"""ONNX export, export parity, and the graph op-type inventory.

Why this module exists
---------------------
PointNet was chosen over DGCNN and PointNet++ substantially on the claim that
*every op it uses is TensorRT-native and needs no custom plugins*
(``docs/task_definition.md`` §9, ``src/pcml/models/pointnet.py`` module docstring).
That claim was an analysis, not a measurement. This module turns it into one, as
far as it can be turned into one without the device: the graph is produced, its
op inventory is enumerated against a justified allowlist, and its numerics are
compared against PyTorch through onnxruntime.

**What this does NOT establish.** TensorRT is not exercised here and cannot be:
the dev laptop has no NVIDIA GPU and no Jetson is reachable. "Every op is
TensorRT-native" is verified only in the sense that *the exported graph contains
only ops from a TensorRT-native allowlist* — the actual `trtexec` build remains
pending on hardware. Do not let a green test here be read as a TensorRT result.

The dynamic batch axis
---------------------
The batch axis is the per-frame region count ``M``, which varies frame to frame.
It is exported **dynamic** on purpose. A fixed batch would force the deploy path
to either pad every frame up to ``max_clusters_per_frame`` (paying for 16
regions when 3 are present, every frame) or re-export per observed ``M``. The
point axis stays fixed at ``N=1024`` because :func:`pcml.data.sampling.resample_to_fixed_size`
guarantees it, and a fixed axis is what lets TensorRT pick kernels for it.

Exporter selection
-----------------
torch 2.9+ defaults to ``dynamo=True`` (the torch.export-based exporter), which
requires the ``onnxscript`` package. It is **not installed in this environment**,
so the default path raises ``ModuleNotFoundError``. This module therefore passes
``dynamo=False`` explicitly and uses the legacy TorchScript exporter, which
works and produces the graphs recorded in ``reports/export_parity.json``.

That is a deliberate, recorded choice, not an oversight:

* the legacy exporter is deprecated and will be removed, so this is a known
  future migration (``pip install onnxscript`` is the whole fix);
* it emits a ``DeprecationWarning`` attributed to *this module's* frame, which
  the repo's ``filterwarnings = ["error::DeprecationWarning:pcml.*"]`` would
  otherwise turn into a hard error. Warnings raised during export are captured
  and returned on :attr:`ExportResult.warnings_raised` rather than suppressed,
  so nothing is hidden — ``scripts/export_model.py`` prints them.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "DEFAULT_NUM_POINTS",
    "DEFAULT_OPSET",
    "EXPECTED_OPS",
    "ExportResult",
    "ParityReport",
    "PARITY_TOLERANCE",
    "build_export_wrapper",
    "check_parity",
    "export_onnx",
    "graph_io_shapes",
    "onnx_op_types",
    "onnxruntime_latency",
    "parity_inputs",
    "unexpected_op_types",
]

#: Points per region. Fixed by the sampler, so fixed in the graph.
DEFAULT_NUM_POINTS = 1024

#: Opset 17.
#:
#: Measured reason, not a default copied from a tutorial: at opset >= 18
#: ``ReduceMax`` takes its ``axes`` as an *input tensor* rather than an
#: attribute, so the graph gains a ``Constant`` node feeding it. At opset 17 the
#: axes stay an attribute and the max-pool is a single node. Both are
#: TensorRT-native; 17 is simply the smaller graph, and TensorRT 8.x on JetPack 6
#: supports it. Opsets 13 and 17 produce an identical op inventory here.
DEFAULT_OPSET = 17

#: Asserted bound on the max absolute logit difference between PyTorch and
#: onnxruntime, both fp32 on CPU.
#:
#: Justification. fp32 eps is ~1.2e-7. A single logit accumulates a 1024-point
#: ReduceMax over a 5-layer Conv1d stack plus a 3-layer head, so the two runtimes
#: can legitimately differ by a few ulps wherever they pick a different
#: accumulation order or a fused kernel. The **measured** max across all three
#: configs at M in {1, 4, 16} is **5.96e-8** (``pointnet_student``; deploy and
#: teacher both 3.35e-8 — see ``reports/export_parity.json``), i.e. this bound
#: carries ~170x headroom.
#:
#: The headroom is the point. This test is not a float-precision assay; it is a
#: tripwire for a *structurally* wrong graph — BatchNorm folded with the wrong
#: statistics, a reduction over the wrong axis, a dropped T-Net matmul. Those
#: failures move logits by O(0.1) to O(10), not by 1e-6. A bound set at the
#: measured noise floor would flake on a different BLAS or thread count while
#: catching nothing extra.
PARITY_TOLERANCE = 1e-5

#: Op type -> why it is expected to be here, and why TensorRT handles it.
#:
#: This is an allowlist **with provenance**. Widening it requires writing down a
#: reason, which is the whole mechanism: an op appearing here that nobody can
#: justify is a finding about the architecture, not a test to relax. If
#: :func:`unexpected_op_types` reports something, report it — do not add a key
#: with a hand-wave.
EXPECTED_OPS: dict[str, str] = {
    # --- compute ---------------------------------------------------------
    "Conv": "nn.Conv1d(kernel_size=1), the shared point-wise MLP primitive",
    "BatchNormalization": (
        "nn.BatchNorm1d in eval mode (affine, running stats). NOT folded into "
        "Conv by do_constant_folding; TensorRT fuses Conv+BN at engine-build time"
    ),
    "Relu": "nn.ReLU after every Conv1d/BatchNorm pair and in the head",
    "ReduceMax": "x.max(dim=2) — the symmetric aggregation that makes the model permutation-invariant",
    "GlobalMaxPool": "alternative lowering of the same max-pool; accepted if the exporter picks it",
    "Gemm": "nn.Linear in the classification head and the T-Net regression head",
    "MatMul": "torch.bmm(t, x) applying a T-Net alignment matrix (teacher config only)",
    # --- shape plumbing, all TensorRT-native -----------------------------
    "Transpose": "layout adjustment the exporter may insert around Gemm/MatMul",
    "Reshape": "TNet.forward's .view(-1, k, k) on the regressed matrix",
    "Concat": "torch.cat([embedding, aux], dim=1) when aux_dim > 0",
    "Identity": (
        "nn.Dropout in eval mode lowers to a no-op Identity rather than "
        "disappearing; TensorRT elides it"
    ),
    "Constant": (
        "the Reshape target shape in TNet, and the ReduceMax axes input at "
        "opset >= 18. A frozen initializer, not a runtime op"
    ),
}


# ---------------------------------------------------------------------------
# Export wrappers
# ---------------------------------------------------------------------------
class _PointsOnly(nn.Module):
    """Pins ``forward(points)`` so the graph has no optional input.

    :meth:`PointNetClassifier.forward` is ``forward(points, aux=None)``. Exporting
    it directly makes ``aux`` an optional/kwarg input, which is exactly the shape
    of graph signature that TensorRT and the deploy-side loader should never have
    to reason about. The wrapper makes the arity explicit and positional.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.model(points)


class _PointsAndAux(nn.Module):
    """Pins ``forward(points, aux)`` for a model built with ``aux_dim > 0``."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, points: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        return self.model(points, aux)


def build_export_wrapper(model: nn.Module) -> nn.Module:
    """Wrap ``model`` so its exported signature is positional and fixed-arity.

    Returns a ``_PointsAndAux`` wrapper when ``model.aux_dim > 0``, else a
    ``_PointsOnly`` wrapper. The wrapper is a pure call-forwarder: it adds no
    ops to the graph and holds no parameters of its own.
    """
    aux_dim = int(getattr(model, "aux_dim", 0) or 0)
    wrapper = _PointsAndAux(model) if aux_dim > 0 else _PointsOnly(model)
    return wrapper.eval()


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
@dataclass
class ExportResult:
    """Everything the export produced, for the report and for the tests."""

    path: Path
    input_names: list[str]
    output_names: list[str]
    num_points: int
    aux_dim: int
    num_classes: int
    opset: int
    dynamic_batch: bool
    exporter: str
    op_types: list[str]
    io_shapes: dict[str, list[Union[int, str]]]
    warnings_raised: list[str] = field(default_factory=list)

    @property
    def unexpected_ops(self) -> list[str]:
        return unexpected_op_types(self.op_types)

    @property
    def batch_axis_is_dynamic(self) -> bool:
        """True when every graph input and output has a symbolic dim 0."""
        dims = [shape[0] for shape in self.io_shapes.values() if shape]
        return bool(dims) and all(isinstance(d, str) for d in dims)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path.name,
            "input_names": self.input_names,
            "output_names": self.output_names,
            "num_points": self.num_points,
            "aux_dim": self.aux_dim,
            "num_classes": self.num_classes,
            "opset": self.opset,
            "dynamic_batch": self.dynamic_batch,
            "exporter": self.exporter,
            "op_types": self.op_types,
            "op_types_with_reason": {
                op: EXPECTED_OPS.get(op, "UNEXPECTED — see unexpected_ops")
                for op in self.op_types
            },
            "unexpected_ops": self.unexpected_ops,
            "io_shapes": {k: list(v) for k, v in self.io_shapes.items()},
            "batch_axis_is_dynamic": self.batch_axis_is_dynamic,
            "warnings_raised": self.warnings_raised,
        }


@dataclass
class ParityReport:
    """PyTorch vs onnxruntime agreement, per batch size, through one engine."""

    tolerance: float
    per_batch: dict[int, dict[str, float]]

    @property
    def max_abs_diff(self) -> float:
        return max(v["max_abs_diff"] for v in self.per_batch.values())

    @property
    def mean_abs_diff(self) -> float:
        vals = [v["mean_abs_diff"] for v in self.per_batch.values()]
        return float(sum(vals) / len(vals))

    @property
    def passed(self) -> bool:
        return self.max_abs_diff < self.tolerance

    def to_dict(self) -> dict[str, Any]:
        return {
            "tolerance": self.tolerance,
            "max_abs_diff": self.max_abs_diff,
            "mean_abs_diff": self.mean_abs_diff,
            "passed": self.passed,
            "per_batch": {str(k): v for k, v in self.per_batch.items()},
        }


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def export_onnx(
    model: nn.Module,
    path: Union[str, Path],
    num_points: int = DEFAULT_NUM_POINTS,
    batch_size: int = 1,
    dynamic_batch: bool = True,
    opset: int = DEFAULT_OPSET,
    batch_dim_name: str = "M",
) -> ExportResult:
    """Export ``model`` to ONNX at ``path`` and return what the graph contains.

    Args:
        model: a :class:`~pcml.models.pointnet.PointNetClassifier` (anything with
            an ``aux_dim`` attribute and a ``(B, C, N) [, (B, aux_dim)] -> (B, K)``
            forward works). Must be in eval mode — see Raises.
        path: output ``.onnx`` path. Parent directories are created.
        num_points: the fixed ``N``. Kept fixed on purpose; the sampler guarantees it.
        batch_size: the ``M`` used for the *tracing* example input. With
            ``dynamic_batch=True`` it does not constrain inference; it is still
            kept >= 2 internally for the trace, because tracing at ``M=1`` lets
            the exporter bake a size-1 axis into reshapes.
        dynamic_batch: mark axis 0 of every input and output symbolic. Defaults
            true because the per-frame region count ``M`` varies. Set false only
            to produce a deliberately fixed-shape engine for comparison.
        opset: ONNX opset. See :data:`DEFAULT_OPSET` for why 17.
        batch_dim_name: symbolic name for the dynamic axis.

    Raises:
        ValueError: if ``model.training`` is true. A model in train mode makes
            BatchNorm use *batch* statistics; the traced graph would then encode
            semantics that differ from deployment, and the parity check would
            compare against the wrong reference. Caller calls ``.eval()``, so the
            mutation stays the caller's decision.
        ValueError: if ``num_points`` or ``batch_size`` is < 1.
    """
    if getattr(model, "training", False):
        raise ValueError(
            "export_onnx requires a model in eval mode: BatchNorm1d and Dropout "
            "behave differently in train mode, so the exported graph would not "
            "match the deployed one. Call model.eval() first."
        )
    if num_points < 1:
        raise ValueError(f"num_points must be >= 1, got {num_points}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    aux_dim = int(getattr(model, "aux_dim", 0) or 0)
    in_channels = int(getattr(model, "in_channels", 3))
    num_classes = int(getattr(model, "num_classes", 0))
    wrapper = build_export_wrapper(model)

    # Trace at >= 2 so no reshape can capture a degenerate size-1 batch axis.
    trace_batch = max(batch_size, 2) if dynamic_batch else batch_size

    example: tuple[torch.Tensor, ...] = (
        torch.zeros(trace_batch, in_channels, num_points),
    )
    input_names = ["points"]
    if aux_dim > 0:
        example = (*example, torch.zeros(trace_batch, aux_dim))
        input_names.append("aux")
    output_names = ["logits"]

    dynamic_axes: Optional[dict[str, dict[int, str]]] = None
    if dynamic_batch:
        dynamic_axes = {
            name: {0: batch_dim_name} for name in (*input_names, *output_names)
        }

    with warnings.catch_warnings(record=True) as caught:
        # "always" + record: captured, not silenced. The legacy-exporter
        # DeprecationWarning is attributed to this frame, and the repo turns
        # DeprecationWarning from pcml.* into an error; the TracerWarnings are
        # real signal (see reports/export_parity.json "findings"). Both are
        # returned on the result and printed by scripts/export_model.py.
        warnings.simplefilter("always")
        torch.onnx.export(
            wrapper,
            example,
            str(path),
            input_names=input_names,
            output_names=output_names,
            opset_version=opset,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes,
            # Explicit: torch >= 2.9 defaults this to True, and that path needs
            # `onnxscript`, which is absent here. See the module docstring.
            dynamo=False,
        )

    warning_lines = [
        f"{w.category.__name__} ({Path(w.filename).name}:{w.lineno}): "
        f"{str(w.message).splitlines()[0][:200]}"
        for w in caught
    ]

    import onnx

    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)

    return ExportResult(
        path=path,
        input_names=input_names,
        output_names=output_names,
        num_points=num_points,
        aux_dim=aux_dim,
        num_classes=num_classes,
        opset=opset,
        dynamic_batch=dynamic_batch,
        exporter="torchscript (dynamo=False)",
        op_types=onnx_op_types(path),
        io_shapes=graph_io_shapes(path),
        warnings_raised=warning_lines,
    )


# ---------------------------------------------------------------------------
# Graph inspection
# ---------------------------------------------------------------------------
def onnx_op_types(path: Union[str, Path]) -> list[str]:
    """Sorted unique op types in the graph. The inventory the allowlist checks."""
    import onnx

    graph = onnx.load(str(path)).graph
    return sorted({node.op_type for node in graph.node})


def unexpected_op_types(
    op_types: Iterable[str], allowlist: Optional[Mapping[str, str]] = None
) -> list[str]:
    """Ops present in the graph that :data:`EXPECTED_OPS` does not account for.

    A non-empty return is a **finding**, not a test to loosen: it means the
    exported graph contains an operation nobody predicted, and whether TensorRT
    implements it natively is then an open question rather than an assumption.
    """
    allowed = EXPECTED_OPS if allowlist is None else allowlist
    return sorted(set(op_types) - set(allowed))


def graph_io_shapes(path: Union[str, Path]) -> dict[str, list[Union[int, str]]]:
    """Declared shapes of every graph input and output.

    A symbolic dim comes back as its ``str`` name (e.g. ``"M"``), a fixed dim as
    an ``int`` — which is exactly the distinction the dynamic-axis test needs.
    """
    import onnx

    graph = onnx.load(str(path)).graph
    shapes: dict[str, list[Union[int, str]]] = {}
    for value in list(graph.input) + list(graph.output):
        dims: list[Union[int, str]] = []
        for dim in value.type.tensor_type.shape.dim:
            dims.append(dim.dim_param if dim.dim_param else int(dim.dim_value))
        shapes[value.name] = dims
    return shapes


# ---------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------
def parity_inputs(
    batch: int,
    num_points: int = DEFAULT_NUM_POINTS,
    in_channels: int = 3,
    aux_dim: int = 0,
    seed: int = 0,
    raw_points_per_region: int = 3000,
) -> dict[str, np.ndarray]:
    """Deterministic, realistic model inputs: ``{"points": ..., ["aux": ...]}``.

    Each of the ``batch`` regions is built by pushing a raw synthetic cloud
    through :func:`pcml.data.sampling.resample_to_fixed_size` with
    ``use_open3d=False``. Two reasons:

    1. **Bit-identical inputs.** The NumPy FPS path is deterministic given an
       explicit ``Generator`` (that is why ``pcml.data.sampling`` takes one), so
       PyTorch and onnxruntime are fed the same bytes and any difference in the
       logits is attributable to the graph. Open3D picks its own FPS start index
       and is not importable here anyway, so ``use_open3d=False`` is both the
       reproducible path and the only available one.
    2. **Realistic magnitudes.** FPS output has the spatial structure and scale
       of a real region. ``randn`` would also work numerically, but parity on
       plausible inputs is the claim worth making.

    ``aux`` is drawn from the same seeded generator rather than computed from
    cluster statistics: the auxiliary vector enters the graph as a plain
    ``Concat`` at the head, so its *provenance* is irrelevant to parity while its
    determinism is not.
    """
    from pcml.data.sampling import resample_to_fixed_size

    rng = np.random.default_rng(seed)
    regions = []
    for _ in range(batch):
        # A region-sized blob: metre-scale, off-origin, anisotropic — i.e. what a
        # cluster looks like before per-region normalisation.
        centre = rng.uniform(-20.0, 20.0, size=3)
        extent = rng.uniform(0.5, 4.0, size=3)
        raw = centre + rng.standard_normal((raw_points_per_region, 3)) * extent
        sampled, _stats = resample_to_fixed_size(
            raw.astype(np.float32), num_points, rng, use_open3d=False
        )
        regions.append(sampled)

    # (M, N, 3) -> (M, 3, N): Conv1d wants channels at dim 1.
    points = np.stack(regions, axis=0).transpose(0, 2, 1)
    points = np.ascontiguousarray(points, dtype=np.float32)

    if in_channels != 3:
        # Pad/trim to the configured channel count so this helper stays usable
        # if in_channels ever stops being 3. xyz is always the leading block.
        padded = np.zeros((batch, in_channels, num_points), dtype=np.float32)
        keep = min(3, in_channels)
        padded[:, :keep] = points[:, :keep]
        points = padded

    out: dict[str, np.ndarray] = {"points": points}
    if aux_dim > 0:
        out["aux"] = np.ascontiguousarray(
            rng.standard_normal((batch, aux_dim)), dtype=np.float32
        )
    return out


def check_parity(
    model: nn.Module,
    onnx_path: Union[str, Path],
    batch_sizes: Sequence[int] = (1, 4, 16),
    num_points: int = DEFAULT_NUM_POINTS,
    tolerance: float = PARITY_TOLERANCE,
    seed: int = 0,
) -> ParityReport:
    """Compare PyTorch and onnxruntime logits across ``batch_sizes``.

    One ``InferenceSession`` is built and every ``M`` runs through *that same
    session*. That is deliberate and is the actual test of the dynamic axis: a
    graph with a baked batch size would fail on the second ``M``, and
    re-exporting per ``M`` would prove nothing.
    """
    import onnxruntime as ort

    if getattr(model, "training", False):
        raise ValueError("check_parity needs the model in eval mode; call .eval()")

    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    output_name = session.get_outputs()[0].name
    aux_dim = int(getattr(model, "aux_dim", 0) or 0)
    in_channels = int(getattr(model, "in_channels", 3))

    per_batch: dict[int, dict[str, float]] = {}
    for batch in batch_sizes:
        feeds = parity_inputs(
            batch,
            num_points=num_points,
            in_channels=in_channels,
            aux_dim=aux_dim,
            seed=seed,
        )
        onnx_logits = session.run([output_name], feeds)[0]

        torch_args = [torch.from_numpy(feeds["points"])]
        if aux_dim > 0:
            torch_args.append(torch.from_numpy(feeds["aux"]))
        with torch.inference_mode():
            torch_logits = model(*torch_args).numpy()

        if torch_logits.shape != onnx_logits.shape:
            raise AssertionError(
                f"shape mismatch at M={batch}: torch {torch_logits.shape} vs "
                f"onnx {onnx_logits.shape}"
            )

        diff = np.abs(torch_logits.astype(np.float64) - onnx_logits.astype(np.float64))
        per_batch[batch] = {
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
            "logit_abs_max": float(np.abs(torch_logits).max()),
            "shape": list(onnx_logits.shape),
        }

    return ParityReport(tolerance=tolerance, per_batch=per_batch)


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------
def onnxruntime_latency(
    onnx_path: Union[str, Path],
    batch_sizes: Sequence[int] = (1, 4, 8, 16),
    num_points: int = DEFAULT_NUM_POINTS,
    in_channels: int = 3,
    aux_dim: int = 0,
    repeats: int = 50,
    warmup: int = 10,
    seed: int = 0,
) -> dict[int, dict[str, float]]:
    """Median / p95 onnxruntime CPU latency per ``M``, in milliseconds.

    Same protocol as ``scripts/model_summary.py``: ``warmup`` untimed passes,
    then ``repeats`` timed ones, reported as median and p95 so the numbers are
    comparable with the torch figures in ``reports/stage1_baseline.md``.

    **This is x86 CPU onnxruntime.** It is not a Jetson number and it is not a
    TensorRT number. It bounds nothing about the device.
    """
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    output_name = session.get_outputs()[0].name

    out: dict[int, dict[str, float]] = {}
    for batch in batch_sizes:
        feeds = parity_inputs(
            batch,
            num_points=num_points,
            in_channels=in_channels,
            aux_dim=aux_dim,
            seed=seed,
        )
        for _ in range(warmup):
            session.run([output_name], feeds)

        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            session.run([output_name], feeds)
            samples.append((time.perf_counter() - start) * 1000.0)

        samples.sort()
        out[batch] = {
            "median_ms": samples[len(samples) // 2],
            "p95_ms": samples[min(int(len(samples) * 0.95), len(samples) - 1)],
            "min_ms": samples[0],
            "max_ms": samples[-1],
            "per_region_ms": samples[len(samples) // 2] / batch,
        }
    return out
