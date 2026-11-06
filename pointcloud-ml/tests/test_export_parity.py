"""ONNX export contract: dynamic batch axis, PyTorch/onnxruntime parity, op inventory.

These tests substantiate the claim the architecture choice rests on — "every op
PointNet uses is TensorRT-native, no custom plugins" (``docs/task_definition.md``
§9) — to the extent it can be substantiated off-device. They need no data and no
trained weights: parity is a property of the *graph*, and a randomly-initialised
model exercises every op in it.

**Scope limit, stated once and meant.** TensorRT is never invoked here. There is
no NVIDIA GPU on this machine and no Jetson in reach. What is verified is that
the exported graph contains only ops from a justified TensorRT-native allowlist
and that it is numerically faithful to PyTorch under onnxruntime. The `trtexec`
build is still pending on hardware.

Nothing here interprets a logit as a prediction. The weights are random.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402

from pcml.compression.export import (  # noqa: E402
    DEFAULT_NUM_POINTS,
    EXPECTED_OPS,
    PARITY_TOLERANCE,
    check_parity,
    export_onnx,
    graph_io_shapes,
    onnx_op_types,
    parity_inputs,
    unexpected_op_types,
)
from pcml.data.clustering import AUX_FEATURE_NAMES  # noqa: E402
from pcml.models import PointNetClassifier, build_model  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CONFIG_NAMES = ["pointnet_deploy", "pointnet_teacher", "pointnet_student"]

#: K=6, from data/label_map.yaml (confirmed 15 Oct 2026). Read rather than
#: hard-coded, so a change to the vocabulary cannot leave this file stale.
LABEL_MAP = REPO / "data" / "label_map.yaml"

#: The M values parity is checked at. 1 is the degenerate single-region frame
#: (and the one that would break if a size-1 batch axis got baked in); 16 is
#: clustering.max_clusters_per_frame.
PARITY_BATCHES = (1, 4, 16)


def _num_classes() -> int:
    from pcml.data.label_mapping import load_label_map

    return load_label_map(LABEL_MAP).num_classes


@pytest.fixture(scope="module")
def num_classes() -> int:
    return _num_classes()


@pytest.fixture(scope="module")
def configs() -> dict[str, dict]:
    out = {}
    for name in CONFIG_NAMES:
        with (REPO / "configs" / "models" / f"{name}.yaml").open() as fh:
            out[name] = yaml.safe_load(fh)["model"]
    return out


@pytest.fixture(scope="module")
def exported(tmp_path_factory, configs, num_classes):
    """Export each config **once** and reuse the artifacts.

    Module-scoped on purpose: every test below must interrogate the *same*
    engine, because "one engine serves every M" is the property under test.
    """
    out_dir = tmp_path_factory.mktemp("onnx")
    built = {}
    for name, cfg in configs.items():
        model = build_model(cfg, num_classes=num_classes).eval()
        result = export_onnx(model, out_dir / f"{name}.onnx", num_points=DEFAULT_NUM_POINTS)
        built[name] = (model, result)
    return built


# ---------------------------------------------------------------------------
class TestExportSucceeds:
    """The export itself. A failure here is a finding about the architecture."""

    @pytest.mark.parametrize("name", CONFIG_NAMES)
    def test_config_exports(self, name, exported):
        _model, result = exported[name]
        assert result.path.exists()
        assert result.path.stat().st_size > 0
        # onnx.checker ran inside export_onnx; re-assert the model parses.
        onnx.checker.check_model(onnx.load(str(result.path)))

    @pytest.mark.parametrize("name", CONFIG_NAMES)
    def test_graph_has_no_optional_inputs(self, name, exported):
        """An xyz-only model takes exactly one input: no kwarg/optional slots.

        This is what ``build_export_wrapper`` buys. ``forward(points, aux=None)``
        exported directly can leave an optional input in the signature, which the
        deploy-side loader and TensorRT should never have to reason about.
        """
        _model, result = exported[name]
        graph = onnx.load(str(result.path)).graph
        assert [i.name for i in graph.input] == ["points"]
        assert [o.name for o in graph.output] == ["logits"]

    def test_train_mode_model_is_refused(self, configs, num_classes, tmp_path):
        """BatchNorm in train mode would bake batch statistics into the graph."""
        model = build_model(configs["pointnet_deploy"], num_classes=num_classes)
        model.train()
        with pytest.raises(ValueError, match="eval mode"):
            export_onnx(model, tmp_path / "nope.onnx")


class TestDynamicBatchAxis:
    """M varies per frame; a fixed batch would force padding or re-export."""

    @pytest.mark.parametrize("name", CONFIG_NAMES)
    def test_batch_axis_is_symbolic_and_point_axis_is_not(self, name, exported):
        _model, result = exported[name]
        shapes = graph_io_shapes(result.path)

        assert shapes["points"][0] == "M", "batch axis must be symbolic"
        assert shapes["points"][1] == 3, "channel axis stays fixed at 3 (xyz)"
        assert shapes["points"][2] == DEFAULT_NUM_POINTS, (
            "point axis must stay FIXED — the sampler guarantees N=1024, and a "
            "fixed axis is what lets TensorRT select kernels for it"
        )
        assert shapes["logits"][0] == "M"
        assert result.batch_axis_is_dynamic

    @pytest.mark.parametrize("name", CONFIG_NAMES)
    @pytest.mark.parametrize("m_regions", [1, 2, 3, 4, 16, 37])
    def test_one_session_serves_any_region_count(self, name, exported, m_regions, num_classes):
        """The dynamic-axis test proper: one session, many M, including M=1.

        37 is deliberately not a power of two and above max_clusters_per_frame —
        if anything in the graph had a baked or padded batch assumption, an
        awkward M is where it shows.
        """
        _model, result = exported[name]
        session = ort.InferenceSession(
            str(result.path), providers=["CPUExecutionProvider"]
        )
        feeds = parity_inputs(m_regions, num_points=DEFAULT_NUM_POINTS)
        logits = session.run(["logits"], feeds)[0]
        assert logits.shape == (m_regions, num_classes)

    def test_fixed_batch_export_really_is_fixed(self, configs, num_classes, tmp_path):
        """Control case: dynamic_batch=False must produce a concrete dim 0.

        Without this, a test asserting "M is symbolic" could pass vacuously if
        the exporter marked everything symbolic regardless of the flag.
        """
        model = build_model(configs["pointnet_student"], num_classes=num_classes).eval()
        result = export_onnx(
            model, tmp_path / "fixed.onnx", batch_size=4, dynamic_batch=False
        )
        shapes = graph_io_shapes(result.path)
        assert shapes["points"][0] == 4
        assert not result.batch_axis_is_dynamic

        session = ort.InferenceSession(
            str(result.path), providers=["CPUExecutionProvider"]
        )
        with pytest.raises(Exception):  # ort raises InvalidArgument on shape mismatch
            session.run(["logits"], parity_inputs(7, num_points=DEFAULT_NUM_POINTS))


class TestParity:
    """PyTorch vs onnxruntime on bit-identical inputs.

    Tolerance: :data:`PARITY_TOLERANCE` = 1e-5, against a measured worst case of
    **5.96e-8** across all three configs and every M (student; deploy and teacher
    both 3.35e-8) — roughly 170x headroom. The reasoning is recorded on the
    constant itself and is worth restating here because it governs what these
    tests can catch: fp32 eps is ~1.2e-7 and a
    logit accumulates a 1024-point reduction over a Conv1d stack, so a few ulps
    of legitimate divergence are expected wherever the two runtimes order an
    accumulation differently. The bound is deliberately far above that noise
    floor. It is a tripwire for a structurally wrong graph — BatchNorm folded
    with wrong statistics, a reduction over the wrong axis, a dropped T-Net
    matmul — and those move logits by O(0.1)-O(10). A bound pinned at 1e-8 would
    flake on a different BLAS or thread count while catching nothing more.

    Note the inputs are **unnormalised metre-scale** region clouds (centres out
    to +/-20 m), which is why the measured difference is larger than the ~1.5e-8
    seen on unit-scale ``randn`` input: absolute error tracks activation
    magnitude. Parity on the realistic scale is the claim worth making.
    """

    @pytest.mark.parametrize("name", CONFIG_NAMES)
    def test_parity_across_region_counts(self, name, exported):
        model, result = exported[name]
        report = check_parity(
            model, result.path, batch_sizes=PARITY_BATCHES, num_points=DEFAULT_NUM_POINTS
        )

        assert report.passed, (
            f"{name}: max |logit difference| {report.max_abs_diff:.3e} exceeds "
            f"tolerance {PARITY_TOLERANCE:.0e}. Per-M: "
            + ", ".join(
                f"M={m}:{v['max_abs_diff']:.3e}" for m, v in report.per_batch.items()
            )
        )
        # Every requested M actually ran, through the one session.
        assert sorted(report.per_batch) == sorted(PARITY_BATCHES)

    @pytest.mark.parametrize("name", CONFIG_NAMES)
    def test_parity_is_not_vacuous(self, name, exported):
        """Guard the guard: logits must be non-trivial for parity to mean anything.

        If the graph emitted zeros (or the inputs were degenerate), the diff
        would be 0.0 and the parity test would pass while proving nothing.
        """
        model, result = exported[name]
        report = check_parity(model, result.path, batch_sizes=(4,))
        assert report.per_batch[4]["logit_abs_max"] > 1e-3

    def test_teacher_tnet_bmm_survives_export(self, configs, num_classes, tmp_path):
        """The teacher enables both T-Nets, adding a bmm. Confirm it exports.

        Called out separately because ``torch.bmm`` on a dynamic batch axis is
        the one op in this architecture with a plausible export failure mode, and
        the teacher config is the only one that uses it.
        """
        cfg = configs["pointnet_teacher"]
        assert cfg["use_input_tnet"] and cfg["use_feature_tnet"], (
            "this test is pointless if the teacher config stops using T-Nets"
        )
        model = build_model(cfg, num_classes=num_classes).eval()
        result = export_onnx(model, tmp_path / "teacher.onnx")
        assert "MatMul" in result.op_types, "the T-Net bmm should appear as MatMul"

        report = check_parity(model, result.path, batch_sizes=PARITY_BATCHES)
        assert report.passed, f"T-Net parity failed: {report.max_abs_diff:.3e}"

    @pytest.mark.parametrize("name", CONFIG_NAMES)
    def test_inputs_are_bit_identical_between_runs(self, name):
        """The premise of the whole parity claim: seeded inputs reproduce exactly.

        ``parity_inputs`` goes through the NumPy FPS path (``use_open3d=False``),
        which ``pcml.data.sampling`` makes deterministic given an explicit
        Generator. If this drifted, a parity difference could no longer be
        attributed to the graph.
        """
        a = parity_inputs(4, num_points=256, seed=11)
        b = parity_inputs(4, num_points=256, seed=11)
        assert np.array_equal(a["points"], b["points"])
        # And a different seed really does produce different data.
        c = parity_inputs(4, num_points=256, seed=12)
        assert not np.array_equal(a["points"], c["points"])


@pytest.fixture(scope="module")
def aux_export(tmp_path_factory, num_classes):
    """The deploy config rebuilt with aux features on: a two-input graph.

    No config in ``configs/models/`` sets ``aux_dim`` yet (all three ship
    ``aux_dim: 0``, flipped to 6 when ``preprocess.features.use_aux`` is turned
    on), so the two-input signature is constructed here rather than loaded.
    """
    model = PointNetClassifier(
        num_classes=num_classes,
        widths=(64, 64, 64, 128, 512),
        head_widths=(256, 128),
        aux_dim=len(AUX_FEATURE_NAMES),
    ).eval()
    path = tmp_path_factory.mktemp("onnx_aux") / "deploy_aux.onnx"
    return model, export_onnx(model, path)


class TestAuxSignature:
    """aux_dim > 0 makes the model two-input. Both signatures must export."""

    def test_two_inputs_both_dynamic(self, aux_export):
        model, result = aux_export
        assert result.aux_dim == len(AUX_FEATURE_NAMES) == 6
        assert result.input_names == ["points", "aux"]

        shapes = graph_io_shapes(result.path)
        assert shapes["points"] == ["M", 3, DEFAULT_NUM_POINTS]
        assert shapes["aux"] == ["M", 6], (
            "aux must share the dynamic batch axis: one aux row per region"
        )
        assert shapes["logits"][0] == "M"
        assert result.batch_axis_is_dynamic

    def test_aux_parity_across_region_counts(self, aux_export):
        model, result = aux_export
        report = check_parity(model, result.path, batch_sizes=PARITY_BATCHES)
        assert report.passed, f"aux parity failed: {report.max_abs_diff:.3e}"

    def test_aux_concat_appears_and_is_expected(self, aux_export):
        """The aux path adds exactly one op: Concat at the head."""
        _model, result = aux_export
        assert "Concat" in result.op_types
        assert not result.unexpected_ops, result.unexpected_ops


class TestOpInventory:
    """The claim under test: no op needs a TensorRT plugin.

    The allowlist (:data:`EXPECTED_OPS`) is a mapping op -> why it is there, so
    widening it costs writing a justification. If a config starts emitting
    something unlisted, that is a **finding about the architecture** and these
    tests are meant to surface it, with the op named, rather than be relaxed.
    """

    @pytest.mark.parametrize("name", CONFIG_NAMES)
    def test_no_unexpected_ops(self, name, exported):
        _model, result = exported[name]
        unexpected = unexpected_op_types(result.op_types)
        assert not unexpected, (
            f"{name} exported unexpected op type(s) {unexpected}. Full inventory: "
            f"{result.op_types}. Do NOT widen EXPECTED_OPS to silence this — "
            "check whether TensorRT implements the op natively and record the "
            "answer in reports/stage1_baseline.md first."
        )

    @pytest.mark.parametrize("name", CONFIG_NAMES)
    def test_core_pointnet_ops_are_all_present(self, name, exported):
        """Positive control: the ops that MUST be there, or the graph is wrong.

        Without this, ``test_no_unexpected_ops`` would pass on an empty graph.
        """
        _model, result = exported[name]
        ops = set(result.op_types)
        assert "Conv" in ops, "the shared point-wise MLP is Conv1d"
        assert "Relu" in ops
        assert "Gemm" in ops, "the classification head is nn.Linear"
        assert ops & {"ReduceMax", "GlobalMaxPool"}, (
            "the symmetric max-pool aggregation must be in the graph"
        )

    @pytest.mark.parametrize("name", CONFIG_NAMES)
    def test_no_dropout_in_eval_graph(self, name, exported):
        """Dropout must not survive export: it would randomise inference."""
        _model, result = exported[name]
        assert "Dropout" not in result.op_types
        assert "RandomNormal" not in result.op_types
        assert "RandomUniform" not in result.op_types

    @pytest.mark.parametrize("name", CONFIG_NAMES)
    def test_no_neighbourhood_search_ops(self, name, exported):
        """The architecture's actual selling point, as an assertion.

        These are the ops a kNN graph (DGCNN) or a ball query (PointNet++) would
        need, and they are the ones that cost custom plugins or fall back to
        slow TensorRT paths. PointNet should contain none of them.
        """
        forbidden = {
            "TopK",  # kNN
            "GatherND",
            "ScatterND",
            "ScatterElements",
            "NonZero",  # dynamic-shape output; TRT-hostile
            "NonMaxSuppression",
            "GridSample",
            "Loop",
            "If",
            "Scan",
        }
        present = forbidden & set(_model_ops(exported, name))
        assert not present, f"{name} contains neighbourhood/control-flow ops {present}"

    def test_allowlist_entries_all_carry_a_reason(self):
        """Structural: an allowlist entry without a justification is not one."""
        for op, reason in EXPECTED_OPS.items():
            assert isinstance(reason, str) and len(reason) > 20, (
                f"EXPECTED_OPS[{op!r}] needs a real justification, got {reason!r}"
            )

    @pytest.mark.parametrize("name", CONFIG_NAMES)
    def test_inventory_is_reported_not_just_checked(self, name, exported):
        """Every op in the graph resolves to a recorded reason in the report."""
        _model, result = exported[name]
        reasons = result.to_dict()["op_types_with_reason"]
        assert set(reasons) == set(result.op_types)
        assert all("UNEXPECTED" not in r for r in reasons.values())


def _model_ops(exported, name) -> list[str]:
    _model, result = exported[name]
    return onnx_op_types(result.path)


class TestLabelMapCoupling:
    """K comes from the label map, never from a model config or this file."""

    def test_logit_width_follows_the_label_map(self, exported, num_classes):
        assert num_classes == 6, (
            "data/label_map.yaml declares 6 classes; if that changed, these "
            "exports and the recorded parity numbers need re-running"
        )
        for name in CONFIG_NAMES:
            _model, result = exported[name]
            assert graph_io_shapes(result.path)["logits"][1] == num_classes
            assert result.num_classes == num_classes
