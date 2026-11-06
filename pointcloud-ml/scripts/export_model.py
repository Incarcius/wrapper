"""Export a model config to ONNX, verify parity, and record the result.

Does four things, in order, and refuses to report success if any of them fails:

  1. exports to ONNX with a **dynamic batch axis** (M = regions per frame) and a
     fixed point axis (N = 1024, guaranteed by the sampler);
  2. enumerates the graph's op types against the justified allowlist in
     ``pcml.compression.export.EXPECTED_OPS``;
  3. measures PyTorch-vs-onnxruntime logit agreement across several M through the
     *same* session, which is what proves the dynamic axis actually works;
  4. times onnxruntime on CPU at each M, in the protocol of
     ``scripts/model_summary.py`` so the numbers are comparable.

Then writes ``reports/export_parity.json``.

LIMITS, which the output restates so nobody can quote a number out of context:

  * **This is not TensorRT.** No NVIDIA GPU on this machine, no Jetson in reach.
    What is verified is that the graph contains only ops from a TensorRT-native
    allowlist, and that it is numerically faithful under onnxruntime. The
    `trtexec` build is pending on hardware.
  * **These are x86 CPU onnxruntime timings**, not device timings.
  * **The weights are random** unless ``--checkpoint`` is given. There is no
    trained checkpoint yet. Parity is a property of the graph and is fully
    meaningful without training, but the logits are not predictions and the
    class outputs mean nothing.

Usage:
    python scripts/export_model.py                         # all three configs
    python scripts/export_model.py --config pointnet_deploy
    python scripts/export_model.py --config pointnet_deploy \
        --checkpoint runs/exp1/best.pt --out artifacts/deploy.onnx
    python scripts/export_model.py --aux                   # two-input signature
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pcml.compression.export import (  # noqa: E402
    DEFAULT_NUM_POINTS,
    DEFAULT_OPSET,
    EXPECTED_OPS,
    PARITY_TOLERANCE,
    check_parity,
    export_onnx,
    onnxruntime_latency,
    unexpected_op_types,
)
from pcml.data.clustering import AUX_FEATURE_NAMES  # noqa: E402
from pcml.data.label_mapping import load_label_map  # noqa: E402
from pcml.models import build_model, count_parameters  # noqa: E402

CONFIGS = ["pointnet_deploy", "pointnet_teacher", "pointnet_student"]
RULE = "=" * 78


def load_config(name: str) -> dict:
    path = REPO / "configs" / "models" / f"{name}.yaml"
    if not path.exists():
        raise SystemExit(f"no such config: {path}")
    with path.open() as fh:
        return yaml.safe_load(fh)["model"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        action="append",
        choices=CONFIGS,
        help="model config to export; repeatable. Default: all three.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="optional state_dict to load. Without it the weights are RANDOM, "
        "which is fine for parity (a graph property) and useless for anything else.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="output .onnx path. Only valid with a single --config; otherwise "
        "files land in --out-dir named after the config.",
    )
    parser.add_argument("--out-dir", type=Path, default=REPO / "artifacts" / "onnx")
    parser.add_argument("--label-map", type=Path, default=REPO / "data" / "label_map.yaml")
    parser.add_argument("--points", type=int, default=DEFAULT_NUM_POINTS)
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    parser.add_argument(
        "--parity-batches", type=int, nargs="+", default=[1, 4, 16],
        help="region counts M to check parity at, through one session",
    )
    parser.add_argument(
        "--latency-batches", type=int, nargs="+", default=[1, 4, 8, 16],
        help="region counts M to time",
    )
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--tolerance", type=float, default=PARITY_TOLERANCE,
        help=f"max permitted |logit difference| (default {PARITY_TOLERANCE:g})",
    )
    parser.add_argument(
        "--aux", action="store_true",
        help=f"force aux_dim={len(AUX_FEATURE_NAMES)} to exercise the two-input "
        "signature, regardless of what the config says",
    )
    parser.add_argument(
        "--no-report", action="store_true", help="skip writing reports/export_parity.json"
    )
    args = parser.parse_args()

    names = args.config or CONFIGS
    if args.out and len(names) > 1:
        raise SystemExit("--out takes a single --config; use --out-dir for several")

    label_map = load_label_map(args.label_map)
    num_classes = label_map.num_classes

    env = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_threads": torch.get_num_threads(),
    }
    try:
        import onnx
        import onnxruntime as ort

        env["onnx"] = onnx.__version__
        env["onnxruntime"] = ort.__version__
        env["ort_providers"] = ort.get_available_providers()
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit(f"export needs onnx and onnxruntime: {exc}")

    print(RULE)
    print("Environment")
    print(RULE)
    for key, value in env.items():
        print(f"  {key:16s} {value}")
    print()
    print("  SCOPE: ONNX + onnxruntime on an x86 CPU. NOT TensorRT, NOT a Jetson.")
    print("         The op allowlist is evidence that no op needs a TRT plugin;")
    print("         it is not a trtexec build. That remains pending on hardware.")
    if args.checkpoint is None:
        print("  WEIGHTS: RANDOM (no trained checkpoint exists). Parity is a graph")
        print("           property and is valid; the class outputs are NOT predictions.")
    print(f"\n  num_classes      {num_classes}  (from {args.label_map.name})")
    print(f"  points/region    {args.points}  (fixed axis)")
    print(f"  opset            {args.opset}")
    print(f"  tolerance        {args.tolerance:g}")

    report: dict = {
        "environment": env,
        "scope": {
            "verified": [
                "ONNX export of all requested configs with a dynamic batch axis",
                "graph op-type inventory against a justified allowlist",
                "PyTorch vs onnxruntime fp32 logit parity across several M, "
                "through one session",
                "onnxruntime CPU latency",
            ],
            "NOT_verified": [
                "TensorRT engine build (trtexec) — no NVIDIA GPU, no Jetson access",
                "TensorRT fp16/int8 numerics",
                "any on-device latency, memory, power or thermal figure",
                "accuracy of any kind — no model is trained and no data exists",
            ],
        },
        "num_classes": num_classes,
        "label_map": str(args.label_map.relative_to(REPO)),
        "points_per_region": args.points,
        "opset": args.opset,
        "tolerance": args.tolerance,
        "weights": "checkpoint" if args.checkpoint else "random (untrained)",
        "allowlist": EXPECTED_OPS,
        "models": {},
    }

    failures: list[str] = []

    for name in names:
        cfg = load_config(name)
        if args.aux:
            cfg = {**cfg, "aux_dim": len(AUX_FEATURE_NAMES)}

        model = build_model(cfg, num_classes=num_classes)
        if args.checkpoint:
            state = torch.load(args.checkpoint, map_location="cpu")
            state = state.get("model", state) if isinstance(state, dict) else state
            model.load_state_dict(state)
        model = model.eval()

        out_path = args.out or (args.out_dir / f"{name}.onnx")

        print("\n" + RULE)
        print(f"{name}   {cfg['widths']} -> {cfg['head_widths']}"
              + (f"   aux_dim={cfg['aux_dim']}" if cfg.get("aux_dim") else ""))
        print(RULE)
        print(f"  parameters       {count_parameters(model):,}")

        result = export_onnx(
            model,
            out_path,
            num_points=args.points,
            dynamic_batch=True,
            opset=args.opset,
        )
        print(f"  wrote            {out_path}  "
              f"({out_path.stat().st_size / 1024:.0f} KiB)")
        print(f"  exporter         {result.exporter}")

        # --- graph shape ---------------------------------------------------
        print("\n  graph I/O (a quoted name is a dynamic axis)")
        for io_name, shape in result.io_shapes.items():
            pretty = ", ".join(f'"{d}"' if isinstance(d, str) else str(d) for d in shape)
            print(f"    {io_name:10s} [{pretty}]")
        dyn_ok = result.batch_axis_is_dynamic
        print(f"    dynamic batch axis: {'YES' if dyn_ok else 'NO'}")
        if not dyn_ok:
            failures.append(f"{name}: batch axis is not dynamic")

        # --- op inventory --------------------------------------------------
        print(f"\n  op-type inventory ({len(result.op_types)} distinct)")
        for op in result.op_types:
            reason = EXPECTED_OPS.get(op)
            mark = " " if reason else "!"
            print(f"   {mark}  {op:20s} {reason or 'UNEXPECTED — not in allowlist'}")
        unexpected = unexpected_op_types(result.op_types)
        if unexpected:
            print(f"\n  !! UNEXPECTED OPS: {unexpected}")
            print("     This is a FINDING about the architecture, not a test to relax.")
            failures.append(f"{name}: unexpected ops {unexpected}")
        else:
            print("    all ops accounted for by the allowlist")

        if result.warnings_raised:
            print(f"\n  warnings raised during export ({len(result.warnings_raised)})")
            for line in result.warnings_raised:
                print(f"    - {line}")

        # --- parity --------------------------------------------------------
        parity = check_parity(
            model,
            out_path,
            batch_sizes=args.parity_batches,
            num_points=args.points,
            tolerance=args.tolerance,
        )
        print("\n  PyTorch vs onnxruntime, one session, fp32")
        print(f"  {'M':>5}  {'max |d|':>11}  {'mean |d|':>11}  {'max |logit|':>11}")
        print(f"  {'-' * 5}  {'-' * 11}  {'-' * 11}  {'-' * 11}")
        for m_regions, stats in parity.per_batch.items():
            print(f"  {m_regions:>5}  {stats['max_abs_diff']:>11.3e}  "
                  f"{stats['mean_abs_diff']:>11.3e}  {stats['logit_abs_max']:>11.3f}")
        print(f"\n    max  |d| over all M : {parity.max_abs_diff:.3e}")
        print(f"    mean |d| over all M : {parity.mean_abs_diff:.3e}")
        print(f"    tolerance           : {args.tolerance:.0e}  "
              f"-> {'PASS' if parity.passed else 'FAIL'}")
        if not parity.passed:
            failures.append(
                f"{name}: parity {parity.max_abs_diff:.3e} exceeds {args.tolerance:g}"
            )

        # --- latency -------------------------------------------------------
        latency = onnxruntime_latency(
            out_path,
            batch_sizes=args.latency_batches,
            num_points=args.points,
            in_channels=int(cfg.get("in_channels", 3)),
            aux_dim=int(cfg.get("aux_dim", 0) or 0),
            repeats=args.repeats,
            warmup=args.warmup,
        )
        print(f"\n  onnxruntime CPU latency  ({args.repeats} repeats, "
              f"{args.warmup} warm-up)  — x86, NOT Jetson")
        print(f"  {'M':>5}  {'median':>10}  {'p95':>10}  {'per-region':>11}")
        print(f"  {'-' * 5}  {'-' * 10}  {'-' * 10}  {'-' * 11}")
        for m_regions, stats in latency.items():
            print(f"  {m_regions:>5}  {stats['median_ms']:>8.2f}ms  "
                  f"{stats['p95_ms']:>8.2f}ms  {stats['per_region_ms']:>9.2f}ms")

        entry = result.to_dict()
        entry["path"] = str(out_path.relative_to(REPO)) if out_path.is_relative_to(REPO) else str(out_path)
        entry["widths"] = cfg["widths"]
        entry["head_widths"] = cfg["head_widths"]
        entry["parameters"] = count_parameters(model)
        entry["onnx_kib"] = round(out_path.stat().st_size / 1024, 1)
        entry["parity"] = parity.to_dict()
        entry["ort_cpu_latency_ms"] = {str(k): v for k, v in latency.items()}
        entry["jetson_tensorrt"] = "pending — requires the device; not measurable here"
        report["models"][name] = entry

    # --- summary -----------------------------------------------------------
    print("\n" + RULE)
    print("Summary")
    print(RULE)
    for name, entry in report["models"].items():
        print(f"  {name:20s} parity max |d| {entry['parity']['max_abs_diff']:.3e}  "
              f"dynamic={entry['batch_axis_is_dynamic']}  "
              f"unexpected_ops={entry['unexpected_ops'] or 'none'}")
    report["passed"] = not failures
    report["failures"] = failures

    if not args.no_report:
        out = REPO / "reports" / "export_parity.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nWrote {out.relative_to(REPO)}")

    if failures:
        print("\nFAILED:")
        for line in failures:
            print(f"  - {line}")
        return 1
    print("\nAll checks passed (ONNX/onnxruntime scope — TensorRT still pending).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
