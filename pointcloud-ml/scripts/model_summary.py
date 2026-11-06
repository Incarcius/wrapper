"""Measure parameter count, checkpoint size and CPU latency for each model config.

These fill the architecture-independent columns of the Phase Two comparison
matrix. They are **not** deployment numbers: this runs on an x86 dev CPU. Every
Jetson column stays `pending` until measured on the device.

Usage:
    python scripts/model_summary.py [--num-classes 5] [--points 1024] [--repeats 50]
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
import time
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pcml.models import build_model, count_parameters  # noqa: E402

CONFIGS = ["pointnet_deploy", "pointnet_teacher", "pointnet_student"]


def checkpoint_size_mb(model: torch.nn.Module) -> float:
    """Actual serialised state_dict size, not a parameter-count estimate."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as fh:
        path = Path(fh.name)
    try:
        torch.save(model.state_dict(), path)
        return path.stat().st_size / (1024 * 1024)
    finally:
        path.unlink(missing_ok=True)


def time_forward(
    model: torch.nn.Module, batch: int, points: int, repeats: int, warmup: int = 10
) -> dict[str, float]:
    """Median / p95 wall-clock latency of one forward pass, in milliseconds."""
    x = torch.randn(batch, 3, points)
    aux = (
        torch.randn(batch, model.aux_dim) if getattr(model, "aux_dim", 0) > 0 else None
    )

    with torch.inference_mode():
        for _ in range(warmup):
            model(x, aux)

        samples = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            model(x, aux)
            samples.append((time.perf_counter() - t0) * 1000.0)

    samples.sort()
    return {
        "median_ms": samples[len(samples) // 2],
        "p95_ms": samples[int(len(samples) * 0.95)],
        "min_ms": samples[0],
        "max_ms": samples[-1],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-classes",
        type=int,
        default=5,
        help="PLACEHOLDER class count. The real K comes from label_map.yaml, "
        "which does not exist yet (task_definition.md P1).",
    )
    parser.add_argument("--points", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=[1, 4, 8, 16],
        help="cluster counts per frame to time",
    )
    args = parser.parse_args()

    env = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_threads": torch.get_num_threads(),
        "note": "x86 dev CPU, NOT the Jetson Orin Nano. Deployment latency is pending.",
    }

    print("=" * 78)
    print("Environment")
    print("=" * 78)
    for k, v in env.items():
        print(f"  {k:16s} {v}")
    print(f"\n  num_classes      {args.num_classes}  (PLACEHOLDER — no label map yet)")
    print(f"  points/cluster   {args.points}")

    results = {"environment": env, "num_classes": args.num_classes, "models": {}}

    for name in CONFIGS:
        cfg_path = REPO / "configs" / "models" / f"{name}.yaml"
        with cfg_path.open() as fh:
            cfg = yaml.safe_load(fh)

        model = build_model(cfg["model"], num_classes=args.num_classes).eval()
        n_params = count_parameters(model)
        size_mb = checkpoint_size_mb(model)

        print("\n" + "=" * 78)
        print(f"{name}   ({cfg['model']['widths']} -> {cfg['model']['head_widths']})")
        print("=" * 78)
        print(f"  parameters       {n_params:,}")
        print(f"  checkpoint       {size_mb:.2f} MB (fp32 state_dict)")
        print(f"  fp16 estimate    {size_mb / 2:.2f} MB")
        print(f"\n  {'clusters':>9}  {'median':>9}  {'p95':>9}  {'per-cluster':>12}")
        print(f"  {'-' * 9}  {'-' * 9}  {'-' * 9}  {'-' * 12}")

        timings = {}
        for b in args.batches:
            t = time_forward(model, b, args.points, args.repeats)
            timings[b] = t
            print(
                f"  {b:>9}  {t['median_ms']:>8.2f}ms  {t['p95_ms']:>8.2f}ms  "
                f"{t['median_ms'] / b:>11.2f}ms"
            )

        results["models"][name] = {
            "widths": cfg["model"]["widths"],
            "head_widths": cfg["model"]["head_widths"],
            "parameters": n_params,
            "checkpoint_mb": round(size_mb, 3),
            "cpu_latency_ms": {str(k): v for k, v in timings.items()},
            "jetson_latency_ms": "pending — requires on-device measurement",
        }

    out = REPO / "reports" / "model_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nWrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
