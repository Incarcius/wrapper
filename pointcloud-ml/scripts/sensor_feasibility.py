"""How many points does a Livox Mid-70 put on a given object at a given range?

This decides what is classifiable at all, and therefore what the class list may
contain and how the drone must be flown. It is a geometry calculation, not a
measurement, and it is deliberately **optimistic** — see the caveats printed at
the end.

Model
-----
The Mid-70 spreads a fixed point budget over a fixed cone, so point density falls
as 1/R^2:

    Omega_fov       = 2*pi*(1 - cos(theta_half))          [steradians]
    pts_per_sr      = points_per_frame / Omega_fov
    pts_on_target   = pts_per_sr * A_eff / R^2 * n_frames

where ``A_eff`` is the target's area projected perpendicular to the line of
sight. For a nadir-looking sensor, ``R`` is the flight altitude and ``A_eff`` is
the plan-view area; for a forward/oblique sensor, ``R`` is the slant range and
``A_eff`` is the silhouette area.

Usage:
    python scripts/sensor_feasibility.py [--min-cluster-points 50] [--frames 1 3]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# --- Livox Mid-70, from the published specification -------------------------
FOV_DEG = 70.4          # circular field of view
POINTS_PER_SECOND = 100_000
FRAME_RATE_HZ = 10      # livox_ros2_driver default publish_freq
POINTS_PER_FRAME = POINTS_PER_SECOND // FRAME_RATE_HZ

# --- Targets: (name, effective area m^2, how the area was derived) ----------
TARGETS: list[tuple[str, float, str]] = [
    ("cricket ball", 0.0036, "72 mm diameter disc"),
    ("cricket stumps", 0.16, "0.71 m tall x 0.23 m wide silhouette"),
    ("sports bag / equipment", 0.25, "~0.6 x 0.4 m"),
    ("person (nadir, from above)", 0.35, "shoulders + head plan view"),
    ("person (oblique/side-on)", 0.85, "1.7 m x 0.5 m silhouette"),
    ("bench", 1.20, "2.0 m x 0.6 m"),
    ("goalpost (football)", 1.50, "crossbar 7.3 m + 2 uprights, 0.12 m section"),
    ("floodlight / fence pole", 2.40, "8 m visible x 0.3 m"),
    ("car (side-on)", 4.00, "4.0 m x 1.0 m visible flank"),
    ("practice net", 8.00, "8 m x 3 m, permeable so returns are partial"),
    ("tree canopy (small)", 10.00, "~3.5 m diameter crown"),
    ("building wall segment", 20.00, "10 m x 2 m of planar wall"),
]

RANGES_M = [5, 10, 15, 20, 25, 30, 40, 50]


def points_per_steradian(points_per_frame: int = POINTS_PER_FRAME) -> float:
    theta_half = math.radians(FOV_DEG / 2.0)
    omega = 2.0 * math.pi * (1.0 - math.cos(theta_half))
    return points_per_frame / omega


def points_on_target(area_m2: float, range_m: float, n_frames: int = 1) -> float:
    return points_per_steradian() * area_m2 / (range_m**2) * n_frames


def nadir_density(altitude_m: float) -> float:
    """Points per square metre of ground, nadir-looking, one frame."""
    footprint_radius = altitude_m * math.tan(math.radians(FOV_DEG / 2.0))
    return POINTS_PER_FRAME / (math.pi * footprint_radius**2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--min-cluster-points",
        type=int,
        default=50,
        help="the clustering threshold below which we decline to classify",
    )
    ap.add_argument("--frames", type=int, nargs="+", default=[1, 3])
    args = ap.parse_args()

    omega = 2.0 * math.pi * (1.0 - math.cos(math.radians(FOV_DEG / 2.0)))
    print("=" * 92)
    print("Livox Mid-70 point budget")
    print("=" * 92)
    print(f"  FOV                   {FOV_DEG}deg circular")
    print(f"  solid angle           {omega:.4f} sr")
    print(f"  points/second         {POINTS_PER_SECOND:,}")
    print(f"  frame rate            {FRAME_RATE_HZ} Hz  ->  {POINTS_PER_FRAME:,} points/frame")
    print(f"  points per steradian  {points_per_steradian():,.0f} per frame")
    print(f"  usable threshold      {args.min_cluster_points} points (min_cluster_points)")

    print()
    print("=" * 92)
    print("Ground sampling density, nadir-looking, one frame")
    print("=" * 92)
    print(f"  {'altitude':>9}  {'footprint':>11}  {'density':>14}")
    print(f"  {'-'*9}  {'-'*11}  {'-'*14}")
    for h in [5, 10, 15, 20, 25, 30, 40]:
        d = 2 * h * math.tan(math.radians(FOV_DEG / 2.0))
        print(f"  {h:>7} m  {d:>9.1f} m  {nadir_density(h):>9.1f} pts/m2")

    results: dict[str, dict] = {}
    for n_frames in args.frames:
        label = f"{n_frames} frame" + ("s" if n_frames > 1 else "")
        secs = n_frames / FRAME_RATE_HZ
        print()
        print("=" * 92)
        print(f"Points on target — {label} ({secs:.1f} s integration)")
        print("=" * 92)
        header = "  " + f"{'target':<28}" + "".join(f"{r:>7}m" for r in RANGES_M)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for name, area, _ in TARGETS:
            cells = []
            for r in RANGES_M:
                p = points_on_target(area, r, n_frames)
                mark = " " if p >= args.min_cluster_points else "."
                cells.append(f"{p:>6.0f}{mark}")
            print(f"  {name:<28}" + "".join(cells))
        print(f"\n  '.' = below {args.min_cluster_points} points: do not classify")

        # Max usable range per target.
        ranges: dict[str, float] = {}
        for name, area, _ in TARGETS:
            r_max = math.sqrt(
                points_per_steradian() * area * n_frames / args.min_cluster_points
            )
            ranges[name] = round(r_max, 1)
        results[label] = ranges

    print()
    print("=" * 92)
    print(f"Maximum range at which a target still yields {args.min_cluster_points} points")
    print("=" * 92)
    print(f"  {'target':<28}" + "".join(f"{k:>12}" for k in results))
    print("  " + "-" * (28 + 12 * len(results)))
    for name, _, derivation in TARGETS:
        row = "".join(f"{results[k][name]:>10.1f} m" for k in results)
        print(f"  {name:<28}{row}")
        print(f"  {'':<28}  ({derivation})")

    out = REPO / "reports" / "sensor_feasibility.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        json.dump(
            {
                "sensor": {
                    "model": "Livox Mid-70",
                    "fov_deg": FOV_DEG,
                    "solid_angle_sr": round(omega, 4),
                    "points_per_second": POINTS_PER_SECOND,
                    "frame_rate_hz": FRAME_RATE_HZ,
                    "points_per_frame": POINTS_PER_FRAME,
                },
                "min_cluster_points": args.min_cluster_points,
                "max_usable_range_m": results,
                "targets": {n: {"area_m2": a, "derivation": d} for n, a, d in TARGETS},
                "caveats": [
                    "Geometry only: assumes the full point budget is spread uniformly "
                    "over the cone. The non-repetitive rosette is NOT uniform within a "
                    "single 100 ms frame, so single-frame counts are optimistic.",
                    "Ignores incidence-angle losses. A surface seen edge-on returns far "
                    "fewer points than its silhouette area implies.",
                    "Ignores occlusion, reflectivity and dropped returns. Dark or wet "
                    "surfaces return less; the Mid-70's 90 m spec is at 10% reflectivity.",
                    "Multi-frame integration assumes a stationary sensor. While moving, "
                    "accumulation requires IMU deskewing, which the Mid-70 cannot supply "
                    "on its own.",
                    "An upper bound, not a prediction. Treat anything within ~2x of the "
                    "threshold as unproven until measured on real recordings.",
                ],
            },
            fh,
            indent=2,
        )
    print(f"\nWrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
