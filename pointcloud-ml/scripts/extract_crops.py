#!/usr/bin/env python3
"""Turn raw recordings into labelled-cluster crops.

    python scripts/extract_crops.py data/raw/session_01_hover_12m \
        --out data/processed

Runs ground removal then region proposal over every frame of every recording
given, and writes one ``<session>_crops.npz`` plus one ``<session>_crops.json``
per session. Reproducible from (raw recording, preprocess.yaml, seed) alone;
all three are recorded in the index.

Accepts a ``.db3`` bag, a rosbag2 directory, ``.npy``/``.npz``/``.pcd``/``.ply``,
or a directory of frame files. With ``--glob`` it walks a directory of
recordings, which is the normal case after a collection session.

``--summary`` prints the cluster-count and cluster-size distributions. Those are
the measurement that re-derives ``max_clusters_per_frame`` and
``min_cluster_points`` from PROVISIONAL to measured (``task_definition.md`` P6),
so run it on the first real recording before tuning anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from pcml.data.config import load_preprocess_config  # noqa: E402
from pcml.data.crops import (  # noqa: E402
    CropError,
    crop_geometry_summary,
    extract_session_crops,
)
from pcml.data.io import DEFAULT_LIVOX_TOPIC, RecordingError, open_recording  # noqa: E402

_FRAME_SUFFIXES = (".npy", ".npz", ".pcd", ".ply")
_RECORDING_SUFFIXES = (".db3", *_FRAME_SUFFIXES)


def _looks_like_a_recording_dir(p: Path) -> bool:
    """A rosbag2 directory, or a directory of per-frame files."""
    if not p.is_dir():
        return False
    if any(p.glob("*.db3")):
        return True
    return any(f.suffix.lower() in _FRAME_SUFFIXES for f in p.iterdir() if f.is_file())


def discover(paths: Sequence[str], use_glob: bool) -> list[Path]:
    """Resolve the command line into a list of recordings.

    Without ``--glob`` each argument **is** one recording. With ``--glob`` each
    directory argument is a container to look inside — the distinction is the
    flag's whole job, and it is not inferred from the contents. A collection
    directory typically holds a bag directory next to loose ``.npz`` files and a
    ``.meta.json`` sidecar for each, which is genuinely ambiguous; guessing there
    produced a single bogus session named after the directory.
    """
    out: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if not use_glob or not p.is_dir():
            out.append(p)
            continue
        for child in sorted(p.iterdir()):
            if child.is_file() and child.suffix.lower() in _RECORDING_SUFFIXES:
                out.append(child)
            elif _looks_like_a_recording_dir(child):
                out.append(child)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("recordings", nargs="+", help="recordings, or a directory of them")
    parser.add_argument(
        "--out", default="data/processed", help="output directory (default: data/processed)"
    )
    parser.add_argument(
        "--config",
        default="configs/preprocess.yaml",
        help="preprocessing config; its sha256 goes into every index",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="overrides the config's seed. Recorded in the index either way, "
        "because the crops are reproducible only from the pair",
    )
    parser.add_argument("--topic", default=DEFAULT_LIVOX_TOPIC, help="bag topic")
    parser.add_argument(
        "--glob",
        action="store_true",
        help="treat each directory argument as a directory *of* recordings",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="process every n-th frame. A 60 s hover is 600 near-identical "
        "frames of the same objects (§4.4), so a stride is often honest",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="stop after n frames")
    parser.add_argument(
        "--no-hash",
        action="store_true",
        help="skip hashing the raw recording (faster; the index then cannot prove "
        "which bytes it came from)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="print the cluster-count/size distributions (task_definition.md P6)",
    )
    parser.add_argument(
        "--summary-json", default=None, help="also write the summary to this path"
    )
    args = parser.parse_args(argv)

    config = load_preprocess_config(args.config)
    recordings = discover(args.recordings, args.glob)
    if not recordings:
        parser.error(f"no recordings found in {args.recordings}")

    print(f"preprocess.yaml sha256 {config.sha256[:12]}…  "
          f"seed {args.seed if args.seed is not None else config.seed}  "
          f"region_proposal.method {config.region_proposal.method}")
    print(f"eps_m {config.clustering.eps_m}  "
          f"min_cluster_points {config.clustering.min_cluster_points}  "
          f"max_clusters {config.clustering.max_clusters_per_frame}")
    print()

    all_crops = []
    seen: dict[str, str] = {}
    failures = 0

    for path in recordings:
        try:
            recording = open_recording(path, topic=args.topic)
        except (RecordingError, FileNotFoundError) as exc:
            print(f"SKIP {path}: {exc}")
            failures += 1
            continue

        if recording.session_id in seen:
            print(
                f"SKIP {path}: session_id {recording.session_id!r} already used by "
                f"{seen[recording.session_id]}. Splits are by session, so a "
                "collision would leak near-duplicate frames across splits. Give it "
                "a distinct session_id in its §4.2 metadata sidecar."
            )
            failures += 1
            continue

        if recording.meta is None:
            print(
                f"  ! {recording.session_id}: no §4.2 metadata sidecar. Altitude and "
                "mount angle cannot be reconstructed from the points later "
                "(docs/data_collection_plan.md §4.2)."
            )
        elif recording.meta.missing_fields():
            print(
                f"  ! {recording.session_id}: metadata is missing "
                f"{list(recording.meta.missing_fields())}"
            )

        try:
            index = extract_session_crops(
                recording,
                args.out,
                config=config,
                seed=args.seed,
                max_frames=args.max_frames,
                frame_stride=args.frame_stride,
                hash_raw=not args.no_hash,
            )
        except CropError as exc:
            print(f"SKIP {path}: {exc}")
            failures += 1
            continue

        seen[recording.session_id] = str(path)
        stats = index.stats
        print(
            f"{index.session_id:<28} {len(index.crops):>5} crops from "
            f"{stats['n_frames_processed']:>4} frames"
            + (
                f"   ({stats['n_frames_skipped_no_ground_plane']} frames skipped: no "
                "ground plane)"
                if stats["n_frames_skipped_no_ground_plane"]
                else ""
            )
        )
        if stats["ground_plane_tilt_deg_mean"] is not None:
            print(
                f"{'':<28} ground tilt mean "
                f"{stats['ground_plane_tilt_deg_mean']:.2f}°, inliers "
                f"{stats['ground_inlier_fraction_mean']:.0%}"
            )
        if stats["n_clusters_dropped_by_max_clusters_cap"]:
            print(
                f"{'':<28} ! {stats['n_clusters_dropped_by_max_clusters_cap']} cluster(s) "
                "dropped by the max_clusters cap — objects were silently discarded, "
                "which is why the count is reported"
            )
        all_crops.extend(index.crops)

    print()
    print(f"{len(seen)} session(s), {len(all_crops)} crops -> {args.out}")
    if failures:
        print(f"{failures} recording(s) skipped")

    if args.summary or args.summary_json:
        summary = crop_geometry_summary(all_crops)
        if args.summary:
            print()
            print("Distributions (task_definition.md P6: re-derive the caps from these)")
            print(json.dumps(summary, indent=2))
        if args.summary_json:
            Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.summary_json).write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
            print(f"summary -> {args.summary_json}")

    if not seen:
        return 1
    print()
    print("Next: label the crops, one decision per static object —")
    for session_id in sorted(seen):
        print(
            f"  python scripts/label_clusters.py "
            f"{Path(args.out) / (session_id + '_crops.json')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
