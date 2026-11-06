#!/usr/bin/env python3
"""Write train / val / test manifests, split BY SESSION.

    python scripts/make_splits.py --crops data/processed --out data/manifests

Splits are by **recording session**, never by frame. Consecutive Mid-70 frames
of one recording are near-duplicates, so a random frame split would put the same
object 100 ms apart on both sides of the boundary and the validation score would
measure memorisation (``docs/task_definition.md`` §7). This script has no option
for a frame-level split, on purpose.

Two defaults worth knowing, both from the collection plan:

* the **test** manifest takes ``annotated`` labels only (§5.3), so the final
  number is not partly a measure of the propagation heuristic;
* crops labelled ``unknown`` go to no manifest — the reject outcome is produced
  by a confidence threshold, not trained (§3). They are listed in
  ``splits.json`` so the threshold can be calibrated against them.

Use ``--pin SESSION:test`` to hold a recording out deliberately: a second
location for the domain-shift set (Q4), or the range-calibration recording
(§4.1), which is diagnostic and does not belong in training.
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

from pcml.data.crops import find_crop_indices, load_crop_index  # noqa: E402
from pcml.data.label_mapping import load_label_map  # noqa: E402
from pcml.data.labels import LabelStore  # noqa: E402
from pcml.data.splits import (  # noqa: E402
    SPLIT_NAMES,
    SplitError,
    assign_sessions_to_splits,
    class_distribution_report,
    format_class_distribution_report,
    write_manifests,
)


def parse_pins(values: Sequence[str]) -> dict[str, str]:
    pins: dict[str, str] = {}
    for value in values:
        if ":" not in value:
            raise SystemExit(f"--pin expects SESSION:SPLIT, got {value!r}")
        session, split = value.rsplit(":", 1)
        if split not in SPLIT_NAMES:
            raise SystemExit(
                f"--pin {value!r}: split must be one of {SPLIT_NAMES}, got {split!r}"
            )
        pins[session] = split
    return pins


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--crops", default="data/processed", help="directory of *_crops.json indices"
    )
    parser.add_argument("--out", default="data/manifests", help="where the manifests go")
    parser.add_argument(
        "--labels-dir",
        default=None,
        help="directory of *_labels.json stores (default: alongside the crop indices)",
    )
    parser.add_argument("--label-map", default="data/label_map.yaml")
    parser.add_argument(
        "--fractions",
        default="0.7,0.15,0.15",
        help="train,val,test session fractions (default: 0.7,0.15,0.15)",
    )
    parser.add_argument("--seed", type=int, default=0, help="split shuffle seed")
    parser.add_argument(
        "--pin",
        action="append",
        default=[],
        metavar="SESSION:SPLIT",
        help="force a session into a split; repeatable",
    )
    parser.add_argument(
        "--allow-propagated-in-test",
        action="store_true",
        help="let the test split use propagated labels. §5.3 says not to: the "
        "number would be partly a measure of the propagation radius",
    )
    parser.add_argument(
        "--include-ambiguous",
        action="store_true",
        help="keep crops whose centroid match had a second object inside the radius",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="also write the class-distribution report here (e.g. reports/splits.md)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the assignment and write nothing"
    )
    args = parser.parse_args(argv)

    try:
        fractions = tuple(float(v) for v in args.fractions.split(","))
    except ValueError:
        raise SystemExit(f"--fractions must be three numbers, got {args.fractions!r}")
    if len(fractions) != 3:
        raise SystemExit(f"--fractions needs exactly three values, got {fractions}")

    label_map = load_label_map(args.label_map)
    if label_map.is_example:
        print(
            "WARNING: the label map declares `example: true`, so no number computed "
            "against it is a result.\n"
        )

    index_paths = find_crop_indices(args.crops)
    if not index_paths:
        raise SystemExit(
            f"no *_crops.json under {args.crops}. Run scripts/extract_crops.py first."
        )
    indices = [load_crop_index(p) for p in index_paths]

    labels_root = Path(args.labels_dir) if args.labels_dir else None
    stores: dict[str, LabelStore] = {}
    unlabelled_sessions: list[str] = []
    for index in indices:
        directory = labels_root or index.index_path.parent
        store_path = LabelStore.path_for(directory, index.session_id)
        if store_path.exists():
            stores[index.session_id] = LabelStore.load(store_path)
        else:
            unlabelled_sessions.append(index.session_id)

    print(f"{len(indices)} session(s), {len(stores)} with labels")
    if unlabelled_sessions:
        print(f"  no label store yet: {unlabelled_sessions}")

    try:
        assignment = assign_sessions_to_splits(
            [i.session_id for i in indices],
            fractions=fractions,
            seed=args.seed,
            pinned=parse_pins(args.pin),
        )
    except SplitError as exc:
        raise SystemExit(f"cannot split: {exc}")

    print()
    print("Assignment (BY SESSION — docs/task_definition.md §7):")
    for name in SPLIT_NAMES:
        sessions = getattr(assignment, name)
        print(f"  {name:<6} {len(sessions):>3} session(s)  {list(sessions)}")
    if assignment.pinned:
        print(f"  pinned: {assignment.pinned}")

    if args.dry_run:
        print("\nDRY RUN: nothing written.")
        return 0

    try:
        written = write_manifests(
            indices,
            stores,
            label_map,
            out_dir=args.out,
            assignment=assignment,
            test_annotated_only=not args.allow_propagated_in_test,
            exclude_ambiguous=not args.include_ambiguous,
        )
    except SplitError as exc:
        raise SystemExit(f"cannot write manifests: {exc}")

    print()
    for name in SPLIT_NAMES:
        doc = json.loads(written[name].read_text(encoding="utf-8"))
        print(f"  {written[name]}  {doc['n_crops']} crops")
    print(f"  {written['splits']}")

    splits_doc = json.loads(written["splits"].read_text(encoding="utf-8"))
    dropped = {k: v for k, v in splits_doc["dropped"].items() if v}
    if dropped:
        print()
        print(f"dropped: {dropped}")
    if splits_doc["reject_labelled_crops"]:
        print(
            f"{len(splits_doc['reject_labelled_crops'])} crop(s) labelled "
            f"{label_map.reserved_class_name!r} were excluded from training and "
            "listed in splits.json — they are what the reject threshold is "
            "calibrated against."
        )

    report = class_distribution_report(
        {name: written[name] for name in SPLIT_NAMES}, label_map
    )
    text = format_class_distribution_report(report)
    print()
    print(text)
    if args.report:
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        print(f"\nreport -> {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
