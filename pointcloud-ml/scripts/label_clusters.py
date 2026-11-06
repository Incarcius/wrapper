#!/usr/bin/env python3
"""Label cluster crops, once per object, and propagate by position (§5.2).

    python scripts/label_clusters.py data/processed/session_01_crops.json \
        --label-map data/label_map.yaml

The tool walks the session's crops in frame order. For each crop it first asks
the label store whether an already-annotated object sits within the match radius;
if one does, the label is **propagated** and nothing is shown. Only an unmatched
cluster reaches the human. On a 60-second hover over 8 static objects that is 8
decisions for ~4,800 labelled crops, which is the difference between a feasible
afternoon and an infeasible month.

Text-first on purpose. It runs over ssh on a field laptop with no display: each
cluster is printed as geometry in metres plus two coarse ASCII projections, and
the answer is a single keypress. ``--viewer`` adds an Open3D window **if** Open3D
imports, which in this environment it does not (missing ``libusb``), so the text
path is the one that has to work and is the one the tests cover.

Resumable by construction: the label store is written after every decision, so
Ctrl-C costs at most the cluster on screen. Re-running picks up exactly where it
stopped, re-propagating from the objects already annotated.

Other modes:
  --report                 print the label summary and exit, change nothing
  --dry-run                propagate and report, write nothing
  --repropagate            re-match every propagated label (after --radius changed
                           or an object was moved), leaving annotations alone
  --write-meta-template P  write a blank §4.2 session metadata sidecar and exit
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

import numpy as np  # noqa: E402

from pcml.data.crops import CropIndex, CropRecord, load_crop_index  # noqa: E402
from pcml.data.io import RecordingMeta  # noqa: E402
from pcml.data.label_mapping import LabelMap, load_label_map  # noqa: E402
from pcml.data.labels import (  # noqa: E402
    DEFAULT_MATCH_RADIUS_M,
    KNOWN_ATTRIBUTES,
    LabelStore,
    describe_crop,
)

QUIT = "__quit__"
SKIP = "__skip__"
BACK = "__back__"


def attributes_for_class(label_name: str) -> dict[str, tuple[type, str]]:
    """The annotation attributes this class's policy asks for.

    Derived from :data:`~pcml.data.labels.KNOWN_ATTRIBUTES` rather than written
    out here, so adding a policy attribute does not mean editing the tool.
    """
    return {
        key: (value_type, why)
        for key, (owner, value_type, why) in KNOWN_ATTRIBUTES.items()
        if owner == label_name
    }


def prompt_attributes(label_name: str, out=None) -> dict[str, object]:
    """Ask for the attributes the class's annotation policy requires.

    Every one is optional — pressing Enter skips it. An annotator who is unsure
    should leave it blank rather than guess, because a wrong ``merged_instances``
    is worse than a missing one: it would be reported as fact.
    """
    import sys as _sys

    out = out or _sys.stdout
    wanted = attributes_for_class(label_name)
    if not wanted:
        return {}

    collected: dict[str, object] = {}
    for key, (value_type, why) in sorted(wanted.items()):
        if value_type is bool:
            hint = "y/n, Enter to skip"
        elif value_type is int:
            hint = "a number, Enter to skip"
        else:
            hint = "text, Enter to skip"
        out.write(f"  {key}? ({hint})\n     {why}\n")
        out.flush()
        raw = _sys.stdin.readline()
        if not raw:
            break
        raw = raw.strip()
        if not raw:
            continue
        if value_type is bool:
            if raw[:1].lower() in ("y", "1", "t"):
                collected[key] = True
            elif raw[:1].lower() in ("n", "0", "f"):
                collected[key] = False
            else:
                out.write(f"     (not a yes/no: {raw!r} ignored)\n")
            continue
        if value_type is int:
            try:
                collected[key] = int(raw)
            except ValueError:
                out.write(f"     (not a number: {raw!r} ignored)\n")
            continue
        collected[key] = raw
    return collected


# ----------------------------------------------------------------------
def read_key(prompt: str, stream=None) -> str:
    """One keypress, or one line when stdin is not a terminal.

    The raw-terminal path makes labelling fast — a key per cluster, no Enter.
    The line-based fallback is what runs under a pipe, in CI and in the tests,
    and it accepts the same characters, so the two paths cannot diverge in
    behaviour.
    """
    out = stream or sys.stdout
    out.write(prompt)
    out.flush()

    if sys.stdin is not None and sys.stdin.isatty():
        try:  # POSIX
            import termios
            import tty

            fd = sys.stdin.fileno()
            saved = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                ch = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, saved)
            out.write(ch + "\n")
            out.flush()
            return ch
        except Exception:  # noqa: BLE001 - fall through to line input
            pass
        try:  # Windows
            import msvcrt

            ch = msvcrt.getwch()
            out.write(ch + "\n")
            out.flush()
            return ch
        except Exception:  # noqa: BLE001
            pass

    line = sys.stdin.readline()
    if not line:  # EOF: treat as quit-and-save rather than crash
        out.write("\n(end of input)\n")
        return "q"
    return line.strip()[:1] or "\n"


def menu_lines(label_map: LabelMap) -> list[str]:
    lines = ["  keys:"]
    for entry in label_map.classes:
        extra = attributes_for_class(entry.name)
        suffix = f"   [will ask: {', '.join(sorted(extra))}]" if extra else ""
        lines.append(f"    {entry.class_id}  {entry.name}{suffix}")
    lines.append(
        f"    u  {label_map.reserved_class_name} "
        f"(id {label_map.reserved_class_id}) — a cluster that is none of the "
        "classes. Recorded, not discarded: with reject.trained_as_class false "
        "these are what the confidence threshold is calibrated against."
    )
    lines.append("    m  moving object: label this frame only, do NOT propagate")
    lines.append("    s  skip this cluster (remembered; not shown again)")
    lines.append("    q  save and quit")
    return lines


def resolve_key(key: str, label_map: LabelMap) -> Optional[tuple[int, bool]]:
    """Map a keypress to ``(label_id, propagate)``, or a sentinel, or None.

    Returns ``None`` for an unrecognised key so the caller can re-prompt rather
    than guess — a mis-keyed class that silently becomes class 0 is worse than
    an extra keystroke.
    """
    if key in ("q", "Q", "\x03", "\x04"):  # q, Ctrl-C, Ctrl-D
        return (QUIT, False)  # type: ignore[return-value]
    if key in ("s", "S"):
        return (SKIP, False)  # type: ignore[return-value]
    if key in ("u", "U"):
        return (label_map.reserved_class_id, True)
    if key in ("m", "M"):
        return (BACK, False)  # type: ignore[return-value]
    if key.isdigit():
        class_id = int(key)
        if label_map.is_trained_class(class_id):
            return (class_id, True)
    return None


def show_in_viewer(points: np.ndarray, title: str) -> bool:
    """Open an Open3D window. Returns False when Open3D is unavailable.

    Guarded with a bare ``except`` and not only ``ImportError``: in this
    environment Open3D is installed but raises ``OSError`` for a missing
    ``libusb-1.0.so.0``, so an ``ImportError``-only guard would still crash the
    labelling session.
    """
    try:
        import open3d as o3d
    except Exception as exc:  # noqa: BLE001
        print(f"  [viewer unavailable: {type(exc).__name__}: {exc}]")
        return False
    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
        o3d.visualization.draw_geometries([pcd], window_name=title)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  [viewer failed: {type(exc).__name__}: {exc}]")
        return False


# ----------------------------------------------------------------------
def ordered_crops(index: CropIndex) -> list[CropRecord]:
    """Frame order, and within a frame largest cluster first.

    Frame order matters: §5.2 only works if the first frame's objects get
    annotated before later frames are considered, so that everything after the
    first frame propagates.
    """
    return sorted(index.crops, key=lambda c: (c.frame_index, -c.n_points, c.crop_id))


def run_labelling(
    index: CropIndex,
    store: LabelStore,
    label_map: LabelMap,
    radius_m: float,
    limit: Optional[int] = None,
    use_viewer: bool = False,
    dry_run: bool = False,
    grid_width: int = 34,
    grid_height: int = 14,
) -> dict[str, int]:
    """The interactive loop. Returns a count breakdown."""
    crops = ordered_crops(index)
    counts = {"annotated": 0, "propagated": 0, "skipped": 0, "already": 0}
    menu = menu_lines(label_map)
    n_decisions = 0

    for crop in crops:
        if crop.crop_id in store.skipped:
            counts["skipped"] += 1
            continue
        if crop.crop_id in store.labels:
            counts["already"] += 1
            continue

        obj, distance, runner_up = store.match(crop.centroid_m, radius_m=radius_m)
        if obj is not None:
            result = store.propagate([crop], radius_m=radius_m)
            counts["propagated"] += result["propagated"]
            if not dry_run:
                store.save()
            continue

        if limit is not None and n_decisions >= limit:
            print(f"\nreached --limit {limit}; stopping. Re-run to continue.")
            break

        print("\n" + "=" * 68)
        print(f"UNMATCHED CLUSTER  ({counts['annotated']} annotated so far, "
              f"{counts['propagated']} propagated)")
        print("=" * 68)
        points = np.array(index.points_for(crop), dtype=np.float32)
        for line in describe_crop(crop, points, width=grid_width, height=grid_height):
            print(line)
        if store.objects:
            nearest = min(
                ((o.distance_to(crop.centroid_m), o) for o in store.objects.values()),
                key=lambda t: t[0],
            )
            print(
                f"nearest labelled object: #{nearest[1].object_id} "
                f"{nearest[1].label_name} at {nearest[0]:.2f} m "
                f"(match radius {radius_m:.2f} m)"
            )
        if use_viewer:
            show_in_viewer(points, crop.crop_id)
        print()
        for line in menu:
            print(line)

        while True:
            key = read_key("  > ")
            resolved = resolve_key(key, label_map)
            if resolved is None:
                print(f"  unrecognised key {key!r}; try again")
                continue
            # The label map's expected_extent_m is a cheap mislabel check: a 9 m
            # person or a 0.4 m structure is nearly always a mis-keyed class or a
            # merged cluster. Advisory — the annotator stays the authority, so a
            # confirmation keeps the label rather than the tool overriding it.
            candidate = resolved[0]
            if isinstance(candidate, int):
                warning = label_map.extent_warning(candidate, crop.largest_extent_m)
                if warning:
                    print(f"  ! {warning}")
                    print(
                        "    Enter to keep this label, or any other key to choose again"
                    )
                    if read_key("  > ").strip():
                        print("  choose again:")
                        continue
            break

        choice, propagate = resolved
        if choice == QUIT:
            print("\nsaving and quitting.")
            if not dry_run:
                store.save()
            break
        if choice == SKIP:
            store.skip(crop)
            counts["skipped"] += 1
            n_decisions += 1
            if not dry_run:
                store.save()
            continue
        if choice == BACK:  # 'm': this frame only, no object registered
            print("  marked as a moving object: which class?")
            for entry in label_map.classes:
                print(f"    {entry.class_id}  {entry.name}")
            while True:
                key = read_key("  > ")
                if key.isdigit() and label_map.is_trained_class(int(key)):
                    moving_attrs = prompt_attributes(label_map.name_for(int(key)))
                    store.annotate(
                        crop,
                        int(key),
                        label_map,
                        notes="moving",
                        new_object=False,
                        attributes=moving_attrs,
                    )
                    counts["annotated"] += 1
                    n_decisions += 1
                    if not dry_run:
                        store.save()
                    break
                print(f"  unrecognised key {key!r}; try again")
            continue

        attributes = prompt_attributes(label_map.name_for(int(choice)))
        store.annotate(
            crop, int(choice), label_map, radius_m=radius_m, attributes=attributes
        )
        counts["annotated"] += 1
        n_decisions += 1
        if not dry_run:
            store.save()

        # Propagate immediately, so the running counts on screen are honest
        # about how much work the last decision actually did.
        result = store.propagate(crops, radius_m=radius_m)
        counts["propagated"] += result["propagated"]
        if not dry_run:
            store.save()
        print(
            f"  recorded {label_map.name_for(int(choice))}; propagated to "
            f"{result['propagated']} further crop(s)"
        )

    return counts


# ----------------------------------------------------------------------
def print_report(index: CropIndex, store: LabelStore, label_map: LabelMap) -> None:
    summary = store.summary()
    total = len(index.crops)
    print(f"session            {store.session_id}")
    print(f"crop index         {index.index_path}")
    print(f"label store        {store.path}")
    print(f"label map          {label_map.path}"
          + ("   *** EXAMPLE MAP, not a confirmed vocabulary ***" if label_map.is_example else ""))
    print(f"match radius       {store.match_radius_m:.2f} m")
    print()
    print(f"crops in session   {total}")
    print(f"labelled           {summary['n_labels']}"
          + (f"  ({100.0 * summary['n_labels'] / total:.1f}%)" if total else ""))
    print(f"  annotated        {summary['n_annotated']}   (human decisions)")
    print(f"  propagated       {summary['n_propagated']}   (§5.2 centroid match)")
    print(f"  ambiguous        {summary['n_ambiguous']}   (two objects inside the radius)")
    print(f"skipped            {summary['n_skipped']}")
    print(f"unlabelled         {total - summary['n_labels'] - summary['n_skipped']}")
    print()
    print("distinct physical objects (§4.4: frames are not samples)")
    if summary["distinct_objects_per_class"]:
        for name, n in sorted(summary["distinct_objects_per_class"].items()):
            crops_for_class = summary["by_class"].get(name, {})
            n_crops = crops_for_class.get("annotated", 0) + crops_for_class.get(
                "propagated", 0
            )
            flag = "" if n >= 10 else "   <-- below §4.4's ~10 instance floor"
            print(f"  {name:<14} {n:>4} object(s)   {n_crops:>6} crop(s){flag}")
    else:
        print("  (none yet)")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "crops",
        nargs="?",
        help="crop index (*_crops.json) or a directory holding exactly one",
    )
    parser.add_argument(
        "--label-map",
        default="data/label_map.yaml",
        help="class vocabulary (default: data/label_map.yaml)",
    )
    parser.add_argument(
        "--labels-dir",
        default=None,
        help="where the label store lives (default: alongside the crop index)",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=DEFAULT_MATCH_RADIUS_M,
        help=f"centroid match radius in metres (default: {DEFAULT_MATCH_RADIUS_M})",
    )
    parser.add_argument("--limit", type=int, default=None, help="stop after N decisions")
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="also open an Open3D window per cluster (needs a working Open3D)",
    )
    parser.add_argument("--report", action="store_true", help="print the summary and exit")
    parser.add_argument(
        "--dry-run", action="store_true", help="never write the label store"
    )
    parser.add_argument(
        "--repropagate",
        action="store_true",
        help="re-match every propagated label (annotations are left alone), then exit",
    )
    parser.add_argument(
        "--write-meta-template",
        metavar="PATH",
        default=None,
        help="write a blank §4.2 session metadata sidecar and exit",
    )
    parser.add_argument("--grid-width", type=int, default=34)
    parser.add_argument("--grid-height", type=int, default=14)
    args = parser.parse_args(argv)

    if args.write_meta_template:
        path = Path(args.write_meta_template)
        path.parent.mkdir(parents=True, exist_ok=True)
        session_id = path.name.split(".")[0]
        path.write_text(
            json.dumps(RecordingMeta.template(session_id), indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {path}")
        print(
            "Fill it in AT CAPTURE TIME — docs/data_collection_plan.md §4.2: none of "
            "these fields can be reconstructed from the points afterwards."
        )
        return 0

    if not args.crops:
        parser.error("the crops argument is required unless --write-meta-template is used")

    index = load_crop_index(args.crops)
    label_map = load_label_map(args.label_map)
    labels_dir = Path(args.labels_dir) if args.labels_dir else index.index_path.parent
    store = LabelStore.load_or_create(
        labels_dir,
        index.session_id,
        match_radius_m=args.radius,
        label_map_sha256=label_map.sha256,
    )

    if label_map.is_example:
        print(
            "WARNING: this label map declares `example: true`. It is the unconfirmed "
            "proposal from docs/data_collection_plan.md §2. Annotation done against "
            "it will have to be re-mapped if the owner changes the class list.\n"
        )

    if args.report:
        print_report(index, store, label_map)
        return 0

    if args.repropagate:
        result = store.propagate(
            index.crops, radius_m=args.radius, overwrite_propagated=True
        )
        if not args.dry_run:
            store.save()
        print(f"re-propagated: {result}")
        print_report(index, store, label_map)
        return 0

    print(f"session {index.session_id}: {len(index.crops)} crops, "
          f"{len(store.labels)} already labelled, {len(store.objects)} objects known")
    print(f"label store: {store.path}"
          + ("   (DRY RUN: nothing will be written)" if args.dry_run else ""))

    counts = run_labelling(
        index,
        store,
        label_map,
        radius_m=args.radius,
        limit=args.limit,
        use_viewer=args.viewer,
        dry_run=args.dry_run,
        grid_width=args.grid_width,
        grid_height=args.grid_height,
    )
    print(f"\nthis run: {counts}")
    print()
    print_report(index, store, label_map)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
