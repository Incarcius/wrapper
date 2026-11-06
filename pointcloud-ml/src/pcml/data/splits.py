"""Train / val / test manifests, split **by recording session**.

``docs/task_definition.md`` §7 states it and this module enforces it: splitting
by random frame would put frame 211 of a hover in train and frame 212 in val.
Those two frames are the same objects, at the same ranges, 100 ms apart — the
validation score would measure memorisation and would look excellent while the
model had learned nothing that survives a new recording.

So session assignment is the only unit here, and
``test_no_session_appears_in_two_splits`` in ``tests/test_splits.py`` is the test
that holds the line. The function below cannot express a frame-level split; there
is no argument for one.

Two further rules, both from the collection plan:

* **The test split uses ``annotated`` labels only** (§5.3). Propagated labels are
  a positional heuristic; evaluating on them would make the headline number
  partly a measure of the propagation radius. On by default, and switching it off
  requires saying so explicitly.
* **Crops whose centroid match was ambiguous are excluded by default** — two
  labelled objects inside the match radius means the label could belong to
  either.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

import numpy as np

from .crops import CropIndex, write_json_atomic
from .label_mapping import LabelMap
from .labels import PROVENANCE_ANNOTATED, LabelStore

__all__ = [
    "SplitError",
    "MANIFEST_VERSION",
    "SPLIT_NAMES",
    "SplitAssignment",
    "assign_sessions_to_splits",
    "write_manifests",
    "load_manifest_sessions",
    "class_distribution_report",
    "format_class_distribution_report",
]

MANIFEST_VERSION = 1
SPLIT_NAMES = ("train", "val", "test")


class SplitError(RuntimeError):
    """A split that would leak between train and test, or cannot be built."""


@dataclass(frozen=True)
class SplitAssignment:
    """Which sessions went to which split. The thing under version control."""

    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]
    seed: int
    fractions: tuple[float, float, float]
    pinned: Mapping[str, str]

    def __post_init__(self) -> None:
        # Belt and braces: the invariant is also a test, but a corrupted hand-edit
        # of a manifest should not get as far as a training run.
        seen: dict[str, str] = {}
        for split in SPLIT_NAMES:
            for session in getattr(self, split):
                if session in seen:
                    raise SplitError(
                        f"session {session!r} is in both {seen[session]!r} and "
                        f"{split!r}. Consecutive frames of one recording are "
                        "near-duplicates, so a session in two splits leaks the test "
                        "set into training (docs/task_definition.md §7)."
                    )
                seen[session] = split

    def split_of(self, session_id: str) -> Optional[str]:
        for split in SPLIT_NAMES:
            if session_id in getattr(self, split):
                return split
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "fractions": list(self.fractions),
            "pinned": dict(self.pinned),
            "train": list(self.train),
            "val": list(self.val),
            "test": list(self.test),
        }


def assign_sessions_to_splits(
    session_ids: Iterable[str],
    fractions: Sequence[float] = (0.7, 0.15, 0.15),
    seed: int = 0,
    pinned: Optional[Mapping[str, str]] = None,
) -> SplitAssignment:
    """Assign whole sessions to train / val / test.

    Args:
        session_ids: all available session ids.
        fractions: ``(train, val, test)``, normalised if they do not sum to 1.
        seed: shuffle seed. The assignment is reproducible from
            ``(sorted session ids, fractions, seed)`` alone, which is why the
            sort happens before the shuffle — dict or filesystem ordering must
            not be able to change a split.
        pinned: ``{session_id: split}`` forced assignments. This is how a
            deliberately held-out recording stays held out: a second location
            for the domain-shift set (``docs/data_collection_plan.md`` Q4), or
            the range-calibration session (§4.1), which is diagnostic and does
            not belong in training.

    Raises:
        SplitError: if there are fewer sessions than splits with a non-zero
            fraction. An empty val or test split is a silent experiment failure:
            every metric comes back as a vacuous number, so it fails here
            instead.
    """
    ids = sorted(set(session_ids))
    if not ids:
        raise SplitError("no sessions to split")

    if len(fractions) != 3:
        raise SplitError(f"fractions must be (train, val, test), got {fractions!r}")
    if any(f < 0 for f in fractions):
        raise SplitError(f"fractions must be non-negative, got {fractions!r}")
    total = float(sum(fractions))
    if total <= 0:
        raise SplitError("fractions must not all be zero")
    norm = tuple(float(f) / total for f in fractions)

    pinned = dict(pinned or {})
    for session, split in pinned.items():
        if split not in SPLIT_NAMES:
            raise SplitError(
                f"pinned session {session!r} asks for split {split!r}; valid: {SPLIT_NAMES}"
            )
        if session not in ids:
            raise SplitError(
                f"pinned session {session!r} is not among the available sessions {ids}"
            )

    free = [s for s in ids if s not in pinned]
    wanted = [i for i, f in enumerate(norm) if f > 0]
    n_pinned_splits = {SPLIT_NAMES.index(v) for v in pinned.values()}
    need_from_free = [i for i in wanted if i not in n_pinned_splits]
    if len(free) < len(need_from_free):
        raise SplitError(
            f"{len(free)} unpinned session(s) {free} cannot fill "
            f"{[SPLIT_NAMES[i] for i in need_from_free]}. Splits are by session and "
            "an empty split makes every metric from it vacuous. Record more sessions, "
            "or set a fraction to 0 to say explicitly that a split is not wanted."
        )

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(free))
    shuffled = [free[i] for i in order]

    buckets: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    for session, split in pinned.items():
        buckets[split].append(session)

    # One session each to the splits that need one, so no non-zero split is empty;
    # then the remainder proportionally, largest remainder first.
    cursor = 0
    for i in need_from_free:
        buckets[SPLIT_NAMES[i]].append(shuffled[cursor])
        cursor += 1

    remaining = shuffled[cursor:]
    if remaining:
        exact = [norm[i] * len(remaining) for i in range(3)]
        counts = [int(math.floor(e)) for e in exact]
        leftover = len(remaining) - sum(counts)
        # Hand out the leftover by descending fractional part; ties by split order,
        # so the result is deterministic rather than dict-order dependent.
        order_by_remainder = sorted(
            range(3), key=lambda i: (-(exact[i] - counts[i]), i)
        )
        for i in order_by_remainder[:leftover]:
            counts[i] += 1
        pos = 0
        for i in range(3):
            if norm[i] <= 0:
                continue
            buckets[SPLIT_NAMES[i]] += remaining[pos : pos + counts[i]]
            pos += counts[i]
        # Anything left over by a rounding quirk goes to train, never to test.
        buckets["train"] += remaining[pos:]

    return SplitAssignment(
        train=tuple(sorted(buckets["train"])),
        val=tuple(sorted(buckets["val"])),
        test=tuple(sorted(buckets["test"])),
        seed=int(seed),
        fractions=(norm[0], norm[1], norm[2]),
        pinned=pinned,
    )


# ======================================================================
def _index_file_ref(index: CropIndex, out_dir: Path) -> str:
    """Path to a crop index, relative to the manifest directory when possible.

    Relative keeps a manifest valid after the project is moved or cloned, which
    matters because manifests are version-controlled and crop stores are not.
    """
    try:
        return index.index_path.resolve().relative_to(out_dir.resolve()).as_posix()
    except ValueError:
        try:
            import os

            return Path(
                os.path.relpath(index.index_path.resolve(), out_dir.resolve())
            ).as_posix()
        except ValueError:  # pragma: no cover - different drives on Windows
            return index.index_path.resolve().as_posix()


def write_manifests(
    indices: Sequence[CropIndex],
    stores: Mapping[str, LabelStore],
    label_map: LabelMap,
    out_dir: Union[str, Path] = "data/manifests",
    assignment: Optional[SplitAssignment] = None,
    fractions: Sequence[float] = (0.7, 0.15, 0.15),
    seed: int = 0,
    pinned: Optional[Mapping[str, str]] = None,
    test_annotated_only: bool = True,
    exclude_ambiguous: bool = True,
    include_reject_class: bool = False,
    preprocess_sha256: Optional[str] = None,
) -> dict[str, Path]:
    """Write ``{train,val,test}.json`` plus ``splits.json``.

    Args:
        indices: one crop index per session.
        stores: ``session_id -> LabelStore``. A session with no store contributes
            nothing and is reported in ``splits.json``.
        label_map: the vocabulary. Its sha256 goes into every manifest.
        out_dir: where the manifests go. Version-controlled (``data/README.md``).
        assignment: a pre-computed assignment; otherwise one is computed from
            ``fractions``/``seed``/``pinned``.
        test_annotated_only: §5.3 — the test manifest takes ``annotated`` labels
            only, so the final number is not partly a measure of the propagation
            heuristic. Turning this off is a deliberate act.
        exclude_ambiguous: drop crops whose centroid match had a second labelled
            object inside the radius.
        include_reject_class: by default crops labelled with the reject id are
            **not** written to any manifest — the reject outcome is produced by a
            confidence threshold, not trained (``task_definition.md`` §3). Those
            crops are what the threshold is calibrated against, and they are
            listed in ``splits.json`` so they can be found.
        preprocess_sha256: recorded so a dataset can refuse a manifest built
            against different clustering parameters. Taken from the indices when
            not given.

    Returns:
        ``{split_name: path}``, plus ``"splits"`` for ``splits.json``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    by_session = {index.session_id: index for index in indices}
    if len(by_session) != len(indices):
        raise SplitError(
            "two crop indices share a session_id; session ids are what the splits "
            "are made by, so this would leak frames across splits"
        )

    if assignment is None:
        assignment = assign_sessions_to_splits(
            by_session, fractions=fractions, seed=seed, pinned=pinned
        )

    assigned = set(assignment.train) | set(assignment.val) | set(assignment.test)
    unknown = sorted(assigned - set(by_session))
    if unknown:
        raise SplitError(f"assignment names session(s) with no crop index: {unknown}")

    shas = {i.preprocess_sha256 for i in indices if i.preprocess_sha256}
    if preprocess_sha256 is None:
        if len(shas) > 1:
            raise SplitError(
                "the crop indices were produced with different preprocess.yaml "
                f"versions ({sorted(s[:12] for s in shas)}). Mixing them would put "
                "clusters from different DBSCAN parameters in one dataset. "
                "Re-extract everything with one config."
            )
        preprocess_sha256 = next(iter(shas), None)

    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLIT_NAMES}
    dropped: dict[str, int] = {
        "no_label_store": 0,
        "unlabelled": 0,
        "ambiguous": 0,
        "reject_class": 0,
        "propagated_in_test": 0,
        "session_not_assigned": 0,
    }
    reject_crops: list[str] = []
    sessions_without_store: list[str] = []

    for session_id, index in sorted(by_session.items()):
        split = assignment.split_of(session_id)
        if split is None:
            dropped["session_not_assigned"] += len(index.crops)
            continue
        store = stores.get(session_id)
        if store is None:
            sessions_without_store.append(session_id)
            dropped["no_label_store"] += len(index.crops)
            continue

        ref = _index_file_ref(index, out)
        for crop in index.crops:
            label = store.labels.get(crop.crop_id)
            if label is None:
                dropped["unlabelled"] += 1
                continue
            if exclude_ambiguous and label.ambiguous:
                dropped["ambiguous"] += 1
                continue
            if not label_map.is_trained_class(label.label_id):
                reject_crops.append(crop.crop_id)
                if not include_reject_class:
                    dropped["reject_class"] += 1
                    continue
            if split == "test" and test_annotated_only and not label.is_annotated:
                dropped["propagated_in_test"] += 1
                continue

            rows[split].append(
                {
                    "crop_id": crop.crop_id,
                    "session_id": session_id,
                    "index_file": ref,
                    "label_id": label.label_id,
                    "label_name": label.label_name,
                    "provenance": label.provenance,
                    "object_id": label.object_id,
                    "n_points": crop.n_points,
                    "frame_index": crop.frame_index,
                }
            )

    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    written: dict[str, Path] = {}

    for split in SPLIT_NAMES:
        sessions = list(getattr(assignment, split))
        counts: dict[str, int] = {}
        provenance: dict[str, int] = {}
        for row in rows[split]:
            counts[row["label_name"]] = counts.get(row["label_name"], 0) + 1
            provenance[row["provenance"]] = provenance.get(row["provenance"], 0) + 1

        path = out / f"{split}.json"
        write_json_atomic(
            path,
            {
                "version": MANIFEST_VERSION,
                "split": split,
                "created": created,
                "split_seed": assignment.seed,
                "split_by": "session",
                "label_map": str(label_map.path) if label_map.path else None,
                "label_map_sha256": label_map.sha256,
                "label_map_is_example": label_map.is_example,
                "preprocess_sha256": preprocess_sha256,
                "num_classes": label_map.num_classes,
                "class_names": list(label_map.names),
                "annotated_only": bool(split == "test" and test_annotated_only),
                "exclude_ambiguous": exclude_ambiguous,
                "sessions": sessions,
                "n_sessions": len(sessions),
                "n_crops": len(rows[split]),
                "class_counts": counts,
                "provenance_counts": provenance,
                "crops": rows[split],
            },
        )
        written[split] = path

    splits_path = out / "splits.json"
    write_json_atomic(
        splits_path,
        {
            "version": MANIFEST_VERSION,
            "created": created,
            "split_by": "session",
            "rationale": (
                "Consecutive Mid-70 frames of one recording are near-duplicates; a "
                "random frame split would measure memorisation "
                "(docs/task_definition.md §7)."
            ),
            "assignment": assignment.to_dict(),
            "label_map_sha256": label_map.sha256,
            "label_map_is_example": label_map.is_example,
            "preprocess_sha256": preprocess_sha256,
            "test_annotated_only": test_annotated_only,
            "exclude_ambiguous": exclude_ambiguous,
            "include_reject_class": include_reject_class,
            "n_crops_per_split": {s: len(rows[s]) for s in SPLIT_NAMES},
            "dropped": dropped,
            "sessions_without_label_store": sessions_without_store,
            "reject_labelled_crops": reject_crops,
        },
    )
    written["splits"] = splits_path
    return written


def load_manifest_sessions(path: Union[str, Path]) -> set[str]:
    """Session ids named by a manifest. Used by the leakage test."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return set(doc.get("sessions", []))


# ======================================================================
def class_distribution_report(
    manifest_paths: Mapping[str, Union[str, Path]],
    label_map: LabelMap,
) -> dict[str, Any]:
    """Class distribution across the splits, with the warnings that matter.

    Not a cosmetic summary. Three things in this project depend on it:

    * **Class weights** for the loss — ``§2`` expects severe imbalance (trees and
      poles in every frame, vehicles rarely), and the weights are computed from
      the *training* distribution, never guessed.
    * **§4.4's counting mistake** — frames are not samples. A class with 4,000
      crops from 2 distinct objects is 2 objects, and macro-F1 on it will not
      survive a new location. ``distinct_objects`` is reported beside the crop
      count so the two cannot be confused.
    * **Empty classes** — a class with no examples in train, or none in test, is
      an experiment that cannot report a number for it.
    """
    report: dict[str, Any] = {
        "label_map": str(label_map.path) if label_map.path else None,
        "label_map_sha256": label_map.sha256,
        "label_map_is_example": label_map.is_example,
        "num_classes": label_map.num_classes,
        "splits": {},
        "warnings": [],
    }

    inverse_freq: dict[str, float] = {}

    for split, path in manifest_paths.items():
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        counts = {name: 0 for name in label_map.names}
        provenance: dict[str, int] = {}
        objects_per_class: dict[str, set[Any]] = {name: set() for name in label_map.names}

        for row in doc.get("crops", []):
            name = row["label_name"]
            counts[name] = counts.get(name, 0) + 1
            provenance[row["provenance"]] = provenance.get(row["provenance"], 0) + 1
            if row.get("object_id") is not None:
                objects_per_class.setdefault(name, set()).add(
                    (row["session_id"], row["object_id"])
                )

        total = sum(counts.values())
        report["splits"][split] = {
            "n_sessions": doc.get("n_sessions", len(doc.get("sessions", []))),
            "sessions": doc.get("sessions", []),
            "n_crops": total,
            "class_counts": counts,
            "class_fractions": {
                name: (n / total if total else 0.0) for name, n in counts.items()
            },
            "distinct_objects_per_class": {
                name: len(ids) for name, ids in objects_per_class.items()
            },
            "provenance_counts": provenance,
        }

        empty = sorted(name for name, n in counts.items() if n == 0)
        if empty:
            report["warnings"].append(
                f"{split}: no examples at all for {empty}. No metric can be reported "
                "for those classes; either collect them or merge them into the reject "
                "outcome (docs/data_collection_plan.md §4.4)."
            )

        thin = sorted(
            name
            for name, ids in objects_per_class.items()
            if 0 < len(ids) < 10 and counts[name] > 0
        )
        if thin:
            report["warnings"].append(
                f"{split}: fewer than 10 distinct physical objects for {thin}. §4.4: "
                "below ~10 instances the model memorises those specific objects and "
                "the score will not survive a new location, however many crops there "
                "are."
            )

        if split == "train" and total:
            present = {name: n for name, n in counts.items() if n > 0}
            if present:
                # Inverse-frequency weights, mean-normalised so the loss scale is
                # unchanged. A starting point for the weighted loss, to be
                # recorded in label_map.yaml once the real data exists.
                raw = {name: total / (len(present) * n) for name, n in present.items()}
                mean = sum(raw.values()) / len(raw)
                inverse_freq = {name: w / mean for name, w in raw.items()}

    if inverse_freq:
        report["suggested_class_weights_from_train"] = {
            name: round(inverse_freq.get(name, 0.0), 4) for name in label_map.names
        }
        report["suggested_class_weights_note"] = (
            "Mean-normalised inverse frequency over classes present in train. A "
            "starting point, not a decision: copy into label_map.yaml:class_weights "
            "only after checking it against macro-F1 on val."
        )

    if label_map.is_example:
        report["warnings"].append(
            "the label map declares example: true. It is the unconfirmed proposal "
            "from docs/data_collection_plan.md §2, so no number computed against it "
            "is a result."
        )

    return report


def format_class_distribution_report(report: Mapping[str, Any]) -> str:
    """The report as a text table, for stdout and for ``reports/``."""
    lines = [
        "Class distribution by split (split BY SESSION, docs/task_definition.md §7)",
        "",
        f"label map: {report.get('label_map')}  "
        f"sha256 {str(report.get('label_map_sha256'))[:12]}…"
        + ("   *** EXAMPLE MAP ***" if report.get("label_map_is_example") else ""),
        "",
    ]

    splits = report.get("splits", {})
    names: list[str] = []
    for data in splits.values():
        for name in data.get("class_counts", {}):
            if name not in names:
                names.append(name)

    header = f"{'class':<14}" + "".join(f"{s:>22}" for s in splits)
    lines.append(header)
    lines.append("-" * len(header))
    for name in names:
        row = f"{name:<14}"
        for data in splits.values():
            n = data["class_counts"].get(name, 0)
            objs = data.get("distinct_objects_per_class", {}).get(name, 0)
            cell = "{} crops / {} obj".format(n, objs)
            row += f"{cell:>22}"
        lines.append(row)
    lines.append("-" * len(header))
    totals = f"{'TOTAL':<14}"
    for data in splits.values():
        cell = str(data["n_crops"]) + " crops"
        totals += f"{cell:>22}"
    lines.append(totals)
    sessions = f"{'sessions':<14}"
    for data in splits.values():
        sessions += f"{data['n_sessions']:>22}"
    lines.append(sessions)

    if report.get("suggested_class_weights_from_train"):
        lines += ["", "suggested class_weights (from train, inverse frequency):"]
        for name, w in report["suggested_class_weights_from_train"].items():
            lines.append(f"  {name:<14} {w:.4f}")
        lines.append(f"  note: {report['suggested_class_weights_note']}")

    warnings = report.get("warnings") or []
    if warnings:
        lines += ["", "WARNINGS:"]
        lines += [f"  - {w}" for w in warnings]
    return "\n".join(lines)
