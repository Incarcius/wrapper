"""The class vocabulary, loaded and validated from ``label_map.yaml``.

``K`` is derived from this file and from nowhere else. No model config, training
script or deployment adapter contains a class count — that is **[DECIDED]** in
``docs/task_definition.md`` §3, and the reason is simple: a checkpoint whose
``num_classes`` disagrees with the label map that produced its training labels is
silently wrong in a way no test on the model alone can catch.

The validation here is deliberately strict and loud. Every rule below corresponds
to a way the label map can silently invalidate a checkpoint:

* **Contiguous ids from 0** — the network's output layer is indexed by position,
  so a gap means an output unit that can never be correct, and a reordering
  remaps every label in every manifest that was written earlier.
* **The reserved id never appears as a class** — ``-1`` is the *output* of
  rejecting (``docs/task_definition.md`` §5), not something an annotator assigns.
  A trained class with id ``-1`` would collide with the reject outcome in the
  output contract.
* **Unique names** — the confusion matrix is read by name. Two classes sharing a
  name produce a report that cannot be interpreted.
* **Every class carries a definition and an annotation policy** — these are
  load-bearing, not documentation garnish. They are what makes two annotators
  agree, and what makes a confusion matrix interpretable six months later. A
  class without them is a class two people will label differently, so the map is
  rejected rather than loaded.

The vocabulary of record is ``data/label_map.yaml``, confirmed by the project
owner on 15 Oct 2026: ``person``, ``vegetation``, ``pole``, ``structure``,
``net_fence``, ``vehicle``, so ``K = 6``. ``-1`` / ``unknown`` is the reject
**outcome** and is not among them.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import yaml

__all__ = [
    "LabelMapError",
    "ClassEntry",
    "LabelMap",
    "load_label_map",
]


class LabelMapError(ValueError):
    """A label map that would silently invalidate a checkpoint or a manifest.

    Separate from ``ValueError`` at the call site so that training scripts can
    report "your label map is wrong" distinctly from "your arguments are wrong".
    """


@dataclass(frozen=True)
class ClassEntry:
    """One trained class.

    ``definition`` and ``annotation_policy`` are carried through rather than
    dropped at load time because they are what makes two annotators agree, and
    what makes a confusion matrix interpretable months later.
    """

    class_id: int
    name: str
    definition: Optional[str] = None
    annotation_policy: Optional[str] = None
    expected_extent_m: Optional[tuple[Optional[float], Optional[float]]] = None
    notes: Optional[str] = None


@dataclass(frozen=True)
class LabelMap:
    """The validated class vocabulary.

    Attributes:
        classes: trained classes, ordered by ``class_id`` (which equals index).
        reserved_class_id: the reject outcome's id, never a trained class.
        reserved_class_name: the name reported for a rejected cluster.
        confidence_threshold: below this, a prediction becomes the reject
            outcome. PROVISIONAL in every map so far; it is set from the
            precision/recall-vs-threshold curve, never guessed.
        reject_trained_as_class: whether negatives were collected and labelled
            as an explicit background class.
        sha256: hash of the file's bytes. Recorded in manifests and alongside
            checkpoints so that "which vocabulary was this trained with" is
            answerable from the artifacts rather than from memory.
        is_example: true when the file declares ``example: true``. Example maps
            must not reach a reported number, so every consumer can check.
    """

    classes: tuple[ClassEntry, ...]
    reserved_class_id: int = -1
    reserved_class_name: str = "unknown"
    confidence_threshold: float = 0.0
    reject_trained_as_class: bool = False
    class_weights: Optional[tuple[float, ...]] = None
    version: Optional[int] = None
    source: Optional[str] = None
    created: Optional[str] = None
    is_example: bool = False
    path: Optional[Path] = None
    sha256: Optional[str] = None
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    @property
    def num_classes(self) -> int:
        """``K``. The one place a class count comes from."""
        return len(self.classes)

    @property
    def names(self) -> tuple[str, ...]:
        """Class names, in id order. Index == class_id."""
        return tuple(c.name for c in self.classes)

    def name_for(self, class_id: int) -> str:
        """Resolve an id to a name, including the reject outcome.

        Raises:
            LabelMapError: for any id that is neither a trained class nor the
                reserved reject id. Returning a placeholder string instead would
                let an off-by-one in the head width reach a report.
        """
        if class_id == self.reserved_class_id:
            return self.reserved_class_name
        if 0 <= class_id < len(self.classes):
            return self.classes[class_id].name
        raise LabelMapError(
            f"class_id {class_id} is not in this label map "
            f"(valid: 0..{len(self.classes) - 1}, or {self.reserved_class_id} for "
            f"{self.reserved_class_name!r})"
        )

    def id_for(self, name: str) -> int:
        """Resolve a name to an id, including the reject outcome."""
        if name == self.reserved_class_name:
            return self.reserved_class_id
        for c in self.classes:
            if c.name == name:
                return c.class_id
        raise LabelMapError(
            f"class name {name!r} is not in this label map; known names: "
            f"{list(self.names) + [self.reserved_class_name]}"
        )

    def is_trained_class(self, class_id: int) -> bool:
        """True for a real output class; False for the reject id and for garbage.

        ``-1`` is a **valid label to store** (negatives are labelled ``unknown``
        so the reject threshold has something to calibrate against) but is not a
        valid training target. Those are different questions, which is why this
        is separate from :meth:`name_for` accepting the reserved id.
        """
        return 0 <= class_id < len(self.classes)

    def is_storable_label(self, class_id: int) -> bool:
        """True for anything an annotator may legitimately record.

        The trained classes plus the reject id. ``reject.trained_as_class`` is
        false in the confirmed map, so ``unknown`` crops are kept and excluded
        from the manifests rather than discarded at annotation time.
        """
        return self.is_trained_class(class_id) or class_id == self.reserved_class_id

    def extent_warning(
        self, class_id: int, largest_extent_m: float
    ) -> Optional[str]:
        """Warn when a cluster's size contradicts its label, else ``None``.

        ``expected_extent_m`` in the label map is a plausible ``[min, max]`` for
        a class's **largest** dimension. A 9 m ``person`` or a 0.4 m ``structure``
        is almost always a mis-keyed label or a merged cluster, and this catches
        it for the price of a comparison — far cheaper than finding it in a
        confusion matrix. Advisory only: it never blocks a label, because a
        genuinely unusual object is possible and the annotator is the authority.
        """
        if not self.is_trained_class(class_id):
            return None
        entry = self.classes[class_id]
        bounds = entry.expected_extent_m
        if bounds is None:
            return None
        lo, hi = bounds
        if lo is not None and largest_extent_m < lo:
            return (
                f"largest extent {largest_extent_m:.2f} m is BELOW the expected "
                f"minimum {lo:.2f} m for {entry.name!r} — a fragment, or the wrong "
                "class?"
            )
        if hi is not None and largest_extent_m > hi:
            return (
                f"largest extent {largest_extent_m:.2f} m is ABOVE the expected "
                f"maximum {hi:.2f} m for {entry.name!r} — merged with something, or "
                "the wrong class?"
            )
        return None


# ----------------------------------------------------------------------
def _require_mapping(obj: Any, what: str) -> Mapping[str, Any]:
    if not isinstance(obj, Mapping):
        raise LabelMapError(f"{what} must be a mapping, got {type(obj).__name__}")
    return obj


def _parse_extent(value: Any, class_name: str) -> Optional[tuple[Optional[float], Optional[float]]]:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise LabelMapError(
            f"class {class_name!r}: expected_extent_m must be a two-element "
            f"[min, max] list or null, got {value!r}"
        )
    out: list[Optional[float]] = []
    for v in value:
        out.append(None if v is None else float(v))
    lo, hi = out
    if lo is not None and hi is not None and lo > hi:
        raise LabelMapError(
            f"class {class_name!r}: expected_extent_m min {lo} exceeds max {hi}"
        )
    return (lo, hi)


def _parse_classes(entries: Any, reserved_id: int, reserved_name: str) -> tuple[ClassEntry, ...]:
    if entries is None or (isinstance(entries, Sequence) and len(entries) == 0):
        raise LabelMapError(
            "label map has no classes, so K would be 0. The confirmed vocabulary is "
            "data/label_map.yaml (6 classes); data/label_map.template.yaml is the "
            "empty structural template and is not loadable by design."
        )
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise LabelMapError(f"`classes` must be a list, got {type(entries).__name__}")

    parsed: list[ClassEntry] = []
    seen_ids: dict[int, str] = {}
    seen_names: dict[str, int] = {}

    for position, raw in enumerate(entries):
        entry = _require_mapping(raw, f"classes[{position}]")
        if "class_id" not in entry:
            raise LabelMapError(f"classes[{position}] is missing `class_id`")
        if "name" not in entry:
            raise LabelMapError(f"classes[{position}] is missing `name`")

        raw_id = entry["class_id"]
        if isinstance(raw_id, bool) or not isinstance(raw_id, int):
            raise LabelMapError(
                f"classes[{position}].class_id must be an integer, got {raw_id!r}"
            )
        class_id = int(raw_id)

        name = entry["name"]
        if not isinstance(name, str) or not name.strip():
            raise LabelMapError(
                f"classes[{position}].name must be a non-empty string, got {name!r}"
            )

        if class_id == reserved_id:
            raise LabelMapError(
                f"classes[{position}] uses the reserved class_id {reserved_id}, which "
                f"is the reject *outcome* and never a trained class "
                f"(docs/task_definition.md §3/§5). Renumber it, or drop the entry and "
                f"let the confidence threshold produce {reserved_name!r}."
            )
        if class_id < 0:
            raise LabelMapError(
                f"classes[{position}].class_id must be >= 0, got {class_id}; negative "
                "ids cannot index a softmax output"
            )
        if class_id in seen_ids:
            raise LabelMapError(
                f"duplicate class_id {class_id}: used by both {seen_ids[class_id]!r} "
                f"and {name!r}"
            )
        if name == reserved_name:
            raise LabelMapError(
                f"classes[{position}] is named {reserved_name!r}, which is reserved for "
                "the reject outcome. If you really want a trained background class, "
                "give it a different name (e.g. 'background') and set "
                "reject.trained_as_class: true."
            )
        if name in seen_names:
            raise LabelMapError(
                f"duplicate class name {name!r} (ids {seen_names[name]} and {class_id}); "
                "the confusion matrix is reported by name and would be unreadable"
            )

        # A class without a written definition and annotation policy is a class
        # two annotators will label differently, and the disagreement surfaces
        # months later as an uninterpretable confusion matrix. Rejected at load
        # time, where it costs five minutes instead of a re-annotation.
        for required in ("definition", "annotation_policy"):
            text = entry.get(required)
            if not isinstance(text, str) or not text.strip():
                raise LabelMapError(
                    f"class {name!r} has no {required}. Both are required for every "
                    "class: they are what makes two annotators agree, and what makes "
                    "the confusion matrix interpretable later. See the entries in "
                    "data/label_map.yaml for the expected level of detail — "
                    "specifically, what to do with a partially occluded, merged or "
                    "ROI-clipped cluster."
                )

        seen_ids[class_id] = name
        seen_names[name] = class_id
        parsed.append(
            ClassEntry(
                class_id=class_id,
                name=name,
                definition=str(entry["definition"]).strip(),
                annotation_policy=str(entry["annotation_policy"]).strip(),
                expected_extent_m=_parse_extent(entry.get("expected_extent_m"), name),
                notes=entry.get("notes"),
            )
        )

    expected = list(range(len(parsed)))
    actual = sorted(seen_ids)
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        detail = []
        if missing:
            detail.append(f"missing {missing}")
        if extra:
            detail.append(f"out of range {extra}")
        raise LabelMapError(
            f"class_id values must be contiguous from 0 with no gaps; got {actual} "
            f"({', '.join(detail)}). A gap leaves a softmax output that can never be "
            "correct, and renumbering invalidates every manifest and checkpoint "
            "written against the old map."
        )

    parsed.sort(key=lambda c: c.class_id)
    return tuple(parsed)


def _parse_class_weights(value: Any, classes: tuple[ClassEntry, ...]) -> Optional[tuple[float, ...]]:
    """Accept ``null``, a list of K numbers, or a ``name: weight`` mapping."""
    if value is None:
        return None
    k = len(classes)
    if isinstance(value, Mapping):
        by_name = {c.name: None for c in classes}
        for name, weight in value.items():
            if name not in by_name:
                raise LabelMapError(
                    f"class_weights names class {name!r}, which is not in `classes`"
                )
            by_name[name] = float(weight)
        missing = [n for n, w in by_name.items() if w is None]
        if missing:
            raise LabelMapError(
                f"class_weights is missing a weight for {missing}; give every class a "
                "weight or set class_weights: null for unweighted cross-entropy"
            )
        return tuple(float(by_name[c.name]) for c in classes)  # type: ignore[arg-type]

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != k:
            raise LabelMapError(
                f"class_weights has {len(value)} entries but there are {k} classes"
            )
        weights = [float(w) for w in value]
        if any(w < 0 for w in weights):
            raise LabelMapError(f"class_weights must be non-negative, got {weights}")
        return tuple(weights)

    raise LabelMapError(
        f"class_weights must be null, a list of {k} numbers, or a name->weight "
        f"mapping; got {type(value).__name__}"
    )


def load_label_map(path: Union[str, Path]) -> LabelMap:
    """Load and validate a label map.

    Args:
        path: path to ``label_map.yaml``.

    Returns:
        The validated :class:`LabelMap`, carrying the file's sha256 so that a
        manifest or checkpoint can record exactly which vocabulary it used.

    Raises:
        FileNotFoundError: with a pointer to the template, because "no label map"
            is the project's known top blocker rather than a mystery.
        LabelMapError: for any map that would silently invalidate a checkpoint.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"no label map at {p}. The vocabulary of record is "
            "data/label_map.yaml (6 classes, owner-confirmed 15 Oct 2026). If it is "
            "genuinely missing, restore it from version control rather than "
            "rebuilding it from data/label_map.template.yaml — the class ids in it "
            "are referenced by every manifest and checkpoint."
        )

    raw_bytes = p.read_bytes()
    sha = hashlib.sha256(raw_bytes).hexdigest()
    try:
        doc = yaml.safe_load(raw_bytes.decode("utf-8"))
    except yaml.YAMLError as exc:
        raise LabelMapError(f"{p} is not valid YAML: {exc}") from exc

    if doc is None:
        raise LabelMapError(f"{p} is empty")
    doc = _require_mapping(doc, str(p))

    reject = doc.get("reject") or {}
    reject = _require_mapping(reject, "`reject`")
    reserved_id = reject.get("reserved_class_id", -1)
    if isinstance(reserved_id, bool) or not isinstance(reserved_id, int):
        raise LabelMapError(
            f"reject.reserved_class_id must be an integer, got {reserved_id!r}"
        )
    if reserved_id >= 0:
        raise LabelMapError(
            f"reject.reserved_class_id must be negative so it cannot collide with a "
            f"trained class id, got {reserved_id}"
        )
    reserved_name = reject.get("reserved_class_name", "unknown")
    if not isinstance(reserved_name, str) or not reserved_name.strip():
        raise LabelMapError(
            f"reject.reserved_class_name must be a non-empty string, got {reserved_name!r}"
        )

    threshold = reject.get("confidence_threshold", 0.0)
    try:
        threshold = float(threshold)
    except (TypeError, ValueError) as exc:
        raise LabelMapError(
            f"reject.confidence_threshold must be a number in [0, 1], got {threshold!r}"
        ) from exc
    if not 0.0 <= threshold <= 1.0:
        raise LabelMapError(
            f"reject.confidence_threshold must be in [0, 1], got {threshold}"
        )

    trained_as_class = bool(reject.get("trained_as_class", False))

    classes = _parse_classes(doc.get("classes"), reserved_id, reserved_name)
    weights = _parse_class_weights(doc.get("class_weights"), classes)

    version = doc.get("version")
    if version is not None and (isinstance(version, bool) or not isinstance(version, int)):
        raise LabelMapError(f"`version` must be an integer or null, got {version!r}")

    return LabelMap(
        classes=classes,
        reserved_class_id=int(reserved_id),
        reserved_class_name=reserved_name,
        confidence_threshold=threshold,
        reject_trained_as_class=trained_as_class,
        class_weights=weights,
        version=None if version is None else int(version),
        source=doc.get("source"),
        created=doc.get("created"),
        is_example=bool(doc.get("example", False)),
        path=p,
        sha256=sha,
        raw=doc,
    )
