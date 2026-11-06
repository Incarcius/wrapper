"""The label store, and the propagation that makes labelling tractable.

``docs/data_collection_plan.md`` §5.2 is the whole reason this module exists.
Most classes on a field are **static** — trees, poles, walls, nets, parked
vehicles — and during a hover the sensor frame is effectively fixed. So an object
is labelled **once per recording, by position**, and that label is propagated to
every frame by matching cluster centroids. A 60-second hover is 600 frames;
labelling 8 objects once yields ~4,800 labelled crops from 8 human decisions.

§5.3 is the other half, and it is not optional: a propagated label is recorded
as ``propagated``, never silently as ground truth. The held-out test set is built
from ``annotated`` labels only, so the final number is not partly a measure of
this heuristic. :class:`LabelStore` therefore has no way to write a label without
a provenance.

Two honest limits of centroid matching, both surfaced rather than hidden:

* **It assumes the object is static and the sensor is not moving.** A walking
  person violates both. The tool annotates such a crop per frame instead; the
  store records nothing automatically for an unmatched cluster.
* **It is ambiguous when two objects sit inside the match radius.** The nearest
  object wins, and the assignment is flagged ``ambiguous`` with the runner-up
  distance, so the ambiguous crops can be counted — or excluded — later.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

import numpy as np

from .crops import CropRecord, write_json_atomic
from .label_mapping import LabelMap

__all__ = [
    "LabelStoreError",
    "PROVENANCE_ANNOTATED",
    "PROVENANCE_PROPAGATED",
    "DEFAULT_MATCH_RADIUS_M",
    "LABEL_STORE_VERSION",
    "KNOWN_ATTRIBUTES",
    "LabelledObject",
    "CropLabel",
    "LabelStore",
    "ascii_projection",
    "describe_crop",
    "validate_attributes",
]

PROVENANCE_ANNOTATED = "annotated"
PROVENANCE_PROPAGATED = "propagated"
_PROVENANCES = (PROVENANCE_ANNOTATED, PROVENANCE_PROPAGATED)

#: Per-crop annotation attributes the confirmed ``label_map.yaml`` policies
#: require, with the class whose policy asks for each. These are **not** extra
#: classes and not model inputs: they are what makes the written annotation
#: policies enforceable and the evaluation able to separate cases it would
#: otherwise hide.
#:
#: The schema is open — :attr:`CropLabel.attributes` accepts any key — because a
#: policy added later must not need a code change. This mapping exists so the
#: labelling tool knows what to *prompt* for, and so a typo in a known key can be
#: pointed out. Unknown keys are kept, with a warning.
KNOWN_ATTRIBUTES: dict[str, tuple[str, type, str]] = {
    # key: (class it belongs to, type, why it is recorded)
    "merged_instances": (
        "person",
        int,
        "number of people in a cluster DBSCAN merged; lets evaluation report "
        "merged and single cases separately instead of hiding the failure",
    ),
    "extended": (
        "vegetation",
        bool,
        "true for a hedge or treeline over ~6 m horizontally, so these can be "
        "analysed apart from individual trees",
    ),
    "subtype": (
        "pole",
        str,
        "e.g. 'goal' for a complete football goal, so goals can be evaluated "
        "separately from masts and fence posts",
    ),
    "edge_on": (
        "net_fence",
        bool,
        "true for an edge-on mesh view — the hard case, easily mistaken for a "
        "pole, and worth measuring on its own",
    ),
}

#: Attribute values are written to JSON and read back by the evaluation, so they
#: are restricted to JSON scalars. A tuple or a numpy scalar would round-trip
#: into something subtly different.
_ATTRIBUTE_VALUE_TYPES = (bool, int, float, str)


def validate_attributes(
    attributes: Optional[Mapping[str, Any]], label_name: Optional[str] = None
) -> tuple[dict[str, Any], list[str]]:
    """Normalise annotation attributes, returning ``(attributes, warnings)``.

    Rejects values that will not survive a JSON round trip, and warns — without
    rejecting — about a key that belongs to a different class or is not in
    :data:`KNOWN_ATTRIBUTES`. Warning rather than rejecting is deliberate: the
    schema has to stay open for policies written after this code, and losing an
    annotator's note because a key was unrecognised would be worse than carrying
    an unused one.
    """
    if not attributes:
        return {}, []

    out: dict[str, Any] = {}
    warnings: list[str] = []
    for key, value in attributes.items():
        if not isinstance(key, str) or not key.strip():
            raise LabelStoreError(f"attribute keys must be non-empty strings, got {key!r}")
        if value is not None and not isinstance(value, _ATTRIBUTE_VALUE_TYPES):
            raise LabelStoreError(
                f"attribute {key!r} has value {value!r} of type "
                f"{type(value).__name__}; attributes are stored as JSON, so values "
                "must be bool, int, float, str or null"
            )
        known = KNOWN_ATTRIBUTES.get(key)
        if known is None:
            warnings.append(
                f"attribute {key!r} is not one of the keys the label-map policies "
                f"define ({sorted(KNOWN_ATTRIBUTES)}); it will be stored anyway"
            )
        else:
            owner_class, expected_type, _why = known
            if label_name is not None and label_name != owner_class:
                warnings.append(
                    f"attribute {key!r} is defined by the {owner_class!r} annotation "
                    f"policy but this crop is labelled {label_name!r}"
                )
            if value is not None and expected_type is int and isinstance(value, bool):
                raise LabelStoreError(
                    f"attribute {key!r} expects an integer, got the boolean {value!r}"
                )
            if value is not None and not isinstance(value, expected_type):
                # float where int is expected is accepted when it is integral,
                # because JSON and YAML both happily produce 2.0 for 2.
                if expected_type is int and isinstance(value, float) and value.is_integer():
                    value = int(value)
                else:
                    raise LabelStoreError(
                        f"attribute {key!r} expects {expected_type.__name__}, got "
                        f"{value!r} ({type(value).__name__})"
                    )
        out[key] = value
    return out, warnings

#: Centroid match radius, metres. 0.5 m is the DBSCAN ``eps_m``, which is the
#: right order of magnitude: a static object's centroid wanders by less than one
#: neighbourhood radius between frames as the rosette fills in different points,
#: and two distinct objects closer than that would have merged into one cluster
#: anyway. Configurable because a windy day moves a canopy centroid further.
DEFAULT_MATCH_RADIUS_M = 0.5

LABEL_STORE_VERSION = 1


class LabelStoreError(RuntimeError):
    """A label store that cannot be read, or an operation that would lose work."""


# ======================================================================
@dataclass(frozen=True)
class LabelledObject:
    """One physical object, labelled once and propagated by position.

    ``position_m`` is the centroid of the cluster the annotator actually looked
    at. ``n_assigned`` is maintained by the store so that §4.4's warning —
    frames are not samples, and 600 frames of one tree is one tree — is visible
    in the report rather than something a reader has to work out.
    """

    object_id: int
    label_id: int
    label_name: str
    position_m: tuple[float, float, float]
    radius_m: float = DEFAULT_MATCH_RADIUS_M
    annotated_crop_id: Optional[str] = None
    frame_index: Optional[int] = None
    notes: Optional[str] = None
    #: Annotation attributes that travel with every propagated crop, because they
    #: are properties of the *object* rather than of the frame: a goal is a goal
    #: in every frame, and a merged pair of people stays merged while neither
    #: moves. See :data:`KNOWN_ATTRIBUTES`.
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def distance_to(self, centroid: Sequence[float]) -> float:
        px, py, pz = self.position_m
        dx = float(centroid[0]) - px
        dy = float(centroid[1]) - py
        dz = float(centroid[2]) - pz
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "label_id": self.label_id,
            "label_name": self.label_name,
            "position_m": list(self.position_m),
            "radius_m": self.radius_m,
            "annotated_crop_id": self.annotated_crop_id,
            "frame_index": self.frame_index,
            "notes": self.notes,
            "attributes": dict(self.attributes),
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "LabelledObject":
        return LabelledObject(
            object_id=int(d["object_id"]),
            label_id=int(d["label_id"]),
            label_name=str(d["label_name"]),
            position_m=tuple(float(v) for v in d["position_m"]),  # type: ignore[arg-type]
            radius_m=float(d.get("radius_m", DEFAULT_MATCH_RADIUS_M)),
            annotated_crop_id=d.get("annotated_crop_id"),
            frame_index=None if d.get("frame_index") is None else int(d["frame_index"]),
            notes=d.get("notes"),
            attributes=dict(d.get("attributes") or {}),
        )


@dataclass(frozen=True)
class CropLabel:
    """One crop's label, with its provenance (§5.3). Provenance is mandatory.

    Not just a class id. ``label_map.yaml``'s annotation policies ask the
    annotator for more than a class in four places — the number of people in a
    merged ``person`` cluster, whether a ``vegetation`` cluster is an extended
    hedge, a ``pole``'s subtype, whether a ``net_fence`` view is edge-on — and a
    schema that held only a class id would make those policies unenforceable and
    the corresponding evaluation splits impossible. :attr:`attributes` carries
    them, with an open key set so a policy written later needs no code change.
    """

    crop_id: str
    label_id: int
    label_name: str
    provenance: str
    object_id: Optional[int] = None
    match_distance_m: Optional[float] = None
    ambiguous: bool = False
    runner_up_distance_m: Optional[float] = None
    #: Per-crop annotation attributes. See :data:`KNOWN_ATTRIBUTES`.
    attributes: Mapping[str, Any] = field(default_factory=dict)
    notes: Optional[str] = None

    def __post_init__(self) -> None:
        if self.provenance not in _PROVENANCES:
            raise LabelStoreError(
                f"provenance must be one of {_PROVENANCES}, got {self.provenance!r}. "
                "A label without a provenance cannot be excluded from the test split, "
                "which docs/data_collection_plan.md §5.3 requires."
            )

    @property
    def is_annotated(self) -> bool:
        return self.provenance == PROVENANCE_ANNOTATED

    def to_dict(self) -> dict[str, Any]:
        return {
            "crop_id": self.crop_id,
            "label_id": self.label_id,
            "label_name": self.label_name,
            "provenance": self.provenance,
            "object_id": self.object_id,
            "match_distance_m": self.match_distance_m,
            "ambiguous": self.ambiguous,
            "runner_up_distance_m": self.runner_up_distance_m,
            "attributes": dict(self.attributes),
            "notes": self.notes,
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "CropLabel":
        return CropLabel(
            crop_id=str(d["crop_id"]),
            label_id=int(d["label_id"]),
            label_name=str(d["label_name"]),
            provenance=str(d["provenance"]),
            object_id=None if d.get("object_id") is None else int(d["object_id"]),
            match_distance_m=(
                None if d.get("match_distance_m") is None else float(d["match_distance_m"])
            ),
            ambiguous=bool(d.get("ambiguous", False)),
            runner_up_distance_m=(
                None
                if d.get("runner_up_distance_m") is None
                else float(d["runner_up_distance_m"])
            ),
            attributes=dict(d.get("attributes") or {}),
            notes=d.get("notes"),
        )


# ======================================================================
@dataclass
class LabelStore:
    """One session's labels. Saved after every decision, so nothing is ever lost.

    On disk: ``<labels_dir>/<session_id>_labels.json``, written through
    :func:`~pcml.data.crops.write_json_atomic`. The labelling tool calls
    :meth:`save` after each keypress; a Ctrl-C or a dead battery costs at most
    the cluster currently on screen.

    The store is deliberately **separate from the crop index**. Crops are
    regenerable from (raw recording + preprocess.yaml + seed); labels are human
    work and are not regenerable at all. Keeping them apart means re-extracting
    crops with a different ``eps_m`` cannot destroy annotation — the labels are
    re-matched by position instead, which is exactly what propagation already
    does.
    """

    session_id: str
    path: Path
    objects: dict[int, LabelledObject] = field(default_factory=dict)
    labels: dict[str, CropLabel] = field(default_factory=dict)
    skipped: set[str] = field(default_factory=set)
    label_map_sha256: Optional[str] = None
    match_radius_m: float = DEFAULT_MATCH_RADIUS_M
    created: Optional[str] = None
    notes: Optional[str] = None

    # ------------------------------------------------------------------
    @staticmethod
    def path_for(labels_dir: Union[str, Path], session_id: str) -> Path:
        safe = session_id.replace("/", "_").replace("\\", "_")
        return Path(labels_dir) / f"{safe}_labels.json"

    @staticmethod
    def load_or_create(
        labels_dir: Union[str, Path],
        session_id: str,
        match_radius_m: float = DEFAULT_MATCH_RADIUS_M,
        label_map_sha256: Optional[str] = None,
    ) -> "LabelStore":
        """Resume an existing store, or start a new one.

        Resumability is the point: the labelling tool is restarted constantly
        (wrong class picked, laptop closed, another look at a cluster wanted),
        and re-annotating from scratch each time is how an afternoon of work gets
        thrown away.
        """
        p = LabelStore.path_for(labels_dir, session_id)
        if p.exists():
            store = LabelStore.load(p)
            if label_map_sha256 and store.label_map_sha256 and (
                store.label_map_sha256 != label_map_sha256
            ):
                raise LabelStoreError(
                    f"{p} was written against label map sha256 "
                    f"{store.label_map_sha256[:12]}… but the current label map is "
                    f"{label_map_sha256[:12]}…. The class ids may mean different "
                    "things now. Resolve it deliberately: either restore the old "
                    "label map, or re-map the ids in this file by hand. Refusing to "
                    "silently reinterpret existing annotation."
                )
            store.match_radius_m = match_radius_m
            if label_map_sha256:
                store.label_map_sha256 = label_map_sha256
            return store
        return LabelStore(
            session_id=session_id,
            path=p,
            match_radius_m=match_radius_m,
            label_map_sha256=label_map_sha256,
            created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    @staticmethod
    def load(path: Union[str, Path]) -> "LabelStore":
        p = Path(path)
        doc = json.loads(p.read_text(encoding="utf-8"))
        version = int(doc.get("version", 0))
        if version != LABEL_STORE_VERSION:
            raise LabelStoreError(
                f"{p} is label store version {version}; this code reads "
                f"{LABEL_STORE_VERSION}. Do not delete it — it holds human work. "
                "Migrate it deliberately."
            )
        return LabelStore(
            session_id=doc["session_id"],
            path=p,
            objects={
                int(o["object_id"]): LabelledObject.from_dict(o)
                for o in doc.get("objects", [])
            },
            labels={
                str(l["crop_id"]): CropLabel.from_dict(l) for l in doc.get("labels", [])
            },
            skipped=set(doc.get("skipped", [])),
            label_map_sha256=doc.get("label_map_sha256"),
            match_radius_m=float(doc.get("match_radius_m", DEFAULT_MATCH_RADIUS_M)),
            created=doc.get("created"),
            notes=doc.get("notes"),
        )

    def save(self) -> None:
        write_json_atomic(
            self.path,
            {
                "version": LABEL_STORE_VERSION,
                "session_id": self.session_id,
                "created": self.created,
                "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "label_map_sha256": self.label_map_sha256,
                "match_radius_m": self.match_radius_m,
                "notes": self.notes,
                "summary": self.summary(),
                "objects": [o.to_dict() for o in self.objects.values()],
                "labels": [l.to_dict() for l in self.labels.values()],
                "skipped": sorted(self.skipped),
            },
        )

    # ------------------------------------------------------------------
    def next_object_id(self) -> int:
        return (max(self.objects) + 1) if self.objects else 0

    def annotate(
        self,
        crop: CropRecord,
        label_id: int,
        label_map: LabelMap,
        radius_m: Optional[float] = None,
        notes: Optional[str] = None,
        new_object: bool = True,
        attributes: Optional[Mapping[str, Any]] = None,
    ) -> tuple[CropLabel, Optional[LabelledObject]]:
        """Record a human decision about one crop.

        This is the only path that produces an ``annotated`` label. By default it
        also registers a :class:`LabelledObject` at the crop's centroid, which is
        what later frames propagate from.

        Args:
            crop: the crop the annotator looked at.
            label_id: a trained class id, or the label map's reserved reject id
                for a cluster that is not any class (§4.3's negatives). The
                reject id is accepted here even though it is not a training
                target: ``reject.trained_as_class`` is false, so those crops are
                kept to calibrate the threshold and excluded from the manifests
                later rather than thrown away now.
            label_map: used to resolve and validate the id. A name is never
                stored from a literal.
            radius_m: match radius for this object. Defaults to the store's.
            notes: free text, e.g. "half behind the roller".
            new_object: ``False`` for a moving object (a walking person), where a
                per-frame decision must **not** become a position to propagate
                from — §5.2 only holds for static objects.
            attributes: the annotation attributes the class's policy asks for
                (``merged_instances``, ``extended``, ``subtype``, ``edge_on``).
                Stored on the object too, so they propagate with the label.

        Returns:
            ``(crop_label, labelled_object_or_None)``.

        Raises:
            LabelStoreError: if ``label_id`` is not storable, or an attribute
                value would not survive a JSON round trip.
        """
        if not label_map.is_storable_label(int(label_id)):
            raise LabelStoreError(
                f"label_id {label_id} is neither a trained class "
                f"(0..{label_map.num_classes - 1}) nor the reject id "
                f"{label_map.reserved_class_id} in {label_map.path}"
            )
        name = label_map.name_for(label_id)
        attrs, _warnings = validate_attributes(attributes, label_name=name)

        obj: Optional[LabelledObject] = None
        if new_object:
            obj = LabelledObject(
                object_id=self.next_object_id(),
                label_id=int(label_id),
                label_name=name,
                position_m=crop.centroid_m,
                radius_m=self.match_radius_m if radius_m is None else float(radius_m),
                annotated_crop_id=crop.crop_id,
                frame_index=crop.frame_index,
                notes=notes,
                attributes=attrs,
            )
            self.objects[obj.object_id] = obj

        label = CropLabel(
            crop_id=crop.crop_id,
            label_id=int(label_id),
            label_name=name,
            provenance=PROVENANCE_ANNOTATED,
            object_id=obj.object_id if obj else None,
            match_distance_m=0.0 if obj else None,
            attributes=attrs,
            notes=notes,
        )
        self.labels[crop.crop_id] = label
        self.skipped.discard(crop.crop_id)
        return label, obj

    def skip(self, crop: CropRecord) -> None:
        """Mark a crop as deliberately not labelled.

        Distinct from "not yet seen": a skipped crop is not re-presented on the
        next run, which is what makes a long session resumable without
        re-deciding the clusters already judged unlabellable.
        """
        self.skipped.add(crop.crop_id)
        self.labels.pop(crop.crop_id, None)

    def match(
        self, centroid: Sequence[float], radius_m: Optional[float] = None
    ) -> tuple[Optional[LabelledObject], Optional[float], Optional[float]]:
        """Nearest labelled object within its radius.

        Returns ``(object, distance, runner_up_distance)``. ``runner_up_distance``
        is the distance to the second-closest object that was *also* inside the
        radius, or ``None``; a non-None value means the match was ambiguous and
        is recorded as such.
        """
        inside: list[tuple[float, LabelledObject]] = []
        for obj in self.objects.values():
            limit = obj.radius_m if radius_m is None else float(radius_m)
            d = obj.distance_to(centroid)
            if d <= limit:
                inside.append((d, obj))
        if not inside:
            return None, None, None
        inside.sort(key=lambda t: t[0])
        best_d, best = inside[0]
        runner_up = inside[1][0] if len(inside) > 1 else None
        return best, best_d, runner_up

    def propagate(
        self,
        crops: Iterable[CropRecord],
        radius_m: Optional[float] = None,
        overwrite_propagated: bool = False,
    ) -> dict[str, int]:
        """Propagate every labelled object's label to matching crops (§5.2).

        Annotated labels are never overwritten — a human decision outranks a
        heuristic, always. Existing *propagated* labels are left alone too unless
        ``overwrite_propagated`` is set, which is what to use after moving an
        object's position or changing the radius.

        Returns a count breakdown: ``propagated``, ``ambiguous``, ``unmatched``,
        ``already_labelled``.
        """
        counts = {"propagated": 0, "ambiguous": 0, "unmatched": 0, "already_labelled": 0}
        for crop in crops:
            existing = self.labels.get(crop.crop_id)
            if existing is not None:
                if existing.is_annotated or not overwrite_propagated:
                    counts["already_labelled"] += 1
                    continue

            obj, distance, runner_up = self.match(crop.centroid_m, radius_m=radius_m)
            if obj is None:
                counts["unmatched"] += 1
                continue

            ambiguous = runner_up is not None
            if ambiguous:
                counts["ambiguous"] += 1
            self.labels[crop.crop_id] = CropLabel(
                crop_id=crop.crop_id,
                label_id=obj.label_id,
                label_name=obj.label_name,
                provenance=PROVENANCE_PROPAGATED,
                object_id=obj.object_id,
                match_distance_m=distance,
                ambiguous=ambiguous,
                runner_up_distance_m=runner_up,
                # Attributes are properties of the object, not of the frame: a
                # goal is a goal in every frame, and a merged pair of people
                # stays merged for as long as neither of them moves (and once one
                # does, the cluster stops matching this object anyway).
                attributes=dict(obj.attributes),
            )
            counts["propagated"] += 1
        return counts

    def move_object(self, object_id: int, position_m: Sequence[float]) -> LabelledObject:
        """Reposition an object. Call :meth:`propagate` with
        ``overwrite_propagated=True`` afterwards to re-match."""
        if object_id not in self.objects:
            raise LabelStoreError(f"no object {object_id} in session {self.session_id!r}")
        obj = replace(
            self.objects[object_id],
            position_m=tuple(float(v) for v in position_m),  # type: ignore[arg-type]
        )
        self.objects[object_id] = obj
        return obj

    def unlabelled(self, crops: Iterable[CropRecord]) -> list[CropRecord]:
        """Crops with neither a label nor a skip. What the tool presents next."""
        return [
            c
            for c in crops
            if c.crop_id not in self.labels and c.crop_id not in self.skipped
        ]

    # ------------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        """Counts for the report, split by provenance.

        ``distinct_objects`` is reported next to the crop count on purpose: §4.4
        warns that frames are not samples, and 4,800 propagated crops from 8
        objects is 8 objects.
        """
        by_class: dict[str, dict[str, int]] = {}
        for label in self.labels.values():
            entry = by_class.setdefault(
                label.label_name, {"annotated": 0, "propagated": 0, "ambiguous": 0}
            )
            entry[label.provenance] += 1
            if label.ambiguous:
                entry["ambiguous"] += 1
        attribute_counts: dict[str, int] = {}
        for label in self.labels.values():
            for key in label.attributes:
                attribute_counts[key] = attribute_counts.get(key, 0) + 1

        return {
            "n_labels": len(self.labels),
            "n_annotated": sum(1 for l in self.labels.values() if l.is_annotated),
            "n_propagated": sum(1 for l in self.labels.values() if not l.is_annotated),
            "n_ambiguous": sum(1 for l in self.labels.values() if l.ambiguous),
            "n_skipped": len(self.skipped),
            "attribute_counts": attribute_counts,
            "distinct_objects": len(self.objects),
            "distinct_objects_per_class": {
                name: sum(1 for o in self.objects.values() if o.label_name == name)
                for name in sorted({o.label_name for o in self.objects.values()})
            },
            "by_class": by_class,
        }


# ======================================================================
# Text rendering: the labelling tool has to work without a GUI.
# ======================================================================
#: Density ramp, sparse to dense. ASCII only, because this gets read over ssh
#: from a field laptop as often as in a desktop terminal.
_RAMP = " .:-=+*#%@"


def ascii_projection(
    points: np.ndarray,
    width: int = 34,
    height: int = 14,
    axes: tuple[int, int] = (0, 1),
) -> list[str]:
    """A coarse density projection of a cluster onto two axes.

    Deliberately crude. The question an annotator answers is "thin-vertical,
    large-planar or volumetric-irregular" (``docs/data_collection_plan.md`` §2),
    and a 34x14 occupancy grid answers it. Anything prettier needs a GUI, which
    is exactly what this avoids depending on.

    The grid preserves **aspect ratio**: the two axes share one metres-per-cell
    scale, so a 6 m pole looks like a pole rather than being stretched to fill
    the box. The caller prints the extents in metres alongside, because the
    projection alone cannot show absolute size.

    Args:
        points: ``(n, >=3)`` metric points.
        width: grid columns (the first axis).
        height: grid rows (the second axis, drawn increasing upward).
        axes: which two coordinate axes to project onto. ``(0, 1)`` is the
            top-down x-y view, ``(0, 2)`` the x-z elevation.

    Returns:
        ``height`` strings of length ``width``.
    """
    if width < 2 or height < 2:
        raise ValueError(f"grid must be at least 2x2, got {width}x{height}")

    blank = [" " * width for _ in range(height)]
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return blank
    pts = pts[np.isfinite(pts[:, :3]).all(axis=1)]
    if pts.shape[0] == 0:
        return blank

    a, b = axes
    u = pts[:, a]
    v = pts[:, b]
    u_span = float(u.max() - u.min())
    v_span = float(v.max() - v.min())

    # One scale for both axes so the shape is not distorted.
    scale = max(u_span / (width - 1), v_span / (height - 1), 1e-6)
    u_centre = 0.5 * (float(u.max()) + float(u.min()))
    v_centre = 0.5 * (float(v.max()) + float(v.min()))

    cols = np.round((u - u_centre) / scale + (width - 1) / 2.0).astype(int)
    rows = np.round((v - v_centre) / scale + (height - 1) / 2.0).astype(int)
    np.clip(cols, 0, width - 1, out=cols)
    np.clip(rows, 0, height - 1, out=rows)

    grid = np.zeros((height, width), dtype=np.int64)
    np.add.at(grid, (rows, cols), 1)

    occupied = grid[grid > 0]
    peak = float(occupied.max()) if occupied.size else 1.0
    # Square-root ramp: linear density mapping makes everything but the densest
    # cell look empty on a cluster with one concentrated surface.
    levels = np.zeros_like(grid)
    nonzero = grid > 0
    levels[nonzero] = 1 + np.floor(
        np.sqrt(grid[nonzero] / peak) * (len(_RAMP) - 2)
    ).astype(np.int64)
    np.clip(levels, 0, len(_RAMP) - 1, out=levels)

    # Row 0 of the grid is the low end of axis `b`; print it last so the view is
    # the right way up.
    return ["".join(_RAMP[int(c)] for c in levels[r]) for r in range(height - 1, -1, -1)]


def describe_crop(
    crop: CropRecord,
    points: Optional[np.ndarray] = None,
    width: int = 34,
    height: int = 14,
) -> list[str]:
    """Everything an annotator needs about one crop, as printable lines.

    Geometry first, in metres, because the class distinctions in §2 are
    shape-and-size distinctions: a 0.3 x 0.3 x 1.8 m cluster is a person and a
    0.2 x 0.2 x 6 m one is a pole, and that is visible in the numbers before any
    picture is drawn. The two projections are there for the cases the numbers
    cannot settle — mesh vs plane, canopy vs wall.
    """
    ex, ey, ez = crop.extent_m
    cx, cy, cz = crop.centroid_m
    aspect = ez / max(ex, ey) if max(ex, ey) > 1e-6 else float("inf")
    height_above = None if crop.ground_z_m is None else cz - crop.ground_z_m

    lines = [
        f"crop      {crop.crop_id}",
        f"frame     {crop.frame_index}   stamp {crop.stamp:.3f} s",
        f"points    {crop.n_points}",
        f"centroid  x={cx:+.2f}  y={cy:+.2f}  z={cz:+.2f} m   range {crop.range_m:.2f} m",
        f"extent    x={ex:.2f}  y={ey:.2f}  z={ez:.2f} m   "
        f"vertical/horizontal aspect {aspect:.2f}",
    ]
    if height_above is not None:
        lines.append(f"centroid height above fitted ground  {height_above:+.2f} m")

    if points is not None:
        top = ascii_projection(points, width=width, height=height // 2 + 1, axes=(0, 1))
        side = ascii_projection(points, width=width, height=height, axes=(0, 2))
        lines.append("")
        lines.append(f"top view  x-y   (x span {ex:.2f} m, y span {ey:.2f} m)")
        lines += [f"  |{row}|" for row in top]
        lines.append(f"elevation x-z   (x span {ex:.2f} m, z span {ez:.2f} m)")
        lines += [f"  |{row}|" for row in side]
    return lines
