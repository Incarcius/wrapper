"""Recordings -> labelled-cluster crops. The annotation and training unit.

The annotation unit is **one DBSCAN cluster**, produced by the same
``cluster_scene`` call with the same ``eps_m`` / ``min_samples`` that run at
inference, after ground removal (``docs/data_collection_plan.md`` §5.1). Nothing
else is labelled: hand-drawn boxes and CAD objects create a train/inference
mismatch that augmentation does not close.

## What a crop stores, and why it is the raw metric points

Each crop keeps the cluster **exactly as DBSCAN returned it** — unnormalised,
unsampled, in sensor-frame metres. Not the 1024-point normalised tensor. That
costs disk and buys three things that are impossible to recover afterwards:

* **Re-running the ablations.** ``num_points``, ``voxel_size_m`` and the
  normalisation switches are all things this project intends to vary. Storing
  the post-FPS tensor bakes one choice of each into the dataset.
* **Honest augmentation.** Jitter in metres and point dropout only mean what
  they say when applied to the metric cluster, before normalisation.
* **The aux features.** ``extent``, ``range_to_centroid`` and
  ``height_above_ground`` come from the metric cluster
  (``docs/task_definition.md`` §4), and a unit sphere has thrown all three away.

## Reproducibility

A crop set is reproducible from **(raw recording, preprocess.yaml, seed)** and
nothing else. The index therefore records the sha256 of the raw recording, the
sha256 of ``preprocess.yaml``, and the seed. Per-frame RNG is derived from
``(seed, frame_index)``, not from one stream advanced frame by frame, so frame
17's crops are identical whether or not frames 0-16 were processed — which is
what makes a partial re-extraction comparable to a full one.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence, Union

import numpy as np

from .clustering import Cluster
from .config import PreprocessConfig, load_preprocess_config
from .ground import split_ground
from .io import Recording, open_recording, sha256_of
from .regions import build_proposer

__all__ = [
    "CropError",
    "CropRecord",
    "CropIndex",
    "CROP_INDEX_VERSION",
    "extract_session_crops",
    "extract_many",
    "load_crop_index",
    "find_crop_indices",
    "crop_geometry_summary",
    "make_crop_id",
    "write_json_atomic",
]

#: Bumped whenever the on-disk layout changes incompatibly. Checked on load, so
#: an old index fails with a version message instead of a KeyError.
CROP_INDEX_VERSION = 1


class CropError(RuntimeError):
    """A crop set that cannot be written or read."""


def write_json_atomic(path: Union[str, Path], payload: Any) -> None:
    """Write JSON via a temp file and ``os.replace``.

    Used for the crop index and the label store. The labelling tool writes after
    **every** human decision, so a crash or a Ctrl-C mid-write must not be able
    to truncate the file and lose an afternoon's annotation.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, p)


# ======================================================================
@dataclass(frozen=True)
class CropRecord:
    """One cluster crop's metadata. The points live in the session's ``.npz``.

    ``crop_id`` is ``"<session>/<frame:06d>/<cluster:02d>"``: stable across
    re-extraction with the same config and seed, sortable into frame order, and
    readable in a prompt, which matters because the labelling tool prints it.
    """

    crop_id: str
    session_id: str
    frame_index: int
    stamp: float
    n_points: int
    centroid_m: tuple[float, float, float]
    extent_m: tuple[float, float, float]
    ground_z_m: Optional[float]
    offset: int  # start row in the session's packed `points` array
    length: int  # number of rows

    @property
    def range_m(self) -> float:
        """Distance from the sensor origin to the cluster centroid."""
        cx, cy, cz = self.centroid_m
        return float(np.sqrt(cx * cx + cy * cy + cz * cz))

    @property
    def largest_extent_m(self) -> float:
        return float(max(self.extent_m))

    def to_dict(self) -> dict[str, Any]:
        return {
            "crop_id": self.crop_id,
            "session_id": self.session_id,
            "frame_index": self.frame_index,
            "stamp": self.stamp,
            "n_points": self.n_points,
            "centroid_m": list(self.centroid_m),
            "extent_m": list(self.extent_m),
            "ground_z_m": self.ground_z_m,
            "offset": self.offset,
            "length": self.length,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "CropRecord":
        return CropRecord(
            crop_id=d["crop_id"],
            session_id=d["session_id"],
            frame_index=int(d["frame_index"]),
            stamp=float(d["stamp"]),
            n_points=int(d["n_points"]),
            centroid_m=tuple(float(v) for v in d["centroid_m"]),  # type: ignore[arg-type]
            extent_m=tuple(float(v) for v in d["extent_m"]),  # type: ignore[arg-type]
            ground_z_m=None if d.get("ground_z_m") is None else float(d["ground_z_m"]),
            offset=int(d["offset"]),
            length=int(d["length"]),
        )


def make_crop_id(session_id: str, frame_index: int, cluster_index: int) -> str:
    return f"{session_id}/{frame_index:06d}/{cluster_index:02d}"


# ======================================================================
@dataclass
class CropIndex:
    """One session's crops: the JSON index plus lazy access to the points.

    The split is deliberate. The index is small, human-readable, diffable and
    version-controllable; the points are a single ``.npz`` that is regenerable
    and therefore *not* version-controlled (``data/README.md``). The labelling
    tool reads only the index until it needs to draw a cluster.
    """

    session_id: str
    index_path: Path
    points_path: Path
    crops: tuple[CropRecord, ...]
    source_file: str
    raw_sha256: Optional[str]
    preprocess_sha256: Optional[str]
    seed: int
    stamp_source: str
    stats: dict[str, Any] = field(default_factory=dict)
    recording_meta: Optional[dict[str, Any]] = None
    created: Optional[str] = None
    _points_cache: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.crops)

    def __iter__(self) -> Iterator[CropRecord]:
        return iter(self.crops)

    @property
    def by_id(self) -> dict[str, CropRecord]:
        return {c.crop_id: c for c in self.crops}

    def frames(self) -> dict[int, list[CropRecord]]:
        """Crops grouped by frame index, each group in extraction order."""
        out: dict[int, list[CropRecord]] = {}
        for c in self.crops:
            out.setdefault(c.frame_index, []).append(c)
        return out

    def _points_array(self) -> np.ndarray:
        if self._points_cache is None:
            if not self.points_path.exists():
                raise CropError(
                    f"crop points file {self.points_path} is missing. It is "
                    "regenerable and not version-controlled: re-run "
                    "scripts/extract_crops (or extract_session_crops) against the "
                    f"raw recording {self.source_file!r} with the same "
                    "preprocess.yaml and seed."
                )
            with np.load(self.points_path, allow_pickle=False) as npz:
                self._points_cache = np.ascontiguousarray(npz["points"], dtype=np.float32)
        return self._points_cache

    def points_for(self, crop: Union[CropRecord, str]) -> np.ndarray:
        """The raw metric points of one crop, ``(n, 3)`` float32.

        A view-free copy is deliberately *not* made: callers that mutate (the
        augmentation path) copy first. The array is read-only to make an
        accidental in-place edit of the shared buffer fail loudly rather than
        corrupt every later item.
        """
        record = self.by_id[crop] if isinstance(crop, str) else crop
        points = self._points_array()
        end = record.offset + record.length
        if end > points.shape[0]:
            raise CropError(
                f"crop {record.crop_id} claims rows [{record.offset}, {end}) but the "
                f"points file holds {points.shape[0]}. Index and points file are out "
                "of sync; re-extract."
            )
        out = points[record.offset : end]
        out.flags.writeable = False
        return out

    def as_cluster(self, crop: Union[CropRecord, str]) -> Cluster:
        """Rebuild a :class:`~pcml.data.clustering.Cluster` from a stored crop."""
        record = self.by_id[crop] if isinstance(crop, str) else crop
        return Cluster.from_points(np.array(self.points_for(record), dtype=np.float32))

    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": CROP_INDEX_VERSION,
            "session_id": self.session_id,
            "created": self.created,
            "source_file": self.source_file,
            "raw_sha256": self.raw_sha256,
            "preprocess_sha256": self.preprocess_sha256,
            "seed": self.seed,
            "stamp_source": self.stamp_source,
            "points_file": self.points_path.name,
            "stats": self.stats,
            "recording_meta": self.recording_meta,
            "crops": [c.to_dict() for c in self.crops],
        }


def load_crop_index(path: Union[str, Path]) -> CropIndex:
    """Load a session's crop index.

    Args:
        path: the ``*_crops.json`` index, or the directory holding exactly one.
    """
    p = Path(path)
    if p.is_dir():
        candidates = sorted(p.glob("*_crops.json"))
        if len(candidates) != 1:
            raise CropError(
                f"{p} holds {len(candidates)} crop indices; name one explicitly "
                f"({[c.name for c in candidates]})"
            )
        p = candidates[0]

    doc = json.loads(p.read_text(encoding="utf-8"))
    version = int(doc.get("version", 0))
    if version != CROP_INDEX_VERSION:
        raise CropError(
            f"{p} is crop index version {version}; this code reads version "
            f"{CROP_INDEX_VERSION}. Re-extract rather than migrating by hand — the "
            "crops are regenerable from the raw recording plus preprocess.yaml."
        )

    return CropIndex(
        session_id=doc["session_id"],
        index_path=p,
        points_path=p.parent / doc["points_file"],
        crops=tuple(CropRecord.from_dict(c) for c in doc["crops"]),
        source_file=doc.get("source_file", ""),
        raw_sha256=doc.get("raw_sha256"),
        preprocess_sha256=doc.get("preprocess_sha256"),
        seed=int(doc.get("seed", 0)),
        stamp_source=doc.get("stamp_source", "unknown"),
        stats=doc.get("stats", {}),
        recording_meta=doc.get("recording_meta"),
        created=doc.get("created"),
    )


def find_crop_indices(root: Union[str, Path]) -> list[Path]:
    """Every ``*_crops.json`` under ``root``, sorted. One per session."""
    return sorted(Path(root).rglob("*_crops.json"))


# ======================================================================
def _frame_rng(seed: int, frame_index: int) -> np.random.Generator:
    """RNG for one frame, derived from ``(seed, frame_index)``.

    Deliberately *not* one generator advanced across frames: that would make
    frame 17's ground fit depend on how many frames were processed before it, so
    re-extracting a subset would silently produce different crops from the same
    inputs.
    """
    return np.random.default_rng([int(seed), int(frame_index)])


def extract_session_crops(
    recording: Union[Recording, str, Path],
    out_dir: Union[str, Path],
    config: Optional[PreprocessConfig] = None,
    config_path: Union[str, Path] = "configs/preprocess.yaml",
    seed: Optional[int] = None,
    max_frames: Optional[int] = None,
    frame_stride: int = 1,
    hash_raw: bool = True,
    topic: Optional[str] = None,
) -> CropIndex:
    """Run ground removal + clustering over a recording and persist the crops.

    One ``.npz`` (the packed points) and one ``.json`` (the index) per session,
    written to ``out_dir``.

    Args:
        recording: an open :class:`~pcml.data.io.Recording`, or a path to pass to
            :func:`~pcml.data.io.open_recording`.
        out_dir: destination directory, created if absent.
        config: parsed ``preprocess.yaml``. Loaded from ``config_path`` if None.
        config_path: where to load the config from when ``config`` is None.
        seed: overrides ``config.seed``. Recorded in the index either way,
            because the crop set is only reproducible from the pair.
        max_frames: stop after this many *read* frames. For smoke-testing a long
            bag without processing all of it.
        frame_stride: process every n-th frame. §4.4's point stands — a 60 s
            hover at 10 Hz is 600 near-identical frames of the same objects, so
            a stride is often the honest choice. ``frame_index`` stays the index
            in the **recording**, not in the kept subset, so stamps and ids
            remain traceable to the source.
        hash_raw: hash the raw recording into the index. Skip only for a
            throwaway extraction; without it the index cannot prove which bytes
            it came from.
        topic: bag topic override.

    Returns:
        The written :class:`CropIndex`.

    Raises:
        CropError: if clustering is disabled in the config (whole-scene mode is
            a debugging setting and cannot produce crops), or if ground removal
            is disabled while the proposer is instance clustering, which §3
            establishes makes the crops meaningless on a field scene.
    """
    cfg = config if config is not None else load_preprocess_config(config_path)
    if seed is None:
        seed = cfg.seed

    instance_mode = cfg.region_proposal.is_instance_clustering

    if instance_mode and not cfg.clustering.enabled:
        raise CropError(
            "clustering.enabled is false in the preprocessing config. Whole-scene "
            "mode is a debugging setting (the wrapper's C2 defect) and cannot "
            "produce cluster crops. Set clustering.enabled: true."
        )
    if instance_mode and not cfg.ground_removal.enabled:
        raise CropError(
            "ground_removal.enabled is false while region_proposal.method is "
            "dbscan_instances. On a field scene that makes the crops meaningless "
            "rather than merely worse: measured in tests/test_ground.py, a mown "
            "field either bridges every object into one field-sized cluster or "
            "shatters into object-sized false objects. See "
            "docs/data_collection_plan.md §3."
        )

    # For a surface proposer (ground_tiles) the ground is the signal, not noise:
    # the plane is still fitted, for the height reference, but the ground points
    # are kept and tiled. This is the seam regions.py exists for.
    propose = build_proposer(cfg.region_proposal.as_proposer_config())

    rec = recording if isinstance(recording, Recording) else open_recording(
        recording, **({"topic": topic} if topic else {})
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if frame_stride < 1:
        raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")

    records: list[CropRecord] = []
    chunks: list[np.ndarray] = []
    offset = 0

    n_frames_read = 0
    n_frames_kept = 0
    n_frames_no_plane = 0
    n_empty_frames = 0
    n_clusters_found = 0
    n_crops_dropped_by_cap = 0
    tilts: list[float] = []
    inlier_fractions: list[float] = []

    for frame_index, (points, stamp) in enumerate(rec.frames()):
        n_frames_read += 1
        if max_frames is not None and n_frames_read > max_frames:
            n_frames_read -= 1
            break
        if frame_index % frame_stride:
            continue
        n_frames_kept += 1

        if points.shape[0] == 0:
            n_empty_frames += 1
            continue

        rng = _frame_rng(seed, frame_index)
        gr = cfg.ground_removal
        _ground, non_ground, plane = split_ground(
            points,
            distance_threshold_m=gr.distance_threshold_m,
            num_iterations=gr.num_iterations,
            max_tilt_deg=gr.max_tilt_deg,
            min_inlier_fraction=gr.min_inlier_fraction,
            min_height_m=gr.min_height_m,
            rng=rng,
        )
        if plane is None:
            # split_ground returns the cloud untouched in this case. Proposing
            # regions from it anyway would produce the field-sized blob of §3 and
            # label it as an object, so the frame is skipped and counted instead.
            n_frames_no_plane += 1
            continue

        tilts.append(plane.tilt_deg)
        inlier_fractions.append(plane.inlier_fraction)
        ground_z = plane.ground_z_at_origin()

        # Instance clustering sees the scene with the ground gone; a surface
        # proposer sees it with the ground kept, because there the ground is the
        # thing being classified.
        scene = non_ground if instance_mode else points
        clusters, found = propose(scene)
        n_clusters_found += found
        n_crops_dropped_by_cap += max(0, found - len(clusters))

        for cluster_index, cluster in enumerate(clusters):
            pts = np.ascontiguousarray(cluster.points[:, :3], dtype=np.float32)
            records.append(
                CropRecord(
                    crop_id=make_crop_id(rec.session_id, frame_index, cluster_index),
                    session_id=rec.session_id,
                    frame_index=frame_index,
                    stamp=float(stamp),
                    n_points=int(pts.shape[0]),
                    centroid_m=cluster.centroid_m,
                    extent_m=cluster.extent_m,
                    ground_z_m=None if np.isnan(ground_z) else float(ground_z),
                    offset=offset,
                    length=int(pts.shape[0]),
                )
            )
            chunks.append(pts)
            offset += int(pts.shape[0])

    packed = (
        np.concatenate(chunks, axis=0)
        if chunks
        else np.zeros((0, 3), dtype=np.float32)
    )
    crop_ids = np.array([r.crop_id for r in records], dtype=object)

    safe_session = rec.session_id.replace("/", "_").replace(os.sep, "_")
    points_path = out / f"{safe_session}_crops.npz"
    index_path = out / f"{safe_session}_crops.json"

    np.savez_compressed(
        points_path,
        points=packed,
        offsets=np.array([r.offset for r in records], dtype=np.int64),
        lengths=np.array([r.length for r in records], dtype=np.int64),
        crop_ids=crop_ids.astype("U"),
    )

    stats = {
        "n_frames_read": n_frames_read,
        "n_frames_processed": n_frames_kept,
        "n_frames_skipped_no_ground_plane": n_frames_no_plane,
        "n_frames_empty": n_empty_frames,
        "n_clusters_found_before_filters": n_clusters_found,
        "n_crops_kept": len(records),
        "n_clusters_dropped_by_max_clusters_cap": n_crops_dropped_by_cap,
        "frame_stride": frame_stride,
        "ground_plane_tilt_deg_mean": float(np.mean(tilts)) if tilts else None,
        "ground_plane_tilt_deg_max": float(np.max(tilts)) if tilts else None,
        "ground_inlier_fraction_mean": (
            float(np.mean(inlier_fractions)) if inlier_fractions else None
        ),
        "total_points_stored": int(packed.shape[0]),
    }

    index = CropIndex(
        session_id=rec.session_id,
        index_path=index_path,
        points_path=points_path,
        crops=tuple(records),
        source_file=str(rec.path),
        raw_sha256=(
            sha256_of(rec.path) if hash_raw and rec.path.is_file() else None
        ),
        preprocess_sha256=cfg.sha256,
        seed=int(seed),
        stamp_source=rec.stamp_source,
        stats=stats,
        recording_meta=rec.meta.to_dict() if rec.meta else None,
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    write_json_atomic(index_path, index.to_dict())
    return index


def extract_many(
    recordings: Iterable[Union[Recording, str, Path]],
    out_dir: Union[str, Path],
    config: Optional[PreprocessConfig] = None,
    config_path: Union[str, Path] = "configs/preprocess.yaml",
    **kwargs: Any,
) -> list[CropIndex]:
    """Extract several sessions into one directory. One index file each."""
    cfg = config if config is not None else load_preprocess_config(config_path)
    seen: dict[str, str] = {}
    out: list[CropIndex] = []
    for rec in recordings:
        index = extract_session_crops(rec, out_dir, config=cfg, **kwargs)
        if index.session_id in seen:
            raise CropError(
                f"two recordings resolved to session_id {index.session_id!r} "
                f"({seen[index.session_id]} and {index.source_file}). Session ids are "
                "what the splits are made by, so a collision would leak near-duplicate "
                "frames across splits. Give each recording a distinct session_id, in "
                "its §4.2 metadata sidecar or via open_recording(session_id=...)."
            )
        seen[index.session_id] = index.source_file
        out.append(index)
    return out


def crop_geometry_summary(crops: Sequence[CropRecord]) -> dict[str, Any]:
    """Distribution summary over crops, for the §P6 "re-derive the caps" work.

    ``max_clusters_per_frame: 16`` and ``min_cluster_points: 50`` are both
    PROVISIONAL and are meant to be re-derived from measured distributions. This
    is the measurement, reported as percentiles rather than a mean, because the
    cap is a worst-case latency bound and a mean says nothing about the tail.
    """
    if not crops:
        return {"n_crops": 0}

    n_points = np.array([c.n_points for c in crops], dtype=np.float64)
    extents = np.array([c.largest_extent_m for c in crops], dtype=np.float64)
    ranges = np.array([c.range_m for c in crops], dtype=np.float64)
    per_frame: dict[tuple[str, int], int] = {}
    for c in crops:
        key = (c.session_id, c.frame_index)
        per_frame[key] = per_frame.get(key, 0) + 1
    counts = np.array(sorted(per_frame.values()), dtype=np.float64)

    def pct(a: np.ndarray, q: float) -> float:
        return float(np.percentile(a, q))

    return {
        "n_crops": len(crops),
        "n_frames_with_crops": int(counts.shape[0]),
        "clusters_per_frame": {
            "mean": float(counts.mean()),
            "p50": pct(counts, 50),
            "p95": pct(counts, 95),
            "max": float(counts.max()),
        },
        "points_per_cluster": {
            "min": float(n_points.min()),
            "p50": pct(n_points, 50),
            "p95": pct(n_points, 95),
            "max": float(n_points.max()),
        },
        "largest_extent_m": {
            "p50": pct(extents, 50),
            "p95": pct(extents, 95),
            "max": float(extents.max()),
        },
        "range_to_centroid_m": {
            "p50": pct(ranges, 50),
            "p95": pct(ranges, 95),
            "max": float(ranges.max()),
        },
    }
