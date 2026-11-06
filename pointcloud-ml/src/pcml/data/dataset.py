"""A torch ``Dataset`` over labelled cluster crops.

Imported explicitly (``from pcml.data.dataset import ClusterCropDataset``) and
**not** re-exported from ``pcml.data``, because this is the only module in the
data package that needs torch. ``pcml.deploy`` runs on the drone with numpy as
its sole dependency besides the inference runtime, and importing ``pcml.data``
there must not drag torch in.

## The transform order, which is the whole design

```
raw metric cluster (as DBSCAN returned it, from the crop store)
  -> augmentation          IN METRES, about the cluster's own centroid
  -> voxel pre-pass + FPS  IN METRES            } prepare_cluster_batch,
  -> per-cluster normalise LAST                 } the same call inference uses
  -> (3, N) + aux (6,)
```

Augmentation goes **first**, on the metric cluster, because that is where the
parameters mean what they say: ``jitter_sigma_m: 0.01`` is a centimetre of Mid-70
ranging noise, and ``point_dropout_max: 0.25`` is a quarter of the returns the
sensor did not get. Applied after unit-sphere normalisation, the same numbers
would mean "1% of this object's own size", which is a different augmentation that
scales with the object and is wrong for every class.

Resampling and normalisation then go through ``prepare_cluster_batch`` — the
*same function the inference path calls* — rather than a reimplementation. A
training-time copy of that transform is how train/inference skew gets introduced.

## Reproducibility

Each view's RNG is derived from ``(seed, epoch, index, view)``, so:

* item 42 is identical on every run with the same seed, whatever the batch size,
  the shuffle order or the number of dataloader workers;
* each epoch sees different augmentation (that is what augmentation is for);
* two views of one crop in one epoch are **independently** augmented, which is
  what makes a consistency or mean-teacher objective expressible (``n_views``);
* with ``augment=False`` the epoch is pinned to 0, so a validation set does not
  quietly change between epochs via the FPS start index.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from .clustering import AUX_FEATURE_NAMES, Cluster, prepare_cluster_batch
from .config import AugmentationConfig, PreprocessConfig, load_preprocess_config
from .crops import CropIndex, CropRecord, load_crop_index
from .label_mapping import LabelMap, load_label_map
from .labels import PROVENANCE_ANNOTATED

__all__ = [
    "DatasetError",
    "DatasetItem",
    "ClusterCropDataset",
    "augment_cluster_points",
    "rotation_matrix",
]


class DatasetError(RuntimeError):
    """A dataset that cannot be assembled from its manifest and crop stores."""


@dataclass(frozen=True)
class DatasetItem:
    """One training example: which crop, which label, and how the label was made."""

    crop_id: str
    session_id: str
    label_id: int
    provenance: str = PROVENANCE_ANNOTATED


# ======================================================================
# Augmentation. Pure functions on metric points, so they are testable alone.
# ======================================================================
def rotation_matrix(yaw_rad: float, pitch_rad: float = 0.0, roll_rad: float = 0.0) -> np.ndarray:
    """Z-Y-X intrinsic rotation, as a ``(3, 3)`` float64 matrix.

    Yaw about z is the one that matters: clusters arrive at arbitrary heading
    relative to the drone, and none of the candidate architectures is
    rotation-invariant (``docs/task_definition.md`` §9.5). Pitch and roll are
    small by design — the drone is roughly level, and a cluster rolled 90° is a
    scene the sensor will never produce.
    """
    cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    cr, sr = math.cos(roll_rad), math.sin(roll_rad)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


def augment_cluster_points(
    points: np.ndarray,
    aug: AugmentationConfig,
    rng: np.random.Generator,
    min_points: int = 16,
) -> np.ndarray:
    """Perturb one cluster, in **metres**, before resampling and normalisation.

    Order: dropout, then rotation, then scale, then translation, then jitter.
    Dropout first because it models a sparser return pattern — it should decide
    *which* points exist before the surviving ones are moved, not thin out a
    cloud that jitter has already smeared.

    Rotation and scaling are applied **about the cluster's own centroid**. Not
    about the sensor origin: that would move the cluster to a different place in
    the scene and so change ``range_to_centroid_m``, fabricating a measurement
    the sensor never made. Rotating in place is the honest version of "the same
    object seen from another bearing".

    Args:
        points: ``(n, 3)`` metric points. Not modified.
        aug: the ``augmentation`` block of ``preprocess.yaml``.
        rng: seeded generator — all randomness comes from here.
        min_points: dropout never takes a cluster below this. A cluster thinned
            to 3 points carries no shape at all, and ``min_cluster_points: 50``
            means the inference path would never have been handed one.

    Returns:
        A new ``(m, 3)`` float32 array, ``m <= n``.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise DatasetError(f"expected (n, >=3) points, got shape {pts.shape}")
    pts = pts[:, :3].copy()
    n = pts.shape[0]
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32)

    # 1. Point dropout: sparser returns, as at longer range.
    if aug.point_dropout_max > 0.0 and n > min_points:
        drop_fraction = float(rng.uniform(0.0, aug.point_dropout_max))
        keep = max(min_points, int(round(n * (1.0 - drop_fraction))))
        if keep < n:
            idx = rng.choice(n, size=keep, replace=False)
            idx.sort()  # keep the original order; cheap, and aids debugging
            pts = pts[idx]

    centroid = pts.mean(axis=0)

    # 2. Rotation about the cluster centroid.
    if aug.yaw_degrees > 0.0 or aug.pitch_roll_degrees > 0.0:
        yaw = math.radians(float(rng.uniform(-aug.yaw_degrees, aug.yaw_degrees)))
        pitch = math.radians(
            float(rng.uniform(-aug.pitch_roll_degrees, aug.pitch_roll_degrees))
        )
        roll = math.radians(
            float(rng.uniform(-aug.pitch_roll_degrees, aug.pitch_roll_degrees))
        )
        rot = rotation_matrix(yaw, pitch, roll)
        pts = (pts - centroid) @ rot.T + centroid

    # 3. Scale about the centroid. Limited: object size is semantically
    #    meaningful here, unlike on a CAD benchmark.
    lo, hi = aug.scale_range
    if not (lo == hi == 1.0):
        s = float(rng.uniform(lo, hi))
        pts = (pts - centroid) * s + centroid

    # 4. Translation. 0.0 by default — pointless after per-cluster centring,
    #    except for its effect on range_to_centroid_m, which is why it exists
    #    as a knob at all.
    if aug.translation_m > 0.0:
        pts = pts + rng.uniform(-aug.translation_m, aug.translation_m, size=3)

    # 5. Ranging noise, in metres, clipped so a tail sample cannot invent a
    #    point outside the sensor's error envelope.
    if aug.jitter_sigma_m > 0.0:
        noise = rng.normal(0.0, aug.jitter_sigma_m, size=pts.shape)
        if aug.jitter_clip_m > 0.0:
            np.clip(noise, -aug.jitter_clip_m, aug.jitter_clip_m, out=noise)
        pts = pts + noise

    return np.ascontiguousarray(pts, dtype=np.float32)


# ======================================================================
class ClusterCropDataset(Dataset):
    """Labelled cluster crops as ``(points (3, N), aux (6,), label)``.

    Args:
        items: the examples, usually from a manifest.
        indices: ``session_id -> CropIndex``, holding the points.
        label_map: the vocabulary. Used to validate every label id, so a
            manifest written against an older map fails at construction rather
            than training a head against shifted ids.
        config: parsed ``preprocess.yaml``.
        augment: apply the config's augmentation block. True for train, False
            for val/test.
        seed: base seed. Defaults to ``config.seed``.
        epoch: changes the augmentation draw. Ignored when ``augment`` is False.
        n_views: how many independently augmented views of each crop to return.
            1 (the default) returns one, which is ordinary supervised training.
            ``>= 2`` returns that many, for a consistency or mean-teacher
            objective — see :meth:`get_view`.

    Returns per item, with ``n_views == 1``:
        ``points``: ``(3, num_points)`` float32 — the layout the model's
        ``forward`` takes directly.
        ``aux``: ``(6,)`` float32, ordered by
        :data:`~pcml.data.clustering.AUX_FEATURE_NAMES`. Always produced, even
        when ``features.use_aux`` is false, so the aux-feature ablation is a
        model-construction choice and not a dataset regeneration.
        ``label``: ``int`` class id.

    With ``n_views >= 2`` the first two gain a leading view axis —
    ``(n_views, 3, num_points)`` and ``(n_views, 6)`` — and the label is shared,
    which is the point: the views are the same object, so a consistency loss can
    require the predictions to agree. The default collate then yields
    ``(B, V, 3, N)``.
    """

    def __init__(
        self,
        items: Sequence[DatasetItem],
        indices: Mapping[str, CropIndex],
        label_map: LabelMap,
        config: PreprocessConfig,
        augment: bool = False,
        seed: Optional[int] = None,
        epoch: int = 0,
        n_views: int = 1,
    ) -> None:
        self.items = list(items)
        self.indices = dict(indices)
        self.label_map = label_map
        self.config = config
        self.augment = bool(augment)
        self.seed = int(config.seed if seed is None else seed)
        self.epoch = int(epoch)

        if n_views < 1:
            raise DatasetError(f"n_views must be >= 1, got {n_views}")
        if n_views > 1 and not self.augment:
            raise DatasetError(
                f"n_views={n_views} with augment=False would return views that "
                "differ only by the FPS draw — near-duplicates, not a consistency "
                "signal, and a consistency loss over them is satisfied trivially. "
                "Pass augment=True, or n_views=1."
            )
        if n_views > 1 and config.augmentation.is_identity:
            raise DatasetError(
                f"n_views={n_views} but every augmentation in {config.path} is "
                "disabled, so the views would be identical. Enable augmentation "
                "or use n_views=1."
            )
        self.n_views = int(n_views)

        missing_sessions = sorted(
            {i.session_id for i in self.items} - set(self.indices)
        )
        if missing_sessions:
            raise DatasetError(
                f"no crop index loaded for session(s) {missing_sessions}; the manifest "
                "references crops whose points are not available"
            )

        for item in self.items:
            if not label_map.is_trained_class(item.label_id):
                raise DatasetError(
                    f"crop {item.crop_id} has label_id {item.label_id}, which is not a "
                    f"trained class in {label_map.path} (valid 0..{label_map.num_classes - 1}). "
                    f"The reject id {label_map.reserved_class_id} is an output, not a "
                    "training target: filter those crops out of the manifest, or add an "
                    "explicit background class to the label map."
                )
            index = self.indices[item.session_id]
            if item.crop_id not in index.by_id:
                raise DatasetError(
                    f"crop {item.crop_id} is not in {index.index_path}. The manifest and "
                    "the crop store disagree — most likely the crops were re-extracted "
                    "with a different preprocess.yaml or seed after the manifest was "
                    "written. Re-generate the manifest."
                )

    # ------------------------------------------------------------------
    @staticmethod
    def from_manifest(
        manifest_path: Union[str, Path],
        label_map: Optional[LabelMap] = None,
        config: Optional[PreprocessConfig] = None,
        augment: Optional[bool] = None,
        crops_root: Optional[Union[str, Path]] = None,
        label_map_path: Union[str, Path] = "data/label_map.yaml",
        config_path: Union[str, Path] = "configs/preprocess.yaml",
        seed: Optional[int] = None,
        n_views: int = 1,
    ) -> "ClusterCropDataset":
        """Build a dataset from a manifest written by :mod:`pcml.data.splits`.

        ``augment`` defaults to ``True`` for the ``train`` split and ``False``
        for everything else — getting that backwards silently is a worse failure
        than a slightly magic default.

        ``n_views >= 2`` therefore only works on a ``train`` manifest unless
        ``augment=True`` is passed explicitly, which is the intended constraint:
        a consistency objective over an eval split would be measuring nothing.
        """
        p = Path(manifest_path)
        doc = json.loads(p.read_text(encoding="utf-8"))
        split = doc.get("split", "unknown")

        lm = label_map if label_map is not None else load_label_map(label_map_path)
        cfg = config if config is not None else load_preprocess_config(config_path)

        if doc.get("label_map_sha256") and lm.sha256 != doc["label_map_sha256"]:
            raise DatasetError(
                f"{p} was written against label map sha256 "
                f"{doc['label_map_sha256'][:12]}… but {lm.path} is {str(lm.sha256)[:12]}…. "
                "The class ids may mean different things. Re-generate the manifests."
            )
        if doc.get("preprocess_sha256") and cfg.sha256 != doc["preprocess_sha256"]:
            raise DatasetError(
                f"{p} was written against preprocess.yaml sha256 "
                f"{doc['preprocess_sha256'][:12]}… but {cfg.path} is "
                f"{str(cfg.sha256)[:12]}…. The crops it names were produced with "
                "different clustering parameters. Re-extract and re-generate."
            )

        indices: dict[str, CropIndex] = {}
        items: list[DatasetItem] = []
        root = Path(crops_root) if crops_root is not None else p.parent

        for entry in doc["crops"]:
            session_id = entry["session_id"]
            if session_id not in indices:
                raw_path = Path(entry["index_file"])
                candidates = [raw_path, root / raw_path, p.parent / raw_path]
                for candidate in candidates:
                    if candidate.is_file():
                        indices[session_id] = load_crop_index(candidate)
                        break
                else:
                    raise DatasetError(
                        f"crop index {entry['index_file']!r} for session "
                        f"{session_id!r} not found (tried {[str(c) for c in candidates]}). "
                        "Pass crops_root=... if the crop store moved."
                    )
            items.append(
                DatasetItem(
                    crop_id=entry["crop_id"],
                    session_id=session_id,
                    label_id=int(entry["label_id"]),
                    provenance=entry.get("provenance", PROVENANCE_ANNOTATED),
                )
            )

        return ClusterCropDataset(
            items=items,
            indices=indices,
            label_map=lm,
            config=cfg,
            augment=(split == "train") if augment is None else bool(augment),
            seed=seed,
            n_views=n_views,
        )

    # ------------------------------------------------------------------
    def set_epoch(self, epoch: int) -> None:
        """Change the augmentation draw. Call once per training epoch.

        Without this, every epoch would re-apply the identical augmentation to
        each item, which is the same as having no augmentation at all plus a
        fixed distortion.
        """
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.items)

    def _rng(self, index: int, view: int = 0) -> np.random.Generator:
        """Generator for one (item, epoch, view). All randomness comes from here.

        Keyed on ``view`` as well as ``index`` and ``epoch`` so that two views of
        the same crop in the same epoch are **independently** augmented. Without
        the view term, asking twice for item 7 returns the identical cloud and a
        two-view consistency objective is inexpressible — the loss would be
        trivially zero.

        Epoch is pinned to 0 without augmentation, so an eval set is identical
        across epochs (the FPS start index also comes from this generator).
        """
        epoch = self.epoch if self.augment else 0
        return np.random.default_rng([self.seed, epoch, int(index), int(view)])

    def get_view(self, index: int, view: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """One augmented view of one item, as ``(points (3, N), aux (6,), label)``.

        ``view`` selects an independent augmentation draw. Reproducible from
        ``(seed, epoch, index, view)``, so view 1 of item 7 in epoch 3 is the same
        cloud on every run and in every process.

        This has to live here rather than in the trainer. The dataset returns
        unit-sphere points, and while yaw commutes with centring and scaling, the
        jitter and point dropout are specified in **metres** and applied to the
        metric cluster before normalisation — a second view made by re-rotating
        the dataset's output would be a yaw-only view with the same dropout
        pattern and the same jitter, which is a weaker perturbation than it looks.
        """
        item = self.items[index]
        crop_index = self.indices[item.session_id]
        record: CropRecord = crop_index.by_id[item.crop_id]
        rng = self._rng(index, view)

        points = np.array(crop_index.points_for(record), dtype=np.float32)
        if self.augment and not self.config.augmentation.is_identity:
            points = augment_cluster_points(points, self.config.augmentation, rng)

        cluster = Cluster.from_points(points)
        batch, aux, _stats = prepare_cluster_batch(
            [cluster],
            num_points=self.config.sampling.num_points,
            rng=rng,
            voxel_size_m=self.config.sampling.voxel_size_m,
            center=self.config.normalization.center,
            scale_to_unit_sphere=self.config.normalization.scale_to_unit_sphere,
            ground_z_m=record.ground_z_m,
            use_open3d=self.config.sampling.use_open3d,
        )

        # (N, 3) -> (3, N): the model's forward() signature. The adapter owns
        # layout at inference time (finding I5); here the contract is fixed.
        points_t = torch.from_numpy(np.ascontiguousarray(batch[0].T, dtype=np.float32))
        aux_t = torch.from_numpy(np.ascontiguousarray(aux[0], dtype=np.float32))
        return points_t, aux_t, int(item.label_id)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        if self.n_views == 1:
            return self.get_view(index, 0)

        views = [self.get_view(index, v) for v in range(self.n_views)]
        points_t = torch.stack([v[0] for v in views], dim=0)
        aux_t = torch.stack([v[1] for v in views], dim=0)
        # One label for all views: they are the same object, which is exactly
        # what a consistency objective asserts.
        return points_t, aux_t, views[0][2]

    # ------------------------------------------------------------------
    @property
    def aux_feature_names(self) -> tuple[str, ...]:
        return AUX_FEATURE_NAMES

    def class_counts(self) -> dict[int, int]:
        """Examples per class id. Input to the class-weighted loss."""
        counts: dict[int, int] = {i: 0 for i in range(self.label_map.num_classes)}
        for item in self.items:
            counts[item.label_id] += 1
        return counts

    def provenance_counts(self) -> dict[str, int]:
        """How many examples are human-annotated vs propagated (§5.3)."""
        counts: dict[str, int] = {}
        for item in self.items:
            counts[item.provenance] = counts.get(item.provenance, 0) + 1
        return counts

    def session_ids(self) -> tuple[str, ...]:
        return tuple(sorted({i.session_id for i in self.items}))

    def describe(self) -> dict[str, Any]:
        return {
            "n_items": len(self.items),
            "n_sessions": len(self.session_ids()),
            "augment": self.augment,
            "n_views": self.n_views,
            "seed": self.seed,
            "num_points": self.config.sampling.num_points,
            "class_counts": {
                self.label_map.names[i]: n for i, n in self.class_counts().items()
            },
            "provenance_counts": self.provenance_counts(),
        }
