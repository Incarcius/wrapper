"""Scene -> object instances, and instances -> model-ready batches.

This module is the fix for the wrapper's finding **C2**: the wrapper normalises a
whole ±10 m scene into one unit sphere and hands it to a classifier trained on
single objects. Here the scene is split into instances first, and normalisation
and resampling are re-scoped to each instance.

The clustering itself is classical (DBSCAN), not learned, per the task definition:
instance separation comes from geometry, the class label comes from the network.

Order of operations is load-bearing. Clustering, voxel pre-pass and FPS all run in
**metres**; normalisation is the *last* step before the network. The wrapper does
it in the opposite order, which is finding C1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .sampling import SamplingStats, resample_to_fixed_size

try:  # pragma: no cover - import guard
    import open3d as o3d

    _HAS_OPEN3D = True
except ImportError:  # pragma: no cover - import guard
    _HAS_OPEN3D = False

try:  # pragma: no cover - import guard
    from sklearn.cluster import DBSCAN

    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover - import guard
    _HAS_SKLEARN = False


__all__ = [
    "Cluster",
    "AUX_FEATURE_NAMES",
    "cluster_scene",
    "prepare_cluster_batch",
    "finite_points_only",
]

#: Order of the auxiliary feature vector. Fixed, because a checkpoint trained
#: with these features is only valid with this exact ordering. Changing the order
#: requires a retrain, so the list is treated as part of the model contract.
AUX_FEATURE_NAMES: tuple[str, ...] = (
    "extent_x_m",
    "extent_y_m",
    "extent_z_m",
    "log_n_points",
    "range_to_centroid_m",
    "height_above_ground_m",
)


@dataclass(frozen=True)
class Cluster:
    """One candidate object, in sensor-frame metres.

    ``points`` is the raw cluster: unnormalised, unsampled, exactly what DBSCAN
    returned. Keeping the raw cluster (rather than only its resampled form) is
    what allows the geometry in ``centroid_m`` / ``extent_m`` to be reported in
    real units after the network has seen a unit-sphere version.
    """

    points: np.ndarray  # (n, 3) float32, metres, sensor frame
    centroid_m: tuple[float, float, float]
    extent_m: tuple[float, float, float]
    n_points_raw: int

    @staticmethod
    def from_points(points: np.ndarray) -> "Cluster":
        xyz = np.ascontiguousarray(points[:, :3], dtype=np.float32)
        centroid = xyz.mean(axis=0)
        extent = xyz.max(axis=0) - xyz.min(axis=0)
        return Cluster(
            points=xyz,
            centroid_m=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
            extent_m=(float(extent[0]), float(extent[1]), float(extent[2])),
            n_points_raw=int(xyz.shape[0]),
        )


def finite_points_only(points: np.ndarray) -> np.ndarray:
    """Drop rows containing NaN or Inf.

    Needed because the wrapper only guards NaNs at the ROS 2 ingestion callback
    (``skip_nans=True``), so any cloud loaded from a file or a recording can carry
    them — finding A6. A single NaN poisons the centroid, hence the whole cloud,
    silently.
    """
    if points.shape[0] == 0:
        return points
    return points[np.isfinite(points[:, :3]).all(axis=1)]


def cluster_scene(
    points: np.ndarray,
    eps_m: float = 0.5,
    min_samples: int = 10,
    min_cluster_points: int = 50,
    max_clusters: Optional[int] = 16,
    use_open3d: bool = True,
) -> tuple[list[Cluster], int]:
    """Split a de-noised scene into object instances.

    Args:
        points: ``(n, >=3)`` array in **metres**, sensor frame, after the
            wrapper's ROI crop and outlier removal.
        eps_m: DBSCAN neighbourhood radius, metres. Units are real here, unlike
            the wrapper's post-normalisation voxel size.
        min_samples: DBSCAN core-point threshold.
        min_cluster_points: clusters smaller than this are discarded. Resampling
            a 20-point cluster to 1024 yields ~98% duplicates and almost no shape
            information, so classifying it is worse than declining to.
        max_clusters: keep at most this many clusters, largest first, to bound
            worst-case latency. ``None`` means no cap.
        use_open3d: prefer Open3D's C++ ``cluster_dbscan``, which is markedly
            faster on-device than scikit-learn's.

    Returns:
        ``(clusters, n_found)`` — ``clusters`` is sorted by descending point count
        and capped; ``n_found`` is the number of clusters found *before* the cap
        and size filter. Reporting both is deliberate: dropping objects must be
        visible, not inferred.

    Raises:
        RuntimeError: if neither Open3D nor scikit-learn is importable.
    """
    pts = finite_points_only(np.asarray(points, dtype=np.float32))
    if pts.shape[0] == 0:
        return [], 0

    xyz = pts[:, :3]

    if use_open3d and _HAS_OPEN3D:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
        labels = np.asarray(
            pcd.cluster_dbscan(eps=eps_m, min_points=min_samples, print_progress=False)
        )
    elif _HAS_SKLEARN:
        labels = DBSCAN(eps=eps_m, min_samples=min_samples).fit_predict(xyz)
    else:  # pragma: no cover - environment guard
        raise RuntimeError(
            "cluster_scene needs Open3D or scikit-learn. Install one, or run with "
            "clustering disabled (whole-scene mode), which is only valid for "
            "debugging — see docs/task_definition.md."
        )

    # label -1 is DBSCAN noise; it is dropped, not classified.
    unique = [c for c in np.unique(labels) if c != -1]
    n_found = len(unique)

    kept: list[Cluster] = []
    for c in unique:
        member = xyz[labels == c]
        if member.shape[0] >= min_cluster_points:
            kept.append(Cluster.from_points(member))

    kept.sort(key=lambda cl: cl.n_points_raw, reverse=True)
    if max_clusters is not None:
        kept = kept[:max_clusters]
    return kept, n_found


def _aux_features(cluster: Cluster, ground_z_m: float) -> np.ndarray:
    """Scale-carrying features, computed from the raw metric cluster.

    Unit-sphere normalisation throws absolute size away, which for CAD benchmarks
    is correct and for LiDAR objects discards one of the strongest available cues
    (see ``docs/task_definition.md`` §4). These six numbers put that information
    back at the classifier head, where it costs almost nothing. Whether they
    actually help is an ablation, not an assumption.
    """
    ex, ey, ez = cluster.extent_m
    cx, cy, cz = cluster.centroid_m
    return np.array(
        [
            ex,
            ey,
            ez,
            np.log1p(float(cluster.n_points_raw)),
            float(np.sqrt(cx * cx + cy * cy + cz * cz)),
            float(cz - ground_z_m),
        ],
        dtype=np.float32,
    )


def prepare_cluster_batch(
    clusters: list[Cluster],
    num_points: int,
    rng: np.random.Generator,
    voxel_size_m: Optional[float] = None,
    center: bool = True,
    scale_to_unit_sphere: bool = True,
    ground_z_m: Optional[float] = None,
    use_open3d: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[SamplingStats]]:
    """Turn clusters into one batched model input.

    All clusters go into a **single** tensor so the network runs once per frame
    rather than once per object. With M separate calls, kernel-launch and
    dispatch overhead is paid M times, which on an Orin Nano dominates the
    arithmetic — see ``docs/task_definition.md`` §9.2.

    Args:
        clusters: output of :func:`cluster_scene`.
        num_points: points per cluster after resampling (1024 by convention).
        rng: seeded generator; makes the whole transform reproducible.
        voxel_size_m: optional voxel pre-pass, in metres, applied before FPS.
        center: subtract each cluster's own centroid.
        scale_to_unit_sphere: divide by each cluster's own max radius.
        ground_z_m: ground height used for the ``height_above_ground_m`` feature.
            Defaults to the minimum z across all clusters, which is a crude proxy
            for the ground plane and is only valid when the ground is in view;
            a RANSAC plane estimate should be passed in once that stage exists.
        use_open3d: forwarded to the sampler.

    Returns:
        ``(batch, aux, stats)``:
          - ``batch``: ``(M, num_points, 3)`` float32, normalised. Layout is
            ``(B, N, C)``; conversion to ``(B, C, N)`` is the adapter's job, so
            that layout stays a config choice (finding I5).
          - ``aux``: ``(M, 6)`` float32, ordered by :data:`AUX_FEATURE_NAMES`,
            computed from the **raw metric** clusters.
          - ``stats``: per-cluster sampling diagnostics.
    """
    if num_points <= 0:
        raise ValueError(f"num_points must be > 0, got {num_points}")

    m = len(clusters)
    batch = np.zeros((m, num_points, 3), dtype=np.float32)
    aux = np.zeros((m, len(AUX_FEATURE_NAMES)), dtype=np.float32)
    stats: list[SamplingStats] = []

    if m == 0:
        return batch, aux, stats

    if ground_z_m is None:
        ground_z_m = float(min(c.points[:, 2].min() for c in clusters))

    for i, cluster in enumerate(clusters):
        # 1. Resample in metres (voxel + FPS), so voxel_size_m means metres.
        sampled, st = resample_to_fixed_size(
            cluster.points,
            num_points,
            rng,
            voxel_size_m=voxel_size_m,
            use_open3d=use_open3d,
        )
        stats.append(st)

        # 2. Normalise last, on exactly the points the network will see — which
        #    is what ModelNet-style training does, and what makes PointNet++'s
        #    absolute ball-query radii meaningful.
        pts = sampled.astype(np.float32, copy=True)
        if center:
            pts -= pts.mean(axis=0)
        if scale_to_unit_sphere:
            max_dist = float(np.linalg.norm(pts, axis=1).max())
            if max_dist > 1e-9:
                pts /= max_dist

        batch[i] = pts
        aux[i] = _aux_features(cluster, ground_z_m)

    return batch, aux, stats
