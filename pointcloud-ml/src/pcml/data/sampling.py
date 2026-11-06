"""Point-cloud sampling primitives.

Every function here is **deterministic given an explicit ``numpy.random.Generator``**.
This is a deliberate departure from the wrapper, whose sampling calls the global
NumPy RNG (``pipeline.py:181/191/196``) and so cannot be reproduced or used for
export-parity testing — finding A1 in ``docs/wrapper_audit.md``.

Open3D is used when available because its C++ FPS is far faster than the NumPy
fallback, but the NumPy path is kept as the reference implementation: it defines
the semantics, and it is what runs in environments without Open3D wheels.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:  # pragma: no cover - import guard
    import open3d as o3d

    _HAS_OPEN3D = True
except ImportError:  # pragma: no cover - import guard
    _HAS_OPEN3D = False


__all__ = [
    "farthest_point_sample",
    "voxel_downsample",
    "resample_to_fixed_size",
    "SamplingStats",
]


class SamplingStats:
    """How a cloud reached its fixed size. Carried into diagnostics rather than discarded.

    ``n_unique`` vs ``n_out`` is the signal that catches the wrapper's C1 defect:
    1024 points of which 121 are unique is a silent catastrophe otherwise.
    """

    __slots__ = ("n_in", "n_unique", "n_out", "mode")

    def __init__(self, n_in: int, n_unique: int, n_out: int, mode: str) -> None:
        self.n_in = n_in
        self.n_unique = n_unique
        self.n_out = n_out
        self.mode = mode

    @property
    def duplicate_fraction(self) -> float:
        """Fraction of output points that are duplicates of another output point."""
        if self.n_out == 0:
            return 0.0
        return 1.0 - (self.n_unique / self.n_out)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"SamplingStats(n_in={self.n_in}, n_unique={self.n_unique}, "
            f"n_out={self.n_out}, mode={self.mode!r}, "
            f"duplicate_fraction={self.duplicate_fraction:.3f})"
        )


def _fps_numpy(points: np.ndarray, k: int, start_index: int) -> np.ndarray:
    """Reference farthest-point sampling. Returns the selected indices.

    O(k * n) distance evaluations, vectorised over n. Deterministic given
    ``start_index``.
    """
    n = points.shape[0]
    selected = np.empty(k, dtype=np.int64)
    # Squared distance from each point to the nearest already-selected point.
    min_sq_dist = np.full(n, np.inf, dtype=np.float64)
    current = int(start_index)

    for i in range(k):
        selected[i] = current
        diff = points - points[current]
        sq_dist = np.einsum("ij,ij->i", diff, diff)
        np.minimum(min_sq_dist, sq_dist, out=min_sq_dist)
        # Next point is the one farthest from the selected set.
        current = int(np.argmax(min_sq_dist))

    return selected


def farthest_point_sample(
    points: np.ndarray,
    k: int,
    rng: np.random.Generator,
    use_open3d: bool = True,
) -> np.ndarray:
    """Select ``k`` well-spread points. Returns an ``(k, 3)`` array.

    Args:
        points: ``(n, 3)`` array. Must satisfy ``n >= k``.
        k: number of points to select.
        rng: generator used only to pick the deterministic FPS start index.
        use_open3d: use Open3D's C++ implementation when available. Open3D picks
            its own start index, so results differ from the NumPy path; both are
            valid FPS, but only the NumPy path is reproducible across Open3D
            versions. Export-parity tests must use ``use_open3d=False``.

    Raises:
        ValueError: if ``n < k``. Padding is the caller's decision, made
            explicitly in :func:`resample_to_fixed_size`, never silently here.
    """
    n = points.shape[0]
    if n < k:
        raise ValueError(f"farthest_point_sample needs n >= k, got n={n}, k={k}")
    if n == k:
        return points.copy()

    if use_open3d and _HAS_OPEN3D:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3].astype(np.float64))
        sampled = np.asarray(pcd.farthest_point_down_sample(k).points)
        if sampled.shape[0] == k:
            return sampled.astype(points.dtype, copy=False)
        # Open3D can return an off-by-small count on degenerate input; fall
        # through to the reference path rather than papering over it with a
        # random resample (which is what the wrapper does at pipeline.py:191).

    start = int(rng.integers(0, n))
    return points[_fps_numpy(points[:, :3].astype(np.float64), k, start)].astype(
        points.dtype, copy=False
    )


def voxel_downsample(points: np.ndarray, voxel_size_m: float) -> np.ndarray:
    """Keep one point per occupied voxel (the centroid of the points in it).

    Pure NumPy so that the result is identical with and without Open3D, which
    matters because this runs in both the training and the inference path.

    Args:
        points: ``(n, 3)`` array, in **metres**. The unit matters: running this
            after unit-sphere normalisation is the wrapper's C1 defect.
        voxel_size_m: edge length of a voxel, in metres. Must be > 0.
    """
    if voxel_size_m <= 0:
        raise ValueError(f"voxel_size_m must be > 0, got {voxel_size_m}")
    if points.shape[0] == 0:
        return points

    xyz = points[:, :3]
    keys = np.floor(xyz / voxel_size_m).astype(np.int64)
    # Unique voxel per row, with an inverse map so we can average within voxels.
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    n_voxels = counts.shape[0]

    sums = np.zeros((n_voxels, 3), dtype=np.float64)
    np.add.at(sums, inverse, xyz.astype(np.float64))
    centroids = sums / counts[:, None]
    return centroids.astype(points.dtype, copy=False)


def resample_to_fixed_size(
    points: np.ndarray,
    target: int,
    rng: np.random.Generator,
    voxel_size_m: Optional[float] = None,
    use_open3d: bool = True,
) -> tuple[np.ndarray, SamplingStats]:
    """Bring a cloud to exactly ``target`` points, reporting how it got there.

    Order of operations, in **metres** (pre-normalisation) — this is the C1 fix:
      1. optional voxel pre-pass, to bound FPS cost on dense clouds;
      2. FPS down to ``target`` when enough points remain;
      3. otherwise pad by sampling with replacement, and say so in the stats.

    Returns:
        ``(points_out, stats)`` where ``points_out`` is ``(target, 3)``.
        An empty input yields zeros and ``mode="empty"`` — the caller is expected
        to check ``stats.mode`` and skip inference rather than classify zeros
        (the wrapper's finding I4).
    """
    n_in = points.shape[0]
    if n_in == 0:
        zeros = np.zeros((target, 3), dtype=np.float32)
        return zeros, SamplingStats(0, 0, target, "empty")

    work = points[:, :3]

    if voxel_size_m is not None and work.shape[0] > target:
        voxelled = voxel_downsample(work, voxel_size_m)
        # Only accept the voxel pre-pass if it leaves us enough points to do a
        # real FPS. Otherwise it would turn a satisfiable request into a padded
        # one, which is strictly worse.
        if voxelled.shape[0] >= target:
            work = voxelled

    if work.shape[0] >= target:
        out = farthest_point_sample(work, target, rng, use_open3d=use_open3d)
        mode = "fps"
    else:
        idx = rng.choice(work.shape[0], size=target, replace=True)
        out = work[idx]
        mode = "pad"

    n_unique = np.unique(out, axis=0).shape[0]
    stats = SamplingStats(n_in, n_unique, target, mode)
    return np.ascontiguousarray(out, dtype=np.float32), stats
