"""
PointCloudPipeline
-------------------
Turns a raw Livox Mid-70 frame (Nx3 or Nx4 numpy array) into a clean,
normalized, fixed-size (target_points, 3) array ready for an ML model.

Designed for real-time use on a Jetson Nano:
- Every stage is independently timed and toggleable via config.
- Heavy stages (outlier removal, FPS) use Open3D's C++ implementations
  rather than pure-Python/sklearn, which are too slow for live frames.
- DBSCAN (as used in the reference paper) is supported but only recommended
  after a voxel pre-downsample, since raw DBSCAN over a full Mid-70 frame
  will not hit real-time budgets on the Nano.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import open3d as o3d
    _HAS_OPEN3D = True
except ImportError:
    _HAS_OPEN3D = False

try:
    from sklearn.cluster import DBSCAN
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

logger = logging.getLogger("pointcloud_pipeline")


@dataclass
class StageTimings:
    roi_filter: float = 0.0
    outlier_removal: float = 0.0
    normalization: float = 0.0
    downsampling: float = 0.0
    total: float = 0.0


class PointCloudPipeline:
    def __init__(self, config: dict):
        """
        config: parsed dict from pipeline_config.yaml (see config/pipeline_config.yaml)
        """
        self.cfg = config
        self._frame_count = 0

        if self.cfg["outlier_removal"]["method"].startswith("open3d") and not _HAS_OPEN3D:
            raise RuntimeError(
                "outlier_removal.method requires open3d but it is not installed. "
                "pip install open3d, or switch to method: dbscan / none."
            )
        if self.cfg["outlier_removal"]["method"] == "dbscan" and not _HAS_SKLEARN:
            raise RuntimeError("outlier_removal.method 'dbscan' requires scikit-learn.")
        if self.cfg["downsampling"]["method"] == "open3d_fps" and not _HAS_OPEN3D:
            raise RuntimeError("downsampling.method 'open3d_fps' requires open3d.")

    # ------------------------------------------------------------------
    # Stage 1: ROI filter (cheapest — run first to shrink point count ASAP)
    # ------------------------------------------------------------------
    def roi_filter(self, points: np.ndarray) -> np.ndarray:
        cfg = self.cfg["roi_filter"]
        if not cfg.get("enabled", True):
            return points

        if cfg["mode"] == "box":
            mask = (
                (points[:, 0] >= cfg["x_min"]) & (points[:, 0] <= cfg["x_max"]) &
                (points[:, 1] >= cfg["y_min"]) & (points[:, 1] <= cfg["y_max"]) &
                (points[:, 2] >= cfg["z_min"]) & (points[:, 2] <= cfg["z_max"])
            )
            return points[mask]

        if cfg["mode"] == "range":
            dist = np.linalg.norm(points[:, :3], axis=1)
            return points[dist <= cfg["max_range_m"]]

        return points  # mode == "none"

    # ------------------------------------------------------------------
    # Stage 2: Outlier / noise removal
    # ------------------------------------------------------------------
    def remove_outliers(self, points: np.ndarray) -> np.ndarray:
        cfg = self.cfg["outlier_removal"]
        method = cfg["method"]

        if method == "none" or len(points) == 0:
            return points

        if method == "open3d_statistical":
            pcd = self._to_o3d(points)
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=cfg["nb_neighbors"], std_ratio=cfg["std_ratio"]
            )
            return np.asarray(pcd.points)

        if method == "open3d_radius":
            pcd = self._to_o3d(points)
            pcd, _ = pcd.remove_radius_outlier(
                nb_points=cfg["radius_nb_points"], radius=cfg["radius"]
            )
            return np.asarray(pcd.points)

        if method == "dbscan":
            return self.remove_outliers_dbscan(
                points,
                eps=cfg["dbscan_eps"],
                min_pts=cfg["dbscan_min_points"],
                pre_voxel=cfg.get("pre_voxel_downsample_for_dbscan"),
            )

        raise ValueError(f"Unknown outlier_removal.method: {method}")

    def remove_outliers_dbscan(
        self,
        points: np.ndarray,
        eps: float = 0.5,
        min_pts: int = 10,
        pre_voxel: Optional[float] = None,
    ) -> np.ndarray:
        """DBSCAN as used in the reference paper (eps=0.5, minPts=10).
        Kept for parity/accuracy comparisons; pre_voxel lets you bound the
        point count before DBSCAN so it stays usable at frame-rate."""
        if len(points) == 0:
            return points

        work_points = points
        if pre_voxel and _HAS_OPEN3D:
            pcd = self._to_o3d(points)
            pcd = pcd.voxel_down_sample(voxel_size=pre_voxel)
            work_points = np.asarray(pcd.points)

        if not _HAS_SKLEARN:
            raise RuntimeError("scikit-learn required for DBSCAN outlier removal.")

        labels = DBSCAN(eps=eps, min_samples=min_pts).fit_predict(work_points[:, :3])
        return work_points[labels != -1]

    # ------------------------------------------------------------------
    # Stage 3: Normalization (zero-center + unit sphere scale)
    # ------------------------------------------------------------------
    def normalize(self, points: np.ndarray) -> np.ndarray:
        cfg = self.cfg["normalization"]
        if len(points) == 0:
            return points

        pts = points[:, :3].astype(np.float64, copy=True)

        if cfg.get("center", True):
            centroid = pts.mean(axis=0)
            pts -= centroid

        if cfg.get("scale_to_unit_sphere", True):
            max_dist = np.linalg.norm(pts, axis=1).max()
            if max_dist > 1e-12:
                pts /= max_dist

        return pts

    # ------------------------------------------------------------------
    # Stage 4: Downsampling to a fixed point count via FPS
    # ------------------------------------------------------------------
    def farthest_point_sampling(self, points: np.ndarray, num_points: int) -> np.ndarray:
        n = len(points)
        if n == 0:
            # degenerate frame — return zeros rather than crashing the pipeline
            logger.warning("Empty point cloud reached FPS stage; returning zeros.")
            return np.zeros((num_points, 3), dtype=np.float32)

        if n <= num_points:
            # pad by resampling with replacement to hit a fixed size
            idx = np.random.choice(n, size=num_points, replace=True)
            return points[idx]

        cfg = self.cfg["downsampling"]
        if cfg["method"] == "open3d_fps" and _HAS_OPEN3D:
            pcd = self._to_o3d(points)
            sampled = pcd.farthest_point_down_sample(num_points)
            out = np.asarray(sampled.points)
            # farthest_point_down_sample can return slightly fewer/more in edge cases; enforce exact size
            if len(out) != num_points:
                idx = np.random.choice(len(out), size=num_points, replace=len(out) < num_points)
                out = out[idx]
            return out

        # fallback: random sampling (fast, lower quality than true FPS)
        idx = np.random.choice(n, size=num_points, replace=False)
        return points[idx]

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def process(self, raw_point_cloud: np.ndarray):
        """
        raw_point_cloud: (N, 3) or (N, 4+) numpy array (xyz[, intensity, ...])
        Returns: (target_points, 3) numpy array or torch tensor per config,
                 plus a StageTimings object for latency profiling.
        """
        t0 = time.perf_counter()
        timings = StageTimings()

        pts = np.asarray(raw_point_cloud, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError(f"Expected (N,>=3) array, got shape {pts.shape}")
        pts = pts[:, :3]

        t = time.perf_counter()
        pts = self.roi_filter(pts)
        timings.roi_filter = time.perf_counter() - t

        t = time.perf_counter()
        pts = self.remove_outliers(pts)
        timings.outlier_removal = time.perf_counter() - t

        t = time.perf_counter()
        pts = self.normalize(pts)
        timings.normalization = time.perf_counter() - t

        t = time.perf_counter()
        target_points = self.cfg["downsampling"]["target_points"]
        # bound cost of FPS by voxel-downsampling first if the cloud is still large
        voxel_size = self.cfg["downsampling"].get("voxel_size_m")
        if voxel_size and _HAS_OPEN3D and len(pts) > target_points * 3:
            pcd = self._to_o3d(pts)
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
            pts = np.asarray(pcd.points)
        pts = self.farthest_point_sampling(pts, target_points)
        timings.downsampling = time.perf_counter() - t

        timings.total = time.perf_counter() - t0

        out_cfg = self.cfg["output"]
        result = pts.astype(np.float32)
        if out_cfg["format"] == "torch":
            import torch
            result = torch.from_numpy(result)

        self._frame_count += 1
        if self.cfg["logging"].get("log_stage_timings") and (
            self._frame_count % self.cfg["logging"].get("log_every_n_frames", 30) == 0
        ):
            logger.info(
                "frame=%d roi=%.1fms outlier=%.1fms norm=%.1fms down=%.1fms total=%.1fms",
                self._frame_count,
                timings.roi_filter * 1000,
                timings.outlier_removal * 1000,
                timings.normalization * 1000,
                timings.downsampling * 1000,
                timings.total * 1000,
            )

        return result, timings

    # ------------------------------------------------------------------
    @staticmethod
    def _to_o3d(points: np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3].astype(np.float64))
        return pcd
