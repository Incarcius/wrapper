import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pipeline import PointCloudPipeline  # noqa: E402


BASE_CONFIG = {
    "roi_filter": {"enabled": False, "mode": "none"},
    "outlier_removal": {"method": "none"},
    "normalization": {"center": True, "scale_to_unit_sphere": True},
    "downsampling": {"voxel_size_m": None, "target_points": 1024, "method": "random"},
    "output": {"format": "numpy", "dtype": "float32"},
    "logging": {"log_stage_timings": False, "log_every_n_frames": 30},
}


def make_pipeline(overrides=None):
    cfg = {k: dict(v) for k, v in BASE_CONFIG.items()}
    if overrides:
        for section, vals in overrides.items():
            cfg[section].update(vals)
    return PointCloudPipeline(cfg)


class TestNormalize:
    def test_zero_centered(self):
        pipeline = make_pipeline()
        pts = np.random.uniform(-10, 10, size=(500, 3)).astype(np.float32) + np.array([5, -3, 2])
        out = pipeline.normalize(pts)
        centroid = out.mean(axis=0)
        assert np.allclose(centroid, 0, atol=1e-6)

    def test_unit_sphere_bound(self):
        pipeline = make_pipeline()
        pts = np.random.uniform(-50, 50, size=(500, 3)).astype(np.float32)
        out = pipeline.normalize(pts)
        max_dist = np.linalg.norm(out, axis=1).max()
        assert max_dist <= 1.0 + 1e-6
        assert max_dist > 1.0 - 1e-3  # farthest point should sit ~on the unit sphere

    def test_empty_input(self):
        pipeline = make_pipeline()
        out = pipeline.normalize(np.zeros((0, 3), dtype=np.float32))
        assert out.shape == (0, 3)


class TestFPS:
    def test_correct_output_count(self):
        pipeline = make_pipeline()
        pts = np.random.uniform(-1, 1, size=(5000, 3)).astype(np.float32)
        out = pipeline.farthest_point_sampling(pts, 1024)
        assert out.shape == (1024, 3)

    def test_upsamples_when_input_smaller_than_target(self):
        pipeline = make_pipeline()
        pts = np.random.uniform(-1, 1, size=(100, 3)).astype(np.float32)
        out = pipeline.farthest_point_sampling(pts, 1024)
        assert out.shape == (1024, 3)

    def test_empty_input_returns_zeros(self):
        pipeline = make_pipeline()
        out = pipeline.farthest_point_sampling(np.zeros((0, 3), dtype=np.float32), 1024)
        assert out.shape == (1024, 3)
        assert np.allclose(out, 0)


class TestOutlierRemovalDBSCAN:
    def test_removes_synthetic_noise(self):
        pipeline = make_pipeline()
        rng = np.random.default_rng(0)
        cluster = rng.normal(loc=[0, 0, 0], scale=0.05, size=(300, 3))
        noise = rng.uniform(-10, 10, size=(30, 3))
        pts = np.vstack([cluster, noise]).astype(np.float32)

        cleaned = pipeline.remove_outliers_dbscan(pts, eps=0.5, min_pts=10)
        # cluster points should mostly survive, isolated noise points should mostly be dropped
        assert len(cleaned) < len(pts)
        assert len(cleaned) >= len(cluster) * 0.9


class TestROIFilter:
    def test_box_filter_bounds(self):
        pipeline = make_pipeline({
            "roi_filter": {
                "enabled": True, "mode": "box",
                "x_min": -1, "x_max": 1, "y_min": -1, "y_max": 1, "z_min": -1, "z_max": 1,
            }
        })
        pts = np.array([[0, 0, 0], [5, 5, 5], [-0.5, 0.5, -0.5]], dtype=np.float32)
        out = pipeline.roi_filter(pts)
        assert len(out) == 2

    def test_range_filter(self):
        pipeline = make_pipeline({
            "roi_filter": {"enabled": True, "mode": "range", "max_range_m": 5.0}
        })
        pts = np.array([[0, 0, 0], [10, 0, 0], [3, 0, 0]], dtype=np.float32)
        out = pipeline.roi_filter(pts)
        assert len(out) == 2


class TestFullProcess:
    def test_output_shape_and_type(self):
        pipeline = make_pipeline()
        pts = np.random.uniform(-5, 5, size=(3000, 3)).astype(np.float32)
        out, timings = pipeline.process(pts)
        assert out.shape == (1024, 3)
        assert out.dtype == np.float32
        assert timings.total >= 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
