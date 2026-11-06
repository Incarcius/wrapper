"""Sampling tests. Includes the regression test for the wrapper's C1 defect.

Every test uses an explicit seeded generator: if any of these becomes flaky, the
determinism guarantee in ``pcml.data.sampling`` has been broken.
"""

from __future__ import annotations

import numpy as np
import pytest

from pcml.data.sampling import (
    farthest_point_sample,
    resample_to_fixed_size,
    voxel_downsample,
)


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def classroom_scene(n: int = 10_000, seed: int = 7) -> np.ndarray:
    """A plausible ~10k-point Mid-70 frame of a room-sized scene, in metres.

    Roughly the shape of the wrapper's own C1 reproduction: a few metres across,
    which is what makes a 0.15 *unit-sphere* voxel catastrophic and a 0.15 *metre*
    voxel harmless.
    """
    g = np.random.default_rng(seed)
    floor = np.column_stack(
        [
            g.uniform(-5, 5, n // 2),
            g.uniform(-5, 5, n // 2),
            g.normal(0, 0.01, n // 2),
        ]
    )
    objects = g.uniform(-4, 4, (n - n // 2, 3)) * np.array([1.0, 1.0, 0.3])
    return np.vstack([floor, objects]).astype(np.float32)


class TestVoxelDownsample:
    def test_units_are_metres(self):
        """A 0.5 m voxel over a 10 m span must leave ~20 cells per axis, not 2."""
        pts = classroom_scene()
        out = voxel_downsample(pts, 0.5)
        # Strictly fewer points, but still a dense representation of the scene.
        assert out.shape[0] < pts.shape[0]
        assert out.shape[0] > 200

    def test_one_point_per_voxel(self):
        # Two tight clusters 10 m apart collapse to exactly two points.
        pts = np.vstack(
            [
                np.zeros((50, 3), dtype=np.float32),
                np.full((50, 3), 10.0, dtype=np.float32),
            ]
        )
        out = voxel_downsample(pts, 1.0)
        assert out.shape[0] == 2

    def test_centroid_semantics(self):
        pts = np.array([[0.1, 0.0, 0.0], [0.3, 0.0, 0.0]], dtype=np.float32)
        out = voxel_downsample(pts, 1.0)
        assert out.shape == (1, 3)
        assert out[0, 0] == pytest.approx(0.2, abs=1e-6)

    def test_rejects_nonpositive_voxel(self):
        with pytest.raises(ValueError, match="must be > 0"):
            voxel_downsample(classroom_scene(100), 0.0)

    def test_empty_passthrough(self):
        out = voxel_downsample(np.zeros((0, 3), dtype=np.float32), 0.1)
        assert out.shape == (0, 3)


class TestFPS:
    def test_exact_count(self):
        pts = classroom_scene(5000)
        out = farthest_point_sample(pts, 1024, rng(), use_open3d=False)
        assert out.shape == (1024, 3)

    def test_deterministic_for_equal_seeds(self):
        pts = classroom_scene(2000)
        a = farthest_point_sample(pts, 256, rng(42), use_open3d=False)
        b = farthest_point_sample(pts, 256, rng(42), use_open3d=False)
        assert np.array_equal(a, b)

    def test_spreads_wider_than_random(self):
        """FPS should cover the scene better than uniform random sampling.

        Measured as the mean nearest-neighbour distance within the sample: a
        well-spread set has a larger one.
        """
        pts = classroom_scene(4000)
        fps = farthest_point_sample(pts, 200, rng(1), use_open3d=False)
        rnd = pts[rng(1).choice(pts.shape[0], 200, replace=False)]

        def mean_nn(sample: np.ndarray) -> float:
            d = np.linalg.norm(sample[:, None, :] - sample[None, :, :], axis=-1)
            np.fill_diagonal(d, np.inf)
            return float(d.min(axis=1).mean())

        assert mean_nn(fps) > mean_nn(rnd)

    def test_refuses_to_pad(self):
        """FPS never silently pads; that decision belongs to the caller."""
        pts = classroom_scene(100)
        with pytest.raises(ValueError, match="needs n >= k"):
            farthest_point_sample(pts, 1024, rng(), use_open3d=False)


class TestResampleToFixedSize:
    def test_c1_regression_voxel_in_metres_keeps_unique_points(self):
        """The wrapper's C1 defect, as an executable test.

        The wrapper voxel-downsamples *after* unit-sphere normalisation, so
        ``voxel_size_m: 0.15`` means 15% of the scene radius and ~88% of the 1024
        points handed to the model are duplicates. Here the voxel pre-pass runs in
        metres, before normalisation, so almost every output point is unique.

        This is the test that was missing from the wrapper's suite: its
        BASE_CONFIG sets ``voxel_size_m: None``, so the buggy branch was never
        executed by a passing test.
        """
        pts = classroom_scene()
        out, stats = resample_to_fixed_size(
            pts, 1024, rng(), voxel_size_m=0.15, use_open3d=False
        )
        assert out.shape == (1024, 3)
        assert stats.mode == "fps"
        assert stats.duplicate_fraction < 0.02, (
            f"expected a near-unique sample, got {stats.duplicate_fraction:.1%} "
            "duplicates — the voxel pre-pass is operating in the wrong units"
        )

    def test_voxel_prepass_is_skipped_when_too_aggressive(self):
        """A voxel size that would starve FPS must be declined, not applied.

        Turning a satisfiable FPS request into a padded one is strictly worse,
        so the pre-pass is only accepted if >= target points survive it.
        """
        pts = classroom_scene()
        out, stats = resample_to_fixed_size(
            pts, 1024, rng(), voxel_size_m=5.0, use_open3d=False
        )
        assert stats.mode == "fps"
        assert stats.duplicate_fraction < 0.02

    def test_pad_mode_is_reported(self):
        pts = classroom_scene(300)
        out, stats = resample_to_fixed_size(pts, 1024, rng(), use_open3d=False)
        assert out.shape == (1024, 3)
        assert stats.mode == "pad"
        assert stats.n_unique <= 300
        assert stats.duplicate_fraction > 0.5

    def test_empty_is_flagged_not_hidden(self):
        """Empty input yields zeros AND mode='empty' so the caller can skip it.

        The wrapper returns zeros with only a log warning (finding I4), which a
        classifier then labels confidently.
        """
        out, stats = resample_to_fixed_size(
            np.zeros((0, 3), dtype=np.float32), 1024, rng()
        )
        assert out.shape == (1024, 3)
        assert np.allclose(out, 0.0)
        assert stats.mode == "empty"

    def test_output_is_contiguous_float32(self):
        out, _ = resample_to_fixed_size(
            classroom_scene(3000), 1024, rng(), use_open3d=False
        )
        assert out.dtype == np.float32
        assert out.flags["C_CONTIGUOUS"]

    def test_reproducible_across_calls(self):
        pts = classroom_scene(500)  # pad path: the stochastic one
        a, _ = resample_to_fixed_size(pts, 1024, rng(3), use_open3d=False)
        b, _ = resample_to_fixed_size(pts, 1024, rng(3), use_open3d=False)
        assert np.array_equal(a, b)
