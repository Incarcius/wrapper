"""Clustering tests — the C2 fix: a scene becomes instances, each normalised alone."""

from __future__ import annotations

import numpy as np
import pytest

from pcml.data.clustering import (
    AUX_FEATURE_NAMES,
    Cluster,
    cluster_scene,
    finite_points_only,
    prepare_cluster_batch,
)


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def blob(center, n=200, scale=0.12, seed=0) -> np.ndarray:
    g = np.random.default_rng(seed)
    return (g.normal(0.0, scale, (n, 3)) + np.asarray(center, dtype=np.float64)).astype(
        np.float32
    )


def three_object_scene() -> np.ndarray:
    """Three well-separated objects of different sizes, in metres."""
    return np.vstack(
        [
            blob([0, 0, 0], n=400, scale=0.15, seed=1),
            blob([5, 0, 0], n=250, scale=0.25, seed=2),
            blob([0, 6, 1], n=120, scale=0.10, seed=3),
        ]
    )


pytest.importorskip("sklearn", reason="clustering needs Open3D or scikit-learn")


class TestClusterScene:
    def test_separates_objects(self):
        clusters, n_found = cluster_scene(
            three_object_scene(), eps_m=0.5, min_samples=10, min_cluster_points=50
        )
        assert n_found == 3
        assert len(clusters) == 3

    def test_sorted_by_size_descending(self):
        clusters, _ = cluster_scene(three_object_scene(), min_cluster_points=50)
        counts = [c.n_points_raw for c in clusters]
        assert counts == sorted(counts, reverse=True)

    def test_small_clusters_dropped_but_counted(self):
        """A tiny cluster is excluded from classification yet still reported.

        n_found vs len(clusters) is how the caller learns that objects were
        dropped rather than never seen.
        """
        scene = np.vstack([blob([0, 0, 0], n=400, seed=1), blob([8, 0, 0], n=60, seed=2)])
        clusters, n_found = cluster_scene(scene, min_cluster_points=100)
        assert n_found == 2
        assert len(clusters) == 1

    def test_max_clusters_cap(self):
        scene = np.vstack([blob([3 * i, 0, 0], n=120, seed=i) for i in range(6)])
        clusters, n_found = cluster_scene(scene, min_cluster_points=50, max_clusters=3)
        assert n_found == 6
        assert len(clusters) == 3

    def test_noise_points_are_not_a_cluster(self):
        scene = np.vstack(
            [blob([0, 0, 0], n=400, seed=1), rng(9).uniform(-20, 20, (15, 3)).astype(np.float32)]
        )
        clusters, _ = cluster_scene(scene, eps_m=0.5, min_samples=10, min_cluster_points=50)
        assert len(clusters) == 1
        assert clusters[0].n_points_raw <= 400

    def test_empty_scene(self):
        clusters, n_found = cluster_scene(np.zeros((0, 3), dtype=np.float32))
        assert clusters == []
        assert n_found == 0

    def test_nan_rows_are_dropped(self):
        scene = three_object_scene()
        poisoned = np.vstack([scene, np.full((5, 3), np.nan, dtype=np.float32)])
        clusters, _ = cluster_scene(poisoned, min_cluster_points=50)
        assert len(clusters) == 3
        assert all(np.isfinite(c.points).all() for c in clusters)

    def test_geometry_is_in_metres(self):
        """Centroid and extent must stay in real units for the output contract."""
        scene = blob([5.0, -2.0, 1.0], n=500, scale=0.2, seed=4)
        clusters, _ = cluster_scene(scene, min_cluster_points=50)
        assert len(clusters) == 1
        cx, cy, cz = clusters[0].centroid_m
        assert cx == pytest.approx(5.0, abs=0.1)
        assert cy == pytest.approx(-2.0, abs=0.1)
        assert cz == pytest.approx(1.0, abs=0.1)
        # ~6 sigma span for a 0.2 m sigma gaussian blob
        assert all(0.5 < e < 2.5 for e in clusters[0].extent_m)


class TestFinitePointsOnly:
    def test_drops_nan_and_inf(self):
        pts = np.array(
            [[0, 0, 0], [np.nan, 0, 0], [0, np.inf, 0], [1, 1, 1]], dtype=np.float32
        )
        assert finite_points_only(pts).shape == (2, 3)


class TestPrepareClusterBatch:
    def test_batch_shape_is_one_tensor_per_frame(self):
        clusters, _ = cluster_scene(three_object_scene(), min_cluster_points=50)
        batch, aux, stats = prepare_cluster_batch(
            clusters, num_points=256, rng=rng(), use_open3d=False
        )
        assert batch.shape == (3, 256, 3)
        assert aux.shape == (3, len(AUX_FEATURE_NAMES))
        assert len(stats) == 3

    def test_each_cluster_normalised_independently(self):
        """The C2 fix: every cluster sits in its own unit sphere, centred.

        Under the wrapper's whole-scene normalisation, a cluster 5 m off-centre
        would be nowhere near the origin.
        """
        clusters, _ = cluster_scene(three_object_scene(), min_cluster_points=50)
        batch, _, _ = prepare_cluster_batch(
            clusters, num_points=256, rng=rng(), use_open3d=False
        )
        for i in range(batch.shape[0]):
            assert np.allclose(batch[i].mean(axis=0), 0.0, atol=1e-5)
            assert np.linalg.norm(batch[i], axis=1).max() == pytest.approx(1.0, abs=1e-4)

    def test_aux_features_carry_real_scale(self):
        """The big object must have a larger extent than the small one.

        This is the information unit-sphere normalisation destroys, which is why
        it is recorded separately.
        """
        clusters, _ = cluster_scene(three_object_scene(), min_cluster_points=50)
        _, aux, _ = prepare_cluster_batch(
            clusters, num_points=256, rng=rng(), use_open3d=False
        )
        extents = aux[:, :3].max(axis=1)
        assert extents.max() > extents.min() * 1.5

        # range_to_centroid for a cluster at ~5 m must be ~5 m
        ranges = aux[:, 4]
        assert ranges.max() > 4.0

    def test_empty_cluster_list(self):
        batch, aux, stats = prepare_cluster_batch([], num_points=64, rng=rng())
        assert batch.shape == (0, 64, 3)
        assert aux.shape == (0, len(AUX_FEATURE_NAMES))
        assert stats == []

    def test_reproducible(self):
        clusters, _ = cluster_scene(three_object_scene(), min_cluster_points=50)
        a, _, _ = prepare_cluster_batch(clusters, 256, rng(5), use_open3d=False)
        b, _, _ = prepare_cluster_batch(clusters, 256, rng(5), use_open3d=False)
        assert np.array_equal(a, b)

    def test_voxel_size_is_metres_here_too(self):
        cluster = Cluster.from_points(blob([0, 0, 0], n=3000, scale=0.5, seed=6))
        _, _, stats = prepare_cluster_batch(
            [cluster], num_points=512, rng=rng(), voxel_size_m=0.05, use_open3d=False
        )
        assert stats[0].mode == "fps"
        assert stats[0].duplicate_fraction < 0.02

    def test_rejects_bad_num_points(self):
        with pytest.raises(ValueError, match="num_points must be > 0"):
            prepare_cluster_batch([], num_points=0, rng=rng())
