"""Region-proposal tests: the seam that lets the domain change without a rewrite.

The point of these is that both proposers return the *same* record type, so
everything downstream — normalisation, FPS, aux features, adapter, model — is
indifferent to which one produced the regions.
"""

from __future__ import annotations

import numpy as np
import pytest

from pcml.data.clustering import Cluster, prepare_cluster_batch
from pcml.data.ground import split_ground
from pcml.data.regions import (
    build_proposer,
    propose_dbscan_instances,
    propose_ground_tiles,
)

from test_ground import OBJECT_POSITIONS, field_scene  # noqa: F401


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def crop_surface(extent_m: float = 6.0, density: float = 60.0, seed: int = 2):
    """A continuous surface with height structure — a crop field, roughly.

    No discrete objects at all: this is the regime where instance clustering is
    the wrong question and tiling is the right one.
    """
    g = np.random.default_rng(seed)
    n = int(density * (2 * extent_m) ** 2)
    x = g.uniform(-extent_m, extent_m, n)
    y = g.uniform(-extent_m, extent_m, n)
    # Row structure: a sinusoidal canopy height across x, plus noise.
    z = 0.4 + 0.25 * np.sin(x * 2.0) + g.normal(0.0, 0.03, n)
    return np.column_stack([x, y, z]).astype(np.float32)


class TestGroundTiles:
    def test_tiles_a_continuous_surface(self):
        pts = crop_surface(extent_m=6.0)
        regions, n_found = propose_ground_tiles(
            pts, tile_size_m=2.0, min_region_points=50, max_regions=None
        )
        # 12 m / 2 m = 6 tiles per axis = ~36 tiles.
        assert 30 <= n_found <= 42
        assert len(regions) == n_found
        assert all(isinstance(r, Cluster) for r in regions)

    def test_tile_extent_matches_tile_size(self):
        pts = crop_surface(extent_m=6.0)
        regions, _ = propose_ground_tiles(pts, tile_size_m=2.0, max_regions=None)
        for r in regions:
            assert r.extent_m[0] <= 2.01
            assert r.extent_m[1] <= 2.01

    def test_smaller_tiles_give_more_regions(self):
        pts = crop_surface(extent_m=6.0)
        coarse, _ = propose_ground_tiles(pts, tile_size_m=3.0, max_regions=None)
        fine, _ = propose_ground_tiles(pts, tile_size_m=1.0, max_regions=None)
        assert len(fine) > len(coarse) * 4

    def test_sparse_tiles_are_dropped_not_padded(self):
        """A tile with too few points carries no surface information."""
        pts = crop_surface(extent_m=6.0, density=2.0)
        regions, _ = propose_ground_tiles(
            pts, tile_size_m=0.5, min_region_points=50, max_regions=None
        )
        assert regions == []

    def test_max_regions_caps_largest_first(self):
        pts = crop_surface(extent_m=6.0)
        regions, n_found = propose_ground_tiles(
            pts, tile_size_m=1.0, max_regions=10
        )
        assert n_found > 10
        assert len(regions) == 10
        counts = [r.n_points_raw for r in regions]
        assert counts == sorted(counts, reverse=True)

    def test_overlap_factor_adds_shifted_grids(self):
        pts = crop_surface(extent_m=6.0)
        plain, n_plain = propose_ground_tiles(
            pts, tile_size_m=2.0, overlap_factor=1, max_regions=None
        )
        overlapped, n_over = propose_ground_tiles(
            pts, tile_size_m=2.0, overlap_factor=2, max_regions=None
        )
        # 2x2 shifted grids: roughly 4x the regions, and tiles still tile-sized.
        assert n_over > n_plain * 3
        for r in overlapped:
            assert r.extent_m[0] <= 2.01

    def test_empty_input(self):
        regions, n_found = propose_ground_tiles(np.zeros((0, 3), dtype=np.float32))
        assert regions == []
        assert n_found == 0

    def test_rejects_bad_parameters(self):
        pts = crop_surface(extent_m=2.0)
        with pytest.raises(ValueError, match="tile_size_m must be > 0"):
            propose_ground_tiles(pts, tile_size_m=0.0)
        with pytest.raises(ValueError, match="overlap_factor must be >= 1"):
            propose_ground_tiles(pts, overlap_factor=0)

    def test_nan_rows_dropped(self):
        pts = np.vstack(
            [crop_surface(extent_m=3.0), np.full((5, 3), np.nan, dtype=np.float32)]
        )
        regions, _ = propose_ground_tiles(pts, tile_size_m=2.0, max_regions=None)
        assert all(np.isfinite(r.points).all() for r in regions)


class TestProposersAreInterchangeable:
    """The property that makes the domain pivot cheap."""

    def test_both_feed_prepare_cluster_batch_identically(self):
        pytest.importorskip("sklearn")
        scene, _ = field_scene(extent_m=8.0, density_per_m2=25.0)
        _, non_ground, plane = split_ground(scene, min_height_m=0.15, rng=rng())
        assert plane is not None

        objects, _ = propose_dbscan_instances(non_ground, use_open3d=False)
        tiles, _ = propose_ground_tiles(scene, tile_size_m=2.0, max_regions=8)

        for regions in (objects, tiles):
            batch, aux, stats = prepare_cluster_batch(
                regions, num_points=256, rng=rng(), use_open3d=False
            )
            assert batch.shape == (len(regions), 256, 3)
            assert aux.shape == (len(regions), 6)
            assert len(stats) == len(regions)

    def test_ground_is_retrievable_as_signal(self):
        """Agriculture and road quality need the ground kept, not discarded."""
        pytest.importorskip("sklearn")
        scene, _ = field_scene(extent_m=8.0, density_per_m2=25.0)
        ground, non_ground, plane = split_ground(scene, min_height_m=0.15, rng=rng())
        assert plane is not None
        # The ground is the majority of the frame and is returned, not dropped.
        assert ground.shape[0] > non_ground.shape[0] * 3
        # And it is genuinely flat: that is what makes it a usable surface.
        heights = plane.height_above(ground.astype(np.float64))
        assert np.abs(heights).max() <= 0.15 + 1e-6

    def test_remove_ground_still_agrees_with_split_ground(self):
        from pcml.data.ground import remove_ground

        scene, _ = field_scene(extent_m=8.0, density_per_m2=25.0)
        kept, plane_a = remove_ground(scene, min_height_m=0.15, rng=rng(1))
        _, non_ground, plane_b = split_ground(scene, min_height_m=0.15, rng=rng(1))
        assert plane_a is not None and plane_b is not None
        assert np.array_equal(kept, non_ground)


class TestBuildProposer:
    def test_builds_dbscan_from_config(self):
        pytest.importorskip("sklearn")
        scene, _ = field_scene(extent_m=8.0, density_per_m2=25.0)
        _, non_ground, _ = split_ground(scene, min_height_m=0.15, rng=rng())
        proposer = build_proposer(
            {"method": "dbscan_instances", "eps_m": 0.5, "use_open3d": False}
        )
        regions, _ = proposer(non_ground)
        assert len(regions) == 3

    def test_builds_tiles_from_config(self):
        proposer = build_proposer(
            {"method": "ground_tiles", "tile_size_m": 2.0, "max_regions": None}
        )
        regions, _ = proposer(crop_surface(extent_m=6.0))
        assert len(regions) > 10

    def test_irrelevant_keys_are_tolerated(self):
        """One config file can carry settings for several methods."""
        proposer = build_proposer(
            {
                "method": "ground_tiles",
                "tile_size_m": 2.0,
                "max_regions": None,
                "eps_m": 0.5,  # a dbscan key, meaningless here
                "min_samples": 10,
            }
        )
        regions, _ = proposer(crop_surface(extent_m=4.0))
        assert regions

    def test_unknown_method_names_alternatives(self):
        with pytest.raises(ValueError, match="available:"):
            build_proposer({"method": "magic"})

    def test_missing_method(self):
        with pytest.raises(KeyError, match="'method'"):
            build_proposer({"tile_size_m": 2.0})
