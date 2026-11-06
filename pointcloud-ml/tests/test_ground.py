"""Ground-removal tests, including the one that justifies the stage existing.

``TestWhyGroundRemovalMatters`` is the important one: on a simulated field,
clustering without ground removal returns a single component containing every
object, and with ground removal returns the objects separately.
"""

from __future__ import annotations

import numpy as np
import pytest

from pcml.data.ground import fit_ground_plane, remove_ground


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


#: Where the three simulated objects stand, in metres (x, y).
OBJECT_POSITIONS = ((0.0, 0.0), (5.0, 0.0), (-5.0, 4.0))

#: ``(name, height_m, half_width_m, n_points)`` per object, in the same order as
#: :data:`OBJECT_POSITIONS`. Point counts are load-bearing, not arbitrary — see
#: :func:`field_scene`'s docstring and :class:`TestFixtureItself`.
OBJECT_SPECS = (
    ("person", 1.75, 0.25, 300),
    ("pole", 6.0, 0.15, 250),
    ("tree", 4.0, 1.5, 1500),
)

#: The DBSCAN radius the fixture is built to satisfy, from configs/preprocess.yaml.
CLUSTERING_EPS_M = 0.5


def _near_an_object(cluster, tol_m: float = 1.5) -> bool:
    """Is this cluster one of the three simulated objects, or ground debris?"""
    return any(
        abs(cluster.centroid_m[0] - ex) < tol_m
        and abs(cluster.centroid_m[1] - ey) < tol_m
        for ex, ey in OBJECT_POSITIONS
    )


def field_scene(
    extent_m: float = 8.0,
    density_per_m2: float = 25.0,
    tilt_deg: float = 0.0,
    noise_m: float = 0.02,
    seed: int = 11,
) -> tuple[np.ndarray, np.ndarray]:
    """A mown field with three objects standing on it.

    ``density_per_m2`` is the parameter that matters for the *ground*. Mid-70
    nadir ground density is ~64 pts/m2 at 10 m altitude and ~16 pts/m2 at 20 m
    (``scripts/sensor_feasibility.py``), and the DBSCAN radius is 0.5 m, so at
    realistic altitudes the mean ground spacing is well below the radius and the
    grass is one connected sheet. The default of 25 pts/m2 (~0.20 m spacing)
    sits in that regime.

    The **objects** have the same constraint, and it is easy to get wrong. Each
    object's point count must give a mean spacing comfortably under the 0.5 m
    DBSCAN radius, or whether it clusters as one object, splits in two, or
    fragments below ``min_cluster_points`` becomes seed-dependent — and a
    fixture whose object count varies with the seed makes every test that counts
    clusters flaky for reasons that have nothing to do with the code under test.
    The tree is the binding case because its points fill a 3 x 3 x 4 m volume:
    at 500 points the spacing is 0.416 m (0.83x the radius) and the cluster count
    over 20 seeds was [2, 3, 4, 5]; at 1500 it is 0.288 m (0.58x) and the count
    is a stable 3 across 20 seeds and all three tilt variants. 1500 is also the
    ceiling imposed by ``test_regions``' ``ground > 3 * non_ground`` invariant —
    2000 breaks it. :func:`TestFixtureItself` guards both bounds.

    Returns ``(points, is_object_mask)``. All three objects touch the ground,
    which is what makes them inseparable until the ground is removed.
    """
    g = np.random.default_rng(seed)

    n_ground = int(density_per_m2 * (2 * extent_m) ** 2)
    gx = g.uniform(-extent_m, extent_m, n_ground)
    gy = g.uniform(-extent_m, extent_m, n_ground)
    slope = np.tan(np.radians(tilt_deg))
    gz = gx * slope + g.normal(0.0, noise_m, n_ground)
    ground = np.column_stack([gx, gy, gz])

    def column(cx, cy, height, radius, n):
        ax = g.uniform(cx - radius, cx + radius, n)
        ay = g.uniform(cy - radius, cy + radius, n)
        az = g.uniform(0.0, height, n) + (ax * slope)
        return np.column_stack([ax, ay, az])

    objects = np.vstack(
        [
            column(cx, cy, height, half_width, n)
            for (cx, cy), (_name, height, half_width, n) in zip(
                OBJECT_POSITIONS, OBJECT_SPECS
            )
        ]
    )
    points = np.vstack([ground, objects]).astype(np.float32)
    mask = np.concatenate(
        [np.zeros(len(ground), dtype=bool), np.ones(len(objects), dtype=bool)]
    )
    return points, mask


class TestFixtureItself:
    """Tests of the fixture, not of the code — because this fixture bit us.

    ``field_scene`` originally gave the tree 500 points across a 3 x 3 x 4 m
    volume: a mean spacing of 0.416 m against a 0.5 m DBSCAN radius. That made
    the tree's cluster count seed-dependent ([2, 3, 4, 5] over 20 seeds), which
    surfaced as two intermittent failures in ``test_crops.py`` that looked like
    pipeline bugs and were not. These tests pin both ends of the usable range so
    the fixture cannot drift back.
    """

    @pytest.mark.parametrize("name,height,half_width,n", OBJECT_SPECS)
    def test_every_object_is_denser_than_the_clustering_radius(
        self, name, height, half_width, n
    ):
        """Mean spacing must sit comfortably below eps, not merely under it.

        At 0.83x eps the tree was unstable; at 0.58x it is not. 0.6x is the
        threshold this fixture is held to.
        """
        volume = (2 * half_width) ** 2 * height
        spacing = (volume / n) ** (1.0 / 3.0)
        assert spacing < CLUSTERING_EPS_M * 0.6, (
            f"{name} has {n} points over {volume:.1f} m3 => {spacing:.3f} m mean "
            f"spacing, which is {spacing / CLUSTERING_EPS_M:.2f}x the {CLUSTERING_EPS_M} m "
            "DBSCAN radius. Its cluster count will be seed-dependent. Raise n."
        )

    @pytest.mark.parametrize("seed", [11, 12, 13, 17, 23])
    def test_cluster_count_is_seed_independent(self, seed):
        """The property the crop tests depend on: exactly three objects, always."""
        from pcml.data.clustering import cluster_scene

        points, _ = field_scene(seed=seed)
        non_ground, plane = remove_ground(points, min_height_m=0.15, rng=rng())
        assert plane is not None
        clusters, _ = cluster_scene(
            non_ground,
            eps_m=CLUSTERING_EPS_M,
            min_samples=10,
            min_cluster_points=50,
            use_open3d=False,
        )
        assert len(clusters) == 3, (
            f"seed {seed} gave {len(clusters)} clusters, not 3 — the fixture's "
            "object point counts are back in the unstable regime"
        )

    def test_ground_still_dominates_the_frame(self):
        """The upper bound on object point counts.

        ``test_regions::test_ground_is_retrievable_as_signal`` asserts the ground
        is more than 3x the non-ground. Raising the tree to 2000 points breaks
        that, so this pins the ceiling from the other side.
        """
        from pcml.data.ground import split_ground

        points, _ = field_scene()
        ground, non_ground, plane = split_ground(points, min_height_m=0.15, rng=rng())
        assert plane is not None
        assert ground.shape[0] > non_ground.shape[0] * 3, (
            f"ground={ground.shape[0]} non_ground={non_ground.shape[0]}: the "
            "objects have grown too large relative to the field"
        )


class TestFitGroundPlane:
    def test_finds_level_plane(self):
        points, _ = field_scene()
        plane = fit_ground_plane(points, rng=rng())
        assert plane is not None
        assert plane.tilt_deg < 2.0
        assert abs(plane.ground_z_at_origin()) < 0.1
        assert plane.inlier_fraction > 0.6

    @pytest.mark.parametrize("tilt", [5.0, 15.0, 25.0])
    def test_finds_tilted_plane(self, tilt):
        """Drone attitude and sloped fields are normal, not error cases."""
        points, _ = field_scene(tilt_deg=tilt)
        plane = fit_ground_plane(points, rng=rng(), max_tilt_deg=30.0)
        assert plane is not None
        assert plane.tilt_deg == pytest.approx(tilt, abs=3.0)

    def test_rejects_a_vertical_wall(self):
        """A big planar wall must not be mistaken for the ground.

        Without the horizontality gate, RANSAC would happily fit the largest
        plane in view — and removing it would delete the wall and keep the grass.
        """
        g = np.random.default_rng(3)
        wall = np.column_stack(
            [
                g.normal(0.0, 0.02, 6000),  # x ~ 0: a y-z plane
                g.uniform(-10, 10, 6000),
                g.uniform(0, 8, 6000),
            ]
        ).astype(np.float32)
        plane = fit_ground_plane(wall, rng=rng(), min_inlier_fraction=0.15)
        # Either no plane, or a plane that is not the wall. Never the wall.
        if plane is not None:
            assert plane.tilt_deg < 30.0
            assert abs(plane.normal[0]) < 0.5

    def test_returns_none_when_nothing_is_planar(self):
        blob = rng(4).uniform(-5, 5, (3000, 3)).astype(np.float32)
        plane = fit_ground_plane(blob, distance_threshold_m=0.05, rng=rng(), min_inlier_fraction=0.5)
        assert plane is None

    def test_too_few_points(self):
        assert fit_ground_plane(np.zeros((2, 3), dtype=np.float32), rng=rng()) is None

    def test_deterministic_for_equal_seeds(self):
        points, _ = field_scene()
        a = fit_ground_plane(points, rng=rng(7))
        b = fit_ground_plane(points, rng=rng(7))
        assert a is not None and b is not None
        assert np.allclose(a.normal, b.normal)
        assert a.offset == pytest.approx(b.offset)

    def test_height_above_is_signed_and_in_metres(self):
        points, _ = field_scene()
        plane = fit_ground_plane(points, rng=rng())
        assert plane is not None
        probe = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, -1.0]], dtype=np.float64)
        heights = plane.height_above(probe)
        assert heights[0] == pytest.approx(2.0, abs=0.1)
        assert heights[1] == pytest.approx(-1.0, abs=0.1)

    def test_rejects_bad_parameters(self):
        points, _ = field_scene(extent_m=2.0, density_per_m2=5.0)
        with pytest.raises(ValueError, match="distance_threshold_m must be > 0"):
            fit_ground_plane(points, distance_threshold_m=0.0)
        with pytest.raises(ValueError, match="min_inlier_fraction"):
            fit_ground_plane(points, min_inlier_fraction=0.0)


class TestRemoveGround:
    def test_removes_ground_keeps_objects(self):
        points, is_object = field_scene()
        kept, plane = remove_ground(points, rng=rng())
        assert plane is not None
        # Most ground gone, most object points kept.
        assert kept.shape[0] < points.shape[0] * 0.4
        assert kept.shape[0] > int(is_object.sum()) * 0.7

    def test_no_plane_means_no_deletion(self):
        """A rejected fit must not delete anything."""
        blob = rng(5).uniform(-5, 5, (2000, 3)).astype(np.float32)
        kept, plane = remove_ground(
            blob, distance_threshold_m=0.05, min_inlier_fraction=0.9, rng=rng()
        )
        assert plane is None
        assert kept.shape[0] == blob.shape[0]

    def test_empty_input(self):
        kept, plane = remove_ground(np.zeros((0, 3), dtype=np.float32), rng=rng())
        assert kept.shape[0] == 0
        assert plane is None

    def test_min_height_trims_the_skirt(self):
        points, _ = field_scene()
        low, _ = remove_ground(points, min_height_m=0.0, rng=rng())
        high, _ = remove_ground(points, min_height_m=0.3, rng=rng())
        assert high.shape[0] < low.shape[0]

    def test_output_dtype_preserved(self):
        points, _ = field_scene()
        kept, _ = remove_ground(points, rng=rng())
        assert kept.dtype == np.float32


pytest.importorskip("sklearn", reason="needs a DBSCAN backend")


class TestWhyGroundRemovalMatters:
    """The justification for this module, as an executable experiment.

    Measured behaviour (see the docstrings): at realistic ground density the
    grass bridges every object into one field-sized component, and at low
    density it instead shatters into object-sized fragments. Both regimes make
    cluster-level classification impossible, for opposite reasons.
    """

    def test_grass_bridges_everything_at_realistic_density(self):
        """Dense ground (0.20 m spacing vs a 0.5 m radius) = one giant blob."""
        from pcml.data.clustering import cluster_scene

        points, _ = field_scene(extent_m=8.0, density_per_m2=25.0)
        clusters, _ = cluster_scene(
            points, eps_m=0.5, min_samples=10, min_cluster_points=50, use_open3d=False
        )
        assert clusters, "expected at least one cluster"
        dominant = clusters[0]
        # Essentially the whole scene, objects included, in one component.
        assert dominant.n_points_raw > points.shape[0] * 0.9
        # Field-scale extent: nothing an object classifier can use.
        assert max(dominant.extent_m) > 10.0

    def test_very_sparse_grass_fragments_but_stays_below_the_size_filter(self):
        """At ~0.45 m spacing the ground shatters, and the filter saves us.

        Corrects an earlier claim in this file. The original test asserted that
        sparse-ground fragments "survive the min_cluster_points filter and would
        be classified as real things", and it passed — but for the wrong reason:
        what produced the extra clusters was the simulated *tree* splitting
        (it was too sparse, see TestFixtureItself), not grass fragments.

        Measured, 5 seeds at density 5/m2: the ground does fragment heavily
        (31-37 raw components) but the largest grass fragment is ~24 points,
        so **zero** of them clear the 50-point floor. Only the three real
        objects are kept, every time.
        """
        from pcml.data.clustering import cluster_scene

        points, _ = field_scene(extent_m=15.0, density_per_m2=5.0)
        clusters, n_found = cluster_scene(
            points,
            eps_m=0.5,
            min_samples=10,
            min_cluster_points=50,
            max_clusters=None,
            use_open3d=False,
        )
        assert n_found > 20, "expected the ground to fragment heavily at this density"
        assert len(clusters) == 3, "no grass fragment should clear the 50-point floor"
        assert all(_near_an_object(c) for c in clusters)

    def test_transitional_density_is_where_false_objects_actually_appear(self):
        """The genuinely dangerous regime: ~0.35 m spacing.

        Between "fragments too small to matter" and "everything bridges into
        one" there is a band where ground fragments are large enough to pass as
        objects. Measured over 5 seeds at density 8/m2: 171-199 raw components,
        6-13 kept, of which **3-10 are grass**, the largest ~62 points. Those
        would be handed to the classifier as though they were real things.

        This is the regime that makes ground removal non-negotiable rather than
        merely helpful, and it is the one a drone meets at higher altitude:
        Mid-70 nadir density is ~16 pts/m2 at 20 m and ~7 at 30 m
        (scripts/sensor_feasibility.py).
        """
        from pcml.data.clustering import cluster_scene

        points, _ = field_scene(extent_m=15.0, density_per_m2=8.0)
        clusters, n_found = cluster_scene(
            points,
            eps_m=0.5,
            min_samples=10,
            min_cluster_points=50,
            max_clusters=None,
            use_open3d=False,
        )
        assert n_found > 100
        assert len(clusters) > 3
        false_objects = [c for c in clusters if not _near_an_object(c)]
        assert false_objects, "expected ground fragments to pass as objects here"
        # They are object-sized, which is exactly why they are dangerous.
        assert max(max(c.extent_m) for c in false_objects) < 10.0

    def test_clustering_recovers_the_objects_after_ground_removal(self):
        from pcml.data.clustering import cluster_scene

        points, _ = field_scene(extent_m=8.0, density_per_m2=25.0)
        kept, plane = remove_ground(points, min_height_m=0.15, rng=rng())
        assert plane is not None
        assert plane.tilt_deg < 1.0

        clusters, _ = cluster_scene(
            kept, eps_m=0.5, min_samples=10, min_cluster_points=50, use_open3d=False
        )
        assert len(clusters) == 3

        # Each is object-sized, not field-sized.
        for c in clusters:
            assert max(c.extent_m) < 8.0

        # And each simulated object is accounted for by some cluster.
        found = [(c.centroid_m[0], c.centroid_m[1]) for c in clusters]
        for ex, ey in OBJECT_POSITIONS:
            assert any(
                abs(fx - ex) < 1.0 and abs(fy - ey) < 1.0 for fx, fy in found
            ), f"no cluster near expected object at ({ex}, {ey}); found {found}"

    def test_ground_plane_feeds_the_height_feature(self):
        """With a plane, height_above_ground stops being a min-z guess."""
        from pcml.data.clustering import cluster_scene, prepare_cluster_batch

        points, _ = field_scene()
        kept, plane = remove_ground(points, min_height_m=0.15, rng=rng())
        assert plane is not None
        clusters, _ = cluster_scene(
            kept, eps_m=0.5, min_samples=10, min_cluster_points=50, use_open3d=False
        )
        _, aux, _ = prepare_cluster_batch(
            clusters,
            num_points=256,
            rng=rng(),
            ground_z_m=plane.ground_z_at_origin(),
            use_open3d=False,
        )
        # The 6 m pole's centroid sits higher above ground than the person's.
        heights = aux[:, 5]
        assert heights.max() > heights.min() + 1.0
