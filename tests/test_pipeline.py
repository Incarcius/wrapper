import numpy as np
import pytest

from livox_wrapper.pipeline import (
    ConfigError,
    PointCloudPipeline,
    StageTimings,
    validate_config,
)


# Open3D is installed in CI but not importable in every dev environment (its
# wheel needs libusb), so the base config picks paths that work on the
# NumPy/scikit-learn fallbacks. The pipeline's behaviour is identical either
# way; only the implementation of FPS and DBSCAN differs.
BASE_CONFIG = {
    "seed": 0,
    "roi_filter": {"enabled": False, "mode": "none"},
    "outlier_removal": {"method": "none"},
    # Off for the object-only fixtures below, which are bare blobs with no ground
    # in them: RANSAC would happily fit a plane through a Gaussian blob and
    # delete most of it. The ground stage has its own fixtures, which include a
    # ground sheet — see TestGroundRemoval.
    "ground_removal": {"enabled": False},
    "clustering": {"enabled": True, "use_open3d": False},
    "normalization": {"center": True, "scale_to_unit_sphere": True},
    "downsampling": {"voxel_size_m": None, "target_points": 1024, "method": "numpy_fps"},
    "output": {"format": "numpy", "dtype": "float32"},
    "logging": {"log_stage_timings": False, "log_every_n_frames": 30},
}


def make_pipeline(overrides=None):
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in BASE_CONFIG.items()}
    if overrides:
        for section, vals in overrides.items():
            if isinstance(vals, dict):
                cfg.setdefault(section, {})
                cfg[section].update(vals)
            else:
                cfg[section] = vals
    return PointCloudPipeline(cfg)


def blob(center, n=400, scale=0.1, seed=0):
    """A compact Gaussian blob: stands in for one object, in metres."""
    rng = np.random.default_rng(seed)
    return (rng.normal(loc=center, scale=scale, size=(n, 3))).astype(np.float32)


def multi_object_scene(n_per=400):
    """Three well-separated objects of visibly different sizes, in metres.

    Separations are far larger than clustering.eps_m (0.5 m), so DBSCAN must
    find exactly three instances; the extents differ so that per-cluster
    normalisation is distinguishable from whole-scene normalisation.
    """
    return np.vstack(
        [
            blob([0.0, 0.0, 0.0], n=n_per, scale=0.10, seed=1),
            blob([5.0, 0.0, 0.0], n=n_per, scale=0.40, seed=2),
            blob([0.0, 6.0, 1.0], n=n_per, scale=0.80, seed=3),
        ]
    ).astype(np.float32)


def field_scene(ground_z=-1.5, n_ground=6000, n_per_object=400, extent=8.0):
    """A sports field: a dense planar ground sheet with objects standing on it.

    This is the fixture the ground stage exists for. The ground spans
    ``2 * extent`` metres with ~15 points/m², i.e. a mean spacing well under the
    0.5 m clustering radius, so the sheet is a single connected component — and
    every object standing on it is connected *through* it to every other object.

    The four objects are vertical columns rising 2 m from the surface, placed at
    the corners so that once the ground is gone they are metres apart and must
    separate. Columns are also the honest shape here: they are thin and vertical,
    which is what the plane fit's horizontality gate has to reject.
    """
    rng = np.random.default_rng(7)
    ground = np.column_stack(
        [
            rng.uniform(-extent, extent, size=n_ground),
            rng.uniform(-extent, extent, size=n_ground),
            ground_z + rng.normal(0.0, 0.02, size=n_ground),
        ]
    )
    objects = []
    for i, (cx, cy) in enumerate([(-6, -6), (6, -6), (-6, 6), (6, 6)]):
        objects.append(
            np.column_stack(
                [
                    cx + rng.normal(0.0, 0.12, size=n_per_object),
                    cy + rng.normal(0.0, 0.12, size=n_per_object),
                    ground_z + rng.uniform(0.05, 2.0, size=n_per_object),
                ]
            )
        )
    return np.vstack([ground] + objects).astype(np.float32)


# ---------------------------------------------------------------------------
# Ground removal. On a field this is a prerequisite for the whole
# cluster-then-classify approach, not an optional preprocessing stage, and the
# first two tests are the measurement that says so.
# ---------------------------------------------------------------------------
class TestGroundRemoval:
    def test_leaving_the_ground_in_merges_the_whole_scene(self):
        """Why the stage is mandatory, stated as a failing scene.

        With the grass still present, DBSCAN at a 0.5 m radius walks from one
        object through the ground sheet to every other, so the frame collapses
        into one component holding nearly all of it. Per-cluster classification
        is impossible in that state: there is one "object", and it is the field.
        """
        pipeline = make_pipeline({"ground_removal": {"enabled": False}})
        scene = field_scene()
        frame, _ = pipeline.process(scene)

        assert frame.ground_plane is None
        assert frame.n_clusters == 1
        assert frame.clusters[0].n_points_raw > 0.9 * len(scene)
        # And the single cluster spans the whole field rather than an object.
        assert max(frame.clusters[0].extent_m) > 10.0

    def test_removing_the_ground_separates_the_objects(self):
        """The same scene, with the stage enabled: one cluster per object."""
        pipeline = make_pipeline({"ground_removal": {"enabled": True}})
        frame, _ = pipeline.process(field_scene())

        assert frame.n_clusters == 4
        assert frame.n_clusters_found == 4
        for cluster in frame.clusters:
            # Each is an object-sized column, not a field-sized blob.
            assert max(cluster.extent_m) < 3.0
            assert cluster.n_points_raw >= 50

    def test_the_plane_is_reported_and_most_points_are_dropped(self):
        pipeline = make_pipeline({"ground_removal": {"enabled": True}})
        scene = field_scene()
        frame, timings = pipeline.process(scene)

        plane = frame.ground_plane
        assert plane is not None
        assert plane.tilt_deg < 5.0  # a level field, fitted from a level sheet
        assert plane.inlier_fraction > 0.5
        assert frame.n_points_clean == len(scene)
        # The 6000 ground points go; the four 400-point columns largely stay.
        assert frame.n_points_non_ground < 0.3 * len(scene)
        assert frame.n_points_non_ground > 1000
        assert timings.ground_removal > 0.0

    def test_the_fitted_plane_feeds_the_height_feature(self):
        """``ground_z_at_origin()`` replaces the minimum-z guess.

        ``prepare_cluster_batch`` falls back to the lowest z across all clusters
        as a ground proxy, which is biased upward by exactly the skirt that
        ``min_height_m`` trims. With a real plane the feature measures height
        above the *surface*, which is what it claims to be.
        """
        ground_z = -1.5
        pipeline = make_pipeline({"ground_removal": {"enabled": True}})
        frame, _ = pipeline.process(field_scene(ground_z=ground_z))

        assert frame.ground_plane is not None
        heights = frame.aux[:, 5]  # height_above_ground_m
        centroid_z = np.array([c.centroid_m[2] for c in frame.clusters])
        assert heights == pytest.approx(centroid_z - ground_z, abs=0.05)

        # The min-z proxy would have measured from the trimmed base instead, so
        # the two differ by about the clearance that was trimmed away.
        proxy_z = min(float(c.points[:, 2].min()) for c in frame.clusters)
        assert proxy_z > ground_z + 0.2
        assert not np.allclose(heights, centroid_z - proxy_z, atol=0.05)

    def test_a_rejected_fit_deletes_nothing(self):
        """Declining beats guessing: removing a wrongly-fitted ground would
        delete real objects, so an unacceptable fit leaves the frame untouched —
        and says so, rather than failing silently."""
        pipeline = make_pipeline(
            {"ground_removal": {"enabled": True, "min_inlier_fraction": 1.0}}
        )
        scene = field_scene()
        frame, _ = pipeline.process(scene)

        assert frame.ground_plane is None
        assert frame.n_points_non_ground == frame.n_points_clean == len(scene)

    def test_a_rejected_fit_is_logged(self, caplog):
        pipeline = make_pipeline(
            {"ground_removal": {"enabled": True, "min_inlier_fraction": 1.0}}
        )
        with caplog.at_level("WARNING", logger="livox_wrapper.pipeline"):
            pipeline.process(field_scene())
        assert any("no ground plane found" in r.getMessage() for r in caplog.records)

    def test_the_stage_is_reproducible_from_the_seed(self):
        """RANSAC draws from the pipeline's seeded generator (finding A1 applies
        to the plane fit as much as to the sampling)."""
        scene = field_scene()
        first, _ = make_pipeline({"ground_removal": {"enabled": True}, "seed": 3}).process(scene)
        second, _ = make_pipeline({"ground_removal": {"enabled": True}, "seed": 3}).process(scene)
        assert first.n_points_non_ground == second.n_points_non_ground
        assert np.array_equal(first.batch, second.batch)

    def test_an_empty_frame_skips_the_stage(self):
        pipeline = make_pipeline({"ground_removal": {"enabled": True}})
        frame, _ = pipeline.process(np.zeros((0, 3), dtype=np.float32))
        assert frame.ground_plane is None
        assert frame.n_clusters == 0

    def test_the_log_line_carries_the_ground_timing(self, caplog):
        pipeline = make_pipeline({
            "ground_removal": {"enabled": True},
            "logging": {"log_stage_timings": True, "log_every_n_frames": 1},
        })
        with caplog.at_level("INFO", logger="livox_wrapper.pipeline"):
            pipeline.process(field_scene())
        assert any("ground=" in r.getMessage() for r in caplog.records)


class TestNormalize:
    def test_zero_centered(self):
        pipeline = make_pipeline()
        pts = np.random.uniform(-10, 10, size=(500, 3)).astype(np.float32) + np.array([5, -3, 2])
        out = pipeline.normalize(pts)
        # atol is float32-appropriate: normalize() no longer upcasts to float64
        # (finding A2), so the tolerance reflects the dtype actually in use.
        assert np.allclose(out.mean(axis=0), 0, atol=1e-5)

    def test_stays_float32(self):
        """Finding A2: the old version returned float64 and doubled the memory
        traffic of every downstream stage on a bandwidth-bound device."""
        pipeline = make_pipeline()
        pts = np.random.uniform(-10, 10, size=(500, 3)).astype(np.float32)
        assert pipeline.normalize(pts).dtype == np.float32

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

    def test_sampling_is_reproducible_from_the_seed(self):
        """Finding A1: the three np.random.choice calls used NumPy's global RNG,
        so the same cluster produced a different model input on every run."""
        pts = np.random.uniform(-1, 1, size=(200, 3)).astype(np.float32)  # < target: pads
        first = make_pipeline({"seed": 7}).farthest_point_sampling(pts, 1024)
        second = make_pipeline({"seed": 7}).farthest_point_sampling(pts, 1024)
        third = make_pipeline({"seed": 8}).farthest_point_sampling(pts, 1024)
        assert np.array_equal(first, second)
        assert not np.array_equal(first, third)


class TestOutlierRemovalDBSCAN:
    def test_removes_synthetic_noise(self):
        # optional dependency (outlier_removal.method: dbscan); exc_type also covers a broken install
        pytest.importorskip("sklearn", exc_type=ImportError)
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


# ---------------------------------------------------------------------------
# Finding A5 / C1: the voxel pre-pass, which is where the critical bug lived.
#
# The old suite set voxel_size_m: None, so the branch containing C1 was never
# executed: ten passing tests and the most damaging defect in the file sat in
# untested code.
#
# Two properties discriminate the orderings, and neither is the output point
# count: once the voxel pass leaves at least target_points points, FPS returns
# exactly target_points distinct points either way, so counts and
# SamplingStats.n_unique are identical under both orderings and carry no
# information. The tests below use properties that do differ.
# ---------------------------------------------------------------------------
def _cube(edge_m, n=6000, seed=0):
    """``n`` points uniformly filling a cube of edge ``edge_m`` metres."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, edge_m, size=(n, 3)).astype(np.float32)


def _single_cluster_batch(points, voxel_size_m, target_points=512, normalise=True):
    pipeline = make_pipeline({
        "downsampling": {"voxel_size_m": voxel_size_m, "target_points": target_points},
        "clustering": {"min_cluster_points": 10, "eps_m": 2.0 * float(points.max()) + 1.0},
        "normalization": {"center": normalise, "scale_to_unit_sphere": normalise},
    })
    frame, _ = pipeline.process(points)
    assert frame.n_clusters == 1, "the fixture must produce exactly one cluster"
    return frame


class TestVoxelPrePassRunsInMetres:
    def test_voxel_size_is_metric_so_the_result_is_scale_dependent(self):
        """The C1 regression test, as an exact identity.

        A voxel grid of edge ``v`` metres applied to a cloud scaled by ``k`` puts
        points in the same cells as a grid of edge ``v / k`` applied to the
        original cloud. Normalisation then removes the overall factor ``k``, so
        the two normalised batches must be *identical*::

            batch(cloud * k, v)  ==  batch(cloud, v / k)

        That identity holds only if the voxel pass sees metres. Under C1 — voxel
        after unit-sphere normalisation — the cloud has already been mapped onto
        radius 1 before voxelisation, so the result is scale-invariant: C1 makes
        ``batch(cloud * k, v) == batch(cloud, v)`` instead, which is what the
        second half of this test rules out.

        ``k`` is a power of two so the scaling itself is exact in float32 and
        cannot be mistaken for a rounding difference.
        """
        cloud = _cube(10.0)
        k, v = 8.0, 0.4

        scaled = _single_cluster_batch(cloud * k, v).batch[0]
        finer = _single_cluster_batch(cloud, v / k).batch[0]
        unchanged = _single_cluster_batch(cloud, v).batch[0]

        assert np.allclose(scaled, finer, atol=1e-5), (
            "voxel_size_m is not acting in metres: scaling the cloud by k did not "
            "match dividing the voxel size by k"
        )
        assert not np.allclose(scaled, unchanged, atol=1e-5), (
            "the voxel pass is scale-invariant, which means it is running after "
            "normalisation — this is finding C1"
        )

    def test_output_points_are_voxel_centroids_in_metres(self):
        """Direct check of the metric semantics the config documents.

        With normalisation switched off the batch is still in metres, so each
        sampled point can be matched against the voxel it came from: the cell
        index is ``floor(p / voxel_size_m)`` with ``voxel_size_m`` in metres, and
        the point is the mean of the raw points in that cell.
        """
        v = 0.5
        cloud = _cube(10.0)
        batch = _single_cluster_batch(cloud, v, target_points=256, normalise=False).batch[0]

        keys = np.floor(batch / v).astype(np.int64)
        assert len(np.unique(keys, axis=0)) == len(keys), "two output points share a voxel"

        raw_keys = np.floor(cloud / v).astype(np.int64)
        for point, key in zip(batch[:20], keys[:20]):
            members = cloud[(raw_keys == key).all(axis=1)]
            assert len(members) > 0, "output point does not sit in an occupied voxel"
            assert point == pytest.approx(members.mean(axis=0), abs=1e-5)

    def test_whether_the_voxel_pass_applies_is_decided_in_metres(self):
        """The metric voxel size decides whether the pre-pass runs at all.

        When it runs, the sampled points are cell *centroids* — averages of
        several raw points, so they are not themselves points of the input cloud.
        When the grid is too coarse to leave ``target_points`` cells the pass is
        declined and FPS selects raw points instead. Which of the two happens is
        a function of the voxel size in metres against the cluster's metric
        extent, so this is also a statement that ``voxel_size_m`` is a length.
        """
        cloud = _cube(10.0, n=20000)
        raw_rows = {tuple(row) for row in cloud}

        def fraction_of_raw_points(voxel_size_m):
            batch = _single_cluster_batch(
                cloud, voxel_size_m, target_points=256, normalise=False
            ).batch[0]
            return sum(tuple(row) in raw_rows for row in batch) / len(batch)

        # 1 m cells over a 10 m cube: ~1000 occupied cells of ~20 points each,
        # so the pass applies and nearly every output point is an average.
        assert fraction_of_raw_points(1.0) < 0.05
        # 5 m cells leave 8 cells, far fewer than the 256 requested: declined,
        # so every output point is a raw point.
        assert fraction_of_raw_points(5.0) == 1.0

    def test_normalisation_still_runs_after_sampling(self):
        """C1 reorders the stages; it does not drop normalisation."""
        batch = _single_cluster_batch(_cube(4.0), 0.2).batch[0]
        assert np.linalg.norm(batch, axis=1).max() == pytest.approx(1.0, abs=1e-5)
        assert np.allclose(batch.mean(axis=0), 0.0, atol=1e-5)

    def test_voxel_pre_pass_is_declined_when_it_would_force_padding(self):
        """A voxel size coarse enough to leave fewer than target_points must be
        skipped: turning a satisfiable FPS request into a padded one is strictly
        worse than not voxelising at all."""
        frame = _single_cluster_batch(_cube(2.0, n=4000), voxel_size_m=5.0, target_points=1024)
        assert frame.stats[0].mode == "fps"
        assert frame.stats[0].n_unique == 1024


# ---------------------------------------------------------------------------
# Finding C2: normalisation is per cluster, not per scene.
# ---------------------------------------------------------------------------
class TestPerClusterNormalisation:
    def test_each_cluster_is_independently_centred_and_scaled(self):
        pipeline = make_pipeline({"clustering": {"eps_m": 0.5, "min_cluster_points": 50}})
        frame, _ = pipeline.process(multi_object_scene())

        assert frame.n_clusters == 3, "three separated objects must give three clusters"
        assert frame.batch.shape == (3, 1024, 3)
        assert frame.batch.dtype == np.float32

        for i in range(frame.n_clusters):
            pts = frame.batch[i]
            assert np.allclose(pts.mean(axis=0), 0.0, atol=1e-5), f"cluster {i} not centred"
            assert np.linalg.norm(pts, axis=1).max() == pytest.approx(1.0, abs=1e-5), (
                f"cluster {i} not on the unit sphere"
            )

    def test_whole_scene_normalisation_would_not_satisfy_that(self):
        """The C2 defect, reproduced deliberately through the debug switch.

        With clustering off, the three objects become one sample: the two
        off-centre objects are then nowhere near the origin and nowhere near
        radius 1. That is precisely what the classifier was being handed.
        """
        pipeline = make_pipeline({"clustering": {"enabled": False}})
        frame, _ = pipeline.process(multi_object_scene())
        assert frame.whole_scene is True
        assert frame.n_clusters == 1
        # One unit sphere for the whole ±6 m scene: each object occupies a small
        # off-centre patch of it.
        assert np.linalg.norm(frame.batch[0], axis=1).max() == pytest.approx(1.0, abs=1e-5)

    def test_metric_geometry_survives_normalisation(self):
        """Invariant 3: centroids and extents come back in sensor-frame metres,
        even though the tensor the network sees has lost absolute scale."""
        pipeline = make_pipeline()
        frame, _ = pipeline.process(multi_object_scene())
        centroids = sorted(c.centroid_m for c in frame.clusters)
        assert centroids[0] == pytest.approx((0.0, 0.0, 0.0), abs=0.1)
        assert centroids[1] == pytest.approx((0.0, 6.0, 1.0), abs=0.2)
        assert centroids[2] == pytest.approx((5.0, 0.0, 0.0), abs=0.2)
        # The widest object must have the largest extent; unit-sphere
        # normalisation would have erased that distinction.
        extents = [max(c.extent_m) for c in frame.clusters]
        assert max(extents) > 3 * min(extents)

    def test_aux_features_are_metric(self):
        pipeline = make_pipeline()
        frame, _ = pipeline.process(multi_object_scene())
        assert frame.aux.shape == (3, 6)
        for cluster, aux in zip(frame.clusters, frame.aux):
            assert tuple(aux[:3]) == pytest.approx(cluster.extent_m, abs=1e-5)


# ---------------------------------------------------------------------------
# Degenerate frames. Invariant 1: these are normal inputs, not errors.
# ---------------------------------------------------------------------------
class TestDegenerateFrames:
    def test_filtering_removes_every_point(self):
        """An ROI box that excludes the whole cloud must yield an empty batch,
        not an exception and not 1024 zeros pretending to be an object."""
        pipeline = make_pipeline({
            "roi_filter": {
                "enabled": True, "mode": "box",
                "x_min": 100.0, "x_max": 101.0,
                "y_min": 100.0, "y_max": 101.0,
                "z_min": 100.0, "z_max": 101.0,
            }
        })
        frame, timings = pipeline.process(multi_object_scene())
        assert frame.n_clusters == 0
        assert frame.n_clusters_found == 0
        assert frame.batch.shape == (0, 1024, 3)
        assert frame.aux.shape == (0, 6)
        assert frame.stats == []
        assert frame.n_points_in == 1200
        assert frame.n_points_clean == 0
        assert timings.n_clusters == 0
        assert timings.total >= 0

    def test_empty_input_frame(self):
        pipeline = make_pipeline()
        frame, _ = pipeline.process(np.zeros((0, 3), dtype=np.float32))
        assert frame.n_clusters == 0
        assert frame.batch.shape == (0, 1024, 3)

    def test_single_point_frame(self):
        pipeline = make_pipeline()
        frame, _ = pipeline.process(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
        assert frame.n_clusters == 0  # below min_cluster_points; declined, not guessed

    def test_non_finite_rows_are_dropped(self):
        """Finding A6: skip_nans guards the ROS 2 callback only. One NaN poisons
        a centroid and from there the whole normalised cloud, silently."""
        pipeline = make_pipeline()
        scene = multi_object_scene()
        poisoned = np.vstack([
            scene,
            np.array([[np.nan, 0.0, 0.0], [0.0, np.inf, 0.0], [0.0, 0.0, -np.inf]],
                     dtype=np.float32),
        ])
        frame, _ = pipeline.process(poisoned)
        assert frame.n_points_in == len(scene) + 3
        assert frame.n_points_clean == len(scene)
        assert np.isfinite(frame.batch).all()
        assert np.isfinite(frame.aux).all()

    def test_all_nan_frame(self):
        pipeline = make_pipeline()
        frame, _ = pipeline.process(np.full((100, 3), np.nan, dtype=np.float32))
        assert frame.n_clusters == 0
        assert frame.n_points_clean == 0

    def test_extra_columns_are_dropped(self):
        pipeline = make_pipeline()
        scene = multi_object_scene()
        with_intensity = np.hstack([scene, np.ones((len(scene), 1), dtype=np.float32)])
        frame, _ = pipeline.process(with_intensity)
        assert frame.batch.shape[2] == 3

    def test_wrong_rank_still_raises(self):
        """A caller handing in a (N,) or (N,2) array has a bug in the caller.
        That is a programming error, not a degenerate frame, so it is loud."""
        pipeline = make_pipeline()
        with pytest.raises(ValueError, match=r"N,>=3"):
            pipeline.process(np.zeros((10, 2), dtype=np.float32))


# ---------------------------------------------------------------------------
# Finding I6: config validation at startup, not KeyError on frame 3000.
# ---------------------------------------------------------------------------
class TestConfigValidation:
    def test_defaults_are_filled(self):
        cfg = validate_config({})
        assert cfg["downsampling"]["target_points"] == 1024
        assert cfg["clustering"]["max_clusters_per_frame"] == 16
        assert cfg["seed"] == 0

    def test_unknown_section_is_rejected(self):
        with pytest.raises(ConfigError, match="unknown config section"):
            validate_config({"downsamplingg": {}})

    def test_unknown_key_is_rejected(self):
        """A silently ignored config key is how a tuning session becomes
        untraceable: the operator thinks they changed something."""
        with pytest.raises(ConfigError, match="unknown key"):
            validate_config({"downsampling": {"target_poinst": 512}})

    def test_bad_enum_value_is_rejected(self):
        with pytest.raises(ConfigError, match="roi_filter.mode"):
            validate_config({"roi_filter": {"mode": "wedge"}})

    def test_non_positive_target_points_is_rejected(self):
        with pytest.raises(ConfigError, match="target_points"):
            validate_config({"downsampling": {"target_points": 0}})

    def test_wrong_type_is_rejected(self):
        with pytest.raises(ConfigError, match="clustering.enabled"):
            validate_config({"clustering": {"enabled": "yes"}})

    def test_negative_eps_is_rejected(self):
        with pytest.raises(ConfigError, match="clustering.eps_m"):
            validate_config({"clustering": {"eps_m": -1.0}})

    def test_ground_removal_keys_are_validated(self):
        with pytest.raises(ConfigError, match="ground_removal.method"):
            validate_config({"ground_removal": {"method": "plane_fit_vibes"}})
        with pytest.raises(ConfigError, match="distance_threshold_m"):
            validate_config({"ground_removal": {"distance_threshold_m": 0.0}})
        with pytest.raises(ConfigError, match="num_iterations"):
            validate_config({"ground_removal": {"num_iterations": 0}})
        with pytest.raises(ConfigError, match="min_height_m"):
            validate_config({"ground_removal": {"min_height_m": -0.1}})

    def test_min_inlier_fraction_must_be_a_fraction(self):
        """It is compared against a ratio, so > 1 can never be satisfied and
        would silently mean 'never remove the ground'."""
        with pytest.raises(ConfigError, match="min_inlier_fraction must be in"):
            validate_config({"ground_removal": {"min_inlier_fraction": 1.5}})
        assert validate_config({"ground_removal": {"min_inlier_fraction": 1.0}})

    def test_ground_removal_defaults_mirror_preprocess_yaml(self):
        cfg = validate_config({})["ground_removal"]
        assert cfg["enabled"] is True
        assert cfg["method"] == "ransac_plane"
        assert cfg["distance_threshold_m"] == pytest.approx(0.15)
        assert cfg["min_height_m"] == pytest.approx(0.15)

    def test_inverted_roi_box_is_rejected(self):
        with pytest.raises(ConfigError, match="is greater than"):
            validate_config({"roi_filter": {"x_min": 5.0, "x_max": -5.0}})

    def test_zero_voxel_size_is_rejected_but_null_is_allowed(self):
        with pytest.raises(ConfigError, match="voxel_size_m"):
            validate_config({"downsampling": {"voxel_size_m": 0.0}})
        assert validate_config({"downsampling": {"voxel_size_m": None}})["downsampling"][
            "voxel_size_m"
        ] is None

    def test_removed_values_name_their_replacement(self):
        with pytest.raises(ConfigError, match="no longer supported"):
            validate_config({"output": {"format": "torch"}})
        with pytest.raises(ConfigError, match="has been removed"):
            validate_config({"downsampling": {"method": "random"}})

    def test_foreign_sections_pass_through(self):
        """`ingestion` and `model` belong to other components; the pipeline must
        neither validate nor discard them."""
        cfg = validate_config({"ingestion": {"mode": "ros2"}, "model": {"enabled": True}})
        assert cfg["ingestion"]["mode"] == "ros2"
        assert cfg["model"]["enabled"] is True

    def test_shipped_config_file_validates(self):
        import pathlib

        import yaml

        path = pathlib.Path(__file__).resolve().parents[1] / "config" / "pipeline_config.yaml"
        cfg = validate_config(yaml.safe_load(path.read_text()))
        # The two caps must agree or clusters are dropped silently; the
        # classifier enforces the same equality from its own side.
        assert cfg["clustering"]["max_clusters_per_frame"] == 16


# ---------------------------------------------------------------------------
# Finding A4: StageTimings had five fixed float fields and no cluster count.
# ---------------------------------------------------------------------------
class TestStageTimings:
    def test_named_stages_exist(self):
        t = StageTimings()
        for stage in ("clustering", "cluster_prep", "inference", "postprocess"):
            t[stage] = 0.001
            assert t[stage] == pytest.approx(0.001)

    def test_unknown_stage_goes_to_extra_rather_than_being_dropped(self):
        t = StageTimings()
        t["frame_accumulation"] = 0.002  # a stage this pipeline does not have yet
        assert t["frame_accumulation"] == pytest.approx(0.002)
        assert "frame_accumulation" in t.extra

    def test_as_ms_converts_and_omits_the_count(self):
        t = StageTimings()
        t["clustering"] = 0.0025
        t["publish"] = 0.001
        t.n_clusters = 4
        ms = t.as_ms()
        assert ms["clustering"] == pytest.approx(2.5)
        assert ms["publish"] == pytest.approx(1.0)
        assert "n_clusters" not in ms

    def test_cluster_count_is_recorded_on_the_frame(self):
        pipeline = make_pipeline()
        _, timings = pipeline.process(multi_object_scene())
        assert timings.n_clusters == 3

    def test_log_line_carries_m(self, caplog):
        pipeline = make_pipeline({"logging": {"log_stage_timings": True, "log_every_n_frames": 1}})
        with caplog.at_level("INFO", logger="livox_wrapper.pipeline"):
            pipeline.process(multi_object_scene())
        assert any("M=3" in r.getMessage() for r in caplog.records)


class TestClusterCap:
    def test_max_clusters_per_frame_is_enforced_and_reported(self):
        """The cap bounds worst-case latency; dropping objects must be visible
        in n_clusters_found rather than inferred from their absence."""
        pipeline = make_pipeline({
            "clustering": {"max_clusters_per_frame": 2, "eps_m": 0.5, "min_cluster_points": 50}
        })
        frame, _ = pipeline.process(multi_object_scene())
        assert frame.n_clusters == 2
        assert frame.n_clusters_found == 3

    def test_largest_clusters_are_kept(self):
        pipeline = make_pipeline({"clustering": {"max_clusters_per_frame": 1}})
        scene = np.vstack([
            blob([0.0, 0.0, 0.0], n=600, scale=0.1, seed=1),
            blob([5.0, 0.0, 0.0], n=100, scale=0.1, seed=2),
        ]).astype(np.float32)
        frame, _ = pipeline.process(scene)
        assert frame.n_clusters == 1
        assert frame.clusters[0].n_points_raw == 600
