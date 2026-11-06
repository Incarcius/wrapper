"""Crop extraction: the recording -> cluster-crop step, and its reproducibility.

``TestReproducibility`` is the class that backs the claim in ``crops.py``'s
docstring: a crop set is reproducible from (raw recording, preprocess.yaml, seed)
and nothing else. It is asserted both ways — same inputs give byte-identical
crops, and a different seed is recorded rather than hidden.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn", reason="needs a DBSCAN backend")

from pcml.data.config import load_preprocess_config  # noqa: E402
from pcml.data.crops import (  # noqa: E402
    CropError,
    crop_geometry_summary,
    extract_many,
    extract_session_crops,
    find_crop_indices,
    load_crop_index,
)
from pcml.data.io import open_recording  # noqa: E402
from synthetic_recordings import (  # noqa: E402
    OBJECT_POSITIONS,
    synthetic_session_frames,
    write_meta_sidecar,
    write_npz_session,
    write_pointcloud2_bag,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PREPROCESS = REPO_ROOT / "configs" / "preprocess.yaml"


@pytest.fixture
def config():
    return load_preprocess_config(PREPROCESS)


@pytest.fixture
def recording(tmp_path):
    """A 3-frame hover over the simulated field, as a packed .npz with metadata."""
    frames = synthetic_session_frames(n_frames=3, extent_m=8.0, density_per_m2=25.0)
    p = write_npz_session(tmp_path / "session_01_hover_12m.npz", frames)
    write_meta_sidecar(p, session_id="session_01_hover_12m")
    return p


class TestExtraction:
    def test_finds_the_three_simulated_objects_in_every_frame(
        self, tmp_path, recording, config
    ):
        index = extract_session_crops(recording, tmp_path / "processed", config=config)

        assert index.session_id == "session_01_hover_12m"
        assert index.stats["n_frames_processed"] == 3
        assert index.stats["n_frames_skipped_no_ground_plane"] == 0
        assert len(index.frames()) == 3

        for frame_index, crops in index.frames().items():
            # Exactly three, in every frame. This was briefly relaxed to ">= 3"
            # when the count came out as 4/3/2 across the three frames, but the
            # cause was the fixture, not the pipeline: the simulated tree's
            # points were sparse enough (0.416 m mean spacing vs the 0.5 m
            # DBSCAN radius) that its cluster count was seed-dependent. Fixed in
            # test_ground.field_scene and pinned by TestFixtureItself there, so
            # an exact count is both correct and the stronger assertion.
            assert len(crops) == 3, f"frame {frame_index} gave {len(crops)} crops"
            found = [(c.centroid_m[0], c.centroid_m[1]) for c in crops]
            for ex, ey in OBJECT_POSITIONS:
                assert any(
                    abs(fx - ex) < 1.0 and abs(fy - ey) < 1.0 for fx, fy in found
                ), f"no crop near the object at ({ex}, {ey}); got {found}"

    def test_crops_are_raw_metric_points_not_normalised(self, tmp_path, recording, config):
        """The stored points must still be in sensor-frame metres.

        Storing the post-FPS unit-sphere tensor would bake in one choice of
        num_points and discard the aux features and metre-domain augmentation.
        """
        index = extract_session_crops(recording, tmp_path / "processed", config=config)
        crop = max(index.crops, key=lambda c: c.extent_m[2])  # the 6 m pole
        points = index.points_for(crop)

        assert points.dtype == np.float32
        assert points.shape == (crop.n_points, 3)
        assert points.shape[0] != config.sampling.num_points or crop.n_points == 1024
        # Metres, not a unit sphere.
        assert points[:, 2].max() > 2.0
        radius = np.linalg.norm(points - points.mean(axis=0), axis=1).max()
        assert radius > 1.0

    def test_variable_length_crops_round_trip(self, tmp_path, recording, config):
        index = extract_session_crops(recording, tmp_path / "processed", config=config)
        reloaded = load_crop_index(index.index_path)
        assert len(reloaded.crops) == len(index.crops)
        for a, b in zip(index.crops, reloaded.crops):
            assert a == b
            assert np.array_equal(index.points_for(a), reloaded.points_for(b))

    def test_stored_points_are_read_only(self, tmp_path, recording, config):
        """A shared buffer that can be mutated in place would corrupt later items."""
        index = extract_session_crops(recording, tmp_path / "processed", config=config)
        points = index.points_for(index.crops[0])
        with pytest.raises(ValueError):
            points[0, 0] = 99.0

    def test_crop_ids_are_stable_sortable_and_unique(self, tmp_path, recording, config):
        index = extract_session_crops(recording, tmp_path / "processed", config=config)
        ids = [c.crop_id for c in index.crops]
        assert len(set(ids)) == len(ids)
        assert all(c.crop_id.startswith("session_01_hover_12m/") for c in index.crops)
        assert ids == sorted(ids)

    def test_geometry_matches_the_points(self, tmp_path, recording, config):
        index = extract_session_crops(recording, tmp_path / "processed", config=config)
        for crop in index.crops:
            points = np.asarray(index.points_for(crop))
            assert np.allclose(crop.centroid_m, points.mean(axis=0), atol=1e-4)
            assert np.allclose(
                crop.extent_m, points.max(axis=0) - points.min(axis=0), atol=1e-4
            )
            assert crop.n_points == points.shape[0]
            assert crop.n_points >= config.clustering.min_cluster_points

    def test_ground_plane_height_is_recorded_per_crop(self, tmp_path, recording, config):
        """ground_z_m is what makes height_above_ground a measurement, not a guess."""
        index = extract_session_crops(recording, tmp_path / "processed", config=config)
        assert all(c.ground_z_m is not None for c in index.crops)
        assert abs(index.crops[0].ground_z_m) < 0.2  # simulated field is near z=0
        assert index.stats["ground_plane_tilt_deg_max"] < 2.0

    def test_stamps_come_from_the_recording(self, tmp_path, recording, config):
        frames = synthetic_session_frames(n_frames=3, extent_m=8.0, density_per_m2=25.0)
        index = extract_session_crops(recording, tmp_path / "processed", config=config)
        assert index.stamp_source == "sensor"
        by_frame = index.frames()
        for frame_index, crops in by_frame.items():
            want = frames[frame_index][1]
            for crop in crops:
                assert crop.stamp == pytest.approx(want, abs=1e-3)

    def test_session_metadata_is_carried_into_the_index(self, tmp_path, recording, config):
        """§4.2 data cannot be reconstructed later, so it travels with the crops."""
        index = extract_session_crops(recording, tmp_path / "processed", config=config)
        assert index.recording_meta is not None
        assert index.recording_meta["altitude_m"] == 12.0
        assert index.recording_meta["mount_angle_deg"] == 35.0
        assert index.recording_meta["motion"] == "hover"

    def test_frame_stride_keeps_source_frame_indices(self, tmp_path, config):
        frames = synthetic_session_frames(n_frames=4, extent_m=8.0, density_per_m2=25.0)
        p = write_npz_session(tmp_path / "s.npz", frames)
        index = extract_session_crops(p, tmp_path / "out", config=config, frame_stride=2)
        assert sorted(index.frames()) == [0, 2]
        assert index.stats["frame_stride"] == 2

    def test_max_frames_stops_early(self, tmp_path, config):
        frames = synthetic_session_frames(n_frames=5, extent_m=8.0, density_per_m2=25.0)
        p = write_npz_session(tmp_path / "s.npz", frames)
        index = extract_session_crops(p, tmp_path / "out", config=config, max_frames=2)
        assert index.stats["n_frames_read"] == 2
        assert len(index.frames()) == 2

    def test_works_from_a_ros2_bag(self, tmp_path, config):
        """The whole path, from a real CDR bag rather than a NumPy convenience."""
        frames = synthetic_session_frames(n_frames=2, extent_m=8.0, density_per_m2=25.0)
        bag_dir = tmp_path / "session_02_hover_15m"
        write_pointcloud2_bag(bag_dir / "session_02_hover_15m_0.db3", frames)
        index = extract_session_crops(bag_dir, tmp_path / "out", config=config)
        assert index.session_id == "session_02_hover_15m"
        assert len(index.frames()) == 2
        assert all(len(c) >= 3 for c in index.frames().values())
        assert index.stamp_source == "sensor"

    def test_empty_frames_are_counted_not_crashed_on(self, tmp_path, config):
        p = tmp_path / "e.npz"
        np.savez(
            p,
            points=np.zeros((0, 3), dtype=np.float32),
            offsets=np.array([0, 0], dtype=np.int64),
            stamps=np.array([1.0]),
        )
        index = extract_session_crops(p, tmp_path / "out", config=config)
        assert index.stats["n_frames_empty"] == 1
        assert len(index.crops) == 0

    def test_frames_with_no_ground_plane_are_skipped_and_counted(self, tmp_path, config):
        """A rejected ground fit must not become a field-sized 'object' (§3)."""
        blob = np.random.default_rng(3).uniform(-5, 5, (2000, 3)).astype(np.float32)
        p = tmp_path / "blob.npz"
        np.savez(
            p,
            points=blob,
            offsets=np.array([0, blob.shape[0]], dtype=np.int64),
            stamps=np.array([1.0]),
        )
        strict = replace(
            config,
            ground_removal=replace(config.ground_removal, min_inlier_fraction=0.95),
        )
        index = extract_session_crops(p, tmp_path / "out", config=strict)
        assert index.stats["n_frames_skipped_no_ground_plane"] == 1
        assert len(index.crops) == 0


class TestRefusals:
    def test_clustering_disabled_is_refused(self, tmp_path, recording, config):
        broken = replace(config, clustering=replace(config.clustering, enabled=False))
        with pytest.raises(CropError, match="Whole-scene mode"):
            extract_session_crops(recording, tmp_path / "out", config=broken)

    def test_ground_removal_disabled_is_refused(self, tmp_path, recording, config):
        """§3 made this a prerequisite, not an ablation."""
        broken = replace(
            config, ground_removal=replace(config.ground_removal, enabled=False)
        )
        with pytest.raises(CropError, match="field-sized cluster"):
            extract_session_crops(recording, tmp_path / "out", config=broken)

    def test_duplicate_session_ids_are_refused(self, tmp_path, config):
        """A collision would leak near-duplicate frames across splits."""
        frames = synthetic_session_frames(n_frames=1, extent_m=6.0, density_per_m2=20.0)
        a = write_npz_session(tmp_path / "a" / "same_name.npz", frames)
        b = write_npz_session(tmp_path / "b" / "same_name.npz", frames)
        with pytest.raises(CropError, match="session_id"):
            extract_many([a, b], tmp_path / "out", config=config)


class TestReproducibility:
    def test_identical_inputs_give_identical_crops(self, tmp_path, recording, config):
        a = extract_session_crops(recording, tmp_path / "a", config=config, seed=0)
        b = extract_session_crops(recording, tmp_path / "b", config=config, seed=0)
        assert [c.crop_id for c in a.crops] == [c.crop_id for c in b.crops]
        for ca, cb in zip(a.crops, b.crops):
            assert ca == cb
            assert np.array_equal(a.points_for(ca), b.points_for(cb))

    def test_frame_results_do_not_depend_on_preceding_frames(self, tmp_path, config):
        """Per-frame RNG is derived from (seed, frame_index), not advanced.

        So extracting a subset gives the same crops for the frames it covers,
        which is what makes a partial re-extraction comparable to a full one.
        """
        frames = synthetic_session_frames(n_frames=4, extent_m=8.0, density_per_m2=25.0)
        p = write_npz_session(tmp_path / "s.npz", frames)
        full = extract_session_crops(p, tmp_path / "full", config=config, seed=0)
        strided = extract_session_crops(
            p, tmp_path / "strided", config=config, seed=0, frame_stride=2
        )
        full_by_id = full.by_id
        assert strided.crops, "stride produced nothing"
        for crop in strided.crops:
            assert crop.crop_id in full_by_id
            other = full_by_id[crop.crop_id]
            # `offset` is a position in the packed points array and legitimately
            # differs when fewer frames were stored; everything describing the
            # cluster itself must be identical.
            assert replace(crop, offset=0) == replace(other, offset=0)
            assert np.array_equal(strided.points_for(crop), full.points_for(other))

    def test_the_index_records_everything_needed_to_reproduce(
        self, tmp_path, recording, config
    ):
        index = extract_session_crops(recording, tmp_path / "out", config=config, seed=7)
        assert index.seed == 7
        assert index.preprocess_sha256 == config.sha256
        assert index.raw_sha256 and len(index.raw_sha256) == 64
        assert index.source_file.endswith("session_01_hover_12m.npz")
        assert index.created

    def test_a_different_seed_is_recorded_not_hidden(self, tmp_path, recording, config):
        a = extract_session_crops(recording, tmp_path / "a", config=config, seed=0)
        b = extract_session_crops(recording, tmp_path / "b", config=config, seed=99)
        assert a.seed == 0 and b.seed == 99
        # Same config file, so the same hash: the seed is the other half of the
        # reproducibility pair and is stored separately on purpose.
        assert a.preprocess_sha256 == b.preprocess_sha256

    def test_index_version_mismatch_is_refused(self, tmp_path, recording, config):
        import json

        index = extract_session_crops(recording, tmp_path / "out", config=config)
        doc = json.loads(index.index_path.read_text(encoding="utf-8"))
        doc["version"] = 999
        index.index_path.write_text(json.dumps(doc), encoding="utf-8")
        with pytest.raises(CropError, match="version 999"):
            load_crop_index(index.index_path)

    def test_missing_points_file_says_how_to_regenerate(self, tmp_path, recording, config):
        index = extract_session_crops(recording, tmp_path / "out", config=config)
        index.points_path.unlink()
        reloaded = load_crop_index(index.index_path)
        with pytest.raises(CropError, match="re-run"):
            reloaded.points_for(reloaded.crops[0])


class TestDiscoveryAndSummary:
    def test_find_crop_indices(self, tmp_path, config):
        frames = synthetic_session_frames(n_frames=1, extent_m=6.0, density_per_m2=20.0)
        out = tmp_path / "processed"
        for name in ("session_01", "session_02"):
            p = write_npz_session(tmp_path / f"{name}.npz", frames)
            extract_session_crops(p, out, config=config)
        found = find_crop_indices(out)
        assert len(found) == 2
        assert {load_crop_index(f).session_id for f in found} == {
            "session_01",
            "session_02",
        }

    def test_geometry_summary_reports_percentiles_for_the_caps(
        self, tmp_path, recording, config
    ):
        """This is the measurement that re-derives P6's provisional caps."""
        index = extract_session_crops(recording, tmp_path / "out", config=config)
        summary = crop_geometry_summary(index.crops)
        assert summary["n_crops"] == 9
        assert summary["n_frames_with_crops"] == 3
        # 3 objects x 3 frames with no frame-to-frame variation, so mean and max
        # both pin to 3 (the summary reports percentiles, not a min, for this key).
        assert summary["clusters_per_frame"]["max"] == 3.0
        assert summary["clusters_per_frame"]["mean"] == 3.0
        assert summary["points_per_cluster"]["min"] >= 50
        assert summary["largest_extent_m"]["max"] > 1.0
        assert "p95" in summary["range_to_centroid_m"]

    def test_geometry_summary_on_nothing(self):
        assert crop_geometry_summary([])["n_crops"] == 0
