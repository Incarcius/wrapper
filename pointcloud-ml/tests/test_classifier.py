"""Unit tests for the deployed classifier and its label map.

Covers the four invariants the wrapper is allowed to rely on
(``docs/wrapper_audit.md`` §5) and the config validation that makes a startup
failure loud. Nothing here tests accuracy: there is no checkpoint and no class
list (``docs/task_definition.md`` P1/P2), so every class assertion is
structural.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from pcml.data.clustering import Cluster
from pcml.data.label_mapping import LabelMapError
from pcml.deploy.adapter import to_model_input
from pcml.deploy.classifier import ClusterClassifier, placeholder_label_map

PREPROCESS = {
    "seed": 0,
    "clustering": {
        "enabled": True,
        "eps_m": 0.5,
        "min_samples": 10,
        "min_cluster_points": 50,
        "max_clusters_per_frame": 4,
        "use_open3d": False,
    },
    "sampling": {"num_points": 128, "voxel_size_m": 0.05, "use_open3d": False},
    "normalization": {"center": True, "scale_to_unit_sphere": True},
    "features": {"use_aux": False},
}

MODEL = {
    "name": "pointnet",
    "in_channels": 3,
    "widths": [16, 16, 32],
    "head_widths": [32],
    "dropout": 0.0,
    "aux_dim": 0,
}


def write_configs(tmp_path, *, deploy_overrides=None, preprocess_overrides=None, model=None):
    """Write a preprocess + deploy config pair and return the deploy path."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    pre = {k: (dict(v) if isinstance(v, dict) else v) for k, v in PREPROCESS.items()}
    for section, vals in (preprocess_overrides or {}).items():
        if isinstance(vals, dict):
            pre[section] = {**pre.get(section, {}), **vals}
        else:
            pre[section] = vals
    (tmp_path / "preprocess.yaml").write_text(yaml.safe_dump(pre))

    deploy = {
        "checkpoint": None,
        "label_map": None,
        "placeholder_num_classes": 3,
        "preprocess": "preprocess.yaml",
        "model": dict(model or MODEL),
        "deployment": {
            "input_layout": "bcn",
            "device": "cpu",
            "warmup_frames": 1,
            "max_clusters": pre["clustering"]["max_clusters_per_frame"],
            "reject_threshold": None,
        },
    }
    for key, value in (deploy_overrides or {}).items():
        if isinstance(value, dict) and isinstance(deploy.get(key), dict):
            deploy[key] = {**deploy[key], **value}
        else:
            deploy[key] = value
    path = tmp_path / "deploy.yaml"
    path.write_text(yaml.safe_dump(deploy))
    return path


def blob(center, n=200, scale=0.1, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(loc=center, scale=scale, size=(n, 3)).astype(np.float32)


def scene():
    return np.vstack([blob([0, 0, 0], seed=1), blob([4, 0, 0], seed=2)]).astype(np.float32)


@pytest.fixture
def classifier(tmp_path):
    return ClusterClassifier(write_configs(tmp_path))


def _class_entry(class_id: int, name: str) -> dict:
    """A label-map class entry that satisfies the loader's validation.

    ``label_mapping`` requires a non-empty ``definition`` and
    ``annotation_policy`` on every class — a class nobody wrote down is a class
    two annotators label differently. Fixtures here are not exercising that
    rule, so they just satisfy it rather than restating it; the rule itself is
    tested in ``tests/test_label_mapping.py``.
    """
    return {
        "class_id": class_id,
        "name": name,
        "definition": f"Test fixture class {name}.",
        "annotation_policy": f"Label {name} when the fixture says so.",
    }


class TestPlaceholderLabelMap:
    """The stand-in vocabulary. Loading and validating *real* label maps is
    ``pcml.data.label_mapping``'s job and belongs with that module's own tests;
    what is checked here is the placeholder, and that the classifier refuses to
    start on a map that module rejects."""

    def test_names_are_self_identifying(self):
        lm = placeholder_label_map(3)
        assert lm.num_classes == 3
        assert all(name.startswith("PLACEHOLDER_") for name in lm.names)

    def test_it_is_flagged_as_an_example(self):
        """`is_example` is the flag the rest of the repository already checks
        before letting a number be reported."""
        assert placeholder_label_map(3).is_example is True

    def test_it_needs_at_least_two_classes(self):
        with pytest.raises(ValueError, match="placeholder_num_classes"):
            placeholder_label_map(1)

    def test_reject_id_resolves_to_the_reject_name(self):
        lm = placeholder_label_map(3)
        assert lm.name_for(-1) == "unknown"
        assert lm.name_for(0) == "PLACEHOLDER_0"

    def test_a_real_map_is_loaded_and_is_not_a_placeholder(self, tmp_path):
        lm_path = tmp_path / "label_map.yaml"
        lm_path.write_text(
            yaml.safe_dump(
                {
                    "version": 2,
                    "reject": {"confidence_threshold": 0.4},
                    "classes": [
                        _class_entry(0, "alpha"),
                        _class_entry(1, "beta"),
                        _class_entry(2, "gamma"),
                    ],
                }
            )
        )
        clf = ClusterClassifier(
            write_configs(tmp_path, deploy_overrides={"label_map": "label_map.yaml"})
        )
        assert clf.label_map.names == ("alpha", "beta", "gamma")
        assert clf.label_map.is_example is False
        # The threshold comes from the label map when the deploy config leaves
        # reject_threshold null.
        assert clf.reject_threshold == pytest.approx(0.4)
        # Still a placeholder overall: there is no checkpoint.
        assert clf.is_placeholder is True

    def test_k_disagreeing_with_the_map_is_impossible(self, tmp_path):
        """K comes from the label map, so a model config cannot override it."""
        lm_path = tmp_path / "label_map.yaml"
        lm_path.write_text(
            yaml.safe_dump(
                {"classes": [_class_entry(i, f"c{i}") for i in range(7)]}
            )
        )
        clf = ClusterClassifier(
            write_configs(tmp_path, deploy_overrides={"label_map": "label_map.yaml"})
        )
        assert clf.model.num_classes == 7

    def test_an_invalid_map_fails_at_startup(self, tmp_path):
        """The shipped template has `classes: []` on purpose. Loading it as a
        real map would give K=0 and a head that cannot be built, so startup must
        fail rather than the first frame."""
        lm_path = tmp_path / "label_map.yaml"
        lm_path.write_text(yaml.safe_dump({"classes": []}))
        with pytest.raises(LabelMapError):
            ClusterClassifier(
                write_configs(tmp_path, deploy_overrides={"label_map": "label_map.yaml"})
            )

    def test_a_missing_map_fails_at_startup(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ClusterClassifier(
                write_configs(tmp_path, deploy_overrides={"label_map": "nope.yaml"})
            )


class TestConstruction:
    def test_k_comes_from_the_label_map(self, classifier):
        assert classifier.label_map.num_classes == 3
        assert classifier.model.num_classes == 3

    def test_placeholder_artifacts_are_flagged(self, classifier):
        assert classifier.is_placeholder is True
        assert classifier.checkpoint_path is None
        assert classifier.label_map.is_example is True

    def test_model_is_in_eval_mode(self, classifier):
        """BatchNorm must use running statistics, and in train() mode a
        single-cluster frame would hit the batch-of-1 variance error."""
        assert classifier.model.training is False

    def test_batch_is_preallocated_at_the_configured_cap(self, classifier):
        assert classifier._batch_buf.shape == (4, 3, 128)

    def test_bnc_layout_changes_the_buffer_shape(self, tmp_path):
        clf = ClusterClassifier(
            write_configs(tmp_path, deploy_overrides={"deployment": {"input_layout": "bnc"}})
        )
        assert clf._batch_buf.shape == (4, 128, 3)

    def test_unknown_layout_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="input_layout"):
            ClusterClassifier(
                write_configs(tmp_path, deploy_overrides={"deployment": {"input_layout": "nhwc"}})
            )

    def test_cap_mismatch_is_rejected(self, tmp_path):
        """The preallocated batch is sized from the cap. A mismatch between the
        two config files would silently drop clusters."""
        with pytest.raises(ValueError, match="max_clusters"):
            ClusterClassifier(
                write_configs(tmp_path, deploy_overrides={"deployment": {"max_clusters": 8}})
            )

    def test_missing_preprocess_is_rejected(self, tmp_path):
        path = write_configs(tmp_path)
        doc = yaml.safe_load(path.read_text())
        del doc["preprocess"]
        path.write_text(yaml.safe_dump(doc))
        with pytest.raises(KeyError, match="preprocess"):
            ClusterClassifier(path)

    def test_aux_mismatch_is_rejected_both_ways(self, tmp_path):
        with pytest.raises(ValueError, match="aux_dim: 0"):
            ClusterClassifier(
                write_configs(tmp_path, preprocess_overrides={"features": {"use_aux": True}})
            )
        with pytest.raises(ValueError, match="use_aux is false"):
            ClusterClassifier(write_configs(tmp_path, model={**MODEL, "aux_dim": 6}))

    def test_out_of_range_reject_threshold_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="reject_threshold"):
            ClusterClassifier(
                write_configs(tmp_path, deploy_overrides={"deployment": {"reject_threshold": 1.5}})
            )


def field_scene(ground_z=-1.5, n_ground=4000, n_per_object=250, extent=6.0):
    """A ground sheet dense enough to bridge clusters, plus columns on it.

    Mirrors ``wrapper/tests/test_pipeline.py::field_scene`` deliberately: the two
    entry points must agree about this scene, and that is only checkable if they
    are given the same one.
    """
    rng = np.random.default_rng(11)
    ground = np.column_stack(
        [
            rng.uniform(-extent, extent, size=n_ground),
            rng.uniform(-extent, extent, size=n_ground),
            ground_z + rng.normal(0.0, 0.02, size=n_ground),
        ]
    )
    objects = [
        np.column_stack(
            [
                cx + rng.normal(0.0, 0.12, size=n_per_object),
                cy + rng.normal(0.0, 0.12, size=n_per_object),
                ground_z + rng.uniform(0.05, 2.0, size=n_per_object),
            ]
        )
        for cx, cy in [(-4, -4), (4, -4), (-4, 4)]
    ]
    return np.vstack([ground] + objects).astype(np.float32)


GROUND_ON = {
    "ground_removal": {
        "enabled": True,
        "method": "ransac_plane",
        "distance_threshold_m": 0.15,
        "num_iterations": 200,
        "max_tilt_deg": 30.0,
        "min_inlier_fraction": 0.15,
        "min_height_m": 0.15,
    }
}


class TestGroundRemoval:
    """``classify_frame`` performs the stage, so it agrees with the wrapper's
    pipeline. If one removed the ground and the other did not, the same frame
    would cluster differently depending on which entry point was called."""

    def test_it_separates_objects_that_the_ground_would_merge(self, tmp_path):
        on = ClusterClassifier(write_configs(tmp_path / "on", preprocess_overrides=GROUND_ON))
        off = ClusterClassifier(write_configs(tmp_path / "off"))
        scene_with_ground = field_scene()

        assert on.classify_frame(scene_with_ground, stamp=0.0).n_clusters_classified == 3
        # Without the stage the sheet connects everything into one component.
        assert off.classify_frame(scene_with_ground, stamp=0.0).n_clusters_classified == 1

    def test_the_stage_is_timed_separately(self, tmp_path):
        clf = ClusterClassifier(write_configs(tmp_path, preprocess_overrides=GROUND_ON))
        pred = clf.classify_frame(field_scene(), stamp=0.0)
        assert pred.latencies_ms["ground_removal"] > 0.0

    def test_it_is_not_timed_away_when_disabled(self, tmp_path):
        clf = ClusterClassifier(write_configs(tmp_path))
        pred = clf.classify_frame(scene(), stamp=0.0)
        # The key is still present, so a latency record has the same shape either
        # way and a consumer does not have to guess whether the stage exists.
        assert pred.latencies_ms["ground_removal"] >= 0.0

    def test_the_fitted_plane_feeds_the_height_feature(self, tmp_path):
        ground_z = -1.5
        clf = ClusterClassifier(
            write_configs(
                tmp_path,
                preprocess_overrides={**GROUND_ON, "features": {"use_aux": True}},
                model={**MODEL, "aux_dim": 6},
            )
        )
        pred = clf.classify_frame(field_scene(ground_z=ground_z), stamp=0.0)
        assert pred.n_clusters_classified == 3
        # Centroids come back in metres, so the clearance above the fitted
        # surface is checkable from the output alone.
        for p in pred.predictions:
            assert p.centroid_m[2] - ground_z == pytest.approx(1.0, abs=0.4)

    def test_an_unimplemented_method_is_refused_at_startup(self, tmp_path):
        with pytest.raises(ValueError, match="is not implemented"):
            ClusterClassifier(
                write_configs(
                    tmp_path,
                    preprocess_overrides={
                        "ground_removal": {"enabled": True, "method": "patchwork"}
                    },
                )
            )

    def test_an_unimplemented_method_is_ignored_when_disabled(self, tmp_path):
        """A parameter for a stage that is switched off is not an error."""
        clf = ClusterClassifier(
            write_configs(
                tmp_path,
                preprocess_overrides={
                    "ground_removal": {"enabled": False, "method": "patchwork"}
                },
            )
        )
        assert clf.ground_removal_enabled is False

    def test_a_rejected_fit_classifies_rather_than_crashing(self, tmp_path):
        """No acceptable plane means the frame passes through unchanged. It will
        cluster badly, which is visible in the counts, but it must not fail."""
        clf = ClusterClassifier(
            write_configs(
                tmp_path,
                preprocess_overrides={
                    "ground_removal": {**GROUND_ON["ground_removal"], "min_inlier_fraction": 1.0}
                },
            )
        )
        pred = clf.classify_frame(field_scene(), stamp=0.0)
        assert pred.diagnostic == "" or "warmup" in pred.diagnostic
        assert pred.n_clusters_classified == 1

    def test_classify_prepared_does_not_remove_ground(self, tmp_path):
        """The wrapper has already done it; a second RANSAC fit on individual
        clusters would fit planes to the objects themselves."""
        clf = ClusterClassifier(write_configs(tmp_path, preprocess_overrides=GROUND_ON))
        clusters = [Cluster.from_points(blob([0, 0, 0], n=200))]
        batch = np.zeros((1, 128, 3), dtype=np.float32)
        pred = clf.classify_prepared(clusters, batch, stamp=0.0)
        assert pred.n_clusters_classified == 1
        assert "ground_removal" not in pred.latencies_ms


class TestRegionProposalSection:
    """``region_proposal`` superseded ``clustering`` in preprocess.yaml. The
    classifier must read whichever the file actually uses: defaulting while the
    config says otherwise is a silent divergence, and the defaults happen to
    match the current values, so it would not show up as a failure."""

    def test_region_proposal_values_are_honoured(self, tmp_path):
        pre = {k: (dict(v) if isinstance(v, dict) else v) for k, v in PREPROCESS.items()}
        del pre["clustering"]
        pre["region_proposal"] = {
            "method": "dbscan_instances",
            "eps_m": 0.5,
            "min_samples": 10,
            "min_region_points": 500,  # above the 200-point blobs this scene has
            "max_regions": 4,
            "use_open3d": False,
        }
        (tmp_path / "preprocess.yaml").write_text(yaml.safe_dump(pre))
        (tmp_path / "deploy.yaml").write_text(
            yaml.safe_dump(
                {
                    "checkpoint": None,
                    "label_map": None,
                    "placeholder_num_classes": 3,
                    "preprocess": "preprocess.yaml",
                    "model": dict(MODEL),
                    "deployment": {"device": "cpu", "max_clusters": 4},
                }
            )
        )
        clf = ClusterClassifier(tmp_path / "deploy.yaml")
        assert clf.min_region_points == 500
        assert clf.region_cap == 4
        pred = clf.classify_frame(scene(), stamp=0.0)
        assert pred.n_clusters_classified == 0
        assert "min_region_points=500" in pred.diagnostic

    def test_legacy_clustering_section_still_works(self, classifier):
        """An older preprocess.yaml, with the pre-rename section."""
        assert classifier.region_method == "dbscan_instances"
        assert classifier.min_region_points == 50
        assert classifier.classify_frame(scene(), stamp=0.0).n_clusters_classified == 2

    def test_unknown_proposal_method_fails_at_startup(self, tmp_path):
        pre = {k: (dict(v) if isinstance(v, dict) else v) for k, v in PREPROCESS.items()}
        pre["region_proposal"] = {"method": "k_means_vibes", "max_regions": 4}
        (tmp_path / "preprocess.yaml").write_text(yaml.safe_dump(pre))
        (tmp_path / "deploy.yaml").write_text(
            yaml.safe_dump(
                {
                    "checkpoint": None,
                    "label_map": None,
                    "placeholder_num_classes": 3,
                    "preprocess": "preprocess.yaml",
                    "model": dict(MODEL),
                    "deployment": {"device": "cpu", "max_clusters": 4},
                }
            )
        )
        with pytest.raises(ValueError, match="unknown region proposal method"):
            ClusterClassifier(tmp_path / "deploy.yaml")


class TestClassifyFrame:
    def test_two_objects_give_two_predictions(self, classifier):
        classifier.warmup(1)
        pred = classifier.classify_frame(scene(), stamp=5.5)
        assert pred.n_clusters_found == 2
        assert pred.n_clusters_classified == 2
        assert pred.stamp == pytest.approx(5.5)
        assert pred.diagnostic == ""
        assert pred.is_placeholder is True
        assert {p.cluster_id for p in pred.predictions} == {0, 1}

    def test_geometry_comes_back_in_metres(self, classifier):
        pred = classifier.classify_frame(scene(), stamp=0.0)
        centroids = sorted(p.centroid_m for p in pred.predictions)
        assert centroids[0] == pytest.approx((0.0, 0.0, 0.0), abs=0.05)
        assert centroids[1] == pytest.approx((4.0, 0.0, 0.0), abs=0.05)

    def test_confidence_is_a_probability(self, classifier):
        pred = classifier.classify_frame(scene(), stamp=0.0)
        assert all(0.0 <= p.confidence <= 1.0 for p in pred.predictions)

    def test_latencies_are_recorded_per_stage(self, classifier):
        pred = classifier.classify_frame(scene(), stamp=0.0)
        for stage in ("validate", "clustering", "cluster_prep", "batching", "inference",
                      "postprocess"):
            assert pred.latencies_ms[stage] >= 0.0

    def test_warmup_state_is_reported_as_a_diagnostic(self, tmp_path):
        """A frame timed before warmup includes one-off kernel costs. Saying so
        is cheaper than explaining a phantom latency spike later."""
        clf = ClusterClassifier(write_configs(tmp_path))
        assert "warmup" in clf.classify_frame(scene(), stamp=0.0).diagnostic
        clf.warmup(1)
        assert clf.classify_frame(scene(), stamp=0.0).diagnostic == ""

    def test_reject_threshold_produces_the_unknown_outcome(self, tmp_path):
        clf = ClusterClassifier(
            write_configs(tmp_path, deploy_overrides={"deployment": {"reject_threshold": 1.0}})
        )
        pred = clf.classify_frame(scene(), stamp=0.0)
        assert pred.predictions
        assert all(p.class_id == -1 and p.class_name == "unknown" for p in pred.predictions)

    def test_cap_bounds_the_batch_and_reports_the_true_count(self, tmp_path):
        clf = ClusterClassifier(
            write_configs(
                tmp_path,
                preprocess_overrides={"clustering": {"max_clusters_per_frame": 2}},
                deploy_overrides={"deployment": {"max_clusters": 2}},
            )
        )
        wide = np.vstack([blob([4.0 * i, 0, 0], seed=i) for i in range(5)]).astype(np.float32)
        pred = clf.classify_frame(wide, stamp=0.0)
        assert pred.n_clusters_found == 5
        assert pred.n_clusters_classified == 2

    def test_whole_scene_mode_classifies_one_sample(self, tmp_path):
        """``clustering.enabled: false`` reproduces the C2 defect on purpose, so
        the old behaviour can be compared against rather than guessed at."""
        clf = ClusterClassifier(
            write_configs(tmp_path, preprocess_overrides={"clustering": {"enabled": False}})
        )
        pred = clf.classify_frame(scene(), stamp=0.0)
        assert pred.n_clusters_classified == 1

    def test_determinism(self, classifier):
        points = blob([0, 0, 0], n=60)  # fewer than num_points: exercises padding
        first = classifier.classify_frame(points, stamp=0.0)
        rerun = ClusterClassifier(classifier.config_path).classify_frame(points, stamp=0.0)
        assert [p.class_id for p in first.predictions] == [p.class_id for p in rerun.predictions]
        assert [p.confidence for p in first.predictions] == pytest.approx(
            [p.confidence for p in rerun.predictions]
        )


class TestNeverRaises:
    @pytest.mark.parametrize(
        "points",
        [
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
            np.full((60, 3), np.nan, dtype=np.float32),
            np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
            np.zeros((10, 2), dtype=np.float32),
            np.zeros((4, 4, 4), dtype=np.float32),
            np.array([[np.inf, 0.0, 0.0]], dtype=np.float32),
        ],
    )
    def test_degenerate_input_returns_a_diagnostic(self, classifier, points):
        pred = classifier.classify_frame(points, stamp=1.0)
        assert pred.predictions == []
        assert pred.n_clusters_classified == 0
        assert pred.diagnostic
        assert pred.stamp == pytest.approx(1.0)

    def test_empty_sampling_mode_is_not_classified(self, classifier):
        """Finding I4: an empty cloud resamples to 1024 zeros. Classifying zeros
        yields a confident label for nothing at all."""
        from pcml.data.sampling import SamplingStats

        clusters = [Cluster.from_points(blob([0, 0, 0], n=60))]
        batch = np.zeros((1, 128, 3), dtype=np.float32)
        pred = classifier.classify_prepared(
            clusters,
            batch,
            stamp=0.0,
            stats=[SamplingStats(0, 0, 128, "empty")],
        )
        assert pred.predictions == []
        assert "non-empty" in pred.diagnostic

    def test_wrong_point_count_is_reported(self, classifier):
        clusters = [Cluster.from_points(blob([0, 0, 0], n=60))]
        pred = classifier.classify_prepared(
            clusters, np.zeros((1, 64, 3), dtype=np.float32), stamp=0.0
        )
        assert pred.predictions == []
        assert "expected (M, 128, 3)" in pred.diagnostic


class TestBufferAndAdapterAgree:
    def test_fill_buffer_matches_to_model_input(self, classifier):
        rng = np.random.default_rng(0)
        batch = rng.standard_normal((3, 128, 3)).astype(np.float32)
        expected = to_model_input(batch, layout="bcn", device="cpu")
        actual = classifier._fill_buffer(batch)
        assert actual.shape == expected.shape == (3, 3, 128)
        assert np.array_equal(actual.numpy(), expected.numpy())
        assert actual.data_ptr() == classifier._batch_buf.data_ptr()

    def test_fill_buffer_matches_to_model_input_bnc(self, tmp_path):
        clf = ClusterClassifier(
            write_configs(tmp_path, deploy_overrides={"deployment": {"input_layout": "bnc"}})
        )
        rng = np.random.default_rng(0)
        batch = rng.standard_normal((2, 128, 3)).astype(np.float32)
        expected = to_model_input(batch, layout="bnc", device="cpu")
        assert np.array_equal(clf._fill_buffer(batch).numpy(), expected.numpy())


class TestAuxFeatures:
    def test_aux_path_runs_when_both_sides_agree(self, tmp_path):
        clf = ClusterClassifier(
            write_configs(
                tmp_path,
                preprocess_overrides={"features": {"use_aux": True}},
                model={**MODEL, "aux_dim": 6},
            )
        )
        clf.warmup(1)
        pred = clf.classify_frame(scene(), stamp=0.0)
        assert pred.n_clusters_classified == 2
        assert pred.diagnostic == ""

    def test_missing_aux_is_reported_not_raised(self, tmp_path):
        clf = ClusterClassifier(
            write_configs(
                tmp_path,
                preprocess_overrides={"features": {"use_aux": True}},
                model={**MODEL, "aux_dim": 6},
            )
        )
        clusters = [Cluster.from_points(blob([0, 0, 0], n=200))]
        batch = np.zeros((1, 128, 3), dtype=np.float32)
        pred = clf.classify_prepared(clusters, batch, aux=None, stamp=0.0)
        assert pred.predictions == []
        assert "aux is None" in pred.diagnostic


class TestWarmup:
    def test_warmup_runs_at_the_full_batch_shape(self, classifier):
        classifier.warmup(2)
        assert classifier._warmed_up is True

    def test_zero_warmup_frames_is_allowed(self, classifier):
        classifier.warmup(0)
        assert classifier._warmed_up is True


def _repo_config(name):
    import pathlib

    return pathlib.Path(__file__).resolve().parents[1] / "configs" / name


class TestShippedPlaceholderConfig:
    """The config that makes the end-to-end path runnable. Since 15 Oct 2026 the
    label map in it is the owner-confirmed one (P1 resolved); only the checkpoint
    is still a placeholder (P2 open)."""

    # The shipped config removes ground, so the input has to contain some. A
    # ground-less scene is not a neutral fixture here: RANSAC fits a plane
    # through the objects themselves and the frame is correctly reported as
    # "every point was ground".
    def test_it_loads_and_runs(self):
        clf = ClusterClassifier(_repo_config("deploy_placeholder.yaml"))
        assert clf.checkpoint_path is None
        assert clf._batch_buf.shape == (16, 3, 1024)
        clf.warmup(1)
        pred = clf.classify_frame(field_scene(), stamp=0.0)
        assert pred.n_clusters_classified == 3

    def test_a_ground_less_scene_is_reported_not_silently_emptied(self):
        """Feeding it a scene with no ground plane is a config/input mismatch,
        and the diagnostic has to say which — an empty prediction list with no
        explanation is indistinguishable from an empty room."""
        clf = ClusterClassifier(_repo_config("deploy_placeholder.yaml"))
        pred = clf.classify_frame(scene(), stamp=0.0)
        assert pred.predictions == []
        assert "every point was ground" in pred.diagnostic

    def test_k_is_six_and_comes_from_the_real_label_map(self):
        clf = ClusterClassifier(_repo_config("deploy_placeholder.yaml"))
        assert clf.label_map.num_classes == 6
        assert clf.model.num_classes == 6
        assert clf.label_map.names == (
            "person",
            "vegetation",
            "pole",
            "structure",
            "net_fence",
            "vehicle",
        )
        # It is the confirmed vocabulary, not the example proposal.
        assert clf.label_map.is_example is False
        assert clf.label_map.version == 1

    def test_random_weights_still_make_every_output_a_placeholder(self):
        """A real class name on a random argmax is more misleading than a
        synthetic one, not less — so the flag must survive P1 being resolved."""
        clf = ClusterClassifier(_repo_config("deploy_placeholder.yaml"))
        pred = clf.classify_frame(field_scene(), stamp=0.0)
        assert clf.is_placeholder is True
        assert pred.is_placeholder is True

    def test_class_names_resolve_through_the_map(self):
        clf = ClusterClassifier(_repo_config("deploy_placeholder.yaml"))
        pred = clf.classify_frame(field_scene(), stamp=0.0)
        assert pred.predictions
        for p in pred.predictions:
            assert 0 <= p.class_id < 6
            assert p.class_name == clf.label_map.names[p.class_id]

    def test_the_reject_name_is_not_a_literal(self):
        """`unknown` comes from `reject.reserved_class_name`, so renaming it in
        the label map changes the output rather than leaving a stale string."""
        clf = ClusterClassifier(_repo_config("deploy_placeholder.yaml"))
        assert clf.label_map.reserved_class_id == -1
        assert clf.label_map.name_for(-1) == clf.label_map.reserved_class_name == "unknown"

    def test_threshold_zero_disables_rejection_deliberately(self):
        """The shipped threshold is 0.0, which means nothing is ever rejected.
        That is config, not a dead branch: a real threshold can only come from a
        precision/recall curve, and there is no trained model to draw one from.
        Asserted explicitly so the 0.0 reads as intentional."""
        clf = ClusterClassifier(_repo_config("deploy_placeholder.yaml"))
        assert clf.reject_threshold == 0.0
        pred = clf.classify_frame(field_scene(), stamp=0.0)
        assert pred.predictions
        assert all(p.class_id != -1 for p in pred.predictions)
        assert all(p.class_name != "unknown" for p in pred.predictions)

    def test_raising_the_threshold_does_reject(self):
        """The branch is live; only the shipped value switches it off."""
        clf = ClusterClassifier(_repo_config("deploy_placeholder.yaml"))
        clf.reject_threshold = 1.0
        pred = clf.classify_frame(field_scene(), stamp=0.0)
        assert pred.predictions  # otherwise the all() below is vacuously true
        assert all(p.class_id == -1 and p.class_name == "unknown" for p in pred.predictions)

    def test_it_performs_ground_removal(self):
        """preprocess.yaml ships `ground_removal.enabled: true`, so the shipped
        config must actually do it."""
        clf = ClusterClassifier(_repo_config("deploy_placeholder.yaml"))
        assert clf.ground_removal_enabled is True
        assert clf.ground_method == "ransac_plane"
        pred = clf.classify_frame(field_scene(), stamp=0.0)
        assert pred.latencies_ms["ground_removal"] > 0.0
        assert pred.n_clusters_classified == 3
