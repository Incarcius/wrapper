"""End to end: a synthetic multi-object scene becomes a FramePrediction.

This is the test that the whole integration exists for. It runs the real path —
pipeline, clustering, per-cluster resampling, one batched forward pass,
post-processing, and the message conversion — against the shipped placeholder
deploy config.

What it can assert: shapes, counts, metric geometry, bounded latency fields,
determinism, graceful handling of degenerate frames, and that the ROS 2 message
would carry the right numbers.

What it cannot assert, and does not: anything about which class is predicted.
The checkpoint is absent and the weights are random (``task_definition.md``
P1/P2). Every assertion about ``class_id`` here is a structural one.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from livox_wrapper.cli import process_frame
from livox_wrapper.pipeline import PointCloudPipeline
from livox_wrapper.ros_publisher import detection_array_fields

from test_pipeline import BASE_CONFIG, blob, field_scene, multi_object_scene

pytest.importorskip("torch", exc_type=ImportError)

DEPLOY_CONFIG = (
    pathlib.Path(__file__).resolve().parents[2]
    / "pointcloud-ml"
    / "configs"
    / "deploy_placeholder.yaml"
)

pytestmark = pytest.mark.skipif(
    not DEPLOY_CONFIG.is_file(),
    reason=f"needs the sibling pointcloud-ml checkout at {DEPLOY_CONFIG}",
)


@pytest.fixture(scope="module")
def classifier():
    from pcml.deploy.classifier import ClusterClassifier

    clf = ClusterClassifier(DEPLOY_CONFIG)
    clf.warmup(2)  # 2 rather than the configured 10: this is a CPU test box
    return clf


def _pipeline(ground_removal: bool):
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in BASE_CONFIG.items()}
    cfg["clustering"].update({"eps_m": 0.5, "min_cluster_points": 50})
    cfg["downsampling"].update({"voxel_size_m": 0.05})
    cfg["ground_removal"] = {"enabled": ground_removal}
    return PointCloudPipeline(cfg)


@pytest.fixture
def pipeline():
    """Ground removal off, because ``multi_object_scene`` has no ground in it.

    Most tests here care about the cluster -> batch -> prediction path, so they
    use bare blobs; RANSAC would fit a plane through one of those and delete it.
    The entry-point parity test below uses a field scene with the stage on, which
    is the configuration that actually ships.
    """
    return _pipeline(ground_removal=False)


class TestEndToEnd:
    def test_scene_becomes_per_cluster_predictions(self, pipeline, classifier):
        scene = multi_object_scene()
        frame, timings = pipeline.process(scene)
        prediction = classifier.classify_prepared(
            frame.clusters,
            frame.batch,
            stamp=1234.5,
            n_clusters_found=frame.n_clusters_found,
            stats=frame.stats,
        )

        assert prediction.stamp == pytest.approx(1234.5)
        assert prediction.n_clusters_found == 3
        assert prediction.n_clusters_classified == 3
        assert len(prediction.predictions) == 3
        assert prediction.diagnostic == ""

        # The outputs are a function of random weights. This flag is how a
        # consumer knows that, and it is set from the config, not asserted into
        # existence here.
        assert prediction.is_placeholder is True

        for pred in prediction.predictions:
            assert 0.0 <= pred.confidence <= 1.0
            # Structural only: with random weights the class id carries no
            # information. It must be a valid index or the reject id.
            assert pred.class_id == -1 or 0 <= pred.class_id < classifier.label_map.num_classes
            # The vocabulary is the owner-confirmed one, so names are real class
            # names. They still carry no information here — with random weights
            # "vegetation" only means the largest logit landed at index 1 — which
            # is what `is_placeholder` above is for. What is checkable is that the
            # name was resolved through the label map rather than written inline.
            if pred.class_id == -1:
                assert pred.class_name == classifier.label_map.reserved_class_name
            else:
                assert pred.class_name == classifier.label_map.names[pred.class_id]
            assert pred.n_points_raw >= 50
            assert len(pred.centroid_m) == 3
            assert len(pred.extent_m) == 3

        # Geometry is reported in sensor-frame metres, matching what went in.
        by_id = {p.cluster_id: p for p in prediction.predictions}
        for cluster_id, pred in by_id.items():
            assert pred.centroid_m == pytest.approx(
                frame.clusters[cluster_id].centroid_m, abs=1e-6
            )
            assert pred.extent_m == pytest.approx(frame.clusters[cluster_id].extent_m, abs=1e-6)

        for stage in ("batching", "inference", "postprocess"):
            assert prediction.latencies_ms[stage] >= 0.0
        assert timings.n_clusters == 3

    def test_the_two_entry_points_agree_on_the_same_scene(self, classifier):
        """``classify_frame`` owns ground removal, clustering and resampling;
        the wrapper's pipeline does the same work and hands the result to
        ``classify_prepared``. Given one scene and one configuration the two must
        reach the same clusters — otherwise the frame's interpretation depends on
        which entry point the caller happened to use.

        Run with a field scene and the ground stage on, which is what ships.
        """
        scene = field_scene()

        via_classify_frame = classifier.classify_frame(scene, stamp=1.0)

        frame, _ = _pipeline(ground_removal=True).process(scene)
        via_pipeline = classifier.classify_prepared(
            frame.clusters,
            frame.batch,
            stamp=1.0,
            n_clusters_found=frame.n_clusters_found,
            stats=frame.stats,
        )

        assert via_classify_frame.n_clusters_classified == 4
        assert via_pipeline.n_clusters_classified == 4
        assert via_classify_frame.n_clusters_found == via_pipeline.n_clusters_found
        assert via_classify_frame.diagnostic == ""

        # Same objects, in the same metric frame, from both routes.
        def geometry(pred):
            return sorted(
                tuple(round(v, 3) for v in p.centroid_m) for p in pred.predictions
            )

        assert geometry(via_classify_frame) == geometry(via_pipeline)

    def test_all_clusters_go_through_one_forward_pass(self, pipeline, classifier):
        """Batching is mandatory, not an optimisation: M separate calls pay
        dispatch overhead M times (task_definition.md §9.2). The preallocated
        buffer is the observable consequence."""
        frame, _ = pipeline.process(multi_object_scene())
        view = classifier._fill_buffer(frame.batch)
        assert view.shape == (3, 3, 1024)
        assert view.is_contiguous()
        # The view must be a slice of the single preallocated tensor, not a new one.
        assert view.data_ptr() == classifier._batch_buf.data_ptr()
        assert classifier._batch_buf.shape == (16, 3, 1024)

    def test_buffer_fill_agrees_with_the_adapter(self, pipeline, classifier):
        """The allocation-free path and ``to_model_input`` must produce the same
        tensor, so the layout rule stays in one place (finding I5)."""
        from pcml.deploy.adapter import to_model_input

        frame, _ = pipeline.process(multi_object_scene())
        expected = to_model_input(frame.batch, layout=classifier.layout, device="cpu")
        actual = classifier._fill_buffer(frame.batch)
        assert actual.shape == expected.shape
        assert np.array_equal(actual.numpy(), expected.numpy())

    def test_deterministic_given_the_seed(self, classifier):
        """Invariant 4. Requires A1: with the global RNG, padding a small cluster
        gave a different model input on every run."""
        scene = multi_object_scene()

        def run():
            pipe = PointCloudPipeline(
                {k: (dict(v) if isinstance(v, dict) else v) for k, v in BASE_CONFIG.items()}
            )
            frame, _ = pipe.process(scene)
            pred = classifier.classify_prepared(
                frame.clusters, frame.batch, stamp=0.0, stats=frame.stats
            )
            return [(p.cluster_id, p.class_id, p.confidence) for p in pred.predictions]

        assert run() == run()


class TestFrameLoop:
    def test_process_frame_returns_a_prediction(self, pipeline, classifier):
        prediction, timings = process_frame(
            pipeline, classifier, None, multi_object_scene(), stamp=7.0
        )
        assert prediction is not None
        assert len(prediction.predictions) == 3
        # The classifier's own stages are folded back into this frame's timings
        # so one frame yields one latency record (finding A4).
        assert timings.inference >= 0.0
        assert timings["classify_total"] >= 0.0
        assert timings.total >= timings.cluster_prep

    def test_inference_is_skipped_on_an_empty_frame(self, pipeline, classifier):
        """Finding I4: the first version padded an empty cloud to 1024 zeros and
        classified it, producing a confident label for an empty room."""
        prediction, timings = process_frame(
            pipeline, classifier, None, np.zeros((0, 3), dtype=np.float32), stamp=8.0
        )
        assert prediction is None
        assert timings.inference == 0.0
        assert timings.n_clusters == 0

    def test_inference_is_skipped_when_no_cluster_is_large_enough(self, pipeline, classifier):
        prediction, _ = process_frame(
            pipeline, classifier, None, blob([0.0, 0.0, 0.0], n=10), stamp=9.0
        )
        assert prediction is None


class TestClassifierNeverRaises:
    """Invariant 1: a perception module must not be able to crash the flight
    software. Each of these is a malformed input, not a programming error."""

    @pytest.mark.parametrize(
        "points, label",
        [
            (np.zeros((0, 3), dtype=np.float32), "empty"),
            (np.full((50, 3), np.nan, dtype=np.float32), "all-NaN"),
            (np.array([[1.0, 2.0, 3.0]], dtype=np.float32), "single point"),
            (np.zeros((10, 2), dtype=np.float32), "wrong width"),
            (np.zeros((5, 3, 3), dtype=np.float32), "wrong rank"),
            (np.full((50, 3), np.inf, dtype=np.float32), "all-Inf"),
        ],
    )
    def test_degenerate_input_yields_a_diagnostic(self, classifier, points, label):
        prediction = classifier.classify_frame(points, stamp=0.0)
        assert prediction.predictions == []
        assert prediction.n_clusters_classified == 0
        assert prediction.diagnostic, f"{label} input produced no diagnostic"

    def test_mismatched_batch_is_reported_not_raised(self, classifier):
        from pcml.data.clustering import Cluster

        clusters = [Cluster.from_points(blob([0.0, 0.0, 0.0], n=100))]
        bad_batch = np.zeros((2, 1024, 3), dtype=np.float32)  # 2 rows, 1 cluster
        prediction = classifier.classify_prepared(clusters, bad_batch, stamp=0.0)
        assert prediction.predictions == []
        assert "does not match" in prediction.diagnostic


class TestDetectionMessageFields:
    """The ROS 2 publisher (finding I8) converted through a pure function, so it
    is testable in an environment with no ROS 2 — which is every environment
    this project currently has. The rclpy class itself is unexecuted."""

    def test_fields_mirror_the_predictions(self, pipeline, classifier):
        frame, _ = pipeline.process(multi_object_scene())
        prediction = classifier.classify_prepared(
            frame.clusters, frame.batch, stamp=12.25, stats=frame.stats
        )
        fields = detection_array_fields(prediction, frame_id="livox_frame")

        assert fields["header"]["frame_id"] == "livox_frame"
        assert fields["header"]["stamp"] == {"sec": 12, "nanosec": 250_000_000}
        assert len(fields["detections"]) == len(prediction.predictions)

        for det, pred in zip(fields["detections"], prediction.predictions):
            cx, cy, cz = pred.centroid_m
            ex, ey, ez = pred.extent_m
            # Centroid in metres, unchanged, in both the box and the hypothesis pose.
            assert det["bbox"]["center"]["position"] == {"x": cx, "y": cy, "z": cz}
            assert det["bbox"]["size"] == {"x": ex, "y": ey, "z": ez}
            # Axis-aligned: the identity quaternion states that no orientation
            # was estimated, rather than claiming the object is axis-aligned.
            assert det["bbox"]["center"]["orientation"]["w"] == 1.0
            hypothesis = det["results"][0]["hypothesis"]
            assert hypothesis["score"] == pytest.approx(pred.confidence)
            assert det["id"] == str(pred.cluster_id)

    def test_rejected_clusters_publish_a_name_not_minus_one(self, pipeline, classifier):
        """A consumer reading only ``class_id`` must not be able to read an
        unknown as class 1. With the reject threshold at 1.0 every cluster is
        rejected, which is the behaviour to check."""
        from pcml.deploy.classifier import ClusterClassifier

        clf = ClusterClassifier(DEPLOY_CONFIG)
        clf.reject_threshold = 1.0  # forces rejection; the config default is 0.0
        frame, _ = pipeline.process(multi_object_scene())
        prediction = clf.classify_prepared(
            frame.clusters, frame.batch, stamp=0.0, stats=frame.stats
        )
        assert all(p.class_id == -1 for p in prediction.predictions)
        assert all(p.class_name == "unknown" for p in prediction.predictions)
        fields = detection_array_fields(prediction, frame_id="livox_frame")
        assert all(
            d["results"][0]["hypothesis"]["class_id"] == "unknown" for d in fields["detections"]
        )

    def test_empty_prediction_publishes_an_empty_array(self, classifier):
        prediction = classifier.classify_frame(np.zeros((0, 3), dtype=np.float32), stamp=3.0)
        fields = detection_array_fields(prediction, frame_id="livox_frame")
        assert fields["detections"] == []
        assert fields["header"]["stamp"]["sec"] == 3
