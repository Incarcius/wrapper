"""
Command-line entry point: streams frames from the Mid-70, runs each one through
the preprocessing pipeline, classifies every cluster in a single batched forward
pass, and publishes the result as ``vision_msgs/Detection3DArray``.

Usage (from the repository root):
    PYTHONPATH=src python -m livox_wrapper --config config/pipeline_config.yaml
or, after `pip install -e .`:
    livox-wrapper --config config/pipeline_config.yaml

The classifier lives in ``pcml.deploy`` (a separate repo, installed alongside);
this module only constructs it, hands it each frame and publishes what comes
back. That is the whole coupling, and it is deliberate: swapping architectures is
a config change in ``pointcloud-ml``, not an edit here.
"""
import argparse
import logging
import time

import yaml

from .livox_interface import LivoxInterface
from .pipeline import ConfigError, PointCloudPipeline, validate_config

logger = logging.getLogger(__name__)


def build_classifier(model_cfg: dict):
    """Construct the ``ClusterClassifier`` named by the ``model:`` config section.

    Imported lazily so that the preprocessing path — and its tests — stay
    importable without torch, which on a Jetson comes from the JetPack wheel
    rather than pip.
    """
    from pcml.deploy.classifier import ClusterClassifier

    return ClusterClassifier(model_cfg["classifier_config"])


def build_publisher(model_cfg: dict):
    """Construct the ROS 2 publisher, or return ``None`` when it is disabled."""
    publish_cfg = model_cfg.get("publish") or {}
    if not publish_cfg.get("enabled", False):
        logger.info("model.publish.enabled is false: predictions are logged, not published.")
        return None
    from .ros_publisher import DetectionPublisher

    return DetectionPublisher(
        topic=publish_cfg.get("topic", "/livox/detections"),
        frame_id=publish_cfg.get("frame_id", "livox_frame"),
        queue_depth=int(publish_cfg.get("queue_depth", 2)),
    )


def process_frame(pipeline, classifier, publisher, raw_points, stamp):
    """One frame: preprocess, classify, publish. Returns ``(prediction, timings)``.

    Separated from the loop so it can be driven from a recording or a test
    without ROS 2 and without a sensor.

    ``prediction`` is ``None`` when inference was skipped. That happens when the
    frame yielded no cluster, or when every cluster resampled from nothing
    (``SamplingStats.mode == "empty"``) — finding I4. The first version padded an
    empty cloud to 1024 zeros and classified it, which produces a confident
    label for an empty room.
    """
    frame, timings = pipeline.process(raw_points)

    live = [
        i for i, st in enumerate(frame.stats) if st.mode != "empty"
    ]
    if not frame.clusters or not live:
        logger.debug(
            "frame skipped: %d cluster(s), %d non-empty (points in=%d, clean=%d)",
            frame.n_clusters,
            len(live),
            frame.n_points_in,
            frame.n_points_clean,
        )
        pipeline.maybe_log_timings(timings)
        return None, timings

    t = time.perf_counter()
    prediction = classifier.classify_prepared(
        frame.clusters,
        frame.batch,
        aux=frame.aux if classifier.aux_dim else None,
        stamp=stamp,
        n_clusters_found=frame.n_clusters_found,
        stats=frame.stats,
    )
    classify_seconds = time.perf_counter() - t

    # Fold the classifier's own per-stage numbers back into this frame's
    # StageTimings (finding A4), so one frame produces one latency record
    # instead of two that have to be correlated by eye.
    timings["inference"] = prediction.latencies_ms.get("inference", 0.0) / 1e3
    timings["postprocess"] = prediction.latencies_ms.get("postprocess", 0.0) / 1e3
    timings["batching"] = prediction.latencies_ms.get("batching", 0.0) / 1e3
    timings["classify_total"] = classify_seconds
    timings["total"] = timings["total"] + classify_seconds

    if publisher is not None:
        t = time.perf_counter()
        publisher.publish(prediction)
        timings["publish"] = time.perf_counter() - t

    pipeline.maybe_log_timings(timings)
    if prediction.diagnostic:
        logger.warning("frame diagnostic: %s", prediction.diagnostic)
    return prediction, timings


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        prog="livox-wrapper",
        description="Stream Livox Mid-70 frames through preprocessing and classification.",
    )
    parser.add_argument(
        "--config",
        default="config/pipeline_config.yaml",
        help="path to the pipeline YAML config (default: %(default)s)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Validate before touching the sensor: a config error should be a startup
    # failure, not a crash on frame 3000 (finding I6).
    config = validate_config(config)
    model_cfg = config.get("model") or {}
    if not model_cfg.get("enabled", False):
        raise ConfigError(
            "model.enabled is false: there is nothing to classify with. Point "
            "model.classifier_config at a deploy config (see "
            "pointcloud-ml/configs/deploy_placeholder.yaml) and set enabled: true."
        )

    pipeline = PointCloudPipeline(config)
    classifier = build_classifier(model_cfg)
    if classifier.is_placeholder:
        logger.warning(
            "Running with PLACEHOLDER model artifacts (checkpoint=%s, label_map=%s). "
            "Latency and plumbing are measurable; the class labels are not "
            "predictions and must not be reported as accuracy.",
            classifier.checkpoint_path,
            classifier.label_map_path,
        )
    publisher = build_publisher(model_cfg)

    # Warm-up before the loop, so the first real frame is not charged for
    # one-off kernel autotuning and CUDA context creation.
    classifier.warmup(int(model_cfg.get("warmup_frames", classifier.warmup_frames)))

    ingest = LivoxInterface(config["ingestion"])
    ingest.start()

    logger.info("Pipeline running. Ctrl+C to stop.")
    try:
        while True:
            frame = ingest.get_latest_frame(timeout=1.0)
            if frame is None:
                continue
            raw_points, stamp = frame
            prediction, _ = process_frame(pipeline, classifier, publisher, raw_points, stamp)
            if prediction is not None:
                logger.debug(
                    "stamp=%.3f found=%d classified=%d",
                    prediction.stamp,
                    prediction.n_clusters_found,
                    prediction.n_clusters_classified,
                )
    finally:
        ingest.stop()
        if publisher is not None:
            publisher.destroy()


if __name__ == "__main__":
    main()
