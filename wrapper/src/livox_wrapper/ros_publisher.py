"""Publish per-cluster predictions back to ROS 2 (finding I8).

The first version of this wrapper subscribed to ``/livox/lidar`` and published
nothing at all, so no downstream node could consume the perception output.

Message choice: ``vision_msgs/Detection3DArray``. It carries, per instance, a
class id, a score and a 3D bounding box, which is exactly the
``ClusterPrediction`` contract. ``vision_msgs/Classification2D`` cannot express
per-instance geometry and ``Detection2DArray`` is the wrong dimensionality;
publishing either would lose the centroid, which is the part a planner actually
needs. (``task_definition.md`` §5, marked PROVISIONAL there pending P3 — who
consumes the topic.)

Structure of this module: the conversion is a **pure function** over plain
Python dicts (:func:`detection_array_fields`), and the ROS 2 class is a thin
shell that copies those dicts into real messages. rclpy and vision_msgs come
from a ROS 2 install, not from pip, so they cannot be imported in a dev venv —
keeping the logic out of the ROS-dependent class is what makes it testable
there. **The ROS 2 class itself has not been executed anywhere yet:** no ROS 2
on the dev machine and no Jetson reachable from it.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:  # pragma: no cover - requires a ROS 2 install
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from vision_msgs.msg import (
        BoundingBox3D,
        Detection3D,
        Detection3DArray,
        ObjectHypothesisWithPose,
    )

    _HAS_ROS2 = True
except ImportError:  # pragma: no cover - the dev-environment path
    _HAS_ROS2 = False

__all__ = [
    "HAS_ROS2",
    "detection_array_fields",
    "DetectionPublisher",
]

#: Whether rclpy and vision_msgs are importable in this process.
HAS_ROS2 = _HAS_ROS2


def _split_stamp(stamp: float) -> tuple[int, int]:
    """Seconds float -> ``(sec, nanosec)``, as ``builtin_interfaces/Time`` wants.

    The sensor stamp is echoed through unchanged (invariant 3): a perception
    message timestamped with the publication time rather than the capture time
    is unusable for any consumer doing temporal association.
    """
    sec = int(stamp // 1)
    nanosec = int(round((stamp - sec) * 1e9))
    if nanosec >= 1_000_000_000:  # rounding can carry
        sec += 1
        nanosec -= 1_000_000_000
    return sec, max(nanosec, 0)


def detection_array_fields(prediction: Any, frame_id: str) -> dict:
    """``FramePrediction`` -> a plain-dict mirror of ``Detection3DArray``.

    Pure, dependency-free and therefore testable without ROS 2. The keys and
    nesting match the message fields one for one, so
    :meth:`DetectionPublisher.publish` is a mechanical copy with no logic in it.

    Geometry passes through in sensor-frame metres. The bounding box is
    axis-aligned, which is what clustering gives: ``center.orientation`` is the
    identity quaternion, and that is an honest statement that no orientation was
    estimated — not a claim that the object is axis-aligned.

    ``class_id`` is a string in ``vision_msgs``; the integer class id is
    stringified, and a rejected cluster carries the label map's reject name
    rather than ``"-1"``, so a consumer reading only ``class_id`` cannot mistake
    an unknown for class 1.
    """
    sec, nanosec = _split_stamp(float(prediction.stamp))
    detections = []
    for pred in prediction.predictions:
        cx, cy, cz = pred.centroid_m
        ex, ey, ez = pred.extent_m
        class_id = (
            pred.class_name if pred.class_id < 0 else f"{pred.class_id}"
        )
        detections.append(
            {
                "header": {"stamp": {"sec": sec, "nanosec": nanosec}, "frame_id": frame_id},
                "id": str(pred.cluster_id),
                "results": [
                    {
                        "hypothesis": {"class_id": class_id, "score": float(pred.confidence)},
                        "pose": {
                            "position": {"x": float(cx), "y": float(cy), "z": float(cz)},
                            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                        },
                    }
                ],
                "bbox": {
                    "center": {
                        "position": {"x": float(cx), "y": float(cy), "z": float(cz)},
                        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                    },
                    "size": {"x": float(ex), "y": float(ey), "z": float(ez)},
                },
            }
        )
    return {
        "header": {"stamp": {"sec": sec, "nanosec": nanosec}, "frame_id": frame_id},
        "detections": detections,
    }


class DetectionPublisher:
    """Publishes ``Detection3DArray`` on a ROS 2 topic.

    Constructed only when ``model.publish.enabled`` is true. If rclpy is not
    importable the constructor raises, rather than degrading to a no-op: a
    perception node that silently stops publishing looks healthy from the
    outside, which is the worst of the available failure modes.

    Args:
        topic: topic name to publish on.
        frame_id: the frame the centroids are expressed in. This is the **Livox
            sensor frame** — nothing in this wrapper transforms to body or world
            (see the audit's data contract), so naming it anything else here
            would be a lie that a TF lookup would then act on.
        node: an existing ``rclpy`` node to publish from. When omitted a private
            node is created, which also means ``rclpy.init`` must already have
            been called by the ingestion thread.
        queue_depth: publisher QoS depth. Small on purpose: a stale detection is
            worse than a dropped one on a moving platform, which is the same
            reasoning as the ingestion queue's drop-oldest policy.
    """

    def __init__(
        self,
        topic: str = "/livox/detections",
        frame_id: str = "livox_frame",
        node: Optional[Any] = None,
        queue_depth: int = 2,
    ) -> None:
        if not _HAS_ROS2:  # pragma: no cover - the dev-environment path
            raise RuntimeError(
                "Publishing requires rclpy and vision_msgs, which come from a ROS 2 "
                "install rather than pip (`apt install ros-humble-vision-msgs`, then "
                "source the setup file). Set model.publish.enabled: false to run the "
                "pipeline without publishing."
            )
        self.topic = topic
        self.frame_id = frame_id
        self._owns_node = node is None
        if node is None:  # pragma: no cover - requires ROS 2
            if not rclpy.ok():
                raise RuntimeError(
                    "rclpy is not initialised. DetectionPublisher expects to share the "
                    "context the ingestion node already created."
                )
            node = Node("livox_detection_publisher")
        self.node = node
        qos = QoSProfile(depth=queue_depth, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.publisher = node.create_publisher(Detection3DArray, topic, qos)
        logger.info("Publishing %s on %s (frame_id=%s)", "Detection3DArray", topic, frame_id)

    # ------------------------------------------------------------------
    def publish(self, prediction: Any) -> Any:  # pragma: no cover - requires ROS 2
        """Convert and publish one ``FramePrediction``. Returns the message sent."""
        msg = self.to_msg(prediction)
        self.publisher.publish(msg)
        return msg

    def to_msg(self, prediction: Any) -> Any:  # pragma: no cover - requires ROS 2
        """Build a ``Detection3DArray`` from the dict that the pure function made."""
        fields = detection_array_fields(prediction, self.frame_id)
        msg = Detection3DArray()
        _apply_header(msg.header, fields["header"])
        for det_fields in fields["detections"]:
            det = Detection3D()
            _apply_header(det.header, det_fields["header"])
            det.id = det_fields["id"]
            hyp = ObjectHypothesisWithPose()
            _apply_hypothesis(hyp, det_fields["results"][0])
            det.results.append(hyp)
            det.bbox = BoundingBox3D()
            _apply_pose(det.bbox.center, det_fields["bbox"]["center"])
            size = det_fields["bbox"]["size"]
            det.bbox.size.x = size["x"]
            det.bbox.size.y = size["y"]
            det.bbox.size.z = size["z"]
            msg.detections.append(det)
        return msg

    def destroy(self) -> None:  # pragma: no cover - requires ROS 2
        if self._owns_node:
            self.node.destroy_node()


def _apply_header(header: Any, fields: dict) -> None:  # pragma: no cover - requires ROS 2
    header.stamp.sec = fields["stamp"]["sec"]
    header.stamp.nanosec = fields["stamp"]["nanosec"]
    header.frame_id = fields["frame_id"]


def _apply_pose(pose: Any, fields: dict) -> None:  # pragma: no cover - requires ROS 2
    pose.position.x = fields["position"]["x"]
    pose.position.y = fields["position"]["y"]
    pose.position.z = fields["position"]["z"]
    pose.orientation.x = fields["orientation"]["x"]
    pose.orientation.y = fields["orientation"]["y"]
    pose.orientation.z = fields["orientation"]["z"]
    pose.orientation.w = fields["orientation"]["w"]


def _apply_hypothesis(hyp: Any, fields: dict) -> None:  # pragma: no cover - requires ROS 2
    """Fill an ``ObjectHypothesisWithPose`` across the two vision_msgs layouts.

    vision_msgs 4.x (Humble and later) nests ``hypothesis.class_id`` /
    ``hypothesis.score``; 3.x (Foxy/Galactic) has flat ``id`` / ``score`` fields.
    The target is Humble, but this is cheap insurance against being handed an
    older install, and the alternative is an AttributeError on the first frame.
    """
    h = fields["hypothesis"]
    if hasattr(hyp, "hypothesis"):
        hyp.hypothesis.class_id = h["class_id"]
        hyp.hypothesis.score = h["score"]
    else:
        hyp.id = h["class_id"]
        hyp.score = h["score"]
    _apply_pose(hyp.pose.pose, fields["pose"])
