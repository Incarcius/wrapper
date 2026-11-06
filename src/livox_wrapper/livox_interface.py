"""
LivoxInterface
--------------
Feeds raw point-cloud frames from the Mid-70 into the processing pipeline.

Two backends:
  - "ros2": subscribes to a sensor_msgs/PointCloud2 topic published by
    livox_ros2_driver (recommended — handles device bring-up, timestamps,
    and plays nicely with the rest of a drone's ROS2 stack).
  - "sdk": talks to livox_sdk Python bindings directly, for setups without ROS2.

Both backends push frames into a bounded queue and DROP the oldest frame
if the consumer (the pipeline) falls behind, rather than blocking or
building up latency — important for a moving drone where a stale point
cloud is often worse than a missing one.
"""

from __future__ import annotations

import queue
import threading
import time
import logging
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


class LivoxInterface:
    def __init__(self, config: dict, on_frame: Optional[Callable[[np.ndarray, float], None]] = None):
        """
        config: the 'ingestion' section of pipeline_config.yaml
        on_frame: optional callback(points, timestamp) invoked per frame;
                  if not provided, use get_latest_frame() / frame stream via .frames()
        """
        self.cfg = config
        self._queue: "queue.Queue[tuple[np.ndarray, float]]" = queue.Queue(
            maxsize=config.get("frame_queue_size", 2)
        )
        self._on_frame = on_frame
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    def start(self):
        mode = self.cfg.get("mode", "ros2")
        if mode == "ros2":
            self._thread = threading.Thread(target=self._run_ros2, daemon=True)
        elif mode == "sdk":
            self._thread = threading.Thread(target=self._run_sdk, daemon=True)
        else:
            raise ValueError(f"Unknown ingestion mode: {mode}")
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    def _push_frame(self, points: np.ndarray, timestamp: float):
        """Push a frame, dropping the oldest queued frame if full (never blocks the ingestion thread)."""
        drop_after = self.cfg.get("drop_stale_after_sec", 0.15)
        if time.time() - timestamp > drop_after:
            logger.debug("Dropping stale frame (%.3fs old)", time.time() - timestamp)
            return

        if self._on_frame is not None:
            self._on_frame(points, timestamp)
            return

        try:
            self._queue.put_nowait((points, timestamp))
        except queue.Full:
            try:
                self._queue.get_nowait()  # drop oldest
            except queue.Empty:
                pass
            self._queue.put_nowait((points, timestamp))

    def get_latest_frame(self, timeout: float = 1.0) -> Optional[tuple[np.ndarray, float]]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    # ------------------------------------------------------------------
    # ROS2 backend
    # ------------------------------------------------------------------
    def _run_ros2(self):
        try:
            import rclpy
            from rclpy.node import Node
            from sensor_msgs.msg import PointCloud2
            import sensor_msgs_py.point_cloud2 as pc2
        except ImportError as e:
            raise RuntimeError(
                "ROS2 ingestion mode requires rclpy + sensor_msgs_py "
                "(source your ROS2 environment / install livox_ros2_driver)."
            ) from e

        parent = self

        class _Sub(Node):
            def __init__(self):
                super().__init__("livox_pipeline_ingest")
                self.create_subscription(
                    PointCloud2, parent.cfg["ros2_topic"], self._cb, 10
                )

            def _cb(self, msg: PointCloud2):
                # structured -> plain (N,3) float32 array; avoid per-point Python loops
                field_names = [f.name for f in msg.fields]
                xyz_fields = [f for f in ("x", "y", "z") if f in field_names]
                cloud = pc2.read_points_numpy(msg, field_names=xyz_fields, skip_nans=True)
                points = np.asarray(cloud, dtype=np.float32).reshape(-1, 3)
                stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                parent._push_frame(points, stamp if stamp > 0 else time.time())

        rclpy.init(args=None)
        node = _Sub()
        try:
            while not self._stop_event.is_set() and rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.05)
        finally:
            node.destroy_node()
            rclpy.shutdown()

    # ------------------------------------------------------------------
    # Direct SDK backend
    # ------------------------------------------------------------------
    def _run_sdk(self):
        try:
            import livox_sdk  # placeholder import name; confirm actual package/binding you install
        except ImportError as e:
            raise RuntimeError(
                "SDK ingestion mode requires the livox_sdk Python bindings. "
                "If these aren't available for your setup, use mode: ros2 instead."
            ) from e

        # NOTE: livox_sdk's Python bindings are thin wrappers over the C SDK and
        # their exact API varies by version/build. Sketch of the expected flow:
        #
        #   device = livox_sdk.connect(...)
        #   device.start()
        #   for frame in device.frame_stream():
        #       points = np.asarray(frame.points, dtype=np.float32)  # (N,3)
        #       self._push_frame(points, time.time())
        #       if self._stop_event.is_set():
        #           break
        #   device.stop()
        #
        # Left as a stub — plug in the exact binding calls for your SDK build/version.
        raise NotImplementedError(
            "Fill in the livox_sdk device connect/stream calls for your SDK version, "
            "or use mode: ros2 (recommended)."
        )
