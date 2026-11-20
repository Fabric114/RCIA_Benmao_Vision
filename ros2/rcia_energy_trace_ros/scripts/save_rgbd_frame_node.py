#!/usr/bin/env python3
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


PROJECT_ROOT = Path(
    os.environ.get(
        "RCIA_ENERGY_TRACE_ROOT",
        Path(__file__).resolve().parents[3],
    )
).resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.rgbd_frame_saver import save_rgbd_frame  # noqa: E402
from scripts.ros_rgbd_sync import StampedFrameBuffer, stamp_to_nanoseconds  # noqa: E402


class SaveRgbdFrameNode(Node):
    def __init__(self):
        super().__init__("save_rgbd_frame_node")
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.depth_buffer = StampedFrameBuffer(max_size=120)
        self.camera_info = None
        self.saved_count = 0
        self.shutdown_requested = False
        self.latest_frame = None
        self.latest_frame_time = 0.0
        self.last_saved_stamp = None
        self.next_frame_index = 0

        self.declare_parameter("color_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter(
            "output_dir",
            str(PROJECT_ROOT / "runtime" / datetime.now().strftime("fp_eval_%Y%m%d_%H%M%S_%f") / "frames"),
        )
        self.declare_parameter("interactive", True)
        # capture_count / shutdown_after_capture apply only to automatic capture.
        self.declare_parameter("capture_count", 1)
        self.declare_parameter("sync_tolerance_ms", 5.0)
        self.declare_parameter("shutdown_after_capture", True)

        self.create_subscription(
            CameraInfo,
            self.get_parameter("camera_info_topic").value,
            self._camera_info_cb,
            10,
        )
        self.create_subscription(
            Image,
            self.get_parameter("depth_topic").value,
            self._depth_cb,
            10,
        )
        self.create_subscription(
            Image,
            self.get_parameter("color_topic").value,
            self._color_cb,
            10,
        )
        self.interactive = bool(self.get_parameter("interactive").value)
        self.output_root = Path(self.get_parameter("output_dir").value).expanduser()
        self.get_logger().info(f"Capture directory: {self.output_root}")
        self.get_logger().info(
            "Click the capture preview: s = save, q = quit"
            if self.interactive else "Waiting for synchronized RGB-D frame to save"
        )

    def _camera_info_cb(self, msg):
        with self.lock:
            self.camera_info = msg

    def _depth_cb(self, msg):
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        with self.lock:
            self.depth_buffer.put(stamp_to_nanoseconds(msg.header.stamp), depth.copy())

    def _color_cb(self, msg):
        capture_count = int(self.get_parameter("capture_count").value)
        if not self.interactive and self.saved_count >= capture_count:
            return

        color = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        color_stamp_ns = stamp_to_nanoseconds(msg.header.stamp)
        tolerance_ns = int(float(self.get_parameter("sync_tolerance_ms").value) * 1_000_000)
        with self.lock:
            matched_depth = self.depth_buffer.get_nearest(color_stamp_ns, tolerance_ns)
            camera_info = self.camera_info

        if matched_depth is None or camera_info is None:
            self.get_logger().debug("Waiting for matching depth and camera_info")
            return

        self.latest_frame = (color, matched_depth[1], camera_info, color_stamp_ns)
        self.latest_frame_time = time.monotonic()
        if not self.interactive:
            self._save_frame(self.latest_frame)

    def _save_frame(self, frame):
        color, depth, camera_info, color_stamp_ns = frame
        # Reserve a new directory, including when resuming an existing dataset.
        self.output_root.mkdir(parents=True, exist_ok=True)
        while True:
            frame_dir = self.output_root / f"frame_{self.next_frame_index:06d}"
            self.next_frame_index += 1
            try:
                frame_dir.mkdir()
                break
            except FileExistsError:
                continue
        save_rgbd_frame(
            frame_dir,
            color,
            depth,
            camera_info,
            color_stamp_ns,
        )
        self.saved_count += 1
        self.last_saved_stamp = color_stamp_ns
        self.get_logger().info(f"Saved RGB-D frame to {frame_dir}")

        if not self.interactive and self.saved_count >= int(self.get_parameter("capture_count").value) and bool(
            self.get_parameter("shutdown_after_capture").value
        ):
            self.shutdown_requested = True

    def update_preview(self, window_name):
        frame = self.latest_frame
        fresh = frame is not None and time.monotonic() - self.latest_frame_time < 1.0
        preview = frame[0].copy() if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        status = f"s: save | q: quit | saved: {self.saved_count}"
        if not fresh:
            status = "Waiting for synchronized RGB-D | q: quit"
        cv2.putText(preview, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0) if fresh else (0, 0, 255), 2)
        cv2.imshow(window_name, preview)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")) or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            self.shutdown_requested = True
        elif key in (ord("s"), ord("S")):
            if not fresh:
                self.get_logger().warning("No fresh synchronized RGB-D frame; nothing saved")
            elif frame[3] == self.last_saved_stamp:
                self.get_logger().warning("This frame was already saved; waiting for a new frame")
            else:
                self._save_frame(frame)


def main():
    rclpy.init()
    node = SaveRgbdFrameNode()
    window_name = "RGB-D Capture"
    try:
        if node.interactive:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        while rclpy.ok() and not node.shutdown_requested:
            rclpy.spin_once(node, timeout_sec=0.01 if node.interactive else 0.1)
            if node.interactive:
                node.update_preview(window_name)
    except KeyboardInterrupt:
        pass
    finally:
        if node.interactive:
            cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
