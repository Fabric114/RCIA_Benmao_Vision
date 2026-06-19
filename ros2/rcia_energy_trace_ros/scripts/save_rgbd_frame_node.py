#!/usr/bin/env python3
import os
import sys
import threading
from pathlib import Path

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

        self.declare_parameter("color_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter(
            "output_dir", str(PROJECT_ROOT / "runtime/rgbd_captures")
        )
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
        self.get_logger().info("Waiting for synchronized RGB-D frame to save")

    def _camera_info_cb(self, msg):
        with self.lock:
            self.camera_info = msg

    def _depth_cb(self, msg):
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        with self.lock:
            self.depth_buffer.put(stamp_to_nanoseconds(msg.header.stamp), depth.copy())

    def _color_cb(self, msg):
        capture_count = int(self.get_parameter("capture_count").value)
        if self.saved_count >= capture_count:
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

        output_root = Path(self.get_parameter("output_dir").value)
        frame_dir = output_root / f"frame_{self.saved_count:06d}"
        save_rgbd_frame(
            frame_dir,
            color,
            matched_depth[1],
            camera_info,
            color_stamp_ns,
        )
        self.saved_count += 1
        self.get_logger().info(f"Saved RGB-D frame to {frame_dir}")

        if self.saved_count >= capture_count and bool(
            self.get_parameter("shutdown_after_capture").value
        ):
            self.shutdown_requested = True


def main():
    rclpy.init()
    node = SaveRgbdFrameNode()
    try:
        while rclpy.ok() and not node.shutdown_requested:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
