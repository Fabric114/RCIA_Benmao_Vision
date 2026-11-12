#!/usr/bin/env python3
import json
import os
import sys
import threading
from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


PROJECT_ROOT = Path(
    os.environ.get(
        "RCIA_ENERGY_TRACE_ROOT",
        Path(__file__).resolve().parents[3],
    )
).resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.diagnostic_visualization import compose_pipeline_view  # noqa: E402


class PipelineVisualizerNode(Node):
    def __init__(self):
        super().__init__("pipeline_visualizer_node")
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.latest = {
            "color": None,
            "yolo": None,
            "mask": None,
            "foundationpose": None,
            "m3t": None,
            "tracking_status": None,
        }
        self.latest_received_ns = {
            "color": 0,
            "yolo": 0,
            "mask": 0,
            "foundationpose": 0,
            "m3t": 0,
        }

        self.declare_parameter("color_topic", "/camera/color/image_raw")
        self.declare_parameter("yolo_topic", "/energy/diagnostics/yolo")
        self.declare_parameter("mask_topic", "/energy/diagnostics/mask")
        self.declare_parameter(
            "foundationpose_topic",
            "/energy/diagnostics/foundationpose",
        )
        self.declare_parameter("m3t_topic", "/energy/diagnostics/m3t_overlay")
        self.declare_parameter("tracking_health_topic", "/energy/tracking_health")
        self.declare_parameter("output_topic", "/energy/diagnostics/pipeline_view")
        self.declare_parameter("show_window", True)
        self.declare_parameter("window_name", "RCIA Energy Tracking Pipeline")
        self.declare_parameter("panel_width", 480)
        self.declare_parameter("panel_height", 360)
        self.declare_parameter("refresh_hz", 10.0)
        self.declare_parameter("stale_panel_after_s", 1.5)

        self.create_subscription(
            Image,
            self.get_parameter("color_topic").value,
            self._image_cb("color"),
            10,
        )
        self.create_subscription(
            Image,
            self.get_parameter("yolo_topic").value,
            self._image_cb("yolo"),
            10,
        )
        self.create_subscription(
            Image,
            self.get_parameter("mask_topic").value,
            self._image_cb("mask"),
            10,
        )
        self.create_subscription(
            Image,
            self.get_parameter("foundationpose_topic").value,
            self._image_cb("foundationpose"),
            10,
        )
        self.create_subscription(
            Image,
            self.get_parameter("m3t_topic").value,
            self._image_cb("m3t"),
            10,
        )
        self.create_subscription(
            String,
            self.get_parameter("tracking_health_topic").value,
            self._tracking_status_cb,
            20,
        )
        self.pipeline_pub = self.create_publisher(
            Image,
            self.get_parameter("output_topic").value,
            10,
        )

        refresh_hz = max(1.0, float(self.get_parameter("refresh_hz").value))
        self.timer = self.create_timer(1.0 / refresh_hz, self._render)
        self.get_logger().info("Pipeline visualizer is ready")

    def _image_cb(self, key):
        def callback(msg):
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with self.lock:
                self.latest[key] = image.copy()
                self.latest_received_ns[key] = self.get_clock().now().nanoseconds

        return callback

    def _tracking_status_cb(self, msg):
        try:
            status = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warning(f"Ignoring invalid tracking health JSON: {exc}")
            return
        with self.lock:
            self.latest["tracking_status"] = status

    def _render(self):
        with self.lock:
            color = None if self.latest["color"] is None else self.latest["color"].copy()
            yolo = None if self.latest["yolo"] is None else self.latest["yolo"].copy()
            mask = None if self.latest["mask"] is None else self.latest["mask"].copy()
            foundationpose = (
                None
                if self.latest["foundationpose"] is None
                else self.latest["foundationpose"].copy()
            )
            m3t = None if self.latest["m3t"] is None else self.latest["m3t"].copy()
            tracking_status = self.latest["tracking_status"]
            received_ns = dict(self.latest_received_ns)

        if yolo is None:
            yolo = color
        now_ns = self.get_clock().now().nanoseconds
        panel_statuses = self._panel_statuses(now_ns, received_ns)
        panel_size = (
            int(self.get_parameter("panel_width").value),
            int(self.get_parameter("panel_height").value),
        )
        view = compose_pipeline_view(
            yolo=yolo,
            mask=mask,
            foundationpose=foundationpose,
            m3t=m3t,
            panel_size=panel_size,
            tracking_status=tracking_status,
            panel_statuses=panel_statuses,
        )
        msg = self.bridge.cv2_to_imgmsg(view, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        self.pipeline_pub.publish(msg)

        if bool(self.get_parameter("show_window").value):
            cv2.imshow(self.get_parameter("window_name").value, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                self.get_logger().info("Pipeline visualizer window requested shutdown")
                rclpy.shutdown()

    def _panel_statuses(self, now_ns, received_ns):
        stale_after_s = max(0.1, float(self.get_parameter("stale_panel_after_s").value))
        labels = {
            "mask": "waiting for FoundationPose request",
            "foundationpose": "waiting for accepted init",
            "m3t": "waiting for supervisor start",
        }
        statuses = {}
        for key, waiting_label in labels.items():
            stamp_ns = int(received_ns.get(key, 0))
            if stamp_ns <= 0:
                statuses[key] = {"label": waiting_label, "stale": False}
                continue
            age_s = max(0.0, (now_ns - stamp_ns) / 1_000_000_000.0)
            if age_s > stale_after_s:
                statuses[key] = {"label": f"stale {age_s:.1f}s", "stale": True}
        return statuses


def main():
    rclpy.init()
    node = PipelineVisualizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
