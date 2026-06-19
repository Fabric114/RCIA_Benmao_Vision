#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


PROJECT_ROOT = Path(
    os.environ.get(
        "RCIA_ENERGY_TRACE_ROOT",
        Path(__file__).resolve().parents[3],
    )
).resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tracking_health_debug_analyzer import analyze_tracking_health  # noqa: E402


class TrackingHealthDebugNode(Node):
    def __init__(self):
        super().__init__("tracking_health_debug_node")
        self.declare_parameter("tracking_health_topic", "/energy/tracking_health")
        self.declare_parameter("debug_topic", "/energy/tracking_debug")
        self.declare_parameter("emit_all", False)
        self.previous_payload = None

        self.debug_pub = self.create_publisher(
            String,
            self.get_parameter("debug_topic").value,
            20,
        )
        self.create_subscription(
            String,
            self.get_parameter("tracking_health_topic").value,
            self._tracking_health_cb,
            50,
        )
        self.get_logger().info(
            "Tracking health debug node is ready: "
            f"input={self.get_parameter('tracking_health_topic').value} "
            f"output={self.get_parameter('debug_topic').value}"
        )

    def _tracking_health_cb(self, msg):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warning(f"Ignoring invalid tracking health JSON: {exc}")
            return

        result = analyze_tracking_health(payload, self.previous_payload)
        self.previous_payload = payload
        if not result.should_emit and not bool(self.get_parameter("emit_all").value):
            return

        output = {
            "stamp_ns": int(payload.get("stamp_ns", 0) or 0),
            **result.to_dict(),
        }
        out_msg = String()
        out_msg.data = json.dumps(output, ensure_ascii=False, sort_keys=True)
        self.debug_pub.publish(out_msg)
        self.get_logger().warning(
            result.summary + " | " + " | ".join(result.evidence)
        )


def main():
    rclpy.init()
    node = TrackingHealthDebugNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
