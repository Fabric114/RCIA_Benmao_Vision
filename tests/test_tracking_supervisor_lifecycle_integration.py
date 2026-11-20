"""Drive the real supervisor tick with real callbacks, lifecycle and tracking filter.

Only the renderer (evaluate_m3t_tracking_health), the ROS clock and the
monotonic clock are replaced. Assertions are on the messages the node publishes.
"""

import importlib.util
import inspect
import json
from pathlib import Path
from types import MethodType, SimpleNamespace
import threading
import unittest
from unittest import mock

import numpy as np

from scripts.initialization_gate import InitializationGateParams
from scripts.m3t_tracking_health_gate import (
    TrackingHealthGateParams,
    TrackingHealthGateResult,
)
from scripts.simple_tracking_lifecycle import (
    LifecycleParams,
    SimpleTrackingLifecycle,
    TRACKING,
)
from scripts.tracking_state_filter import TrackingStateFilter, TrackingStateFilterParams
from scripts.yolo_roi_gate import YoloRoiGateParams


MS = 1_000_000
WALL0 = 1_700_000_000_000_000_000
BBOX = [100.0, 100.0, 200.0, 200.0]


def load_tracking_supervisor_node():
    path = (
        Path(__file__).resolve().parents[1]
        / "ros2/rcia_energy_trace_ros/scripts/tracking_supervisor_node.py"
    )
    spec = importlib.util.spec_from_file_location("tracking_supervisor_node_it", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def good_health():
    return TrackingHealthGateResult(
        tracking_ok=True,
        depth_ok=True,
        projection_ok=True,
        projection_center_ok=True,
        projection_area_ok=True,
        black_white_ok=True,
        smoothness_ok=True,
        depth_ratio=0.9,
        projection_iou=0.8,
        projection_center_distance_ratio=0.1,
        projection_area_ratio=0.9,
        translation_jump_m=0.0,
        rotation_jump_deg=0.0,
        rendered_pixel_count=8000,
        reason="",
        health_evaluated=True,
        valid_depth_overlap_count=8000,
        depth_evaluable=True,
    )


def make_pair(stamp_ns):
    return SimpleNamespace(
        color_stamp_ns=int(stamp_ns),
        depth_stamp_ns=int(stamp_ns),
        color=np.zeros((4, 4, 3), dtype=np.uint8),
        depth=np.zeros((4, 4), dtype=np.uint16),
    )


def pose_msg(stamp_ns):
    return SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(
                sec=stamp_ns // 1_000_000_000, nanosec=stamp_ns % 1_000_000_000
            ),
            frame_id="camera_color_optical_frame",
        ),
        pose=SimpleNamespace(
            position=SimpleNamespace(x=0.0, y=0.0, z=0.5),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        ),
    )


def json_msg(payload):
    return SimpleNamespace(data=json.dumps(payload))


class Harness:
    def __init__(self, module):
        self.module = module
        self.clock = {"mono": 10_000 * MS}
        self.logger = mock.Mock()
        params = {
            "target_class": 0,
            "max_yolo_age_ms": 500.0,
            "max_pose_age_ms": 250.0,
            "sync_tolerance_ms": 15.0,
            "latest_frame_max_age_ms": 250.0,
            "tracking_pose_frame_tolerance_ms": 15.0,
            "tracking_yolo_frame_tolerance_ms": 80.0,
            "candidate_match_tolerance_ms": 80.0,
            "candidate_timeout_ms": 8000.0,
            "preinit_roi_gate_enabled": False,
            "m3t_stop_ack_timeout_ms": 2000.0,
        }
        self.node = self._make_node(params)
        self.module_time = SimpleNamespace(monotonic_ns=lambda: self.clock["mono"])
        self.evaluate_patch = mock.patch.object(
            module, "evaluate_m3t_tracking_health", return_value=good_health()
        )
        self.time_patch = mock.patch.object(module, "time", self.module_time)

    def start(self):
        self.evaluate_patch.start()
        self.time_patch.start()

    def stop(self):
        self.time_patch.stop()
        self.evaluate_patch.stop()

    @property
    def now_ns(self):
        return WALL0 + self.clock["mono"]

    def advance(self, ms):
        self.clock["mono"] += int(ms * MS)

    def tick(self, ms=125):
        self.advance(ms)
        self.node._yolo_cb(self.yolo_msg())
        self.node._tick()

    def yolo_msg(self, detected=True):
        detections = (
            [{"class_id": 0, "confidence": 0.9, "bbox_xyxy": BBOX}] if detected else []
        )
        return json_msg(
            {"stamp_ns": self.now_ns, "detections": detections}
        )

    def status(self, state, initial_stamp=None, **extra):
        payload = {"state": state, "stamp_ns": self.now_ns}
        if initial_stamp is not None:
            payload["initial_pose_stamp_ns"] = int(initial_stamp)
        payload.update(extra)
        self.node._m3t_status_cb(json_msg(payload))

    def feed_pose(self, age_ms=10):
        self.node._m3t_pose_cb(pose_msg(self.now_ns - int(age_ms * MS)))

    def _make_node(self, params):
        module = self.module
        self_ref = self
        node = SimpleNamespace(
            get_parameter=lambda name: SimpleNamespace(value=params[name]),
            get_logger=lambda: self_ref.logger,
            _now_ns=lambda: self_ref.now_ns,
            lock=threading.Lock(),
            bridge=mock.Mock(
                imgmsg_to_cv2=mock.Mock(return_value=np.zeros((4, 4, 3), np.uint8))
            ),
            latest_buffer=mock.Mock(),
            latest_camera_info=SimpleNamespace(k=[1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0]),
            latest_yolo=None,
            latest_candidate_pose=None,
            latest_candidate_status=None,
            latest_m3t_status={"state": "stopped", "stamp_ns": 0},
            latest_m3t_pose=None,
            previous_valid_m3t_pose=None,
            active_request=None,
            request_counter=0,
            stop_in_flight=False,
            session_initial_pose_stamp_ns=0,
            stop_sent_mono_ns=None,
            stop_sent_status_seq=0,
            m3t_status_seq=0,
            stop_ack_timeout_logged=False,
            health_cache=None,
            last_pose_color_stamp_ns=0,
            last_color_stamp_ns=None,
            camera_time_reversed=False,
            vertices=None,
            faces=None,
            tracking_health_params=TrackingHealthGateParams(),
            initialization_params=InitializationGateParams(),
            yolo_roi_gate_params=YoloRoiGateParams(),
            yaw_canonicalizer=None,
            lifecycle=SimpleTrackingLifecycle(
                LifecycleParams(stable_yolo_frames=2, lost_frames=3, out_of_view_frames=2)
            ),
            tracking_state_filter=TrackingStateFilter(TrackingStateFilterParams()),
            init_request_pub=mock.Mock(),
            initial_pose_pub=mock.Mock(),
            stop_pub=mock.Mock(),
            trusted_pose_pub=mock.Mock(),
            state_pub=mock.Mock(),
            health_pub=mock.Mock(),
        )
        node.latest_buffer.get_latest_pair.return_value = None
        node.latest_buffer.get_pair_at.side_effect = lambda stamp, tol: make_pair(stamp)
        # Bind only methods the node class defines itself. Inherited rclpy methods
        # such as get_parameter must keep the fake's versions.
        for name, attr in vars(module.TrackingSupervisorNode).items():
            if name.startswith("__"):
                continue
            if isinstance(attr, staticmethod):
                setattr(node, name, attr.__func__)
            elif inspect.isfunction(attr):
                setattr(node, name, MethodType(attr, node))
        # The class defines _now_ns itself, so it is rebound above. Restore the
        # simulated clock so the test controls wall time.
        node._now_ns = lambda: self_ref.now_ns
        return node

    def start_session(self):
        """Drive YOLO to a candidate request, then accept the candidate.

        Returns the initial pose stamp of the new session.
        """
        self.tick()
        self.tick()
        request_id = self.node.active_request["request_id"]
        stamp = self.now_ns
        self.node._candidate_pose_cb(pose_msg(stamp))
        self.node._candidate_status_cb(
            json_msg({"request_id": request_id, "frame_stamp_ns": stamp})
        )
        self.tick()
        return stamp


class TrackingSupervisorLifecycleIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_tracking_supervisor_node()

    def setUp(self):
        self.h = Harness(self.module)
        self.h.start()
        self.addCleanup(self.h.stop)
        self.node = self.h.node

    def test_candidate_starts_session_and_trusted_pose_waits_for_ownership(self):
        session_stamp = self.h.start_session()

        self.assertEqual(self.h.node.lifecycle.state, TRACKING)
        self.node.initial_pose_pub.publish.assert_called_once()
        self.assertEqual(self.node.session_initial_pose_stamp_ns, session_stamp)

        self.h.status("starting", initial_stamp=session_stamp)
        self.h.feed_pose()
        self.h.tick()
        self.node.trusted_pose_pub.publish.assert_not_called()

        self.h.status("tracking", initial_stamp=session_stamp)
        published_while_confirming = []
        for _ in range(6):
            self.h.feed_pose()
            self.h.tick()
            published_while_confirming.append(self.node.trusted_pose_pub.publish.call_count)

        # Ownership needs three new supporting samples, so the first tick publishes nothing.
        self.assertEqual(published_while_confirming[0], 0)
        self.assertGreater(published_while_confirming[-1], 0)
        self.node.stop_pub.publish.assert_not_called()

    def test_startup_timeout_sends_one_stop_and_no_trusted_pose(self):
        session_stamp = self.h.start_session()
        self.h.status("starting", initial_stamp=session_stamp)

        for _ in range(42):  # about 5.25 s at 125 ms per tick
            self.h.tick()

        self.node.stop_pub.publish.assert_called_once()
        self.node.trusted_pose_pub.publish.assert_not_called()
        self.assertEqual(self.node.lifecycle.state, "NO_TRACK")
        self.assertTrue(self.node.stop_in_flight)

    def test_no_new_initialization_until_stopped_then_one_request(self):
        session_stamp = self.h.start_session()
        self.h.status("starting", initial_stamp=session_stamp)
        for _ in range(42):
            self.h.tick()
        requests_before = self.node.init_request_pub.publish.call_count

        # Two seconds pass with the target visible and no stop acknowledgement.
        for _ in range(20):
            self.h.tick()
        self.assertEqual(self.node.init_request_pub.publish.call_count, requests_before)
        self.assertTrue(self.node.stop_in_flight)
        self.assertEqual(self.h.logger.error.call_count, 1)

        # A stopped status from a different session does not acknowledge our stop.
        self.h.status("stopped", initial_stamp=session_stamp - 1)
        self.assertTrue(self.node.stop_in_flight)

        self.h.status("stopped", initial_stamp=session_stamp)
        self.assertFalse(self.node.stop_in_flight)
        for _ in range(3):
            self.h.tick()
        self.assertEqual(self.node.init_request_pub.publish.call_count, requests_before + 1)

    def test_stale_stopped_status_before_the_stop_does_not_acknowledge_it(self):
        session_stamp = self.h.start_session()
        self.h.status("starting", initial_stamp=session_stamp)
        for _ in range(42):
            self.h.tick()
        self.assertTrue(self.node.stop_in_flight)

        # A stopped message accepted before the stop was sent has a lower sequence.
        self.node.stop_sent_status_seq = self.node.m3t_status_seq + 1
        self.h.status("stopped", initial_stamp=session_stamp)
        self.assertTrue(self.node.stop_in_flight)

    def test_camera_time_reversal_stops_the_session_once(self):
        session_stamp = self.h.start_session()
        self.h.status("tracking", initial_stamp=session_stamp)
        self.h.feed_pose()
        self.h.tick()
        self.assertEqual(self.node.lifecycle.state, TRACKING)

        later = SimpleNamespace(
            header=SimpleNamespace(
                stamp=SimpleNamespace(sec=self.h.now_ns // 1_000_000_000, nanosec=self.h.now_ns % 1_000_000_000)
            )
        )
        earlier = SimpleNamespace(
            header=SimpleNamespace(
                stamp=SimpleNamespace(
                    sec=(self.h.now_ns - 500 * MS) // 1_000_000_000,
                    nanosec=(self.h.now_ns - 500 * MS) % 1_000_000_000,
                )
            )
        )
        self.node._color_cb(later)
        self.node._color_cb(earlier)
        self.h.tick()
        self.h.tick()

        self.node.stop_pub.publish.assert_called_once()
        self.assertEqual(self.node.lifecycle.state, "NO_TRACK")


if __name__ == "__main__":
    unittest.main()
