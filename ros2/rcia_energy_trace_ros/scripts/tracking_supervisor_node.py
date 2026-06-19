#!/usr/bin/env python3
import json
import math
import os
import sys
import threading
from pathlib import Path

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Empty, String


PROJECT_ROOT = Path(
    os.environ.get(
        "RCIA_ENERGY_TRACE_ROOT",
        Path(__file__).resolve().parents[3],
    )
).resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.latest_rgbd_buffer import LatestRgbdBuffer  # noqa: E402
from scripts.linemod_silhouette_renderer import load_obj_mesh  # noqa: E402
from scripts.ros_rgbd_sync import stamp_to_nanoseconds  # noqa: E402
from scripts.initialization_gate import (  # noqa: E402
    InitializationGateParams,
    InitializationGateResult,
    validate_initialization_candidate,
)
from scripts.m3t_tracking_health_gate import (  # noqa: E402
    TrackingHealthGateParams,
    TrackingHealthGateResult,
    evaluate_m3t_tracking_health,
)
from scripts.simple_tracking_lifecycle import (  # noqa: E402
    LifecycleObservation,
    LifecycleParams,
    SimpleTrackingLifecycle,
    TRACKING,
)
from scripts.supervisor_messages import (  # noqa: E402
    candidate_matches_request,
    candidate_status_is_failure,
    make_init_request_json,
    parse_m3t_status_json,
)
from scripts.tracking_state_filter import (  # noqa: E402
    TrackingStateFilter,
    TrackingStateFilterParams,
    TrackingStateObservation,
)
from scripts.yolo_roi_gate import YoloRoiGateParams, validate_yolo_roi  # noqa: E402


def camera_info_to_matrix(msg):
    return np.array(msg.k, dtype=np.float64).reshape(3, 3)


def pose_msg_to_matrix(msg):
    qx = float(msg.pose.orientation.x)
    qy = float(msg.pose.orientation.y)
    qz = float(msg.pose.orientation.z)
    qw = float(msg.pose.orientation.w)
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 1e-12:
        raise ValueError("pose quaternion norm must be non-zero")
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )
    matrix[:3, 3] = [
        float(msg.pose.position.x),
        float(msg.pose.position.y),
        float(msg.pose.position.z),
    ]
    return matrix


def parse_yolo_detections(payload, target_class=None):
    data = json.loads(payload)
    detections = []
    for item in data.get("detections", []):
        class_id = int(item.get("class_id", -1))
        if target_class is not None and class_id != int(target_class):
            continue
        bbox = item.get("bbox_xyxy", [])
        if len(bbox) < 4:
            continue
        detections.append(
            {
                "class_id": class_id,
                "confidence": float(item.get("confidence", 0.0)),
                "bbox_xyxy": [float(v) for v in bbox[:4]],
            }
        )
    return int(data.get("stamp_ns", 0)), detections


def select_best_detection(detections):
    if not detections:
        return None
    return max(
        detections,
        key=lambda item: (
            float(item.get("confidence", 0.0)),
            _bbox_area(item.get("bbox_xyxy", [0, 0, 0, 0])),
        ),
    )


def _bbox_area(bbox):
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


class TrackingSupervisorNode(Node):
    def __init__(self):
        super().__init__("tracking_supervisor_node")
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.latest_camera_info = None
        self.latest_yolo = None
        self.latest_candidate_pose = None
        self.latest_candidate_status = None
        self.latest_m3t_status = {"state": "stopped", "stamp_ns": 0}
        self.latest_m3t_pose = None
        self.previous_valid_m3t_pose = None
        self.active_request = None
        self.request_counter = 0
        self.stop_in_flight = False

        self.declare_parameter("color_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter("yolo_detections_topic", "/energy/yolo_detections")
        self.declare_parameter("candidate_pose_topic", "/energy/foundationpose_candidate_pose")
        self.declare_parameter(
            "candidate_status_topic",
            "/energy/foundationpose_candidate_status",
        )
        self.declare_parameter("m3t_status_topic", "/energy/m3t_status")
        self.declare_parameter("m3t_pose_topic", "/energy/m3t_pose")
        self.declare_parameter("init_request_topic", "/energy/init_request")
        self.declare_parameter("initial_pose_topic", "/energy/initial_pose")
        self.declare_parameter("m3t_stop_topic", "/energy/m3t_stop")
        self.declare_parameter("trusted_pose_topic", "/energy/trusted_pose")
        self.declare_parameter("tracking_state_topic", "/energy/tracking_state")
        self.declare_parameter("tracking_health_topic", "/energy/tracking_health")
        self.declare_parameter("mesh_file", str(PROJECT_ROOT / "models/object/Str3D.obj"))
        self.declare_parameter("target_class", 0)
        self.declare_parameter("supervisor_rate_hz", 8.0)
        self.declare_parameter("sync_tolerance_ms", 15.0)
        self.declare_parameter("latest_frame_max_age_ms", 250.0)
        self.declare_parameter("max_yolo_age_ms", 500.0)
        self.declare_parameter("max_pose_age_ms", 250.0)
        self.declare_parameter("candidate_timeout_ms", 8000.0)
        self.declare_parameter("candidate_match_tolerance_ms", 80.0)
        self.declare_parameter("candidate_frame_tolerance_ms", 40.0)
        self.declare_parameter("rgbd_history_size", 240)
        self.declare_parameter("stable_yolo_frames", 3)
        self.declare_parameter("lost_frames", 6)
        self.declare_parameter("out_of_view_frames", 10)
        self.declare_parameter("max_candidate_age_ms", 10000.0)
        self.declare_parameter("enforce_candidate_freshness", False)
        self.declare_parameter("tracking_min_depth_consistency_ratio", 0.60)
        self.declare_parameter("min_rendered_overlap_pixels", 80)
        self.declare_parameter("min_yolo_projection_iou", 0.35)
        self.declare_parameter("max_projection_center_distance_ratio", 0.75)
        self.declare_parameter("min_projection_area_ratio", 0.25)
        self.declare_parameter("max_depth_residual_m", 0.02)
        self.declare_parameter("max_pose_jump_translation_m", 0.08)
        self.declare_parameter("max_pose_jump_rotation_deg", 30.0)
        self.declare_parameter("require_smooth_pose", True)
        self.declare_parameter("depth_scale", 0.001)
        self.declare_parameter("preinit_roi_gate_enabled", False)
        self.declare_parameter("preinit_min_valid_depth_ratio", 0.20)
        self.declare_parameter("preinit_min_low_saturation_ratio", 0.45)
        self.declare_parameter("preinit_min_black_white_support", 0.20)
        self.declare_parameter("filter_initial_presence_score", 0.80)
        self.declare_parameter("filter_initial_health_score", 0.65)
        self.declare_parameter("filter_startup_grace_frames", 40)
        self.declare_parameter("filter_occlusion_grace_frames", 24)
        self.declare_parameter("filter_owned_confirm_frames", 3)
        self.declare_parameter("filter_recover_frames", 2)
        self.declare_parameter("filter_suspect_frames", 6)
        self.declare_parameter("filter_reinit_conflict_frames", 5)
        self.declare_parameter("filter_bad_tracking_frames", 5)
        self.declare_parameter("filter_out_of_view_frames", 12)
        self.declare_parameter("filter_min_rendered_pixels_for_owned", 40)
        self.declare_parameter("filter_locked_health_threshold", 0.70)
        self.declare_parameter("filter_suspect_health_threshold", 0.40)
        self.declare_parameter("filter_degraded_health_threshold", 0.40)
        self.declare_parameter("filter_degraded_presence_threshold", 0.55)
        self.declare_parameter("filter_lost_health_threshold", 0.35)
        self.declare_parameter("filter_lost_presence_threshold", 0.35)

        self.vertices, self.faces = load_obj_mesh(self.get_parameter("mesh_file").value)
        self.latest_buffer = LatestRgbdBuffer(
            color_history_size=int(self.get_parameter("rgbd_history_size").value),
            depth_history_size=int(self.get_parameter("rgbd_history_size").value),
        )
        self.lifecycle = SimpleTrackingLifecycle(
            LifecycleParams(
                stable_yolo_frames=int(self.get_parameter("stable_yolo_frames").value),
                lost_frames=int(self.get_parameter("lost_frames").value),
                out_of_view_frames=int(self.get_parameter("out_of_view_frames").value),
            )
        )
        self.tracking_health_params = TrackingHealthGateParams(
            min_rendered_overlap_pixels=int(
                self.get_parameter("min_rendered_overlap_pixels").value
            ),
            min_depth_consistency_ratio=float(
                self.get_parameter("tracking_min_depth_consistency_ratio").value
            ),
            max_depth_residual_m=float(self.get_parameter("max_depth_residual_m").value),
            min_projection_iou=float(
                self.get_parameter("min_yolo_projection_iou").value
            ),
            max_projection_center_distance_ratio=float(
                self.get_parameter("max_projection_center_distance_ratio").value
            ),
            min_projection_area_ratio=float(
                self.get_parameter("min_projection_area_ratio").value
            ),
            max_pose_jump_translation_m=float(
                self.get_parameter("max_pose_jump_translation_m").value
            ),
            max_pose_jump_rotation_deg=float(
                self.get_parameter("max_pose_jump_rotation_deg").value
            ),
            depth_scale=float(self.get_parameter("depth_scale").value),
            require_smooth_pose=bool(self.get_parameter("require_smooth_pose").value),
        )
        self.initialization_params = InitializationGateParams(
            max_candidate_age_ms=float(
                self.get_parameter("max_candidate_age_ms").value
            ),
            enforce_freshness=bool(
                self.get_parameter("enforce_candidate_freshness").value
            ),
        )
        self.yolo_roi_gate_params = YoloRoiGateParams(
            min_valid_depth_ratio=float(
                self.get_parameter("preinit_min_valid_depth_ratio").value
            ),
            min_low_saturation_ratio=float(
                self.get_parameter("preinit_min_low_saturation_ratio").value
            ),
            min_black_white_support=float(
                self.get_parameter("preinit_min_black_white_support").value
            ),
        )
        self.tracking_state_filter = TrackingStateFilter(
            TrackingStateFilterParams(
                initial_presence_score=float(
                    self.get_parameter("filter_initial_presence_score").value
                ),
                initial_health_score=float(
                    self.get_parameter("filter_initial_health_score").value
                ),
                startup_grace_frames=int(
                    self.get_parameter("filter_startup_grace_frames").value
                ),
                occlusion_grace_frames=int(
                    self.get_parameter("filter_occlusion_grace_frames").value
                ),
                owned_confirm_frames=int(
                    self.get_parameter("filter_owned_confirm_frames").value
                ),
                recover_frames=int(
                    self.get_parameter("filter_recover_frames").value
                ),
                suspect_frames=int(
                    self.get_parameter("filter_suspect_frames").value
                ),
                reinit_conflict_frames=int(
                    self.get_parameter("filter_reinit_conflict_frames").value
                ),
                bad_tracking_frames=int(
                    self.get_parameter("filter_bad_tracking_frames").value
                ),
                out_of_view_frames=int(
                    self.get_parameter("filter_out_of_view_frames").value
                ),
                min_rendered_pixels_for_owned=int(
                    self.get_parameter("filter_min_rendered_pixels_for_owned").value
                ),
                locked_health_threshold=float(
                    self.get_parameter("filter_locked_health_threshold").value
                ),
                suspect_health_threshold=float(
                    self.get_parameter("filter_suspect_health_threshold").value
                ),
                degraded_health_threshold=float(
                    self.get_parameter("filter_degraded_health_threshold").value
                ),
                degraded_presence_threshold=float(
                    self.get_parameter("filter_degraded_presence_threshold").value
                ),
                lost_health_threshold=float(
                    self.get_parameter("filter_lost_health_threshold").value
                ),
                lost_presence_threshold=float(
                    self.get_parameter("filter_lost_presence_threshold").value
                ),
                max_fresh_pose_age_ms=float(
                    self.get_parameter("max_pose_age_ms").value
                ),
                min_rendered_pixels=int(
                    self.get_parameter("min_rendered_overlap_pixels").value
                ),
            )
        )

        self.init_request_pub = self.create_publisher(
            String, self.get_parameter("init_request_topic").value, 10
        )
        self.initial_pose_pub = self.create_publisher(
            PoseStamped, self.get_parameter("initial_pose_topic").value, 10
        )
        self.stop_pub = self.create_publisher(
            Empty, self.get_parameter("m3t_stop_topic").value, 10
        )
        self.trusted_pose_pub = self.create_publisher(
            PoseStamped, self.get_parameter("trusted_pose_topic").value, 10
        )
        self.state_pub = self.create_publisher(
            String, self.get_parameter("tracking_state_topic").value, 20
        )
        self.health_pub = self.create_publisher(
            String, self.get_parameter("tracking_health_topic").value, 20
        )

        self.create_subscription(Image, self.get_parameter("color_topic").value, self._color_cb, 2)
        self.create_subscription(Image, self.get_parameter("depth_topic").value, self._depth_cb, 2)
        self.create_subscription(
            CameraInfo,
            self.get_parameter("camera_info_topic").value,
            self._camera_info_cb,
            10,
        )
        self.create_subscription(
            String,
            self.get_parameter("yolo_detections_topic").value,
            self._yolo_cb,
            20,
        )
        self.create_subscription(
            PoseStamped,
            self.get_parameter("candidate_pose_topic").value,
            self._candidate_pose_cb,
            10,
        )
        self.create_subscription(
            String,
            self.get_parameter("candidate_status_topic").value,
            self._candidate_status_cb,
            10,
        )
        self.create_subscription(
            String,
            self.get_parameter("m3t_status_topic").value,
            self._m3t_status_cb,
            20,
        )
        self.create_subscription(
            PoseStamped,
            self.get_parameter("m3t_pose_topic").value,
            self._m3t_pose_cb,
            20,
        )

        rate_hz = max(1.0, float(self.get_parameter("supervisor_rate_hz").value))
        self.timer = self.create_timer(1.0 / rate_hz, self._tick)
        self.get_logger().info("Tracking supervisor is ready")

    def _now_ns(self):
        return int(self.get_clock().now().nanoseconds)

    def _color_cb(self, msg):
        stamp_ns = stamp_to_nanoseconds(msg.header.stamp)
        color = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        with self.lock:
            self.latest_buffer.put_color(stamp_ns, color)

    def _depth_cb(self, msg):
        stamp_ns = stamp_to_nanoseconds(msg.header.stamp)
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        with self.lock:
            self.latest_buffer.put_depth(stamp_ns, depth)

    def _camera_info_cb(self, msg):
        with self.lock:
            self.latest_camera_info = msg

    def _yolo_cb(self, msg):
        target_class = int(self.get_parameter("target_class").value)
        target_class = None if target_class < 0 else target_class
        try:
            stamp_ns, detections = parse_yolo_detections(msg.data, target_class)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self.get_logger().warning(f"Ignoring invalid yolo detections: {exc}")
            return
        with self.lock:
            self.latest_yolo = {
                "stamp_ns": stamp_ns,
                "detections": detections,
                "selected": select_best_detection(detections),
            }

    def _candidate_pose_cb(self, msg):
        try:
            matrix = pose_msg_to_matrix(msg)
        except ValueError as exc:
            self.get_logger().warning(f"Ignoring invalid candidate pose: {exc}")
            return
        with self.lock:
            self.latest_candidate_pose = {
                "stamp_ns": stamp_to_nanoseconds(msg.header.stamp),
                "matrix": matrix,
                "msg": msg,
            }

    def _candidate_status_cb(self, msg):
        try:
            status = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warning(f"Ignoring invalid candidate status: {exc}")
            return
        with self.lock:
            self.latest_candidate_status = status

    def _m3t_status_cb(self, msg):
        try:
            status = parse_m3t_status_json(msg.data)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self.get_logger().warning(f"Ignoring invalid M3T status: {exc}")
            return
        with self.lock:
            self.latest_m3t_status = status
            if status.get("state") == "stopped":
                self.stop_in_flight = False

    def _m3t_pose_cb(self, msg):
        try:
            matrix = pose_msg_to_matrix(msg)
        except ValueError as exc:
            self.get_logger().warning(f"Ignoring invalid M3T pose: {exc}")
            return
        with self.lock:
            self.latest_m3t_pose = {
                "stamp_ns": stamp_to_nanoseconds(msg.header.stamp),
                "matrix": matrix,
                "msg": msg,
            }

    def _latest_inputs(self):
        now_ns = self._now_ns()
        tolerance_ns = int(float(self.get_parameter("sync_tolerance_ms").value) * 1_000_000)
        max_age_ns = int(float(self.get_parameter("latest_frame_max_age_ms").value) * 1_000_000)
        with self.lock:
            pair = self.latest_buffer.get_latest_pair(now_ns, tolerance_ns, max_age_ns)
            return {
                "now_ns": now_ns,
                "pair": pair,
                "camera_info": self.latest_camera_info,
                "yolo": None if self.latest_yolo is None else dict(self.latest_yolo),
                "candidate_pose": self.latest_candidate_pose,
                "candidate_status": self.latest_candidate_status,
                "m3t_status": dict(self.latest_m3t_status),
                "m3t_pose": self.latest_m3t_pose,
                "active_request": None if self.active_request is None else dict(self.active_request),
            }

    def _tick(self):
        inputs = self._latest_inputs()
        raw_yolo_bbox, raw_yolo_visible, yolo_conf = self._active_yolo(inputs)
        yolo_gate = self._preinit_yolo_gate(inputs, raw_yolo_bbox)
        yolo_visible = raw_yolo_visible and yolo_gate.success
        yolo_bbox = raw_yolo_bbox if yolo_visible else None
        tracking_yolo_bbox = raw_yolo_bbox if raw_yolo_visible else None
        candidate_result, candidate_available, candidate_pose_msg = self._candidate_validation(
            inputs, yolo_bbox
        )
        tracking_health, m3t_pose_available = self._m3t_health(inputs, tracking_yolo_bbox)
        tracking_filter_decision = self._tracking_filter_decision(
            inputs=inputs,
            raw_yolo_visible=raw_yolo_visible,
            yolo_conf=yolo_conf,
            tracking_health=tracking_health,
            m3t_pose_available=m3t_pose_available,
        )
        candidate_timed_out = self._candidate_timed_out(inputs["now_ns"], inputs["active_request"])
        observation = LifecycleObservation(
            yolo_visible=yolo_visible,
            candidate_available=candidate_available,
            candidate_valid=candidate_result.success,
            candidate_timed_out=candidate_timed_out,
            m3t_status=inputs["m3t_status"].get("state", ""),
            m3t_pose_available=m3t_pose_available,
            tracking_health_ok=tracking_health.tracking_ok,
            tracking_reliability_state=(
                "" if tracking_filter_decision is None else tracking_filter_decision.reliability_state
            ),
            filter_publish_trusted_pose=(
                False if tracking_filter_decision is None else tracking_filter_decision.publish_trusted_pose
            ),
            filter_keep_m3t_alive=(
                False if tracking_filter_decision is None else tracking_filter_decision.keep_m3t_alive
            ),
            filter_request_reinit=(
                False if tracking_filter_decision is None else tracking_filter_decision.request_reinit
            ),
            filter_stop_m3t=(
                False if tracking_filter_decision is None else tracking_filter_decision.stop_m3t
            ),
        )
        decision = self.lifecycle.update(observation)
        self._log_candidate_decision(decision, candidate_result)
        self._apply_decision(
            decision=decision,
            now_ns=inputs["now_ns"],
            yolo_bbox=yolo_bbox,
            yolo_conf=yolo_conf,
            candidate_pose_msg=candidate_pose_msg,
            m3t_pose=inputs["m3t_pose"],
        )
        self._publish_state(decision, inputs["now_ns"])
        self._publish_health(
            decision,
            candidate_result,
            tracking_health,
            tracking_filter_decision,
            yolo_gate,
            inputs["now_ns"],
        )
        if (
            m3t_pose_available
            and tracking_filter_decision is not None
            and tracking_filter_decision.publish_trusted_pose
        ):
            self.previous_valid_m3t_pose = inputs["m3t_pose"]["matrix"]

    def _log_candidate_decision(self, decision, candidate_result):
        if decision.action == "reject_candidate":
            self.get_logger().warning(
                "Candidate rejected: "
                f"pose_valid={candidate_result.pose_valid} "
                f"pose_age_ms={candidate_result.pose_age_ms:.1f} "
                f"reason={candidate_result.reason}"
            )
        elif decision.publish_initial_pose:
            self.get_logger().info(
                "Candidate accepted for M3T: "
                f"pose_age_ms={candidate_result.pose_age_ms:.1f}"
            )

    def _active_yolo(self, inputs):
        yolo = inputs["yolo"]
        if not yolo or not yolo.get("selected"):
            return None, False, 0.0
        age_ms = (inputs["now_ns"] - int(yolo.get("stamp_ns", 0))) / 1_000_000.0
        if age_ms > float(self.get_parameter("max_yolo_age_ms").value):
            return None, False, float(yolo["selected"].get("confidence", 0.0))
        return (
            yolo["selected"]["bbox_xyxy"],
            True,
            float(yolo["selected"].get("confidence", 0.0)),
        )

    def _preinit_yolo_gate(self, inputs, yolo_bbox):
        if yolo_bbox is None:
            return validate_yolo_roi(None, None, None, self.yolo_roi_gate_params)
        if not bool(self.get_parameter("preinit_roi_gate_enabled").value):
            result = validate_yolo_roi(None, None, None, self.yolo_roi_gate_params)
            return result.__class__(
                success=True,
                valid_depth_ratio=0.0,
                low_saturation_ratio=0.0,
                black_white_support=0.0,
                roi_area_px=0,
                reason="preinit ROI gate disabled",
            )
        pair = inputs["pair"]
        if pair is None:
            return validate_yolo_roi(None, None, yolo_bbox, self.yolo_roi_gate_params)
        return validate_yolo_roi(
            color_bgr=pair.color,
            depth_image=pair.depth,
            bbox_xyxy=yolo_bbox,
            params=self.yolo_roi_gate_params,
        )

    def _candidate_validation(self, inputs, yolo_bbox):
        request = inputs["active_request"]
        candidate_pose = inputs["candidate_pose"]
        candidate_status = inputs["candidate_status"]
        if request is None:
            return self._candidate_result(False, "waiting for candidate"), False, None
        if (
            candidate_pose is None
            and candidate_status is not None
            and candidate_matches_request(
                request,
                candidate_status,
                max_stamp_delta_ms=float(
                    self.get_parameter("candidate_match_tolerance_ms").value
                ),
            )
            and candidate_status_is_failure(candidate_status)
        ):
            return (
                self._candidate_result(
                    False,
                    str(candidate_status.get("error", candidate_status.get("event", "candidate failed"))),
                ),
                True,
                None,
            )
        if candidate_pose is None:
            return self._candidate_result(False, "waiting for candidate"), False, None
        if not candidate_matches_request(
            request,
            candidate_status or {"frame_stamp_ns": candidate_pose["stamp_ns"]},
            max_stamp_delta_ms=float(self.get_parameter("candidate_match_tolerance_ms").value),
        ):
            return self._candidate_result(False, "candidate does not match request"), False, None
        result = validate_initialization_candidate(
            candidate_pose=candidate_pose["matrix"],
            now_ns=inputs["now_ns"],
            candidate_stamp_ns=candidate_pose["stamp_ns"],
            candidate_status=candidate_status,
            params=self.initialization_params,
        )
        return result, True, candidate_pose["msg"]

    def _candidate_rgbd_pair(self, inputs, candidate_status, candidate_pose):
        frame_stamp_ns = 0
        if isinstance(candidate_status, dict):
            frame_stamp_ns = int(candidate_status.get("frame_stamp_ns", 0) or 0)
        if frame_stamp_ns <= 0:
            frame_stamp_ns = int(candidate_pose.get("stamp_ns", 0) or 0)
        tolerance_ns = int(
            float(self.get_parameter("candidate_frame_tolerance_ms").value) * 1_000_000
        )
        if frame_stamp_ns > 0:
            with self.lock:
                historical_pair = self.latest_buffer.get_pair_at(frame_stamp_ns, tolerance_ns)
            if historical_pair is not None:
                return historical_pair
        return inputs["pair"]

    def _m3t_health(self, inputs, yolo_bbox):
        m3t_pose = inputs["m3t_pose"]
        if m3t_pose is None:
            return self._empty_health("missing M3T pose", yolo_bbox), False
        age_ms = (inputs["now_ns"] - int(m3t_pose["stamp_ns"])) / 1_000_000.0
        if age_ms > float(self.get_parameter("max_pose_age_ms").value):
            return self._empty_health("stale M3T pose", yolo_bbox), False
        pair = inputs["pair"]
        camera_info = inputs["camera_info"]
        if pair is None or camera_info is None:
            return self._empty_health("missing RGB-D/camera_info", yolo_bbox), True
        try:
            return (
                evaluate_m3t_tracking_health(
                    vertices=self.vertices,
                    faces=self.faces,
                    object_in_camera=m3t_pose["matrix"],
                    previous_pose=self.previous_valid_m3t_pose,
                    k_matrix=camera_info_to_matrix(camera_info),
                    color_bgr=pair.color,
                    observed_depth_image=pair.depth,
                    yolo_bbox=yolo_bbox,
                    params=self.tracking_health_params,
                ),
                True,
            )
        except Exception as exc:  # noqa: BLE001
            return self._empty_health(f"validation error: {exc}", yolo_bbox), True

    def _tracking_filter_decision(
        self,
        inputs,
        raw_yolo_visible,
        yolo_conf,
        tracking_health,
        m3t_pose_available,
    ):
        if self.lifecycle.state != TRACKING:
            return None
        m3t_pose_age_ms = 0.0
        if inputs["m3t_pose"] is not None:
            m3t_pose_age_ms = (
                inputs["now_ns"] - int(inputs["m3t_pose"].get("stamp_ns", inputs["now_ns"]))
            ) / 1_000_000.0
        observation = TrackingStateObservation(
            yolo_visible=bool(raw_yolo_visible),
            yolo_confidence=float(yolo_conf),
            m3t_status=str(inputs["m3t_status"].get("state", "")),
            m3t_pose_available=bool(m3t_pose_available),
            m3t_pose_age_ms=float(m3t_pose_age_ms),
            depth_ok=bool(tracking_health.depth_ok),
            depth_ratio=float(tracking_health.depth_ratio),
            black_white_ok=bool(tracking_health.black_white_ok),
            projection_ok=bool(tracking_health.projection_ok),
            projection_center_ok=bool(tracking_health.projection_center_ok),
            projection_area_ok=bool(tracking_health.projection_area_ok),
            smoothness_ok=bool(tracking_health.smoothness_ok),
            rendered_pixel_count=int(tracking_health.rendered_pixel_count),
        )
        return self.tracking_state_filter.update(observation)

    def _apply_decision(
        self,
        decision,
        now_ns,
        yolo_bbox,
        yolo_conf,
        candidate_pose_msg,
        m3t_pose,
    ):
        if decision.publish_init_request and yolo_bbox is not None:
            self._publish_init_request(now_ns, yolo_bbox, yolo_conf, decision.reason)
        if decision.clear_candidate:
            self._clear_active_request()
        if decision.publish_stop and not self.stop_in_flight:
            self.stop_pub.publish(Empty())
            self.stop_in_flight = True
            self.tracking_state_filter.reset_to_search()
        if decision.publish_initial_pose and candidate_pose_msg is not None:
            self.initial_pose_pub.publish(candidate_pose_msg)
            self.tracking_state_filter.reset_for_tracking()
            self._clear_active_request()
        if decision.publish_trusted_pose and m3t_pose is not None:
            self.trusted_pose_pub.publish(m3t_pose["msg"])

    def _publish_init_request(self, now_ns, yolo_bbox, yolo_conf, reason):
        if self.active_request is not None:
            return
        self.request_counter += 1
        request_id = f"supervisor-{self.request_counter}"
        target_class = int(self.get_parameter("target_class").value)
        payload = make_init_request_json(
            request_id=request_id,
            stamp_ns=now_ns,
            target_class=target_class,
            bbox_xyxy=yolo_bbox,
            confidence=yolo_conf,
            reason=reason,
        )
        self.active_request = json.loads(payload)
        self.active_request["request_sent_ns"] = now_ns
        msg = String()
        msg.data = payload
        self.init_request_pub.publish(msg)
        self.get_logger().warning(
            "Published FoundationPose init request: "
            f"request_id={request_id} bbox={self.active_request.get('bbox_xyxy')} "
            f"confidence={yolo_conf:.3f} reason={reason}"
        )

    def _clear_active_request(self):
        with self.lock:
            self.active_request = None
            self.latest_candidate_pose = None
            self.latest_candidate_status = None

    def _candidate_timed_out(self, now_ns, request):
        if request is None:
            return False
        sent_ns = int(request.get("request_sent_ns", request.get("stamp_ns", now_ns)))
        age_ms = (now_ns - sent_ns) / 1_000_000.0
        return age_ms > float(self.get_parameter("candidate_timeout_ms").value)

    def _publish_state(self, decision, now_ns):
        payload = {
            "stamp_ns": int(now_ns),
            "state": decision.state,
            "action": decision.action,
            "trusted": bool(decision.publish_trusted_pose),
            "reason": decision.reason,
        }
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        self.state_pub.publish(msg)

    def _publish_health(
        self,
        decision,
        candidate_result,
        m3t_health,
        tracking_filter_decision,
        yolo_gate,
        now_ns,
    ):
        payload = {
            "stamp_ns": int(now_ns),
            "state": decision.state,
            "action": decision.action,
            "trusted": bool(decision.publish_trusted_pose),
            "preinit_yolo_gate": yolo_gate.to_dict(),
            "candidate": candidate_result.to_dict(),
            "m3t": m3t_health.to_dict(),
            "tracking_filter": (
                None
                if tracking_filter_decision is None
                else tracking_filter_decision.to_dict()
            ),
        }
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        self.health_pub.publish(msg)

    @staticmethod
    def _candidate_result(success, reason):
        return InitializationGateResult(
            success=success,
            pose_valid=False,
            pose_age_ms=0.0,
            reason=reason,
        )

    @staticmethod
    def _empty_health(reason, yolo_bbox):
        return TrackingHealthGateResult(
            tracking_ok=False,
            depth_ok=False,
            projection_ok=yolo_bbox is None,
            projection_center_ok=yolo_bbox is None,
            projection_area_ok=yolo_bbox is None,
            black_white_ok=False,
            smoothness_ok=True,
            depth_ratio=0.0,
            projection_iou=0.0,
            projection_center_distance_ratio=0.0 if yolo_bbox is None else float("inf"),
            projection_area_ratio=1.0 if yolo_bbox is None else 0.0,
            translation_jump_m=0.0,
            rotation_jump_deg=0.0,
            rendered_pixel_count=0,
            reason=reason,
        )


def main():
    rclpy.init()
    node = TrackingSupervisorNode()
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
