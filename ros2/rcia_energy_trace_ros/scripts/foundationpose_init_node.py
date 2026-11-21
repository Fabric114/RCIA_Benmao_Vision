#!/usr/bin/env python3
import os
import sys
import threading
import traceback
from pathlib import Path

import json
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String


PROJECT_ROOT = Path(
    os.environ.get(
        "RCIA_ENERGY_TRACE_ROOT",
        Path(__file__).resolve().parents[3],
    )
).resolve()
FOUNDATIONPOSE_DIR = PROJECT_ROOT / "foundation_pose"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(FOUNDATIONPOSE_DIR))

from scripts.auto_register import AutoRegisterGate  # noqa: E402
from scripts.diagnostic_visualization import (  # noqa: E402
    draw_yolo_detections,
    overlay_mask,
    save_debug_image,
)
from scripts.foundationpose_candidate_policy import FoundationPoseCandidatePolicy  # noqa: E402
from scripts.initial_pose_policy import InitialPosePublishPolicy  # noqa: E402
from scripts.latest_rgbd_buffer import LatestRgbdBuffer  # noqa: E402
from scripts.pose_end_disambiguation import (  # noqa: E402
    disambiguate_zone_pose,
    mesh_endpoints_from_vertices,
)
from scripts.ros_pose_utils import matrix_to_pose_components  # noqa: E402
from scripts.ros_rgbd_sync import stamp_to_nanoseconds  # noqa: E402
from scripts.timing_diagnostics import make_timing_event  # noqa: E402
from scripts.yaw_canonicalization import canonicalize_yaw  # noqa: E402
from scripts.yolo_detection_message import detections_to_json  # noqa: E402
from scripts.foundationpose_init import MaskGenerator  # noqa: E402
from scripts.foundationpose_factory import EstimatorConfig, build_estimator  # noqa: E402
from Utils import (  # noqa: E402
    draw_posed_3d_box,
    draw_xyz_axis,
)


def camera_info_to_matrix(msg):
    return np.array(msg.k, dtype=np.float64).reshape(3, 3)


def depth_image_to_meters(depth):
    if depth.dtype == np.uint16:
        converted = depth.astype(np.float32) / 1000.0
    else:
        converted = depth.astype(np.float32)
        if converted.size and converted.max() > 100.0:
            converted /= 1000.0
    converted[(converted < 0.001) | (~np.isfinite(converted))] = 0
    return converted


class FoundationPoseInitNode(Node):
    def __init__(self):
        super().__init__("foundationpose_init_node")
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.latest_camera_info = None
        self.initialized = False
        self.active_init_request = None

        self.declare_parameter("color_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter("initial_pose_topic", "/energy/initial_pose")
        self.declare_parameter("init_request_topic", "/energy/init_request")
        self.declare_parameter("reinit_request_topic", "/energy/reinit_request")
        self.declare_parameter(
            "candidate_pose_topic",
            "/energy/foundationpose_candidate_pose",
        )
        self.declare_parameter(
            "candidate_status_topic",
            "/energy/foundationpose_candidate_status",
        )
        self.declare_parameter("direct_initial_pose_enabled", False)
        self.declare_parameter("request_required", True)
        self.declare_parameter("output_frame_id", "camera_color_optical_frame")
        self.declare_parameter("mesh_file", str(PROJECT_ROOT / "models/object/Str3D.obj"))
        self.declare_parameter("yolo_model", str(PROJECT_ROOT / "models/yolo/1117_openvino_model"))
        self.declare_parameter("mask_center_frac", 0.35)
        self.declare_parameter("depth_window_half_m", 0.08)
        self.declare_parameter("target_class", -1)
        self.declare_parameter("detect_conf", 0.75)
        self.declare_parameter("register_conf", 0.75)
        self.declare_parameter("stable_frames", 3)
        self.declare_parameter("est_refine_iter", 2)
        self.declare_parameter("symmetry_mode", "z12")
        self.declare_parameter("device", "cpu")
        self.declare_parameter("render_backend", "moderngl")
        self.declare_parameter("net_backend", "openvino")
        self.declare_parameter("ov_device", "CPU")
        self.declare_parameter("ov_precision", "fp32")
        self.declare_parameter("ov_num_threads", 0)  # 0 表示由 OpenVINO 自动决定
        self.declare_parameter("ov_cache_dir", "none")  # none/off/空字符串表示不启用 OpenVINO 编译缓存
        self.declare_parameter("canonicalize_yaw", True)
        self.declare_parameter("yaw_fallback_deg", 15.0)
        self.declare_parameter("zone_disambiguation_enabled", False)
        self.declare_parameter("zone_disambiguation_radius_m", 0.14)
        self.declare_parameter("sync_tolerance_ms", 5.0)
        self.declare_parameter("sync_queue_size", 60)
        self.declare_parameter("latest_frame_max_age_ms", 120.0)
        self.declare_parameter("drop_stale_rgbd", True)
        self.declare_parameter("process_timer_hz", 30.0)
        self.declare_parameter("debug", 1)
        self.declare_parameter("debug_dir", str(PROJECT_ROOT / "runtime/foundationpose_ros_debug"))
        self.declare_parameter("diagnostic_visualization", True)
        self.declare_parameter("show_diagnostic_windows", False)
        self.declare_parameter("diagnostic_yolo_topic", "/energy/diagnostics/yolo")
        self.declare_parameter("yolo_detections_topic", "/energy/yolo_detections")
        self.declare_parameter("diagnostic_mask_topic", "/energy/diagnostics/mask")
        self.declare_parameter(
            "diagnostic_foundationpose_topic",
            "/energy/diagnostics/foundationpose",
        )
        self.declare_parameter("timing_diagnostics", True)
        self.declare_parameter("timing_topic", "/energy/timing_events")
        self.declare_parameter("monitor_after_init", True)
        self.declare_parameter("monitor_stride", 3)
        self.declare_parameter("supervisor_enabled", True)
        self.declare_parameter("supervisor_frames", 20)
        self.declare_parameter("track_refine_iter", 2)
        self.declare_parameter("initial_pose_publish_policy", "fresh_supervisor")
        self.declare_parameter("max_initial_pose_age_ms", 80.0)
        self.declare_parameter("fresh_pose_required_frames", 2)
        self.declare_parameter(
            "supervisor_pose_topic",
            "/energy/foundationpose_supervisor_pose",
        )
        self.declare_parameter("pose_output", str(PROJECT_ROOT / "runtime/foundation_pose.txt"))
        self.declare_parameter("publish_once", True)
        zone_radius_m = float(self.get_parameter("zone_disambiguation_radius_m").value)
        if not np.isfinite(zone_radius_m) or zone_radius_m <= 0:
            raise ValueError(
                f"zone_disambiguation_radius_m must be finite and > 0, got {zone_radius_m}"
            )
        self.color_frame_count = 0
        self.supervisor_active = False
        self.supervisor_remaining = 0
        self.processing_frame = False
        self.last_processed_color_stamp_ns = None
        self.latest_buffer = LatestRgbdBuffer(
            depth_history_size=int(self.get_parameter("sync_queue_size").value)
        )
        self.initial_pose_policy = InitialPosePublishPolicy(
            mode=str(self.get_parameter("initial_pose_publish_policy").value),
            max_initial_pose_age_ms=float(
                self.get_parameter("max_initial_pose_age_ms").value
            ),
            fresh_pose_required_frames=int(
                self.get_parameter("fresh_pose_required_frames").value
            ),
        )
        self.candidate_policy = FoundationPoseCandidatePolicy(
            direct_initial_pose_enabled=bool(
                self.get_parameter("direct_initial_pose_enabled").value
            ),
            request_required=bool(self.get_parameter("request_required").value),
        )

        self.pose_pub = self.create_publisher(
            PoseStamped,
            self.get_parameter("initial_pose_topic").value,
            10,
        )
        self.supervisor_pose_pub = self.create_publisher(
            PoseStamped,
            self.get_parameter("supervisor_pose_topic").value,
            10,
        )
        self.candidate_pose_pub = self.create_publisher(
            PoseStamped,
            self.get_parameter("candidate_pose_topic").value,
            10,
        )
        self.candidate_status_pub = self.create_publisher(
            String,
            self.get_parameter("candidate_status_topic").value,
            20,
        )
        self.yolo_diag_pub = self.create_publisher(
            Image,
            self.get_parameter("diagnostic_yolo_topic").value,
            10,
        )
        self.yolo_detections_pub = self.create_publisher(
            String,
            self.get_parameter("yolo_detections_topic").value,
            20,
        )
        self.mask_diag_pub = self.create_publisher(
            Image,
            self.get_parameter("diagnostic_mask_topic").value,
            10,
        )
        self.foundationpose_diag_pub = self.create_publisher(
            Image,
            self.get_parameter("diagnostic_foundationpose_topic").value,
            10,
        )
        self.timing_pub = self.create_publisher(
            String,
            self.get_parameter("timing_topic").value,
            50,
        )
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
            1,
        )
        self.create_subscription(
            Image,
            self.get_parameter("color_topic").value,
            self._color_cb,
            1,
        )
        self.create_subscription(
            String,
            self.get_parameter("init_request_topic").value,
            self._init_request_cb,
            10,
        )
        self.create_subscription(
            String,
            self.get_parameter("reinit_request_topic").value,
            self._reinit_request_cb,
            10,
        )
        process_timer_hz = max(1.0, float(self.get_parameter("process_timer_hz").value))
        self.process_timer = self.create_timer(1.0 / process_timer_hz, self._process_latest_frame)

        target_class = self.get_parameter("target_class").value
        self.target_class = None if int(target_class) < 0 else int(target_class)
        self.gate = AutoRegisterGate(
            min_conf=float(self.get_parameter("detect_conf").value),
            stable_frames=int(self.get_parameter("stable_frames").value),
        )

        self.get_logger().info("Loading FoundationPose models for ROS2 initialization")
        self.mask_gen = MaskGenerator(
            yolo_path=self.get_parameter("yolo_model").value,
            device=self.get_parameter("device").value,
            center_frac=float(self.get_parameter("mask_center_frac").value),
            half_window_m=float(self.get_parameter("depth_window_half_m").value),
        )
        self.estimator = self._create_estimator()
        self.get_logger().info("FoundationPose ROS2 initializer is ready")

    def _now_ns(self):
        return int(self.get_clock().now().nanoseconds)

    def _publish_timing(self, event, frame_stamp_ns, started_ns=None, extra=None):
        if not bool(self.get_parameter("timing_diagnostics").value):
            return
        payload = make_timing_event(
            event,
            frame_stamp_ns=frame_stamp_ns,
            node_now_ns=self._now_ns(),
            started_ns=started_ns,
            extra=extra,
        )
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        self.timing_pub.publish(msg)
        self.get_logger().info(f"TIMING {msg.data}")

    def _create_estimator(self):
        ov_num_threads = int(self.get_parameter("ov_num_threads").value)
        ov_cache_dir = str(self.get_parameter("ov_cache_dir").value).strip()
        if ov_cache_dir.lower() in ("", "none", "off"):
            ov_cache_dir = ""
        bundle = build_estimator(
            EstimatorConfig(
                mesh_file=self.get_parameter("mesh_file").value,
                debug=int(self.get_parameter("debug").value),
                debug_dir=self.get_parameter("debug_dir").value,
                symmetry_mode=self.get_parameter("symmetry_mode").value,
                device=self.get_parameter("device").value,
                render_backend=self.get_parameter("render_backend").value,
                net_backend=self.get_parameter("net_backend").value,
                ov_device=self.get_parameter("ov_device").value,
                ov_precision=str(self.get_parameter("ov_precision").value),
                ov_num_threads=None if ov_num_threads <= 0 else ov_num_threads,
                ov_cache_dir=None if not ov_cache_dir else str(PROJECT_ROOT / ov_cache_dir),
            )
        )
        self.estimator_bundle = bundle
        self.to_origin = bundle.to_origin
        self.bbox_vis = bundle.bbox_vis
        self.model_center, self.large_tip, self.small_tip = mesh_endpoints_from_vertices(
            bundle.mesh.vertices
        )
        return bundle.estimator

    def _canonicalize_pose(self, pose):
        if not bool(self.get_parameter("canonicalize_yaw").value):
            return pose
        return canonicalize_yaw(
            pose,
            self.estimator_bundle.model_center,
            float(self.get_parameter("yaw_fallback_deg").value),
        )

    def _sync_estimator_pose_last(self, world_pose):
        last = self.estimator.pose_last
        tf_to_center = self.estimator.get_tf_to_centered_mesh()
        self.estimator.pose_last = (
            torch.as_tensor(world_pose, dtype=last.dtype, device=last.device)
            @ torch.linalg.inv(tf_to_center)
        )

    def _camera_info_cb(self, msg):
        with self.lock:
            self.latest_camera_info = msg

    def _depth_cb(self, msg):
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        stamp_ns = stamp_to_nanoseconds(msg.header.stamp)
        with self.lock:
            self.latest_buffer.put_depth(stamp_ns, depth_image_to_meters(depth))

    def _color_cb(self, msg):
        self.color_frame_count += 1
        color = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        stamp_ns = stamp_to_nanoseconds(msg.header.stamp)
        with self.lock:
            self.latest_buffer.put_color(stamp_ns, (color, msg.header))

    def _reinit_request_cb(self, msg):
        self._init_request_cb(msg)

    def _init_request_cb(self, msg):
        try:
            request = json.loads(msg.data)
        except json.JSONDecodeError:
            request = {
                "request_id": f"legacy-{self._now_ns()}",
                "reason": msg.data,
            }
        if "request_id" not in request:
            request["request_id"] = f"request-{self._now_ns()}"
        if "stamp_ns" not in request:
            request["stamp_ns"] = self._now_ns()
        with self.lock:
            self.active_init_request = dict(request)
        self.initialized = False
        self.supervisor_active = False
        self.supervisor_remaining = 0
        self.last_processed_color_stamp_ns = None
        self.initial_pose_policy.reset()
        self.gate.reset()
        self.get_logger().warning(
            "Received FoundationPose candidate request; "
            f"request_id={request.get('request_id')} reason={request.get('reason', '')}"
        )

    def _get_active_init_request(self):
        with self.lock:
            return None if self.active_init_request is None else dict(self.active_init_request)

    def _clear_active_init_request(self, request_id):
        with self.lock:
            if (
                self.active_init_request is not None
                and str(self.active_init_request.get("request_id", "")) == str(request_id)
            ):
                self.active_init_request = None

    def _latest_pair(self):
        now_ns = self._now_ns()
        tolerance_ns = int(float(self.get_parameter("sync_tolerance_ms").value) * 1_000_000)
        if bool(self.get_parameter("drop_stale_rgbd").value):
            max_age_ns = int(
                float(self.get_parameter("latest_frame_max_age_ms").value) * 1_000_000
            )
        else:
            max_age_ns = 10**18
        with self.lock:
            pair = self.latest_buffer.get_latest_pair(
                now_ns=now_ns,
                tolerance_ns=tolerance_ns,
                max_age_ns=max_age_ns,
            )
            camera_info = self.latest_camera_info
        return pair, camera_info, now_ns

    def _process_latest_frame(self):
        if self.processing_frame:
            return
        pair, camera_info, now_ns = self._latest_pair()
        if pair is None or camera_info is None:
            return
        if pair.color_stamp_ns == self.last_processed_color_stamp_ns:
            return

        self.processing_frame = True
        self.last_processed_color_stamp_ns = pair.color_stamp_ns
        try:
            color, header = pair.color
            self._process_rgbd_pair(
                color=color,
                depth=pair.depth.copy(),
                camera_info=camera_info,
                header=header,
                color_stamp_ns=pair.color_stamp_ns,
                depth_stamp_ns=pair.depth_stamp_ns,
            )
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            self.get_logger().error(
                f"FoundationPose frame processing failed: {exc}\n{tb}"
            )
            self._publish_candidate_failure_status(
                frame_stamp_ns=pair.color_stamp_ns,
                error=f"{type(exc).__name__}: {exc}",
                traceback_text=tb,
            )
            self.initialized = False
            self.supervisor_active = False
            self.supervisor_remaining = 0
            self.last_processed_color_stamp_ns = None
        finally:
            self.processing_frame = False

    def _process_rgbd_pair(
        self,
        color,
        depth,
        camera_info,
        header,
        color_stamp_ns,
        depth_stamp_ns,
    ):
        active_request = self._get_active_init_request()
        if self.initialized and bool(self.get_parameter("publish_once").value) and active_request is None:
            if self.supervisor_active:
                if self._publish_next_supervisor_pose(
                    color=color,
                    depth=depth,
                    camera_info=camera_info,
                    header=header,
                    color_stamp_ns=color_stamp_ns,
                ):
                    return
                self.get_logger().warning(
                    "FoundationPose supervisor could not update on this frame; falling back to YOLO monitor"
                )
            if bool(self.get_parameter("monitor_after_init").value):
                stride = max(1, int(self.get_parameter("monitor_stride").value))
                if self.color_frame_count % stride == 0:
                    detections = self.mask_gen.detect_only(
                        color,
                        target_class=self.target_class,
                        conf=float(self.get_parameter("detect_conf").value),
                    )
                    self._publish_yolo_detections(detections, color_stamp_ns)
                    self._publish_yolo_visualization(color, detections, None, header)
            return
        if depth_stamp_ns != color_stamp_ns:
            self.get_logger().debug(
                f"Using nearest RGB-D pair: color_stamp_ns={color_stamp_ns}, "
                f"depth_stamp_ns={depth_stamp_ns}"
            )

        detections = self.mask_gen.detect_only(
            color,
            target_class=self.target_class,
            conf=float(self.get_parameter("detect_conf").value),
        )
        self._publish_yolo_detections(detections, color_stamp_ns)
        selected = self.gate.update(detections)
        self._publish_yolo_visualization(color, detections, selected, header)
        if active_request is None and bool(self.get_parameter("request_required").value):
            return
        if selected is None and active_request is None:
            return
        if selected is None and active_request is not None:
            selected = self._request_detection(active_request)

        self.get_logger().info(
            f"Stable detection accepted: bbox={selected[:4]}, conf={selected[4]:.3f}, cls={selected[5]}"
        )
        init_started_ns = self._now_ns()
        self._publish_timing(
            "foundationpose_stable_detection_accepted",
            frame_stamp_ns=color_stamp_ns,
            started_ns=None,
            extra={
                "color_frame_count": int(self.color_frame_count),
                "depth_stamp_ns": None if depth_stamp_ns is None else int(depth_stamp_ns),
                "rgb_depth_delta_ms": (
                    None
                    if depth_stamp_ns is None
                    else abs(color_stamp_ns - depth_stamp_ns) / 1_000_000.0
                ),
                "selected_bbox": [float(v) for v in selected[:4]],
                "selected_conf": float(selected[4]),
                "selected_class": int(selected[5]),
            },
        )
        mask, bbox = self.mask_gen.generate(
            color,
            depth,
            target_class=self.target_class,
            conf=float(self.get_parameter("register_conf").value),
        )
        if mask.sum() == 0:
            self.get_logger().warning("YOLO box has no valid depth near the object; skipping this frame")
            return

        k_matrix = camera_info_to_matrix(camera_info)
        pose = self.estimator.register(
            K=k_matrix,
            rgb=color,
            depth=depth,
            ob_mask=mask,
            iteration=int(self.get_parameter("est_refine_iter").value),
            bbox_xyxy=bbox,
        )
        zone_extra = {}
        if bool(self.get_parameter("zone_disambiguation_enabled").value):
            zone_radius_m = float(self.get_parameter("zone_disambiguation_radius_m").value)
            zone_res = disambiguate_zone_pose(
                pose=pose,
                depth_m=depth,
                K=k_matrix,
                model_center=self.model_center,
                large_tip=self.large_tip,
                small_tip=self.small_tip,
                radius_m=zone_radius_m,
            )
            pose = zone_res.pose
            zone_extra = {
                "zone_disambiguation_flipped": bool(zone_res.flipped),
                "zone_n_large": int(zone_res.n_large),
                "zone_n_small": int(zone_res.n_small),
            }
            if zone_res.flipped:
                if getattr(self.estimator, "pose_last", None) is not None:
                    self._sync_estimator_pose_last(pose)
                self.get_logger().info(
                    "Zone disambiguation flipped the FoundationPose pose: "
                    f"n_large={zone_res.n_large}, n_small={zone_res.n_small}, "
                    f"radius_m={zone_radius_m}"
                )
        pose = self._canonicalize_pose(pose)
        self._publish_timing(
            "foundationpose_register_done",
            frame_stamp_ns=color_stamp_ns,
            started_ns=init_started_ns,
            extra={
                "bbox": [] if bbox is None else [float(v) for v in bbox],
                "mask_pixels": int(mask.sum()),
                **zone_extra,
            },
        )
        register_age_ms = (self._now_ns() - color_stamp_ns) / 1_000_000.0
        candidate_decision = self.candidate_policy.on_register_result(active_request)
        if candidate_decision.publish_initial_pose:
            self._publish_pose(pose, header.stamp)
        if candidate_decision.publish_candidate_pose:
            self._publish_candidate_pose(pose, header.stamp)
            self._publish_candidate_status(
                request=active_request,
                frame_stamp_ns=color_stamp_ns,
                pose_age_ms=register_age_ms,
                bbox=bbox,
                mask_pixels=int(mask.sum()),
                event=candidate_decision.event,
            )
            self._clear_active_init_request(candidate_decision.request_id)
        self._publish_supervisor_pose(pose, header.stamp)
        self._publish_timing(
            candidate_decision.event,
            frame_stamp_ns=color_stamp_ns,
            started_ns=init_started_ns,
            extra={
                "age_ms_at_decision": register_age_ms,
                "request_id": candidate_decision.request_id,
                "pose_topic": self.get_parameter("initial_pose_topic").value,
                "candidate_pose_topic": self.get_parameter("candidate_pose_topic").value,
                "published_initial_pose": bool(candidate_decision.publish_initial_pose),
                "published_candidate_pose": bool(candidate_decision.publish_candidate_pose),
            },
        )
        self._publish_foundationpose_visualization(
            color,
            camera_info,
            mask,
            bbox,
            pose,
            header,
        )
        pose_output = Path(self.get_parameter("pose_output").value)
        pose_output.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(pose_output, pose.reshape(4, 4))
        self.initialized = True
        if bool(self.get_parameter("supervisor_enabled").value) and bool(
            self.get_parameter("direct_initial_pose_enabled").value
        ):
            self.supervisor_active = True
            self.supervisor_remaining = max(
                0,
                int(self.get_parameter("supervisor_frames").value),
            )
        else:
            self.supervisor_active = False
            self.supervisor_remaining = 0
        self.get_logger().info(
            f"FoundationPose register seed ready from RGB-D stamp_ns={color_stamp_ns}, bbox={bbox}"
        )

    def _publish_pose(self, matrix, stamp):
        self._publish_pose_to(self.pose_pub, matrix, stamp)

    def _publish_supervisor_pose(self, matrix, stamp):
        self._publish_pose_to(self.supervisor_pose_pub, matrix, stamp)

    def _publish_candidate_pose(self, matrix, stamp):
        self._publish_pose_to(self.candidate_pose_pub, matrix, stamp)

    def _publish_candidate_status(
        self,
        request,
        frame_stamp_ns,
        pose_age_ms,
        bbox,
        mask_pixels,
        event,
    ):
        payload = {
            "event": event,
            "success": True,
            "request_id": None if request is None else request.get("request_id"),
            "request_stamp_ns": None if request is None else request.get("stamp_ns"),
            "frame_stamp_ns": int(frame_stamp_ns),
            "pose_age_ms": float(pose_age_ms),
            "bbox": [] if bbox is None else [float(v) for v in bbox],
            "mask_pixels": int(mask_pixels),
            "candidate_pose_topic": self.get_parameter("candidate_pose_topic").value,
        }
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        self.candidate_status_pub.publish(msg)

    def _publish_candidate_failure_status(self, frame_stamp_ns, error, traceback_text):
        request = self._get_active_init_request()
        if request is None:
            return
        payload = {
            "event": "foundationpose_candidate_failed",
            "success": False,
            "request_id": request.get("request_id"),
            "request_stamp_ns": request.get("stamp_ns"),
            "frame_stamp_ns": int(frame_stamp_ns),
            "error": str(error),
            "traceback_tail": str(traceback_text)[-2000:],
            "candidate_pose_topic": self.get_parameter("candidate_pose_topic").value,
        }
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        self.candidate_status_pub.publish(msg)
        self._publish_timing(
            "foundationpose_candidate_failed",
            frame_stamp_ns=frame_stamp_ns,
            started_ns=None,
            extra={
                "request_id": request.get("request_id"),
                "error": str(error),
            },
        )
        self._clear_active_init_request(request.get("request_id"))

    def _publish_pose_to(self, publisher, matrix, stamp):
        pose = matrix_to_pose_components(matrix.reshape(4, 4).tolist())
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.get_parameter("output_frame_id").value
        msg.pose.position.x = pose["position"]["x"]
        msg.pose.position.y = pose["position"]["y"]
        msg.pose.position.z = pose["position"]["z"]
        msg.pose.orientation.x = pose["orientation"]["x"]
        msg.pose.orientation.y = pose["orientation"]["y"]
        msg.pose.orientation.z = pose["orientation"]["z"]
        msg.pose.orientation.w = pose["orientation"]["w"]
        publisher.publish(msg)

    def _request_detection(self, request):
        bbox = request.get("bbox_xyxy", [0, 0, 0, 0])
        class_id = request.get(
            "target_class",
            self.target_class if self.target_class is not None else -1,
        )
        return (
            int(float(bbox[0])),
            int(float(bbox[1])),
            int(float(bbox[2])),
            int(float(bbox[3])),
            float(request.get("confidence", 1.0)),
            int(class_id) if class_id is not None else -1,
        )

    def _publish_next_supervisor_pose(
        self,
        color,
        depth,
        camera_info,
        header,
        color_stamp_ns,
    ):
        if self.supervisor_remaining <= 0:
            self.supervisor_active = False
            return False

        started_ns = self._now_ns()
        pose = self.estimator.track_one(
            rgb=color,
            depth=depth,
            K=camera_info_to_matrix(camera_info),
            iteration=int(self.get_parameter("track_refine_iter").value),
        )
        pose = self._canonicalize_pose(pose)
        self._publish_supervisor_pose(pose, header.stamp)
        pose_age_ms = (self._now_ns() - color_stamp_ns) / 1_000_000.0
        decision = self.initial_pose_policy.on_track_pose(pose_age_ms)
        if decision.publish_initial_pose and bool(
            self.get_parameter("direct_initial_pose_enabled").value
        ):
            self._publish_pose(pose, header.stamp)
            self.supervisor_remaining = max(
                0,
                int(self.get_parameter("supervisor_frames").value),
            )
        self._publish_timing(
            decision.event,
            frame_stamp_ns=color_stamp_ns,
            started_ns=started_ns,
            extra={
                "age_ms_at_decision": pose_age_ms,
                "fresh_count": int(decision.fresh_count),
                "pose_topic": self.get_parameter("initial_pose_topic").value,
                "published_initial_pose": bool(decision.publish_initial_pose),
            },
        )
        self._publish_timing(
            "foundationpose_supervisor_track_one_published",
            frame_stamp_ns=color_stamp_ns,
            started_ns=started_ns,
            extra={
                "remaining_after_publish": int(max(0, self.supervisor_remaining - 1)),
                "pose_topic": self.get_parameter("supervisor_pose_topic").value,
            },
        )
        if self.initial_pose_policy.initial_pose_published:
            self.supervisor_remaining -= 1
            if self.supervisor_remaining <= 0:
                self.supervisor_active = False
                self.get_logger().info("FoundationPose supervisor warm-up finished")
        return True

    def _diagnostic_enabled(self):
        return bool(self.get_parameter("diagnostic_visualization").value)

    def _diagnostic_dir(self):
        return Path(self.get_parameter("debug_dir").value) / "diagnostics"

    def _maybe_show(self, name, image):
        if bool(self.get_parameter("show_diagnostic_windows").value):
            import cv2

            cv2.imshow(name, image)
            cv2.waitKey(1)

    def _publish_diag_image(self, publisher, image, header):
        msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
        msg.header = header
        publisher.publish(msg)

    def _publish_yolo_visualization(self, color, detections, selected, header):
        if not self._diagnostic_enabled():
            return
        vis = draw_yolo_detections(color, detections, selected=selected)
        save_debug_image(self._diagnostic_dir() / "yolo_latest.png", vis)
        if selected is not None:
            save_debug_image(self._diagnostic_dir() / "yolo_selected.png", vis)
        self._publish_diag_image(self.yolo_diag_pub, vis, header)
        self._maybe_show("YOLO diagnostics", vis)

    def _publish_yolo_detections(self, detections, stamp_ns):
        msg = String()
        msg.data = detections_to_json(stamp_ns, detections)
        self.yolo_detections_pub.publish(msg)

    def _publish_foundationpose_visualization(
        self,
        color,
        camera_info,
        mask,
        bbox,
        pose,
        header,
    ):
        if not self._diagnostic_enabled():
            return
        k_matrix = camera_info_to_matrix(camera_info)
        detection_vis = draw_yolo_detections(
            overlay_mask(color, mask),
            [] if bbox is None else [(*bbox, 1.0, self.target_class)],
        )
        center_pose = pose @ np.linalg.inv(self.to_origin)
        pose_vis = draw_posed_3d_box(
            k_matrix,
            img=color.copy(),
            ob_in_cam=center_pose,
            bbox=self.bbox_vis,
        )
        pose_vis = draw_xyz_axis(
            pose_vis,
            ob_in_cam=center_pose,
            scale=0.05,
            K=k_matrix,
            thickness=3,
            transparency=0,
            is_input_rgb=False,
        )
        import cv2

        cv2.putText(
            detection_vis,
            "YOLO + depth-window mask used for FoundationPose init",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            pose_vis,
            "FoundationPose global init result",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        canvas = np.concatenate([detection_vis, pose_vis], axis=1)
        self._publish_diag_image(self.mask_diag_pub, detection_vis, header)
        self._publish_diag_image(self.foundationpose_diag_pub, pose_vis, header)
        save_debug_image(self._diagnostic_dir() / "foundationpose_initialization.png", canvas)
        self._maybe_show("FoundationPose init diagnostics", canvas)


def main():
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "runtime/matplotlib"))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / "runtime/ultralytics"))
    rclpy.init()
    node = FoundationPoseInitNode()
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
