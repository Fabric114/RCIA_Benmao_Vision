from dataclasses import asdict, dataclass
import math

import numpy as np

from scripts.tracking_watchdog_validation import (
    WatchdogValidationParams,
    pose_jump_ok,
    validate_m3t_pose_health,
)


@dataclass(frozen=True)
class TrackingHealthGateParams:
    min_rendered_overlap_pixels: int = 80
    min_depth_consistency_ratio: float = 0.55
    max_depth_residual_m: float = 0.02
    min_projection_iou: float = 0.20
    max_projection_center_distance_ratio: float = 0.75
    min_projection_area_ratio: float = 0.25
    max_pose_jump_translation_m: float = 0.08
    max_pose_jump_rotation_deg: float = 30.0
    depth_scale: float = 0.001
    require_smooth_pose: bool = True


@dataclass(frozen=True)
class TrackingHealthGateResult:
    tracking_ok: bool
    depth_ok: bool
    projection_ok: bool
    projection_center_ok: bool
    projection_area_ok: bool
    black_white_ok: bool
    smoothness_ok: bool
    depth_ratio: float
    projection_iou: float
    projection_center_distance_ratio: float
    projection_area_ratio: float
    translation_jump_m: float
    rotation_jump_deg: float
    rendered_pixel_count: int
    reason: str = ""
    health_evaluated: bool = False
    valid_depth_overlap_count: int = 0
    depth_evaluable: bool = False

    def to_dict(self):
        data = asdict(self)
        data["depth_consistency_ratio"] = self.depth_ratio
        data["pose_valid"] = self.tracking_ok
        return data


def evaluate_m3t_tracking_health(
    vertices,
    faces,
    object_in_camera,
    previous_pose,
    k_matrix,
    color_bgr,
    observed_depth_image,
    yolo_bbox,
    params=None,
):
    params = params or TrackingHealthGateParams()
    if color_bgr is None or observed_depth_image is None:
        return unevaluated_tracking_health("missing RGB-D frame for tracking health")
    if not np.all(np.isfinite(np.asarray(object_in_camera, dtype=np.float64))):
        return unevaluated_tracking_health("non-finite M3T pose; not rendered")

    health = validate_m3t_pose_health(
        vertices=vertices,
        faces=faces,
        object_in_camera=object_in_camera,
        k_matrix=k_matrix,
        color_bgr=color_bgr,
        observed_depth_image=observed_depth_image,
        yolo_bbox=yolo_bbox,
        previous_pose=None,
        params=WatchdogValidationParams(
            min_rendered_overlap_pixels=params.min_rendered_overlap_pixels,
            tracking_min_depth_consistency_ratio=params.min_depth_consistency_ratio,
            max_depth_residual_m=params.max_depth_residual_m,
            min_yolo_projection_iou=params.min_projection_iou,
            max_projection_center_distance_ratio=params.max_projection_center_distance_ratio,
            min_projection_area_ratio=params.min_projection_area_ratio,
            depth_scale=params.depth_scale,
        ),
    )
    translation_jump_m, rotation_jump_deg = pose_delta(previous_pose, object_in_camera)
    smoothness_ok = pose_jump_ok(
        previous_pose,
        object_in_camera,
        max_translation_m=params.max_pose_jump_translation_m,
        max_rotation_deg=params.max_pose_jump_rotation_deg,
    )
    depth_ok = bool(health.depth_success)
    black_white_ok = bool(health.black_white_success)
    projection_ok = (
        yolo_bbox is None
        or float(health.projection_iou) >= float(params.min_projection_iou)
    )
    projection_center_ok = bool(health.projection_center_success)
    projection_area_ok = bool(health.projection_area_success)
    reasons = []
    if not depth_ok:
        reasons.append("depth consistency failed")
    if not projection_ok:
        reasons.append(
            f"projection IoU {health.projection_iou:.3f} below {params.min_projection_iou:.3f}"
        )
    if not projection_center_ok:
        reasons.append(
            "projection center distance ratio "
            f"{health.projection_center_distance_ratio:.3f} above "
            f"{params.max_projection_center_distance_ratio:.3f}"
        )
    if not projection_area_ok:
        reasons.append(
            f"projection area ratio {health.projection_area_ratio:.3f} below "
            f"{params.min_projection_area_ratio:.3f}"
        )
    if not black_white_ok:
        reasons.append("black/white appearance failed")
    if not smoothness_ok:
        reasons.append("pose smoothness warning")
    if health.reason:
        reasons.append(health.reason)

    smoothness_gate_ok = smoothness_ok or not bool(params.require_smooth_pose)
    tracking_ok = (
        depth_ok
        and projection_ok
        and projection_center_ok
        and projection_area_ok
        and black_white_ok
        and smoothness_gate_ok
    )
    return TrackingHealthGateResult(
        tracking_ok=tracking_ok,
        depth_ok=depth_ok,
        projection_ok=projection_ok,
        projection_center_ok=projection_center_ok,
        projection_area_ok=projection_area_ok,
        black_white_ok=black_white_ok,
        smoothness_ok=bool(smoothness_ok),
        depth_ratio=float(health.depth_consistency_ratio),
        projection_iou=float(health.projection_iou),
        projection_center_distance_ratio=float(
            health.projection_center_distance_ratio
        ),
        projection_area_ratio=float(health.projection_area_ratio),
        translation_jump_m=float(translation_jump_m),
        rotation_jump_deg=float(rotation_jump_deg),
        rendered_pixel_count=int(health.rendered_pixel_count),
        reason="; ".join(reason for reason in reasons if reason),
        health_evaluated=True,
        valid_depth_overlap_count=int(health.valid_depth_overlap_count),
        depth_evaluable=int(health.valid_depth_overlap_count)
        >= int(params.min_rendered_overlap_pixels),
    )


def unevaluated_tracking_health(reason):
    """Result for a frame where health was not computed at all."""
    return TrackingHealthGateResult(
        tracking_ok=False,
        depth_ok=False,
        projection_ok=False,
        projection_center_ok=False,
        projection_area_ok=False,
        black_white_ok=False,
        smoothness_ok=False,
        depth_ratio=0.0,
        projection_iou=0.0,
        projection_center_distance_ratio=float("inf"),
        projection_area_ratio=0.0,
        translation_jump_m=0.0,
        rotation_jump_deg=0.0,
        rendered_pixel_count=0,
        reason=reason,
        health_evaluated=False,
        valid_depth_overlap_count=0,
        depth_evaluable=False,
    )


def pose_delta(previous_pose, current_pose):
    if previous_pose is None or current_pose is None:
        return 0.0, 0.0
    previous = np.asarray(previous_pose, dtype=np.float64).reshape(4, 4)
    current = np.asarray(current_pose, dtype=np.float64).reshape(4, 4)
    translation_jump_m = float(np.linalg.norm(current[:3, 3] - previous[:3, 3]))
    rotation_delta = previous[:3, :3].T @ current[:3, :3]
    trace = float(np.trace(rotation_delta))
    cos_angle = max(-1.0, min(1.0, (trace - 1.0) * 0.5))
    rotation_jump_deg = float(math.degrees(math.acos(cos_angle)))
    return translation_jump_m, rotation_jump_deg
