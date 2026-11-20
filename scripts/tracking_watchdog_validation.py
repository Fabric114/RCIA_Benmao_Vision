from dataclasses import asdict, dataclass
import math

import numpy as np

from scripts.linemod_silhouette_renderer import render_model_buffers
from scripts.linemod_validator import (
    validate_black_white_appearance,
    validate_rendered_depth_consistency,
)


@dataclass(frozen=True)
class WatchdogValidationParams:
    min_rendered_overlap_pixels: int = 80
    tracking_min_depth_consistency_ratio: float = 0.60
    max_depth_residual_m: float = 0.02
    min_yolo_projection_iou: float = 0.20
    max_projection_center_distance_ratio: float = 0.75
    min_projection_area_ratio: float = 0.25
    max_pose_jump_translation_m: float = 0.08
    max_pose_jump_rotation_deg: float = 30.0
    depth_scale: float = 0.001


@dataclass(frozen=True)
class WatchdogHealth:
    yolo_visible: bool
    depth_success: bool
    depth_consistency_ratio: float
    black_white_success: bool
    projection_iou: float
    projection_center_distance_ratio: float
    projection_area_ratio: float
    projection_center_success: bool
    projection_area_success: bool
    pose_jump_ok: bool
    pose_valid: bool
    rendered_pixel_count: int
    reason: str = ""
    valid_depth_overlap_count: int = 0

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class CandidateValidationResult:
    success: bool
    depth_consistency_ratio: float
    projection_iou: float
    black_white_success: bool
    rendered_pixel_count: int
    reason: str = ""

    def to_dict(self):
        return asdict(self)


def bbox_iou(a, b):
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in b[:4]]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return float(inter_area / union)


def mask_bbox(mask):
    mask_array = np.asarray(mask) > 0
    ys, xs = np.nonzero(mask_array)
    if xs.size == 0:
        return None
    return (
        float(xs.min()),
        float(ys.min()),
        float(xs.max()),
        float(ys.max()),
    )


def projection_iou_from_mask(mask, yolo_bbox):
    return bbox_iou(mask_bbox(mask), yolo_bbox)


def bbox_center_distance_ratio(a, b):
    if a is None or b is None:
        return float("inf")
    ax1, ay1, ax2, ay2 = [float(v) for v in a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in b[:4]]
    aw = max(0.0, ax2 - ax1)
    ah = max(0.0, ay2 - ay1)
    bw = max(0.0, bx2 - bx1)
    bh = max(0.0, by2 - by1)
    if aw <= 0.0 or ah <= 0.0 or bw <= 0.0 or bh <= 0.0:
        return float("inf")
    acx = (ax1 + ax2) * 0.5
    acy = (ay1 + ay2) * 0.5
    bcx = (bx1 + bx2) * 0.5
    bcy = (by1 + by2) * 0.5
    reference = math.hypot(bw, bh)
    if reference <= 1e-9:
        return float("inf")
    return float(math.hypot(acx - bcx, acy - bcy) / reference)


def bbox_area_ratio(a, b):
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in b[:4]]
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    larger = max(area_a, area_b)
    if larger <= 0.0:
        return 0.0
    return float(min(area_a, area_b) / larger)


def pose_jump_ok(
    previous_pose,
    current_pose,
    max_translation_m=0.08,
    max_rotation_deg=30.0,
):
    if previous_pose is None or current_pose is None:
        return True
    previous = np.asarray(previous_pose, dtype=np.float64).reshape(4, 4)
    current = np.asarray(current_pose, dtype=np.float64).reshape(4, 4)
    translation_delta = np.linalg.norm(current[:3, 3] - previous[:3, 3])
    if translation_delta > float(max_translation_m):
        return False

    rotation_delta = previous[:3, :3].T @ current[:3, :3]
    trace = float(np.trace(rotation_delta))
    cos_angle = max(-1.0, min(1.0, (trace - 1.0) * 0.5))
    angle_deg = math.degrees(math.acos(cos_angle))
    return angle_deg <= float(max_rotation_deg)


def build_watchdog_health(
    yolo_bbox,
    rendered_mask,
    depth_result,
    appearance_result,
    previous_pose,
    current_pose,
    params=None,
):
    params = params or WatchdogValidationParams()
    rendered_pixel_count = int(getattr(depth_result, "rendered_pixel_count", 0))
    depth_ratio = float(getattr(depth_result, "consistency_ratio", 0.0))
    depth_success = bool(getattr(depth_result, "success", False))
    projected_bbox = mask_bbox(rendered_mask)
    projection_iou = bbox_iou(projected_bbox, yolo_bbox)
    projection_center_distance = bbox_center_distance_ratio(projected_bbox, yolo_bbox)
    projection_area = bbox_area_ratio(projected_bbox, yolo_bbox)
    projection_center_success = (
        yolo_bbox is None
        or projection_center_distance
        <= float(params.max_projection_center_distance_ratio)
    )
    projection_area_success = (
        yolo_bbox is None
        or projection_area >= float(params.min_projection_area_ratio)
    )
    jump_ok = pose_jump_ok(
        previous_pose,
        current_pose,
        max_translation_m=params.max_pose_jump_translation_m,
        max_rotation_deg=params.max_pose_jump_rotation_deg,
    )

    reasons = []
    if rendered_pixel_count < int(params.min_rendered_overlap_pixels):
        reasons.append(
            f"too few rendered pixels: {rendered_pixel_count} < {int(params.min_rendered_overlap_pixels)}"
        )
        black_white_success = False
    else:
        black_white_success = bool(getattr(appearance_result, "success", False))

    if not depth_success:
        reason = getattr(depth_result, "reason", "")
        reasons.append(reason or "depth consistency failed")
    if yolo_bbox is not None and projection_iou < params.min_yolo_projection_iou:
        reasons.append(
            f"projection IoU {projection_iou:.3f} below {params.min_yolo_projection_iou:.3f}"
        )
    if not projection_center_success:
        reasons.append(
            "projection center distance ratio "
            f"{projection_center_distance:.3f} above "
            f"{float(params.max_projection_center_distance_ratio):.3f}"
        )
    if not projection_area_success:
        reasons.append(
            f"projection area ratio {projection_area:.3f} below "
            f"{float(params.min_projection_area_ratio):.3f}"
        )
    if not black_white_success and rendered_pixel_count >= int(params.min_rendered_overlap_pixels):
        reason = getattr(appearance_result, "reason", "")
        reasons.append(reason or "black/white appearance failed")
    if not jump_ok:
        reasons.append("pose jump exceeded threshold")

    pose_valid = (
        rendered_pixel_count >= int(params.min_rendered_overlap_pixels)
        and depth_success
        and black_white_success
        and jump_ok
        and (
            yolo_bbox is None
            or projection_iou >= float(params.min_yolo_projection_iou)
        )
        and projection_center_success
        and projection_area_success
    )

    return WatchdogHealth(
        yolo_visible=yolo_bbox is not None,
        depth_success=depth_success,
        depth_consistency_ratio=depth_ratio,
        black_white_success=black_white_success,
        projection_iou=projection_iou,
        projection_center_distance_ratio=projection_center_distance,
        projection_area_ratio=projection_area,
        projection_center_success=projection_center_success,
        projection_area_success=projection_area_success,
        pose_jump_ok=jump_ok,
        pose_valid=pose_valid,
        rendered_pixel_count=rendered_pixel_count,
        reason="; ".join(reason for reason in reasons if reason),
        valid_depth_overlap_count=int(getattr(depth_result, "valid_overlap_count", 0)),
    )


def validate_m3t_pose_health(
    vertices,
    faces,
    object_in_camera,
    k_matrix,
    color_bgr,
    observed_depth_image,
    yolo_bbox=None,
    previous_pose=None,
    params=None,
):
    params = params or WatchdogValidationParams()
    height, width = np.asarray(color_bgr).shape[:2]
    image_size = (width, height)
    _, rendered_mask, _, _, _ = render_model_buffers(
        vertices,
        faces,
        object_in_camera,
        k_matrix,
        image_size=image_size,
    )
    depth_result = validate_rendered_depth_consistency(
        vertices,
        faces,
        object_in_camera,
        k_matrix,
        observed_depth_image,
        image_size=image_size,
        depth_scale=params.depth_scale,
        max_depth_residual_m=params.max_depth_residual_m,
        min_consistency_ratio=params.tracking_min_depth_consistency_ratio,
        min_overlap_pixels=params.min_rendered_overlap_pixels,
    )
    appearance_result = validate_black_white_appearance(color_bgr, rendered_mask)
    return build_watchdog_health(
        yolo_bbox=yolo_bbox,
        rendered_mask=rendered_mask,
        depth_result=depth_result,
        appearance_result=appearance_result,
        previous_pose=previous_pose,
        current_pose=object_in_camera,
        params=params,
    )


def validate_candidate_pose(
    vertices,
    faces,
    candidate_pose,
    k_matrix,
    color_bgr,
    observed_depth_image,
    yolo_bbox,
    params=None,
):
    params = params or WatchdogValidationParams()
    if color_bgr is None or observed_depth_image is None:
        return CandidateValidationResult(
            success=False,
            depth_consistency_ratio=0.0,
            projection_iou=0.0,
            black_white_success=False,
            rendered_pixel_count=0,
            reason="missing RGB-D frame for candidate validation",
        )

    height, width = np.asarray(color_bgr).shape[:2]
    image_size = (width, height)
    _, rendered_mask, _, _, _ = render_model_buffers(
        vertices,
        faces,
        candidate_pose,
        k_matrix,
        image_size=image_size,
    )
    depth_result = validate_rendered_depth_consistency(
        vertices,
        faces,
        candidate_pose,
        k_matrix,
        observed_depth_image,
        image_size=image_size,
        depth_scale=params.depth_scale,
        max_depth_residual_m=params.max_depth_residual_m,
        min_consistency_ratio=params.tracking_min_depth_consistency_ratio,
        min_overlap_pixels=params.min_rendered_overlap_pixels,
    )
    appearance_result = validate_black_white_appearance(color_bgr, rendered_mask)
    projection_iou = projection_iou_from_mask(rendered_mask, yolo_bbox)

    reasons = []
    if not depth_result.success:
        reasons.append(depth_result.reason or "depth consistency failed")
    if not appearance_result.success:
        reasons.append(appearance_result.reason or "black/white appearance failed")
    if projection_iou < float(params.min_yolo_projection_iou):
        reasons.append(
            f"projection IoU {projection_iou:.3f} below {params.min_yolo_projection_iou:.3f}"
        )

    success = not reasons
    return CandidateValidationResult(
        success=success,
        depth_consistency_ratio=float(depth_result.consistency_ratio),
        projection_iou=float(projection_iou),
        black_white_success=bool(appearance_result.success),
        rendered_pixel_count=int(depth_result.rendered_pixel_count),
        reason="; ".join(reasons),
    )
