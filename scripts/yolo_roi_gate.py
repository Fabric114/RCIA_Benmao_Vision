from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class YoloRoiGateParams:
    min_roi_area_px: int = 400
    min_valid_depth_ratio: float = 0.30
    min_low_saturation_ratio: float = 0.55
    min_black_white_support: float = 0.25
    max_saturation: int = 55
    black_v_max: int = 90
    white_v_min: int = 145


@dataclass(frozen=True)
class YoloRoiGateResult:
    success: bool
    valid_depth_ratio: float
    low_saturation_ratio: float
    black_white_support: float
    roi_area_px: int
    reason: str = ""

    def to_dict(self):
        return asdict(self)


def validate_yolo_roi(color_bgr, depth_image, bbox_xyxy, params=None):
    params = params or YoloRoiGateParams()
    if color_bgr is None or depth_image is None or bbox_xyxy is None:
        return _result(False, "missing RGB-D or bbox")

    color = np.asarray(color_bgr)
    depth = np.asarray(depth_image)
    if color.ndim < 3 or depth.ndim < 2:
        return _result(False, "invalid RGB-D shape")

    height, width = color.shape[:2]
    x1, y1, x2, y2 = _clip_bbox(bbox_xyxy, width, height)
    roi_area = max(0, x2 - x1) * max(0, y2 - y1)
    if roi_area < int(params.min_roi_area_px):
        return _result(False, f"roi area {roi_area} below {int(params.min_roi_area_px)}", roi_area=roi_area)

    color_roi = color[y1:y2, x1:x2]
    depth_roi = depth[y1:y2, x1:x2]
    valid_depth_ratio = float(np.count_nonzero(depth_roi > 0) / max(1, depth_roi.size))

    hsv = cv2.cvtColor(color_roi, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    low_sat = saturation < int(params.max_saturation)
    black = np.logical_and(low_sat, value <= int(params.black_v_max))
    white = np.logical_and(low_sat, value >= int(params.white_v_min))
    low_saturation_ratio = float(np.count_nonzero(low_sat) / max(1, low_sat.size))
    black_white_support = float(
        (np.count_nonzero(black) + np.count_nonzero(white)) / max(1, low_sat.size)
    )

    reasons = []
    if valid_depth_ratio < float(params.min_valid_depth_ratio):
        reasons.append(
            f"depth ratio {valid_depth_ratio:.3f} below {float(params.min_valid_depth_ratio):.3f}"
        )
    if low_saturation_ratio < float(params.min_low_saturation_ratio):
        reasons.append(
            f"saturation ratio {low_saturation_ratio:.3f} below {float(params.min_low_saturation_ratio):.3f}"
        )
    if black_white_support < float(params.min_black_white_support):
        reasons.append(
            f"black/white support {black_white_support:.3f} below {float(params.min_black_white_support):.3f}"
        )

    return YoloRoiGateResult(
        success=not reasons,
        valid_depth_ratio=valid_depth_ratio,
        low_saturation_ratio=low_saturation_ratio,
        black_white_support=black_white_support,
        roi_area_px=roi_area,
        reason="; ".join(reasons),
    )


def _clip_bbox(bbox_xyxy, width, height):
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox_xyxy[:4]]
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    return x1, y1, x2, y2


def _result(success, reason, roi_area=0):
    return YoloRoiGateResult(
        success=success,
        valid_depth_ratio=0.0,
        low_saturation_ratio=0.0,
        black_white_support=0.0,
        roi_area_px=int(roi_area),
        reason=reason,
    )
