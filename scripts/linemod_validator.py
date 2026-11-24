from dataclasses import asdict, dataclass

import cv2
import numpy as np

from scripts.linemod_depth_validation import depth_image_to_meters
from scripts.linemod_silhouette_renderer import render_model_buffers


@dataclass(frozen=True)
class RenderedDepthConsistencyResult:
    success: bool
    consistency_ratio: float
    rendered_pixel_count: int
    valid_overlap_count: int
    supported_pixel_count: int
    mean_abs_depth_residual_m: float | None
    reason: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class BlackWhiteAppearanceResult:
    success: bool
    low_saturation_ratio: float
    black_ratio: float
    white_ratio: float
    black_white_support_ratio: float
    sampled_pixel_count: int
    reason: str = ""

    def to_dict(self):
        return asdict(self)


def validate_black_white_appearance(
    color_bgr,
    object_mask,
    max_saturation=45,
    black_v_max=80,
    white_v_min=150,
    min_low_saturation_ratio=0.65,
    min_black_white_support=0.45,
    min_black_ratio=0.05,
    min_white_ratio=0.05,
):
    mask = np.asarray(object_mask) > 0
    sampled_count = int(np.count_nonzero(mask))
    if sampled_count == 0:
        return BlackWhiteAppearanceResult(
            success=False,
            low_saturation_ratio=0.0,
            black_ratio=0.0,
            white_ratio=0.0,
            black_white_support_ratio=0.0,
            sampled_pixel_count=0,
            reason="Projected mask has no pixels for black/white appearance validation",
        )

    hsv = cv2.cvtColor(np.asarray(color_bgr), cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1][mask]
    value = hsv[:, :, 2][mask]
    low_saturation = saturation <= float(max_saturation)
    black = low_saturation & (value <= float(black_v_max))
    white = low_saturation & (value >= float(white_v_min))

    low_saturation_ratio = float(np.count_nonzero(low_saturation)) / float(sampled_count)
    black_ratio = float(np.count_nonzero(black)) / float(sampled_count)
    white_ratio = float(np.count_nonzero(white)) / float(sampled_count)
    support_ratio = black_ratio + white_ratio
    success = (
        low_saturation_ratio >= float(min_low_saturation_ratio)
        and support_ratio >= float(min_black_white_support)
        and black_ratio >= float(min_black_ratio)
        and white_ratio >= float(min_white_ratio)
    )
    if success:
        reason = ""
    elif low_saturation_ratio < float(min_low_saturation_ratio):
        reason = (
            f"Low-saturation ratio {low_saturation_ratio:.3f} is below "
            f"{float(min_low_saturation_ratio):.3f}"
        )
    elif support_ratio < float(min_black_white_support):
        reason = (
            f"Black/white support ratio {support_ratio:.3f} is below "
            f"{float(min_black_white_support):.3f}"
        )
    elif black_ratio < float(min_black_ratio):
        reason = f"Black ratio {black_ratio:.3f} is below {float(min_black_ratio):.3f}"
    else:
        reason = f"White ratio {white_ratio:.3f} is below {float(min_white_ratio):.3f}"

    return BlackWhiteAppearanceResult(
        success=success,
        low_saturation_ratio=low_saturation_ratio,
        black_ratio=black_ratio,
        white_ratio=white_ratio,
        black_white_support_ratio=support_ratio,
        sampled_pixel_count=sampled_count,
        reason=reason,
    )


def validate_rendered_depth_consistency(
    vertices,
    faces,
    object_in_camera,
    k_matrix,
    observed_depth_image,
    image_size,
    depth_scale=0.001,
    max_depth_residual_m=0.02,
    min_consistency_ratio=0.70,
    min_overlap_pixels=50,
):
    _, rendered_mask, _, rendered_depth, _ = render_model_buffers(
        vertices,
        faces,
        object_in_camera,
        k_matrix,
        image_size=image_size,
    )
    return validate_pre_rendered_depth_consistency(
        rendered_mask,
        rendered_depth,
        observed_depth_image,
        depth_scale=depth_scale,
        max_depth_residual_m=max_depth_residual_m,
        min_consistency_ratio=min_consistency_ratio,
        min_overlap_pixels=min_overlap_pixels,
    )


def validate_pre_rendered_depth_consistency(
    rendered_mask,
    rendered_depth,
    observed_depth_image,
    depth_scale=0.001,
    max_depth_residual_m=0.02,
    min_consistency_ratio=0.70,
    min_overlap_pixels=50,
):
    rendered_valid = (np.asarray(rendered_mask) > 0) & np.isfinite(rendered_depth) & (rendered_depth > 0.0)
    rendered_count = int(np.count_nonzero(rendered_valid))
    if rendered_count == 0:
        return RenderedDepthConsistencyResult(
            success=False,
            consistency_ratio=0.0,
            rendered_pixel_count=0,
            valid_overlap_count=0,
            supported_pixel_count=0,
            mean_abs_depth_residual_m=None,
            reason="Rendered pose has no visible model pixels",
        )

    observed_depth_m = depth_image_to_meters(observed_depth_image, depth_scale=depth_scale)
    observed_valid = np.isfinite(observed_depth_m) & (observed_depth_m > 0.0)
    overlap = rendered_valid & observed_valid
    overlap_count = int(np.count_nonzero(overlap))
    if overlap_count < int(min_overlap_pixels):
        return RenderedDepthConsistencyResult(
            success=False,
            consistency_ratio=0.0,
            rendered_pixel_count=rendered_count,
            valid_overlap_count=overlap_count,
            supported_pixel_count=0,
            mean_abs_depth_residual_m=None,
            reason=f"Need at least {int(min_overlap_pixels)} rendered/observed depth overlap pixels, got {overlap_count}",
        )

    residual = np.abs(rendered_depth[overlap] - observed_depth_m[overlap])
    supported = residual <= float(max_depth_residual_m)
    supported_count = int(np.count_nonzero(supported))
    consistency_ratio = float(supported_count) / float(overlap_count)
    mean_abs_residual = float(np.mean(residual))
    success = consistency_ratio >= float(min_consistency_ratio)
    return RenderedDepthConsistencyResult(
        success=success,
        consistency_ratio=consistency_ratio,
        rendered_pixel_count=rendered_count,
        valid_overlap_count=overlap_count,
        supported_pixel_count=supported_count,
        mean_abs_depth_residual_m=mean_abs_residual,
        reason=""
        if success
        else (
            f"Rendered depth consistency ratio {consistency_ratio:.3f} "
            f"is below {float(min_consistency_ratio):.3f}"
        ),
    )
