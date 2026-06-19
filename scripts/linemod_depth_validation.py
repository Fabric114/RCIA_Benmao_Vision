from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class DepthValidationResult:
    success: bool
    valid_depth_count: int
    observed_median_depth_m: float | None
    pose_center_depth_m: float
    abs_z_error_m: float | None
    reason: str = ""

    def to_dict(self):
        return asdict(self)


def depth_image_to_meters(depth_image, depth_scale=0.001):
    depth = np.asarray(depth_image)
    if depth.dtype.kind in ("u", "i"):
        return depth.astype(np.float64) * float(depth_scale)
    depth_m = depth.astype(np.float64)
    finite = depth_m[np.isfinite(depth_m)]
    if finite.size and float(np.nanmax(finite)) > 20.0:
        return depth_m * float(depth_scale)
    return depth_m


def _feature_depth_samples(depth_m, match, template, window_radius=1):
    height, width = depth_m.shape[:2]
    samples = []
    for feature in template.features:
        u = int(round(float(match.x) + float(feature.x)))
        v = int(round(float(match.y) + float(feature.y)))
        x0 = max(0, u - int(window_radius))
        x1 = min(width, u + int(window_radius) + 1)
        y0 = max(0, v - int(window_radius))
        y1 = min(height, v + int(window_radius) + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        patch = depth_m[y0:y1, x0:x1]
        valid = patch[np.isfinite(patch) & (patch > 0.0)]
        if valid.size:
            samples.append(float(np.median(valid)))
    return np.asarray(samples, dtype=np.float64)


def median_depth_in_window(depth_image, center_u, center_v, depth_scale=0.001, radius=3):
    depth_m = depth_image_to_meters(depth_image, depth_scale=depth_scale)
    height, width = depth_m.shape[:2]
    u = int(round(float(center_u)))
    v = int(round(float(center_v)))
    x0 = max(0, u - int(radius))
    x1 = min(width, u + int(radius) + 1)
    y0 = max(0, v - int(radius))
    y1 = min(height, v + int(radius) + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    patch = depth_m[y0:y1, x0:x1]
    valid = patch[np.isfinite(patch) & (patch > 0.0)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def validate_depth_support(
    depth_image,
    match,
    template,
    object_in_camera,
    k_matrix=None,
    depth_scale=0.001,
    max_abs_z_error_m=0.06,
    min_valid_depths=8,
    window_radius=1,
):
    del k_matrix
    pose = np.asarray(object_in_camera, dtype=np.float64).reshape(4, 4)
    pose_center_depth_m = float(pose[2, 3])
    if pose_center_depth_m <= 0.0:
        return DepthValidationResult(
            success=False,
            valid_depth_count=0,
            observed_median_depth_m=None,
            pose_center_depth_m=pose_center_depth_m,
            abs_z_error_m=None,
            reason="Pose center depth is not in front of the camera",
        )

    depth_m = depth_image_to_meters(depth_image, depth_scale=depth_scale)
    samples = _feature_depth_samples(depth_m, match, template, window_radius=window_radius)
    valid_count = int(samples.size)
    if valid_count < int(min_valid_depths):
        return DepthValidationResult(
            success=False,
            valid_depth_count=valid_count,
            observed_median_depth_m=None,
            pose_center_depth_m=pose_center_depth_m,
            abs_z_error_m=None,
            reason=f"Need at least {int(min_valid_depths)} valid depth samples, got {valid_count}",
        )

    observed_median = float(np.median(samples))
    abs_z_error = abs(observed_median - pose_center_depth_m)
    if abs_z_error > float(max_abs_z_error_m):
        return DepthValidationResult(
            success=False,
            valid_depth_count=valid_count,
            observed_median_depth_m=observed_median,
            pose_center_depth_m=pose_center_depth_m,
            abs_z_error_m=abs_z_error,
            reason=(
                f"Depth median differs from pose z by {abs_z_error:.4f} m, "
                f"limit {float(max_abs_z_error_m):.4f} m"
            ),
        )

    return DepthValidationResult(
        success=True,
        valid_depth_count=valid_count,
        observed_median_depth_m=observed_median,
        pose_center_depth_m=pose_center_depth_m,
        abs_z_error_m=abs_z_error,
    )
