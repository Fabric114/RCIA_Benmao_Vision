import math
from dataclasses import dataclass

import numpy as np

END_BAND_FRAC = 0.15
RADIAL_TIE_TOL_M = 1e-4
MIN_VALID_DEPTH_M = 1e-3


@dataclass(frozen=True, slots=True)
class PoseEndDisambiguationResult:
    pose: np.ndarray
    flipped: bool
    n_large: int
    n_small: int


def mesh_endpoints_from_vertices(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (model_center, large_tip, small_tip)，large/small 指 Z 两端中半径较大/较小的一端。"""
    v = np.asarray(vertices, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != 3 or v.shape[0] == 0:
        raise ValueError(f"vertices must have shape (N, 3) with N > 0, got {v.shape}")
    if not np.all(np.isfinite(v)):
        raise ValueError("vertices must be finite")

    v_min = v.min(axis=0)
    v_max = v.max(axis=0)
    model_center = (v_min + v_max) / 2.0
    cx, cy = model_center[0], model_center[1]
    z_min, z_max = v[:, 2].min(), v[:, 2].max()
    span_z = z_max - z_min
    if span_z <= 0:
        raise ValueError("vertices must have a positive Z span")

    # 只看两端 15% 高度内的顶点，比较它们到轴线的最大径向距离
    radial = np.hypot(v[:, 0] - cx, v[:, 1] - cy)
    band = END_BAND_FRAC * span_z
    r_max_zmin = radial[v[:, 2] <= z_min + band].max()
    r_max_zmax = radial[v[:, 2] >= z_max - band].max()

    if r_max_zmin >= r_max_zmax - RADIAL_TIE_TOL_M:
        large_z, small_z = z_min, z_max
    else:
        large_z, small_z = z_max, z_min

    large_tip = np.array([cx, cy, large_z], dtype=np.float64)
    small_tip = np.array([cx, cy, small_z], dtype=np.float64)
    return model_center, large_tip, small_tip


def flip_pose_around_center(pose: np.ndarray, model_center: np.ndarray) -> np.ndarray:
    """绕模型中心翻转 180 度（局部 Y、Z 取反），保持中心在相机系的位置和 det(R)=+1。"""
    pose = np.asarray(pose, dtype=np.float64)
    center = np.asarray(model_center, dtype=np.float64).reshape(3)

    t_pos = np.eye(4, dtype=np.float64)
    t_pos[:3, 3] = center
    t_neg = np.eye(4, dtype=np.float64)
    t_neg[:3, 3] = -center
    flip = np.diag([1.0, -1.0, -1.0, 1.0])

    return pose @ t_pos @ flip @ t_neg


def count_points_in_sphere(depth_m: np.ndarray, K: np.ndarray,
                           center_cam: np.ndarray, radius_m: float) -> int:
    """统计深度图中落在相机系球体内的像素数。只在投影 ROI 内反投影，避免整幅图计算。"""
    if not (math.isfinite(radius_m) and radius_m > 0):
        raise ValueError(f"radius_m must be positive and finite, got {radius_m}")
    depth = np.asarray(depth_m, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError(f"depth_m must be 2D, got shape {depth.shape}")
    K = np.asarray(K, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"K must have shape (3, 3), got {K.shape}")

    center = np.asarray(center_cam, dtype=np.float64).reshape(-1)
    cx, cy, cz = center[0], center[1], center[2]
    if not np.all(np.isfinite([cx, cy, cz])) or cz <= -radius_m:
        return 0

    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    px, py = K[0, 2], K[1, 2]
    if not (math.isfinite(fx) and fx > 0 and math.isfinite(fy) and fy > 0):
        raise ValueError(f"focal lengths must be positive and finite, got fx={fx}, fy={fy}")
    z_lo = max(MIN_VALID_DEPTH_M, cz - radius_m)
    z_hi = cz + radius_m

    if cz <= radius_m:
        # 球体贴近或穿过相机光心，投影范围不可控，使用整幅图
        r0, r1, c0, c1 = 0, H, 0, W
    else:
        # 用圆锥切线斜率求球体在归一化平面上 x/z、y/z 的精确包络（避免大偏角下近似偏小）
        denom_z = cz * cz - radius_m * radius_m
        r2 = radius_m * radius_m
        sx = math.sqrt(max(0.0, cx * cx + denom_z))
        sy = math.sqrt(max(0.0, cy * cy + denom_z))
        mx_min = (cx * cz - radius_m * sx) / denom_z
        mx_max = (cx * cz + radius_m * sx) / denom_z
        my_min = (cy * cz - radius_m * sy) / denom_z
        my_max = (cy * cz + radius_m * sy) / denom_z
        c0 = int(max(0, math.floor(fx * mx_min + px)))
        c1 = int(min(W, math.ceil(fx * mx_max + px) + 1))
        r0 = int(max(0, math.floor(fy * my_min + py)))
        r1 = int(min(H, math.ceil(fy * my_max + py) + 1))

    if c1 <= c0 or r1 <= r0:
        return 0

    sub = depth[r0:r1, c0:c1]
    rows, cols = np.nonzero(np.isfinite(sub) & (sub >= z_lo) & (sub <= z_hi))
    if rows.size == 0:
        return 0

    z = sub[rows, cols]
    u = cols + c0
    v = rows + r0
    x = (u - px) * z / fx
    y = (v - py) * z / fy
    d2 = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
    return int(np.count_nonzero(d2 <= radius_m ** 2))


def disambiguate_zone_pose(pose: np.ndarray, depth_m: np.ndarray, K: np.ndarray,
                           model_center: np.ndarray, large_tip: np.ndarray,
                           small_tip: np.ndarray,
                           radius_m: float = 0.14) -> PoseEndDisambiguationResult:
    """用深度点数判断大端朝向：大端周围点更多则翻转位姿，平局不翻转。"""
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError("pose must be a finite (4, 4) matrix")
    K = np.asarray(K, dtype=np.float64)
    if K.shape != (3, 3) or not np.all(np.isfinite(K)):
        raise ValueError("K must be a finite (3, 3) matrix")
    if not (math.isfinite(radius_m) and radius_m > 0):
        raise ValueError(f"radius_m must be positive and finite, got {radius_m}")

    large_cam = pose[:3, :3] @ np.asarray(large_tip, dtype=np.float64).reshape(3) + pose[:3, 3]
    small_cam = pose[:3, :3] @ np.asarray(small_tip, dtype=np.float64).reshape(3) + pose[:3, 3]

    n_large = count_points_in_sphere(depth_m, K, large_cam, radius_m)
    n_small = count_points_in_sphere(depth_m, K, small_cam, radius_m)

    flipped = n_large > n_small
    if flipped:
        out_pose = flip_pose_around_center(pose, model_center)
    else:
        out_pose = pose.copy()

    return PoseEndDisambiguationResult(pose=out_pose, flipped=flipped,
                                       n_large=n_large, n_small=n_small)
