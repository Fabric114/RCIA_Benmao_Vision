import cv2
import numpy as np

from scripts.linemod_line2d import Line2DFeature


INVALID_ORIENTATION = 255


def compute_depth_normal_image(depth_image, k_matrix, depth_scale=0.001):
    depth = np.asarray(depth_image)
    if depth.dtype.kind in ("u", "i"):
        depth_m = depth.astype(np.float64) * float(depth_scale)
    else:
        depth_m = depth.astype(np.float64)
        finite = depth_m[np.isfinite(depth_m)]
        if finite.size and float(np.nanmax(finite)) > 20.0:
            depth_m = depth_m * float(depth_scale)

    k_matrix = np.asarray(k_matrix, dtype=np.float64).reshape(3, 3)
    fx, fy = float(k_matrix[0, 0]), float(k_matrix[1, 1])
    cx, cy = float(k_matrix[0, 2]), float(k_matrix[1, 2])
    height, width = depth_m.shape[:2]
    u_grid, v_grid = np.meshgrid(np.arange(width), np.arange(height))
    points = np.zeros((height, width, 3), dtype=np.float64)
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    points[:, :, 0] = (u_grid - cx) * depth_m / fx
    points[:, :, 1] = (v_grid - cy) * depth_m / fy
    points[:, :, 2] = depth_m

    normal_image = np.full((height, width, 3), np.nan, dtype=np.float64)
    center_valid = (
        valid[1:-1, 1:-1]
        & valid[1:-1, :-2]
        & valid[1:-1, 2:]
        & valid[:-2, 1:-1]
        & valid[2:, 1:-1]
    )
    dx = points[1:-1, 2:] - points[1:-1, :-2]
    dy = points[2:, 1:-1] - points[:-2, 1:-1]
    normals = np.cross(dx, dy)
    norm = np.linalg.norm(normals, axis=2)
    center_valid &= norm > 1e-9
    normals[center_valid] = normals[center_valid] / norm[center_valid, None]
    normal_image[1:-1, 1:-1][center_valid] = normals[center_valid]
    return normal_image


def quantize_normal_orientations(normal_image):
    normal_image = np.asarray(normal_image, dtype=np.float64)
    if normal_image.ndim != 3 or normal_image.shape[2] != 3:
        raise ValueError("normal_image must have shape HxWx3")
    quantized = np.full(normal_image.shape[:2], INVALID_ORIENTATION, dtype=np.uint8)
    valid = np.all(np.isfinite(normal_image), axis=2)
    xy_norm = np.linalg.norm(normal_image[:, :, :2], axis=2)
    valid &= xy_norm > 1e-9
    angle = np.degrees(np.arctan2(normal_image[:, :, 1], normal_image[:, :, 0]))
    bins = np.floor(((angle[valid] + 22.5) % 360.0) / 45.0).astype(np.uint8)
    quantized[valid] = bins
    return quantized


def extract_depth_normal_features(
    normal_image,
    object_mask,
    bbox,
    model_point_image=None,
    max_features=64,
    feature_spacing=8,
    erosion_iterations=2,
):
    mask = (np.asarray(object_mask) > 0).astype(np.uint8)
    if erosion_iterations > 0:
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.erode(mask, kernel, iterations=int(erosion_iterations))
    quantized = quantize_normal_orientations(normal_image)
    valid = (mask > 0) & (quantized != INVALID_ORIENTATION)

    if model_point_image is not None:
        model_point_image = np.asarray(model_point_image, dtype=np.float64)
        if model_point_image.shape[:2] != mask.shape or model_point_image.shape[2] != 3:
            raise ValueError("model_point_image must have shape HxWx3 matching object_mask")
        valid &= np.all(np.isfinite(model_point_image), axis=2)

    stable_distance = cv2.distanceTransform(valid.astype(np.uint8), cv2.DIST_L2, 3)
    ys, xs = np.where(valid)
    x0, y0, _, _ = tuple(int(value) for value in bbox)
    selected = []
    spacing_sq = int(feature_spacing) * int(feature_spacing)
    candidates = sorted(
        zip(xs.tolist(), ys.tolist()),
        key=lambda point: (
            float(stable_distance[point[1], point[0]]),
            -abs(int(point[0]) - int(x0)),
            -abs(int(point[1]) - int(y0)),
        ),
        reverse=True,
    )
    for px, py in candidates:
        if len(selected) >= int(max_features):
            break
        if any(
            (px - sx) * (px - sx) + (py - sy) * (py - sy) < spacing_sq
            for sx, sy, _ in selected
        ):
            continue
        model_point = None
        if model_point_image is not None:
            model_point = tuple(float(value) for value in model_point_image[py, px].tolist())
        selected.append((px, py, model_point))

    return [
        Line2DFeature(
            x=int(px - x0),
            y=int(py - y0),
            orientation=int(quantized[py, px]),
            model_point=model_point,
        )
        for px, py, model_point in selected
    ]
