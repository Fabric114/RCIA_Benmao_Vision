from pathlib import Path

import cv2
import numpy as np


def load_obj_mesh(path):
    vertices = []
    faces = []
    for raw_line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0] == "v" and len(parts) >= 4:
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif parts[0] == "f" and len(parts) >= 4:
            indices = []
            for token in parts[1:]:
                index_text = token.split("/")[0]
                index = int(index_text)
                if index < 0:
                    index = len(vertices) + index
                else:
                    index -= 1
                indices.append(index)
            for i in range(1, len(indices) - 1):
                faces.append([indices[0], indices[i], indices[i + 1]])
    if not vertices or not faces:
        raise ValueError(f"OBJ mesh has no vertices or faces: {path}")
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def _project_points(points_camera, k_matrix):
    projected = (k_matrix @ points_camera.T).T
    return projected[:, :2] / projected[:, 2:3]


def _rasterize_triangle_model_points(
    points_uv,
    points_camera,
    points_model,
    normal_camera,
    mask,
    depth_buffer,
    model_point_image,
    normal_image,
):
    height, width = mask.shape
    min_x = max(0, int(np.floor(np.min(points_uv[:, 0]))))
    max_x = min(width - 1, int(np.ceil(np.max(points_uv[:, 0]))))
    min_y = max(0, int(np.floor(np.min(points_uv[:, 1]))))
    max_y = min(height - 1, int(np.ceil(np.max(points_uv[:, 1]))))
    if min_x > max_x or min_y > max_y:
        return

    p0, p1, p2 = points_uv.astype(np.float64)
    denominator = (
        (p1[1] - p2[1]) * (p0[0] - p2[0])
        + (p2[0] - p1[0]) * (p0[1] - p2[1])
    )
    if abs(denominator) <= 1e-12:
        return

    xs = np.arange(min_x, max_x + 1, dtype=np.float64)
    ys = np.arange(min_y, max_y + 1, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(xs, ys)
    w0 = ((p1[1] - p2[1]) * (grid_x - p2[0]) + (p2[0] - p1[0]) * (grid_y - p2[1])) / denominator
    w1 = ((p2[1] - p0[1]) * (grid_x - p2[0]) + (p0[0] - p2[0]) * (grid_y - p2[1])) / denominator
    w2 = 1.0 - w0 - w1
    inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
    if not np.any(inside):
        return

    patch_depth = depth_buffer[min_y : max_y + 1, min_x : max_x + 1]
    interpolated_z = (
        w0 * points_camera[0, 2]
        + w1 * points_camera[1, 2]
        + w2 * points_camera[2, 2]
    )
    update = inside & (interpolated_z < patch_depth)
    if not np.any(update):
        return

    patch_depth[update] = interpolated_z[update]
    patch_mask = mask[min_y : max_y + 1, min_x : max_x + 1]
    patch_mask[update] = 255
    interpolated_model = (
        w0[..., None] * points_model[0]
        + w1[..., None] * points_model[1]
        + w2[..., None] * points_model[2]
    )
    patch_model_points = model_point_image[min_y : max_y + 1, min_x : max_x + 1]
    patch_model_points[update] = interpolated_model[update]
    patch_normals = normal_image[min_y : max_y + 1, min_x : max_x + 1]
    patch_normals[update] = normal_camera


def _face_normal(points):
    normal = np.cross(points[1] - points[0], points[2] - points[0])
    norm = np.linalg.norm(normal)
    if norm <= 1e-12:
        return None
    return normal / norm


def render_model_buffers(vertices, faces, object_in_camera, k_matrix, image_size):
    width, height = int(image_size[0]), int(image_size[1])
    color = np.zeros((height, width, 3), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    model_point_image = np.full((height, width, 3), np.nan, dtype=np.float64)
    normal_image = np.full((height, width, 3), np.nan, dtype=np.float64)
    depth_buffer = np.full((height, width), np.inf, dtype=np.float64)

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    object_in_camera = np.asarray(object_in_camera, dtype=np.float64).reshape(4, 4)
    k_matrix = np.asarray(k_matrix, dtype=np.float64).reshape(3, 3)
    homogeneous = np.concatenate([vertices, np.ones((vertices.shape[0], 1))], axis=1)
    vertices_camera = (object_in_camera @ homogeneous.T).T[:, :3]

    for face in faces:
        points_camera = vertices_camera[face]
        if np.any(points_camera[:, 2] <= 0.001):
            continue
        normal_camera = _face_normal(points_camera)
        if normal_camera is None:
            continue
        points_uv = _project_points(points_camera, k_matrix)
        if (
            np.all(points_uv[:, 0] < 0)
            or np.all(points_uv[:, 0] >= width)
            or np.all(points_uv[:, 1] < 0)
            or np.all(points_uv[:, 1] >= height)
        ):
            continue
        _rasterize_triangle_model_points(
            points_uv=points_uv,
            points_camera=points_camera,
            points_model=vertices[face],
            normal_camera=normal_camera,
            mask=mask,
            depth_buffer=depth_buffer,
            model_point_image=model_point_image,
            normal_image=normal_image,
        )

    color[mask > 0] = (255, 255, 255)
    edges = cv2.Canny(mask, 50, 150)
    color[edges > 0] = (0, 0, 0)
    depth_image = depth_buffer.copy()
    depth_image[~np.isfinite(depth_image)] = 0.0
    return color, mask, model_point_image, depth_image, normal_image


def render_silhouette_with_model_points(vertices, faces, object_in_camera, k_matrix, image_size):
    color, mask, model_point_image, _, _ = render_model_buffers(
        vertices,
        faces,
        object_in_camera,
        k_matrix,
        image_size,
    )
    return color, mask, model_point_image


def render_silhouette(vertices, faces, object_in_camera, k_matrix, image_size):
    color, mask, _ = render_silhouette_with_model_points(
        vertices,
        faces,
        object_in_camera,
        k_matrix,
        image_size,
    )
    return color, mask
