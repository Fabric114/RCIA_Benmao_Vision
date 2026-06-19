import math

import numpy as np


def _normalize(vector):
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector)
    if norm <= 1e-12:
        raise ValueError("Cannot normalize a near-zero vector")
    return vector / norm


def sample_hemisphere_views(radius, azimuth_count, elevation_degrees):
    if radius <= 0:
        raise ValueError("radius must be positive")
    if azimuth_count <= 0:
        raise ValueError("azimuth_count must be positive")

    views = []
    for elevation_deg in elevation_degrees:
        elevation = math.radians(float(elevation_deg))
        z = -radius * math.cos(elevation)
        xy_radius = radius * math.sin(elevation)
        for i in range(int(azimuth_count)):
            azimuth = 2.0 * math.pi * float(i) / float(azimuth_count)
            x = xy_radius * math.cos(azimuth)
            y = xy_radius * math.sin(azimuth)
            views.append([x, y, z])
    return np.asarray(views, dtype=np.float64)


def look_at_object_in_camera(camera_position_object, target_object=None, up_object=None):
    camera_position_object = np.asarray(camera_position_object, dtype=np.float64).reshape(3)
    target_object = (
        np.zeros(3, dtype=np.float64)
        if target_object is None
        else np.asarray(target_object, dtype=np.float64).reshape(3)
    )
    up_object = (
        np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if up_object is None
        else np.asarray(up_object, dtype=np.float64).reshape(3)
    )

    z_axis_object = _normalize(target_object - camera_position_object)
    x_axis_object = np.cross(up_object, z_axis_object)
    if np.linalg.norm(x_axis_object) <= 1e-9:
        x_axis_object = np.cross(np.array([1.0, 0.0, 0.0]), z_axis_object)
    x_axis_object = _normalize(x_axis_object)
    y_axis_object = _normalize(np.cross(z_axis_object, x_axis_object))

    rotation_camera_from_object = np.stack(
        [x_axis_object, y_axis_object, z_axis_object], axis=0
    )
    translation_camera_from_object = -rotation_camera_from_object @ camera_position_object

    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation_camera_from_object
    pose[:3, 3] = translation_camera_from_object
    return pose


def backproject_pixel(u, v, depth_m, k_matrix):
    if depth_m <= 0:
        raise ValueError("depth_m must be positive")
    k_matrix = np.asarray(k_matrix, dtype=np.float64).reshape(3, 3)
    fx = k_matrix[0, 0]
    fy = k_matrix[1, 1]
    cx = k_matrix[0, 2]
    cy = k_matrix[1, 2]
    if abs(fx) <= 1e-12 or abs(fy) <= 1e-12:
        raise ValueError("camera intrinsics must have non-zero focal lengths")
    x = (float(u) - cx) * float(depth_m) / fx
    y = (float(v) - cy) * float(depth_m) / fy
    return np.array([x, y, float(depth_m)], dtype=np.float64)


def replace_pose_translation_from_depth(
    template_pose, center_u, center_v, depth_m, k_matrix
):
    pose = np.asarray(template_pose, dtype=np.float64).reshape(4, 4).copy()
    pose[:3, 3] = backproject_pixel(center_u, center_v, depth_m, k_matrix)
    pose[3, :] = [0.0, 0.0, 0.0, 1.0]
    return pose


def transform_model_points(points_model, object_in_camera):
    points_model = np.asarray(points_model, dtype=np.float64).reshape(-1, 3)
    object_in_camera = np.asarray(object_in_camera, dtype=np.float64).reshape(4, 4)
    homogeneous = np.concatenate([points_model, np.ones((points_model.shape[0], 1))], axis=1)
    return (object_in_camera @ homogeneous.T).T[:, :3]


def project_camera_points(points_camera, k_matrix):
    points_camera = np.asarray(points_camera, dtype=np.float64).reshape(-1, 3)
    k_matrix = np.asarray(k_matrix, dtype=np.float64).reshape(3, 3)
    if np.any(points_camera[:, 2] <= 0.0):
        raise ValueError("all camera points must have positive z")
    projected = (k_matrix @ points_camera.T).T
    return projected[:, :2] / projected[:, 2:3]


def project_model_points(points_model, object_in_camera, k_matrix):
    return project_camera_points(transform_model_points(points_model, object_in_camera), k_matrix)


def mean_reprojection_error(points_model, image_points, object_in_camera, k_matrix):
    projected = project_model_points(points_model, object_in_camera, k_matrix)
    image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    if projected.shape != image_points.shape:
        raise ValueError("points_model and image_points must contain the same number of points")
    return float(np.mean(np.linalg.norm(projected - image_points, axis=1)))


def object_diameter(points_model):
    points_model = np.asarray(points_model, dtype=np.float64).reshape(-1, 3)
    if points_model.shape[0] == 0:
        raise ValueError("points_model must contain at least one point")
    extents = np.max(points_model, axis=0) - np.min(points_model, axis=0)
    return float(np.linalg.norm(extents))


def add_error(points_model, expected_object_in_camera, actual_object_in_camera):
    expected_points = transform_model_points(points_model, expected_object_in_camera)
    actual_points = transform_model_points(points_model, actual_object_in_camera)
    return float(np.mean(np.linalg.norm(expected_points - actual_points, axis=1)))
