from dataclasses import dataclass

import cv2
import numpy as np

from scripts.linemod_pose_utils import mean_reprojection_error


@dataclass(frozen=True)
class PnPResult:
    success: bool
    object_in_camera: np.ndarray | None
    inlier_count: int
    mean_reprojection_error_px: float
    reason: str = ""


def _collect_correspondences(match, template):
    object_points = []
    image_points = []
    for feature in template.features:
        if feature.model_point is None:
            continue
        object_points.append(feature.model_point)
        image_points.append([float(match.x) + float(feature.x), float(match.y) + float(feature.y)])
    return (
        np.asarray(object_points, dtype=np.float64).reshape(-1, 3),
        np.asarray(image_points, dtype=np.float64).reshape(-1, 2),
    )


def _pose_from_rvec_tvec(rvec, tvec):
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return pose


def estimate_pose_from_match(
    match,
    template,
    k_matrix,
    reprojection_error_px=8.0,
    min_correspondences=6,
):
    object_points, image_points = _collect_correspondences(match, template)
    if object_points.shape[0] < int(min_correspondences):
        return PnPResult(
            success=False,
            object_in_camera=None,
            inlier_count=0,
            mean_reprojection_error_px=float("inf"),
            reason=f"Need at least {int(min_correspondences)} correspondences, got {object_points.shape[0]}",
        )

    k_matrix = np.asarray(k_matrix, dtype=np.float64).reshape(3, 3)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=object_points.astype(np.float32),
        imagePoints=image_points.astype(np.float32),
        cameraMatrix=k_matrix,
        distCoeffs=np.zeros((4, 1), dtype=np.float64),
        iterationsCount=100,
        reprojectionError=float(reprojection_error_px),
        confidence=0.99,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok or inliers is None or len(inliers) < int(min_correspondences):
        inlier_count = 0 if inliers is None else int(len(inliers))
        return PnPResult(
            success=False,
            object_in_camera=None,
            inlier_count=inlier_count,
            mean_reprojection_error_px=float("inf"),
            reason="solvePnPRansac failed to find a valid pose",
        )

    pose = _pose_from_rvec_tvec(rvec, tvec)
    inlier_indices = np.asarray(inliers, dtype=np.int32).reshape(-1)
    reprojection_error = mean_reprojection_error(
        object_points[inlier_indices],
        image_points[inlier_indices],
        pose,
        k_matrix,
    )
    if pose[2, 3] <= 0.0:
        return PnPResult(
            success=False,
            object_in_camera=None,
            inlier_count=int(len(inlier_indices)),
            mean_reprojection_error_px=reprojection_error,
            reason="Recovered pose is behind the camera",
        )

    return PnPResult(
        success=True,
        object_in_camera=pose,
        inlier_count=int(len(inlier_indices)),
        mean_reprojection_error_px=reprojection_error,
    )
