import numpy as np


def pose_center(pose: np.ndarray, model_center: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    return pose[:3, :3] @ np.asarray(model_center, dtype=np.float64) + pose[:3, 3]


def pose_errors(pose_ref: np.ndarray, pose_test: np.ndarray, model_center: np.ndarray) -> tuple[float, float]:
    """返回 (中心误差米, 长轴夹角度)。长轴是物体 z 轴，两端不对称，不取绝对值。"""
    pose_ref = np.asarray(pose_ref, dtype=np.float64)
    pose_test = np.asarray(pose_test, dtype=np.float64)
    center_err = float(np.linalg.norm(pose_center(pose_ref, model_center) - pose_center(pose_test, model_center)))
    axis_ref = pose_ref[:3, 2] / np.linalg.norm(pose_ref[:3, 2])
    axis_test = pose_test[:3, 2] / np.linalg.norm(pose_test[:3, 2])
    axis_err = float(np.degrees(np.arccos(np.clip(np.dot(axis_ref, axis_test), -1.0, 1.0))))
    return center_err, axis_err
