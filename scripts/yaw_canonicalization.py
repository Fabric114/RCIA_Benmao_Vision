import numpy as np

REF_TOWARD_CAMERA = np.array([0.0, 0.0, -1.0])
REF_IMAGE_UP = np.array([0.0, -1.0, 0.0])
PREVIOUS_X_MAX_ALIGNMENT = 0.9


def mesh_center_from_vertices(vertices: np.ndarray) -> np.ndarray:
    v = np.asarray(vertices, dtype=np.float64)
    return (v.min(axis=0) + v.max(axis=0)) / 2.0


def canonicalize_yaw(pose: np.ndarray, model_center: np.ndarray,
                     fallback_deg: float = 15.0,
                     previous_x: np.ndarray | None = None) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    center = np.asarray(model_center, dtype=np.float64)

    centered = pose.copy()
    centered[:3, 3] = pose[:3, :3] @ center + pose[:3, 3]

    z = centered[:3, 2] / np.linalg.norm(centered[:3, 2])
    ref = REF_TOWARD_CAMERA
    if abs(np.dot(ref, z)) > np.cos(np.radians(fallback_deg)):
        ref = REF_IMAGE_UP
        if previous_x is not None:
            prev = np.asarray(previous_x, dtype=np.float64)
            if abs(np.dot(prev, z)) < PREVIOUS_X_MAX_ALIGNMENT:
                ref = prev

    x = ref - np.dot(ref, z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    centered[:3, 0] = x
    centered[:3, 1] = y
    centered[:3, 2] = z

    out = centered.copy()
    out[:3, 3] = centered[:3, 3] - centered[:3, :3] @ center
    return out


class YawCanonicalizer:
    def __init__(self, model_center: np.ndarray, fallback_deg: float = 15.0):
        self.model_center = np.asarray(model_center, dtype=np.float64)
        self.fallback_deg = float(fallback_deg)
        self.previous_x = None

    def apply(self, pose: np.ndarray) -> np.ndarray:
        out = canonicalize_yaw(pose, self.model_center, self.fallback_deg, self.previous_x)
        self.previous_x = out[:3, 0].copy()
        return out

    def reset(self) -> None:
        self.previous_x = None
