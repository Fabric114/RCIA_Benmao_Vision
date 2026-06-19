import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _validate_pose_matrix(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError("object_in_camera must be a 4x4 homogeneous matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("object_in_camera must contain finite values")
    if not np.allclose(matrix[3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError("object_in_camera must have homogeneous last row")
    return matrix


@dataclass
class TemplatePoseMetadata:
    class_id: str
    template_id: int
    object_in_camera: np.ndarray
    match_center: list
    source: str = ""

    def __post_init__(self):
        self.class_id = str(self.class_id)
        self.template_id = int(self.template_id)
        self.object_in_camera = _validate_pose_matrix(self.object_in_camera)
        if len(self.match_center) != 2:
            raise ValueError("match_center must contain [u, v]")
        self.match_center = [float(self.match_center[0]), float(self.match_center[1])]
        self.source = str(self.source)

    def to_dict(self):
        return {
            "class_id": self.class_id,
            "template_id": self.template_id,
            "object_in_camera": self.object_in_camera.tolist(),
            "match_center": self.match_center,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            class_id=data["class_id"],
            template_id=data["template_id"],
            object_in_camera=data["object_in_camera"],
            match_center=data["match_center"],
            source=data.get("source", ""),
        )


def save_template_metadata(path, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "templates": [item.to_dict() for item in metadata],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_template_metadata(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("version") != 1:
        raise ValueError("Unsupported LINEMOD template metadata version")
    return [TemplatePoseMetadata.from_dict(item) for item in payload["templates"]]


def find_template_metadata(metadata, class_id, template_id):
    for item in metadata:
        if item.class_id == str(class_id) and item.template_id == int(template_id):
            return item
    raise KeyError(f"No template metadata for class_id={class_id}, template_id={template_id}")
