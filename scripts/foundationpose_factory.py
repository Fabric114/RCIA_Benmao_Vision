import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FOUNDATIONPOSE_DIR = PROJECT_ROOT / "foundation_pose"
if str(FOUNDATIONPOSE_DIR) not in sys.path:
    sys.path.insert(0, str(FOUNDATIONPOSE_DIR))

from estimater import FoundationPose  # noqa: E402
from learning.training.predict_pose_refine import PoseRefinePredictor  # noqa: E402
from learning.training.predict_score import ScorePredictor  # noqa: E402
from Utils import NVDIFFRAST_IMPORT_ERROR, dr, set_logging_format, set_seed  # noqa: E402


@dataclass
class EstimatorConfig:
    mesh_file: str
    debug: int = 0
    debug_dir: str = "runtime/foundationpose_debug"
    symmetry_mode: str = "none"
    device: str = "cuda"
    net_backend: str = "torch"
    ov_device: str = "CPU"

# 约束能量单元绕z旋转角共12个，每30 1
def z_symmetry_tfs(step_deg: float = 30.0) -> np.ndarray:
    angles = np.radians(np.arange(0.0, 360.0, step_deg))
    tfs = np.tile(np.eye(4), (len(angles), 1, 1))
    tfs[:, 0, 0] = np.cos(angles)
    tfs[:, 0, 1] = -np.sin(angles)
    tfs[:, 1, 0] = np.sin(angles)
    tfs[:, 1, 1] = np.cos(angles)
    return tfs


def symmetry_tfs_for_mode(mode: str):
    if mode == "none":
        return None
    if mode == "z12":
        return z_symmetry_tfs(30.0)
    raise ValueError(f"unknown symmetry_mode: {mode}")


@dataclass
class EstimatorBundle:
    estimator: FoundationPose
    mesh: trimesh.Trimesh
    to_origin: np.ndarray
    bbox_vis: np.ndarray
    model_center: np.ndarray


def build_estimator(cfg: EstimatorConfig) -> EstimatorBundle:
    if cfg.device == "cuda" and dr is None:
        raise ImportError(
            "nvdiffrast 导入失败，请先 `source scripts/foundationpose_env.sh` 再运行"
        ) from NVDIFFRAST_IMPORT_ERROR
    if cfg.net_backend not in ("torch", "openvino"):
        raise ValueError(f"unknown net_backend: {cfg.net_backend}")
    if cfg.ov_device not in ("CPU", "GPU"):
        raise ValueError(f"unknown ov_device: {cfg.ov_device}")
    symmetry_tfs = symmetry_tfs_for_mode(cfg.symmetry_mode)
    set_logging_format()
    set_seed(0)
    mesh = trimesh.load(cfg.mesh_file, process=False)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox_vis = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
    estimator = FoundationPose(
        model_pts=mesh.vertices.copy(),
        model_normals=mesh.vertex_normals.copy(),
        symmetry_tfs=symmetry_tfs,
        mesh=mesh,
        scorer=ScorePredictor(device=cfg.device, net_backend=cfg.net_backend, ov_device=cfg.ov_device),
        refiner=PoseRefinePredictor(device=cfg.device, net_backend=cfg.net_backend, ov_device=cfg.ov_device),
        debug_dir=cfg.debug_dir,
        debug=int(cfg.debug),
        glctx=dr.RasterizeCudaContext() if cfg.device == "cuda" else None,
        device=cfg.device,
    )
    return EstimatorBundle(
        estimator=estimator,
        mesh=mesh,
        to_origin=to_origin,
        bbox_vis=bbox_vis,
        model_center=np.asarray(estimator.model_center, dtype=np.float64).copy(),
    )
