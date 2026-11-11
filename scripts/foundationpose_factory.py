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
from Utils import dr, set_logging_format, set_seed  # noqa: E402


@dataclass
class EstimatorConfig:
    mesh_file: str
    debug: int = 0
    debug_dir: str = "runtime/foundationpose_debug"


@dataclass
class EstimatorBundle:
    estimator: FoundationPose
    mesh: trimesh.Trimesh
    to_origin: np.ndarray
    bbox_vis: np.ndarray
    model_center: np.ndarray


def build_estimator(cfg: EstimatorConfig) -> EstimatorBundle:
    set_logging_format()
    set_seed(0)
    mesh = trimesh.load(cfg.mesh_file, process=False)
    if mesh.vertex_normals is None:
        mesh.compute_vertex_normals()
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox_vis = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
    estimator = FoundationPose(
        model_pts=mesh.vertices.copy(),
        model_normals=mesh.vertex_normals.copy(),
        mesh=mesh,
        scorer=ScorePredictor(),
        refiner=PoseRefinePredictor(),
        debug_dir=cfg.debug_dir,
        debug=int(cfg.debug),
        glctx=dr.RasterizeCudaContext(),
    )
    return EstimatorBundle(
        estimator=estimator,
        mesh=mesh,
        to_origin=to_origin,
        bbox_vis=bbox_vis,
        model_center=np.asarray(estimator.model_center, dtype=np.float64).copy(),
    )
