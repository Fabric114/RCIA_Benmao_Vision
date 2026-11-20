import logging
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
from learning.ov_model import OV_PRECISIONS  # noqa: E402
from learning.training.predict_pose_refine import PoseRefinePredictor  # noqa: E402
from learning.training.predict_score import ScorePredictor  # noqa: E402
from Utils import NVDIFFRAST_IMPORT_ERROR, dr, set_logging_format, set_seed  # noqa: E402
from moderngl_renderer import ModernglBatchRenderer  # noqa: E402


@dataclass
class EstimatorConfig:
    mesh_file: str
    debug: int = 0
    debug_dir: str = "runtime/foundationpose_debug"
    symmetry_mode: str = "none"
    device: str = "cuda"
    net_backend: str = "torch"
    ov_device: str = "CPU"
    ov_precision: str = "fp32"  # "fp32" | "fp16" | "int8"
    ov_num_threads: int | None = None  # None 表示由 OpenVINO 决定
    render_backend: str = "nvdiffrast"
    warmup: bool = True
    ov_cache_dir: str | None = None

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
    if cfg.render_backend not in ("nvdiffrast", "moderngl"):
        raise ValueError(f"unknown render_backend: {cfg.render_backend}")
    if cfg.render_backend == "nvdiffrast" and cfg.device != "cuda":
        raise ValueError("render_backend=nvdiffrast requires device=cuda")
    if cfg.render_backend == "nvdiffrast" and dr is None:
        raise ImportError(
            "nvdiffrast 导入失败，请先 `source scripts/foundationpose_env.sh` 再运行"
        ) from NVDIFFRAST_IMPORT_ERROR
    if cfg.net_backend not in ("torch", "openvino"):
        raise ValueError(f"unknown net_backend: {cfg.net_backend}")
    if cfg.ov_device not in ("CPU", "GPU"):
        raise ValueError(f"unknown ov_device: {cfg.ov_device}")
    if cfg.ov_precision not in OV_PRECISIONS:
        raise ValueError(f"unknown ov_precision: {cfg.ov_precision}, expected one of {OV_PRECISIONS}")
    # 下面两项校验放在加载 IR 之前，IR 不存在时也能报出配置错误。
    if cfg.net_backend == "openvino" and cfg.ov_precision == "fp16" and cfg.ov_device != "GPU":
        # CPU 上设置 f16 不报错，但实际仍按 f32 运行（实测），所以直接拒绝。
        raise ValueError("ov_precision=fp16 只支持 ov_device=GPU")
    if cfg.net_backend == "openvino" and cfg.ov_num_threads is not None and cfg.ov_device == "GPU":
        # 核显插件不接受 INFERENCE_NUM_THREADS，编译会报错（实测）。
        raise ValueError("ov_num_threads 只能用于 ov_device=CPU")
    symmetry_tfs = symmetry_tfs_for_mode(cfg.symmetry_mode)
    if cfg.render_backend == "moderngl":
        glctx = ModernglBatchRenderer(backend="egl")
    elif cfg.device == "cuda":
        glctx = dr.RasterizeCudaContext()
    else:
        glctx = None
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
        scorer=ScorePredictor(device=cfg.device, net_backend=cfg.net_backend, ov_device=cfg.ov_device,
                              ov_cache_dir=cfg.ov_cache_dir, ov_precision=cfg.ov_precision,
                              ov_num_threads=cfg.ov_num_threads),
        refiner=PoseRefinePredictor(device=cfg.device, net_backend=cfg.net_backend, ov_device=cfg.ov_device,
                                    ov_cache_dir=cfg.ov_cache_dir, ov_precision=cfg.ov_precision,
                                    ov_num_threads=cfg.ov_num_threads),
        debug_dir=cfg.debug_dir,
        debug=int(cfg.debug),
        glctx=glctx,
        device=cfg.device,
    )
    if cfg.warmup:
        _warmup_estimator(estimator, cfg)
    return EstimatorBundle(
        estimator=estimator,
        mesh=mesh,
        to_origin=to_origin,
        bbox_vis=bbox_vis,
        model_center=np.asarray(estimator.model_center, dtype=np.float64).copy(),
    )


def _warmup_estimator(estimator: FoundationPose, cfg: EstimatorConfig) -> None:
    """启动时提前完成首帧需要的一次性开销：OpenVINO 静态图编译，以及 moderngl 的 FBO/VBO/VAO 分配。"""
    n_hypo = len(estimator.rot_grid)
    if cfg.net_backend == "openvino" and cfg.ov_device == "GPU":
        # 核显每个 batch 大小的编译约 2 秒。不预热的话，这些编译会全部算进第一次 register。
        # refine 同时预热 batch=1（track_one 使用）。CPU 路径不预热，保持原行为。
        estimator.refiner.model.warmup((n_hypo, 1))
        estimator.scorer.model.warmup((n_hypo,))
    if cfg.render_backend == "moderngl":
        # ModernglBatchRenderer 只保留一个尺寸的 FBO，所以只渲染一次 register 的完整批次。
        # FBO 尺寸由 批大小 和 input_resize 决定，与相机分辨率无关，这里用占位内参即可。
        k_dummy = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
        resize = tuple(int(v) for v in estimator.refiner.cfg["input_resize"])
        estimator.glctx.render(K=k_dummy, H=480, W=640, ob_in_cams=estimator.rot_grid,
                               mesh_tensors=estimator.mesh_tensors, output_size=resize)
    logging.info(f"estimator warmup done: n_hypo={n_hypo}, net={cfg.net_backend}, render={cfg.render_backend}")
