#!/usr/bin/env python3
"""FoundationPose refine/score 网络的 INT8 训练后量化（NNCF）。

collect：用部署管线（CPU + moderngl + torch + z12 + bbox_depth mask）跑校准帧，记录网络输入 (A, B)。
quantize：用记录的 (A, B) 作为校准集，把 FP32 IR 量化成 refine_int8.xml 和 score_int8.xml。
"""
import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
FOUNDATIONPOSE_DIR = PROJECT_ROOT / "foundation_pose"
if str(FOUNDATIONPOSE_DIR) not in sys.path:
    sys.path.insert(0, str(FOUNDATIONPOSE_DIR))

from learning.ov_model import refine_ir_path, score_ir_path  # noqa: E402

MESH_FILE = str(PROJECT_ROOT / "models/object/Str3D.obj")
MASK_CENTER_FRAC = 0.35  # 与 foundationpose_eval.py run 的默认值一致
DEPTH_WINDOW_HALF_M = 0.11


class _RefineReservoir(list):
    """refine 的 calib_sink：对所有 N=1 样本做蓄水池抽样，最多保留 capacity 个。

    每次 register 的每个迭代都会推理整批候选，样本总数远大于 capacity，全部保存会占用数 GB 内存。
    """

    def __init__(self, capacity, seed):
        super().__init__()
        self.capacity = capacity
        self.seen = 0
        self.rng = np.random.default_rng(seed)

    def append(self, pair):
        A, B = pair
        for i in range(A.shape[0]):
            self.seen += 1
            item = (A[i:i + 1].copy(), B[i:i + 1].copy())
            if len(self) < self.capacity:
                list.append(self, item)
            else:
                j = int(self.rng.integers(0, self.seen))
                if j < self.capacity:
                    self[j] = item


def collect_calibration(frames_root, out_dir, est_refine_iter: int = 2,
                        max_refine_samples: int = 300, seed: int = 0) -> dict[str, int]:
    """写出 out_dir/refine_calib.npz（A、B 各 (M,6,160,160)）和 out_dir/score_calib.npz
    （A_0、B_0、A_1、B_1…，每对是一帧的整批候选）。返回 {"refine_samples": M, "score_batches": K}。"""
    from scripts.bbox_depth_mask import bbox_depth_mask
    from scripts.foundationpose_eval import read_camera_k, read_depth_m, list_frame_dirs
    from scripts.foundationpose_factory import EstimatorConfig, build_estimator
    import cv2
    import json

    cfg = EstimatorConfig(mesh_file=MESH_FILE, device="cpu", render_backend="moderngl",
                          net_backend="torch", symmetry_mode="z12")
    bundle = build_estimator(cfg)
    est = bundle.estimator
    refine_sink = _RefineReservoir(max_refine_samples, seed)
    score_sink = []
    est.refiner.calib_sink = refine_sink
    est.scorer.calib_sink = score_sink
    frames_used = 0
    try:
        for frame_dir in list_frame_dirs(frames_root):
            bbox_path = frame_dir / "bbox.json"
            if not bbox_path.exists():
                print(f"{frame_dir.name}: bbox.json missing, skipped")
                continue
            color = cv2.imread(str(frame_dir / "color.png"), cv2.IMREAD_COLOR)
            rgb = np.ascontiguousarray(color[..., ::-1])
            depth = read_depth_m(frame_dir / "depth.png")
            K = read_camera_k(frame_dir / "camera_info.yaml")
            bbox = json.loads(bbox_path.read_text(encoding="utf-8"))["bbox_xyxy"]
            mask = bbox_depth_mask(depth, bbox, center_frac=MASK_CENTER_FRAC, half_window_m=DEPTH_WINDOW_HALF_M)
            est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask, iteration=est_refine_iter, bbox_xyxy=bbox)
            frames_used += 1
            print(f"{frame_dir.name}: recorded")
    finally:
        if hasattr(est.glctx, "release"):
            est.glctx.release()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not refine_sink:
        raise RuntimeError("没有记录到 refine 样本，请检查 frames_root 中是否有 bbox.json")
    # 每项是 (1,6,160,160)，拼接成 (M,6,160,160)。
    refine_A = np.concatenate([a for a, _ in refine_sink], axis=0)
    refine_B = np.concatenate([b for _, b in refine_sink], axis=0)
    np.savez(out_dir / "refine_calib.npz", A=refine_A, B=refine_B)

    score_arrays = {}
    for k, (a, b) in enumerate(score_sink):
        score_arrays[f"A_{k}"] = a
        score_arrays[f"B_{k}"] = b
    np.savez(out_dir / "score_calib.npz", **score_arrays)

    counts = {"refine_samples": int(refine_A.shape[0]), "score_batches": len(score_sink)}
    print(f"frames used: {frames_used}, refine_samples: {counts['refine_samples']}, "
          f"score_batches: {counts['score_batches']}")
    return counts


def quantize_ir(fp32_xml, int8_xml, batches, conv_only: bool = False) -> None:
    """batches: list[tuple[np.ndarray, np.ndarray]]，每个元素是一次推理的 (A, B)。"""
    import nncf
    import openvino as ov

    model = ov.Core().read_model(str(fp32_xml))
    dataset = nncf.Dataset(batches, lambda b: {0: b[0], 1: b[1]})
    kwargs = {}
    if conv_only:
        kwargs["ignored_scope"] = nncf.IgnoredScope(types=["MatMul"])
    quantized = nncf.quantize(
        model, dataset,
        preset=nncf.QuantizationPreset.MIXED,
        model_type=nncf.ModelType.TRANSFORMER,
        subset_size=len(batches),
        **kwargs,
    )
    Path(int8_xml).parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(quantized, str(int8_xml), compress_to_fp16=False)


def count_fake_quantize(xml_path) -> int:
    import openvino as ov

    model = ov.Core().read_model(str(xml_path))
    return sum(1 for op in model.get_ops() if op.get_type_name() == "FakeQuantize")


def _load_refine_batches(calib_dir):
    # 每个数组只读一次：NpzFile 每次按键取值都会重新解压整个数组，
    # 切片又会持有整块数组的引用，放在循环里会反复占用几百 MB。
    data = np.load(Path(calib_dir) / "refine_calib.npz")
    A, B = data["A"], data["B"]
    return [(A[i:i + 1], B[i:i + 1]) for i in range(A.shape[0])]


def _load_score_batches(calib_dir):
    data = np.load(Path(calib_dir) / "score_calib.npz")
    k = 0
    batches = []
    while f"A_{k}" in data.files:
        batches.append((data[f"A_{k}"], data[f"B_{k}"]))
        k += 1
    return batches


def cmd_collect(args):
    counts = collect_calibration(args.frames_root, args.out, est_refine_iter=args.est_refine_iter,
                                 max_refine_samples=args.max_refine_samples, seed=args.seed)
    print(counts)


def cmd_quantize(args):
    weights_root = Path(args.weights_root)
    refine_batches = _load_refine_batches(args.calib)
    score_batches = _load_score_batches(args.calib)
    print(f"refine batches: {len(refine_batches)}, score batches: {len(score_batches)}")
    # refine 始终全量量化（卷积 + MatMul）；score 由 --score_conv_only 决定是否只量化卷积。
    quantize_ir(refine_ir_path(weights_root, "fp32"), refine_ir_path(weights_root, "int8"),
                refine_batches, conv_only=False)
    quantize_ir(score_ir_path(weights_root, "fp32"), score_ir_path(weights_root, "int8"),
                score_batches, conv_only=args.score_conv_only)
    print(f"wrote {refine_ir_path(weights_root, 'int8')} and {score_ir_path(weights_root, 'int8')}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("collect", help="用校准帧记录网络输入")
    c.add_argument("--frames_root", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--est_refine_iter", type=int, default=2)
    c.add_argument("--max_refine_samples", type=int, default=300)
    c.add_argument("--seed", type=int, default=0)
    c.set_defaults(func=cmd_collect)

    q = sub.add_parser("quantize", help="量化 FP32 IR 生成 INT8 IR")
    q.add_argument("--calib", required=True)
    q.add_argument("--weights_root", default=str(FOUNDATIONPOSE_DIR / "weights"))
    q.add_argument("--score_conv_only", action="store_true",
                   help="score 网络只量化卷积，MatMul 保持 FP32（refine 始终全量量化）")
    q.set_defaults(func=cmd_quantize)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
