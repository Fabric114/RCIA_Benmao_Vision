#!/usr/bin/env python3
"""FoundationPose 真实帧评测：prepare（YOLO 取框 + mask）、run（register 并记录位姿与耗时）、compare（与基线对比）。"""
import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.bbox_depth_mask import bbox_depth_mask  # noqa: E402
from scripts.pose_eval_metrics import pose_errors  # noqa: E402


def list_frame_dirs(frames_root):
    return sorted(p for p in Path(frames_root).iterdir() if (p / "color.png").exists() and (p / "depth.png").exists())


def read_camera_k(path):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("k:"):
            values = [float(v) for v in line.split(":", 1)[1].strip().strip("[]").split(",")]
            return np.array(values, dtype=np.float64).reshape(3, 3)
    raise ValueError(f"no 'k:' line in {path}")


def read_depth_m(path):
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    depth[(depth < 0.001) | (~np.isfinite(depth))] = 0
    return depth


def model_center_from_mesh(mesh_file):
    import trimesh

    vertices = trimesh.load(mesh_file, process=False).vertices
    return (vertices.min(axis=0) + vertices.max(axis=0)) / 2


def cmd_prepare(args):
    import torch

    from scripts.foundationpose_init import MaskGenerator

    def now():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    mask_gen = MaskGenerator(yolo_path=args.yolo_model)
    for frame_dir in list_frame_dirs(args.frames_root):
        color = cv2.imread(str(frame_dir / "color.png"), cv2.IMREAD_COLOR)
        t0 = now()
        dets = mask_gen.detect_only(color, target_class=args.target_class, conf=args.conf)
        t1 = now()
        if not dets:
            logging.warning(f"{frame_dir.name}: no detection, skipped")
            continue
        x1, y1, x2, y2, conf, _ = max(dets, key=lambda d: (d[2] - d[0]) * (d[3] - d[1]))
        (frame_dir / "bbox.json").write_text(
            json.dumps(
                {
                    "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                    "conf": float(conf),
                    "yolo_ms": (t1 - t0) * 1000.0,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"{frame_dir.name}: bbox={[x1, y1, x2, y2]} conf={conf:.3f}")


def cmd_run(args):
    from scripts.foundationpose_factory import EstimatorConfig, build_estimator

    cfg = EstimatorConfig(**{f.name: getattr(args, f.name) for f in fields(EstimatorConfig)})
    bundle = build_estimator(cfg)
    est = bundle.estimator
    results = {
        "config": {**asdict(cfg), "mask_source": args.mask_source, "est_refine_iter": args.est_refine_iter},
        "frames": {},
    }
    for frame_dir in list_frame_dirs(args.frames_root):
        color = cv2.imread(str(frame_dir / "color.png"), cv2.IMREAD_COLOR)
        rgb = np.ascontiguousarray(color[..., ::-1])
        depth = read_depth_m(frame_dir / "depth.png")
        K = read_camera_k(frame_dir / "camera_info.yaml")
        bbox_path = frame_dir / "bbox.json"
        if args.mask_source == "bbox_depth":
            if not bbox_path.exists():
                logging.warning(f"{frame_dir.name}: bbox.json missing, skipped")
                continue
            bbox = json.loads(bbox_path.read_text(encoding="utf-8"))["bbox_xyxy"]
            t0 = time.perf_counter()
            mask = bbox_depth_mask(depth, bbox, center_frac=args.mask_center_frac, half_window_m=args.depth_window_half_m)
            mask_ms = (time.perf_counter() - t0) * 1000.0
        elif args.mask_source == "sam_file":
            mask_path = frame_dir / "mask_sam.png"
            if not mask_path.exists():
                logging.warning(f"{frame_dir.name}: {mask_path.name} missing, skipped")
                continue
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0
            bbox = None
            mask_ms = 0.0
        else:
            raise ValueError(f"unknown mask_source: {args.mask_source}")
        pose = est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask, iteration=args.est_refine_iter, bbox_xyxy=bbox)
        timing = dict(est.last_timing_ms)
        timing["mask"] = mask_ms
        timing.update({f"refine_{k}": v for k, v in est.refiner.last_timing_ms.items()})
        timing.update({f"score_{k}": v for k, v in est.scorer.last_timing_ms.items()})
        results["frames"][frame_dir.name] = {"pose": np.asarray(pose).tolist(), "timing_ms": timing}
        print(f"{frame_dir.name}: total={timing['total']:.1f} ms")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out} ({len(results['frames'])} frames)")


def cmd_compare(args):
    ref = json.loads(Path(args.ref).read_text(encoding="utf-8"))["frames"]
    test = json.loads(Path(args.test).read_text(encoding="utf-8"))["frames"]
    center = model_center_from_mesh(args.mesh_file)
    names = sorted(set(ref) & set(test))
    missing = sorted(set(ref) ^ set(test))
    if missing:
        print(f"frames only in one side (ignored): {missing}")
    if not names:
        print("no common frames")
        return 1

    centers_mm, axes_deg, failed = [], [], []
    print(f"{'frame':<24}{'center_mm':>10}{'axis_deg':>10}{'ref_ms':>10}{'test_ms':>10}")
    for name in names:
        c, a = pose_errors(np.array(ref[name]["pose"]), np.array(test[name]["pose"]), center)
        c_mm = c * 1000.0
        centers_mm.append(c_mm)
        axes_deg.append(a)
        flag = ""
        if c_mm > args.max_center_mm or a > args.max_axis_deg:
            failed.append(name)
            flag = "  FAIL"
        ref_ms = ref[name].get("timing_ms", {}).get("total", float("nan"))
        test_ms = test[name].get("timing_ms", {}).get("total", float("nan"))
        print(f"{name:<24}{c_mm:>10.2f}{a:>10.2f}{ref_ms:>10.1f}{test_ms:>10.1f}{flag}")
    print(f"center_mm median={np.median(centers_mm):.2f} max={np.max(centers_mm):.2f}")
    print(f"axis_deg  median={np.median(axes_deg):.2f} max={np.max(axes_deg):.2f}")
    if failed:
        print(f"{len(failed)} frame(s) over limit: {failed}")
        return 1
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--frames_root", required=True)
    p.add_argument("--yolo_model", default=str(PROJECT_ROOT / "models/yolo/best_openvino_model"))
    p.add_argument("--target_class", type=int, default=0)
    p.add_argument("--conf", type=float, default=0.75)
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("run")
    p.add_argument("--frames_root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mask_source", choices=["bbox_depth", "sam_file"], default="bbox_depth")
    p.add_argument("--mask_center_frac", type=float, default=0.35)
    p.add_argument("--depth_window_half_m", type=float, default=0.11)
    p.add_argument("--est_refine_iter", type=int, default=5)
    p.add_argument("--symmetry_mode", choices=["none", "z12"], default="none")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--net_backend", choices=["torch", "openvino"], default="torch")
    p.add_argument("--ov_device", choices=["CPU", "GPU"], default="CPU")
    p.add_argument("--mesh_file", default=str(PROJECT_ROOT / "models/object/Str3D.obj"))
    p.add_argument("--debug", type=int, default=0)
    p.add_argument("--debug_dir", default=str(PROJECT_ROOT / "runtime/foundationpose_debug"))
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("compare")
    p.add_argument("--ref", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--mesh_file", default=str(PROJECT_ROOT / "models/object/Str3D.obj"))
    p.add_argument("--max_center_mm", type=float, default=10.0)
    p.add_argument("--max_axis_deg", type=float, default=5.0)
    p.set_defaults(func=cmd_compare)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
