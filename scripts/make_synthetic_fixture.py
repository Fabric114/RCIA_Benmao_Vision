#!/usr/bin/env python3
"""用 nvdiffrast 渲染 Str3D 的合成 RGB-D 场景，作为 FoundationPose register 的回归 fixture。"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "foundation_pose"))

from Utils import dr, make_mesh_tensors, nvdiffrast_render  # noqa: E402

H, W = 480, 640
K = np.array([[615.0, 0.0, 320.0], [0.0, 615.0, 240.0], [0.0, 0.0, 1.0]])
TILT_DEG = 30.0
CENTER_CAM = np.array([0.02, -0.01, 0.55])


def centered_pose():
    # 物体 z（长轴）→ 相机 -Y，再绕相机 X 轴向镜头方向倾斜
    R0 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    a = np.radians(TILT_DEG)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(a), -np.sin(a)], [0.0, np.sin(a), np.cos(a)]])
    pose = np.eye(4)
    pose[:3, :3] = Rx @ R0
    pose[:3, 3] = CENTER_CAM
    return pose


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(PROJECT_ROOT / "tests/data/synthetic_str3d"))
    parser.add_argument("--mesh_file", default=str(PROJECT_ROOT / "models/object/Str3D.obj"))
    args = parser.parse_args()

    mesh = trimesh.load(args.mesh_file, process=False)
    model_center = (mesh.vertices.min(axis=0) + mesh.vertices.max(axis=0)) / 2
    mesh_centered = mesh.copy()
    mesh_centered.vertices = mesh_centered.vertices - model_center.reshape(1, 3)

    pose_centered = centered_pose()
    color, depth, _ = nvdiffrast_render(
        K=K, H=H, W=W,
        ob_in_cams=torch.as_tensor(pose_centered[None], device="cuda", dtype=torch.float),
        glctx=dr.RasterizeCudaContext(),
        mesh_tensors=make_mesh_tensors(mesh_centered),
        use_light=True,
    )
    color = (color[0].clamp(0, 1) * 255).round().byte().cpu().numpy()
    depth = depth[0].cpu().numpy()

    tf_to_center = np.eye(4)
    tf_to_center[:3, 3] = -model_center
    pose = pose_centered @ tf_to_center

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "color.png"), cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / "depth.png"), np.round(depth * 1000.0).clip(0, 65535).astype(np.uint16))
    np.savetxt(out / "K.txt", K)
    np.savetxt(out / "pose.txt", pose)
    print(f"wrote {out}, depth nonzero pixels: {int((depth > 0.001).sum())}")


if __name__ == "__main__":
    main()
