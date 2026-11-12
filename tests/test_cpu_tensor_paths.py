import sys
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "foundation_pose"))

from Utils import compute_crop_window_tf_batch, compute_mesh_diameter, depth2xyzmap_batch  # noqa: E402

MESH = str(PROJECT_ROOT / "models/object/Str3D.obj")
WEIGHTS_OK = all(
    (PROJECT_ROOT / "foundation_pose/weights" / run / "model_best.pth").exists()
    for run in ("2023-10-28-18-33-37", "2024-01-11-20-02-45")
)


class CpuTensorPathsTest(unittest.TestCase):
    def test_depth2xyzmap_batch_on_cpu(self):
        out = depth2xyzmap_batch(torch.full((1, 4, 5), 0.5), torch.eye(3)[None], zfar=np.inf)
        self.assertEqual(out.device.type, "cpu")

    def test_compute_crop_window_tf_batch_on_cpu(self):
        import trimesh

        mesh = trimesh.load(MESH, process=False)
        diameter = compute_mesh_diameter(mesh=mesh)
        poses = torch.eye(4)[None].repeat(2, 1, 1)
        poses[:, :3, 3] = torch.tensor([[0.0, 0.0, 0.5], [0.05, 0.02, 0.6]])
        K = np.array([[615.0, 0, 320.0], [0, 615.0, 240.0], [0, 0, 1.0]])
        tf = compute_crop_window_tf_batch(pts=mesh.vertices, H=480, W=640, poses=poses, K=K,
                                          crop_ratio=1.2, out_size=(160, 160), method="box_3d",
                                          mesh_diameter=diameter)
        self.assertEqual(tf.device.type, "cpu")
        self.assertEqual(tuple(tf.shape), (2, 3, 3))

    @unittest.skipUnless(WEIGHTS_OK, "needs weights")
    def test_refine_net_forward_on_cpu(self):
        from learning.training.predict_pose_refine import PoseRefinePredictor

        p = PoseRefinePredictor(device="cpu")
        out = p.model(torch.rand(2, 6, 160, 160), torch.rand(2, 6, 160, 160))
        self.assertEqual(tuple(out["trans"].shape), (2, 3))
        self.assertEqual(out["trans"].device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
