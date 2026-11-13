import sys
import unittest
from pathlib import Path

import moderngl
import numpy as np
import torch
import trimesh

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "foundation_pose"))

import Utils  # noqa: E402
from moderngl_renderer import ModernglBatchRenderer, is_software_renderer  # noqa: E402

MESH = str(PROJECT_ROOT / "models/object/Str3D.obj")
K = np.array([[615.0, 0, 320], [0, 615.0, 240], [0, 0, 1]])


def _egl_available():
    try:
        moderngl.create_standalone_context(backend="egl").release()
        return True
    except Exception:
        return False


EGL_OK = _egl_available()


def cylinder_pose(z=0.5):
    # 圆柱长轴沿相机 -Y，中心在 (0, 0, z)
    R = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = [0, 0, z]
    return pose


def cylinder_mesh_tensors(device="cpu"):
    mesh = trimesh.creation.cylinder(radius=0.0475, height=0.15, sections=64)
    return Utils.make_mesh_tensors(mesh, device=device)


def random_poses(n, seed):
    rng = np.random.default_rng(seed)
    poses = []
    for _ in range(n):
        R = trimesh.transformations.random_rotation_matrix(rng.random(3))[:3, :3]
        t = [rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1), rng.uniform(0.3, 0.8)]
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = t
        poses.append(pose)
    return np.stack(poses)


class ModernglRendererTest(unittest.TestCase):
    def test_is_software_renderer(self):
        self.assertTrue(is_software_renderer("llvmpipe (LLVM 15.0.7, 256 bits)"))
        self.assertFalse(is_software_renderer("Mesa Intel(R) Graphics (ADL GT2)"))


@unittest.skipUnless(EGL_OK, "needs EGL")
class RendererTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.renderer = ModernglBatchRenderer()
        cls.mt = cylinder_mesh_tensors()

    def test_upright_cylinder_depth(self):
        pose = torch.tensor(cylinder_pose(), dtype=torch.float32)[None]
        extra = {}
        rgb, depth, _ = self.renderer.render(K, 480, 640, pose, self.mt, output_size=(480, 640), extra=extra)
        self.assertAlmostEqual(depth[0, 240, 320].item(), 0.5 - 0.0475, delta=0.002)
        self.assertEqual(depth[0, 0, 0].item(), 0.0)
        self.assertTrue((rgb[0][depth[0] > 0] > 0).all())
        torch.testing.assert_close(extra["xyz_map"][0, ..., 2], depth[0])

    def test_single_pose_with_bbox(self):
        pose = torch.tensor(cylinder_pose(), dtype=torch.float32)[None]
        bbox = torch.tensor([[240.0, 130.0, 400.0, 350.0]])
        rgb, depth, _ = self.renderer.render(K, 480, 640, pose, self.mt, output_size=(160, 160), bbox2d=bbox)
        self.assertEqual(tuple(depth.shape), (1, 160, 160))
        self.assertGreater((depth > 0).sum().item(), 500)

    def test_multi_pass_matches_single_pass(self):
        poses = torch.tensor(random_poses(20, seed=3), dtype=torch.float32)
        self.renderer.max_tiles_per_pass = 8
        a = self.renderer.render(K, 480, 640, poses, self.mt, output_size=(120, 160), use_light=True, extra={})
        self.renderer.max_tiles_per_pass = 1024
        b = self.renderer.render(K, 480, 640, poses, self.mt, output_size=(120, 160), use_light=True, extra={})
        self.assertTrue(torch.equal(a[1] > 0, b[1] > 0))
        torch.testing.assert_close(a[0], b[0], atol=2e-3, rtol=0)
        torch.testing.assert_close(a[1], b[1], atol=1e-3, rtol=0)

    def test_uses_mesh_vertex_color(self):
        mt = cylinder_mesh_tensors()
        mt["vertex_color"][:] = torch.tensor([1.0, 0.0, 0.0])
        pose = torch.tensor(cylinder_pose(), dtype=torch.float32)[None]
        rgb, depth, _ = self.renderer.render(K, 480, 640, pose, mt, output_size=(480, 640), use_light=False)
        fg = rgb[0][depth[0] > 0]
        self.assertTrue((fg[:, 0] > 0.5).all())
        self.assertTrue((fg[:, 1:] < 1e-6).all())

    def test_textured_mesh_not_supported(self):
        pose = torch.tensor(cylinder_pose(), dtype=torch.float32)[None]
        with self.assertRaises(NotImplementedError):
            self.renderer.render(K, 480, 640, pose, {**self.mt, "tex": torch.zeros(1, 4, 4, 3)},
                                 output_size=(480, 640))

    def test_framebuffer_resize_releases_old_attachments(self):
        renderer = ModernglBatchRenderer()
        renderer._framebuffer(64, 64)
        old = list(renderer._fbo_attachments)
        renderer._framebuffer(32, 32)
        self.assertEqual(len(old), 3)
        for obj in old:
            self.assertEqual(type(obj.mglo).__name__, "InvalidObject")
        self.assertEqual(len(renderer._fbo_attachments), 3)
        self.assertNotEqual(type(renderer._fbo_attachments[0].mglo).__name__, "InvalidObject")

    def test_release_is_idempotent_and_blocks_render(self):
        renderer = ModernglBatchRenderer()
        pose = torch.tensor(cylinder_pose(), dtype=torch.float32)[None]
        renderer.render(K, 480, 640, pose, self.mt, output_size=(480, 640))
        renderer.release()
        renderer.release()
        with self.assertRaises(RuntimeError):
            renderer.render(K, 480, 640, pose, self.mt, output_size=(480, 640))

    def test_releasing_one_renderer_keeps_another_working(self):
        pose = torch.tensor(cylinder_pose(), dtype=torch.float32)[None]
        a = ModernglBatchRenderer()
        b = ModernglBatchRenderer()
        self.addCleanup(a.release)
        _, before, _ = a.render(K, 480, 640, pose, self.mt, output_size=(480, 640))
        b.release()
        _, after, _ = a.render(K, 480, 640, pose, self.mt, output_size=(480, 640))
        self.assertTrue(torch.equal(before > 0, after > 0))
        self.assertGreater((after > 0).sum().item(), 1000)

    def test_context_manager_releases(self):
        with ModernglBatchRenderer() as renderer:
            pose = torch.tensor(cylinder_pose(), dtype=torch.float32)[None]
            renderer.render(K, 480, 640, pose, self.mt, output_size=(480, 640))
        self.assertTrue(renderer._released)

    def test_str3d_bbox_multi_pass_statistics(self):
        # 用与真实管线相同的逐位姿裁剪 bbox；边缘像素的三角形覆盖翻转会让极少数像素差异较大，所以用占比而不是最大值
        mesh = trimesh.load(MESH, process=False)
        mt = Utils.make_mesh_tensors(mesh, device="cpu")
        poses = torch.tensor(random_poses(20, seed=5), dtype=torch.float32)
        tf = Utils.compute_crop_window_tf_batch(pts=mesh.vertices, H=480, W=640, poses=poses, K=K, crop_ratio=1.2,
                                                out_size=(160, 160), method="box_3d",
                                                mesh_diameter=Utils.compute_mesh_diameter(mesh=mesh))
        crop = torch.tensor([[0, 0, 159, 159]], dtype=torch.float32).reshape(2, 2)
        bbox = Utils.transform_pts(crop, tf.inverse()).reshape(-1, 4)
        renderer = ModernglBatchRenderer()
        self.addCleanup(renderer.release)
        renderer.max_tiles_per_pass = 8
        a = renderer.render(K, 480, 640, poses, mt, output_size=(160, 160), bbox2d=bbox, use_light=True, extra={})
        renderer.max_tiles_per_pass = 1024
        b = renderer.render(K, 480, 640, poses, mt, output_size=(160, 160), bbox2d=bbox, use_light=True, extra={})
        fg_a, fg_b = a[1] > 0, b[1] > 0
        self.assertGreater(fg_a.sum().item(), 10000)
        self.assertTrue(torch.equal(fg_a, fg_b))
        both = fg_a & fg_b
        rgb_diff = (a[0] - b[0]).abs().amax(dim=-1)[both]
        self.assertLess((rgb_diff > 2e-3).float().mean().item(), 1e-4)
        depth_diff = (a[1] - b[1]).abs()[both]
        self.assertLess((depth_diff > 1e-3).float().mean().item(), 1e-4)
        self.assertLess(depth_diff.max().item(), 1e-2)

    def test_failed_construction_returns_context_reference(self):
        import moderngl_renderer as mr

        keep = ModernglBatchRenderer()
        self.addCleanup(keep.release)
        key = (keep._backend, keep._owner_thread)
        before = mr._CONTEXTS[key][1]
        original = mr.FRAGMENT_SHADER
        mr.FRAGMENT_SHADER = "not valid glsl"
        try:
            with self.assertRaises(Exception):
                ModernglBatchRenderer()
        finally:
            mr.FRAGMENT_SHADER = original
        self.assertEqual(mr._CONTEXTS[key][1], before)

    def test_render_from_other_thread_raises(self):
        import threading

        pose = torch.tensor(cylinder_pose(), dtype=torch.float32)[None]
        errors = []

        def run():
            try:
                self.renderer.render(K, 480, 640, pose, self.mt, output_size=(480, 640))
            except Exception as e:
                errors.append(e)

        t = threading.Thread(target=run)
        t.start()
        t.join()
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], RuntimeError)


@unittest.skipUnless(EGL_OK and torch.cuda.is_available(), "needs EGL and CUDA")
class NvdiffrastParityTest(unittest.TestCase):
    def test_matches_nvdiffrast(self):
        import trimesh as tm

        mesh = tm.load(MESH, process=False)
        poses = random_poses(20, seed=0)
        poses_t = torch.tensor(poses, dtype=torch.float32)
        diameter = Utils.compute_mesh_diameter(mesh=mesh)
        tf_crop = Utils.compute_crop_window_tf_batch(pts=mesh.vertices, H=480, W=640, poses=poses_t, K=K,
                                                     crop_ratio=1.2, out_size=(160, 160), method="box_3d",
                                                     mesh_diameter=diameter)
        bbox_crop = torch.tensor([[0, 0, 159, 159]], dtype=torch.float32).reshape(2, 2)
        bbox_ori = Utils.transform_pts(bbox_crop, tf_crop.inverse()).reshape(-1, 4).cuda()

        mt = Utils.make_mesh_tensors(mesh, device="cuda")
        glctx = Utils.dr.RasterizeCudaContext()
        extra_nv = {}
        rgb_nv, d_nv, _ = Utils.nvdiffrast_render(K=K, H=480, W=640, ob_in_cams=poses_t.cuda(), glctx=glctx,
                                                  mesh_tensors=mt, output_size=(160, 160), bbox2d=bbox_ori,
                                                  use_light=True, extra=extra_nv)
        renderer = ModernglBatchRenderer()
        extra_mgl = {}
        rgb_mgl, d_mgl, _ = renderer.render(K, 480, 640, poses_t.cuda(), mt, output_size=(160, 160),
                                            bbox2d=bbox_ori, use_light=True, extra=extra_mgl)

        both_fg = (d_mgl > 0) & (d_nv > 0)
        either_fg = (d_mgl > 0) | (d_nv > 0)
        iou = (both_fg.sum() / either_fg.sum()).item()
        self.assertGreaterEqual(iou, 0.98)
        dz = (d_mgl - d_nv).abs()[both_fg]
        self.assertLess(dz.median().item(), 0.001)
        self.assertLess(dz.quantile(0.95).item(), 0.003)
        self.assertLess((rgb_mgl - rgb_nv).abs()[both_fg].mean().item(), 0.02)
        xyz_mgl = extra_mgl["xyz_map"]
        xyz_nv = extra_nv["xyz_map"]
        self.assertLess((xyz_mgl - xyz_nv).norm(dim=-1)[both_fg].median().item(), 0.001)


if __name__ == "__main__":
    unittest.main()
