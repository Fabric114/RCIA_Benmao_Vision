import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.linemod_silhouette_renderer import (
    load_obj_mesh,
    render_mask_and_depth,
    render_model_buffers,
    render_silhouette,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
STR3D_PATH = REPO_ROOT / "models" / "object" / "Str3D.obj"
K_160_120 = np.array(
    [
        [600.0, 0.0, 80.0],
        [0.0, 600.0, 60.0],
        [0.0, 0.0, 1.0],
    ]
)
K_640_480 = np.array(
    [
        [600.0, 0.0, 320.0],
        [0.0, 600.0, 240.0],
        [0.0, 0.0, 1.0],
    ]
)


class LinemodSilhouetteRendererTest(unittest.TestCase):
    def test_load_obj_mesh_reads_vertices_and_triangle_faces(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "triangle.obj"
            path.write_text(
                "\n".join(
                    [
                        "v 0 0 0",
                        "v 1 0 0",
                        "v 0 1 0",
                        "f 1 2 3",
                    ]
                ),
                encoding="utf-8",
            )

            vertices, faces = load_obj_mesh(path)

        self.assertEqual(vertices.shape, (3, 3))
        self.assertEqual(faces.shape, (1, 3))
        np.testing.assert_array_equal(faces[0], [0, 1, 2])

    def test_render_silhouette_projects_mesh_into_image(self):
        vertices = np.array(
            [
                [-0.05, -0.05, 0.0],
                [0.05, -0.05, 0.0],
                [0.05, 0.05, 0.0],
                [-0.05, 0.05, 0.0],
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        object_in_camera = np.eye(4)
        object_in_camera[:3, 3] = [0.0, 0.0, 0.5]
        k_matrix = np.array(
            [
                [600.0, 0.0, 80.0],
                [0.0, 600.0, 60.0],
                [0.0, 0.0, 1.0],
            ]
        )

        color, mask = render_silhouette(
            vertices,
            faces,
            object_in_camera,
            k_matrix,
            image_size=(160, 120),
        )

        self.assertEqual(color.shape, (120, 160, 3))
        self.assertEqual(mask.shape, (120, 160))
        self.assertGreater(int(mask.sum()), 0)
        self.assertTrue(np.any(color[mask > 0] != 0))

    def test_render_mask_and_depth_matches_render_model_buffers_on_quad(self):
        vertices = np.array(
            [
                [-0.05, -0.05, 0.0],
                [0.05, -0.05, 0.0],
                [0.05, 0.05, 0.0],
                [-0.05, 0.05, 0.0],
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        pose = np.eye(4)
        pose[:3, 3] = [0.0, 0.0, 0.5]

        _, expected_mask, _, expected_depth, _ = render_model_buffers(
            vertices, faces, pose, K_160_120, image_size=(160, 120)
        )
        mask, depth = render_mask_and_depth(
            vertices, faces, pose, K_160_120, image_size=(160, 120)
        )

        self.assertGreater(int(np.count_nonzero(mask)), 0)
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(mask.shape, (120, 160))
        self.assertEqual(depth.dtype, np.float64)
        self.assertEqual(depth.shape, (120, 160))
        np.testing.assert_array_equal(mask, expected_mask)
        np.testing.assert_array_equal(depth, expected_depth)
        self.assertTrue(np.all(depth[mask == 0] == 0.0))
        self.assertTrue(np.all(depth[mask > 0] > 0.0))
        np.testing.assert_allclose(depth[mask > 0], 0.5)

    def test_render_mask_and_depth_keeps_nearest_surface_when_far_face_comes_first(self):
        vertices = np.array(
            [
                [-0.05, -0.05, 0.6],
                [0.05, -0.05, 0.6],
                [0.0, 0.05, 0.6],
                [-0.05, -0.05, 0.4],
                [0.05, -0.05, 0.4],
                [0.0, 0.05, 0.4],
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        pose = np.eye(4)

        mask, depth = render_mask_and_depth(
            vertices, faces, pose, K_160_120, image_size=(160, 120)
        )
        _, expected_mask, _, expected_depth, _ = render_model_buffers(
            vertices, faces, pose, K_160_120, image_size=(160, 120)
        )

        center = (60, 80)
        self.assertAlmostEqual(float(depth[center]), 0.4)
        np.testing.assert_array_equal(mask, expected_mask)
        np.testing.assert_array_equal(depth, expected_depth)

    def test_render_mask_and_depth_skips_triangles_behind_or_outside_view(self):
        vertices = np.array(
            [
                [-0.05, -0.05, -0.2],
                [0.05, -0.05, -0.2],
                [0.0, 0.05, -0.2],
                [5.0, 5.0, 0.5],
                [5.1, 5.0, 0.5],
                [5.0, 5.1, 0.5],
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)

        mask, depth = render_mask_and_depth(
            vertices, faces, np.eye(4), K_160_120, image_size=(160, 120)
        )

        self.assertEqual(int(np.count_nonzero(mask)), 0)
        self.assertTrue(np.all(depth == 0.0))

    def test_render_mask_and_depth_all_triangles_behind_camera(self):
        vertices = np.array(
            [
                [-0.05, -0.05, -0.2],
                [0.05, -0.05, -0.2],
                [0.0, 0.05, -0.2],
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2]], dtype=np.int32)

        mask, depth = render_mask_and_depth(
            vertices, faces, np.eye(4), K_160_120, image_size=(160, 120)
        )

        self.assertEqual(mask.shape, (120, 160))
        self.assertEqual(int(np.count_nonzero(mask)), 0)
        self.assertTrue(np.all(depth == 0.0))

    def test_render_mask_and_depth_matches_render_model_buffers_on_str3d(self):
        if not STR3D_PATH.exists():
            self.skipTest("models/object/Str3D.obj is not available")
        vertices, faces = load_obj_mesh(STR3D_PATH)
        pose = np.eye(4)
        pose[:3, 3] = [0.0, 0.0, 0.5]

        _, expected_mask, _, expected_depth, _ = render_model_buffers(
            vertices, faces, pose, K_640_480, image_size=(640, 480)
        )
        mask, depth = render_mask_and_depth(
            vertices, faces, pose, K_640_480, image_size=(640, 480)
        )

        self.assertGreater(int(np.count_nonzero(mask)), 1000)
        np.testing.assert_array_equal(mask, expected_mask)
        np.testing.assert_allclose(depth, expected_depth, rtol=0.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
