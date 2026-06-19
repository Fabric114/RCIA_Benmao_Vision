import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.linemod_silhouette_renderer import load_obj_mesh, render_silhouette


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


if __name__ == "__main__":
    unittest.main()
