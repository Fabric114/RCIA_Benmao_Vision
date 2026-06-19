import argparse
import unittest

import numpy as np

from scripts.linemod_template_builder import (
    apply_camera_inplane_rotation,
    iter_template_pose_specs,
)


class LinemodTemplateBuilderTest(unittest.TestCase):
    def test_iter_template_pose_specs_expands_distances_elevations_and_inplane_rolls(self):
        args = argparse.Namespace(
            distance=0.5,
            distances="0.4,0.6",
            azimuth_count=3,
            elevations="0,30",
            inplane_degrees="0,90",
        )

        specs = list(iter_template_pose_specs(args))

        self.assertEqual(len(specs), 24)
        self.assertEqual({spec.distance for spec in specs}, {0.4, 0.6})
        self.assertEqual({spec.inplane_degree for spec in specs}, {0.0, 90.0})

    def test_apply_camera_inplane_rotation_preserves_translation_and_rolls_rotation(self):
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = [0.1, -0.2, 0.5]

        rolled = apply_camera_inplane_rotation(pose, 90.0)

        np.testing.assert_allclose(rolled[:3, 3], pose[:3, 3], atol=1e-9)
        np.testing.assert_allclose(
            rolled[:3, :3],
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            atol=1e-9,
        )


if __name__ == "__main__":
    unittest.main()
