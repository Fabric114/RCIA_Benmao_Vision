import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.linemod_template_store import (
    TemplatePoseMetadata,
    find_template_metadata,
    load_template_metadata,
    save_template_metadata,
)


class LinemodTemplateStoreTest(unittest.TestCase):
    def test_save_and_load_template_metadata_round_trips_pose(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "poses.json"
            pose = np.eye(4)
            pose[:3, 3] = [0.1, -0.2, 0.5]
            metadata = [
                TemplatePoseMetadata(
                    class_id="Str3D",
                    template_id=7,
                    object_in_camera=pose,
                    match_center=[320.0, 240.0],
                    source="unit-test",
                )
            ]

            save_template_metadata(path, metadata)
            loaded = load_template_metadata(path)

            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].class_id, "Str3D")
            self.assertEqual(loaded[0].template_id, 7)
            self.assertEqual(loaded[0].match_center, [320.0, 240.0])
            np.testing.assert_allclose(loaded[0].object_in_camera, pose)

    def test_template_metadata_rejects_non_homogeneous_matrix(self):
        bad_pose = np.eye(3)

        with self.assertRaises(ValueError):
            TemplatePoseMetadata(
                class_id="Str3D",
                template_id=1,
                object_in_camera=bad_pose,
                match_center=[0.0, 0.0],
            )

    def test_find_template_metadata_returns_matching_class_and_template_id(self):
        metadata = [
            TemplatePoseMetadata("A", 1, np.eye(4), [10.0, 10.0]),
            TemplatePoseMetadata("B", 2, np.eye(4), [20.0, 20.0]),
        ]

        found = find_template_metadata(metadata, class_id="B", template_id=2)

        self.assertEqual(found.class_id, "B")
        self.assertEqual(found.template_id, 2)

    def test_find_template_metadata_raises_for_missing_id(self):
        metadata = [TemplatePoseMetadata("A", 1, np.eye(4), [10.0, 10.0])]

        with self.assertRaises(KeyError):
            find_template_metadata(metadata, class_id="A", template_id=99)


if __name__ == "__main__":
    unittest.main()
