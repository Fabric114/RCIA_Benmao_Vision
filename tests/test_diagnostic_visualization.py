import unittest

import numpy as np

from scripts.diagnostic_visualization import (
    compose_pipeline_view,
    draw_panel_status,
    draw_tracking_status_overlay,
    draw_yolo_detections,
    make_placeholder_panel,
    overlay_mask,
)


class DiagnosticVisualizationTest(unittest.TestCase):
    def test_draw_yolo_detections_marks_bbox_edges(self):
        image = np.zeros((40, 50, 3), dtype=np.uint8)

        rendered = draw_yolo_detections(image, [(10, 8, 30, 28, 0.91, 1)])

        self.assertTrue(np.any(rendered[8, 10:31] != 0))
        self.assertTrue(np.array_equal(image, np.zeros_like(image)))

    def test_overlay_mask_changes_only_masked_region(self):
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:10, 5:10] = True

        rendered = overlay_mask(image, mask, color=(0, 255, 0), alpha=0.5)

        self.assertTrue(np.any(rendered[7, 7] != 0))
        self.assertTrue(np.all(rendered[0, 0] == 0))

    def test_make_placeholder_panel_has_stable_size(self):
        rendered = make_placeholder_panel("Waiting", panel_size=(64, 48))

        self.assertEqual(rendered.shape, (48, 64, 3))
        self.assertEqual(rendered.dtype, np.uint8)

    def test_compose_pipeline_view_builds_two_by_two_grid(self):
        yolo = np.full((20, 30, 3), (0, 255, 0), dtype=np.uint8)
        foundationpose = np.full((10, 10, 3), (255, 0, 0), dtype=np.uint8)

        rendered = compose_pipeline_view(
            yolo=yolo,
            sam=None,
            foundationpose=foundationpose,
            m3t=None,
            panel_size=(80, 60),
        )

        self.assertEqual(rendered.shape, (120, 160, 3))
        self.assertTrue(np.any(rendered[:60, :80] != 0))
        self.assertTrue(np.any(rendered[60:, :80] != 0))

    def test_tracking_status_overlay_draws_status_bar(self):
        image = np.zeros((80, 120, 3), dtype=np.uint8)

        rendered = draw_tracking_status_overlay(
            image,
            {
                "state": "M3T_SUSPECT",
                "trusted": False,
                "depth_consistency_ratio": 0.31,
                "yolo_visible": True,
                "action": "request_reinitialization",
            },
        )

        self.assertTrue(np.any(rendered[-24:, :] != 0))
        self.assertTrue(np.array_equal(image, np.zeros_like(image)))

    def test_compose_pipeline_view_accepts_tracking_status(self):
        rendered = compose_pipeline_view(
            yolo=np.zeros((20, 20, 3), dtype=np.uint8),
            sam=None,
            foundationpose=None,
            m3t=None,
            panel_size=(80, 60),
            tracking_status={
                "state": "M3T_TRACKING_OK",
                "trusted": True,
                "depth_consistency_ratio": 0.82,
                "yolo_visible": True,
                "action": "trust_tracking",
            },
        )

        self.assertEqual(rendered.shape, (120, 160, 3))
        self.assertTrue(np.any(rendered[-24:, :] != 0))

    def test_tracking_status_overlay_accepts_supervisor_nested_health(self):
        image = np.zeros((80, 120, 3), dtype=np.uint8)

        rendered = draw_tracking_status_overlay(
            image,
            {
                "state": "TRACKING_SUSPECT",
                "action": "mark_untrusted",
                "trusted": False,
                "m3t": {
                    "depth_consistency_ratio": 0.27,
                    "yolo_visible": True,
                    "reason": "projection IoU too low",
                },
                "candidate": {"success": False, "reason": "waiting"},
            },
        )

        self.assertTrue(np.any(rendered[-24:, :] != 0))

    def test_draw_panel_status_marks_stale_panel(self):
        image = np.zeros((80, 120, 3), dtype=np.uint8)

        rendered = draw_panel_status(image, "stale 2.3s", stale=True)

        self.assertTrue(np.any(rendered[30:54, :] != 0))
        self.assertTrue(np.array_equal(image, np.zeros_like(image)))

    def test_compose_pipeline_view_accepts_panel_statuses(self):
        rendered = compose_pipeline_view(
            yolo=np.zeros((20, 20, 3), dtype=np.uint8),
            sam=np.zeros((20, 20, 3), dtype=np.uint8),
            foundationpose=None,
            m3t=None,
            panel_size=(80, 60),
            panel_statuses={
                "sam": {"label": "stale 3.1s", "stale": True},
                "foundationpose": {"label": "waiting request", "stale": False},
            },
        )

        self.assertEqual(rendered.shape, (120, 160, 3))
        self.assertTrue(np.any(rendered[:60, 80:] != 0))


if __name__ == "__main__":
    unittest.main()
