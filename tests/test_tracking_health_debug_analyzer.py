import unittest

from scripts.tracking_health_debug_analyzer import analyze_tracking_health


class TrackingHealthDebugAnalyzerTest(unittest.TestCase):
    def test_reinit_needed_reports_external_conflict_as_primary_cause(self):
        payload = {
            "action": "stop_m3t_for_reinit",
            "tracking_filter": {
                "reliability_state": "REINIT_NEEDED",
                "request_reinit": True,
                "stop_m3t": False,
                "reason": "sustained external conflict",
                "m3t_good_streak": 0,
                "m3t_bad_streak": 1,
                "external_conflict_streak": 5,
                "out_of_view_streak": 0,
                "weak_validation_fail_streak": 0,
            },
            "m3t": {
                "depth_consistency_ratio": 0.01,
                "projection_iou": 0.0,
                "black_white_ok": False,
                "smoothness_ok": False,
                "rendered_pixel_count": 200,
                "reason": "projection IoU below threshold; depth consistency failed",
            },
        }

        result = analyze_tracking_health(payload)

        self.assertTrue(result.should_emit)
        self.assertEqual(result.primary_cause, "external_conflict")
        self.assertIn("REINIT_NEEDED", result.summary)
        self.assertIn("projection", " ".join(result.evidence).lower())

    def test_lost_reports_out_of_view_as_primary_cause(self):
        payload = {
            "action": "stop_m3t",
            "tracking_filter": {
                "reliability_state": "LOST",
                "request_reinit": False,
                "stop_m3t": True,
                "reason": "target out of view",
                "m3t_good_streak": 0,
                "m3t_bad_streak": 8,
                "external_conflict_streak": 0,
                "out_of_view_streak": 12,
                "weak_validation_fail_streak": 0,
            },
            "m3t": {
                "depth_consistency_ratio": 0.0,
                "projection_iou": 0.0,
                "black_white_ok": False,
                "smoothness_ok": False,
                "rendered_pixel_count": 0,
                "reason": "missing rendered support",
            },
        }

        result = analyze_tracking_health(payload)

        self.assertTrue(result.should_emit)
        self.assertEqual(result.primary_cause, "out_of_view")
        self.assertIn("stop", result.summary.lower())

    def test_startup_grace_expired_is_reported_as_startup_timeout(self):
        payload = {
            "action": "stop_m3t",
            "tracking_filter": {
                "reliability_state": "LOST",
                "request_reinit": False,
                "stop_m3t": True,
                "reason": "startup grace expired",
                "m3t_good_streak": 1,
                "m3t_bad_streak": 0,
                "external_conflict_streak": 0,
                "out_of_view_streak": 0,
                "weak_validation_fail_streak": 0,
            },
            "m3t": {
                "depth_consistency_ratio": 0.984,
                "projection_iou": 0.792,
                "black_white_ok": False,
                "smoothness_ok": True,
                "rendered_pixel_count": 8019,
                "reason": "black/white appearance failed",
            },
        }

        result = analyze_tracking_health(payload)

        self.assertTrue(result.should_emit)
        self.assertEqual(result.primary_cause, "startup_timeout")
        self.assertIn("startup grace expired", result.summary)

    def test_owned_without_action_is_quiet_by_default(self):
        payload = {
            "action": "publish_trusted_pose",
            "tracking_filter": {
                "reliability_state": "M3T_OWNED",
                "request_reinit": False,
                "stop_m3t": False,
                "reason": "m3t-owned",
                "m3t_good_streak": 20,
                "m3t_bad_streak": 0,
                "external_conflict_streak": 0,
                "out_of_view_streak": 0,
                "weak_validation_fail_streak": 0,
            },
            "m3t": {
                "depth_consistency_ratio": 0.75,
                "projection_iou": 0.5,
                "black_white_ok": True,
                "smoothness_ok": True,
                "rendered_pixel_count": 5000,
                "reason": "",
            },
        }

        result = analyze_tracking_health(payload)

        self.assertFalse(result.should_emit)
        self.assertEqual(result.primary_cause, "healthy")

    def test_state_change_is_reported_even_without_stop_or_reinit(self):
        previous = {
            "tracking_filter": {
                "reliability_state": "M3T_OWNED",
                "m3t_good_streak": 10,
                "external_conflict_streak": 0,
                "out_of_view_streak": 0,
                "weak_validation_fail_streak": 0,
            }
        }
        current = {
            "action": "watch_tracking",
            "tracking_filter": {
                "reliability_state": "M3T_SUSPECT",
                "request_reinit": False,
                "stop_m3t": False,
                "reason": "weak validation failures",
                "m3t_good_streak": 14,
                "m3t_bad_streak": 0,
                "external_conflict_streak": 0,
                "out_of_view_streak": 0,
                "weak_validation_fail_streak": 6,
            },
            "m3t": {
                "depth_consistency_ratio": 0.0,
                "black_white_ok": False,
                "smoothness_ok": True,
                "rendered_pixel_count": 80,
            },
        }

        result = analyze_tracking_health(current, previous)

        self.assertTrue(result.should_emit)
        self.assertEqual(result.primary_cause, "weak_validation")
        self.assertIn("M3T_OWNED -> M3T_SUSPECT", result.summary)


if __name__ == "__main__":
    unittest.main()
