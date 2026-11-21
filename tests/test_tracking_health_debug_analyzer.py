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

    def test_observation_unavailable_is_primary_cause(self):
        payload = _payload(
            "M3T_SUSPECT",
            "watch_tracking",
            reason="observation_unavailable",
            weak_validation_fail_streak=3,
        )

        result = analyze_tracking_health(payload)

        self.assertEqual(result.primary_cause, "observation_unavailable")

    def test_support_timeout_is_primary_cause(self):
        payload = _payload(
            "LOST",
            "stop_m3t",
            reason="support_timeout",
            stop_m3t=True,
            weak_validation_fail_streak=4,
        )

        result = analyze_tracking_health(payload)

        self.assertTrue(result.should_emit)
        self.assertEqual(result.primary_cause, "support_timeout")

    def test_out_of_view_reason_is_primary_cause(self):
        payload = _payload(
            "LOST",
            "stop_m3t",
            reason="out_of_view",
            stop_m3t=True,
            out_of_view_streak=9,
        )

        result = analyze_tracking_health(payload)

        self.assertTrue(result.should_emit)
        self.assertEqual(result.primary_cause, "out_of_view")

    def test_legacy_out_of_view_reason_is_primary_cause(self):
        payload = _payload(
            "M3T_SUSPECT",
            "watch_tracking",
            reason="out of view",
            out_of_view_streak=5,
        )

        result = analyze_tracking_health(payload)

        self.assertEqual(result.primary_cause, "out_of_view")

    def test_startup_timeout_reason_is_primary_cause(self):
        payload = _payload(
            "LOST",
            "stop_m3t",
            reason="startup_timeout",
            stop_m3t=True,
        )

        result = analyze_tracking_health(payload)

        self.assertTrue(result.should_emit)
        self.assertEqual(result.primary_cause, "startup_timeout")

    def test_stop_ack_timeout_action_is_primary_cause_and_emits(self):
        payload = _payload(
            "NO_TRACK",
            "stop_ack_timeout",
            reason="stop ack timeout",
        )

        result = analyze_tracking_health(payload)

        self.assertTrue(result.should_emit)
        self.assertEqual(result.primary_cause, "stop_ack_timeout")
        self.assertIn("stop_ack_timeout", result.summary)

    def test_recovered_to_owned_is_emitted(self):
        previous = _payload("M3T_SUSPECT", "watch_tracking", reason="weak validation failures")
        current = _payload("M3T_OWNED", "publish_trusted_pose", reason="m3t-owned", good=2)

        result = analyze_tracking_health(current, previous)

        self.assertTrue(result.should_emit)
        self.assertIn("M3T_SUSPECT -> M3T_OWNED", result.summary)

    def test_unchanged_owned_is_not_emitted(self):
        previous = _payload("M3T_OWNED", "publish_trusted_pose", reason="m3t-owned", good=20)
        current = _payload("M3T_OWNED", "publish_trusted_pose", reason="m3t-owned", good=21)

        result = analyze_tracking_health(current, previous)

        self.assertFalse(result.should_emit)
        self.assertEqual(result.primary_cause, "healthy")

    def test_evidence_includes_recovery_and_health_fields(self):
        payload = _payload("M3T_SUSPECT", "watch_tracking", reason="weak validation failures")
        payload["tracking_filter"].update(
            recovery_streak=1,
            suspect_elapsed_ms=250.0,
            evidence_kind="weak",
            observation_is_new=True,
        )
        payload["m3t"].update(
            health_evaluated=True,
            depth_evaluable=False,
            valid_depth_overlap_count=12,
        )

        result = analyze_tracking_health(payload)
        text = " ".join(result.evidence)

        self.assertIn("recovery_streak=1", text)
        self.assertIn("suspect_elapsed_ms=250.0", text)
        self.assertIn("evidence_kind=weak", text)
        self.assertIn("observation_is_new=True", text)
        self.assertIn("health_evaluated=True", text)
        self.assertIn("depth_evaluable=False", text)
        self.assertIn("valid_depth_overlap_count=12", text)


def _payload(state, action, reason="", good=0, stop_m3t=False, weak_validation_fail_streak=0, out_of_view_streak=0):
    return {
        "action": action,
        "tracking_filter": {
            "reliability_state": state,
            "request_reinit": False,
            "stop_m3t": stop_m3t,
            "reason": reason,
            "m3t_good_streak": good,
            "m3t_bad_streak": 0,
            "external_conflict_streak": 0,
            "out_of_view_streak": out_of_view_streak,
            "weak_validation_fail_streak": weak_validation_fail_streak,
        },
        "m3t": {
            "depth_consistency_ratio": 0.0,
            "projection_iou": 0.0,
            "black_white_ok": False,
            "smoothness_ok": False,
            "rendered_pixel_count": 0,
            "reason": "",
        },
    }


if __name__ == "__main__":
    unittest.main()
