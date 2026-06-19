import unittest

from scripts.tracking_state_filter import (
    LOST,
    M3T_OWNED,
    M3T_STARTING,
    M3T_SUSPECT,
    REINIT_NEEDED,
    TrackingStateFilter,
    TrackingStateFilterParams,
    TrackingStateObservation,
)


class TrackingStateFilterTest(unittest.TestCase):
    def make_filter(self):
        return TrackingStateFilter(
            TrackingStateFilterParams(
                startup_grace_frames=5,
                owned_confirm_frames=3,
                recover_frames=2,
                suspect_frames=4,
                reinit_conflict_frames=4,
                bad_tracking_frames=4,
                out_of_view_frames=6,
                min_rendered_pixels_for_owned=40,
                max_fresh_pose_age_ms=250.0,
            )
        )

    def m3t_good(self, yolo_visible=False, rendered_pixel_count=120):
        return TrackingStateObservation(
            yolo_visible=yolo_visible,
            yolo_confidence=0.80 if yolo_visible else 0.0,
            m3t_status="tracking",
            m3t_pose_available=True,
            m3t_pose_age_ms=20.0,
            depth_ok=True,
            depth_ratio=0.45,
            black_white_ok=True,
            projection_ok=True,
            projection_center_ok=True,
            projection_area_ok=True,
            smoothness_ok=True,
            rendered_pixel_count=rendered_pixel_count,
        )

    def strong_m3t_evidence_with_appearance_failure(self):
        return TrackingStateObservation(
            yolo_visible=False,
            yolo_confidence=0.0,
            m3t_status="tracking",
            m3t_pose_available=True,
            m3t_pose_age_ms=20.0,
            depth_ok=True,
            depth_ratio=0.98,
            black_white_ok=False,
            projection_ok=True,
            projection_center_ok=True,
            projection_area_ok=True,
            smoothness_ok=True,
            rendered_pixel_count=8000,
        )

    def strong_m3t_evidence_with_smoothness_warning(self):
        return TrackingStateObservation(
            yolo_visible=False,
            yolo_confidence=0.0,
            m3t_status="tracking",
            m3t_pose_available=True,
            m3t_pose_age_ms=20.0,
            depth_ok=False,
            depth_ratio=0.599,
            black_white_ok=False,
            projection_ok=True,
            projection_center_ok=True,
            projection_area_ok=True,
            smoothness_ok=False,
            rendered_pixel_count=17475,
        )

    def startup_m3t_evidence_with_smoothness_warning(self):
        return TrackingStateObservation(
            yolo_visible=False,
            yolo_confidence=0.0,
            m3t_status="tracking",
            m3t_pose_available=True,
            m3t_pose_age_ms=20.0,
            depth_ok=True,
            depth_ratio=0.958,
            black_white_ok=True,
            projection_ok=True,
            projection_center_ok=True,
            projection_area_ok=True,
            smoothness_ok=False,
            rendered_pixel_count=21719,
        )

    def black_white_only_without_depth_support(self):
        return TrackingStateObservation(
            yolo_visible=False,
            yolo_confidence=0.0,
            m3t_status="tracking",
            m3t_pose_available=True,
            m3t_pose_age_ms=20.0,
            depth_ok=False,
            depth_ratio=0.0,
            black_white_ok=True,
            projection_ok=True,
            projection_center_ok=True,
            projection_area_ok=True,
            smoothness_ok=False,
            rendered_pixel_count=13054,
        )

    def rendered_but_depth_absent(self):
        return TrackingStateObservation(
            yolo_visible=False,
            yolo_confidence=0.0,
            m3t_status="tracking",
            m3t_pose_available=True,
            m3t_pose_age_ms=20.0,
            depth_ok=False,
            depth_ratio=0.0,
            black_white_ok=False,
            projection_ok=True,
            projection_center_ok=True,
            projection_area_ok=True,
            smoothness_ok=True,
            rendered_pixel_count=15155,
        )

    def weak_external_failure_but_m3t_good(self):
        return TrackingStateObservation(
            yolo_visible=False,
            yolo_confidence=0.0,
            m3t_status="tracking",
            m3t_pose_available=True,
            m3t_pose_age_ms=20.0,
            depth_ok=False,
            depth_ratio=0.20,
            black_white_ok=False,
            projection_ok=True,
            projection_center_ok=True,
            projection_area_ok=True,
            smoothness_ok=True,
            rendered_pixel_count=80,
        )

    def wrong_instance_conflict(self):
        return TrackingStateObservation(
            yolo_visible=True,
            yolo_confidence=0.85,
            m3t_status="tracking",
            m3t_pose_available=True,
            m3t_pose_age_ms=20.0,
            depth_ok=False,
            depth_ratio=0.0,
            black_white_ok=False,
            projection_ok=False,
            projection_center_ok=False,
            projection_area_ok=False,
            smoothness_ok=False,
            rendered_pixel_count=100,
        )

    def target_left_view(self):
        return TrackingStateObservation(
            yolo_visible=False,
            yolo_confidence=0.0,
            m3t_status="tracking",
            m3t_pose_available=True,
            m3t_pose_age_ms=20.0,
            depth_ok=False,
            depth_ratio=0.0,
            black_white_ok=False,
            projection_ok=True,
            projection_center_ok=True,
            projection_area_ok=True,
            smoothness_ok=False,
            rendered_pixel_count=0,
        )

    def stale_m3t_pose(self):
        return TrackingStateObservation(
            yolo_visible=False,
            yolo_confidence=0.0,
            m3t_status="tracking",
            m3t_pose_available=False,
            m3t_pose_age_ms=999.0,
            depth_ok=False,
            depth_ratio=0.0,
            black_white_ok=False,
            projection_ok=True,
            projection_center_ok=True,
            projection_area_ok=True,
            smoothness_ok=True,
            rendered_pixel_count=0,
        )

    def enter_owned_state(self):
        state_filter = self.make_filter()
        state_filter.reset_for_tracking()
        decision = None
        for _ in range(3):
            decision = state_filter.update(self.m3t_good(yolo_visible=False))
        self.assertEqual(decision.reliability_state, M3T_OWNED)
        return state_filter

    def test_starting_waits_for_first_live_m3t_pose(self):
        state_filter = self.make_filter()
        state_filter.reset_for_tracking()

        decision = None
        for _ in range(5):
            decision = state_filter.update(self.stale_m3t_pose())

        self.assertEqual(decision.reliability_state, M3T_STARTING)
        self.assertTrue(decision.keep_m3t_alive)
        self.assertFalse(decision.request_reinit)
        self.assertFalse(decision.stop_m3t)

    def test_starting_fails_if_m3t_never_becomes_good(self):
        state_filter = self.make_filter()
        state_filter.reset_for_tracking()

        decision = None
        for _ in range(9):
            decision = state_filter.update(self.stale_m3t_pose())

        self.assertEqual(decision.reliability_state, LOST)
        self.assertTrue(decision.stop_m3t)

    def test_starting_accepts_strong_m3t_evidence_despite_black_white_failure(self):
        state_filter = self.make_filter()
        state_filter.reset_for_tracking()

        decision = state_filter.update(self.strong_m3t_evidence_with_appearance_failure())

        self.assertEqual(decision.reliability_state, M3T_OWNED)
        self.assertTrue(decision.publish_trusted_pose)
        self.assertFalse(decision.stop_m3t)
        self.assertGreaterEqual(decision.startup_good_total, 1)

    def test_starting_accepts_strong_m3t_evidence_despite_smoothness_warning(self):
        state_filter = self.make_filter()
        state_filter.reset_for_tracking()

        decision = state_filter.update(
            self.startup_m3t_evidence_with_smoothness_warning()
        )

        self.assertEqual(decision.reliability_state, M3T_OWNED)
        self.assertTrue(decision.publish_trusted_pose)
        self.assertFalse(decision.stop_m3t)
        self.assertGreaterEqual(decision.startup_good_total, 1)

    def test_starting_uses_cumulative_good_frames_not_only_consecutive_frames(self):
        state_filter = self.make_filter()
        state_filter.reset_for_tracking()

        state_filter.update(self.m3t_good(yolo_visible=False))
        state_filter.update(self.stale_m3t_pose())
        state_filter.update(self.m3t_good(yolo_visible=False))
        decision = state_filter.update(self.m3t_good(yolo_visible=False))

        self.assertEqual(decision.reliability_state, M3T_OWNED)
        self.assertGreaterEqual(decision.startup_good_total, 3)

    def test_starting_rejects_black_white_only_without_depth_support(self):
        state_filter = self.make_filter()
        state_filter.reset_for_tracking()

        decision = None
        for _ in range(5):
            decision = state_filter.update(
                self.black_white_only_without_depth_support()
            )

        self.assertEqual(decision.reliability_state, M3T_STARTING)
        self.assertEqual(decision.startup_good_total, 0)
        self.assertFalse(decision.publish_trusted_pose)
        self.assertTrue(decision.keep_m3t_alive)

    def test_owned_tracking_survives_unstable_yolo_without_reinit(self):
        state_filter = self.enter_owned_state()

        decision = None
        for _ in range(20):
            decision = state_filter.update(self.m3t_good(yolo_visible=False))

        self.assertEqual(decision.reliability_state, M3T_OWNED)
        self.assertTrue(decision.publish_trusted_pose)
        self.assertFalse(decision.request_reinit)
        self.assertFalse(decision.stop_m3t)

    def test_owned_tracking_becomes_suspect_but_alive_on_weak_failures(self):
        state_filter = self.enter_owned_state()

        decision = None
        for _ in range(4):
            decision = state_filter.update(self.weak_external_failure_but_m3t_good())

        self.assertEqual(decision.reliability_state, M3T_SUSPECT)
        self.assertTrue(decision.keep_m3t_alive)
        self.assertFalse(decision.request_reinit)
        self.assertFalse(decision.stop_m3t)

    def test_owned_tracking_keeps_m3t_alive_when_only_smoothness_warns(self):
        state_filter = self.enter_owned_state()

        decision = None
        for _ in range(8):
            decision = state_filter.update(
                self.strong_m3t_evidence_with_smoothness_warning()
            )

        self.assertIn(decision.reliability_state, {M3T_OWNED, M3T_SUSPECT})
        self.assertTrue(decision.keep_m3t_alive)
        self.assertFalse(decision.request_reinit)
        self.assertFalse(decision.stop_m3t)

    def test_suspect_recovers_to_owned_after_good_m3t_frames(self):
        state_filter = self.enter_owned_state()
        for _ in range(4):
            state_filter.update(self.weak_external_failure_but_m3t_good())

        decision = None
        for _ in range(2):
            decision = state_filter.update(self.m3t_good(yolo_visible=False))

        self.assertEqual(decision.reliability_state, M3T_OWNED)
        self.assertTrue(decision.publish_trusted_pose)

    def test_wrong_instance_never_enters_owned_and_requests_reinit(self):
        state_filter = self.make_filter()
        state_filter.reset_for_tracking()

        decision = None
        for _ in range(8):
            decision = state_filter.update(self.wrong_instance_conflict())

        self.assertEqual(decision.reliability_state, REINIT_NEEDED)
        self.assertTrue(decision.request_reinit)
        self.assertFalse(decision.stop_m3t)
        self.assertEqual(decision.owned_confirm_streak, 0)

    def test_owned_tracking_reinitializes_on_sustained_external_conflict(self):
        state_filter = self.enter_owned_state()

        decision = None
        for _ in range(4):
            decision = state_filter.update(self.wrong_instance_conflict())

        self.assertEqual(decision.reliability_state, REINIT_NEEDED)
        self.assertTrue(decision.request_reinit)
        self.assertFalse(decision.stop_m3t)

    def test_owned_tracking_stops_when_target_leaves_image_long_enough(self):
        state_filter = self.enter_owned_state()

        decision = None
        for _ in range(6):
            decision = state_filter.update(self.target_left_view())

        self.assertEqual(decision.reliability_state, LOST)
        self.assertTrue(decision.stop_m3t)
        self.assertFalse(decision.request_reinit)

    def test_owned_tracking_reinitializes_when_rendered_pose_has_no_depth_support(self):
        state_filter = self.enter_owned_state()

        decision = None
        for _ in range(4):
            decision = state_filter.update(self.rendered_but_depth_absent())

        self.assertEqual(decision.reliability_state, REINIT_NEEDED)
        self.assertTrue(decision.request_reinit)
        self.assertFalse(decision.stop_m3t)

    def test_diagnostics_include_streak_counters(self):
        state_filter = self.enter_owned_state()

        payload = state_filter.update(self.m3t_good(yolo_visible=False)).to_dict()

        self.assertIn("m3t_good_streak", payload)
        self.assertIn("m3t_bad_streak", payload)
        self.assertIn("external_conflict_streak", payload)
        self.assertIn("out_of_view_streak", payload)
        self.assertIn("weak_validation_fail_streak", payload)


if __name__ == "__main__":
    unittest.main()
