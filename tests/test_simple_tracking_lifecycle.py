import unittest

from scripts.simple_tracking_lifecycle import (
    LifecycleObservation,
    LifecycleParams,
    SimpleTrackingLifecycle,
)


class SimpleTrackingLifecycleTest(unittest.TestCase):
    def make_lifecycle(self):
        return SimpleTrackingLifecycle(
            LifecycleParams(
                stable_yolo_frames=2,
                lost_frames=2,
                out_of_view_frames=2,
                candidate_reject_cooldown_frames=2,
            )
        )

    def test_no_track_requests_candidate_after_stable_yolo(self):
        lifecycle = self.make_lifecycle()

        first = lifecycle.update(LifecycleObservation(yolo_visible=True))
        second = lifecycle.update(LifecycleObservation(yolo_visible=True))

        self.assertEqual(first.state, "NO_TRACK")
        self.assertFalse(first.publish_init_request)
        self.assertEqual(second.state, "NO_TRACK")
        self.assertTrue(second.publish_init_request)
        self.assertEqual(second.action, "request_candidate")

    def test_valid_candidate_starts_m3t_and_enters_tracking(self):
        lifecycle = self.make_lifecycle()
        lifecycle.update(LifecycleObservation(yolo_visible=True))
        lifecycle.update(LifecycleObservation(yolo_visible=True))

        decision = lifecycle.update(
            LifecycleObservation(
                yolo_visible=True,
                candidate_available=True,
                candidate_valid=True,
            )
        )

        self.assertEqual(decision.state, "TRACKING")
        self.assertEqual(decision.action, "start_m3t")
        self.assertTrue(decision.publish_initial_pose)

    def test_invalid_candidate_clears_candidate_and_stays_no_track(self):
        lifecycle = self.make_lifecycle()
        lifecycle.update(LifecycleObservation(yolo_visible=True))
        lifecycle.update(LifecycleObservation(yolo_visible=True))

        decision = lifecycle.update(
            LifecycleObservation(
                yolo_visible=True,
                candidate_available=True,
                candidate_valid=False,
            )
        )

        self.assertEqual(decision.state, "NO_TRACK")
        self.assertEqual(decision.action, "reject_candidate")
        self.assertTrue(decision.clear_candidate)
        self.assertFalse(decision.publish_initial_pose)

    def test_rejected_candidate_requires_cooldown_before_next_request(self):
        lifecycle = self.make_lifecycle()
        lifecycle.update(LifecycleObservation(yolo_visible=True))
        lifecycle.update(LifecycleObservation(yolo_visible=True))
        lifecycle.update(
            LifecycleObservation(
                yolo_visible=True,
                candidate_available=True,
                candidate_valid=False,
            )
        )

        cooldown = lifecycle.update(LifecycleObservation(yolo_visible=True))
        ready_again_first = lifecycle.update(LifecycleObservation(yolo_visible=True))
        ready_again_second = lifecycle.update(LifecycleObservation(yolo_visible=True))

        self.assertEqual(cooldown.action, "candidate_reject_cooldown")
        self.assertFalse(cooldown.publish_init_request)
        self.assertFalse(ready_again_first.publish_init_request)
        self.assertTrue(ready_again_second.publish_init_request)

    def test_tracking_health_good_publishes_trusted_pose(self):
        lifecycle = self.tracking_lifecycle()

        decision = lifecycle.update(
            LifecycleObservation(
                yolo_visible=True,
                m3t_status="tracking",
                m3t_pose_available=True,
                tracking_health_ok=True,
            )
        )

        self.assertEqual(decision.state, "TRACKING")
        self.assertEqual(decision.action, "publish_trusted_pose")
        self.assertTrue(decision.publish_trusted_pose)

    def test_owned_filter_decision_publishes_trusted_pose(self):
        lifecycle = self.tracking_lifecycle()

        decision = lifecycle.update(
            LifecycleObservation(
                yolo_visible=False,
                m3t_status="tracking",
                m3t_pose_available=True,
                tracking_reliability_state="M3T_OWNED",
                filter_publish_trusted_pose=True,
                filter_keep_m3t_alive=True,
            )
        )

        self.assertEqual(decision.state, "TRACKING")
        self.assertEqual(decision.action, "publish_trusted_pose")
        self.assertTrue(decision.publish_trusted_pose)

    def test_reinit_needed_filter_decision_stops_and_enters_reinit(self):
        lifecycle = self.tracking_lifecycle()

        decision = lifecycle.update(
            LifecycleObservation(
                yolo_visible=True,
                m3t_status="tracking",
                m3t_pose_available=True,
                tracking_reliability_state="REINIT_NEEDED",
                filter_request_reinit=True,
                filter_keep_m3t_alive=True,
            )
        )

        self.assertEqual(decision.state, "REINIT")
        self.assertEqual(decision.action, "stop_m3t_for_reinit")
        self.assertTrue(decision.publish_stop)

    def test_lost_filter_decision_stops_and_returns_no_track(self):
        lifecycle = self.tracking_lifecycle()

        decision = lifecycle.update(
            LifecycleObservation(
                yolo_visible=False,
                m3t_status="tracking",
                m3t_pose_available=True,
                tracking_reliability_state="LOST",
                filter_stop_m3t=True,
                filter_keep_m3t_alive=False,
            )
        )

        self.assertEqual(decision.state, "NO_TRACK")
        self.assertEqual(decision.action, "stop_m3t")
        self.assertTrue(decision.publish_stop)

    def test_m3t_suspect_filter_decision_keeps_m3t_alive(self):
        lifecycle = self.tracking_lifecycle()

        decision = lifecycle.update(
            LifecycleObservation(
                yolo_visible=False,
                m3t_status="tracking",
                m3t_pose_available=True,
                tracking_reliability_state="M3T_SUSPECT",
                filter_keep_m3t_alive=True,
            )
        )

        self.assertEqual(decision.state, "TRACKING")
        self.assertEqual(decision.action, "watch_tracking")
        self.assertFalse(decision.publish_stop)
        self.assertFalse(decision.publish_trusted_pose)

    def test_soft_health_failure_with_visible_yolo_keeps_m3t_alive(self):
        lifecycle = self.tracking_lifecycle()

        lifecycle.update(
            LifecycleObservation(
                yolo_visible=True,
                m3t_status="tracking",
                m3t_pose_available=True,
                tracking_health_ok=False,
            )
        )
        decision = lifecycle.update(
            LifecycleObservation(
                yolo_visible=True,
                m3t_status="tracking",
                m3t_pose_available=True,
                tracking_health_ok=False,
            )
        )

        self.assertEqual(decision.state, "TRACKING")
        self.assertEqual(decision.action, "watch_tracking")
        self.assertFalse(decision.publish_stop)
        self.assertFalse(decision.publish_trusted_pose)

    def test_soft_health_failure_without_yolo_keeps_m3t_alive(self):
        lifecycle = self.tracking_lifecycle()

        lifecycle.update(
            LifecycleObservation(
                yolo_visible=False,
                m3t_status="tracking",
                m3t_pose_available=True,
                tracking_health_ok=False,
            )
        )
        decision = lifecycle.update(
            LifecycleObservation(
                yolo_visible=False,
                m3t_status="tracking",
                m3t_pose_available=True,
                tracking_health_ok=False,
            )
        )

        self.assertEqual(decision.state, "TRACKING")
        self.assertEqual(decision.action, "watch_tracking")
        self.assertFalse(decision.publish_stop)
        self.assertFalse(decision.publish_trusted_pose)

    def test_missing_m3t_pose_with_visible_yolo_stops_and_enters_reinit(self):
        lifecycle = self.tracking_lifecycle()

        lifecycle.update(
            LifecycleObservation(
                yolo_visible=True,
                m3t_status="tracking",
                m3t_pose_available=False,
                tracking_health_ok=False,
            )
        )
        decision = lifecycle.update(
            LifecycleObservation(
                yolo_visible=True,
                m3t_status="tracking",
                m3t_pose_available=False,
                tracking_health_ok=False,
            )
        )

        self.assertEqual(decision.state, "REINIT")
        self.assertEqual(decision.action, "stop_m3t_for_reinit")
        self.assertTrue(decision.publish_stop)

    def test_missing_m3t_pose_without_yolo_stops_and_returns_no_track(self):
        lifecycle = self.tracking_lifecycle()

        lifecycle.update(
            LifecycleObservation(
                yolo_visible=False,
                m3t_status="tracking",
                m3t_pose_available=False,
                tracking_health_ok=False,
            )
        )
        decision = lifecycle.update(
            LifecycleObservation(
                yolo_visible=False,
                m3t_status="tracking",
                m3t_pose_available=False,
                tracking_health_ok=False,
            )
        )

        self.assertEqual(decision.state, "NO_TRACK")
        self.assertEqual(decision.action, "stop_m3t")
        self.assertTrue(decision.publish_stop)

    def test_non_tracking_m3t_status_with_visible_yolo_stops_and_enters_reinit(self):
        lifecycle = self.tracking_lifecycle()

        lifecycle.update(
            LifecycleObservation(
                yolo_visible=True,
                m3t_status="error",
                m3t_pose_available=True,
                tracking_health_ok=False,
            )
        )
        decision = lifecycle.update(
            LifecycleObservation(
                yolo_visible=True,
                m3t_status="error",
                m3t_pose_available=True,
                tracking_health_ok=False,
            )
        )

        self.assertEqual(decision.state, "REINIT")
        self.assertEqual(decision.action, "stop_m3t_for_reinit")
        self.assertTrue(decision.publish_stop)

    def test_reinit_waits_for_stop_then_requests_candidate(self):
        lifecycle = self.tracking_lifecycle()
        for _ in range(2):
            lifecycle.update(
                LifecycleObservation(
                    yolo_visible=True,
                    m3t_status="tracking",
                    m3t_pose_available=False,
                    tracking_health_ok=False,
                )
            )

        waiting = lifecycle.update(LifecycleObservation(yolo_visible=True, m3t_status="stopping"))
        stopped = lifecycle.update(LifecycleObservation(yolo_visible=True, m3t_status="stopped"))

        self.assertEqual(waiting.state, "REINIT")
        self.assertEqual(waiting.action, "wait_for_m3t_stop")
        self.assertEqual(stopped.state, "REINIT")
        self.assertEqual(stopped.action, "request_candidate")
        self.assertTrue(stopped.publish_init_request)

    def tracking_lifecycle(self):
        lifecycle = self.make_lifecycle()
        lifecycle.update(LifecycleObservation(yolo_visible=True))
        lifecycle.update(LifecycleObservation(yolo_visible=True))
        lifecycle.update(
            LifecycleObservation(
                yolo_visible=True,
                candidate_available=True,
                candidate_valid=True,
            )
        )
        return lifecycle


if __name__ == "__main__":
    unittest.main()
