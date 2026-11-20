import unittest

from scripts.tracking_state_filter import (
    EVIDENCE_ABSENCE,
    EVIDENCE_SUPPORT,
    LOST,
    M3T_OWNED,
    M3T_STARTING,
    M3T_SUSPECT,
    REINIT_NEEDED,
    TrackingStateFilter,
    TrackingStateFilterParams,
    TrackingStateObservation,
)


BASE_NS = 1_000_000_000


def observation_at(ms, **changes):
    """Default is a fresh, fully evaluated, positively supported sample.

    `ms` sets the pose stamp, observation stamp and monotonic time. Equal `ms`
    values make a duplicate sample; a smaller value makes a backward sample.
    """
    stamp = BASE_NS + int(ms) * 1_000_000
    fields = dict(
        yolo_visible=False,
        yolo_confidence=0.0,
        yolo_observation_valid=True,
        m3t_status="tracking",
        m3t_pose_available=True,
        m3t_pose_age_ms=20.0,
        health_evaluated=True,
        valid_depth_overlap_count=8000,
        depth_evaluable=True,
        depth_ok=True,
        depth_ratio=0.80,
        black_white_ok=True,
        projection_evaluated=True,
        projection_ok=True,
        projection_center_ok=True,
        projection_area_ok=True,
        smoothness_ok=True,
        rendered_pixel_count=8000,
        pose_stamp_ns=stamp,
        observation_stamp_ns=stamp,
        now_monotonic_ns=stamp,
    )
    fields.update(changes)
    return TrackingStateObservation(**fields)


def good(ms, **changes):
    return observation_at(ms, **changes)


def weak(ms, **changes):
    # Rendered and depth-evaluable, but the depth agreement is below 0.60.
    return observation_at(ms, depth_ratio=0.20, black_white_ok=False, **changes)


def unknown(ms, **changes):
    # Depth overlap is too small to evaluate, so nothing can be measured.
    return observation_at(
        ms,
        valid_depth_overlap_count=0,
        depth_evaluable=False,
        depth_ratio=0.0,
        black_white_ok=False,
        **changes,
    )


def stale(ms, **changes):
    return observation_at(ms, m3t_pose_available=False, m3t_pose_age_ms=999.0, **changes)


def absent(ms, **changes):
    # Fresh YOLO with no target, and the rendered projection is empty.
    return observation_at(
        ms,
        yolo_visible=False,
        rendered_pixel_count=0,
        valid_depth_overlap_count=0,
        depth_evaluable=False,
        depth_ratio=0.0,
        black_white_ok=False,
        smoothness_ok=False,
        **changes,
    )


def conflict(ms, **changes):
    # YOLO sees the target, but the synchronized projection disagrees.
    return observation_at(
        ms,
        yolo_visible=True,
        yolo_confidence=0.85,
        projection_ok=False,
        projection_center_ok=False,
        projection_area_ok=False,
        depth_ratio=0.0,
        valid_depth_overlap_count=0,
        depth_evaluable=False,
        rendered_pixel_count=100,
        **changes,
    )


def depth_contradiction(ms, **changes):
    # Enough overlap to measure depth, and the measured agreement is zero.
    return observation_at(ms, depth_ratio=0.0, black_white_ok=True, **changes)


def owned_filter(params=None):
    state_filter = TrackingStateFilter(params or make_params())
    state_filter.reset_for_tracking(now_monotonic_ns=BASE_NS)
    for ms in (0, 125, 250):
        decision = state_filter.update(good(ms))
    assert decision.reliability_state == M3T_OWNED
    return state_filter


def make_params():
    return TrackingStateFilterParams(
        owned_confirm_frames=3,
        recover_frames=2,
        max_fresh_pose_age_ms=250.0,
        min_rendered_pixels_for_owned=40,
    )


class TrackingStateFilterTest(unittest.TestCase):
    def run_samples(self, state_filter, samples):
        return [state_filter.update(sample) for sample in samples]

    def fresh_filter(self):
        state_filter = TrackingStateFilter(make_params())
        state_filter.reset_for_tracking(now_monotonic_ns=BASE_NS)
        return state_filter

    # Startup

    def test_starting_waits_for_first_live_m3t_pose(self):
        state_filter = self.fresh_filter()
        decisions = self.run_samples(state_filter, [stale(ms) for ms in range(0, 625, 125)])

        self.assertEqual(decisions[-1].reliability_state, M3T_STARTING)
        self.assertTrue(decisions[-1].keep_m3t_alive)
        self.assertFalse(decisions[-1].request_reinit)
        self.assertFalse(decisions[-1].stop_m3t)

    def test_starting_fails_on_startup_timeout_not_after_frame_count(self):
        state_filter = self.fresh_filter()
        decisions = self.run_samples(
            state_filter, [stale(ms) for ms in range(0, 5001, 125)]
        )

        self.assertEqual(decisions[-2].reliability_state, M3T_STARTING)
        self.assertEqual(decisions[-1].reliability_state, LOST)
        self.assertEqual(decisions[-1].reason, "startup_timeout")
        self.assertTrue(decisions[-1].stop_m3t)

    def test_strong_evidence_needs_three_consecutive_new_samples(self):
        state_filter = self.fresh_filter()
        decisions = self.run_samples(
            state_filter,
            [good(0, black_white_ok=False), good(125, black_white_ok=False)],
        )
        self.assertEqual(decisions[-1].reliability_state, M3T_STARTING)

        decision = state_filter.update(good(250, black_white_ok=False))
        self.assertEqual(decision.reliability_state, M3T_OWNED)
        self.assertTrue(decision.publish_trusted_pose)
        self.assertFalse(decision.stop_m3t)

    def test_startup_requires_consecutive_support(self):
        state_filter = self.fresh_filter()
        samples = [
            good(0), stale(125), good(250), stale(375), good(500), good(625),
        ]
        decisions = self.run_samples(state_filter, samples)
        self.assertNotEqual(decisions[-1].reliability_state, M3T_OWNED)

        decision = state_filter.update(good(750))
        self.assertEqual(decision.reliability_state, M3T_OWNED)

    def test_startup_unknown_depth_is_not_a_timeout_before_startup_deadline(self):
        state_filter = self.fresh_filter()
        decisions = self.run_samples(
            state_filter, [unknown(ms) for ms in range(0, 1001, 125)]
        )
        self.assertEqual(decisions[-1].reliability_state, M3T_STARTING)
        self.assertFalse(decisions[-1].publish_trusted_pose)

    def test_wrong_instance_never_enters_owned_and_requests_reinit(self):
        state_filter = self.fresh_filter()
        decisions = self.run_samples(
            state_filter, [conflict(ms) for ms in range(0, 751, 125)]
        )

        reinit_index = next(
            i for i, d in enumerate(decisions) if d.reliability_state == REINIT_NEEDED
        )
        self.assertEqual(decisions[reinit_index].reason, "external_conflict")
        self.assertEqual(reinit_index, 5)  # 625 ms after the first conflict sample
        self.assertTrue(decisions[-1].request_reinit)
        self.assertFalse(decisions[-1].stop_m3t)
        self.assertEqual(decisions[-1].owned_confirm_streak, 0)

    def test_future_age_nan_or_inf_is_not_support(self):
        cases = [
            dict(m3t_pose_age_ms=-5.0),
            dict(depth_ratio=float("nan")),
            dict(depth_ratio=float("inf")),
        ]
        for changes in cases:
            with self.subTest(changes=changes):
                state_filter = self.fresh_filter()
                decisions = self.run_samples(
                    state_filter,
                    [observation_at(ms, **changes) for ms in (0, 125, 250, 375)],
                )
                self.assertFalse(any(d.reliability_state == M3T_OWNED for d in decisions))
                self.assertFalse(any(d.publish_trusted_pose for d in decisions))

    # Owned state: outside, weak support, unknown evidence

    def test_owned_tracking_survives_unstable_yolo_without_reinit(self):
        state_filter = owned_filter()
        decisions = self.run_samples(
            state_filter, [good(375 + 125 * i, yolo_visible=False) for i in range(20)]
        )
        decision = decisions[-1]
        self.assertEqual(decision.reliability_state, M3T_OWNED)
        self.assertTrue(decision.publish_trusted_pose)
        self.assertFalse(decision.request_reinit)
        self.assertFalse(decision.stop_m3t)

    def test_static_outside_eventually_stops_after_absence_timeout(self):
        state_filter = owned_filter()
        samples = [absent(ms) for ms in range(375, 1876, 125)]
        decisions = self.run_samples(state_filter, samples)

        self.assertEqual(decisions[0].reliability_state, M3T_SUSPECT)
        self.assertEqual(decisions[0].evidence_kind, EVIDENCE_ABSENCE)
        self.assertTrue(all(not d.publish_trusted_pose for d in decisions))
        self.assertEqual(decisions[-2].reliability_state, M3T_SUSPECT)
        self.assertEqual(decisions[-1].reliability_state, LOST)
        self.assertEqual(decisions[-1].reason, "out_of_view")
        self.assertTrue(decisions[-1].stop_m3t)

    def test_weak_depth_without_yolo_times_out_at_500ms(self):
        state_filter = owned_filter()
        samples = [weak(ms) for ms in range(375, 876, 125)]
        decisions = self.run_samples(state_filter, samples)

        self.assertEqual(decisions[0].reliability_state, M3T_SUSPECT)
        self.assertFalse(any(d.reliability_state == M3T_OWNED for d in decisions))
        self.assertEqual(decisions[-3].reliability_state, M3T_SUSPECT)  # 625 ms: 250 ms elapsed
        self.assertEqual(decisions[-1].reliability_state, LOST)
        self.assertEqual(decisions[-1].reason, "support_timeout")
        self.assertTrue(decisions[-1].stop_m3t)

    def test_unknown_depth_without_yolo_uses_500ms_timeout(self):
        state_filter = owned_filter()
        samples = [unknown(ms) for ms in range(375, 876, 125)]
        decisions = self.run_samples(state_filter, samples)

        self.assertEqual(decisions[-2].reliability_state, M3T_SUSPECT)  # 750 ms: 375 ms elapsed
        self.assertEqual(decisions[-1].reliability_state, LOST)  # 875 ms: 500 ms elapsed
        self.assertEqual(decisions[-1].reason, "observation_unavailable")
        self.assertTrue(decisions[-1].stop_m3t)

    def test_visible_target_conflict_requests_reinit_after_625ms(self):
        state_filter = owned_filter()
        samples = [conflict(ms) for ms in (375, 500, 625, 750, 875, 1000)]
        decisions = self.run_samples(state_filter, samples)

        self.assertEqual(decisions[4].reliability_state, M3T_SUSPECT)  # 500 ms elapsed
        self.assertEqual(decisions[-1].reliability_state, REINIT_NEEDED)  # 625 ms elapsed
        self.assertEqual(decisions[-1].reason, "external_conflict")
        self.assertTrue(decisions[-1].request_reinit)
        self.assertFalse(decisions[-1].stop_m3t)

    def test_owned_tracking_reinitializes_on_sustained_external_conflict(self):
        state_filter = owned_filter()
        decisions = self.run_samples(
            state_filter, [conflict(ms) for ms in (375, 500, 625, 750, 875, 1000)]
        )
        self.assertEqual(decisions[-1].reliability_state, REINIT_NEEDED)
        self.assertTrue(decisions[-1].request_reinit)
        self.assertFalse(decisions[-1].stop_m3t)

    def test_visible_target_weak_support_keeps_2000ms_suspect_deadline(self):
        state_filter = owned_filter()
        samples = [weak(ms, yolo_visible=True, yolo_confidence=0.8) for ms in range(375, 2376, 125)]
        decisions = self.run_samples(state_filter, samples)

        self.assertFalse(any(d.publish_trusted_pose for d in decisions))
        self.assertEqual(decisions[-2].reliability_state, M3T_SUSPECT)  # 2250 ms
        self.assertEqual(decisions[-1].reliability_state, REINIT_NEEDED)  # 2375 ms
        self.assertEqual(decisions[-1].reason, "support_timeout")
        self.assertFalse(decisions[-1].stop_m3t)

    def test_depth_contradiction_without_yolo_is_not_out_of_view(self):
        state_filter = owned_filter()
        decisions = self.run_samples(
            state_filter, [depth_contradiction(ms) for ms in (375, 500, 625, 750, 875)]
        )
        self.assertEqual(decisions[-1].reliability_state, LOST)
        self.assertEqual(decisions[-1].reason, "support_timeout")

    def test_appearance_pass_cannot_override_bad_depth(self):
        state_filter = owned_filter()
        decisions = self.run_samples(
            state_filter,
            [depth_contradiction(ms, smoothness_ok=True) for ms in (375, 500, 625, 750, 875)],
        )
        self.assertFalse(any(d.reliability_state == M3T_OWNED for d in decisions))
        self.assertEqual(decisions[-1].reliability_state, LOST)

    # Recovery, duplicates and deadlines

    def test_short_occlusion_recovers_without_stop_or_reinit(self):
        state_filter = owned_filter()
        samples = [
            weak(375), weak(500), weak(625),  # 250 ms of weak evidence
            good(750), good(875),
        ]
        decisions = self.run_samples(state_filter, samples)

        self.assertEqual(decisions[3].reliability_state, M3T_SUSPECT)
        self.assertEqual(decisions[-1].reliability_state, M3T_OWNED)
        self.assertFalse(any(d.stop_m3t or d.request_reinit for d in decisions))

    def test_recovery_requires_two_new_supporting_observations(self):
        state_filter = owned_filter()
        self.run_samples(state_filter, [weak(375)])

        first_new_good = state_filter.update(good(500))
        self.assertEqual(first_new_good.reliability_state, M3T_SUSPECT)
        self.assertEqual(first_new_good.recovery_streak, 1)

        repeated_same_good = state_filter.update(good(500))
        self.assertEqual(repeated_same_good.reliability_state, M3T_SUSPECT)
        self.assertEqual(repeated_same_good.recovery_streak, 1)
        self.assertFalse(repeated_same_good.observation_is_new)

        second_new_good = state_filter.update(good(625))
        self.assertEqual(second_new_good.reliability_state, M3T_OWNED)
        self.assertTrue(second_new_good.publish_trusted_pose)

    def test_owned_pose_going_stale_drops_trusted_output(self):
        state_filter = owned_filter()
        decision = state_filter.update(stale(375))
        self.assertEqual(decision.reliability_state, M3T_SUSPECT)
        self.assertFalse(decision.publish_trusted_pose)

    def test_duplicate_or_backward_stamps_do_not_confirm(self):
        state_filter = self.fresh_filter()
        decisions = self.run_samples(
            state_filter, [good(0), good(0), good(0), good(-125)]
        )
        self.assertEqual(decisions[-1].reliability_state, M3T_STARTING)
        self.assertEqual(decisions[-1].owned_confirm_streak, 1)
        self.assertFalse(any(d.observation_is_new for d in decisions[1:]))

        decision = state_filter.update(good(125))
        self.assertEqual(decision.owned_confirm_streak, 2)
        decision = state_filter.update(good(250))
        self.assertEqual(decision.reliability_state, M3T_OWNED)

    def test_duplicate_samples_do_not_refresh_the_no_target_deadline(self):
        state_filter = owned_filter()
        self.run_samples(state_filter, [unknown(375)])

        # Same pose stamp, but the monotonic clock has advanced to 875 ms.
        duplicate = unknown(375, now_monotonic_ns=BASE_NS + 875 * 1_000_000)
        decision = state_filter.update(duplicate)
        self.assertEqual(decision.reliability_state, LOST)
        self.assertEqual(decision.reason, "observation_unavailable")

    def test_good_sample_cannot_cancel_an_expired_deadline(self):
        state_filter = owned_filter()
        self.run_samples(state_filter, [unknown(ms) for ms in (375, 500, 625, 750)])

        decision = state_filter.update(good(875))
        self.assertEqual(decision.reliability_state, LOST)
        self.assertEqual(decision.reason, "observation_unavailable")
        self.assertTrue(decision.stop_m3t)

    def test_absence_evidence_is_counted_as_absence(self):
        state_filter = owned_filter()
        decision = state_filter.update(absent(375))
        self.assertEqual(decision.evidence_kind, EVIDENCE_ABSENCE)
        self.assertEqual(decision.out_of_view_streak, 1)

    def test_positive_support_is_reported_as_support(self):
        state_filter = owned_filter()
        decision = state_filter.update(good(375))
        self.assertEqual(decision.evidence_kind, EVIDENCE_SUPPORT)
        self.assertTrue(decision.observation_is_new)

    def test_diagnostics_include_streak_counters_and_deadline_fields(self):
        state_filter = owned_filter()
        payload = state_filter.update(good(375)).to_dict()

        for key in (
            "m3t_good_streak",
            "m3t_bad_streak",
            "external_conflict_streak",
            "out_of_view_streak",
            "weak_validation_fail_streak",
            "recovery_streak",
            "suspect_elapsed_ms",
            "evidence_kind",
            "observation_is_new",
        ):
            self.assertIn(key, payload)


if __name__ == "__main__":
    unittest.main()
