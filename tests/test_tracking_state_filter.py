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


def stale_pose_at(ms):
    # The pose stamp stays at the last live pose (BASE_NS), so the sample is a duplicate.
    return stale(0, now_monotonic_ns=BASE_NS + int(ms) * 1_000_000)


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
    # YOLO sees the target during startup, so its absence timer starts later.
    for ms in (0, 125, 250):
        decision = state_filter.update(good(ms, yolo_visible=True))
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
        self.assertFalse(decisions[-1].has_been_owned)

    def test_single_conflict_followed_by_stale_pose_does_not_trigger_reinit(self):
        sample_sets = {
            "stale pose stamp frozen": [conflict(0)]
            + [stale_pose_at(ms) for ms in range(125, 751, 125)],
            "stale pose stamp advancing": [conflict(0)]
            + [stale(ms) for ms in range(125, 751, 125)],
            "single stall to 700 ms": [conflict(0), stale_pose_at(700)],
        }
        for name, samples in sample_sets.items():
            with self.subTest(name=name):
                state_filter = self.fresh_filter()
                decisions = self.run_samples(state_filter, samples)

                self.assertFalse(any(d.request_reinit for d in decisions))
                self.assertFalse(any(d.reason == "external_conflict" for d in decisions))
                self.assertEqual(decisions[-1].reliability_state, M3T_STARTING)

    def test_intermittent_conflict_broken_by_stale_or_non_conflict_resets_conflict_timer(self):
        # A stale duplicate at 375 ms breaks the run, so the timer restarts at 500 ms.
        state_filter = self.fresh_filter()
        samples = [
            conflict(0), conflict(125), conflict(250), stale_pose_at(375),
            conflict(500), conflict(625), conflict(750), conflict(875), conflict(1000),
            conflict(1125),
        ]
        decisions = self.run_samples(state_filter, samples)
        self.assertFalse(any(d.request_reinit for d in decisions[:9]))
        self.assertEqual(decisions[8].reliability_state, M3T_STARTING)
        self.assertEqual(decisions[9].reliability_state, REINIT_NEEDED)
        self.assertEqual(decisions[9].reason, "external_conflict")

        # A new weak sample at 250 ms breaks the run, so the timer restarts at 375 ms.
        state_filter = self.fresh_filter()
        samples = [
            conflict(0), conflict(125), weak(250),
            conflict(375), conflict(500), conflict(625), conflict(750),
            conflict(875), conflict(1000),
        ]
        decisions = self.run_samples(state_filter, samples)
        self.assertFalse(any(d.request_reinit for d in decisions[:8]))
        self.assertEqual(decisions[7].reliability_state, M3T_STARTING)
        self.assertEqual(decisions[8].reliability_state, REINIT_NEEDED)
        self.assertEqual(decisions[8].reason, "external_conflict")

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

    # Owned state: brief YOLO dropouts, absence, conflict

    def test_owned_tracking_survives_brief_yolo_dropouts(self):
        state_filter = owned_filter()
        samples = [
            good(375, yolo_visible=False),
            good(500, yolo_visible=False),  # 250 ms dropout
            good(625, yolo_visible=True),
            good(750, yolo_visible=False),
            good(875, yolo_visible=False),  # 250 ms dropout
            good(1000, yolo_visible=True),
        ]
        decisions = self.run_samples(state_filter, samples)

        self.assertTrue(all(d.reliability_state == M3T_OWNED for d in decisions))
        self.assertTrue(all(d.publish_trusted_pose for d in decisions))
        self.assertFalse(any(d.request_reinit or d.stop_m3t for d in decisions))

    def test_owned_good_depth_with_continuous_yolo_absence_stops_at_750ms(self):
        state_filter = owned_filter()
        samples = [good(ms, yolo_visible=False) for ms in range(375, 1126, 125)]
        decisions = self.run_samples(state_filter, samples)

        self.assertEqual(decisions[5].reliability_state, M3T_OWNED)  # 625 ms absent
        self.assertTrue(decisions[5].publish_trusted_pose)
        self.assertEqual(decisions[6].reliability_state, LOST)  # 750 ms absent
        self.assertEqual(decisions[6].reason, "out_of_view")
        self.assertTrue(decisions[6].stop_m3t)
        self.assertTrue(decisions[6].has_been_owned)
        self.assertTrue(decisions[6].absence_confirmed)

    def test_static_outside_eventually_stops_after_absence_timeout(self):
        state_filter = owned_filter()
        samples = [absent(ms) for ms in range(375, 1876, 125)]
        decisions = self.run_samples(state_filter, samples)

        self.assertEqual(decisions[0].reliability_state, M3T_SUSPECT)
        self.assertEqual(decisions[0].evidence_kind, EVIDENCE_ABSENCE)
        self.assertTrue(all(not d.publish_trusted_pose for d in decisions))
        self.assertEqual(decisions[5].reliability_state, M3T_SUSPECT)  # 1000 ms: 625 ms elapsed
        self.assertEqual(decisions[6].reliability_state, LOST)  # 1125 ms: 750 ms elapsed
        self.assertEqual(decisions[6].reason, "out_of_view")
        self.assertTrue(decisions[6].stop_m3t)
        self.assertTrue(decisions[6].absence_confirmed)

    def test_weak_depth_with_absent_yolo_times_out_at_750ms_as_out_of_view(self):
        state_filter = owned_filter()
        samples = [weak(ms) for ms in range(375, 1126, 125)]
        decisions = self.run_samples(state_filter, samples)

        self.assertEqual(decisions[0].reliability_state, M3T_SUSPECT)
        self.assertFalse(any(d.reliability_state == M3T_OWNED for d in decisions))
        self.assertEqual(decisions[5].reliability_state, M3T_SUSPECT)  # 625 ms elapsed
        self.assertEqual(decisions[6].reliability_state, LOST)  # 750 ms elapsed
        self.assertEqual(decisions[6].reason, "out_of_view")
        self.assertTrue(decisions[6].stop_m3t)
        self.assertTrue(decisions[6].absence_confirmed)

    def test_unknown_depth_without_yolo_times_out_at_750ms_with_absence_confirmed(self):
        state_filter = owned_filter()
        samples = [unknown(ms) for ms in range(375, 1126, 125)]
        decisions = self.run_samples(state_filter, samples)

        self.assertEqual(decisions[5].reliability_state, M3T_SUSPECT)  # 625 ms elapsed
        self.assertEqual(decisions[6].reliability_state, LOST)  # 750 ms elapsed
        self.assertEqual(decisions[6].reason, "out_of_view")
        self.assertTrue(decisions[6].stop_m3t)
        self.assertTrue(decisions[6].absence_confirmed)

    def test_owned_conflict_goes_to_suspect_then_lost_never_reinit(self):
        state_filter = owned_filter()
        samples = [conflict(ms) for ms in range(375, 1376, 125)]
        decisions = self.run_samples(state_filter, samples)

        self.assertFalse(any(d.request_reinit for d in decisions))
        self.assertEqual(decisions[0].reliability_state, M3T_SUSPECT)
        self.assertEqual(decisions[-2].reliability_state, M3T_SUSPECT)  # 875 ms elapsed
        self.assertEqual(decisions[-1].reliability_state, LOST)  # 1000 ms elapsed
        self.assertEqual(decisions[-1].reason, "support_timeout")
        self.assertTrue(decisions[-1].stop_m3t)
        self.assertTrue(decisions[-1].has_been_owned)
        self.assertFalse(decisions[-1].absence_confirmed)

    def test_visible_target_weak_support_keeps_1000ms_suspect_deadline(self):
        state_filter = owned_filter()
        samples = [weak(ms, yolo_visible=True, yolo_confidence=0.8) for ms in range(375, 1376, 125)]
        decisions = self.run_samples(state_filter, samples)

        self.assertFalse(any(d.publish_trusted_pose for d in decisions))
        self.assertFalse(any(d.request_reinit for d in decisions))
        self.assertEqual(decisions[-2].reliability_state, M3T_SUSPECT)  # 1250 ms: 875 ms elapsed
        self.assertEqual(decisions[-1].reliability_state, LOST)  # 1375 ms: 1000 ms elapsed
        self.assertEqual(decisions[-1].reason, "support_timeout")
        self.assertTrue(decisions[-1].stop_m3t)
        self.assertFalse(decisions[-1].absence_confirmed)

    def test_depth_contradiction_with_absent_yolo_stops_as_out_of_view(self):
        state_filter = owned_filter()
        decisions = self.run_samples(
            state_filter, [depth_contradiction(ms) for ms in range(375, 1126, 125)]
        )
        self.assertEqual(decisions[-1].reliability_state, LOST)
        self.assertEqual(decisions[-1].reason, "out_of_view")
        self.assertTrue(decisions[-1].absence_confirmed)

    def test_appearance_pass_cannot_override_bad_depth(self):
        state_filter = owned_filter()
        decisions = self.run_samples(
            state_filter,
            [depth_contradiction(ms, smoothness_ok=True) for ms in range(375, 1126, 125)],
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

    def test_duplicate_samples_do_not_refresh_the_absence_deadline(self):
        state_filter = owned_filter()
        self.run_samples(state_filter, [unknown(375)])

        # Same pose stamp, but the monotonic clock has advanced to 1125 ms.
        duplicate = unknown(375, now_monotonic_ns=BASE_NS + 1125 * 1_000_000)
        decision = state_filter.update(duplicate)
        self.assertEqual(decision.reliability_state, LOST)
        self.assertEqual(decision.reason, "out_of_view")
        self.assertFalse(decision.observation_is_new)

    def test_good_sample_cannot_cancel_an_expired_deadline(self):
        state_filter = owned_filter()
        self.run_samples(state_filter, [unknown(ms) for ms in (375, 500, 625, 750, 875, 1000)])

        decision = state_filter.update(good(1125, yolo_visible=False))
        self.assertEqual(decision.reliability_state, LOST)
        self.assertEqual(decision.reason, "out_of_view")
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
            "has_been_owned",
            "absence_confirmed",
        ):
            self.assertIn(key, payload)

    def test_startup_yolo_absence_does_not_carry_over_into_owned(self):
        state_filter = self.fresh_filter()
        # 1000 ms of YOLO absence during startup (> 750 ms absence_timeout_ms).
        for ms in range(0, 876, 125):
            state_filter.update(unknown(ms, yolo_visible=False))
        # Confirm ownership with good support while YOLO is still temporarily absent.
        for ms in (1000, 1125, 1250):
            decision = state_filter.update(good(ms, yolo_visible=False))
        self.assertEqual(decision.reliability_state, M3T_OWNED)
        # Absence timer should start fresh from OWNED, not fire immediately at 1375 ms.
        next_decision = state_filter.update(good(1375, yolo_visible=False))
        self.assertEqual(next_decision.reliability_state, M3T_OWNED)

    def test_single_frame_yolo_flicker_on_suspect_timeout_does_not_confirm_absence(self):
        state_filter = owned_filter()
        # Keep YOLO visible up to 875 ms of weak support (suspect starts at 375 ms).
        self.run_samples(state_filter, [weak(ms, yolo_visible=True) for ms in (375, 500, 625, 750, 875, 1000, 1125, 1250)])
        # At 1375 ms (1000 ms suspect timeout), YOLO flickers False for one frame only.
        decision = state_filter.update(weak(1375, yolo_visible=False))
        self.assertEqual(decision.reliability_state, LOST)
        self.assertTrue(decision.has_been_owned)
        self.assertFalse(decision.absence_confirmed)

    def test_unknown_depth_with_visible_or_invalid_yolo_reports_observation_unavailable(self):
        for changes in (
            dict(yolo_visible=True, yolo_observation_valid=True),
            dict(yolo_visible=False, yolo_observation_valid=False),
        ):
            with self.subTest(changes=changes):
                state_filter = owned_filter()
                decisions = self.run_samples(
                    state_filter,
                    [unknown(ms, **changes) for ms in range(375, 1376, 125)],
                )
                self.assertEqual(decisions[-2].reliability_state, M3T_SUSPECT)
                self.assertEqual(decisions[-1].reliability_state, LOST)
                self.assertEqual(decisions[-1].reason, "observation_unavailable")
                self.assertFalse(decisions[-1].absence_confirmed)


if __name__ == "__main__":
    unittest.main()
