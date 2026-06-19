import unittest

from scripts.initial_pose_policy import InitialPosePublishPolicy


class InitialPosePublishPolicyTest(unittest.TestCase):
    def test_register_pose_does_not_publish_in_fresh_supervisor_mode(self):
        policy = InitialPosePublishPolicy(
            mode="fresh_supervisor",
            max_initial_pose_age_ms=80.0,
            fresh_pose_required_frames=2,
        )

        decision = policy.on_register_pose(age_ms=4000.0)

        self.assertFalse(decision.publish_initial_pose)
        self.assertEqual(decision.event, "foundationpose_register_seed_ready")

    def test_consecutive_fresh_track_poses_publish_once(self):
        policy = InitialPosePublishPolicy(
            mode="fresh_supervisor",
            max_initial_pose_age_ms=80.0,
            fresh_pose_required_frames=2,
        )

        first = policy.on_track_pose(age_ms=60.0)
        second = policy.on_track_pose(age_ms=55.0)
        third = policy.on_track_pose(age_ms=50.0)

        self.assertFalse(first.publish_initial_pose)
        self.assertTrue(second.publish_initial_pose)
        self.assertFalse(third.publish_initial_pose)
        self.assertEqual(second.event, "foundationpose_fresh_initial_pose_published")

    def test_stale_pose_resets_fresh_counter(self):
        policy = InitialPosePublishPolicy(
            mode="fresh_supervisor",
            max_initial_pose_age_ms=80.0,
            fresh_pose_required_frames=2,
        )

        policy.on_track_pose(age_ms=60.0)
        stale = policy.on_track_pose(age_ms=120.0)
        next_fresh = policy.on_track_pose(age_ms=50.0)

        self.assertEqual(stale.event, "foundationpose_fresh_pose_rejected")
        self.assertFalse(next_fresh.publish_initial_pose)

    def test_register_immediate_mode_publishes_register_pose(self):
        policy = InitialPosePublishPolicy(
            mode="register_immediate",
            max_initial_pose_age_ms=80.0,
            fresh_pose_required_frames=2,
        )

        decision = policy.on_register_pose(age_ms=4000.0)

        self.assertTrue(decision.publish_initial_pose)

    def test_reset_allows_fresh_supervisor_policy_to_publish_again(self):
        policy = InitialPosePublishPolicy(
            mode="fresh_supervisor",
            max_initial_pose_age_ms=80.0,
            fresh_pose_required_frames=1,
        )
        first = policy.on_track_pose(age_ms=20.0)

        policy.reset()
        second = policy.on_track_pose(age_ms=20.0)

        self.assertTrue(first.publish_initial_pose)
        self.assertTrue(second.publish_initial_pose)


if __name__ == "__main__":
    unittest.main()
