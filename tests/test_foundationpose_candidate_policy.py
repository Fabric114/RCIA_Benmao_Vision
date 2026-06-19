import unittest

from scripts.foundationpose_candidate_policy import FoundationPoseCandidatePolicy


class FoundationPoseCandidatePolicyTest(unittest.TestCase):
    def test_default_policy_does_not_publish_direct_initial_pose(self):
        policy = FoundationPoseCandidatePolicy(
            direct_initial_pose_enabled=False,
            request_required=True,
        )

        decision = policy.on_register_result(active_request=None)

        self.assertFalse(decision.publish_initial_pose)
        self.assertFalse(decision.publish_candidate_pose)
        self.assertEqual(decision.event, "foundationpose_waiting_for_init_request")

    def test_active_request_emits_candidate_pose_not_initial_pose(self):
        policy = FoundationPoseCandidatePolicy(
            direct_initial_pose_enabled=False,
            request_required=True,
        )

        decision = policy.on_register_result(
            active_request={"request_id": "req-1", "stamp_ns": 123}
        )

        self.assertFalse(decision.publish_initial_pose)
        self.assertTrue(decision.publish_candidate_pose)
        self.assertEqual(decision.request_id, "req-1")
        self.assertEqual(decision.event, "foundationpose_candidate_pose_published")

    def test_legacy_direct_mode_can_publish_initial_pose(self):
        policy = FoundationPoseCandidatePolicy(
            direct_initial_pose_enabled=True,
            request_required=False,
        )

        decision = policy.on_register_result(active_request=None)

        self.assertTrue(decision.publish_initial_pose)
        self.assertFalse(decision.publish_candidate_pose)


if __name__ == "__main__":
    unittest.main()
