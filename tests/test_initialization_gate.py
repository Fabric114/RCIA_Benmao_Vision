import unittest

import numpy as np

from scripts.initialization_gate import (
    InitializationGateParams,
    validate_initialization_candidate,
)


class InitializationGateTest(unittest.TestCase):
    def setUp(self):
        self.pose = np.eye(4, dtype=np.float64)
        self.pose[:3, 3] = [0.0, 0.0, 0.35]
        self.params = InitializationGateParams(max_candidate_age_ms=500.0)

    def test_fresh_finite_candidate_passes_without_depth_or_appearance(self):
        result = validate_initialization_candidate(
            candidate_pose=self.pose,
            now_ns=1_300_000_000,
            candidate_stamp_ns=1_000_000_000,
            candidate_status={"pose_age_ms": 250.0},
            params=self.params,
        )

        self.assertTrue(result.success, result.reason)
        self.assertTrue(result.pose_valid)
        self.assertEqual(result.pose_age_ms, 250.0)

    def test_stale_candidate_passes_by_default_as_diagnostic_warning(self):
        result = validate_initialization_candidate(
            candidate_pose=self.pose,
            now_ns=1_800_000_000,
            candidate_stamp_ns=1_000_000_000,
            candidate_status={},
            params=self.params,
        )

        self.assertTrue(result.success, result.reason)
        self.assertIn("stale", result.reason)

    def test_stale_candidate_fails_when_freshness_is_enforced(self):
        result = validate_initialization_candidate(
            candidate_pose=self.pose,
            now_ns=1_800_000_000,
            candidate_stamp_ns=1_000_000_000,
            candidate_status={},
            params=InitializationGateParams(
                max_candidate_age_ms=500.0,
                enforce_freshness=True,
            ),
        )

        self.assertFalse(result.success)
        self.assertIn("stale", result.reason)

    def test_non_finite_pose_fails(self):
        pose = self.pose.copy()
        pose[0, 3] = np.nan

        result = validate_initialization_candidate(
            candidate_pose=pose,
            now_ns=1_100_000_000,
            candidate_stamp_ns=1_000_000_000,
            candidate_status={"pose_age_ms": 100.0},
            params=self.params,
        )

        self.assertFalse(result.success)
        self.assertIn("finite", result.reason)

    def test_invalid_homogeneous_row_fails(self):
        pose = self.pose.copy()
        pose[3, :] = [0.0, 0.0, 0.0, 2.0]

        result = validate_initialization_candidate(
            candidate_pose=pose,
            now_ns=1_100_000_000,
            candidate_stamp_ns=1_000_000_000,
            candidate_status={"pose_age_ms": 100.0},
            params=self.params,
        )

        self.assertFalse(result.success)
        self.assertIn("homogeneous", result.reason)


if __name__ == "__main__":
    unittest.main()
