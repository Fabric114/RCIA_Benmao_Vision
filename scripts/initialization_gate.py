from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class InitializationGateParams:
    max_candidate_age_ms: float = 10000.0
    enforce_freshness: bool = False


@dataclass(frozen=True)
class InitializationGateResult:
    success: bool
    pose_valid: bool
    pose_age_ms: float
    reason: str = ""
    depth_ratio: float = 0.0
    projection_iou: float = 0.0
    black_white_success: bool = True
    rendered_pixel_count: int = 0

    def to_dict(self):
        data = asdict(self)
        data["depth_consistency_ratio"] = self.depth_ratio
        return data


def validate_initialization_candidate(
    candidate_pose,
    now_ns,
    candidate_stamp_ns,
    candidate_status=None,
    params=None,
):
    params = params or InitializationGateParams()
    pose = np.asarray(candidate_pose, dtype=np.float64)
    if pose.shape != (4, 4):
        return _result(False, 0.0, "pose matrix must be 4x4")
    if not np.all(np.isfinite(pose)):
        return _result(False, 0.0, "pose matrix must contain only finite values")
    if not np.allclose(pose[3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        return _result(False, 0.0, "pose matrix must have homogeneous last row")

    pose_age_ms = _candidate_age_ms(now_ns, candidate_stamp_ns, candidate_status)
    if pose_age_ms > float(params.max_candidate_age_ms):
        reason = (
            f"stale candidate pose: {pose_age_ms:.1f}ms > "
            f"{float(params.max_candidate_age_ms):.1f}ms"
        )
        if not bool(params.enforce_freshness):
            return _result(True, pose_age_ms, reason)
        return _result(
            False,
            pose_age_ms,
            reason,
        )
    return _result(True, pose_age_ms, "")


def _candidate_age_ms(now_ns, candidate_stamp_ns, candidate_status):
    if isinstance(candidate_status, dict) and candidate_status.get("pose_age_ms") is not None:
        try:
            return float(candidate_status.get("pose_age_ms"))
        except (TypeError, ValueError):
            pass
    now = int(now_ns or 0)
    stamp = int(candidate_stamp_ns or 0)
    if now <= 0 or stamp <= 0:
        return 0.0
    return max(0.0, (now - stamp) / 1_000_000.0)


def _result(success, pose_age_ms, reason):
    return InitializationGateResult(
        success=bool(success),
        pose_valid=bool(success),
        pose_age_ms=float(pose_age_ms),
        reason=str(reason),
    )
