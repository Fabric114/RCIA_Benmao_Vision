from dataclasses import dataclass


@dataclass(frozen=True)
class InitialPoseDecision:
    publish_initial_pose: bool
    event: str
    fresh_count: int


class InitialPosePublishPolicy:
    def __init__(self, mode, max_initial_pose_age_ms, fresh_pose_required_frames):
        if mode not in {"register_immediate", "fresh_supervisor"}:
            raise ValueError("mode must be register_immediate or fresh_supervisor")
        self.mode = mode
        self.max_initial_pose_age_ms = float(max_initial_pose_age_ms)
        self.fresh_pose_required_frames = max(1, int(fresh_pose_required_frames))
        self.fresh_count = 0
        self.initial_pose_published = False

    def reset(self):
        self.fresh_count = 0
        self.initial_pose_published = False

    def on_register_pose(self, age_ms):
        self.fresh_count = 0
        if self.mode == "register_immediate" and not self.initial_pose_published:
            self.initial_pose_published = True
            return InitialPoseDecision(
                publish_initial_pose=True,
                event="foundationpose_initial_pose_published",
                fresh_count=self.fresh_count,
            )
        return InitialPoseDecision(
            publish_initial_pose=False,
            event="foundationpose_register_seed_ready",
            fresh_count=self.fresh_count,
        )

    def on_track_pose(self, age_ms):
        if self.initial_pose_published:
            return InitialPoseDecision(
                publish_initial_pose=False,
                event="foundationpose_fresh_pose_candidate",
                fresh_count=self.fresh_count,
            )
        if float(age_ms) > self.max_initial_pose_age_ms:
            self.fresh_count = 0
            return InitialPoseDecision(
                publish_initial_pose=False,
                event="foundationpose_fresh_pose_rejected",
                fresh_count=self.fresh_count,
            )

        self.fresh_count += 1
        if self.fresh_count >= self.fresh_pose_required_frames:
            self.initial_pose_published = True
            return InitialPoseDecision(
                publish_initial_pose=True,
                event="foundationpose_fresh_initial_pose_published",
                fresh_count=self.fresh_count,
            )
        return InitialPoseDecision(
            publish_initial_pose=False,
            event="foundationpose_fresh_pose_candidate",
            fresh_count=self.fresh_count,
        )
