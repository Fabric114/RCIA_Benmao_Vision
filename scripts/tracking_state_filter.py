from dataclasses import asdict, dataclass


NO_TRACK = "NO_TRACK"
M3T_STARTING = "M3T_STARTING"
M3T_OWNED = "M3T_OWNED"
M3T_SUSPECT = "M3T_SUSPECT"
REINIT_NEEDED = "REINIT_NEEDED"
LOST = "LOST"

# Backward-compatible names for older diagnostics/tests.
LOCKED = M3T_OWNED
SUSPECT = M3T_SUSPECT
DEGRADED = REINIT_NEEDED


@dataclass(frozen=True)
class TrackingStateFilterParams:
    initial_presence_score: float = 0.80
    initial_health_score: float = 0.65
    startup_grace_frames: int = 40
    owned_confirm_frames: int = 3
    recover_frames: int = 2
    suspect_frames: int = 6
    reinit_conflict_frames: int = 5
    bad_tracking_frames: int = 5
    out_of_view_frames: int = 12
    min_rendered_pixels_for_owned: int = 40
    max_fresh_pose_age_ms: float = 250.0

    # Compatibility fields accepted by existing launch files. The ownership
    # filter no longer uses these thresholds directly.
    occlusion_grace_frames: int = 24
    locked_health_threshold: float = 0.70
    suspect_health_threshold: float = 0.40
    degraded_health_threshold: float = 0.40
    degraded_presence_threshold: float = 0.55
    lost_health_threshold: float = 0.35
    lost_presence_threshold: float = 0.35
    min_rendered_pixels: int = 80


@dataclass(frozen=True)
class TrackingStateObservation:
    yolo_visible: bool = False
    yolo_confidence: float = 0.0
    m3t_status: str = ""
    m3t_pose_available: bool = False
    m3t_pose_age_ms: float = 0.0
    depth_ok: bool = False
    depth_ratio: float = 0.0
    black_white_ok: bool = False
    projection_ok: bool = True
    projection_center_ok: bool = True
    projection_area_ok: bool = True
    smoothness_ok: bool = True
    rendered_pixel_count: int = 0


@dataclass(frozen=True)
class TrackingStateDecision:
    reliability_state: str
    publish_trusted_pose: bool
    keep_m3t_alive: bool
    request_reinit: bool
    stop_m3t: bool
    target_presence_score: float
    tracker_health_score: float
    reason: str
    owned_confirm_streak: int = 0
    m3t_good_streak: int = 0
    m3t_bad_streak: int = 0
    external_conflict_streak: int = 0
    out_of_view_streak: int = 0
    weak_validation_fail_streak: int = 0
    startup_good_total: int = 0

    def to_dict(self):
        return asdict(self)


class TrackingStateFilter:
    def __init__(self, params=None):
        self.params = params or TrackingStateFilterParams()
        self.state = M3T_STARTING
        self.target_presence_score = float(self.params.initial_presence_score)
        self.tracker_health_score = float(self.params.initial_health_score)
        self.owned_confirm_streak = 0
        self.m3t_good_streak = 0
        self.m3t_bad_streak = 0
        self.external_conflict_streak = 0
        self.out_of_view_streak = 0
        self.weak_validation_fail_streak = 0
        self.startup_good_total = 0
        self._startup_frames_seen = 0

    def reset_for_tracking(
        self,
        target_presence_score=None,
        tracker_health_score=None,
    ):
        self.state = M3T_STARTING
        self.target_presence_score = self._clamp(
            self.params.initial_presence_score
            if target_presence_score is None
            else target_presence_score
        )
        self.tracker_health_score = self._clamp(
            self.params.initial_health_score
            if tracker_health_score is None
            else tracker_health_score
        )
        self.owned_confirm_streak = 0
        self.m3t_good_streak = 0
        self.m3t_bad_streak = 0
        self.external_conflict_streak = 0
        self.out_of_view_streak = 0
        self.weak_validation_fail_streak = 0
        self.startup_good_total = 0
        self._startup_frames_seen = 0

    def reset_to_search(self):
        self.state = LOST
        self.target_presence_score = 0.0
        self.tracker_health_score = 0.0
        self.owned_confirm_streak = 0
        self.m3t_good_streak = 0
        self.m3t_bad_streak = 0
        self.external_conflict_streak = 0
        self.out_of_view_streak = 0
        self.weak_validation_fail_streak = 0
        self.startup_good_total = 0
        self._startup_frames_seen = 0

    def update(self, observation):
        self._update_streaks(observation)
        self._update_scores(observation)

        if self.state == M3T_STARTING:
            return self._update_starting(observation)
        if self.state == M3T_OWNED:
            return self._update_owned(observation)
        if self.state == M3T_SUSPECT:
            return self._update_suspect(observation)
        if self.state in {REINIT_NEEDED, LOST}:
            return self._decision_for_state(self.state, f"state held: {self.state}")

        self.state = M3T_STARTING
        return self._decision_for_state(self.state, "reset unknown filter state")

    def _update_starting(self, observation):
        self._startup_frames_seen += 1
        if (
            self.owned_confirm_streak >= int(self.params.owned_confirm_frames)
            or self.startup_good_total >= int(self.params.owned_confirm_frames)
            or self._strong_startup_m3t_evidence(observation)
        ):
            self.state = M3T_OWNED
            return self._decision_for_state(self.state, "m3t ownership confirmed")

        if self.external_conflict_streak >= int(self.params.reinit_conflict_frames):
            self.state = REINIT_NEEDED
            return self._decision_for_state(self.state, "startup external conflict")

        if self._startup_frames_seen > int(self.params.startup_grace_frames):
            self.state = LOST
            return self._decision_for_state(self.state, "startup grace expired")

        return self._decision_for_state(self.state, "waiting for m3t ownership")

    def _update_owned(self, observation):
        if self.out_of_view_streak >= int(self.params.out_of_view_frames):
            self.state = LOST
            return self._decision_for_state(self.state, "target out of view")

        if self.external_conflict_streak >= int(self.params.reinit_conflict_frames):
            self.state = REINIT_NEEDED
            return self._decision_for_state(self.state, "sustained external conflict")

        if (
            self.m3t_bad_streak >= int(self.params.bad_tracking_frames)
            and self._target_likely_present_for_reinit(observation)
        ):
            self.state = REINIT_NEEDED
            return self._decision_for_state(self.state, "sustained bad m3t tracking")

        if (
            self.weak_validation_fail_streak >= int(self.params.suspect_frames)
            and self._m3t_good(observation)
        ):
            self.state = M3T_SUSPECT
            return self._decision_for_state(self.state, "weak validation failures")

        return self._decision_for_state(self.state, "m3t-owned")

    def _update_suspect(self, observation):
        if self.out_of_view_streak >= int(self.params.out_of_view_frames):
            self.state = LOST
            return self._decision_for_state(self.state, "target out of view")

        if self.external_conflict_streak >= int(self.params.reinit_conflict_frames):
            self.state = REINIT_NEEDED
            return self._decision_for_state(self.state, "sustained external conflict")

        if (
            self.m3t_bad_streak >= int(self.params.bad_tracking_frames)
            and self._target_likely_present_for_reinit(observation)
        ):
            self.state = REINIT_NEEDED
            return self._decision_for_state(self.state, "sustained bad m3t tracking")

        if self.m3t_good_streak >= int(self.params.recover_frames):
            self.state = M3T_OWNED
            return self._decision_for_state(self.state, "m3t recovered")

        return self._decision_for_state(self.state, "m3t suspect")

    def _update_streaks(self, observation):
        m3t_good = self._m3t_good(observation)
        external_conflict = self._external_conflict(observation)
        out_of_view = self._out_of_view_evidence(observation)
        weak_failure = self._weak_validation_failed(observation)

        self.m3t_good_streak = self._count_if(m3t_good, self.m3t_good_streak)
        self.m3t_bad_streak = self._count_if(not m3t_good, self.m3t_bad_streak)
        self.external_conflict_streak = self._count_if(
            external_conflict,
            self.external_conflict_streak,
        )
        self.out_of_view_streak = self._count_if(out_of_view, self.out_of_view_streak)
        self.weak_validation_fail_streak = self._count_if(
            weak_failure,
            self.weak_validation_fail_streak,
        )
        if self.state == M3T_STARTING:
            if m3t_good and not external_conflict:
                self.startup_good_total += 1
            self.owned_confirm_streak = self._count_if(
                m3t_good and not external_conflict,
                self.owned_confirm_streak,
            )
        elif self.state == M3T_OWNED:
            self.owned_confirm_streak = max(self.owned_confirm_streak, self.m3t_good_streak)

    def _m3t_good(self, observation):
        if not self._m3t_pose_fresh_and_tracking(observation):
            return False
        if self._hard_visual_contradiction(observation):
            return False
        return self._m3t_motion_good(observation) or self._strong_visual_support(
            observation
        )

    def _m3t_pose_fresh_and_tracking(self, observation):
        return (
            observation.m3t_status == "tracking"
            and observation.m3t_pose_available
            and float(observation.m3t_pose_age_ms)
            <= float(self.params.max_fresh_pose_age_ms)
        )

    @staticmethod
    def _m3t_motion_good(observation):
        return bool(observation.smoothness_ok)

    def _m3t_rendered_visible(self, observation):
        return int(observation.rendered_pixel_count) >= int(
            self.params.min_rendered_pixels_for_owned
        )

    def _strong_visual_support(self, observation):
        if not self._m3t_rendered_visible(observation):
            return False
        return self._depth_support(observation)

    @staticmethod
    def _depth_support(observation):
        return (
            bool(observation.depth_ok)
            or float(observation.depth_ratio) >= 0.50
        )

    def _hard_visual_contradiction(self, observation):
        return (
            self._m3t_rendered_visible(observation)
            and not self._depth_support(observation)
            and float(observation.depth_ratio) <= 0.05
            and not bool(observation.black_white_ok)
        )

    def _strong_startup_m3t_evidence(self, observation):
        if not self._m3t_pose_fresh_and_tracking(observation):
            return False
        if self._external_conflict(observation):
            return False
        return (
            self._m3t_rendered_visible(observation)
            and float(observation.depth_ratio) >= 0.70
            and observation.projection_ok
            and observation.projection_center_ok
            and observation.projection_area_ok
        )

    def _weak_validation_failed(self, observation):
        if not self._m3t_good(observation):
            return False
        failures = 0
        if not observation.yolo_visible:
            failures += 1
        if not observation.depth_ok or float(observation.depth_ratio) <= 0.10:
            failures += 1
        if not observation.black_white_ok:
            failures += 1
        if not self._m3t_rendered_visible(observation):
            failures += 1
        return failures >= 2

    def _external_conflict(self, observation):
        if self._out_of_view_evidence(observation):
            return False
        if not self._m3t_pose_fresh_and_tracking(observation):
            return False
        if self._hard_visual_contradiction(observation):
            return True
        projection_failed = not (
            observation.projection_ok
            and observation.projection_center_ok
            and observation.projection_area_ok
        )
        if observation.yolo_visible and projection_failed:
            return True
        external_failures = 0
        if not observation.depth_ok or float(observation.depth_ratio) <= 0.05:
            external_failures += 1
        if not observation.black_white_ok:
            external_failures += 1
        if projection_failed:
            external_failures += 1
        if not self._m3t_rendered_visible(observation):
            external_failures += 1
        return external_failures >= 3 and not observation.smoothness_ok

    def _out_of_view_evidence(self, observation):
        if not self._m3t_pose_fresh_and_tracking(observation):
            return not observation.yolo_visible
        return (
            not observation.yolo_visible
            and not self._m3t_rendered_visible(observation)
            and float(observation.depth_ratio) <= 0.02
            and not observation.black_white_ok
            and not observation.smoothness_ok
        )

    def _target_likely_present_for_reinit(self, observation):
        return (
            observation.yolo_visible
            or self._m3t_rendered_visible(observation)
            or float(observation.depth_ratio) > 0.10
            or bool(observation.black_white_ok)
        )

    def _update_scores(self, observation):
        if self._out_of_view_evidence(observation):
            self.target_presence_score = self._clamp(self.target_presence_score - 0.12)
        elif observation.yolo_visible or self._m3t_rendered_visible(observation):
            self.target_presence_score = self._clamp(self.target_presence_score + 0.04)

        if self._m3t_good(observation):
            self.tracker_health_score = self._clamp(self.tracker_health_score + 0.08)
        elif self._external_conflict(observation) or self._out_of_view_evidence(observation):
            self.tracker_health_score = self._clamp(self.tracker_health_score - 0.12)
        else:
            self.tracker_health_score = self._clamp(self.tracker_health_score - 0.03)

    def _decision_for_state(self, state, reason):
        publish_trusted_pose = state == M3T_OWNED
        keep_m3t_alive = state in {M3T_STARTING, M3T_OWNED, M3T_SUSPECT}
        request_reinit = state == REINIT_NEEDED
        stop_m3t = state == LOST
        return TrackingStateDecision(
            reliability_state=state,
            publish_trusted_pose=publish_trusted_pose,
            keep_m3t_alive=keep_m3t_alive,
            request_reinit=request_reinit,
            stop_m3t=stop_m3t,
            target_presence_score=float(self.target_presence_score),
            tracker_health_score=float(self.tracker_health_score),
            reason=reason,
            owned_confirm_streak=int(self.owned_confirm_streak),
            m3t_good_streak=int(self.m3t_good_streak),
            m3t_bad_streak=int(self.m3t_bad_streak),
            external_conflict_streak=int(self.external_conflict_streak),
            out_of_view_streak=int(self.out_of_view_streak),
            weak_validation_fail_streak=int(self.weak_validation_fail_streak),
            startup_good_total=int(self.startup_good_total),
        )

    @staticmethod
    def _count_if(condition, current):
        return current + 1 if condition else 0

    @staticmethod
    def _clamp(value):
        return max(0.0, min(1.0, float(value)))
