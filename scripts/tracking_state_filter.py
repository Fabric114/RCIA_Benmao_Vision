import math
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

# Per-sample evidence classes, reported as TrackingStateDecision.evidence_kind.
EVIDENCE_SUPPORT = "support"
EVIDENCE_ABSENCE = "absence"
EVIDENCE_CONFLICT = "conflict"
EVIDENCE_WEAK = "weak"
EVIDENCE_UNKNOWN = "unknown"

_NS_PER_MS = 1_000_000


@dataclass(frozen=True)
class TrackingStateFilterParams:
    initial_presence_score: float = 0.80
    initial_health_score: float = 0.65
    owned_confirm_frames: int = 3
    recover_frames: int = 2
    max_fresh_pose_age_ms: float = 250.0
    min_rendered_pixels: int = 80
    min_rendered_pixels_for_owned: int = 40
    min_valid_depth_overlap_pixels: int = 80
    min_depth_support_ratio: float = 0.60
    hard_depth_mismatch_ratio: float = 0.05
    absence_timeout_ms: float = 1500.0
    conflict_timeout_ms: float = 625.0
    suspect_timeout_ms: float = 2000.0
    no_target_support_timeout_ms: float = 500.0
    startup_timeout_ms: float = 5000.0

    # Deprecated frame-count fields. They are accepted so existing launch files
    # still load, but no deadline uses them any more.
    startup_grace_frames: int = 40
    occlusion_grace_frames: int = 24
    suspect_frames: int = 6
    reinit_conflict_frames: int = 5
    bad_tracking_frames: int = 5
    out_of_view_frames: int = 12
    locked_health_threshold: float = 0.70
    suspect_health_threshold: float = 0.40
    degraded_health_threshold: float = 0.40
    degraded_presence_threshold: float = 0.55
    lost_health_threshold: float = 0.35
    lost_presence_threshold: float = 0.35


@dataclass(frozen=True)
class TrackingStateObservation:
    yolo_visible: bool = False
    yolo_confidence: float = 0.0
    yolo_observation_valid: bool = False
    m3t_status: str = ""
    m3t_pose_available: bool = False
    m3t_pose_age_ms: float = 0.0
    health_evaluated: bool = False
    valid_depth_overlap_count: int = 0
    depth_evaluable: bool = False
    depth_ok: bool = False
    depth_ratio: float = 0.0
    black_white_ok: bool = False
    projection_evaluated: bool = False
    projection_ok: bool = True
    projection_center_ok: bool = True
    projection_area_ok: bool = True
    smoothness_ok: bool = True
    rendered_pixel_count: int = 0
    pose_stamp_ns: int = 0
    observation_stamp_ns: int = 0
    now_monotonic_ns: int = 0


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
    recovery_streak: int = 0
    suspect_elapsed_ms: float = 0.0
    evidence_kind: str = EVIDENCE_UNKNOWN
    observation_is_new: bool = False

    def to_dict(self):
        return asdict(self)


class TrackingStateFilter:
    def __init__(self, params=None):
        self.params = params or TrackingStateFilterParams()
        self.state = M3T_STARTING
        self.target_presence_score = float(self.params.initial_presence_score)
        self.tracker_health_score = float(self.params.initial_health_score)
        self._clear_counters_and_timers()

    def reset_for_tracking(
        self,
        target_presence_score=None,
        tracker_health_score=None,
        *,
        now_monotonic_ns=None,
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
        self._clear_counters_and_timers()
        if now_monotonic_ns is not None:
            self._startup_start_ns = int(now_monotonic_ns)

    def reset_to_search(self):
        self.state = LOST
        self.target_presence_score = 0.0
        self.tracker_health_score = 0.0
        self._clear_counters_and_timers()

    def _clear_counters_and_timers(self):
        self.owned_confirm_streak = 0
        self.m3t_good_streak = 0
        self.m3t_bad_streak = 0
        self.external_conflict_streak = 0
        self.out_of_view_streak = 0
        self.weak_validation_fail_streak = 0
        self.startup_good_total = 0
        self._last_pose_stamp_ns = 0
        self._last_now_ns = None
        self._startup_start_ns = None
        self._suspect_since_ns = None
        self._conflict_since_ns = None
        self._absence_since_ns = None
        self._no_target_since_ns = None
        self._no_target_reason = None

    def update(self, observation):
        now = int(observation.now_monotonic_ns)
        if self._startup_start_ns is None:
            self._startup_start_ns = now
        kind = self._classify(observation)
        is_new = self._is_new_sample(observation, now)

        if self.state in {REINIT_NEEDED, LOST}:
            return self._decision(self.state, f"state held: {self.state}", kind, False, now)

        # A deadline that expired before this sample wins over it, so a late
        # good sample cannot cancel a timeout that already ran out.
        expired = self._expired_deadline(observation, now)
        self._apply_sample(observation, now, is_new, kind)
        if expired is None:
            expired = self._expired_deadline(observation, now)
        if expired is not None:
            self.state, reason = expired
            return self._decision(self.state, reason, kind, is_new, now)
        return self._advance(observation, now, is_new, kind)

    def _advance(self, observation, now, is_new, kind):
        if self.state == M3T_STARTING:
            if self.m3t_good_streak >= int(self.params.owned_confirm_frames):
                self.state = M3T_OWNED
                self._suspect_since_ns = None
                return self._decision(self.state, "m3t ownership confirmed", kind, is_new, now)
            return self._decision(self.state, "waiting for m3t ownership", kind, is_new, now)

        if self.state == M3T_OWNED:
            lost_support = not self._fresh_tracking(observation) or (
                is_new and kind != EVIDENCE_SUPPORT
            )
            if lost_support:
                self._enter_suspect(now)
                return self._decision(self.state, "positive support lost", kind, is_new, now)
            return self._decision(self.state, "m3t-owned", kind, is_new, now)

        if self.state == M3T_SUSPECT:
            if self.m3t_good_streak >= int(self.params.recover_frames):
                self.state = M3T_OWNED
                self._suspect_since_ns = None
                return self._decision(self.state, "m3t recovered", kind, is_new, now)
            return self._decision(self.state, "m3t suspect", kind, is_new, now)

        return self._decision(self.state, f"no transition from {self.state}", kind, is_new, now)

    def _enter_suspect(self, now):
        self.state = M3T_SUSPECT
        # Recovery counts only new samples received after entering SUSPECT.
        self.m3t_good_streak = 0
        self.owned_confirm_streak = 0
        if self._suspect_since_ns is None:
            self._suspect_since_ns = now

    def _expired_deadline(self, observation, now):
        p = self.params
        if self.state == M3T_STARTING:
            if self._expired(self._startup_start_ns, now, p.startup_timeout_ms):
                return LOST, "startup_timeout"
            if self._expired(self._conflict_since_ns, now, p.conflict_timeout_ms):
                return REINIT_NEEDED, "external_conflict"
            return None
        if self.state != M3T_SUSPECT:
            return None
        if self._expired(self._absence_since_ns, now, p.absence_timeout_ms):
            return LOST, "out_of_view"
        if self._expired(self._no_target_since_ns, now, p.no_target_support_timeout_ms):
            return LOST, self._no_target_reason
        if self._expired(self._conflict_since_ns, now, p.conflict_timeout_ms):
            return REINIT_NEEDED, "external_conflict"
        if self._expired(self._suspect_since_ns, now, p.suspect_timeout_ms):
            if observation.yolo_observation_valid and observation.yolo_visible:
                return REINIT_NEEDED, "support_timeout"
            return LOST, "support_timeout"
        return None

    def _apply_sample(self, observation, now, is_new, kind):
        # Duplicate and backward samples change no counter or timer.
        if not is_new:
            return
        self._last_pose_stamp_ns = int(observation.pose_stamp_ns)
        self._last_now_ns = now

        positive = kind == EVIDENCE_SUPPORT
        self.m3t_good_streak = self._count_if(positive, self.m3t_good_streak)
        self.owned_confirm_streak = self.m3t_good_streak
        self.m3t_bad_streak = self._count_if(not positive, self.m3t_bad_streak)
        self.external_conflict_streak = self._count_if(
            kind == EVIDENCE_CONFLICT, self.external_conflict_streak
        )
        self.out_of_view_streak = self._count_if(
            kind == EVIDENCE_ABSENCE, self.out_of_view_streak
        )
        self.weak_validation_fail_streak = self._count_if(
            kind in (EVIDENCE_WEAK, EVIDENCE_UNKNOWN),
            self.weak_validation_fail_streak,
        )
        if positive and self.state == M3T_STARTING:
            self.startup_good_total += 1

        self._conflict_since_ns = self._since(
            kind == EVIDENCE_CONFLICT, self._conflict_since_ns, now
        )
        self._absence_since_ns = self._since(
            kind == EVIDENCE_ABSENCE, self._absence_since_ns, now
        )
        # With no usable YOLO target, weak or unknown evidence gets a short
        # deadline. A visible target uses the suspect deadline instead.
        no_target = kind in (EVIDENCE_WEAK, EVIDENCE_UNKNOWN) and not (
            observation.yolo_observation_valid and observation.yolo_visible
        )
        if no_target and self._no_target_since_ns is None:
            self._no_target_reason = (
                "observation_unavailable" if kind == EVIDENCE_UNKNOWN else "support_timeout"
            )
        self._no_target_since_ns = self._since(no_target, self._no_target_since_ns, now)
        self._update_scores(observation, kind)

    @staticmethod
    def _since(condition, since_ns, now_ns):
        if not condition:
            return None
        return since_ns if since_ns is not None else now_ns

    @staticmethod
    def _expired(since_ns, now_ns, timeout_ms):
        return since_ns is not None and now_ns - since_ns >= float(timeout_ms) * _NS_PER_MS

    def _is_new_sample(self, observation, now_ns):
        pose_stamp = int(observation.pose_stamp_ns)
        if pose_stamp <= 0 or pose_stamp <= self._last_pose_stamp_ns:
            return False
        if self._last_now_ns is not None and now_ns < self._last_now_ns:
            return False
        return True

    def _classify(self, observation):
        if not (self._fresh_tracking(observation) and observation.health_evaluated):
            return EVIDENCE_UNKNOWN
        if self._has_positive_support(observation):
            return EVIDENCE_SUPPORT
        if self._has_absence_evidence(observation):
            return EVIDENCE_ABSENCE
        if self._has_conflict_evidence(observation):
            return EVIDENCE_CONFLICT
        if not self._depth_evaluable(observation):
            return EVIDENCE_UNKNOWN
        return EVIDENCE_WEAK

    def _has_positive_support(self, observation):
        # Appearance, smoothness and YOLO are diagnostics only. Support needs a
        # fresh pose with enough rendered and depth-overlap pixels that agree.
        return bool(
            self._fresh_tracking(observation)
            and observation.health_evaluated
            and observation.projection_evaluated
            and observation.projection_ok
            and observation.projection_center_ok
            and observation.projection_area_ok
            and int(observation.rendered_pixel_count) >= int(self.params.min_rendered_pixels)
            and self._depth_evaluable(observation)
            and self._finite(observation.depth_ratio)
            and float(observation.depth_ratio) >= float(self.params.min_depth_support_ratio)
        )

    def _has_absence_evidence(self, observation):
        return bool(
            observation.yolo_observation_valid
            and not observation.yolo_visible
            and observation.projection_evaluated
            and int(observation.rendered_pixel_count)
            < int(self.params.min_rendered_pixels_for_owned)
        )

    def _has_conflict_evidence(self, observation):
        if not (observation.yolo_observation_valid and observation.yolo_visible):
            return False
        return self._projection_conflict(observation) or self._hard_depth_mismatch(observation)

    def _projection_conflict(self, observation):
        return bool(
            observation.projection_evaluated
            and not (
                observation.projection_ok
                and observation.projection_center_ok
                and observation.projection_area_ok
            )
        )

    def _hard_depth_mismatch(self, observation):
        return bool(
            self._depth_evaluable(observation)
            and self._finite(observation.depth_ratio)
            and float(observation.depth_ratio) <= float(self.params.hard_depth_mismatch_ratio)
        )

    def _depth_evaluable(self, observation):
        return bool(observation.depth_evaluable) and int(
            observation.valid_depth_overlap_count
        ) >= int(self.params.min_valid_depth_overlap_pixels)

    def _fresh_tracking(self, observation):
        return bool(
            observation.m3t_status == "tracking"
            and observation.m3t_pose_available
            and self._finite(observation.m3t_pose_age_ms)
            and 0.0 <= float(observation.m3t_pose_age_ms) <= float(self.params.max_fresh_pose_age_ms)
        )

    def _update_scores(self, observation, kind):
        yolo_target = bool(observation.yolo_observation_valid and observation.yolo_visible)
        if kind == EVIDENCE_ABSENCE:
            self.target_presence_score = self._clamp(self.target_presence_score - 0.12)
        elif yolo_target or kind == EVIDENCE_SUPPORT:
            self.target_presence_score = self._clamp(self.target_presence_score + 0.04)

        if kind == EVIDENCE_SUPPORT:
            self.tracker_health_score = self._clamp(self.tracker_health_score + 0.08)
        elif kind in (EVIDENCE_CONFLICT, EVIDENCE_ABSENCE):
            self.tracker_health_score = self._clamp(self.tracker_health_score - 0.12)
        else:
            self.tracker_health_score = self._clamp(self.tracker_health_score - 0.03)

    def _decision(self, state, reason, kind, is_new, now_ns):
        suspect_elapsed_ms = 0.0
        if self._suspect_since_ns is not None:
            suspect_elapsed_ms = max(0.0, (now_ns - self._suspect_since_ns) / _NS_PER_MS)
        return TrackingStateDecision(
            reliability_state=state,
            publish_trusted_pose=state == M3T_OWNED,
            keep_m3t_alive=state in {M3T_STARTING, M3T_OWNED, M3T_SUSPECT},
            request_reinit=state == REINIT_NEEDED,
            stop_m3t=state == LOST,
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
            recovery_streak=int(self.m3t_good_streak),
            suspect_elapsed_ms=float(suspect_elapsed_ms),
            evidence_kind=kind,
            observation_is_new=bool(is_new),
        )

    @staticmethod
    def _finite(value):
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _count_if(condition, current):
        return current + 1 if condition else 0

    @staticmethod
    def _clamp(value):
        return max(0.0, min(1.0, float(value)))
