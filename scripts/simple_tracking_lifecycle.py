from dataclasses import dataclass


NO_TRACK = "NO_TRACK"
TRACKING = "TRACKING"
REINIT = "REINIT"


@dataclass(frozen=True)
class LifecycleParams:
    stable_yolo_frames: int = 3
    lost_frames: int = 5
    out_of_view_frames: int = 5
    candidate_reject_cooldown_frames: int = 5


@dataclass(frozen=True)
class LifecycleObservation:
    yolo_visible: bool = False
    candidate_available: bool = False
    candidate_valid: bool = False
    candidate_timed_out: bool = False
    m3t_status: str = ""
    m3t_pose_available: bool = False
    tracking_health_ok: bool = False
    tracking_reliability_state: str = ""
    filter_publish_trusted_pose: bool = False
    filter_keep_m3t_alive: bool = False
    filter_request_reinit: bool = False
    filter_stop_m3t: bool = False
    now_monotonic_ns: int = 0
    m3t_startup_pending: bool = False
    m3t_startup_failed: bool = False
    m3t_stop_pending: bool = False
    m3t_stop_ack_timed_out: bool = False


@dataclass(frozen=True)
class LifecycleDecision:
    state: str
    action: str
    publish_init_request: bool = False
    publish_initial_pose: bool = False
    publish_stop: bool = False
    publish_trusted_pose: bool = False
    clear_candidate: bool = False
    reason: str = ""


class SimpleTrackingLifecycle:
    def __init__(self, params=None):
        self.params = params or LifecycleParams()
        self.state = NO_TRACK
        self._stable_yolo_count = 0
        self._candidate_in_flight = False
        self._bad_tracking_count = 0
        self._out_of_view_count = 0
        self._candidate_reject_cooldown = 0

    def update(self, observation):
        # Old M3T output may still be running, so nothing new starts until the
        # stop is acknowledged. Candidates from before the stop are dropped.
        if observation.m3t_stop_pending:
            return self._wait_for_m3t_stop(observation)
        if self.state == NO_TRACK:
            return self._update_no_track(observation)
        if self.state == TRACKING:
            return self._update_tracking(observation)
        if self.state == REINIT:
            return self._update_reinit(observation)
        self.state = NO_TRACK
        self._reset_no_track_counts()
        return self._decision("reset_unknown_state", reason="unknown state")

    def _wait_for_m3t_stop(self, observation):
        # Stable-YOLO counting restarts after the stop is acknowledged, so a
        # target seen during the wait must stay visible for a full stable run.
        self._candidate_in_flight = False
        self._stable_yolo_count = 0
        if observation.m3t_stop_ack_timed_out:
            return self._decision(
                "stop_ack_timeout",
                clear_candidate=True,
                reason="stop ack timeout: no new initialization until m3t reports stopped",
            )
        return self._decision(
            "wait_for_m3t_stop",
            clear_candidate=True,
            reason="waiting for m3t stop ack",
        )

    def _update_no_track(self, observation):
        if observation.candidate_available:
            self._candidate_in_flight = False
            if observation.candidate_valid:
                self.state = TRACKING
                self._reset_tracking_counts()
                return self._decision(
                    "start_m3t",
                    publish_initial_pose=True,
                    clear_candidate=True,
                    reason="candidate accepted",
                )
            self._stable_yolo_count = 0
            self._candidate_reject_cooldown = max(
                0,
                int(self.params.candidate_reject_cooldown_frames) - 1,
            )
            return self._decision(
                "reject_candidate",
                clear_candidate=True,
                reason="candidate rejected",
            )

        if observation.candidate_timed_out:
            self._candidate_in_flight = False
            self._stable_yolo_count = 0
            self._candidate_reject_cooldown = max(
                0,
                int(self.params.candidate_reject_cooldown_frames) - 1,
            )
            return self._decision(
                "clear_timed_out_candidate",
                clear_candidate=True,
                reason="candidate timed out",
            )

        if self._candidate_reject_cooldown > 0:
            self._candidate_reject_cooldown -= 1
            self._stable_yolo_count = 0
            return self._decision(
                "candidate_reject_cooldown",
                reason="waiting after rejected candidate",
            )

        self._stable_yolo_count = self._count_if(
            observation.yolo_visible, self._stable_yolo_count
        )
        if (
            observation.yolo_visible
            and not self._candidate_in_flight
            and self._stable_yolo_count >= self.params.stable_yolo_frames
        ):
            self._candidate_in_flight = True
            return self._decision(
                "request_candidate",
                publish_init_request=True,
                reason="stable yolo target",
            )
        return self._decision(
            "wait_for_stable_yolo" if observation.yolo_visible else "wait_for_target",
            reason="waiting for target",
        )

    def _update_tracking(self, observation):
        if observation.m3t_startup_failed:
            self.state = NO_TRACK
            self._reset_no_track_counts()
            return self._decision(
                "stop_m3t_startup_failed",
                publish_stop=True,
                clear_candidate=True,
                reason="m3t reported setup failure",
            )

        if observation.tracking_reliability_state:
            return self._update_tracking_from_filter(observation)

        if observation.m3t_startup_pending:
            # Startup has its own deadline in the tracking filter. Until then it
            # neither publishes a pose nor counts toward ordinary loss.
            return self._decision(
                "wait_for_m3t_startup",
                reason="m3t starting; trusted output held",
            )

        if (
            observation.m3t_status == "tracking"
            and observation.m3t_pose_available
            and observation.tracking_health_ok
        ):
            self._reset_tracking_counts()
            return self._decision(
                "publish_trusted_pose",
                publish_trusted_pose=True,
                reason="tracking healthy",
            )

        if observation.m3t_status == "tracking" and observation.m3t_pose_available:
            self._reset_tracking_counts()
            return self._decision(
                "watch_tracking",
                reason="m3t alive but pose not trusted",
            )

        if observation.yolo_visible:
            self._bad_tracking_count += 1
            self._out_of_view_count = 0
            if self._bad_tracking_count >= self.params.lost_frames:
                self.state = REINIT
                self._candidate_in_flight = False
                return self._decision(
                    "stop_m3t_for_reinit",
                    publish_stop=True,
                    clear_candidate=True,
                    reason="target visible but m3t unhealthy",
                )
            return self._decision("watch_tracking", reason="tracking unhealthy")

        self._out_of_view_count += 1
        self._bad_tracking_count = 0
        if self._out_of_view_count >= self.params.out_of_view_frames:
            self.state = NO_TRACK
            self._reset_no_track_counts()
            return self._decision(
                "stop_m3t",
                publish_stop=True,
                clear_candidate=True,
                reason="target out of view",
            )
        return self._decision("watch_tracking", reason="target not visible")

    def _update_tracking_from_filter(self, observation):
        if observation.filter_stop_m3t:
            self.state = NO_TRACK
            self._reset_no_track_counts()
            return self._decision(
                "stop_m3t",
                publish_stop=True,
                clear_candidate=True,
                reason="tracking filter marked target lost",
            )

        if observation.filter_request_reinit:
            self.state = REINIT
            self._candidate_in_flight = False
            return self._decision(
                "stop_m3t_for_reinit",
                publish_stop=True,
                clear_candidate=True,
                reason="tracking filter marked tracker degraded",
            )

        if observation.m3t_startup_pending:
            return self._decision(
                "wait_for_m3t_startup",
                reason="m3t starting; trusted output held",
            )

        if observation.filter_publish_trusted_pose:
            self._reset_tracking_counts()
            return self._decision(
                "publish_trusted_pose",
                publish_trusted_pose=True,
                reason="tracking filter locked",
            )

        if observation.filter_keep_m3t_alive:
            self._reset_tracking_counts()
            return self._decision(
                "watch_tracking",
                reason="tracking filter kept m3t alive",
            )

        return self._decision(
            "watch_tracking",
            reason="tracking filter produced no terminal action",
        )

    def _update_reinit(self, observation):
        if observation.candidate_available:
            self._candidate_in_flight = False
            if observation.candidate_valid:
                self.state = TRACKING
                self._reset_tracking_counts()
                return self._decision(
                    "start_m3t",
                    publish_initial_pose=True,
                    clear_candidate=True,
                    reason="reinit candidate accepted",
                )
            self.state = NO_TRACK
            self._reset_no_track_counts()
            return self._decision(
                "reject_candidate",
                clear_candidate=True,
                reason="reinit candidate rejected",
            )

        if observation.m3t_status != "stopped":
            return self._decision("wait_for_m3t_stop", reason="waiting for old m3t stop")

        if observation.candidate_timed_out:
            self._candidate_in_flight = False
            return self._decision(
                "clear_timed_out_candidate",
                clear_candidate=True,
                reason="reinit candidate timed out",
            )

        if observation.yolo_visible and not self._candidate_in_flight:
            self._candidate_in_flight = True
            return self._decision(
                "request_candidate",
                publish_init_request=True,
                reason="reinitialize visible target",
            )

        if not observation.yolo_visible:
            self.state = NO_TRACK
            self._reset_no_track_counts()
            return self._decision("wait_for_target", reason="target no longer visible")
        return self._decision("wait_for_candidate", reason="candidate in flight")

    def _decision(
        self,
        action,
        publish_init_request=False,
        publish_initial_pose=False,
        publish_stop=False,
        publish_trusted_pose=False,
        clear_candidate=False,
        reason="",
    ):
        return LifecycleDecision(
            state=self.state,
            action=action,
            publish_init_request=publish_init_request,
            publish_initial_pose=publish_initial_pose,
            publish_stop=publish_stop,
            publish_trusted_pose=publish_trusted_pose,
            clear_candidate=clear_candidate,
            reason=reason,
        )

    def _reset_no_track_counts(self):
        self._stable_yolo_count = 0
        self._candidate_in_flight = False
        self._candidate_reject_cooldown = 0
        self._reset_tracking_counts()

    def _reset_tracking_counts(self):
        self._bad_tracking_count = 0
        self._out_of_view_count = 0

    @staticmethod
    def _count_if(condition, current):
        return current + 1 if condition else 0
