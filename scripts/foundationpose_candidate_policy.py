from dataclasses import dataclass


@dataclass(frozen=True)
class FoundationPoseOutputDecision:
    publish_initial_pose: bool
    publish_candidate_pose: bool
    event: str
    request_id: str | None = None


class FoundationPoseCandidatePolicy:
    def __init__(self, direct_initial_pose_enabled=False, request_required=True):
        self.direct_initial_pose_enabled = bool(direct_initial_pose_enabled)
        self.request_required = bool(request_required)

    def on_register_result(self, active_request):
        if active_request is None:
            if self.request_required:
                return FoundationPoseOutputDecision(
                    publish_initial_pose=False,
                    publish_candidate_pose=False,
                    event="foundationpose_waiting_for_init_request",
                )
            return FoundationPoseOutputDecision(
                publish_initial_pose=self.direct_initial_pose_enabled,
                publish_candidate_pose=False,
                event="foundationpose_initial_pose_published"
                if self.direct_initial_pose_enabled
                else "foundationpose_register_result_ignored",
            )

        request_id = str(active_request.get("request_id", ""))
        return FoundationPoseOutputDecision(
            publish_initial_pose=self.direct_initial_pose_enabled,
            publish_candidate_pose=True,
            event="foundationpose_candidate_pose_published",
            request_id=request_id,
        )
