"""State-conditioned V4.1 Teacher over one WAIT and 25 RELEASE proposals."""
from __future__ import annotations

from ..core.config import RingRole
from ..core.low_level_joint_action_v4_1 import JointActionV41
from ..core.low_level_joint_action_v4_2 import ACTION_SCHEMA_V4_2
from ..core.merge_scheduler import LowLevelIntent, RepairCandidate
from ..core.uav import FlightPhase
from ..envs.parallel_env import TerminalParallelEnv, WorldSnapshot


def _release_context(env: TerminalParallelEnv, uav) -> tuple[RingRole | str, str] | None:
    if uav.committed_plan is not None or env.now < uav.safety_retry_after:
        return None
    if uav.phase is FlightPhase.RING_OUTER:
        return RingRole.MIDDLE, "o2m"
    if uav.phase is FlightPhase.RING_MIDDLE:
        return RingRole.INNER, "m2i"
    if uav.phase is FlightPhase.RING_INNER:
        return "final", "i2f"
    return None


def _candidate_score(candidate: JointActionV41) -> dict[str, float | int | str]:
    # WAIT has no residual candidate or residual-ranking term in V4.1.
    score: dict[str, float | int | str] = {
        "preference_group": "RELEASE" if candidate.release_request > 0.0 else "WAIT",
        "release_priority": 0 if candidate.release_request > 0.0 else 1,
        "joint_index": candidate.index,
    }
    if candidate.release_request > 0.0:
        score.update({"abs_speed_residual": abs(float(candidate.speed_residual)),
                      "abs_merge_time_residual": abs(float(candidate.merge_time_residual))})
    return score


def teacher_joint_expert_v4_1(env: TerminalParallelEnv, uav, *, audit_catalogue: bool = False) -> tuple[JointActionV41, tuple[dict, ...]]:
    if env.config.action_schema_version not in {"low_action_schema_v4_1_joint_discrete_26", ACTION_SCHEMA_V4_2}:
        raise ValueError("V4.1 semantic Teacher requires the V4.1 or V4.2 action schema")
    if env.teacher_scheduler is None:
        raise RuntimeError("V4.1 joint Teacher requires TeacherScheduler")
    context = _release_context(env, uav)
    snapshot = WorldSnapshot(env.now, env.active, env.airspace)
    audit: list[dict] = []
    selected: JointActionV41 | None = None
    for rank, candidate in enumerate(env.teacher_scheduler.joint_catalogue()):
        feasible, reason = False, ""
        if candidate.release_request <= 0.0:
            feasible, reason = True, "wait_legal"
        elif context is None:
            reason = "release_context_masked"
        else:
            target, interface = context
            if interface in env.airspace.affected_interfaces():
                reason = "geometry_frozen"
            else:
                lz_id = None
                if target == "final":
                    if sum(item.phase is FlightPhase.FINAL for item in env.active.values()) >= env.config.final_queue_capacity:
                        reason = "final_capacity"
                    else:
                        lz_id = env.lz_manager.choose_rule_zone(uav, env.now)
                        if lz_id is None:
                            reason = "lz_unavailable"
                if not reason:
                    proposal, plan, reason = env._candidate_plan(
                        uav, RepairCandidate("v4_1_joint_teacher", LowLevelIntent(*candidate.normalized), 0.0),
                        target, snapshot, lz_id)
                    feasible = proposal is not None and plan is not None and reason is None
                    reason = "certified" if feasible else str(reason or "candidate_rejected")
        audit.append({"rank": rank, "joint_index": candidate.index, "decoded": candidate.as_dict(),
                      "candidate_score": _candidate_score(candidate), "feasible": feasible, "reason": reason,
                      "tie_break": "catalogue_rank_release_then_abs_residual_then_index" if candidate.release_request > 0 else "canonical_wait"})
        if feasible and selected is None:
            selected = candidate
            if not audit_catalogue:
                break
    if selected is None:
        selected = next(row for row in env.teacher_scheduler.joint_catalogue() if row.release_request == 0.0)
    return selected, tuple(audit)


def teacher_joint_label_v4_1(env: TerminalParallelEnv, uav) -> JointActionV41:
    return teacher_joint_expert_v4_1(env, uav)[0]


def component_disagreement_v4_1(requested: JointActionV41, teacher: JointActionV41) -> dict[str, bool]:
    return {"release": requested.release_request != teacher.release_request,
            "speed": requested.speed_residual != teacher.speed_residual,
            "merge_time": requested.merge_time_residual != teacher.merge_time_residual}
