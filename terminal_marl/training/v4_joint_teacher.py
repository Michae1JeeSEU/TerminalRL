"""Bounded training-time V4 joint-action Teacher query.

The runtime scheduler is deliberately not involved here: it receives precisely
one decoded policy request.  This helper is for offline labels and smoke
qualification only, where :class:`TeacherScheduler` is allowed to select a
deterministic best action from the finite V4 catalogue.
"""
from __future__ import annotations

from ..core.config import RingRole
from ..core.low_level_joint_action import JointAction
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


def _candidate_score(candidate: JointAction) -> dict[str, float | int | str]:
    """The explicit, state-independent V4 catalogue preference key."""
    return {
        "preference_group": "RELEASE" if candidate.release_request > 0.0 else "WAIT",
        "release_priority": 0 if candidate.release_request > 0.0 else 1,
        "abs_speed_residual": abs(float(candidate.speed_residual)),
        "abs_merge_time_residual": abs(float(candidate.merge_time_residual)),
        "joint_index": candidate.index,
    }


def teacher_joint_expert(env: TerminalParallelEnv, uav, *, audit_catalogue: bool = False) -> tuple[JointAction, tuple[dict, ...]]:
    """Evaluate the complete V4 catalogue on one immutable decision state.

    Candidate checks run through the same proposal/trajectory/safety path as
    execution, but never reserve or commit.  The returned expert target is
    deliberately independent of a policy request; audit rows expose the exact
    feasibility and deterministic rank used to establish that property when
    ``audit_catalogue`` is requested.  Ordinary label collection preserves
    the previous bounded first-certified-candidate cost.
    """
    if env.teacher_scheduler is None:
        raise RuntimeError("V4 joint Teacher requires TeacherScheduler")

    context = _release_context(env, uav)
    snapshot = WorldSnapshot(env.now, env.active, env.airspace)
    audit: list[dict] = []
    selected: JointAction | None = None
    for rank, candidate in enumerate(env.teacher_scheduler.joint_catalogue()):
        feasible = False
        reason = ""
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
                        uav,
                        RepairCandidate("v4_joint_teacher", LowLevelIntent(*candidate.normalized), 0.0),
                        target, snapshot, lz_id,
                    )
                    feasible = proposal is not None and plan is not None and reason is None
                    reason = "certified" if feasible else str(reason or "candidate_rejected")
        audit.append({"rank": rank, "joint_index": candidate.index, "decoded": candidate.as_dict(),
                      "candidate_score": _candidate_score(candidate), "feasible": feasible,
                      "reason": reason, "tie_break": "catalogue_rank_release_then_abs_residual_then_index"})
        if feasible and selected is None:
            selected = candidate
            if not audit_catalogue:
                break
    if selected is None:
        # The nominal WAIT fallback remains a total state-conditioned expert.
        selected = next(row for row in env.teacher_scheduler.joint_catalogue()
                        if row.release_request == 0.0 and row.speed_residual == 0.0)
    return selected, tuple(audit)


def teacher_joint_label(env: TerminalParallelEnv, uav) -> JointAction:
    """Compatibility label API: return the state-conditioned V4 expert only."""
    return teacher_joint_expert(env, uav, audit_catalogue=False)[0]


def component_disagreement(requested: JointAction, teacher: JointAction) -> dict[str, bool]:
    """Diagnostic-only decomposition; it is never used as BC supervision."""
    return {
        "release": requested.release_request != teacher.release_request,
        "speed": requested.speed_residual != teacher.speed_residual,
        "merge_time": requested.merge_time_residual != teacher.merge_time_residual,
    }
