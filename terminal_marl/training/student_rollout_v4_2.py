"""88D student-only rollout shared by future DAgger and admitted PPO.

No CLI starts training here. DAgger labels are optional sidecars and never
enter env.step. Empty arrival/service intervals remain in team-time GAE.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..algo.joint_ppo_v4_2 import JointPPOV42, hierarchical_joint_logprob
from ..algo.student_ppo_v4_2 import StudentStepV42
from ..core.low_level_joint_action_v4_2 import (ACTION_SCHEMA_V4_2, RELEASE_GATE,
    HierarchicalActionV42, decode_hierarchical_actions_v4_2,
    hierarchical_from_joint_index_v4_1)
from ..core.runtime_traffic_safety import SafetyInvariantError
from ..core.safety_trace import SafetyTraceRecorder, _digest
from ..envs.parallel_env import TerminalParallelEnv
from .v4_1_joint_teacher import teacher_joint_expert_v4_1

OBS_SCHEMA = "phaseA_low_obs_v2_1_retry_state"


def _validate_label_stride_steps(label_stride_steps: int) -> int:
    if isinstance(label_stride_steps, bool) or not isinstance(label_stride_steps, (int, np.integer)):
        raise ValueError("label_stride_steps must be a positive integer")
    value = int(label_stride_steps)
    if value < 1:
        raise ValueError("label_stride_steps must be a positive integer")
    return value


def _label_query_uids(
    *,
    tick: int,
    uids: list[str],
    actor_mask: np.ndarray,
    gates: np.ndarray,
    label_stride_steps: int,
    previous_actor_mask: dict[str, bool],
    previous_requested_gates: dict[str, int],
) -> set[str]:
    """Select rows for bounded Teacher queries and update selector memory."""

    stride = _validate_label_stride_steps(label_stride_steps)
    periodic = int(tick) % stride == 0
    selected: set[str] = set()
    for index, uid in enumerate(uids):
        uid_key = str(uid)
        actor = bool(actor_mask[index])
        requested_gate = int(gates[index])
        first_actor = actor and not previous_actor_mask.get(uid_key, False)
        gate_changed = (
            actor
            and uid_key in previous_requested_gates
            and previous_requested_gates[uid_key] != requested_gate
        )
        if periodic or first_actor or gate_changed:
            selected.add(uid_key)
        previous_actor_mask[uid_key] = actor
        previous_requested_gates[uid_key] = requested_gate
    # An active UID missing from one observation is non-actor for the purpose
    # of the next appearance.  This keeps short-lived arrivals from being
    # skipped merely because the previous actor state was cached.
    current_uids = {str(uid) for uid in uids}
    for uid_key in tuple(previous_actor_mask):
        if uid_key not in current_uids:
            previous_actor_mask[uid_key] = False
    return selected


def validate_student_config(config, *, collect_labels: bool = False) -> None:
    if config.action_schema_version != ACTION_SCHEMA_V4_2 or config.observation_schema_version != OBS_SCHEMA:
        raise ValueError("student rollout requires V4.2 / 88D V2.1")
    labeler_only = collect_labels and config.teacher_enabled and config.teacher_mode == "teacher_label_only"
    if (config.teacher_enabled and not labeler_only) or config.scheduler_mode != "execution" or config.teacher_release_label_enabled:
        raise ValueError("student execution must disable Teacher substitution")
    if collect_labels and not labeler_only:
        raise ValueError("DAgger labels require an explicitly label-only Teacher")
    if config.high_level_enabled or not config.shield_enabled:
        raise ValueError("PPO admission requires fixed geometry and enabled safety shield")


def world_value(policy: JointPPOV42, world: np.ndarray) -> float:
    values = world if len(world) else np.zeros((1, policy.observation_dim), dtype=np.float32)
    mask = torch.full((1, len(values)), bool(len(world)), device=policy.device, dtype=torch.bool)
    with torch.no_grad():
        return float(policy.model.critic(torch.as_tensor(values[None], dtype=torch.float32, device=policy.device), mask)[0])


def _student_request_details(policy: JointPPOV42, env, observations: dict, *, deterministic: bool,
                             shortlist_k: int):
    if shortlist_k not in {1, 3}:
        raise ValueError("student runtime shortlist must have K=1 or K=3")
    uids = sorted(observations)
    world = np.asarray([observations[uid] for uid in uids], dtype=np.float32).reshape(-1, policy.observation_dim)
    actor_mask = np.asarray([
        env.active[int(uid)].committed_plan is None
        and env.active[int(uid)].phase.value in {"ring_outer", "ring_middle", "ring_inner"}
        and int(uid) not in env.future_release_reservations
        for uid in uids], dtype=bool)
    if not uids:
        return (uids, world, actor_mask, {}, np.zeros(0, dtype=int), np.zeros(0, dtype=int),
                np.zeros(0), np.zeros(0), {}, {})
    tensor = torch.as_tensor(world, device=policy.device)
    with torch.no_grad():
        gates, proposals = torch.zeros(len(uids), dtype=torch.long, device=policy.device), torch.full((len(uids),), -1, dtype=torch.long, device=policy.device)
        gate_dist, proposal_dist = policy.model.distributions(tensor)
        mask = torch.as_tensor(actor_mask, device=policy.device)
        if mask.any():
            logits = gate_dist.logits[mask]
            gates[mask] = logits.argmax(-1) if deterministic else torch.distributions.Categorical(logits=logits).sample()
        release = gates == 1
        if release.any():
            logits = proposal_dist.logits[release]
            proposals[release] = logits.argmax(-1) if deterministic else torch.distributions.Categorical(logits=logits).sample()
        gate_lp, proposal_lp, joint_lp = hierarchical_joint_logprob(gate_dist, proposal_dist, gates, proposals)
    gate_np, proposal_np = gates.cpu().numpy(), proposals.cpu().numpy()
    decoded = decode_hierarchical_actions_v4_2(gate_np, np.asarray([None if p == -1 else int(p) for p in proposal_np], dtype=object))
    actions = {uid: decoded[i] for i, uid in enumerate(uids)}
    likelihoods = {uid: {"gate": float(gate_lp[i]), "proposal": None if proposal_np[i] == -1 else float(proposal_lp[i]), "joint": float(joint_lp[i])} for i, uid in enumerate(uids)}
    shortlists: dict[str, tuple[np.ndarray, ...]] = {}
    if shortlist_k > 1:
        # The requested proposal remains rank 1 so PPO likelihood provenance
        # is unchanged.  Deterministic execution therefore uses the policy's
        # ordinary descending ranking; stochastic collection places its one
        # sampled request first, followed by the highest remaining logits.
        proposal_logits = proposal_dist.logits.detach().cpu()
        for index, uid in enumerate(uids):
            if not actor_mask[index] or gate_np[index] != RELEASE_GATE:
                continue
            selected = int(proposal_np[index])
            ranked = torch.argsort(proposal_logits[index], descending=True, stable=True).tolist()
            ordered = [selected] + [int(value) for value in ranked if int(value) != selected]
            shortlists[uid] = tuple(
                HierarchicalActionV42(RELEASE_GATE, proposal).normalized.copy()
                for proposal in ordered[:shortlist_k]
            )
    return (uids, world, actor_mask, actions, gate_np, proposal_np, gate_lp.cpu().numpy(),
            proposal_lp.cpu().numpy(), likelihoods, shortlists)


def student_request(policy: JointPPOV42, env, observations: dict, *, deterministic: bool):
    """Backward-compatible single-request policy API."""
    return _student_request_details(
        policy, env, observations, deterministic=deterministic, shortlist_k=1,
    )[:-1]


def _teacher_label(
    env,
    uav,
    *,
    audit_catalogue: bool,
    state_only: bool,
):
    if state_only:
        return teacher_joint_expert_v4_1(
            env, uav, audit_catalogue=audit_catalogue, state_only=True,
        )
    # Keep the historical call signature untouched for existing callers and
    # monkeypatched Teacher tests when the external labeler is disabled.
    return teacher_joint_expert_v4_1(env, uav, audit_catalogue=audit_catalogue)


def _accumulate_reward_components(totals: dict[str, float], reward_components: dict) -> None:
    """Accumulate post-step component values without double-counting ``total``."""

    for components in reward_components.values():
        for name, value in components.items():
            if name == "total":
                continue
            totals[name] = totals.get(name, 0.0) + float(value)


def rollout(policy: JointPPOV42, config, *, seed: int, max_steps: int,
            output: Path, deterministic: bool = False, collect_labels: bool = False,
            collect_diagnostics: bool = False,
            collect_catalogue_diagnostics: bool = False,
            catalogue_scope: str = "requested_release",
            label_stride_steps: int = 1,
            state_only_labels: bool = False) -> tuple[list[StudentStepV42], dict]:
    if state_only_labels and not collect_labels:
        raise ValueError("state_only_labels requires collect_labels")
    if collect_diagnostics and not collect_labels:
        raise ValueError("diagnostic attribution requires label-only Teacher rows")
    if collect_catalogue_diagnostics and not collect_diagnostics:
        raise ValueError("catalogue diagnostics require diagnostic attribution rows")
    if catalogue_scope not in {"requested_release", "teacher_release"}:
        raise ValueError("catalogue_scope must be requested_release or teacher_release")
    label_stride_steps = _validate_label_stride_steps(label_stride_steps)
    if state_only_labels:
        if (
            config.teacher_enabled
            or config.scheduler_mode != "execution"
            or config.teacher_release_label_enabled
        ):
            raise ValueError(
                "state-only labels require teacher_enabled=False, "
                "scheduler_mode='execution', and teacher_release_label_enabled=False"
            )
        validate_student_config(config, collect_labels=False)
    else:
        validate_student_config(config, collect_labels=collect_labels)
    if policy.observation_dim != 88 or max_steps < 1:
        raise ValueError("88D policy and positive rollout horizon required")
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    env = TerminalParallelEnv(config)
    observations, _ = env.reset(seed)
    tape = SafetyTraceRecorder(output / "action_tapes", dt=config.dt)
    tape.start_episode(env, episode_seed=seed, policy_rng_state={"seed": seed, "mode": "student_only_v2_1"})
    steps, events, queues = [], [], []
    reward_component_totals: dict[str, float] = {}
    safety_error = None
    counts = {"WAIT": 0, "RELEASE": 0, "shield": 0, "shield_actor": 0, "actor_decisions": 0}
    previous_actor_mask: dict[str, bool] = {}
    previous_requested_gates: dict[str, int] = {}
    for tick in range(max_steps):
        (uids, world, actor_mask, requested, gates, proposals, gate_lp, proposal_lp,
         likelihoods, proposal_shortlists) = _student_request_details(
            policy, env, observations, deterministic=deterministic,
            shortlist_k=config.runtime_proposal_shortlist_k,
        )
        value = world_value(policy, world)
        labels = {}
        pending = []
        state_hash = _digest(SafetyTraceRecorder._state(env))
        label_uids = _label_query_uids(
            tick=tick,
            uids=uids,
            actor_mask=actor_mask,
            gates=gates,
            label_stride_steps=label_stride_steps,
            previous_actor_mask=previous_actor_mask,
            previous_requested_gates=previous_requested_gates,
        ) if collect_labels else set()
        for i, uid in enumerate(uids):
            counts["actor_decisions"] += int(actor_mask[i])
            if actor_mask[i]:
                counts["RELEASE" if gates[i] else "WAIT"] += 1
            if collect_labels and str(uid) in label_uids:
                catalogue = ()
                if collect_catalogue_diagnostics and catalogue_scope == "teacher_release":
                    label, _ = _teacher_label(
                        env, env.active[int(uid)], audit_catalogue=False,
                        state_only=state_only_labels,
                    )
                    catalogue_requested = bool(actor_mask[i] and label.index > 0)
                    if catalogue_requested:
                        full_label, catalogue = _teacher_label(
                            env, env.active[int(uid)], audit_catalogue=True,
                            state_only=state_only_labels,
                        )
                        if full_label.index != label.index:
                            raise RuntimeError("catalogue Teacher changed the label")
                else:
                    catalogue_requested = bool(
                        collect_catalogue_diagnostics and actor_mask[i] and gates[i] == 1
                    )
                    label, catalogue = _teacher_label(
                        env, env.active[int(uid)], audit_catalogue=catalogue_requested,
                        state_only=state_only_labels,
                    )
                repeated, _ = _teacher_label(
                    env, env.active[int(uid)], audit_catalogue=False,
                    state_only=state_only_labels,
                )
                if label.index != repeated.index:
                    raise RuntimeError("nondeterministic DAgger label")
                target = hierarchical_from_joint_index_v4_1(label.index)
                labels[uid] = {"gate": target.gate, "proposal": target.proposal, "joint_index_v4_1": target.joint_index_v4_1}
                pending.append({"observation_schema": OBS_SCHEMA, "action_schema": ACTION_SCHEMA_V4_2,
                    "observation_v2_1": world[i].tolist(), "teacher_gate": target.gate,
                    "teacher_proposal": target.proposal, "teacher_joint_index": target.joint_index_v4_1,
                    "source": "student_visited_label_only", "seed": seed,
                    "phase": env.active[int(uid)].phase.value, "step": tick,
                    "observed_soc": float(env.active[int(uid)].soc),
                    "base_priority": int(env.active[int(uid)].base_priority),
                    "episode_id": f"student:{seed}",
                    "lifecycle_event_id": f"student:{seed}:uav:{uid}", "decision_event_id": f"student:{seed}:uav:{uid}:step:{tick}",
                    "observation_hash": _digest(world[i]), "state_hash": state_hash, "uav_id": uid,
                    "requested_gate": int(gates[i]), "requested_proposal": None if proposals[i] == -1 else int(proposals[i]),
                    "gate_logprob": float(gate_lp[i]), "proposal_logprob": likelihoods[uid]["proposal"],
                    "requested_logprob": likelihoods[uid]["joint"], "actor_mask": bool(actor_mask[i]),
                    **({
                        "sim_time_before": float(env.now),
                        "arrival_id": int(env.active[int(uid)].arrival_id),
                        "active_uavs_before": int(len(env.active)),
                        "completed_uavs_before": int(len(env.completed_uavs)),
                    } if collect_diagnostics else {}),
                    **({
                        "teacher_catalogue_complete": bool(catalogue_requested),
                        "teacher_catalogue_scope": catalogue_scope,
                        "teacher_feasible_proposals": [
                            int(item["joint_index"]) - 1 for item in catalogue
                            if bool(item.get("feasible")) and int(item.get("joint_index", 0)) > 0
                        ],
                    } if collect_catalogue_diagnostics and actor_mask[i] else {})})
        if collect_labels and _digest(SafetyTraceRecorder._state(env)) != state_hash:
            raise RuntimeError("Teacher label query mutated environment state")
        record = tape.record_pre_step(
            env, observations, requested, likelihoods,
            proposal_shortlists=proposal_shortlists,
        )
        record["actor_mask"] = {uid: bool(actor_mask[i]) for i, uid in enumerate(uids)}
        try:
            following, rewards, _, _, info = env.step(requested, proposal_shortlists)
        except SafetyInvariantError as exc:
            safety_error = str(exc)
            tape.record_safety_invariant(record, exc)
            break
        tape.record_post_step(env, record, info)
        _accumulate_reward_components(reward_component_totals, info["reward_components"])
        if collect_labels:
            tape.record_teacher_labels(record, labels)
        if any(trace.get("teacher_used") for trace in info["action_traces"].values()):
            raise RuntimeError("Teacher replaced student behavior")
        for uid in uids:
            if not np.array_equal(np.asarray(info["action_traces"][uid]["requested_low_action"], dtype=np.float32), requested[uid]):
                raise RuntimeError("requested-action provenance mismatch")
        for row in pending:
            trace = info["action_traces"][row["uav_id"]]
            row.update(execution_applied_action=trace["execution_applied_action"], shield_applied_action=trace["shield_applied_action"], post_state_hash=_digest(record["post_state"]))
            if collect_diagnostics:
                release = env.last_release_telemetry.get(int(row["uav_id"]))
                history = [] if release is None else list(release.get("decision_history", ()))
                row.update({
                    "release_decision": history[-1] if history else None,
                    "release_interface": None if release is None else release.get("interface"),
                    "release_attempt_count": None if release is None else release.get("attempt_count"),
                    "release_retry_count": None if release is None else release.get("retry_count"),
                    "release_primary_blocker": None if release is None else release.get("primary_blocker"),
                    "release_secondary_constraints": [] if release is None else list(release.get("secondary_constraints", ())),
                    "release_diagnostic_present": bool(release is not None),
                })
        events.extend(pending)
        counts["shield"] += sum(bool(t.get("shield_used")) for t in info["action_traces"].values())
        counts["shield_actor"] += sum(bool(info["action_traces"][uid].get("shield_used"))
                                      for i, uid in enumerate(uids) if actor_mask[i])
        queues.append(int(info["queue_size"]))
        terminal = env.arrivals.exhausted() and not env.active and not env.admission.queue.items and not env.service_uavs
        end = terminal or tick + 1 == max_steps or env.now >= config.max_simulation_seconds
        next_world = np.asarray([following[uid] for uid in sorted(following)], dtype=np.float32).reshape(-1, 88)
        step = StudentStepV42(world.copy(), gates.copy(), proposals.copy(), gate_lp.copy(), proposal_lp.copy(), actor_mask.copy(), value,
                             0. if terminal else world_value(policy, next_world), float(sum(rewards.values())), bool(terminal), bool(end),
                             stochastic_behavior=not deterministic)
        step.validate(88)
        steps.append(step)
        observations = following
        if (tick + 1) % 25 == 0 or end:
            (output / "progress.json").write_text(json.dumps({"seed": seed, "steps": tick + 1,
                "seconds": env.now, "counts": counts, "completed": len(env.completed_uavs),
                "queue": queues[-1], "terminal": terminal}), encoding="utf-8")
        if end:
            break
    tape_path = tape.dump(reason="student_only_v2_1")
    with (output / "labels.jsonl").open("w", encoding="utf-8") as handle:
        for row in events:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    safety = {k: int(env.telemetry.get(k, 0)) for k in ("collisions", "nmac", "hard_separation_violations", "uncertified_execution")}
    safety["SafetyInvariantError"] = int(safety_error is not None)
    runtime_shortlist = {key: int(env.telemetry.get(key, 0)) for key in (
        "runtime_shortlist_candidates_checked", "runtime_shortlist_fallback_selected",
        "runtime_shortlist_exhausted",
    )}
    reward_total = float(sum(step.reward for step in steps))
    reward_component_totals = {key: float(value) for key, value in reward_component_totals.items()}
    result = {"seed": seed, "steps": len(steps), "seconds": env.now, "counts": counts,
              "safety": safety, "safety_error": safety_error, "completed": len(env.completed_uavs),
              "outcomes": dict(env.ledger.outcomes), "queue": queues,
              "terminal": bool(steps and steps[-1].terminated), "action_tape": str(tape_path),
              "label_rows": len(events), "runtime_shortlist": runtime_shortlist,
              "reward_component_totals": reward_component_totals,
              "reward_total": reward_total,
              "reward_components_consistent": bool(np.isclose(
                  sum(reward_component_totals.values()), reward_total,
                  rtol=1e-6, atol=1e-6,
              )),
              "label_stride_steps": int(label_stride_steps),
              "state_only_labels": bool(state_only_labels),
              "PPO_ALLOWED": False}
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return steps, result
