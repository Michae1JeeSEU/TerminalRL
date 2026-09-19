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
from ..core.low_level_joint_action_v4_2 import ACTION_SCHEMA_V4_2, decode_hierarchical_actions_v4_2, hierarchical_from_joint_index_v4_1
from ..core.runtime_traffic_safety import SafetyInvariantError
from ..core.safety_trace import SafetyTraceRecorder, _digest
from ..envs.parallel_env import TerminalParallelEnv
from .v4_1_joint_teacher import teacher_joint_expert_v4_1

OBS_SCHEMA = "phaseA_low_obs_v2_1_retry_state"


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


def student_request(policy: JointPPOV42, env, observations: dict, *, deterministic: bool):
    uids = sorted(observations)
    world = np.asarray([observations[uid] for uid in uids], dtype=np.float32).reshape(-1, policy.observation_dim)
    actor_mask = np.asarray([
        env.active[int(uid)].committed_plan is None
        and env.active[int(uid)].phase.value in {"ring_outer", "ring_middle", "ring_inner"}
        and int(uid) not in env.future_release_reservations
        for uid in uids], dtype=bool)
    if not uids:
        return uids, world, actor_mask, {}, np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros(0), np.zeros(0), {}
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
    return uids, world, actor_mask, actions, gate_np, proposal_np, gate_lp.cpu().numpy(), proposal_lp.cpu().numpy(), likelihoods


def rollout(policy: JointPPOV42, config, *, seed: int, max_steps: int,
            output: Path, deterministic: bool = False, collect_labels: bool = False,
            collect_diagnostics: bool = False) -> tuple[list[StudentStepV42], dict]:
    if collect_diagnostics and not collect_labels:
        raise ValueError("diagnostic attribution requires label-only Teacher rows")
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
    safety_error = None
    counts = {"WAIT": 0, "RELEASE": 0, "shield": 0, "shield_actor": 0, "actor_decisions": 0}
    for tick in range(max_steps):
        uids, world, actor_mask, requested, gates, proposals, gate_lp, proposal_lp, likelihoods = student_request(policy, env, observations, deterministic=deterministic)
        value = world_value(policy, world)
        labels = {}
        pending = []
        state_hash = _digest(SafetyTraceRecorder._state(env))
        for i, uid in enumerate(uids):
            counts["actor_decisions"] += int(actor_mask[i])
            if actor_mask[i]:
                counts["RELEASE" if gates[i] else "WAIT"] += 1
            if collect_labels:
                label, _ = teacher_joint_expert_v4_1(env, env.active[int(uid)], audit_catalogue=False)
                repeated, _ = teacher_joint_expert_v4_1(env, env.active[int(uid)], audit_catalogue=False)
                if label.index != repeated.index:
                    raise RuntimeError("nondeterministic DAgger label")
                target = hierarchical_from_joint_index_v4_1(label.index)
                labels[uid] = {"gate": target.gate, "proposal": target.proposal, "joint_index_v4_1": target.joint_index_v4_1}
                pending.append({"observation_schema": OBS_SCHEMA, "action_schema": ACTION_SCHEMA_V4_2,
                    "observation_v2_1": world[i].tolist(), "teacher_gate": target.gate,
                    "teacher_proposal": target.proposal, "teacher_joint_index": target.joint_index_v4_1,
                    "source": "student_visited_label_only", "seed": seed,
                    "phase": env.active[int(uid)].phase.value, "step": tick,
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
                    } if collect_diagnostics else {})})
        record = tape.record_pre_step(env, observations, requested, likelihoods)
        record["actor_mask"] = {uid: bool(actor_mask[i]) for i, uid in enumerate(uids)}
        try:
            following, rewards, _, _, info = env.step(requested)
        except SafetyInvariantError as exc:
            safety_error = str(exc)
            tape.record_safety_invariant(record, exc)
            break
        tape.record_post_step(env, record, info)
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
    result = {"seed": seed, "steps": len(steps), "seconds": env.now, "counts": counts,
              "safety": safety, "safety_error": safety_error, "completed": len(env.completed_uavs),
              "outcomes": dict(env.ledger.outcomes), "queue": queues,
              "terminal": bool(steps and steps[-1].terminated), "action_tape": str(tape_path),
              "label_rows": len(events), "PPO_ALLOWED": False}
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return steps, result
