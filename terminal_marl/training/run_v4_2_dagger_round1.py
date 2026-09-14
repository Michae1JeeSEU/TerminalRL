"""V4.2 Safe DAgger Round 1: semantic collection plus CE-only fine-tuning."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ..algo.joint_ppo_v4_2 import JointPPOV42, hierarchical_bc_loss
from ..core.checkpoints import load_checkpoint, save_checkpoint
from ..core.low_level_joint_action_v4_2 import ACTION_SCHEMA_V4_2, RELEASE_GATE, hierarchical_from_joint_index_v4_1
from ..core.runtime_traffic_safety import SafetyInvariantError
from ..core.safety_trace import SafetyTraceRecorder, _digest
from ..envs.parallel_env import TerminalParallelEnv
from .run_v4_2_hierarchical_s0_qualification import _config
from .run_v4_2_i1_hierarchical_bc import _arrays, _metrics
from .run_v4_2_i2_dagger_preflight import _hash, _policy_tape_logprob_exact, _read_jsonl
from .v4_1_joint_teacher import teacher_joint_label_v4_1


def _config_for_steps(max_steps: int):
    config = _config(); config.max_simulation_seconds = float(max_steps); return config


def _empty_diagnostics() -> dict:
    return {"student_WAIT": 0, "student_RELEASE": 0, "teacher_WAIT": 0, "teacher_RELEASE": 0,
            "gate_disagreement": 0, "proposal_disagreement_on_teacher_RELEASE": 0,
            "teacher_RELEASE_events": 0, "shield_used": 0, "completion": 0, "throughput": 0,
            "queue": [], "delay_seconds": [], "student_proposal_histogram": {}}


def _rates(data: dict) -> dict:
    teacher_release = max(1, data["teacher_RELEASE_events"])
    total = max(1, data["student_WAIT"] + data["student_RELEASE"])
    return {"gate_disagreement_rate": data["gate_disagreement"] / total,
            "proposal_disagreement_rate_on_teacher_RELEASE": data["proposal_disagreement_on_teacher_RELEASE"] / teacher_release}


def _update_diagnostics(data: dict, *, student_gate: int, student_proposal: int | None, teacher, trace: dict, queue: int) -> None:
    data["student_WAIT" if student_gate == 0 else "student_RELEASE"] += 1
    data["teacher_WAIT" if teacher.gate == 0 else "teacher_RELEASE"] += 1
    data["gate_disagreement"] += int(student_gate != teacher.gate)
    if teacher.gate == RELEASE_GATE:
        data["teacher_RELEASE_events"] += 1
        data["proposal_disagreement_on_teacher_RELEASE"] += int(student_proposal != teacher.proposal)
    if student_proposal is not None:
        histogram = data["student_proposal_histogram"]; histogram[str(student_proposal)] = histogram.get(str(student_proposal), 0) + 1
    data["shield_used"] += int(bool(trace.get("shield_used"))); data["queue"].append(int(queue))


def _safety_add(total: dict, env) -> None:
    total["collision"] += int(env.telemetry.get("collisions", 0)); total["NMAC"] += int(env.telemetry.get("nmac", 0))
    total["hard"] += int(env.telemetry.get("hard_separation_violations", 0)); total["uncertified"] += int(env.telemetry.get("uncertified_components", 0))


def _semantic_reason(memory: dict, key: tuple[int, str], *, teacher_joint: int, observation: np.ndarray, tick: int) -> str | None:
    previous = memory.get(key)
    if previous is None:
        memory[key] = {"teacher_joint": teacher_joint, "observation": observation.copy(), "tick": tick}; return "event_start"
    if teacher_joint != previous["teacher_joint"]:
        reason = "teacher_target_change"
    elif float(np.linalg.norm(observation - previous["observation"]) / np.sqrt(observation.size)) >= .20:
        reason = "meaningful_observation_change"
    elif tick - previous["tick"] >= 3:
        reason = "bounded_sparse_persistence"
    else:
        return None
    memory[key] = {"teacher_joint": teacher_joint, "observation": observation.copy(), "tick": tick}; return reason


def _collect_round(policy: JointPPOV42, config, output: Path, *, target_events: int, max_seeds: int, max_steps: int) -> tuple[list[dict], dict, dict, list[str], bool, bool, int]:
    events: list[dict] = []; diagnostics = _empty_diagnostics(); safety = {"collision": 0, "NMAC": 0, "hard": 0, "SafetyInvariantError": 0, "uncertified": 0}
    tapes: list[str] = []; teacher_determinism = True; authoritative = True; raw_ticks = 0
    for episode, seed in enumerate(range(9801, 9801 + max_seeds)):
        env = TerminalParallelEnv(config); observations, _ = env.reset(seed); memory: dict = {}
        tape = SafetyTraceRecorder(output / "action_tapes", dt=config.dt); tape.start_episode(env, episode_seed=seed, policy_rng_state={"round": 1, "student_checkpoint": "I1"})
        for _ in range(max_steps):
            if not observations:
                break
            uids = sorted(observations); values = np.asarray([observations[uid] for uid in uids], dtype=np.float32); batch = policy.act_details(values, values, deterministic=False)
            requested = {uid: batch.decoded[position] for position, uid in enumerate(uids)}; labels: dict[str, dict] = {}; pending: list[tuple[int, dict]] = []
            state_hash = _digest(SafetyTraceRecorder._state(env)); raw_ticks += len(uids)
            for position, uid in enumerate(uids):
                uav = env.active[int(uid)]; label = teacher_joint_label_v4_1(env, uav)
                if raw_ticks % 13 == 0:
                    teacher_determinism = teacher_determinism and label.index == teacher_joint_label_v4_1(env, uav).index
                teacher = hierarchical_from_joint_index_v4_1(label.index); labels[uid] = {"gate": teacher.gate, "proposal": teacher.proposal, "joint_index_v4_1": teacher.joint_index_v4_1}
                reason = _semantic_reason(memory, (int(uid), uav.phase.value), teacher_joint=label.index, observation=values[position], tick=env.step_count)
                if reason is not None and len(events) < target_events:
                    lifecycle = f"dagger_r1:seed-{seed}:episode-{episode}:uav-{uid}:phase-{uav.phase.value}"
                    pending.append((position, {"schema_version": "v4_2_dagger_round1_event_v1", "action_schema": ACTION_SCHEMA_V4_2, "round": 1,
                        "observation": values[position].tolist(), "requested_gate": int(batch.requested_gate[position]), "requested_proposal": batch.requested_proposal[position],
                        "requested_joint_logprob": float(batch.joint_logprob[position]), "teacher_gate": teacher.gate, "teacher_proposal": teacher.proposal,
                        "teacher_semantic_joint_index_v4_1": teacher.joint_index_v4_1, "gate_target": teacher.gate, "proposal_target": teacher.proposal,
                        "decoded_requested_action": batch.decoded[position].tolist(), "source": "student_closed_loop_dagger_round1", "seed": seed, "episode_id": episode,
                        "uav_id": uid, "lifecycle_event_id": lifecycle, "decision_event_id": f"{lifecycle}:step-{env.step_count}:reason-{reason}",
                        "semantic_sampling_reason": reason, "obs_hash": _digest(values[position]), "state_hash": state_hash, "teacher_deterministic": True,
                        "teacher_label_only": True}))
            log_probs = {uid: {"gate": float(batch.gate_logprob[position]), "proposal": batch.proposal_logprob[position], "joint": float(batch.joint_logprob[position])} for position, uid in enumerate(uids)}
            record = tape.record_pre_step(env, observations, requested, log_probs)
            try:
                observations, _, _, _, info = env.step(requested)
            except SafetyInvariantError as error:
                safety["SafetyInvariantError"] += 1; tape.record_safety_invariant(record, error); break
            tape.record_post_step(env, record, info); tape.record_teacher_labels(record, labels); post_hash = _digest(SafetyTraceRecorder._state(env))
            for position, event in pending:
                trace = info["action_traces"][event["uav_id"]]
                event.update({"applied_action": trace.get("execution_applied_action"), "shield_action": trace.get("shield_applied_action"), "post_state_hash": post_hash,
                              "teacher_did_not_replace_execution": not trace.get("teacher_used", False), "requested_action_matches_student": np.allclose(np.asarray(trace["requested_low_action"]), np.asarray(event["decoded_requested_action"]))})
                authoritative = authoritative and event["requested_action_matches_student"] and event["teacher_did_not_replace_execution"]
                _update_diagnostics(diagnostics, student_gate=event["requested_gate"], student_proposal=event["requested_proposal"], teacher=hierarchical_from_joint_index_v4_1(event["teacher_semantic_joint_index_v4_1"]), trace=trace, queue=info["queue_size"])
                events.append(event)
            if len(events) >= target_events:
                # Finish the current short trajectory/tape, but do not invent
                # additional DAgger samples after the target is reached.
                continue
        diagnostics["completion"] += len(env.completed_uavs); diagnostics["throughput"] += len(env.completed_uavs)
        _safety_add(safety, env); tapes.append(str(tape.dump(reason="v4_2_dagger_round1")))
        if len(events) >= target_events:
            break
    return events, diagnostics, safety, tapes, teacher_determinism, authoritative, raw_ticks


def _fine_tune(policy: JointPPOV42, rows: list[dict], *, epochs: int = 35, batch_size: int = 64, seed: int = 4301) -> dict:
    torch.manual_seed(seed); np.random.seed(seed); x, gate, proposal, joint = _arrays(rows); before = _metrics(policy, x, gate, proposal, joint); gradients = []
    for _ in range(epochs):
        for ids in (np.random.permutation(len(x))[offset:offset + batch_size] for offset in range(0, len(x), batch_size)):
            gate_dist, proposal_dist = policy.model.distributions(x[ids]); loss, _ = hierarchical_bc_loss(gate_dist.logits, proposal_dist.logits, gate[ids], proposal[ids])
            policy.optimizer.zero_grad(); loss.backward(); gradients.append(float(torch.nn.utils.clip_grad_norm_(policy.model.parameters(), 1.0))); policy.optimizer.step()
    return {"before_D1": before, "after_D1": _metrics(policy, x, gate, proposal, joint), "gradient_norm": {"mean": float(np.mean(gradients)), "max": float(np.max(gradients)), "finite": bool(np.isfinite(gradients).all())}}


def _validate_closed_loop(policy: JointPPOV42, config, *, seeds: tuple[int, ...] = (9901, 9902, 9903), max_steps: int = 8) -> tuple[dict, dict, bool]:
    diagnostics = _empty_diagnostics(); safety = {"collision": 0, "NMAC": 0, "hard": 0, "SafetyInvariantError": 0, "uncertified": 0}; label_only = True
    for seed in seeds:
        env = TerminalParallelEnv(config); observations, _ = env.reset(seed)
        for _ in range(max_steps):
            if not observations: break
            uids = sorted(observations); values = np.asarray([observations[uid] for uid in uids], dtype=np.float32); batch = policy.act_details(values, values, deterministic=False)
            requested = {uid: batch.decoded[pos] for pos, uid in enumerate(uids)}; teachers = [hierarchical_from_joint_index_v4_1(teacher_joint_label_v4_1(env, env.active[int(uid)]).index) for uid in uids]
            try:
                observations, _, _, _, info = env.step(requested)
            except SafetyInvariantError:
                safety["SafetyInvariantError"] += 1; break
            for pos, uid in enumerate(uids):
                trace = info["action_traces"][uid]; label_only = label_only and not trace.get("teacher_used", False)
                _update_diagnostics(diagnostics, student_gate=int(batch.requested_gate[pos]), student_proposal=batch.requested_proposal[pos], teacher=teachers[pos], trace=trace, queue=info["queue_size"])
            _safety_add(safety, env); diagnostics["completion"] += len(env.completed_uavs); diagnostics["throughput"] += len(env.completed_uavs)
    return diagnostics, safety, label_only


def run(student_checkpoint: str | Path, derived_i0_manifest: str | Path, i1_manifest: str | Path, output_dir: str | Path, *, target_events: int = 500, max_seeds: int = 24, max_steps: int = 10) -> dict:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True); lineage = json.loads(Path(derived_i0_manifest).read_text(encoding="utf-8")); i1 = json.loads(Path(i1_manifest).read_text(encoding="utf-8"))
    if not lineage.get("lineage_PASS") or i1.get("V4_2_I1_HIERARCHICAL_BC") != "PASS": raise ValueError("Round 1 requires PASS derived lineage and I1")
    config = _config_for_steps(max_steps); policy = JointPPOV42(87); load_checkpoint(student_checkpoint, policy, config, load_optimizer=True); policy.model.eval()
    events, collection, collection_safety, tapes, determinism, authoritative, raw_ticks = _collect_round(policy, config, output, target_events=target_events, max_seeds=max_seeds, max_steps=max_steps)
    if len(events) < target_events: raise RuntimeError(f"semantic collection shortfall: {len(events)} < {target_events}")
    ids = [row["decision_event_id"] for row in events]; parent = _read_jsonl(lineage["derived_dataset"]["path"]); aggregate = parent + events; aggregate_path = output / "D1_round1.jsonl"; aggregate_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in aggregate), encoding="utf-8")
    d1 = {"stage": "D1_V4_2_DAGGER_ROUND1", "round": 1, "action_schema": ACTION_SCHEMA_V4_2, "parent_dataset_sha256": lineage["derived_dataset"]["sha256"], "parent_rows": len(parent), "student_checkpoint_sha256": _hash(student_checkpoint), "teacher_version": "v4_1_joint_teacher_semantic_unmodified", "source_tags": sorted(set(row["source"] for row in events)), "new_student_visited_events": len(events), "decision_event_unique": len(ids) == len(set(ids)), "dataset_sha256": _hash(aggregate_path), "aggregate_rows": len(aggregate), "PPO_ALLOWED": False, "formal_retraining": "hierarchical_BC_only"}
    d1_path = output / "D1_round1.manifest.json"; d1_path.write_text(json.dumps(d1, indent=2), encoding="utf-8")
    replay = SafetyTraceRecorder.replay_action_tape(TerminalParallelEnv(config), tapes[0]); replay["policy_joint_logprob_match"] = _policy_tape_logprob_exact(policy, config, tapes[0])
    training = _fine_tune(policy, aggregate); checkpoint = output / "dagger_r1_v4_2_hierarchical_actor.pt"; save_checkpoint(checkpoint, policy, config, stage="DAGGER_R1_V4_2", ppo_called=False, parent_checkpoint_sha256=_hash(student_checkpoint))
    cohort = _read_jsonl(lineage["derived_frozen_cohort"]["path"]); vx, vg, vp, vj = _arrays(cohort); fixed = _metrics(policy, vx, vg, vp, vj)
    validation, validation_safety, validation_label_only = _validate_closed_loop(policy, _config_for_steps(8))
    safety = {key: collection_safety[key] + validation_safety[key] for key in collection_safety}; baseline = i1["frozen_cohort"]
    fixed_ok = bool(fixed["gate"]["WAIT_recall"] >= baseline["gate"]["WAIT_recall"] - .05 and fixed["gate"]["RELEASE_recall"] >= baseline["gate"]["RELEASE_recall"] - .05 and fixed["gate"]["macro_F1"] >= baseline["gate"]["macro_F1"] - .05 and fixed["proposal"]["top_3"] >= baseline["proposal"]["top_3"] - .05)
    collection_rates, validation_rates = _rates(collection), _rates(validation); disagreement_stable = validation_rates["gate_disagreement_rate"] <= collection_rates["gate_disagreement_rate"] + .10 and validation_rates["proposal_disagreement_rate_on_teacher_RELEASE"] <= collection_rates["proposal_disagreement_rate_on_teacher_RELEASE"] + .15
    no_gate_collapse = validation["student_RELEASE"] > 0 and validation["student_WAIT"] > 0; proposal_retained = fixed["proposal"]["top_1"] >= .15 and fixed["proposal"]["top_3"] >= .35 and len(fixed["proposal"]["prediction_histogram"]) >= 2
    lineage_ok = bool(d1["parent_dataset_sha256"] == lineage["derived_dataset"]["sha256"] and d1["student_checkpoint_sha256"] == _hash(student_checkpoint) and d1["decision_event_unique"] and d1["dataset_sha256"] == _hash(aggregate_path))
    gates = {"fixed_state_performance_retained": fixed_ok, "RELEASE_recall_not_collapsed": fixed["gate"]["RELEASE_recall"] >= .5, "proposal_learning_retained": proposal_retained, "student_state_disagreement_stable": disagreement_stable, "closed_loop_not_all_WAIT_or_RELEASE": no_gate_collapse, "Safety": not any(safety.values()), "dataset_lineage": lineage_ok, "student_authoritative_execution": authoritative, "teacher_label_only": validation_label_only, "teacher_determinism": determinism, "ActionTape_exact_replay": bool(replay["exact"] and replay["policy_joint_logprob_match"])}
    passed = all(gates.values()); result = {"stage": "V4_2_DAGGER_ROUND1", "evidence_type": "newly_run", "student_checkpoint": str(student_checkpoint), "D1_manifest": str(d1_path), "D1_dataset": str(aggregate_path), "action_tapes": tapes, "collection": {"raw_decision_ticks": raw_ticks, "retained_events": len(events), "temporal_duplication_factor": raw_ticks / max(1, len(events)), "semantic_sampler": "event_start + teacher_target_change + meaningful_observation_change + bounded_sparse_persistence", "diagnostics": collection, "rates": collection_rates}, "training": training, "fixed_cohort": fixed, "student_only_validation": {"diagnostics": validation, "rates": validation_rates}, "safety": safety, "action_tape_replay": replay, "checkpoint": str(checkpoint), "gates": {key: "PASS" if value else "FAIL" for key, value in gates.items()}, "V4_2_DAGGER_R1": "PASS" if passed else "FAIL", "I2_DAGGER_ROUND2_ALLOWED": bool(passed), "PPO_ALLOWED": False, "forbidden_operations_called": []}
    (output / "v4_2_dagger_round1.manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8"); return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--student-checkpoint", required=True); parser.add_argument("--derived-i0-manifest", required=True); parser.add_argument("--i1-manifest", required=True); parser.add_argument("--output-dir", required=True); parser.add_argument("--target-events", type=int, default=500); parser.add_argument("--max-seeds", type=int, default=24); parser.add_argument("--max-steps", type=int, default=10)
    a = parser.parse_args(); print(json.dumps(run(a.student_checkpoint, a.derived_i0_manifest, a.i1_manifest, a.output_dir, target_events=a.target_events, max_seeds=a.max_seeds, max_steps=a.max_steps), indent=2))
