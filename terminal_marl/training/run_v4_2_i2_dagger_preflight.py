"""Short, label-only V4.2 DAgger closed-loop preflight; no retraining or PPO."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ..algo.joint_ppo_v4_2 import JointPPOV42, hierarchical_joint_logprob
from ..core.checkpoints import load_checkpoint
from ..core.low_level_joint_action_v4_2 import ACTION_SCHEMA_V4_2, RELEASE_GATE, hierarchical_from_joint_index_v4_1
from ..core.runtime_traffic_safety import SafetyInvariantError
from ..core.safety_trace import SafetyTraceRecorder, _digest
from ..envs.parallel_env import TerminalParallelEnv
from .run_v4_2_hierarchical_s0_qualification import _config
from .v4_1_joint_teacher import teacher_joint_label_v4_1


def _hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def _policy_tape_logprob_exact(policy: JointPPOV42, config, tape_path: str | Path, *, atol: float = 1e-6) -> bool:
    """Recompute only requested-action likelihoods; never resample actions."""
    payload = json.loads(Path(tape_path).read_text(encoding="utf-8")); env = TerminalParallelEnv(config)
    observations, _ = env.reset(int(payload["episode_seed"]))
    for row in payload["action_tape"]:
        uids = sorted(row["a_requested"])
        values = torch.as_tensor(np.asarray([observations[uid] for uid in uids], dtype=np.float32))
        gates = torch.as_tensor([row["requested_gate"][uid] for uid in uids], dtype=torch.long)
        proposals = torch.as_tensor([-1 if row["requested_proposal"][uid] is None else row["requested_proposal"][uid] for uid in uids], dtype=torch.long)
        with torch.no_grad():
            gate_distribution, proposal_distribution = policy.model.distributions(values)
            gate_lp, proposal_lp, joint_lp = hierarchical_joint_logprob(gate_distribution, proposal_distribution, gates, proposals)
        for offset, uid in enumerate(uids):
            if abs(float(gate_lp[offset]) - float(row["gate_logprob"][uid])) > atol or abs(float(joint_lp[offset]) - float(row["joint_log_prob"][uid])) > atol:
                return False
            expected = row["proposal_logprob"].get(uid)
            if expected is None:
                if not bool(torch.isnan(proposal_lp[offset])):
                    return False
            elif abs(float(proposal_lp[offset]) - float(expected)) > atol:
                return False
        actions = {uid: np.asarray(value, dtype=np.float32) for uid, value in row["a_requested"].items()}
        observations, *_ = env.step(actions)
    return True


def _config_for_preflight(max_steps: int):
    config = _config(); config.max_simulation_seconds = float(max_steps); return config


def run(student_checkpoint: str | Path, derived_i0_manifest: str | Path, output_dir: str | Path, *, seeds: tuple[int, ...] = (9701, 9702), max_steps: int = 6) -> dict:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    lineage = json.loads(Path(derived_i0_manifest).read_text(encoding="utf-8"))
    if lineage.get("derived_target_schema") != ACTION_SCHEMA_V4_2 or not lineage.get("lineage_PASS"):
        raise ValueError("I2 requires a lineage-PASS V4.2 derived I0 view")
    config = _config_for_preflight(max_steps); policy = JointPPOV42(87); payload = load_checkpoint(student_checkpoint, policy, config, load_optimizer=False)
    if payload.get("stage") != "I1_V4_2_HIERARCHICAL_BC" or not payload.get("fresh_actor") or payload.get("ppo_called"):
        raise ValueError("I2 requires the fresh PPO-free V4.2 I1 checkpoint")
    policy.model.eval(); new_events: list[dict] = []; tapes: list[str] = []; replay_results: list[dict] = []
    diagnostics = {"student_WAIT": 0, "student_RELEASE": 0, "teacher_WAIT": 0, "teacher_RELEASE": 0,
                   "gate_disagreement": 0, "proposal_disagreement_on_teacher_RELEASE": 0, "shield_used": 0,
                   "completion": 0, "throughput": 0, "queue": [], "student_proposal_histogram": {}}
    safety = {"collision": 0, "NMAC": 0, "hard": 0, "SafetyInvariantError": 0, "uncertified": 0}
    authoritative = True; label_only = True; determinism = True; episode_id = 0
    for seed in seeds:
        env = TerminalParallelEnv(config); observations, _ = env.reset(seed)
        tape = SafetyTraceRecorder(output / "action_tapes", dt=config.dt); tape.start_episode(env, episode_seed=seed, policy_rng_state={"student_checkpoint_sha256": _hash(student_checkpoint), "seed": seed})
        for _ in range(max_steps):
            if not observations:
                break
            uids = sorted(observations); values = np.asarray([observations[uid] for uid in uids], dtype=np.float32)
            batch = policy.act_details(values, values, deterministic=False)
            requested = {uid: batch.decoded[position] for position, uid in enumerate(uids)}
            labels: dict[str, dict] = {}; state_hash = _digest(SafetyTraceRecorder._state(env))
            for position, uid in enumerate(uids):
                uav = env.active[int(uid)]; first = teacher_joint_label_v4_1(env, uav); second = teacher_joint_label_v4_1(env, uav)
                target = hierarchical_from_joint_index_v4_1(first.index); determinism = determinism and first.index == second.index
                labels[uid] = {"gate": target.gate, "proposal": target.proposal, "joint_index_v4_1": target.joint_index_v4_1}
                lifecycle = f"i2_preflight_student_closed_loop:seed-{seed}:episode-{episode_id}:uav-{uid}:phase-{uav.phase.value}"
                student_gate, student_proposal = int(batch.requested_gate[position]), batch.requested_proposal[position]
                diagnostics["student_WAIT" if student_gate == 0 else "student_RELEASE"] += 1
                diagnostics["teacher_WAIT" if target.gate == 0 else "teacher_RELEASE"] += 1
                diagnostics["gate_disagreement"] += int(student_gate != target.gate)
                if target.gate == RELEASE_GATE:
                    diagnostics["proposal_disagreement_on_teacher_RELEASE"] += int(student_proposal != target.proposal)
                if student_proposal is not None:
                    histogram = diagnostics["student_proposal_histogram"]; histogram[str(student_proposal)] = histogram.get(str(student_proposal), 0) + 1
                new_events.append({"schema_version": "v4_2_i2_dagger_preflight_event_v1", "action_schema": ACTION_SCHEMA_V4_2,
                                   "round": "preflight", "observation": values[position].tolist(), "requested_gate": student_gate,
                                   "requested_proposal": student_proposal, "requested_joint_logprob": float(batch.joint_logprob[position]),
                                   "teacher_gate": target.gate, "teacher_proposal": target.proposal, "teacher_semantic_joint_index_v4_1": target.joint_index_v4_1,
                                   "decoded_requested_action": batch.decoded[position].tolist(), "source": "student_closed_loop_preflight",
                                   "seed": seed, "episode_id": episode_id, "uav_id": uid, "lifecycle_event_id": lifecycle,
                                   "decision_event_id": f"{lifecycle}:step-{env.step_count}", "obs_hash": _digest(values[position]), "state_hash": state_hash,
                                   "teacher_deterministic": first.index == second.index, "teacher_label_only": True})
            log_probs = {uid: {"gate": float(batch.gate_logprob[position]), "proposal": batch.proposal_logprob[position], "joint": float(batch.joint_logprob[position])}
                         for position, uid in enumerate(uids)}
            record = tape.record_pre_step(env, observations, requested, log_probs)
            try:
                observations, _, _, _, info = env.step(requested)
            except SafetyInvariantError as error:
                safety["SafetyInvariantError"] += 1; tape.record_safety_invariant(record, error); break
            tape.record_post_step(env, record, info); tape.record_teacher_labels(record, labels)
            post_hash = _digest(SafetyTraceRecorder._state(env))
            for event in new_events[-len(uids):]:
                trace = info["action_traces"][event["uav_id"]]
                event.update({"applied_action": trace.get("execution_applied_action"), "shield_action": trace.get("shield_applied_action"),
                              "post_state_hash": post_hash, "teacher_did_not_replace_execution": not trace.get("teacher_used", False),
                              "requested_action_matches_student": np.allclose(np.asarray(trace["requested_low_action"]), np.asarray(event["decoded_requested_action"]))})
                authoritative = authoritative and event["requested_action_matches_student"]
                label_only = label_only and event["teacher_did_not_replace_execution"] and event["teacher_label_only"]
                diagnostics["shield_used"] += int(bool(trace.get("shield_used")))
            diagnostics["queue"].append(int(info["queue_size"]))
        diagnostics["completion"] += len(env.completed_uavs); diagnostics["throughput"] += len(env.completed_uavs)
        safety["collision"] += int(env.telemetry.get("collisions", 0)); safety["NMAC"] += int(env.telemetry.get("nmac", 0)); safety["hard"] += int(env.telemetry.get("hard_separation_violations", 0))
        tape_path = tape.dump(reason="v4_2_i2_dagger_preflight"); tapes.append(str(tape_path))
        replay = SafetyTraceRecorder.replay_action_tape(TerminalParallelEnv(config), tape_path)
        replay["policy_joint_logprob_match"] = _policy_tape_logprob_exact(policy, config, tape_path)
        replay_results.append(replay); episode_id += 1
    all_ids = [row["decision_event_id"] for row in new_events]; identity = len(all_ids) == len(set(all_ids))
    parent_rows = _read_jsonl(lineage["derived_dataset"]["path"]); aggregate = parent_rows + new_events
    aggregate_path = output / "D1_preflight.jsonl"; aggregate_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in aggregate), encoding="utf-8")
    lineage_manifest = {"stage": "D1_V4_2_DAGGER_PREFLIGHT", "round": "preflight", "action_schema": ACTION_SCHEMA_V4_2,
                        "parent_dataset_sha256": lineage["derived_dataset"]["sha256"], "parent_rows": len(parent_rows),
                        "student_checkpoint_sha256": _hash(student_checkpoint), "teacher_version": "v4_1_joint_teacher_semantic_unmodified",
                        "source_tags": sorted(set(row.get("source", "") for row in new_events)), "new_student_visited_events": len(new_events),
                        "decision_event_unique": identity, "aggregate_rows": len(aggregate), "dataset_sha256": _hash(aggregate_path),
                        "PPO_ALLOWED": False, "formal_retraining_called": False}
    lineage_path = output / "D1_preflight.manifest.json"; lineage_path.write_text(json.dumps(lineage_manifest, indent=2), encoding="utf-8")
    action_provenance = all(np.isfinite(row["requested_joint_logprob"]) and (row["requested_proposal"] is not None or row["requested_gate"] == 0) for row in new_events)
    replay_ok = bool(replay_results) and all(row["exact"] and row["policy_joint_logprob_match"] for row in replay_results)
    lineage_ok = bool(lineage_manifest["parent_dataset_sha256"] == lineage["derived_dataset"]["sha256"] and
                      lineage_manifest["student_checkpoint_sha256"] == _hash(student_checkpoint) and
                      lineage_manifest["source_tags"] == ["student_closed_loop_preflight"] and
                      lineage_manifest["dataset_sha256"] == _hash(aggregate_path) and lineage_manifest["decision_event_unique"])
    gates = {"student_authoritative_execution": authoritative, "teacher_label_only": label_only,
             "hierarchical_action_provenance": action_provenance, "dagger_dataset_lineage": lineage_ok,
             "decision_event_identity": identity, "teacher_determinism": determinism, "ActionTape_exact_replay": replay_ok,
             "Safety": not any(safety.values())}
    passed = all(gates.values())
    result = {"stage": "I2_V4_2_DAGGER_PREFLIGHT", "evidence_type": "newly_run", "student_checkpoint": str(student_checkpoint),
              "derived_i0_manifest": str(derived_i0_manifest), "D1_preflight_manifest": str(lineage_path), "action_tapes": tapes,
              "seeds": list(seeds), "max_steps": max_steps, "diagnostics": {**diagnostics, "all_WAIT": diagnostics["student_RELEASE"] == 0,
              "all_RELEASE": diagnostics["student_WAIT"] == 0, "single_proposal_collapse": len(diagnostics["student_proposal_histogram"]) <= 1},
              "safety": safety, "action_tape_replay": replay_results, "gates": {key: "PASS" if value else "FAIL" for key, value in gates.items()},
              "V4_2_I2_DAGGER_PREFLIGHT": "PASS" if passed else "FAIL", "I2_DAGGER_ROUND1_ALLOWED": bool(passed),
              "PPO_ALLOWED": False, "forbidden_operations_called": []}
    (output / "v4_2_i2_dagger_preflight.manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--student-checkpoint", required=True); parser.add_argument("--derived-i0-manifest", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", default="9701,9702"); parser.add_argument("--max-steps", type=int, default=6); arguments = parser.parse_args()
    print(json.dumps(run(arguments.student_checkpoint, arguments.derived_i0_manifest, arguments.output_dir, seeds=tuple(int(value) for value in arguments.seeds.split(",")), max_steps=arguments.max_steps), indent=2))
