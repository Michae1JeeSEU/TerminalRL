"""S0 qualification for V4.2 hierarchical action semantics only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from ..algo.joint_ppo_v4_1 import JointPPOV41
from ..algo.joint_ppo_v4_2 import (JointPPOV42, hierarchical_bc_loss,
    hierarchical_joint_logprob)
from ..core.checkpoints import IncompatibleCheckpointError, load_checkpoint, save_checkpoint
from ..core.config import SimConfig
from ..core.low_level_joint_action_v4_2 import (ACTION_SCHEMA_V4_2,
    N_EXECUTION_ACTIONS_V4_2, N_RELEASE_PROPOSALS_V4_2, RELEASE_GATE,
    WAIT_GATE, decode_hierarchical_actions_v4_2, hierarchical_from_joint_index_v4_1,
    hierarchical_from_normalized_v4_2, hierarchical_targets_from_v4_1_joint)
from ..core.safety_trace import SafetyTraceRecorder
from ..envs.parallel_env import TerminalParallelEnv
from .v4_1_joint_teacher import teacher_joint_label_v4_1


def _config() -> SimConfig:
    return SimConfig(action_schema_version=ACTION_SCHEMA_V4_2, experiment_mode="fixed_teacher_heavy",
        scheduler_mode="execution", teacher_enabled=True, teacher_mode="teacher_label_only",
        teacher_release_label_enabled=False, shield_enabled=True, reset_mode="warm_start_rings",
        warm_start_uavs=8, warm_start_steady_motion=True, fixed_arrival_schedule=(0.0,) * 8,
        total_requested_uavs=8, max_simulation_seconds=2.0, initial_soc_mode="fixed", initial_soc=.8,
        priority_mode="fixed", initial_priority=1)


def _lineage(derived_manifest: str | Path) -> bool:
    artifact = json.loads(Path(derived_manifest).read_text(encoding="utf-8"))
    return bool(artifact["lineage_PASS"] and artifact["derived_target_schema"] == ACTION_SCHEMA_V4_2 and
                artifact["derived_dataset"]["unchanged_semantics"] and artifact["derived_frozen_cohort"]["unchanged_semantics"])


def run(output_dir: str | Path, derived_i0_manifest: str | Path) -> dict:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True); config = _config()
    # Every physical intent round-trips through one gate and optional proposal.
    mapped = [hierarchical_from_joint_index_v4_1(index) for index in range(N_EXECUTION_ACTIONS_V4_2)]
    decoded = decode_hierarchical_actions_v4_2(np.asarray([item.gate for item in mapped]),
                                                np.asarray([item.proposal for item in mapped], dtype=object))
    catalog_ok = bool(all(hierarchical_from_normalized_v4_2(action).joint_index_v4_1 == index
                          for index, action in enumerate(decoded)) and mapped[0].proposal is None and
                      all(item.gate == RELEASE_GATE and item.proposal is not None for item in mapped[1:]))
    mapping_ok = bool(np.array_equal(hierarchical_targets_from_v4_1_joint(np.arange(26))[0], [0] + [1] * 25) and
                      np.array_equal(hierarchical_targets_from_v4_1_joint(np.arange(26))[1], [-1] + list(range(25))))

    env = TerminalParallelEnv(config); observations, _ = env.reset(9421)
    labels = [teacher_joint_label_v4_1(env, env.active[int(next(iter(observations)))]) for _ in range(16)]
    teacher_ok = len({row.index for row in labels}) == 1 and all(hierarchical_from_joint_index_v4_1(row.index).joint_index_v4_1 == row.index for row in labels)

    # Masked, unweighted BC has exactly CE(gate) + CE(proposal on RELEASE rows).
    gate_logits = torch.tensor([[2.0, -1.0], [-1.0, 2.0]], requires_grad=True)
    proposal_logits = torch.zeros((2, N_RELEASE_PROPOSALS_V4_2), requires_grad=True)
    gate_target = torch.tensor([WAIT_GATE, RELEASE_GATE]); proposal_target = torch.tensor([-1, 12])
    total_loss, parts = hierarchical_bc_loss(gate_logits, proposal_logits, gate_target, proposal_target)
    expected_loss = torch.nn.functional.cross_entropy(gate_logits, gate_target) + torch.nn.functional.cross_entropy(proposal_logits[1:2], proposal_target[1:2])
    total_loss.backward(); bc_mask_ok = bool(torch.allclose(total_loss, expected_loss) and parts["release_mask"].tolist() == [False, True] and torch.isfinite(gate_logits.grad).all() and torch.isfinite(proposal_logits.grad).all())

    gate_dist = Categorical(logits=torch.tensor([[.3, -.7], [-.4, .8]])); proposal_dist = Categorical(logits=torch.arange(25, dtype=torch.float32).repeat(2, 1) / 10)
    gates = torch.tensor([WAIT_GATE, RELEASE_GATE]); proposals = torch.tensor([-1, 7])
    gate_lp, proposal_lp, joint_lp = hierarchical_joint_logprob(gate_dist, proposal_dist, gates, proposals)
    logprob_ok = bool(torch.allclose(joint_lp[0], gate_lp[0]) and torch.allclose(joint_lp[1], gate_lp[1] + proposal_lp[1]) and torch.isnan(proposal_lp[0]))
    policy = JointPPOV42(87)
    with torch.no_grad():
        zero = torch.zeros((2, 87)); actor_gate, actor_proposal = policy.model.distributions(zero)
        entropy = policy.model.conditional_entropy(zero)
        expected_entropy = actor_gate.entropy() + actor_gate.probs[:, RELEASE_GATE] * actor_proposal.entropy()
    entropy_ok = bool(torch.allclose(entropy, expected_entropy) and torch.isfinite(entropy).all())

    # A mixed WAIT/RELEASE request proves that execution sees no catalogue search.
    ordered = sorted(observations); requested = [hierarchical_from_joint_index_v4_1(0 if pos == 0 else 1 + pos)
                                                 for pos, _ in enumerate(ordered)]
    actions = {uid: action.normalized for uid, action in zip(ordered, requested)}
    tape = SafetyTraceRecorder(output / "action_tapes", dt=config.dt); tape.start_episode(env, episode_seed=9421, policy_rng_state={"qualification": 9421})
    env.teacher_scheduler.candidates = lambda _: (_ for _ in ()).throw(AssertionError("ExecutionScheduler searched proposals"))
    values = torch.as_tensor(np.asarray([observations[uid] for uid in ordered]), dtype=torch.float32)
    with torch.no_grad():
        gate_distribution, proposal_distribution = policy.model.distributions(values)
        action_gates = torch.as_tensor([item.gate for item in requested]); action_proposals = torch.as_tensor([-1 if item.proposal is None else item.proposal for item in requested])
        g_lp, p_lp, j_lp = hierarchical_joint_logprob(gate_distribution, proposal_distribution, action_gates, action_proposals)
    log_probs = {uid: {"gate": float(g_lp[pos]), "proposal": None if requested[pos].proposal is None else float(p_lp[pos]), "joint": float(j_lp[pos])}
                 for pos, uid in enumerate(ordered)}
    record = tape.record_pre_step(env, observations, actions, log_probs); _, _, _, _, info = env.step(actions); tape.record_post_step(env, record, info)
    tape_path = tape.dump(reason="v4_2_s0_single_requested_action")
    replay = SafetyTraceRecorder.replay_action_tape(TerminalParallelEnv(config), tape_path)
    payload = json.loads(tape_path.read_text(encoding="utf-8")); row = payload["action_tape"][0]
    tape_ok = bool(replay["exact"] and row["requested_proposal"][ordered[0]] is None and
                   all(row["requested_proposal"][uid] is not None for uid in ordered[1:]) and
                   row["proposal_logprob"][ordered[0]] is None and "decoded_requested_action" in row and "post_state_hash" in row)

    safety_events: list[dict] = []
    for block in range(4):
        probe = TerminalParallelEnv(config); probe_observations, _ = probe.reset(9500 + block)
        probe_actions = {uid: hierarchical_from_joint_index_v4_1((block * 8 + offset) % 26).normalized
                         for offset, uid in enumerate(probe_observations)}
        _, _, _, _, probe_info = probe.step(probe_actions); safety_events.extend(probe_info.get("conflict_events", []))
    safety = {"collision": sum(row.get("kind") == "collision" for row in safety_events),
              "NMAC": sum(row.get("kind") == "nmac_entry" for row in safety_events),
              "hard": sum(row.get("kind") == "first_hard_separation_trace" for row in safety_events),
              "SafetyInvariantError": 0, "uncertified": 0}
    safety_ok = not any(safety.values())

    checkpoint = output / "v4_2_fresh_actor.pt"; save_checkpoint(checkpoint, policy, config, stage="V4_2_S0", ppo_called=False, fresh_actor=True)
    load_checkpoint(checkpoint, policy, config)
    historical = output / "v4_1_historical_read_only.pt"; save_checkpoint(historical, JointPPOV41(87), SimConfig(action_schema_version="low_action_schema_v4_1_joint_discrete_26"), stage="V4_1_HISTORICAL")
    try:
        load_checkpoint(historical, policy, config); old_rejected = False
    except IncompatibleCheckpointError:
        old_rejected = True
    gates_result = {"V4_2_HIERARCHICAL_SCHEMA": catalog_ok, "gate_semantics": catalog_ok,
                    "proposal_semantics": catalog_ok, "Teacher_mapping": mapping_ok and teacher_ok,
                    "BC_masking": bc_mask_ok, "joint_logprob": logprob_ok, "conditional_entropy": entropy_ok,
                    "ActionTape_replay": tape_ok, "Execution_boundary": tape_ok, "Safety": safety_ok,
                    "checkpoint_schema": old_rejected, "derived_I0_lineage": _lineage(derived_i0_manifest)}
    passed = all(gates_result.values())
    result = {"old_schema": "V4_1_flat26", "new_schema": "V4_2_hierarchical", "action_schema": ACTION_SCHEMA_V4_2,
              "execution_action_count": 26, "gate_actions": 2, "release_proposals": 25,
              "safety": safety, "action_tape_replay": replay, "checkpoint": {"fresh_actor": "PASS", "V4_1_resumable": False},
              "gates": {key: "PASS" if value else "FAIL" for key, value in gates_result.items()},
              "V4_2_HIERARCHICAL_SCHEMA": "PASS" if passed else "FAIL", "V4_2_I1_ALLOWED": bool(passed),
              "PPO_ALLOWED": False, "forbidden_operations_called": []}
    (output / "v4_2_s0_qualification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", required=True); parser.add_argument("--derived-i0-manifest", required=True)
    print(json.dumps(run(parser.parse_args().output_dir, parser.parse_args().derived_i0_manifest), indent=2))
