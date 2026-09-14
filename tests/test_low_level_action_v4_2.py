import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch.distributions import Categorical

from terminal_marl.algo.joint_ppo_v4_1 import JointPPOV41
from terminal_marl.algo.joint_ppo_v4_2 import (JointPPOV42, hierarchical_bc_loss,
    hierarchical_joint_logprob)
from terminal_marl.core.checkpoints import IncompatibleCheckpointError, load_checkpoint, save_checkpoint
from terminal_marl.core.config import SimConfig
from terminal_marl.core.observations import (LOW_OBS_V2, LOW_OBS_V2_1,
    build_low_observation_v2, build_low_observation_v2_1)
from terminal_marl.core.low_level_joint_action_v4_2 import (ACTION_SCHEMA_V4_2,
    RELEASE_GATE, WAIT_GATE, decode_hierarchical_actions_v4_2,
    hierarchical_from_joint_index_v4_1, hierarchical_from_normalized_v4_2,
    hierarchical_targets_from_v4_1_joint)
from terminal_marl.core.safety_trace import SafetyTraceRecorder
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training.v4_1_joint_teacher import teacher_joint_label_v4_1
from terminal_marl.training.run_v4_2_dagger_round1 import _semantic_reason
from terminal_marl.training.run_v4_2_dagger_r1_release_retention_diagnostic import _root_cause
from terminal_marl.training.run_v4_2_dagger_d1_conflict_source_attribution import _family
from terminal_marl.training.run_v4_2_transition_wait_observability_audit import _observation_mapping
from terminal_marl.training.run_v4_2_observation_v2_1_qualification import (
    OBSERVATION_V2_1, audit_retry_state_provenance)
from terminal_marl.training.run_v4_2_observation_v2_1_teacher_predicate_closure_audit import (
    _predicate_inventory)
from terminal_marl.training.run_v4_2_privileged_teacher_gate_factorization_diagnostic import (
    _closure_reclassification)
from terminal_marl.training.run_v4_2_v2_1_confirmatory_closure_audit import (
    SOURCE_SPECS, _source_config, _state_only_pass)
from terminal_marl.training.run_v4_2_v2_1_retry_closure_diagnostic import _powered_protocol
from terminal_marl.training.run_v4_2_v2_1_powered_confirmation import (
    MINIMUM_SUPPORT as POWERED_MINIMUM_SUPPORT, SOURCE_SPECS as POWERED_SOURCE_SPECS,
    _retry_sanity_pass, _support_reached,
)
from terminal_marl.training.run_v4_2_v2_1_formal_i0 import (
    FROZEN_TARGET as FORMAL_V2_1_FROZEN_TARGET, OBSERVATION_SCHEMA as FORMAL_V2_1_OBSERVATION,
    SOURCES as FORMAL_V2_1_SOURCES, TRAIN_TARGET as FORMAL_V2_1_TRAIN_TARGET, _config as _formal_config,
    _source_reset_recipe,
)
from terminal_marl.training.run_v4_2_v2_1_i0_recollection_preflight import (
    DYNAMIC_TAPES_PER_FAMILY, FAMILY_QUOTAS,
)
from terminal_marl.training.run_v4_2_v2_1_release_manifold_audit import (
    PREFERENCE_RANK, _canonical_preference_catalogue, _proposal_row,
)
from terminal_marl.training.run_v4_2_v2_1_predicate_release_reachability import (
    classify_family_flags, validate_recipe,
)
from terminal_marl.training.run_v4_2_v2_1_teacher_selected_support_closure import (
    STABLE_MINIMUM, STRATA as SUPPORT_STRATA, classify_selected_support,
)
from terminal_marl.training.run_v4_2_v2_1_formal_i0_r import (
    FAMILY_MINIMUMS as FORMAL_I0_R_FAMILY_MINIMUMS,
    FROZEN_MAX_ROWS as FORMAL_I0_R_FROZEN_MAX_ROWS,
    RELEASE_RICH_CONTINUATION, REQUIRED_SELECTED_SUPPORT,
    STRATA as FORMAL_I0_R_STRATA, _scheduled_stratum,
)
from terminal_marl.training.run_v4_2_v2_1_i1_bc import (
    EXPECTED_FROZEN_SHA256, EXPECTED_I0_MANIFEST_SHA256, EXPECTED_TRAIN_SHA256,
    STABLE_JOINT_SUPPORT as FORMAL_I1_STABLE_SUPPORT,
    _architecture_fingerprint, _arrays as _formal_i1_arrays,
    load_formal_i1_checkpoint,
)
from terminal_marl.training.run_v4_2_v2_1_i1_q_cohort import (
    MAX_ROWS as I1_Q_MAX_ROWS, MINIMUMS as I1_Q_MINIMUMS,
    SEED_END as I1_Q_SEED_END, SEED_START as I1_Q_SEED_START,
    STRATA as I1_Q_STRATA,
)


def _config():
    return SimConfig(action_schema_version=ACTION_SCHEMA_V4_2, teacher_enabled=True,
        teacher_mode="teacher_label_only", scheduler_mode="execution", experiment_mode="fixed_teacher_heavy",
        reset_mode="warm_start_rings", warm_start_uavs=8, warm_start_steady_motion=True,
        fixed_arrival_schedule=(0.0,) * 8, total_requested_uavs=8, max_simulation_seconds=2.0)


def test_v4_2_preserves_all_26_execution_decodes_and_has_no_wait_proposal():
    actions = [hierarchical_from_joint_index_v4_1(index) for index in range(26)]
    decoded = decode_hierarchical_actions_v4_2([row.gate for row in actions], [row.proposal for row in actions])
    assert actions[0].gate == WAIT_GATE and actions[0].proposal is None
    assert all(row.gate == RELEASE_GATE and row.proposal is not None for row in actions[1:])
    assert [hierarchical_from_normalized_v4_2(row).joint_index_v4_1 for row in decoded] == list(range(26))
    with pytest.raises(ValueError, match="WAIT proposal"):
        decode_hierarchical_actions_v4_2([WAIT_GATE], [0])


def test_v4_2_teacher_encoding_bc_mask_logprob_and_conditional_entropy():
    gate_target, proposal_target = hierarchical_targets_from_v4_1_joint(np.arange(26))
    assert gate_target.tolist() == [0] + [1] * 25 and proposal_target.tolist() == [-1] + list(range(25))
    gate_logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True); proposal_logits = torch.zeros((2, 25), requires_grad=True)
    loss, parts = hierarchical_bc_loss(gate_logits, proposal_logits, torch.tensor([0, 1]), torch.tensor([-1, 4]))
    expected = torch.nn.functional.cross_entropy(gate_logits, torch.tensor([0, 1])) + torch.nn.functional.cross_entropy(proposal_logits[1:], torch.tensor([4]))
    assert torch.allclose(loss, expected) and parts["release_mask"].tolist() == [False, True]
    gd = Categorical(logits=torch.tensor([[1.0, 0.0], [0.0, 1.0]])); pd = Categorical(logits=torch.zeros((2, 25)))
    gate_lp, proposal_lp, joint_lp = hierarchical_joint_logprob(gd, pd, torch.tensor([0, 1]), torch.tensor([-1, 4]))
    assert torch.allclose(joint_lp[0], gate_lp[0]) and torch.allclose(joint_lp[1], gate_lp[1] + proposal_lp[1]) and torch.isnan(proposal_lp[0])
    policy = JointPPOV42(4); observations = torch.zeros((2, 4)); gate, proposal = policy.model.distributions(observations)
    assert torch.allclose(policy.model.conditional_entropy(observations), gate.entropy() + gate.probs[:, 1] * proposal.entropy())
    with torch.no_grad():
        policy.model.gate_head.weight.zero_(); policy.model.gate_head.bias[:] = torch.tensor([10.0, -10.0])
    waiting = policy.act_details(np.zeros((2, 4), np.float32), np.zeros((2, 4), np.float32), deterministic=True)
    assert waiting.requested_proposal == (None, None)
    with torch.no_grad():
        policy.model.gate_head.bias[:] = torch.tensor([-10.0, 10.0])
    releasing = policy.act_details(np.zeros((2, 4), np.float32), np.zeros((2, 4), np.float32), deterministic=True)
    assert all(value is not None for value in releasing.requested_proposal)


def test_v4_2_tape_replay_and_checkpoint_boundary(tmp_path):
    config = _config(); env = TerminalParallelEnv(config); observations, _ = env.reset(61); ordered = sorted(observations)
    requested = [hierarchical_from_joint_index_v4_1(0 if index == 0 else 13) for index, _ in enumerate(ordered)]
    teacher_labels = {uid: hierarchical_from_joint_index_v4_1(teacher_joint_label_v4_1(env, env.active[int(uid)]).index)
                      for uid in ordered}
    tape = SafetyTraceRecorder(tmp_path, dt=config.dt); tape.start_episode(env, episode_seed=61, policy_rng_state={})
    env.teacher_scheduler.candidates = lambda _: (_ for _ in ()).throw(AssertionError("runtime search"))
    actions = {uid: value.normalized for uid, value in zip(ordered, requested)}
    probs = {uid: {"gate": -.5, "proposal": None if value.proposal is None else -.7, "joint": -1.2} for uid, value in zip(ordered, requested)}
    record = tape.record_pre_step(env, observations, actions, probs); _, _, _, _, info = env.step(actions); tape.record_post_step(env, record, info)
    tape.record_teacher_labels(record, {uid: {"gate": label.gate, "proposal": label.proposal, "joint_index_v4_1": label.joint_index_v4_1}
                                        for uid, label in teacher_labels.items()})
    path = tape.dump(reason="test_v4_2"); payload = json.loads(path.read_text()); row = payload["action_tape"][0]
    assert row["action_schema_version"] == ACTION_SCHEMA_V4_2 and row["requested_proposal"][ordered[0]] is None
    replay = SafetyTraceRecorder.replay_action_tape(TerminalParallelEnv(config), path)
    assert replay["exact"] and replay["steps"][0]["teacher_label_match"]
    policy = JointPPOV42(4); checkpoint = tmp_path / "v4_2.pt"; save_checkpoint(checkpoint, policy, config); load_checkpoint(checkpoint, policy, config)
    old = tmp_path / "v4_1.pt"; save_checkpoint(old, JointPPOV41(4), SimConfig(action_schema_version="low_action_schema_v4_1_joint_discrete_26"))
    with pytest.raises(IncompatibleCheckpointError):
        load_checkpoint(old, policy, config)
    uav = env.active[int(ordered[0])]
    assert teacher_joint_label_v4_1(env, uav).index in range(26)


def test_v4_2_dagger_sampler_retains_semantic_changes_not_every_raw_tick():
    memory = {}; observation = np.zeros(87, dtype=np.float32)
    assert _semantic_reason(memory, (1, "RING_OUTER"), teacher_joint=0, observation=observation, tick=0) == "event_start"
    assert _semantic_reason(memory, (1, "RING_OUTER"), teacher_joint=0, observation=observation, tick=1) is None
    assert _semantic_reason(memory, (1, "RING_OUTER"), teacher_joint=0, observation=observation, tick=3) == "bounded_sparse_persistence"
    assert _semantic_reason(memory, (1, "RING_OUTER"), teacher_joint=13, observation=observation, tick=4) == "teacher_target_change"


def test_v4_2_r1_diagnostic_root_cause_does_not_promote_a_passing_epoch():
    def point(epoch, release, disagreement):
        return {"epoch": epoch, "frozen_cohort": {"gate": {"RELEASE_recall": release}},
                "D1_teacher_disagreement": {"decoded_action_disagreement": disagreement}}
    trajectory = [point(0, .7361, .50), point(2, .70, .40), point(35, .6806, .30)]
    seed_rows = [{"RELEASE_recall": value} for value in (.6806, .67, .69, .68, .67)]
    root, evidence = _root_cause(trajectory, seed_rows, i1_release_recall=.7361)
    assert root == "OVER_FINETUNING" and evidence["early_pass_with_lower_D1_disagreement"]


def test_v4_2_d1_attribution_family_partition_is_semantic_and_exclusive():
    def row(phase, gate, reason):
        return {"lifecycle_event_id": f"dagger:phase-{phase}", "gate_target": gate, "semantic_sampling_reason": reason}
    assert _family(row("ring_outer", 1, "event_start")) == "clear_outer_RELEASE"
    assert _family(row("ring_middle", 1, "event_start")) == "middle_or_interface_RELEASE"
    assert _family(row("ring_middle", 0, "bounded_sparse_persistence")) == "persistent_middle_WAIT"
    assert _family(row("ring_outer", 0, "bounded_sparse_persistence")) == "persistent_other_WAIT"
    assert _family(row("ring_outer", 0, "teacher_target_change")) == "transition_WAIT"


def test_v4_2_observability_audit_keeps_retry_and_history_out_of_v2_features():
    status = _observation_mapping()["teacher_variable_status"]
    assert status["safety_retry_after_or_retry_remaining"] == "absent"
    assert status["previous_gate_or_action"] == "absent"
    assert status["target_predecessor_successor"] == "present"


def test_observation_v2_1_appends_only_exact_retry_state_and_rejects_short_horizon():
    config = _config(); config.observation_schema_version = OBSERVATION_V2_1
    env = TerminalParallelEnv(config); env.reset(73)
    uav = next(iter(env.active.values()))
    uav.safety_retry_after = env.now + config.retry_norm_horizon_seconds / 2.0
    v2, v21 = build_low_observation_v2(uav, env), build_low_observation_v2_1(uav, env)
    assert LOW_OBS_V2.dimension == 87 and LOW_OBS_V2_1.dimension == 88
    assert np.array_equal(v2, v21[:87]) and v21[87] == pytest.approx(.5)
    uav.safety_retry_after = env.now
    assert build_low_observation_v2_1(uav, env)[87] == 0.0
    with pytest.raises(ValueError, match="retry normalization horizon"):
        SimConfig(observation_schema_version=OBSERVATION_V2_1, retry_norm_horizon_seconds=29.0)


def test_observation_v2_1_provenance_fails_closed_without_exact_snapshot_retry(tmp_path):
    row = {"decision_event_id": "test:0", "uav_id": 7, "state_snapshot": {"time": 2.0, "uavs": [{"uid": 7}]}}
    dataset, cohort = tmp_path / "train.jsonl", tmp_path / "cohort.jsonl"
    dataset.write_text(json.dumps(row) + "\n", encoding="utf-8")
    cohort.write_text(json.dumps(row) + "\n", encoding="utf-8")
    manifest = tmp_path / "parent.json"; manifest.write_text(json.dumps({
        "derived_dataset": {"path": str(dataset)}, "derived_frozen_cohort": {"path": str(cohort)},
    }), encoding="utf-8")
    result = audit_retry_state_provenance(manifest)
    assert result["DATASET_V2_1_DERIVATION"] == "FAIL"
    assert result["splits"]["train"]["missing_by_reason"] == {"missing_safety_retry_after": 1}


def test_v2_1_teacher_predicate_inventory_separates_state_and_action_conditioned_inputs():
    inventory = {row["name"]: row for row in _predicate_inventory()}
    assert inventory["retry_remaining"]["class"] == "A"
    assert inventory["retry_remaining"]["v2_1_status"] == "direct:index_87"
    assert inventory["proposal_reservation_feasibility"]["class"] == "B"
    assert inventory["candidate_safety_certificate_and_holder_tube"]["v2_1_status"] == "ACTION_CONDITIONED"
    assert inventory["last_release_or_safety_decision_and_lifecycle_age"]["class"] == "C"


def test_v2_1_closure_reclassification_keeps_proposal_verdicts_out_of_state_only_gate():
    result = _closure_reclassification()
    assert result["reservation"]["classification"] == "ACTION_CONDITIONED"
    assert result["interface"]["classification"] == "STATE_ONLY"
    assert result["headway"]["classification"] == "ACTION_CONDITIONED"
    assert "proposal reservation" in result["revised_closure_requirements"]["action_conditioned"]


def test_v2_1_confirmatory_sources_preserve_runtime_schemas_and_closure_fails_closed():
    for _, dynamic, warm_start_uavs in SOURCE_SPECS:
        config = _source_config(dynamic=dynamic, warm_start_uavs=warm_start_uavs)
        assert config.action_schema_version == ACTION_SCHEMA_V4_2
        assert config.observation_schema_version == "phaseA_low_obs_v2_1_retry_state"
        assert config.teacher_mode == "teacher_label_only"
        assert config.total_requested_uavs == warm_start_uavs
    coverage = {"all_effective_state_only_predicates_covered": True, "unresolved_state_only_missing_features": []}
    assert _state_only_pass(coverage, {"retry": {"status": "PASS"}})
    assert not _state_only_pass(coverage, {"retry": {"status": "NOT_AVAILABLE"}})


def test_v2_1_powered_confirmation_protocol_freezes_counts_without_authorizing_collection():
    protocol = _powered_protocol()
    assert protocol["status"] == "FROZEN_PREPARATION_ONLY"
    assert protocol["minimum_counts"] == {"train_Teacher_RELEASE": 100, "dev_Teacher_RELEASE": 40, "dev_retry_active_WAIT": 40}
    assert protocol["no_collection_or_promotion_this_turn"]


def test_v2_1_powered_confirmation_uses_preregistered_support_not_target_selection():
    assert POWERED_MINIMUM_SUPPORT == {"train_Teacher_RELEASE": 100, "dev_Teacher_RELEASE": 40, "dev_retry_active_WAIT": 40}
    assert {spec[0] for spec in POWERED_SOURCE_SPECS} == {
        "powered_release_ready_low_traffic", "powered_clear_outer_release", "powered_middle_interface_release",
        "powered_transition_retry", "powered_mixed_traffic", "powered_resource_pressure",
    }
    assert not _support_reached({"train_Teacher_RELEASE": 100, "dev_Teacher_RELEASE": 39, "dev_retry_active_WAIT": 40})
    assert _support_reached(POWERED_MINIMUM_SUPPORT)
    assert _retry_sanity_pass({"feature_sanity": {"no_future_leakage": True, "replay_exact": True,
                                                       "finite": True, "in_unit_interval": True, "deterministic": True}})


def test_v2_1_formal_i0_protocol_is_fresh_88d_hierarchical_and_sized_before_collection():
    assert FORMAL_V2_1_OBSERVATION == "phaseA_low_obs_v2_1_retry_state"
    assert (FORMAL_V2_1_TRAIN_TARGET, FORMAL_V2_1_FROZEN_TARGET) == (1800, 300)
    assert {source.source for source in FORMAL_V2_1_SOURCES} >= {
        "clear_release_opportunity", "transition_retry_wait", "post_conflict_recovery",
        "post_wait_accumulated_release", "interface_headway", "resource_final_pressure",
    }
    assert all(source.max_steps == 12 for source in FORMAL_V2_1_SOURCES)


def test_dynamic_collector_reset_recipe_replays_the_post_setup_environment(tmp_path):
    spec = next(item for item in FORMAL_V2_1_SOURCES if item.dynamic)
    config = _formal_config(spec); env = TerminalParallelEnv(config); env.reset(81001)
    recipe, observations = _source_reset_recipe(env, spec)
    tape = SafetyTraceRecorder(tmp_path, dt=config.dt)
    tape.start_episode(env, episode_seed=81001, policy_rng_state={}, reset_recipe=recipe)
    actions = {uid: hierarchical_from_joint_index_v4_1(0).normalized for uid in observations}
    labels = {}
    for uid in actions:
        target = hierarchical_from_joint_index_v4_1(teacher_joint_label_v4_1(env, env.active[int(uid)]).index)
        labels[uid] = {"gate": target.gate, "proposal": target.proposal, "joint_index_v4_1": target.joint_index_v4_1}
    record = tape.record_pre_step(env, observations, actions); _, _, _, _, info = env.step(actions)
    tape.record_post_step(env, record, info); tape.record_teacher_labels(record, labels)
    replay = SafetyTraceRecorder.replay_action_tape(TerminalParallelEnv(config), tape.dump(reason="dynamic_recipe"))
    assert recipe["kind"] == "high_level_action_after_reset" and replay["exact"]
    assert replay["base_reset_state_match"] and replay["reset_recipe_match"]


def test_collector_repair_preflight_freezes_dynamic_replay_and_natural_family_quotas():
    assert DYNAMIC_TAPES_PER_FAMILY == 10
    assert FAMILY_QUOTAS == {"clear_outer_RELEASE": 15, "middle_interface_RELEASE": 15,
                             "post_conflict_RELEASE": 10, "post_WAIT_RELEASE": 10}


def test_release_manifold_audit_uses_proposal_ids_not_teacher_audit_positions():
    assert _proposal_row(10) == {"proposal_id": 10, "joint_index": 11,
                                 "speed_residual": 0.0, "merge_residual": -1.0}
    catalogue = _canonical_preference_catalogue()
    assert len(catalogue) == 25
    assert sorted(PREFERENCE_RANK.values()) == list(range(25))
    assert {row["proposal_id"] for row in catalogue} == set(range(25))


def test_predicate_release_families_are_source_independent_and_allow_overlap():
    previous = {"uid": 7, "lifecycle_id": "seed-1:uid-7", "teacher_gate": 0,
                "wait_rationale": "geometry_conflict", "time": 2.0, "decision_event_id": "wait"}
    current = {"uid": 7, "lifecycle_id": "seed-1:uid-7", "teacher_gate": 1,
               "retry_active": False, "committed": False, "interface_open": True,
               "resource_valid": True, "conflict_state": False, "phase": "ring_middle"}
    flags = classify_family_flags(current, previous)
    assert flags == {"clear_outer": False, "middle_interface": True, "post_wait": True, "post_conflict": True}
    assert validate_recipe("post_conflict_RELEASE", current, previous, flags)["predicate_exact"]
    # Collection source is deliberately not an argument to the classifier.
    assert classify_family_flags(dict(current, collection_source="expert_rollout"), previous) == flags


def test_predicate_release_families_reject_lifecycle_and_uid_breaks():
    previous = {"uid": 7, "lifecycle_id": "episode-a:uid-7", "teacher_gate": 0,
                "wait_rationale": "geometry_conflict", "time": 2.0, "decision_event_id": "wait"}
    base = {"uid": 7, "lifecycle_id": "episode-a:uid-7", "teacher_gate": 1,
            "retry_active": False, "committed": False, "interface_open": True,
            "resource_valid": True, "conflict_state": False, "phase": "ring_outer"}
    assert classify_family_flags(dict(base, lifecycle_id="episode-b:uid-7"), previous)["post_wait"] is False
    assert classify_family_flags(dict(base, uid=8, lifecycle_id="episode-a:uid-8"), previous)["post_conflict"] is False
    # A reset starts a new lifecycle even if a later simulator UID is reused.
    assert classify_family_flags(dict(base, lifecycle_id="reset-episode:uid-7"), previous)["post_wait"] is False
    assert classify_family_flags(dict(base, teacher_gate=0), previous)["clear_outer"] is False


def test_teacher_selected_support_requires_independent_seed_and_lifecycle_evidence():
    stable = [{"teacher_joint_index": 12, "seed": seed, "lifecycle_id": f"s{seed}:u1"}
              for seed in range(STABLE_MINIMUM)]
    rare = [{"teacher_joint_index": 14, "seed": 99, "lifecycle_id": "s99:u1"}]
    result = classify_selected_support(stable + rare)
    assert result["stable"] == [12]
    assert result["rare"] == [14]
    assert 13 in result["not_observed"]
    duplicated = [{"teacher_joint_index": 11, "seed": 1, "lifecycle_id": "same"}
                  for _ in range(STABLE_MINIMUM)]
    assert classify_selected_support(duplicated)["rare"] == [11]


def test_support_closure_strata_are_new_and_include_targeted_and_nominal_contexts():
    names = {row.name for row in SUPPORT_STRATA}
    assert {"target_proposal_12_context", "target_proposal_14_context",
            "nominal_mixed_traffic", "release_ready_low_traffic", "resource_pressure",
            "interface_transition", "ordinary_expert_rollout"} == names
    seeds = [seed for row in SUPPORT_STRATA for seed in range(row.seed_start, row.seed_start + row.seed_count)]
    assert len(seeds) == len(set(seeds))


def test_formal_i0_r_freezes_approved_support_and_whole_episode_continuation():
    assert REQUIRED_SELECTED_SUPPORT == (11, 12, 13, 14)
    assert FORMAL_I0_R_FAMILY_MINIMUMS == {
        "clear_outer": 15, "middle_interface": 15,
        "post_wait": 10, "post_conflict": 10,
    }
    assert FORMAL_I0_R_FROZEN_MAX_ROWS >= 800
    assert [_scheduled_stratum(index) for index in range(len(FORMAL_I0_R_STRATA))] == list(FORMAL_I0_R_STRATA)
    assert {_scheduled_stratum(len(FORMAL_I0_R_STRATA) + index).name
            for index in range(len(RELEASE_RICH_CONTINUATION))} == {
        row.name for row in RELEASE_RICH_CONTINUATION
    }
    assert all(not row.spec.dynamic and row.spec.final_capacity > 0 for row in RELEASE_RICH_CONTINUATION)


def test_formal_v2_1_i1_mapping_is_zero_based_and_wait_is_masked():
    rows = []
    for joint in (0, *FORMAL_I1_STABLE_SUPPORT):
        target = hierarchical_from_joint_index_v4_1(joint)
        rows.append({
            "observation_v2_1": [0.0] * 88,
            "teacher_gate": target.gate,
            "teacher_proposal": target.proposal,
            "teacher_joint_index": joint,
        })
    observations, gates, proposals, joint = _formal_i1_arrays(rows)
    assert observations.shape == (5, 88)
    assert gates.tolist() == [0, 1, 1, 1, 1]
    assert proposals.tolist() == [-1, 10, 11, 12, 13]
    assert joint.tolist() == [0, 11, 12, 13, 14]


def test_formal_v2_1_i1_loader_rejects_wrong_dataset_lineage(tmp_path):
    config = replace(_config(), observation_schema_version="phaseA_low_obs_v2_1_retry_state")
    policy = JointPPOV42(88)
    checkpoint = tmp_path / "formal.pt"
    extras = {
        "training_dataset_sha256": EXPECTED_TRAIN_SHA256,
        "frozen_dataset_sha256": EXPECTED_FROZEN_SHA256,
        "i0_r_manifest_sha256": EXPECTED_I0_MANIFEST_SHA256,
        "architecture_fingerprint": _architecture_fingerprint(policy),
    }
    save_checkpoint(checkpoint, policy, config, **extras)
    load_formal_i1_checkpoint(checkpoint, policy, config)
    payload = torch.load(checkpoint, weights_only=False)
    payload["training_dataset_sha256"] = "wrong-lineage"
    wrong = tmp_path / "wrong.pt"
    torch.save(payload, wrong)
    with pytest.raises(IncompatibleCheckpointError, match="lineage mismatch"):
        load_formal_i1_checkpoint(wrong, policy, config)


def test_v2_1_i1_qualification_protocol_is_frozen_before_new_cohort():
    path = Path("terminal_marl/configs/low_v4_2/i1_v2_1_qualification_protocol.yaml")
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert protocol["protocol"] == "V4_2_V2_1_I1_QUALIFICATION_V1"
    assert protocol["applies_to"] == "future_fresh_I1_only"
    assert protocol["diagnostic_checkpoint_4201_eligible"] is False
    assert protocol["fresh_actor_seeds"] == [4202, 4203, 4204]
    assert protocol["canonical_checkpoint_seed"] == 4202
    assert protocol["multi_seed_rule"] == "all_three_seeds_must_pass_every_per_seed_hard_gate"
    assert protocol["q_cohort_model_selection"] is False
    assert protocol["q_cohort_threshold_tuning"] is False
    assert protocol["per_seed_hard_gates"] == {
        "wait_recall_min": 0.90,
        "release_recall_min": 0.70,
        "gate_macro_f1_min": 0.80,
        "proposal_top1_min": 0.85,
        "proposal_top3_min": 0.95,
        "decoded_joint_accuracy_min": 0.85,
        "stable_teacher_proposals_required": [11, 12, 13, 14],
        "every_stable_proposal_predicted": True,
        "finite_gradients": True,
        "gate_class_prediction_share_min": 0.05,
        "proposal_dominant_prediction_share_max_exclusive": 0.95,
    }


def test_v2_1_i1_q_collector_matches_frozen_collection_protocol():
    assert (I1_Q_SEED_START, I1_Q_SEED_END, I1_Q_MAX_ROWS) == (210001, 210999, 1200)
    assert [row.name for row in I1_Q_STRATA] == [
        "clear_outer_low_traffic", "middle_interface", "post_WAIT_release",
        "conflict_recovery", "transition_retry", "nominal_mixed",
        "resource_pressure", "proposal_12_context", "proposal_14_context",
    ]
    assert I1_Q_MINIMUMS == {
        "WAIT_rows": 300, "WAIT_lifecycles": 250,
        "RELEASE_rows": 160, "RELEASE_lifecycles": 150,
        "proposal_11_lifecycles": 15, "proposal_12_lifecycles": 15,
        "proposal_13_lifecycles": 30, "proposal_14_lifecycles": 15,
        "clear_outer": 30, "middle_interface": 30, "post_WAIT": 30,
        "post_conflict": 30, "retry_active_WAIT": 60,
        "retry_expired_RELEASE": 60,
    }
    source = Path("terminal_marl/training/run_v4_2_v2_1_i1_q_cohort.py").read_text(encoding="utf-8")
    assert "JointPPOV42" not in source
    assert "from ..algo" not in source


def test_v4_2_v2_1_multiseed_protocol_and_q_access_guard(tmp_path):
    from terminal_marl.training.run_v4_2_v2_1_i1_multiseed_qualification import (
        _install_q_access_guard,
        _validate_protocol,
    )

    protocol = _validate_protocol(
        Path("terminal_marl/configs/low_v4_2/i1_v2_1_qualification_protocol.yaml")
    )
    assert protocol["fresh_actor_seeds"] == [4202, 4203, 4204]
    assert protocol["canonical_checkpoint_seed"] == 4202
    assert protocol["checkpoint_selection"] == "final_epoch"

    q_dataset = tmp_path / "immutable_q.jsonl"
    q_manifest = tmp_path / "immutable_q_manifest.json"
    state = {"phase": "TRAINING", "q_access_during_training": False}
    _install_q_access_guard((q_dataset, q_manifest), state)
    with pytest.raises(RuntimeError, match="Q cohort access attempted during training"):
        q_dataset.write_text("forbidden", encoding="utf-8")
    assert state["q_access_during_training"] is True

    state["phase"] = "Q_EVALUATION"
    q_dataset.write_text("allowed-after-freeze", encoding="utf-8")
    assert q_dataset.read_text(encoding="utf-8") == "allowed-after-freeze"
