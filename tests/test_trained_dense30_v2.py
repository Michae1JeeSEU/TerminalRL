from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest
import torch

import terminal_marl.training.diagnose_trained_dense30_v2 as diagnostic
from terminal_marl.algo.gate_calibrated_adapter_v4_2 import (
    CALIBRATED_ARCHITECTURE_ID,
    TemperatureCalibratedGateResidualAdapterStudentPPOV42,
)
from terminal_marl.core.checkpoints import checkpoint_metadata
from terminal_marl.core.low_level_joint_action_v4_2 import hierarchical_from_normalized_v4_2
from terminal_marl.core.runtime_traffic_safety import SafetyInvariantError
from terminal_marl.envs.parallel_env import TerminalParallelEnv


def _tiny_config():
    return replace(
        diagnostic.build_dense30_config(),
        warm_start_uavs=2,
        total_requested_uavs=2,
        fixed_arrival_schedule=(0.0, 0.0),
        max_simulation_seconds=5.0,
    )


def _release_policy():
    policy = TemperatureCalibratedGateResidualAdapterStudentPPOV42(88)
    with torch.no_grad():
        for parameter in policy.model.parameters():
            parameter.zero_()
        policy.model.gate_head.bias.copy_(torch.tensor([-1.0, 1.0]))
    policy.model.eval()
    return policy


def test_dense30_config_and_controller_identity():
    config = diagnostic.build_dense30_config()
    policy = _release_policy()

    assert config.observation_schema_version == diagnostic.OBSERVATION_SCHEMA
    assert config.action_schema_version == diagnostic.ACTION_SCHEMA
    assert config.neighbor_comm_range_m is None
    assert config.runtime_proposal_shortlist_k == 3
    assert config.experiment_mode == "fixed_low_student"
    assert config.rule_flow_coordination_enabled is False
    assert config.teacher_enabled is False
    assert config.teacher_release_label_enabled is False
    assert config.scheduler_mode == "execution"
    assert config.vehicle_profile.profile_version == diagnostic.TARGET_VEHICLE_PROFILE_VERSION
    assert policy.observation_dim == diagnostic.OBSERVATION_DIM
    assert policy.model.architecture_id == CALIBRATED_ARCHITECTURE_ID
    expected = checkpoint_metadata(config)
    assert expected["low_level_action_dim"] == 3
    assert expected["high_level_action_dim"] == 10
    assert expected["high_level_enabled"] is False
    assert expected["policy_kind"] == "low_level"


def test_metadata_rejects_schema_hash_and_unapproved_profile_changes():
    config = diagnostic.build_dense30_config()
    expected = checkpoint_metadata(config)
    source = dict(expected)
    source["vehicle_profile_version"] = diagnostic.SOURCE_VEHICLE_PROFILE_VERSION

    migration = diagnostic.validate_checkpoint_metadata_for_diagnostic(source, config)
    assert migration["applied_deltas"] == [{
        "field": "vehicle_profile_version",
        "source": diagnostic.SOURCE_VEHICLE_PROFILE_VERSION,
        "target": diagnostic.TARGET_VEHICLE_PROFILE_VERSION,
    }]

    bad_hash = dict(source)
    bad_hash["observation_feature_hash"] = "wrong-feature-hash"
    with pytest.raises(diagnostic.DiagnosticInputError):
        diagnostic.validate_checkpoint_metadata_for_diagnostic(bad_hash, config)

    bad_schema = dict(source)
    bad_schema["observation_schema_version"] = "phaseA_low_obs_v2_0"
    with pytest.raises(diagnostic.DiagnosticInputError):
        diagnostic.validate_checkpoint_metadata_for_diagnostic(bad_schema, config)

    bad_profile = dict(source)
    bad_profile["vehicle_profile_version"] = "unapproved_profile"
    with pytest.raises(diagnostic.DiagnosticInputError):
        diagnostic.validate_checkpoint_metadata_for_diagnostic(bad_profile, config)


def test_payload_bindings_reject_changed_source_contract():
    bindings = {
        "formal_protocol_sha256": "protocol",
        "initial_checkpoint_sha256": "initial",
    }
    payload = {
        "stage": "V4_2_P2_RUNTIME_SHORTLIST_FORMAL_V1",
        "formal": True,
        "ppo_called": True,
        "protocol_sha256": "protocol",
        "initial_checkpoint_sha256": "initial",
        "checkpoint_promotable": False,
    }
    assert diagnostic._validate_payload_bindings(payload, bindings)["formal"] is True
    payload["protocol_sha256"] = "changed"
    with pytest.raises(diagnostic.DiagnosticInputError):
        diagnostic._validate_payload_bindings(payload, bindings)


def test_checkpoint_hash_and_fresh_output_guards_precede_execution(tmp_path, monkeypatch):
    checkpoint = tmp_path / "invalid_checkpoint.pt"
    checkpoint.write_bytes(b"not a torch checkpoint")

    def fail_torch_load(*args, **kwargs):
        raise AssertionError("torch.load must not run after a SHA mismatch")

    monkeypatch.setattr(diagnostic.torch, "load", fail_torch_load)
    with pytest.raises(diagnostic.DiagnosticInputError, match="SHA mismatch"):
        diagnostic.load_trained_checkpoint(checkpoint, diagnostic.build_dense30_config())

    existing = tmp_path / "existing_output"
    existing.mkdir()

    def fail_reset(*args, **kwargs):
        raise AssertionError("existing output must be rejected before env.reset")

    monkeypatch.setattr(diagnostic.TerminalParallelEnv, "reset", fail_reset)
    with pytest.raises(FileExistsError):
        diagnostic.run_rollout(
            _release_policy(),
            _tiny_config(),
            output=existing,
            seed=3,
            max_steps=1,
        )


def test_rollout_records_actor_mask_trace_identity_and_serialized_output(tmp_path):
    output = tmp_path / "rollout"
    summary = diagnostic.run_rollout(
        _release_policy(),
        _tiny_config(),
        output=output,
        seed=3,
        max_steps=2,
    )

    assert summary["action_rows"] == 4
    assert 0 < summary["actor_decisions"] < summary["action_rows"]
    assert summary["teacher_used"] is False
    assert summary["PPO_ALLOWED"] is False
    assert summary["checkpoint_promotable"] is False
    assert summary["weight_digest_unchanged"] is True
    assert summary["unresolved_identities"]

    rows = [json.loads(line) for line in (output / "actor_steps.jsonl").read_text().splitlines()]
    assert any(row["actor_mask"] for row in rows)
    assert any(not row["actor_mask"] for row in rows)
    for row in rows:
        trace = row["action_trace"]
        assert trace["teacher_used"] is False
        np.testing.assert_array_equal(
            row["requested_action"], trace["requested_low_action"]
        )
    json.loads((output / "config.json").read_text())
    json.loads((output / "metadata.json").read_text())
    json.loads((output / "migration.json").read_text())
    json.loads((output / "summary.json").read_text())


def test_rollout_records_release_veto(tmp_path, monkeypatch):
    original_step = diagnostic.TerminalParallelEnv.step

    def forced_veto(self, requested, shortlists):
        observation, rewards, terminated, truncated, info = original_step(
            self, requested, shortlists
        )
        for uid, decision in list(info["release_decisions"].items()):
            if decision in diagnostic.SAFETY_DECISION_ACCEPTED:
                info["release_decisions"][uid] = "forced_veto"
                info["release_intent_telemetry"].setdefault(uid, {})[
                    "safety_reason"
                ] = "forced_veto"
        return observation, rewards, terminated, truncated, info

    monkeypatch.setattr(diagnostic.TerminalParallelEnv, "step", forced_veto)
    summary = diagnostic.run_rollout(
        _release_policy(),
        _tiny_config(),
        output=tmp_path / "veto",
        seed=3,
        max_steps=1,
    )
    assert summary["release_decisions"] == 2
    assert summary["release_vetoes"] == 2
    rows = [json.loads(line) for line in (tmp_path / "veto" / "actor_steps.jsonl").read_text().splitlines()]
    assert all(row["veto"]["vetoed"] for row in rows)


def test_safety_invariant_error_keeps_usable_terminal_output(tmp_path, monkeypatch):
    def fail_step(self, requested, shortlists):
        raise SafetyInvariantError("forced diagnostic failure")

    monkeypatch.setattr(diagnostic.TerminalParallelEnv, "step", fail_step)
    output = tmp_path / "safety_error"
    summary = diagnostic.run_rollout(
        _release_policy(),
        _tiny_config(),
        output=output,
        seed=3,
        max_steps=2,
    )
    assert summary["safety_error"] == "forced diagnostic failure"
    assert summary["steps_executed"] == 1
    assert json.loads((output / "summary.json").read_text())["safety_error"]
    assert json.loads((output / "progress.json").read_text())["status"] == "safety_error"
    rows = [json.loads(line) for line in (output / "actor_steps.jsonl").read_text().splitlines()]
    assert rows and all(row["status"] == "safety_invariant_error" for row in rows)


def test_nominal_request_details_uses_canonical_proposal12_and_same_actor_mask():
    env = TerminalParallelEnv(_tiny_config())
    observations, _ = env.reset(seed=3)
    env.active[1].committed_plan = object()

    details = diagnostic._nominal_request_details(env, observations, shortlist_k=3)
    uids, _, actor_mask, actions, gates, proposals, gate_lp, proposal_lp, likelihoods, shortlists = details

    assert uids == ["0", "1"]
    assert actor_mask.tolist() == [True, False]
    assert gates.tolist() == [1, 0]
    assert proposals.tolist() == [12, -1]
    np.testing.assert_array_equal(actions["0"], np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(actions["1"], np.asarray([0.0, 0.0, 0.0], dtype=np.float32))
    assert hierarchical_from_normalized_v4_2(actions["0"]).proposal == 12
    assert hierarchical_from_normalized_v4_2(actions["0"]).joint_index_v4_1 == 13
    assert list(shortlists) == ["0"]
    assert len(shortlists["0"]) == 1
    assert all(value == 0.0 for value in gate_lp)
    assert all(value == 0.0 for value in proposal_lp)
    assert likelihoods["0"] == {"gate": 0.0, "proposal": 0.0, "joint": 0.0}


def test_shadow_probe_records_positive_witness_and_preserves_original_state():
    env = TerminalParallelEnv(_tiny_config())
    observations, _ = env.reset(seed=3)
    details = diagnostic._nominal_request_details(env, observations, shortlist_k=3)
    rows = diagnostic._shadow_probe_rows(
        env,
        time_value=env.now,
        uids=details[0],
        actor_mask=details[2],
        gates=details[4],
        proposals=details[5],
        requested=details[3],
        shortlists=details[-1],
    )

    assert rows
    assert all(row["original_state_unchanged"] for row in rows)
    assert all(row["shadow_decision"] == "approved" for row in rows)
    assert all(row["approved_witness"] for row in rows)
    assert all(row["joint_feasibility_claim"] is False for row in rows)
    assert all(row["shadow_release_telemetry"]["secondary_constraints"] == [] for row in rows)
    assert all(row["shadow_action_trace"]["requested_hierarchical_action"]["proposal"] == 12 for row in rows)


def test_shadow_probe_records_negative_safety_invariant_without_witness(monkeypatch):
    env = TerminalParallelEnv(_tiny_config())
    observations, _ = env.reset(seed=3)
    details = diagnostic._nominal_request_details(env, observations, shortlist_k=3)

    def fail_shadow_step(self, requested, shortlists):
        raise SafetyInvariantError("forced shadow probe failure")

    monkeypatch.setattr(TerminalParallelEnv, "step", fail_shadow_step)
    rows = diagnostic._shadow_probe_rows(
        env,
        time_value=env.now,
        uids=details[0],
        actor_mask=details[2],
        gates=details[4],
        proposals=details[5],
        requested=details[3],
        shortlists=details[-1],
    )

    assert rows
    assert all(row["original_state_unchanged"] for row in rows)
    assert all(row["shadow_safety_error"] == "forced shadow probe failure" for row in rows)
    assert all(row["approved_witness"] is False for row in rows)
