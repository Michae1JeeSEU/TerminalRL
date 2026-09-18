from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import terminal_marl.training.collect_v4_2_p2_gate_support_v3 as runner


def _row(index: int, interface: str, gate: int, *, split: str = "train", seed: int | None = None):
    phase = {"o2m": "ring_outer", "m2i": "ring_middle", "i2f": "ring_inner"}[interface]
    return {
        "seed": runner.FORMAL_TRAIN_SEEDS[0] if seed is None else seed,
        "phase": phase,
        "step": index,
        "uav_id": index % 12,
        "decision_event_id": f"event-v3-{split}-{index}",
        "observation_v2_1": np.zeros(88, dtype=np.float32).tolist(),
        "actor_mask": True,
        "teacher_gate": gate,
        "teacher_proposal": 0 if gate else None,
        "requested_gate": gate,
        "support_split": split,
        "scenario": "flow6",
        "interface": interface,
    }


def _passing_rows(split: str, count: int, seeds: tuple[int, ...]):
    rows = []
    for index in range(count):
        interface = runner.INTERFACES[index % len(runner.INTERFACES)]
        gate = (index // len(runner.INTERFACES)) % 2
        rows.append(_row(index, interface, gate, split=split, seed=seeds[index % len(seeds)]))
    return rows


def _records(split: str, seeds: tuple[int, ...], *, fraction_gate: bool = True):
    shield_actor_count = 3 if fraction_gate else 48
    fraction = shield_actor_count / 959
    return [{
        "split": split,
        "seed": seed,
        "requested": 1,
        "completed": 1,
        "episode_execution_complete": True,
        "shield_actor_count": shield_actor_count,
        "actor_decisions": 959,
        "shield_actor_fraction": fraction,
        "shield_actor_fraction_within_inherited_limit": fraction_gate,
        "gates": {
            "replay": True,
            "label_only": True,
            "shield_actor_fraction_within_inherited_limit": fraction_gate,
        },
    } for seed in seeds]


def _failed_qualification_manifest(*, row_mixing: bool = False, auth: bool = False) -> dict:
    return {
        "stage": runner.QUALIFICATION_STAGE,
        "status": "FAIL",
        "P2_GATE_STUDENT_QUALIFICATION": "FAIL",
        "P2_EFFECTIVENESS_EVALUATION_ALLOWED": auth,
        "PPO_ALLOWED": False,
        "P3_ALLOWED": False,
        "curriculum_promotion_allowed": False,
        "checkpoint_promotable": False,
        "qualification_only": True,
        "recovered_from_completed_artifacts": True,
        "formal_rollouts_repeated": False,
        "formal_seed_resets": 0,
        "environment_rollout": True,
        "teacher_called": True,
        "support_v2_rows_mixed": row_mixing,
        "completeness": {"episodes": 6, "expected_episodes": 6, "requested": 52, "completed": 52},
        "records": [{} for _ in range(6)],
        "sampled_action_checks": {"false_WAIT_rate": False, "false_RELEASE_rate": True},
        "probability_checks": {
            "macro_F1": True,
            "WAIT_recall": True,
            "RELEASE_recall": False,
            "brier": True,
            "binary_nll": True,
        },
        "sampled_action_metrics": {
            "false_wait_rate": 106 / 318,
            "false_release_rate": 29 / 665,
        },
        "probability_metrics": {"RELEASE_recall": 0.625786},
        "row_checks": {"all": True},
    }


def _support_v2_manifest(*, row_mixing: bool = False, auth: bool = False) -> dict:
    return {
        "stage": "V4_2_P2_GATE_SUPPORT_COLLECTION_V2",
        "SUPPORT_COLLECTION": "PASS",
        "SUPPORT_FROZEN": True,
        "LEARNABILITY_AUDIT_ALLOWED": True,
        "GATE_REPAIR_ALLOWED": auth,
        "PPO_ALLOWED": False,
        "curriculum_promotion_allowed": False,
        "future_qualification_rollouts": 0,
        "future_qualification_consumed_marker": False,
        "support_v2_rows_mixed": row_mixing,
    }


def test_protocol_freezes_v3_candidate_predecessors_and_seed_groups():
    value = runner.protocol()
    assert tuple(value["formal"]["train_support_seeds"]) == runner.FORMAL_TRAIN_SEEDS
    assert tuple(value["formal"]["dev_support_seeds"]) == runner.FORMAL_DEV_SEEDS
    assert tuple(value["formal"]["future_qualification_seeds"]) == runner.FUTURE_QUALIFICATION_SEEDS
    groups = [set(runner.FORMAL_TRAIN_SEEDS), set(runner.FORMAL_DEV_SEEDS),
              set(runner.FUTURE_QUALIFICATION_SEEDS)]
    assert all(not (left & right) for index, left in enumerate(groups) for right in groups[index + 1:])
    assert runner._sha(runner.PROTOCOL) == runner.PROTOCOL_SHA
    assert value["candidate_checkpoint_sha256"] == runner.CANDIDATE_SHA
    assert value["candidate_architecture_id"] == runner.CANDIDATE_ARCHITECTURE_ID
    assert value["candidate_temperature"] == runner.CANDIDATE_TEMPERATURE
    assert value["collection"]["failed_qualification_rows_mixed"] is False
    assert value["bindings"]["source_usage"] == "manifest_only_predecessor_bindings_no_rows_mixed"


def test_binding_validators_reject_row_mixing_and_authorization_mismatch(monkeypatch, tmp_path):
    support_path = tmp_path / "support.json"
    support_path.write_text(json.dumps(_support_v2_manifest()), encoding="utf-8")
    monkeypatch.setattr(runner, "_sha", lambda path: runner.SUPPORT_V2_MANIFEST_SHA)
    binding = runner._validate_support_v2_manifest(support_path)
    assert binding["rows_opened"] is False

    support_path.write_text(json.dumps(_support_v2_manifest(row_mixing=True)), encoding="utf-8")
    with pytest.raises(ValueError, match="row-mixing"):
        runner._validate_support_v2_manifest(support_path)
    support_path.write_text(json.dumps(_support_v2_manifest(auth=True)), encoding="utf-8")
    with pytest.raises(ValueError, match="GATE_REPAIR_ALLOWED"):
        runner._validate_support_v2_manifest(support_path)

    failed_path = tmp_path / "failed.json"
    failed_path.write_text(json.dumps(_failed_qualification_manifest()), encoding="utf-8")
    monkeypatch.setattr(runner, "_sha", lambda path: runner.FAILED_QUALIFICATION_MANIFEST_SHA)
    failed_binding = runner._validate_failed_qualification_manifest(failed_path)
    assert failed_binding["rows_opened"] is False
    failed_path.write_text(json.dumps(_failed_qualification_manifest(row_mixing=True)), encoding="utf-8")
    with pytest.raises(ValueError, match="row-mixing"):
        runner._validate_failed_qualification_manifest(failed_path)
    failed_path.write_text(json.dumps(_failed_qualification_manifest(auth=True)), encoding="utf-8")
    with pytest.raises(ValueError, match="P2_EFFECTIVENESS"):
        runner._validate_failed_qualification_manifest(failed_path)


def test_calibrated_loader_uses_qualification_loader_and_checks_binding(monkeypatch, tmp_path):
    payload = {
        "architecture_id": runner.CANDIDATE_ARCHITECTURE_ID,
        "temperature": runner.CANDIDATE_TEMPERATURE,
        "STUDENT_QUALIFICATION_ALLOWED": True,
    }
    calls = []

    def fake_loader(path, *, expected_sha256):
        calls.append((path, expected_sha256))
        return object(), payload

    monkeypatch.setattr(runner.qualification, "_load_calibrated_checkpoint", fake_loader)
    checkpoint = tmp_path / "calibrated.pt"
    policy, returned = runner._load_calibrated_candidate(checkpoint)
    assert policy is not None and returned is payload
    assert calls == [(checkpoint, runner.CANDIDATE_SHA)]

    payload["STUDENT_QUALIFICATION_ALLOWED"] = False
    with pytest.raises(ValueError, match="qualification authorization"):
        runner._load_calibrated_candidate(checkpoint)


def test_smoke_has_no_marker_or_authorization(monkeypatch, tmp_path):
    payload = {
        "stage": "V4_2_P2_GATE_TEMPERATURE_CALIBRATION_V3",
        "architecture_id": runner.CANDIDATE_ARCHITECTURE_ID,
        "temperature": runner.CANDIDATE_TEMPERATURE,
        "qualification_only": True,
        "checkpoint_promotable": False,
    }
    monkeypatch.setattr(
        runner,
        "_validate_candidate_inputs",
        lambda checkpoint, support, failed: ({"rows_opened": False}, {"rows_opened": False}, object(), payload),
    )

    def fake_episode(policy, *, split, seed, scenario, output, max_steps):
        return {
            "split": split,
            "seed": seed,
            "scenario": scenario,
            "requested": 6,
            "completed": 6,
            "episode_execution_complete": True,
            "safety": {},
            "gates": {"synthetic": True},
        }, []

    monkeypatch.setattr(runner, "_episode_record", fake_episode)
    result = runner.run(
        tmp_path / "smoke",
        tmp_path / "calibrated.pt",
        tmp_path / "support_v2.json",
        tmp_path / "qualification.json",
        tmp_path,
        smoke=True,
    )
    assert result["stage"] == runner.STAGE
    assert result["smoke"] is True
    assert result["authorization"]["SUPPORT_COLLECTION"] == "FAIL"
    assert result["SUPPORT_FROZEN"] is False
    assert result["LEARNABILITY_AUDIT_ALLOWED"] is False
    assert result["formal_marker_created"] is False
    assert result["formal_marker"] is None
    assert not list(tmp_path.glob(runner.RESERVATION_MARKER))
    assert result["support_v2_rows_mixed"] is False
    assert result["failed_qualification_rows_mixed"] is False


def test_formal_reservation_is_one_shot_and_rejects_historical_collision(monkeypatch, tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "candidate.pt"
    output = tmp_path / "formal"
    monkeypatch.setattr(runner, "historical_registry", lambda history, out: {"seeds": set()})
    marker, registry = runner._reserve_formal(checkpoint, tmp_path, output)
    assert marker.name == runner.RESERVATION_MARKER
    assert marker.is_file()
    assert registry["checked_future_seeds"] is True
    with pytest.raises(ValueError, match="consumed marker"):
        runner._reserve_formal(checkpoint, tmp_path, tmp_path / "formal-again")

    collision_dir = tmp_path / "collision-checkpoint"
    collision_dir.mkdir()
    collision_checkpoint = collision_dir / "candidate.pt"
    monkeypatch.setattr(
        runner,
        "historical_registry",
        lambda history, out: {"seeds": {runner.FORMAL_TRAIN_SEEDS[0]}},
    )
    with pytest.raises(ValueError, match="seed already consumed"):
        runner._reserve_formal(collision_checkpoint, tmp_path, tmp_path / "collision")
    assert not (collision_dir / runner.RESERVATION_MARKER).exists()


def test_inherited_shield_gate_and_support_thresholds_are_reused():
    rows = _passing_rows("train", 3000, runner.FORMAL_TRAIN_SEEDS)
    result = runner.support_gates(
        rows,
        "train",
        _records("train", runner.FORMAL_TRAIN_SEEDS),
        runner.FORMAL_TRAIN_SEEDS,
    )
    assert result["gates"]["all_episode_shield_actor_fraction_within_inherited_limit"]
    assert result["gates"]["all"]

    over = runner.support_gates(
        _passing_rows("dev", 1800, runner.FORMAL_DEV_SEEDS),
        "dev",
        _records("dev", runner.FORMAL_DEV_SEEDS, fraction_gate=False),
        runner.FORMAL_DEV_SEEDS,
    )
    assert not over["gates"]["all_episode_shield_actor_fraction_within_inherited_limit"]
    assert over["gates"]["schema_88d_finite"]
    assert not over["gates"]["all"]


def test_manifest_stage_and_authorization_boundary_are_explicit():
    value = runner.protocol()
    assert value["protocol"] == runner.STAGE
    assert value["authorization"]["always_false"] == [
        "GATE_REPAIR_ALLOWED", "PPO_ALLOWED", "curriculum_promotion_allowed",
    ]
    assert value["prohibited"]
    assert "support_v2_row_mixing" in value["prohibited"]
    assert "failed_qualification_row_mixing" in value["prohibited"]
