from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from terminal_marl.algo.gate_calibrated_adapter_v4_2 import (
    TemperatureCalibratedGateResidualAdapterStudentPPOV42,
)
from terminal_marl.core.checkpoints import save_checkpoint
from terminal_marl.training import qualify_v4_2_p2_gate_student_v1 as v1
from terminal_marl.training.run_v4_2_ppo_p2_resume_v4 import p2_config

runner = importlib.import_module("terminal_marl.training.qualify_v4_2_p2_gate_student_v2")


def _row(index: int, teacher: int, requested: int, interface: str = "o2m") -> dict:
    phase = {"o2m": "ring_outer", "m2i": "ring_middle", "i2f": "ring_inner"}[interface]
    observation = np.zeros(88, dtype=np.float32)
    observation[index % 8] = 1.0
    return {
        "decision_event_id": f"qualification-v2-event-{index}",
        "seed": 782001,
        "observation_v2_1": observation.tolist(),
        "teacher_gate": teacher,
        "requested_gate": requested,
        "actor_mask": True,
        "scenario": "flow6",
        "interface": interface,
        "phase": phase,
    }


def _episode_summary(**overrides) -> dict:
    value = {
        "requested": 2,
        "completed": 2,
        "terminal": True,
        "queue": [0],
        "steps": 4,
        "safety": {"collisions": 0, "nmac": 0, "hard_separation_violations": 0,
                   "uncertified_execution": 0, "SafetyInvariantError": 0},
        "counts": {"actor_decisions": 2, "shield_actor": 0},
        "replay_exact": True,
        "requested_logprob_exact": True,
        "teacher_label_only": True,
        "student_authoritative": True,
        "actor_rows": 2,
        "training_called": False,
        "optimizer_called": False,
        "policy_update": False,
    }
    value.update(overrides)
    return value


def test_protocol_binds_repair_v3_and_fresh_v2_seeds() -> None:
    value = runner.protocol()
    assert tuple(value["formal"]["seeds"]) == runner.FORMAL_SEEDS
    assert tuple(value["formal"]["scenarios"]) == runner.FORMAL_SCENARIOS
    assert value["predecessor"]["repair_manifest_sha256"] == runner.REPAIR_MANIFEST_SHA
    assert value["predecessor"]["repaired_checkpoint_sha256"] == runner.REPAIRED_CHECKPOINT_SHA
    assert tuple(value["formal"]["old_qualification_seeds_forbidden"]) == runner.OLD_QUALIFICATION_SEEDS
    assert runner.formal_horizons() == (1210, 1330, 1570, 1210, 1330, 1570)


def test_episode_gates_reject_incomplete_safety_and_shield() -> None:
    assert runner.episode_gate_checks(_episode_summary(), 10)["all"]
    assert not runner.episode_gate_checks(_episode_summary(completed=1), 10)["all"]
    assert not runner.episode_gate_checks(_episode_summary(safety={"collisions": 1}), 10)["all"]
    assert not runner.episode_gate_checks(
        _episode_summary(counts={"actor_decisions": 10, "shield_actor": 1}), 10
    )["all"]


def test_sampled_and_probability_thresholds_are_inclusive() -> None:
    rows = [_row(0, 1, 0), _row(1, 0, 1), _row(2, 1, 1), _row(3, 0, 0)]
    metrics = runner.sampled_action_metrics(rows)
    assert metrics["false_wait_rate"] == .5
    assert metrics["false_release_rate"] == .5
    assert metrics["both_teacher_classes"] and metrics["both_requested_gates"]
    sampled = {
        "both_teacher_classes": True, "both_requested_gates": True,
        "false_wait_rate": .30, "false_release_rate": .30,
        "interface": {
            name: {"both_teacher_classes": True, "false_wait_rate": .40,
                   "false_release_rate": .40}
            for name in runner.INTERFACES
        },
    }
    assert runner.sampled_action_threshold_checks(sampled)["all"]
    sampled["interface"]["m2i"]["false_release_rate"] = .400001
    assert not runner.sampled_action_threshold_checks(sampled)["all"]
    probabilities = {
        "macro_F1": .75, "WAIT_recall": .65, "RELEASE_recall": .75,
        "brier": .20, "binary_nll": .70,
        "interface": {name: {"macro_F1": .65} for name in runner.INTERFACES},
    }
    assert runner.probability_threshold_checks(probabilities)["all"]
    probabilities["binary_nll"] = .700001
    assert not runner.probability_threshold_checks(probabilities)["all"]


def test_repaired_checkpoint_round_trip_keeps_optimizer_present_but_unloaded(tmp_path: Path) -> None:
    torch.manual_seed(7)
    source = TemperatureCalibratedGateResidualAdapterStudentPPOV42(88, temperature=2.5)
    checkpoint = tmp_path / "repair-v3.pt"
    save_checkpoint(
        checkpoint, source, p2_config(6),
        stage=runner.REPAIR_STAGE,
        architecture_id=runner.REPAIR_ARCHITECTURE_ID,
        temperature=2.5,
        qualification_pass=True,
        STUDENT_QUALIFICATION_ALLOWED=True,
        checkpoint_role="qualification_candidate",
        qualification_only=True,
        checkpoint_promotable=False,
        PPO_ALLOWED=False,
        GATE_REPAIR_ALLOWED=False,
        curriculum_promotion_allowed=False,
    )
    restored, payload = runner._load_repaired_checkpoint(checkpoint, expected_sha256=None)
    assert isinstance(restored, TemperatureCalibratedGateResidualAdapterStudentPPOV42)
    assert "optimizer" in payload
    assert len(restored.optimizer.state) == 0
    for name, value in source.model.state_dict().items():
        assert torch.equal(value, restored.model.state_dict()[name])


def test_smoke_never_authorizes_or_creates_formal_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    history = tmp_path / "history"
    history.mkdir()
    fake_policy = SimpleNamespace(model=object())
    monkeypatch.setattr(runner, "_load_inputs", lambda *args: (
        {"status": "PASS", "repair_pass": True, "STUDENT_QUALIFICATION_ALLOWED": True},
        fake_policy,
        {"stage": runner.REPAIR_STAGE, "architecture_id": runner.REPAIR_ARCHITECTURE_ID,
         "temperature": 2.5, "qualification_pass": True, "qualification_only": True,
         "checkpoint_role": "qualification_candidate", "checkpoint_promotable": False,
         "optimizer": {}},
    ))
    observed = {}

    def fake_episode_record(*args, **kwargs):
        observed["output"] = kwargs["output"]
        assert not kwargs["output"].exists()
        return _episode_summary(actor_rows=0), []

    monkeypatch.setattr(runner.support_v2, "_episode_record", fake_episode_record)
    result = runner.run(tmp_path / "smoke", tmp_path / "repair.json",
                        tmp_path / "checkpoint.pt", history, smoke=True)
    assert result["status"] == "FAIL"
    assert result["P2_EFFECTIVENESS_EVALUATION_ALLOWED"] is False
    assert result["formal_marker_created"] is False
    assert observed["output"].name == "seed782000_flow6"
    assert not (tmp_path / runner.RESERVATION_MARKER).exists()


def test_formal_reservation_is_one_shot_and_checks_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    history = tmp_path / "history"
    history.mkdir()
    output = tmp_path / "attempt"
    marker = tmp_path / runner.RESERVATION_MARKER
    monkeypatch.setattr(runner.support_v2, "historical_registry",
                        lambda history_root, output_root: {"seeds": set()})
    reserved = runner.reserve_formal_attempt(marker, history, output, protocol_sha256="test")
    assert marker.is_file()
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["formal_seeds"] == list(runner.FORMAL_SEEDS)
    assert payload["repaired_checkpoint_sha256"] == runner.REPAIRED_CHECKPOINT_SHA
    assert reserved["payload"]["reserved_before_first_reset"] is True
    with pytest.raises(ValueError, match="already exists"):
        runner.reserve_formal_attempt(marker, history, output, protocol_sha256="test")


def test_jsonable_serializes_sets_deterministically() -> None:
    assert runner._jsonable({3, 1, 2}) == [1, 2, 3]
    assert runner._jsonable(frozenset({"b", "a"})) == ["a", "b"]
    assert runner._jsonable({("b", 2), ("a", 1)}) == [["a", 1], ["b", 2]]


def _recovery_scaffold(tmp_path: Path):
    output = tmp_path / "completed"
    output.mkdir()
    history = tmp_path / "history"
    history.mkdir()
    repair_manifest = tmp_path / "repair.json"
    repair_manifest.write_text("{}", encoding="utf-8")
    checkpoint = tmp_path / "repair.pt"
    checkpoint.write_bytes(b"synthetic-checkpoint")
    marker = runner.reservation_marker_path(checkpoint)
    marker_payload = {
        "protocol": runner.QUALIFICATION_STAGE,
        "protocol_sha256": runner.PROTOCOL_SHA,
        "formal_seeds": list(runner.FORMAL_SEEDS),
        "old_qualification_seeds": list(runner.OLD_QUALIFICATION_SEEDS),
        "output": str(output),
        "repaired_checkpoint_sha256": runner.REPAIRED_CHECKPOINT_SHA,
        "reserved_before_first_reset": True,
    }
    marker.write_text(json.dumps(marker_payload), encoding="utf-8")
    (output / "progress.json").write_text(json.dumps({
        "stage": runner.QUALIFICATION_STAGE, "smoke": False,
        "episodes_completed": 6, "episodes_planned": 6,
        "seeds_completed": list(runner.FORMAL_SEEDS),
    }), encoding="utf-8")
    (output / "pre_reset_registry.json").write_text(json.dumps({
        "built_before_first_reset": True, "historical_registry_checked": True,
        "formal_seeds": list(runner.FORMAL_SEEDS),
        "protocol_sha256": runner.PROTOCOL_SHA,
        "reservation_marker": str(marker),
    }), encoding="utf-8")
    interfaces = ("ring_outer", "ring_middle", "ring_inner", "ring_outer", "ring_middle", "ring_inner")
    for index, (seed, scenario, phase) in enumerate(zip(
        runner.FORMAL_SEEDS, runner.FORMAL_SCENARIOS, interfaces,
    )):
        episode = output / f"seed{seed}_{scenario}"
        episode.mkdir()
        tape = episode / "action_tape.json"
        tape.write_text("{}", encoding="utf-8")
        interface = {"ring_outer": "o2m", "ring_middle": "m2i", "ring_inner": "i2f"}[phase]
        row = _row(index, index % 2, index % 2, interface)
        row["seed"] = seed
        row["phase"] = phase
        row["decision_event_id"] = f"recovery-v2-event-{index}"
        (episode / "labels.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        requested = int(scenario.removeprefix("flow"))
        (episode / "summary.json").write_text(json.dumps({
            "seed": seed, "steps": 1,
            "counts": {"actor_decisions": 1, "shield_actor": 0, "shield": 0},
            "safety": {"collisions": 0, "nmac": 0, "hard_separation_violations": 0,
                       "uncertified_execution": 0, "SafetyInvariantError": 0},
            "completed": requested, "queue": [0], "terminal": True,
            "action_tape": str(tape), "label_rows": 1,
        }), encoding="utf-8")
    return output, history, repair_manifest, checkpoint, marker


def _patch_recovery_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_policy = SimpleNamespace(model=object())
    monkeypatch.setattr(runner, "_load_inputs", lambda *args: (
        {"status": "PASS", "repair_pass": True, "STUDENT_QUALIFICATION_ALLOWED": True},
        fake_policy,
        {"stage": runner.REPAIR_STAGE, "architecture_id": runner.REPAIR_ARCHITECTURE_ID,
         "temperature": 2.5, "qualification_pass": True, "qualification_only": True,
         "checkpoint_role": "qualification_candidate", "checkpoint_promotable": False,
         "optimizer": {}},
    ))
    monkeypatch.setattr(runner, "TerminalParallelEnv", lambda config: object())
    monkeypatch.setattr(runner.SafetyTraceRecorder, "replay_action_tape",
                        staticmethod(lambda env, tape: {"exact": True}))
    monkeypatch.setattr(runner, "_policy_tape_logprob_exact", lambda *args, **kwargs: True)
    monkeypatch.setattr(runner.support_v2, "_tape_flags",
                        lambda path: {"teacher_label_only": True, "student_authoritative": True})
    monkeypatch.setattr(v1, "_policy_tape_logprob_exact", lambda *args, **kwargs: True)
    monkeypatch.setattr(v1.SafetyTraceRecorder, "replay_action_tape",
                        staticmethod(lambda env, tape: {"exact": True}))
    monkeypatch.setattr(v1, "TerminalParallelEnv", lambda config: object())
    monkeypatch.setattr(runner.support_v2, "_episode_record",
                        lambda *args, **kwargs: pytest.fail("formal rollout was repeated"))
    monkeypatch.setattr(runner, "reserve_formal_attempt",
                        lambda *args, **kwargs: pytest.fail("reservation was repeated"))
    monkeypatch.setattr(runner, "probability_metrics", lambda *args, **kwargs: {
        "macro_F1": 0.0, "WAIT_recall": 0.0, "RELEASE_recall": 0.0,
        "brier": 1.0, "binary_nll": 1.0,
        "interface": {name: {"macro_F1": 0.0} for name in runner.INTERFACES},
    })


def test_finalize_existing_rejects_incomplete_or_mismatched_artifacts(tmp_path: Path,
                                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    output, history, repair_manifest, checkpoint, marker = _recovery_scaffold(tmp_path)
    _patch_recovery_runtime(monkeypatch)
    (output / "progress.json").write_text(json.dumps({
        "stage": runner.QUALIFICATION_STAGE, "smoke": False,
        "episodes_completed": 5, "episodes_planned": 6,
        "seeds_completed": list(runner.FORMAL_SEEDS[:5]),
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="6/6"):
        runner.finalize_existing(output, repair_manifest, checkpoint, history)
    (output / "progress.json").write_text(json.dumps({
        "stage": runner.QUALIFICATION_STAGE, "smoke": False,
        "episodes_completed": 6, "episodes_planned": 6,
        "seeds_completed": list(runner.FORMAL_SEEDS),
    }), encoding="utf-8")
    marker.write_text(json.dumps({
        "protocol": runner.QUALIFICATION_STAGE,
        "protocol_sha256": runner.PROTOCOL_SHA,
        "formal_seeds": list(runner.FORMAL_SEEDS),
        "old_qualification_seeds": list(runner.OLD_QUALIFICATION_SEEDS),
        "output": str(tmp_path / "wrong-output"),
        "repaired_checkpoint_sha256": runner.REPAIRED_CHECKPOINT_SHA,
        "reserved_before_first_reset": True,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="output mismatch"):
        runner.finalize_existing(output, repair_manifest, checkpoint, history)


def test_finalize_existing_recovers_once_without_rollout_or_marker_write(tmp_path: Path,
                                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    output, history, repair_manifest, checkpoint, marker = _recovery_scaffold(tmp_path)
    _patch_recovery_runtime(monkeypatch)
    marker_before = marker.read_bytes()
    result = runner.finalize_existing(output, repair_manifest, checkpoint, history)
    assert result["status"] == "FAIL"
    assert result["P2_EFFECTIVENESS_EVALUATION_ALLOWED"] is False
    assert result["PPO_ALLOWED"] is False
    assert result["P3_ALLOWED"] is False
    assert result["recovered_from_completed_artifacts"] is True
    assert result["formal_rollouts_repeated"] is False
    assert result["formal_seed_resets"] == 0
    assert result["environment_rollout"] is True
    assert result["exception"] is None
    assert (output / "manifest.json").is_file()
    assert marker.read_bytes() == marker_before
    with pytest.raises(ValueError, match="manifest already exists"):
        runner.finalize_existing(output, repair_manifest, checkpoint, history)
