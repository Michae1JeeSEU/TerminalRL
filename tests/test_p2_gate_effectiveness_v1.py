from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from terminal_marl.algo.gate_calibrated_adapter_v4_2 import (
    TemperatureCalibratedGateResidualAdapterStudentPPOV42,
)
from terminal_marl.core.checkpoints import save_checkpoint
from terminal_marl.training.run_v4_2_ppo_p2_resume_v4 import p2_config


runner = importlib.import_module(
    "terminal_marl.training.evaluate_v4_2_p2_gate_effectiveness_v1"
)


def _aggregate(wait: float, touchdown: float) -> dict:
    return {
        "weighted_mean_wait_seconds": wait,
        "weighted_mean_request_to_touchdown_seconds": touchdown,
        "all_gates": True,
        "requests": 52,
        "completed": 52,
    }


def test_protocol_binds_predecessors_controllers_and_fresh_schedule() -> None:
    value = runner.protocol()
    assert value["protocol"] == runner.EFFECTIVENESS_STAGE
    assert value["predecessor"]["qualification_manifest_sha256"] == runner.QUALIFICATION_MANIFEST_SHA
    assert value["predecessor"]["repair_manifest_sha256"] == runner.REPAIR_MANIFEST_SHA
    assert tuple(value["formal"]["seeds"]) == runner.FORMAL_SEEDS
    assert tuple(value["formal"]["scenarios"]) == runner.FORMAL_SCENARIOS
    assert tuple(value["formal"]["controllers"]) == runner.CONTROLLERS
    assert value["formal"]["expected_requests_per_controller"] == 52
    assert value["smoke"]["formal_marker"] is False


def test_gap_closure_is_inclusive_and_handles_nonpositive_gaps() -> None:
    assert runner.gap_closure(9.0, 10.0, 8.0) == pytest.approx(0.50)
    assert runner.gap_closure(9.0, 9.0, 10.0) == 1.0
    assert runner.gap_closure(10.0, 9.0, 9.0) < 0.0
    assert runner.gap_closure(float("nan"), 1.0, 0.0) is None


def test_effectiveness_gate_checks_pass_at_exact_half_closure() -> None:
    checks = runner.effectiveness_gate_checks({
        "candidate": _aggregate(9.0, 18.0),
        "baseline": _aggregate(10.0, 20.0),
        "rule": _aggregate(8.0, 16.0),
    })
    assert checks["candidate_wait_le_baseline"]
    assert checks["candidate_request_to_touchdown_le_baseline"]
    assert checks["wait_gap_closure"] == pytest.approx(0.50)
    assert checks["request_to_touchdown_gap_closure"] == pytest.approx(0.50)
    assert checks["all"]


def test_effectiveness_gate_checks_reject_candidate_regression() -> None:
    checks = runner.effectiveness_gate_checks({
        "candidate": _aggregate(10.1, 20.1),
        "baseline": _aggregate(10.0, 20.0),
        "rule": _aggregate(8.0, 16.0),
    })
    assert not checks["candidate_wait_le_baseline"]
    assert not checks["candidate_request_to_touchdown_le_baseline"]
    assert not checks["all"]


def test_formal_shape_requires_three_controllers_and_scenario_cycle() -> None:
    records = []
    for controller in runner.CONTROLLERS:
        for seed, scenario in zip(runner.FORMAL_SEEDS, runner.FORMAL_SCENARIOS):
            records.append({
                "controller": controller,
                "seed": seed,
                "scenario": scenario,
                "result": {"seed": seed, "gates": {"all": True}},
            })
    assert runner._formal_shape(records)
    records[-1]["scenario"] = "flow6"
    assert not runner._formal_shape(records)


def test_pair_deltas_read_raw_episode_metrics_and_emit_standardized_keys() -> None:
    records = []
    values = {
        "candidate": (10.0, 3.0, 8.0),
        "baseline": (12.0, 5.0, 9.0),
        "rule": (14.0, 7.0, 10.0),
    }
    for controller, (seconds, wait, touchdown) in values.items():
        records.append({
            "controller": controller,
            "seed": runner.FORMAL_SEEDS[0],
            "scenario": runner.FORMAL_SCENARIOS[0],
            "result": {
                "seed": runner.FORMAL_SEEDS[0],
                "seconds": seconds,
                "weighted_wait_seconds": wait,
                "request_to_touchdown_seconds": touchdown,
            },
        })
    delta = runner.pair_deltas(records)[0]
    assert delta["candidate_minus_baseline"] == {
        "completion_seconds": -2.0,
        "weighted_mean_wait_seconds": -2.0,
        "weighted_mean_request_to_touchdown_seconds": -1.0,
    }
    assert delta["candidate_minus_rule"] == {
        "completion_seconds": -4.0,
        "weighted_mean_wait_seconds": -4.0,
        "weighted_mean_request_to_touchdown_seconds": -2.0,
    }


def test_smoke_semantics_never_reserves_or_authorizes() -> None:
    checks = runner.smoke_semantics(
        smoke=True, seeds=[runner.SMOKE_SEED],
        scenarios=[runner.SMOKE_SCENARIO], steps=runner.SMOKE_STEPS,
    )
    assert all(checks.values())
    assert not runner.smoke_semantics(
        smoke=False, seeds=list(runner.FORMAL_SEEDS),
        scenarios=list(runner.FORMAL_SCENARIOS), steps=0,
    )["no_reservation"]


def _save_synthetic(path: Path, *, stage: str, qualification_pass: bool = True) -> None:
    policy = TemperatureCalibratedGateResidualAdapterStudentPPOV42(88, temperature=2.5)
    save_checkpoint(
        path, policy, p2_config(6), stage=stage,
        architecture_id=runner.ARCHITECTURE_ID,
        temperature=runner.TEMPERATURE,
        qualification_only=True,
        qualification_pass=qualification_pass,
        checkpoint_role="qualification_candidate",
        checkpoint_promotable=False,
        PPO_ALLOWED=False,
        curriculum_promotion_allowed=False,
    )


def test_candidate_and_baseline_loaders_do_not_load_optimizer_state(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidate.pt"
    baseline_path = tmp_path / "baseline.pt"
    _save_synthetic(
        candidate_path,
        stage="V4_2_P2_GATE_RESIDUAL_ADAPTER_REPAIR_V3",
    )
    _save_synthetic(
        baseline_path,
        stage="V4_2_P2_GATE_TEMPERATURE_CALIBRATION_V3",
    )
    candidate, candidate_payload = runner.load_candidate_checkpoint(
        candidate_path, expected_sha256=None,
    )
    baseline, baseline_payload = runner.load_baseline_checkpoint(
        baseline_path, expected_sha256=None,
    )
    assert "optimizer" in candidate_payload
    assert "optimizer" in baseline_payload
    assert len(candidate.optimizer.state) == 0
    assert len(baseline.optimizer.state) == 0


def test_candidate_loader_falls_back_to_temperature_buffer(tmp_path: Path) -> None:
    checkpoint = tmp_path / "candidate-missing-temperature.pt"
    _save_synthetic(
        checkpoint,
        stage="V4_2_P2_GATE_RESIDUAL_ADAPTER_REPAIR_V3",
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload.pop("temperature", None)
    torch.save(payload, checkpoint)
    restored, loaded = runner.load_candidate_checkpoint(
        checkpoint, expected_sha256=None,
    )
    assert loaded["temperature"] == runner.TEMPERATURE
    assert float(restored.model.gate_temperature.detach().cpu()) == runner.TEMPERATURE
    assert len(restored.optimizer.state) == 0

    wrong = tmp_path / "candidate-wrong-buffer.pt"
    wrong_policy = TemperatureCalibratedGateResidualAdapterStudentPPOV42(88, temperature=2.5)
    wrong_policy.model.gate_temperature.data.fill_(3.0)
    save_checkpoint(
        wrong, wrong_policy, p2_config(6),
        stage="V4_2_P2_GATE_RESIDUAL_ADAPTER_REPAIR_V3",
        architecture_id=runner.ARCHITECTURE_ID,
        qualification_only=True,
        qualification_pass=True,
        checkpoint_role="qualification_candidate",
        checkpoint_promotable=False,
        PPO_ALLOWED=False,
        curriculum_promotion_allowed=False,
    )
    with pytest.raises(runner.EffectivenessInputError, match="temperature"):
        runner.load_candidate_checkpoint(wrong, expected_sha256=None)


def test_smoke_run_has_three_records_and_no_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_policy = SimpleNamespace()
    monkeypatch.setattr(runner, "_load_inputs", lambda *args, **kwargs: {
        "qualification_manifest": {"status": "PASS"},
        "repair_manifest": {"status": "PASS"},
        "baseline_manifest": {"status": "PASS"},
        "candidate": fake_policy,
        "candidate_payload": {},
        "baseline": fake_policy,
        "baseline_payload": {},
    })

    def fake_student(policy, config, seed, output, *, smoke):
        output.mkdir(parents=True, exist_ok=False)
        return {
            "seed": seed, "requests": 6, "completed": 0, "terminal": False,
            "seconds": 160.0, "safety": {}, "shield": 0, "shield_actor": 0,
            "actor_decisions": 1, "shield_actor_fraction": 0.0,
            "weighted_wait_seconds": None, "request_to_touchdown_seconds": None,
            "weighted_mean_wait_seconds": None,
            "weighted_mean_request_to_touchdown_seconds": None,
            "replay_exact": True, "requested_logprob_exact": True,
            "teacher_enabled": False, "label_rows": 0, "training_called": False,
            "geometry": {"samples": 1, "reverse_samples": 0, "max_inner_excursion_m": 0.0},
            "gates": {"all": False},
        }

    def fake_rule(config, seed, output, *, smoke):
        output.mkdir(parents=True, exist_ok=False)
        return {
            "seed": seed, "requests": 6, "completed": 0, "terminal": False,
            "seconds": 160.0, "safety": {}, "shield": 0,
            "weighted_wait_seconds": None, "request_to_touchdown_seconds": None,
            "weighted_mean_wait_seconds": None,
            "weighted_mean_request_to_touchdown_seconds": None,
            "replay_exact": True, "teacher_enabled": False, "training_called": False,
            "geometry": {"samples": 1, "reverse_samples": 0, "max_inner_excursion_m": 0.0},
            "gates": {"all": False},
        }

    monkeypatch.setattr(runner, "_student_result", fake_student)
    monkeypatch.setattr(runner, "_rule_result", fake_rule)
    for path in (
        tmp_path / "qualification.json", tmp_path / "repair.json",
        tmp_path / "candidate.pt", tmp_path / "baseline.json", tmp_path / "baseline.pt",
    ):
        path.write_bytes(b"synthetic")
    result = runner.run(
        tmp_path / "smoke", tmp_path / "qualification.json", tmp_path / "repair.json",
        tmp_path / "candidate.pt", tmp_path / "baseline.json", tmp_path / "baseline.pt",
        tmp_path / "history", smoke=True,
    )
    assert result["status"] == "FAIL"
    assert len(result["records"]) == 3
    assert result["formal_marker_created"] is False
    assert result["P2_PPO_ADMISSION_REVIEW_ALLOWED"] is False
    assert not (tmp_path / runner.RESERVATION_MARKER).exists()
    assert json.loads((tmp_path / "smoke" / "manifest.json").read_text())[
        "PPO_ALLOWED"
    ] is False
