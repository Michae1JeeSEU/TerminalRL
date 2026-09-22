"""Focused checks for the diagnostic-only high-SOC P3 pilot."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from terminal_marl.training import (
    diagnose_v4_2_p3_high_soc_pilot_v1 as pilot,
)


def test_protocol_override_is_deep_copied_and_preserves_other_semantics() -> None:
    original = pilot.safety_energy.load_scenario_protocol()
    overridden = pilot.override_high_soc_protocol(original)

    assert original["attributes"]["initial_soc_min"] == 0.55
    assert original["attributes"]["initial_soc_max"] == 0.90
    assert overridden is not original
    assert overridden["attributes"] is not original["attributes"]
    assert overridden["attributes"]["initial_soc_min"] == pytest.approx(0.90)
    assert overridden["attributes"]["initial_soc_max"] == pytest.approx(1.00)
    assert overridden["reward"] == original["reward"]
    assert overridden["scenarios"] == original["scenarios"]


def test_high_soc_config_keeps_fixed_dense24_constants() -> None:
    config = pilot.scenario_config()

    assert config.initial_soc_mode == "uniform"
    assert config.initial_soc_min == pytest.approx(0.90)
    assert config.initial_soc_max == pytest.approx(1.00)
    assert config.total_requested_uavs == 24
    assert config.warm_start_uavs == 12
    assert config.safe_distance == pytest.approx(80.0)
    assert config.nmac_distance == pytest.approx(40.0)
    assert config.collision_distance == pytest.approx(10.0)
    assert config.energy_capacity == pytest.approx(1500.0)
    assert config.max_simulation_seconds == pytest.approx(1800.0)
    assert config.low_level_priority_time_weight == pytest.approx(1.0)
    assert config.low_level_soc_urgency_time_weight == pytest.approx(2.0)
    assert config.low_level_soc_urgency_threshold == pytest.approx(0.65)


def test_case_matrix_is_fixed_80m_rule_k3_pair_per_seed() -> None:
    cases = pilot.build_case_matrix([901, 902])

    assert len(cases) == 4
    assert [row["case_id"] for row in cases] == [
        "dense24_seed901_safe80_rule",
        "dense24_seed901_safe80_original_k3",
        "dense24_seed902_safe80_rule",
        "dense24_seed902_safe80_original_k3",
    ]
    assert all(row["safe_distance"] == 80.0 for row in cases)
    assert all({row["controller"] for row in cases if row["seed"] == seed}
               == {"rule", "original_k3"} for seed in (901, 902))


def _record(
    *,
    status: str = "completed",
    terminal: bool = True,
    requests: int | None = 24,
    completed: int | None = 24,
    exception: str | None = None,
    exact: bool = True,
) -> dict:
    return {
        "status": status,
        "terminal": terminal,
        "requests": requests,
        "completed": completed,
        "unresolved": None if requests is None or completed is None else requests - completed,
        "low_soc_diversions": 2,
        "safety_invariant_errors": 0,
        "uncertified_execution": 0,
        "hard_separation_violations": 0,
        "nmac": 0,
        "collisions": 0,
        "exception": exception,
        "replay": {"exact": exact, "steps_replayed": 10, "steps_expected": 10},
    }


def test_incomplete_and_error_records_fail_closed() -> None:
    incomplete = pilot._with_summary(_record(terminal=False, completed=12))
    errored = pilot._with_summary(_record(status="failed", exception="boom"))

    assert incomplete["complete"] is False
    assert errored["complete"] is False
    result = pilot.aggregate_records([incomplete, errored], expected_record_count=2)
    assert result["diagnostic_pass"] is False
    assert result["gates"]["all_records_complete"] is False
    assert result["gates"]["no_record_errors"] is False


def test_run_case_passes_high_soc_reward_config_to_both_controllers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def summary_for(config, output_dir: Path) -> dict:
        output_dir.mkdir(parents=True, exist_ok=False)
        return {
            "action_tape": "synthetic-tape.json",
            "requests": 24,
            "completed": 24,
            "terminal": True,
            "outcomes": {},
            "safety": {},
            "seconds": 1.0,
        }

    def fake_episode(config, seed, output_dir, *, policy, max_steps):
        del seed, policy, max_steps
        captured["rule"] = config
        return summary_for(config, output_dir)

    def fake_rollout(
        policy, config, *, seed, max_steps, output, deterministic, collect_labels
    ):
        del policy, seed, max_steps, deterministic, collect_labels
        captured["original_k3"] = config
        return [], summary_for(config, output)

    monkeypatch.setattr(pilot.safety_energy.curriculum, "episode", fake_episode)
    monkeypatch.setattr(
        pilot.safety_energy.student_rollout_v4_2, "rollout", fake_rollout
    )
    monkeypatch.setattr(
        pilot.safety_energy,
        "_replay_telemetry",
        lambda config, tape: ({}, {"exact": True}, None, None),
    )
    raw = pilot.load_scenario_protocol()
    for controller in ("rule", "original_k3"):
        case = {
            "case_id": f"dense24_seed901_safe80_{controller}",
            "scenario": "dense24",
            "seed": 901,
            "safe_distance": 80.0,
            "controller": controller,
        }
        result = pilot.run_case(
            case,
            output=tmp_path,
            policy=object(),
            raw_protocol=raw,
            max_steps=10,
        )
        assert result["status"] == "completed"

    assert set(captured) == {"rule", "original_k3"}
    for config in captured.values():
        assert config.initial_soc_mode == "uniform"
        assert config.initial_soc_min == pytest.approx(0.90)
        assert config.initial_soc_max == pytest.approx(1.00)
        assert config.low_level_priority_time_weight == pytest.approx(1.0)
        assert config.low_level_soc_urgency_time_weight == pytest.approx(2.0)
        assert config.low_level_soc_urgency_threshold == pytest.approx(0.65)


def test_run_serializes_progress_and_manifest_with_paired_summaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "original_k3.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(pilot.effectiveness, "_sha256", lambda path: pilot.ORIGINAL_K3_SHA256)
    monkeypatch.setattr(pilot.effectiveness, "load_candidate_checkpoint", lambda path: (object(), {}))
    calls: list[object] = []

    def fake_run_case(case, *, output, policy, raw_protocol, max_steps):
        calls.append(policy)
        assert raw_protocol["attributes"]["initial_soc_min"] == pytest.approx(0.90)
        assert raw_protocol["attributes"]["initial_soc_max"] == pytest.approx(1.00)
        assert case["safe_distance"] == 80.0
        return {
            **case,
            "status": "completed",
            "terminal": True,
            "requests": 24,
            "completed": 24,
            "unresolved": 0,
            "low_soc_diversions": 0,
            "safety_invariant_errors": 0,
            "uncertified_execution": 0,
            "hard_separation_violations": 0,
            "nmac": 0,
            "collisions": 0,
            "exception": None,
            "replay": {"exact": True, "steps_replayed": 1, "steps_expected": 1},
        }

    monkeypatch.setattr(pilot, "run_case", fake_run_case)
    output = tmp_path / "pilot"
    manifest = pilot.run_matrix(
        output=output,
        checkpoint=checkpoint,
        seeds=[901],
        max_steps=10,
    )

    assert len(calls) == 2
    assert calls[0] is calls[1]
    assert manifest["PASS"] is True
    assert manifest["diagnostic_only"] is True
    assert manifest["qualification"] is False
    assert manifest["PPO_ALLOWED"] is False
    assert manifest["aggregate"]["completed"] == 48
    assert manifest["aggregate"]["requests"] == 48
    assert manifest["aggregate"]["diversions"] == 0
    assert (output / "progress.json").is_file()
    assert (output / "manifest.json").is_file()
    serialized = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert len(serialized["records"]) == 2
    assert all("complete" in row and "safety_summary" in row and "replay_summary" in row
               for row in serialized["records"])
