from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from terminal_marl.training import diagnose_emergency_priority_guard_pilot_v1 as runner


def _historical_source_hashes() -> dict[Path, str]:
    snapshot = runner.yaml.safe_load(runner.PROTOCOL_PATH.read_text(encoding="utf-8-sig"))
    return {
        (runner.PROJECT_ROOT / relative).resolve(): str(expected).lower()
        for relative, expected in snapshot["source_bindings"].items()
    }


def test_protocol_and_guard_config_bindings_are_frozen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This historical snapshot test replaces current source hashes with the
    # frozen V1 values for protocol structure and serialization coverage. The
    # V1 runner itself remains fail-closed when current hashes differ.
    real_sha = runner._sha256
    historical = _historical_source_hashes()
    monkeypatch.setattr(
        runner,
        "_sha256",
        lambda path: historical.get(Path(path).resolve(), real_sha(path)),
    )
    protocol = runner.load_protocol()
    scenario = protocol["scenario"]
    assert tuple(protocol["matrix"]["seeds"]) == (840401, 840402)
    assert protocol["matrix"]["controllers"] == ["rule", "original_k3"]
    assert protocol["matrix"]["guard_modes"] == [False, True]
    assert protocol["matrix"]["expected_records"] == 8
    assert protocol["smoke"] == {"seed": 840400, "steps": 12, "formal_marker": False}
    off = runner.build_config(protocol, guard_enabled=False)
    on = runner.build_config(protocol, guard_enabled=True)
    assert off.emergency_priority_guard_enabled is False
    assert on.emergency_priority_guard_enabled is True
    assert on.action_schema_version == runner.ACTION_SCHEMA
    assert on.observation_schema_version == runner.OBSERVATION_SCHEMA
    assert on.runtime_proposal_shortlist_k == 3
    assert on.total_requested_uavs == 12
    assert on.fixed_arrival_schedule == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0)
    assert on.energy_model_mode == "phase_motion_v2"
    assert on.emergency_landing_enabled is True
    assert on.emergency_trigger_soc == pytest.approx(0.40)
    assert scenario["initial_soc_min"] == pytest.approx(0.35)
    assert scenario["initial_soc_max"] == pytest.approx(0.55)


def _record(controller: str, guard_enabled: bool, *, failed: int, delay: float) -> dict:
    return {
        "controller": controller,
        "seed": 840401,
        "guard_enabled": guard_enabled,
        "failed_energy": failed,
        "emergency_landed": 3,
        "completed": 5,
        "successes": 8,
        "upstream_low_soc_diversions": 1,
        "in_terminal_low_soc_diversions": 0,
        "all_request_horizon_penalized_delay": {"mean_seconds": delay},
        "safety": {key: 0 for key in runner.SAFETY_COUNTER_KEYS},
    }


def test_paired_delta_is_guard_on_minus_off() -> None:
    rows = runner.paired_deltas([
        _record("rule", False, failed=2, delay=100.0),
        _record("rule", True, failed=0, delay=80.0),
    ])
    assert rows == [{
        "seed": 840401,
        "controller": "rule",
        "guard_on_minus_off": {
            "failed_energy": -2,
            "emergency_landed": 0,
            "completed": 0,
            "successes": 0,
            "upstream_low_soc_diversions": 0,
            "in_terminal_low_soc_diversions": 0,
            "all_request_horizon_penalized_delay_mean_seconds": -20.0,
            "safety": {key: 0 for key in runner.SAFETY_COUNTER_KEYS},
        },
        "guard_telemetry": None,
        "guard_telemetry_available": False,
    }]


def test_guard_off_failure_is_descriptive_and_does_not_fail_case_gate() -> None:
    base = {
        "failed_energy": 2,
        "hard_gates": {
            "exact_request_accounting": True,
            "all_records_terminal": True,
            "zero_safety_events": True,
            "zero_failed_energy": False,
            "zero_in_terminal_low_soc_diversion": True,
            "exact_action_replay": True,
            "no_teacher": True,
            "no_labels": True,
            "no_updates": True,
            "runtime_shortlist_k3": True,
            "all": False,
        },
    }
    off = runner._guard_record(base, guard_enabled=False)
    assert off["zero_failed_energy_observed"] is False
    assert off["guard_off_failed_energy_descriptive_only"] is True
    assert "guard_off_failed_energy_descriptive_only" not in off["hard_gates"]
    assert off["hard_gates"]["zero_failed_energy"] is True
    assert off["hard_gates"]["guard_on_zero_failed_energy"] is True
    assert off["hard_gates"]["all"] is True

    on = runner._guard_record({**base, "failed_energy": 0}, guard_enabled=True)
    assert on["hard_gates"]["all"] is True
    on_failed = runner._guard_record(base, guard_enabled=True)
    assert on_failed["hard_gates"]["all"] is False


def test_smoke_path_serializes_four_cases_and_refuses_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"test checkpoint")
    real_sha = runner._sha256
    historical = _historical_source_hashes()

    def fake_sha(path: str | Path) -> str:
        resolved = Path(path).resolve()
        if resolved == runner.PROTOCOL_PATH.resolve():
            return runner.PROTOCOL_SHA256
        if resolved == checkpoint.resolve():
            return runner.CHECKPOINT_SHA256
        if resolved in historical:
            return historical[resolved]
        return real_sha(path)

    monkeypatch.setattr(runner, "_sha256", fake_sha)

    class DummyPolicy:
        optimizer = SimpleNamespace(state={})

    monkeypatch.setattr(
        runner.paired.effectiveness,
        "load_candidate_checkpoint",
        lambda path, expected_sha256: (DummyPolicy(), {"stage": "test", "architecture_id": "test", "qualification_pass": True, "PPO_ALLOWED": False}),
    )

    def fake_case(policy, config, seed, controller, guard_enabled, output, *, max_steps):
        output.mkdir(parents=False, exist_ok=False)
        return {
            "controller": controller,
            "seed": seed,
            "guard_enabled": guard_enabled,
            "failed_energy": 0,
            "emergency_landed": 0,
            "completed": 0,
            "successes": 0,
            "upstream_low_soc_diversions": 0,
            "in_terminal_low_soc_diversions": 0,
            "outcome_counts": {},
            "safety": {key: 0 for key in runner.SAFETY_COUNTER_KEYS},
            "hard_gates": {key: True for key in (
                "exact_request_accounting", "all_records_terminal", "zero_safety_events",
                "zero_in_terminal_low_soc_diversion", "exact_action_replay", "no_teacher",
                "no_labels", "no_updates", "runtime_shortlist_k3",
            )},
            "all_request_horizon_penalized_delay": {"mean_seconds": 0.0},
        }

    monkeypatch.setattr(runner, "_run_case", fake_case)
    output = tmp_path / "smoke"
    manifest = runner.run(checkpoint=checkpoint, output=output, smoke=True)
    assert manifest["smoke"] is True
    assert manifest["PPO_ALLOWED"] is False
    assert len(manifest["records"]) == 4
    assert {record["guard_enabled"] for record in manifest["records"]} == {False, True}
    assert (output / "manifest.json").is_file()
    with pytest.raises(runner.ProtocolError, match="overwrite"):
        runner.run(checkpoint=checkpoint, output=output, smoke=True)
