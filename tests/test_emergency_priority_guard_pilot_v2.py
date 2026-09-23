from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from terminal_marl.training import diagnose_emergency_priority_guard_pilot_v2 as runner


def test_protocol_and_guard_on_off_bindings_are_frozen() -> None:
    protocol = runner.load_protocol()
    scenario = protocol["scenario"]
    assert tuple(protocol["matrix"]["seeds"]) == (840501, 840502)
    assert protocol["matrix"]["controllers"] == ["rule", "original_k3"]
    assert protocol["matrix"]["guard_modes"] == [False, True]
    assert protocol["matrix"]["expected_records"] == 8
    assert protocol["smoke"] == {"seed": 840500, "steps": 12, "formal_marker": False}
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
        "seed": 840501,
        "guard_enabled": guard_enabled,
        "failed_energy": failed,
        "emergency_landed": 2,
        "completed": 6,
        "successes": 8,
        "upstream_low_soc_diversions": 1,
        "in_terminal_low_soc_diversions": 0,
        "all_request_horizon_penalized_delay": {"mean_seconds": delay},
        "safety": {key: 0 for key in runner.SAFETY_COUNTER_KEYS},
    }


def test_paired_delta_is_guard_on_minus_off() -> None:
    rows = runner.paired_deltas([
        _record("original_k3", False, failed=1, delay=120.0),
        _record("original_k3", True, failed=0, delay=90.0),
    ])
    assert rows == [{
        "seed": 840501,
        "controller": "original_k3",
        "guard_on_minus_off": {
            "failed_energy": -1,
            "emergency_landed": 0,
            "completed": 0,
            "successes": 0,
            "upstream_low_soc_diversions": 0,
            "in_terminal_low_soc_diversions": 0,
            "all_request_horizon_penalized_delay_mean_seconds": -30.0,
            "safety": {key: 0 for key in runner.SAFETY_COUNTER_KEYS},
        },
        "guard_telemetry": None,
        "guard_telemetry_available": False,
    }]


def test_smoke_serializes_four_guard_controller_cases_and_refuses_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"test checkpoint")
    real_sha = runner._sha256

    def fake_sha(path: str | Path) -> str:
        resolved = Path(path).resolve()
        if resolved == runner.PROTOCOL_PATH.resolve():
            return runner.PROTOCOL_SHA256
        if resolved == checkpoint.resolve():
            return runner.CHECKPOINT_SHA256
        return real_sha(path)

    monkeypatch.setattr(runner, "_sha256", fake_sha)

    class DummyPolicy:
        optimizer = SimpleNamespace(state={})

    monkeypatch.setattr(
        runner.v1.paired.effectiveness,
        "load_candidate_checkpoint",
        lambda path, expected_sha256: (
            DummyPolicy(),
            {
                "stage": "test",
                "architecture_id": "test",
                "qualification_pass": True,
                "PPO_ALLOWED": False,
            },
        ),
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
            "all_request_horizon_penalized_delay": {"mean_seconds": 0.0},
        }

    monkeypatch.setattr(runner, "_run_case", fake_case)
    output = tmp_path / "smoke"
    manifest = runner.run(checkpoint=checkpoint, output=output, smoke=True)
    assert manifest["smoke"] is True
    assert manifest["PPO_ALLOWED"] is False
    assert manifest["seeds"] == [840500]
    assert len(manifest["records"]) == 4
    assert len(manifest["paired_deltas"]) == 2
    assert {record["controller"] for record in manifest["records"]} == {"rule", "original_k3"}
    assert {record["guard_enabled"] for record in manifest["records"]} == {False, True}
    assert manifest["protocol_sha256"] == runner.PROTOCOL_SHA256
    assert (output / "manifest.json").is_file()
    with pytest.raises(runner.ProtocolError, match="overwrite"):
        runner.run(checkpoint=checkpoint, output=output, smoke=True)


def test_main_smoke_prints_summary_and_returns_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = {"aggregate": {"PASS": False}, "smoke": True, "records": [{"seed": 840500}]}
    monkeypatch.setattr(runner, "run", lambda **kwargs: manifest)
    code = runner.main([
        "--checkpoint", str(tmp_path / "checkpoint.pt"),
        "--output", str(tmp_path / "output"),
        "--smoke",
    ])
    assert code == 0
    assert json.loads(capsys.readouterr().out) == {
        "PASS": False,
        "records": 1,
        "smoke": True,
    }
