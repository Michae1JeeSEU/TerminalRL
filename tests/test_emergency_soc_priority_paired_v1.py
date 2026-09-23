from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from terminal_marl.training import diagnose_emergency_soc_priority_paired_v1 as runner


def test_protocol_and_config_bindings_are_frozen() -> None:
    protocol = runner.load_protocol()
    scenario = protocol["scenario"]
    assert tuple(protocol["matrix"]["seeds"]) == (840301, 840302)
    assert protocol["matrix"]["controllers"] == ["rule", "original_k3"]
    assert protocol["smoke"] == {"seed": 840300, "steps": 12, "formal_marker": False}
    config = runner.build_config(protocol)
    assert config.action_schema_version == runner.ACTION_SCHEMA
    assert config.observation_schema_version == runner.OBSERVATION_SCHEMA
    assert config.runtime_proposal_shortlist_k == 3
    assert config.total_requested_uavs == 12
    assert config.fixed_arrival_schedule == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0)
    assert config.energy_model_mode == "phase_motion_v2"
    assert config.phase_motion_nominal_solo_energy == pytest.approx(240.9)
    assert config.emergency_landing_enabled is True
    assert config.emergency_trigger_soc == pytest.approx(0.40)
    assert config.teacher_enabled is False
    assert config.high_level_enabled is False
    assert scenario["initial_soc_min"] == pytest.approx(0.35)
    assert scenario["initial_soc_max"] == pytest.approx(0.55)


def test_horizon_penalized_delay_charges_unresolved_requests() -> None:
    result = runner._horizon_penalized_delays(
        {0: 0.0, 1: 15.0, 2: 30.0},
        {0: 100.0},
        horizon_seconds=1200.0,
    )
    assert result["per_request_seconds"] == {"0": 100.0, "1": 1185.0, "2": 1170.0}
    assert result["requests"] == 3
    assert result["mean_seconds"] == pytest.approx((100.0 + 1185.0 + 1170.0) / 3.0)


def test_paired_delta_is_original_k3_minus_rule() -> None:
    def record(controller: str, delay: float, successes: int) -> dict:
        return {
            "controller": controller,
            "seed": 840301,
            "successes": successes,
            "emergency_landed": 1,
            "upstream_low_soc_diversions": 2,
            "in_terminal_low_soc_diversions": 0,
            "failed_energy": 0,
            "all_request_horizon_penalized_delay": {"mean_seconds": delay},
            "normal_touchdown_delay_seconds": {"mean": 20.0},
            "emergency_touchdown_delay_seconds": {"mean": 30.0},
        }

    rows = runner.paired_deltas([record("rule", 80.0, 8), record("original_k3", 100.0, 7)])
    assert rows == [{
        "seed": 840301,
        "original_k3_minus_rule": {
            "all_request_horizon_penalized_delay_mean_seconds": 20.0,
            "normal_touchdown_delay_mean_seconds": 0.0,
            "emergency_touchdown_delay_mean_seconds": 0.0,
            "successes": -1,
            "emergency_landed": 0,
            "upstream_low_soc_diversions": 0,
            "in_terminal_low_soc_diversions": 0,
            "failed_energy": 0,
        },
    }]


def test_smoke_path_serializes_manifest_and_refuses_overwrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"test checkpoint")
    protocol_sha = runner.PROTOCOL_SHA256
    real_sha = runner._sha256

    def fake_sha(path: str | Path) -> str:
        resolved = Path(path).resolve()
        if resolved == runner.PROTOCOL_PATH.resolve():
            return protocol_sha
        if resolved == checkpoint.resolve():
            return runner.CHECKPOINT_SHA256
        return real_sha(path)

    monkeypatch.setattr(runner, "_sha256", fake_sha)

    class DummyPolicy:
        optimizer = SimpleNamespace(state={})

    monkeypatch.setattr(
        runner.effectiveness,
        "load_candidate_checkpoint",
        lambda path, expected_sha256: (DummyPolicy(), {
            "stage": "V4_2_P2_GATE_RESIDUAL_ADAPTER_REPAIR_V3",
            "architecture_id": "test",
            "qualification_pass": True,
            "PPO_ALLOWED": False,
        }),
    )

    def fake_case(policy, config, seed, controller, output, *, max_steps):
        output.mkdir(parents=True, exist_ok=False)
        return {
            "controller": controller,
            "seed": seed,
            "successes": 0,
            "emergency_landed": 0,
            "upstream_low_soc_diversions": 0,
            "in_terminal_low_soc_diversions": 0,
            "failed_energy": 0,
            "outcome_counts": {},
            "safety": {key: 0 for key in runner.SAFETY_COUNTER_KEYS},
            "hard_gates": {key: False for key in (
                "exact_request_accounting", "all_records_terminal", "zero_safety_events",
                "zero_failed_energy", "zero_in_terminal_low_soc_diversion",
                "exact_action_replay", "no_teacher", "no_labels", "no_updates",
                "runtime_shortlist_k3",
            )},
            "all_request_horizon_penalized_delay": {"mean_seconds": 0.0},
            "normal_touchdown_delay_seconds": {"mean": None},
            "emergency_touchdown_delay_seconds": {"mean": None},
        }

    monkeypatch.setattr(runner, "_run_case", fake_case)
    output = tmp_path / "smoke"
    manifest = runner.run(checkpoint=checkpoint, output=output, smoke=True)
    assert manifest["smoke"] is True
    assert manifest["PPO_ALLOWED"] is False
    assert len(manifest["records"]) == 2
    assert (output / "manifest.json").is_file()
    with pytest.raises(runner.ProtocolError, match="overwrite"):
        runner.run(checkpoint=checkpoint, output=output, smoke=True)
