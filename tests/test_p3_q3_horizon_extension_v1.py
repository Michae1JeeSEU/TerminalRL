from pathlib import Path
from types import SimpleNamespace

import pytest

from terminal_marl.core.runtime_traffic_safety import SafetyInvariantError
from terminal_marl.training import diagnose_v4_2_p3_q3_horizon_extension_v1 as diagnostic


def test_protocol_binds_exposed_seed_cap_and_nonpromotion():
    value = diagnostic.protocol()

    assert value["protocol"] == diagnostic.STAGE
    assert value["status"] == diagnostic.STATUS
    assert value["bindings"]["seed"] == 826006
    assert value["bindings"]["scenario"] == "dense24"
    assert value["bindings"]["extended_horizon_seconds"] == 1800.0
    assert value["scenario"]["same_config_except_max_simulation_seconds"] is True
    assert value["measurement"]["completed_only_waits"] is True
    assert value["authorization"]["diagnostic_only"] is True
    assert value["authorization"]["P3_QUALIFIED"] is False
    assert value["authorization"]["checkpoint_promotable"] is False
    assert value["authorization"]["P4_ALLOWED"] is False


def test_extended_config_changes_only_simulation_cap():
    base, extended = diagnostic._extended_config()

    assert base.max_simulation_seconds == 1080.0
    assert extended.max_simulation_seconds == 1800.0
    assert extended.dt == base.dt == 1.0
    assert extended.total_requested_uavs == 24
    assert extended.low_level_priority_time_weight == 1.0
    assert extended.low_level_soc_urgency_time_weight == 2.0
    assert extended.low_level_soc_urgency_threshold == 0.65


def test_checkpoint_hash_and_output_preflight_fail_closed(tmp_path, monkeypatch):
    trained = tmp_path / "trained.pt"
    initial = tmp_path / "initial.pt"
    trained.write_bytes(b"trained")
    initial.write_bytes(b"initial")
    output = tmp_path / "output"
    monkeypatch.setattr(diagnostic, "_sha", lambda path: "0" * 64)

    with pytest.raises(diagnostic.HorizonExtensionInputError, match="trained checkpoint SHA mismatch"):
        diagnostic._validate_checkpoint_inputs(trained, initial, output)
    assert not output.exists()


def test_completed_waits_are_labeled_censored():
    uav = SimpleNamespace(
        external_wait_time=3.0,
        internal_wait_time=5.0,
        touchdown_time=12.0,
        request_time=2.0,
    )

    waits = diagnostic._completed_waits([uav])

    assert waits["completed_count"] == 1
    assert waits["completed_only"] is True
    assert waits["censored"] is True
    assert waits["mean_wait_seconds"] == 8.0
    assert waits["mean_request_to_touchdown_seconds"] == 10.0


def test_safety_error_stops_one_controller_but_run_continues_and_never_promotes(
    tmp_path, monkeypatch
):
    trained = tmp_path / "trained.pt"
    initial = tmp_path / "initial.pt"
    trained.write_bytes(b"trained")
    initial.write_bytes(b"initial")
    output = tmp_path / "output"
    calls = []

    monkeypatch.setattr(
        diagnostic,
        "_sha",
        lambda path: (
            diagnostic.TRAINED_CHECKPOINT_SHA
            if Path(path).name == "trained.pt"
            else diagnostic.INITIAL_CHECKPOINT_SHA
        ),
    )
    monkeypatch.setattr(
        diagnostic.q3,
        "protocol",
        lambda: {
            "protocol": diagnostic.q3.STAGE,
            "bindings": {
                "trained_checkpoint_sha256": diagnostic.TRAINED_CHECKPOINT_SHA,
                "initial_checkpoint_sha256": diagnostic.INITIAL_CHECKPOINT_SHA,
                "training_protocol_sha256": diagnostic.TRAINING_PROTOCOL_SHA,
            },
        },
    )
    monkeypatch.setattr(
        diagnostic.q3,
        "load_trained_checkpoint",
        lambda path, p: ("trained-policy", {"stage": "trained"}),
    )
    monkeypatch.setattr(
        diagnostic.q3,
        "load_initial_checkpoint",
        lambda path, p: ("initial-policy", {"stage": "initial"}),
    )
    config = SimpleNamespace(dt=1.0, max_simulation_seconds=1800.0)
    monkeypatch.setattr(diagnostic, "_extended_config", lambda: (SimpleNamespace(max_simulation_seconds=1080.0), config))
    monkeypatch.setattr(
        diagnostic,
        "_lifecycle_snapshot",
        lambda config, tape: {
            "replay_exact": True,
            "replay_steps_replayed": 1,
            "replay_steps_expected": 1,
            "replay_error": None,
            "first_touchdown_seconds": 10.0,
            "last_touchdown_seconds": 10.0,
            "final_active": 0,
            "final_queue": 0,
            "arrivals_generated": 24,
            "completed_after_replay": 24,
            "waits": {
                "completed_count": 24,
                "completed_only": True,
                "censored": False,
                "mean_wait_seconds": 2.0,
                "mean_request_to_touchdown_seconds": 3.0,
            },
        },
    )

    def fake_rollout(policy, config, *, seed, max_steps, output, deterministic, collect_labels):
        calls.append(policy)
        output.mkdir(parents=True, exist_ok=False)
        if policy == "trained-policy":
            raise SafetyInvariantError("synthetic safety stop")
        tape = output / "action_tape.json"
        tape.write_text("{}", encoding="utf-8")
        return [], {
            "requests": 24,
            "completed": 24,
            "terminal": True,
            "seconds": 10.0,
            "safety": {"SafetyInvariantError": 0},
            "safety_error": None,
            "action_tape": str(tape),
        }

    def fake_rule(config, seed, output, *, max_steps):
        calls.append("rule")
        output.mkdir(parents=True, exist_ok=False)
        tape = output / "action_tape.json"
        tape.write_text("{}", encoding="utf-8")
        return {
            "requests": 24,
            "completed": 24,
            "terminal": True,
            "seconds": 10.0,
            "safety": {"SafetyInvariantError": 0},
            "safety_error": None,
            "action_tape": str(tape),
        }

    monkeypatch.setattr(diagnostic.student_rollout_v4_2, "rollout", fake_rollout)
    monkeypatch.setattr(diagnostic, "rule_episode", fake_rule)

    result = diagnostic.run(
        trained_checkpoint=trained,
        initial_checkpoint=initial,
        output=output,
    )

    assert calls == ["trained-policy", "initial-policy", "rule"]
    assert len(result["records"]) == 3
    assert result["PASS"] is False
    assert result["checkpoint_promotable"] is False
    assert result["P3_QUALIFIED"] is False
    assert result["P4_ALLOWED"] is False
    assert result["records"][0]["safety_error"] == "synthetic safety stop"
    assert result["records"][0]["run_integrity"] is False
    assert result["records"][1]["run_integrity"] is True
    assert result["records"][2]["run_integrity"] is True
