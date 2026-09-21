import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from terminal_marl.core.safety_trace import SafetyTraceRecorder
from terminal_marl.training import diagnose_v4_2_p3_q3_dense_stall_timeline_v1 as diagnostic


Q3_OUTPUT = Path(
    r"D:\7_codespace\training_outputs\terminal\p3_v3_early_stop_q3_v1_20260921"
)
Q3_MANIFEST = Q3_OUTPUT / "manifest.json"
ACTION_TAPE = (
    Q3_OUTPUT
    / "trained_dense24_826006"
    / "action_tapes"
    / "safety_trace_seed_826006_step_1079.json"
)


def _timeline_row(time_before: int, *, active: int = 1) -> dict:
    return {
        "step": time_before,
        "time_before": float(time_before),
        "time_after": float(time_before + 1),
        "active_count": active,
        "phase_counts": {"inner": active},
        "completed": time_before // 60,
        "arrivals_generated": time_before // 30,
        "queue_size": time_before % 3,
        "landing_zones": {"free": 4, "reserved": 1, "occupied": 1},
        "requested_release": 2,
        "requested_wait": 1,
        "release_approved": 1,
        "release_safety_reject": 0,
        "release_safety_wait": 1,
        "shield_count": 0,
        "release_decision_counts": {"approved": 1, "safety_wait": 1},
        "primary_blockers": {"headway": 1},
        "secondary_constraints": {"capacity": 1},
    }


def test_protocol_is_frozen_and_diagnostic_only():
    value = diagnostic.protocol()

    assert diagnostic.PROTOCOL_SHA != "TO_BE_FROZEN"
    assert diagnostic._sha(diagnostic.PROTOCOL) == diagnostic.PROTOCOL_SHA
    assert value["protocol"] == diagnostic.STAGE
    assert value["status"] == "PREREGISTERED_READ_ONLY_REPLAY_DIAGNOSTIC"
    assert value["bindings"]["seed"] == 826006
    assert value["bindings"]["scenario"] == "dense24"
    assert value["bindings"]["expected_steps"] == 1080
    assert value["authorization"] == {
        "diagnostic_only": True,
        "PASS_used_for_qualification": False,
        "checkpoint_promotable": False,
        "P3_QUALIFIED": False,
        "P4_ALLOWED": False,
        "no_seed_reservation": True,
        "no_seed_consumption": True,
    }


def test_source_hashes_are_frozen_and_cover_replay_semantics():
    hashes = diagnostic.source_hashes()

    assert hashes == diagnostic.SOURCE_EXPECTED
    assert all(len(value) == 64 for value in hashes.values())
    assert {
        "terminal_marl/core/safety_trace.py",
        "terminal_marl/envs/parallel_env.py",
        "terminal_marl/core/rewards.py",
        "terminal_marl/core/config.py",
    } <= set(hashes)


def test_fixed_time_bins_aggregate_observed_counts():
    bins = diagnostic.protocol()["time_bins"]
    rows = [_timeline_row(int(spec["start"])) for spec in bins]

    result = diagnostic._aggregate_bins(rows, bins)

    assert [item["name"] for item in result] == [
        "0-180", "180-380", "380-600", "600-800", "800-960", "960-1080",
    ]
    assert [item["steps"] for item in result] == [1] * len(bins)
    assert all(item["requested_release"] == 2 for item in result)
    assert all(item["release_approved"] == 1 for item in result)
    assert all(item["decision_counts"] == {"approved": 1, "safety_wait": 1} for item in result)
    assert all(item["primary_blockers"] == {"headway": 1} for item in result)
    assert all(item["secondary_constraints"] == {"capacity": 1} for item in result)


def test_input_validation_fails_closed_without_replay(tmp_path):
    protocol = diagnostic.protocol()
    manifest = tmp_path / "manifest.json"
    tape = tmp_path / "tape.json"
    manifest.write_text(json.dumps({"stage": diagnostic.STAGE}), encoding="utf-8")
    tape.write_text(json.dumps({"episode_seed": diagnostic.SEED}), encoding="utf-8")

    with pytest.raises(diagnostic.TimelineInputError):
        diagnostic._validate_q3_manifest(manifest, protocol)
    with pytest.raises(diagnostic.TimelineInputError):
        diagnostic._validate_action_tape(tape, protocol, {})


def test_synthetic_early_observation_mismatch_is_replay_error(monkeypatch):
    config = SimpleNamespace()

    class FakeEnv:
        def __init__(self, passed_config):
            self.config = passed_config
            self.step_count = 0
            self.now = 0.0
            self._observations_value = {"0": np.asarray([1.0], dtype=np.float32)}

        def reset(self, seed):
            assert seed == diagnostic.SEED
            return self._observations_value, {}

    monkeypatch.setattr(diagnostic, "TerminalParallelEnv", FakeEnv)
    monkeypatch.setattr(
        SafetyTraceRecorder,
        "_state",
        staticmethod(lambda env: {"fake": True}),
    )
    monkeypatch.setattr(
        SafetyTraceRecorder,
        "_apply_reset_recipe",
        staticmethod(lambda env, recipe: env._observations_value),
    )
    recipe = {"kind": "generic_reset", "steps": []}
    payload = {
        "reset_recipe": recipe,
        "reset_recipe_hash": diagnostic._digest(recipe),
        "reset_state_hash": diagnostic._digest({"fake": True}),
        "environment_config_hash": diagnostic._digest(vars(config)),
        "action_tape": [{
            "step": 0,
            "time_before": 0.0,
            "observation_hash": "0" * 64,
        }],
    }

    with pytest.raises(diagnostic.TimelineReplayError, match="observation hash mismatch"):
        diagnostic._replay_timeline(config, payload, {"time_bins": diagnostic.protocol()["time_bins"]})


@pytest.mark.skipif(
    not (Q3_MANIFEST.is_file() and ACTION_TAPE.is_file()),
    reason="synchronized Q3 evidence is unavailable",
)
def test_real_q3_manifest_and_tape_validate_without_environment_reset():
    protocol = diagnostic.protocol()
    q3 = diagnostic._validate_q3_manifest(Q3_MANIFEST, protocol)
    payload = diagnostic._validate_action_tape(ACTION_TAPE, protocol, q3)

    assert payload["episode_seed"] == diagnostic.SEED
    assert len(payload["action_tape"]) == diagnostic.EXPECTED_STEPS


def test_run_is_read_only_replay_and_never_promotes():
    source = inspect.getsource(diagnostic.run)

    assert "_replay_timeline" in source
    assert '"diagnostic_only": True' in source
    assert '"checkpoint_promotable": False' in source
    assert '"P3_QUALIFIED": False' in source
    assert '"P4_ALLOWED": False' in source
    assert "reserve_once" not in source
    assert "torch.optim" not in source
