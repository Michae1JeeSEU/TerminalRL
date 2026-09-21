import inspect
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from terminal_marl.training import (
    run_v4_2_p3_mixed_priority_soc_training_v4 as training,
)


def test_v4_protocol_binds_original_checkpoint_and_frozen_dense_schedule():
    value = training.protocol()
    assert value["protocol"] == training.STAGE
    assert value["status"] == training.STATUS
    assert value["bindings"]["initial_checkpoint_sha256"] == training.INITIAL_CHECKPOINT_SHA
    assert value["bindings"]["superseded_training_manifest_sha256"] == training.V3_ROOT_MANIFEST_SHA
    assert value["bindings"]["forbidden_v3_checkpoint_sha256"] == training.V3_FINAL_CHECKPOINT_SHA
    assert value["scenarios"]["dense24"]["full_horizon_seconds"] == 1800
    assert value["scenarios"]["flow6"]["full_horizon_seconds"] == 1210
    assert value["scenarios"]["flow12"]["full_horizon_seconds"] == 1570


def test_v4_seed_order_is_fresh_and_heldout_is_not_used_by_training():
    value = training.protocol()
    scheduled = (
        value["entry"]["seeds"] + value["formal"]["seeds"]
        + value["dev"]["seeds"] + value["heldout"]["seeds"]
    )
    assert len(scheduled) == len(set(scheduled))
    assert not set(scheduled) & set(value["history"]["forbidden_seeds"])
    assert not set(value["heldout"]["seeds"]) & set(value["formal"]["seeds"])
    assert value["heldout"]["untouched"] is True
    assert value["reservation"]["reserve_pipeline_and_heldout_before_first_reset"] is True


def test_v4_scenario_horizons_and_exact_urgency_reward_values():
    value = training.protocol()
    expected = {"dense24": 1800.0, "flow6": 1210.0, "flow12": 1570.0}
    for name, horizon in expected.items():
        config = training.scenario_config(name, value)
        assert config.max_simulation_seconds == horizon
        assert config.low_level_priority_time_weight == 1.0
        assert config.low_level_soc_urgency_time_weight == 2.0
        assert config.low_level_soc_urgency_threshold == 0.65
        configured, steps = training._scenario_horizon(
            config, name, stage="formal", index=0, p=value,
        )
        assert configured.max_simulation_seconds == horizon
        assert steps == int(horizon / configured.dt)

    entry_config = training.scenario_config("dense24", value)
    short, steps = training._scenario_horizon(
        entry_config, "dense24", stage="entry", index=0, p=value,
    )
    assert steps == 320
    assert short.max_simulation_seconds == 320.0


def test_v4_initialization_is_original_and_fresh_for_entry_and_formal():
    value = training.protocol()
    entry = training.stage_initialization_contract("entry", value)
    formal = training.stage_initialization_contract("formal", value)
    assert entry["checkpoint_sha256"] == training.INITIAL_CHECKPOINT_SHA
    assert formal["checkpoint_sha256"] == training.INITIAL_CHECKPOINT_SHA
    assert entry["fresh_optimizer"] is True
    assert formal["fresh_optimizer"] is True
    assert entry["entry_updates_discarded"] is True
    assert formal["formal_initialization_is_original"] is True
    assert training.INITIAL_CHECKPOINT_SHA != training.V3_FINAL_CHECKPOINT_SHA


def test_v4_selection_requires_first_complete_strict_priority_improvement():
    initial = {
        "completion_fraction": 1.0,
        "priority_weighted_wait_seconds": 100.0,
        "low_soc_mean_wait_seconds": 60.0,
    }
    candidate = {
        "completion_fraction": 1.0,
        "priority_weighted_wait_seconds": 99.0,
        "low_soc_mean_wait_seconds": 60.0,
    }
    passed, gates = training._dev_pass(initial, candidate)
    assert passed is True
    assert all(gates.values())

    equal_priority = dict(candidate, priority_weighted_wait_seconds=100.0)
    passed, gates = training._dev_pass(initial, equal_priority)
    assert passed is False
    assert gates["priority_wait_strictly_better"] is False


def test_v4_incomplete_dev_cannot_pass_despite_lower_wait():
    initial = {
        "completion_fraction": 1.0,
        "priority_weighted_wait_seconds": 100.0,
        "low_soc_mean_wait_seconds": 60.0,
    }
    incomplete = {
        "completion_fraction": 23 / 24,
        "priority_weighted_wait_seconds": 1.0,
        "low_soc_mean_wait_seconds": 1.0,
    }
    passed, gates = training._dev_pass(initial, incomplete)
    assert passed is False
    assert gates["candidate_complete"] is False
    assert gates["priority_wait_strictly_better"] is True


def test_v4_missing_metric_fails_closed():
    initial = {
        "completion_fraction": 1.0,
        "priority_weighted_wait_seconds": 100.0,
        "low_soc_mean_wait_seconds": None,
    }
    candidate = {
        "completion_fraction": 1.0,
        "priority_weighted_wait_seconds": 99.0,
        "low_soc_mean_wait_seconds": 1.0,
    }
    passed, gates = training._dev_pass(initial, candidate)
    assert passed is False
    assert gates["metrics_defined"] is False


def test_v4_no_heldout_evaluation_or_promotion_in_training_source():
    source = inspect.getsource(training.run)
    formal_source = inspect.getsource(training._formal_stage)
    assert "heldout" in source
    assert "evaluate_heldout" not in source
    assert '"P3_QUALIFIED": False' in source
    assert '"P4_ALLOWED": False' in source
    assert '"checkpoint_promotable": False' in source
    assert "_evaluate_dev" in formal_source
    assert "completion_report_only" in formal_source


def test_v4_marker_is_reserved_before_entry_reset_and_output_is_preflighted():
    source = inspect.getsource(training.run)
    assert '"reserved_before_first_reset": True' in source
    assert source.index("_require(not output.exists()") < source.index("output.mkdir")
    assert source.index("reserve_once(marker") < source.index("_entry_stage")


def test_v4_source_hashes_are_relative_and_include_runner_protocol():
    hashes = training.source_hashes()
    assert hashes[training.PROTOCOL.as_posix()] == training.PROTOCOL_SHA
    assert hashes[training.RUNNER.as_posix()] == training._sha(training.RUNNER)
    assert all(not key.startswith(("/", "\\")) and ":" not in key for key in hashes)
    assert all(len(value) == 64 for value in hashes.values())


def test_v4_root_validation_is_fail_closed(tmp_path, monkeypatch):
    value = training.protocol()
    root = {
        "stage": training.v3.STAGE,
        "PASS": True,
        "HELDOUT_EVALUATION_ALLOWED": True,
        "heldout_untouched": True,
        "P3_QUALIFIED": False,
        "P4_ALLOWED": False,
        "checkpoint_promotable": False,
        "protocol_sha256": training.V3_PROTOCOL_SHA,
        "initial_checkpoint_sha256": training.INITIAL_CHECKPOINT_SHA,
        "formal": {"PASS": True, "final_checkpoint_sha256": training.V3_FINAL_CHECKPOINT_SHA},
    }
    path = tmp_path / "root.json"
    path.write_text(json.dumps(root), encoding="utf-8")
    monkeypatch.setattr(training, "_sha256", lambda _: training.V3_ROOT_MANIFEST_SHA)
    assert training.validate_superseded_v3_manifest(path, value)["PASS"] is True

    root["formal"]["PASS"] = False
    path.write_text(json.dumps(root), encoding="utf-8")
    with pytest.raises(ValueError):
        training.validate_superseded_v3_manifest(path, value)


def test_v4_formal_stops_after_first_passing_checkpoint(tmp_path, monkeypatch):
    value = training.protocol()
    rollout_seeds = []
    dev_labels = []

    class Config:
        dt = 1.0
        max_simulation_seconds = 1800.0
        total_requested_uavs = 24

    class State:
        def state_dict(self):
            return {}

    policy = SimpleNamespace(model=State(), optimizer=State())
    monkeypatch.setattr(
        training, "_fresh_policy", lambda checkpoint, p: (policy, {}, ("fake",)),
    )
    monkeypatch.setattr(training, "scenario_config", lambda name, p: Config())
    monkeypatch.setattr(
        training, "_scenario_horizon", lambda config, name, *, stage, index, p: (config, 2),
    )

    def fake_rollout(policy_arg, config, *, seed, max_steps, output, deterministic, collect_labels):
        rollout_seeds.append(seed)
        return [object()], {"seed": seed, "action_tape": str(tmp_path / f"tape_{seed}.json"), "completed": 24}

    monkeypatch.setattr(training, "rollout", fake_rollout)
    monkeypatch.setattr(
        training,
        "_rollout_gates",
        lambda *args, **kwargs: ({"hard": True}, {}, {}, 1.0),
    )
    monkeypatch.setattr(
        training,
        "_training_update",
        lambda *args, **kwargs: ({"update": 1}, {"update": True}),
    )
    monkeypatch.setattr(training, "state_equal", lambda left, right: True)

    def fake_save(path, *args, **kwargs):
        path.write_bytes(b"checkpoint")

    monkeypatch.setattr(training, "save_checkpoint", fake_save)
    monkeypatch.setattr(training, "_restored_policy", lambda *args, **kwargs: policy)
    rows = [
        {"base_priority": priority, "initial_soc": soc, "wait_seconds": 1.0}
        for priority, soc in ((1, 0.55), (3, 0.70), (5, 0.85))
    ]
    monkeypatch.setattr(
        training.pilot_v2,
        "lifecycle_metrics",
        lambda *args, **kwargs: {"replay_exact": True, "rows": rows},
    )

    def fake_dev(policy_arg, config, p, output, *, label):
        dev_labels.append(label)
        if label == "initial":
            aggregate = {
                "completion_fraction": 1.0,
                "priority_weighted_wait_seconds": 100.0,
                "low_soc_mean_wait_seconds": 50.0,
            }
        else:
            aggregate = {
                "completion_fraction": 1.0,
                "priority_weighted_wait_seconds": 99.0,
                "low_soc_mean_wait_seconds": 50.0,
            }
        return {
            "label": label, "records": [], "aggregate": aggregate,
            "complete_records": True, "PASS": True,
            "fatal_hard_failure": False, "performance_failure_only": False,
        }

    monkeypatch.setattr(training, "_evaluate_dev", fake_dev)
    result = training._formal_stage(
        output=tmp_path / "formal", checkpoint=tmp_path / "initial.pt", p=value,
    )

    assert result["PASS"] is True
    assert result["selected_checkpoint_index"] == 1
    assert result["selected_checkpoint_sha256"]
    assert rollout_seeds == [827101]
    assert dev_labels == ["initial", "checkpoint_1"]
    assert len(result["records"]) == 1
