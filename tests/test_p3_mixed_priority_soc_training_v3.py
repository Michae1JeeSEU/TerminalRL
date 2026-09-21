import inspect
import json
from dataclasses import replace

import pytest

from terminal_marl.training import (
    run_v4_2_p3_mixed_priority_soc_training_v3 as training,
)


def _failed_manifest(protocol=None) -> dict:
    if protocol is None:
        protocol = training.protocol()
    return {
        "stage": training.FAILED_HELDOUT_STAGE,
        "PASS": False,
        "P3_QUALIFIED": False,
        "checkpoint_promotable": False,
        "P4_ALLOWED": False,
        "HELDOUT_EVALUATION_ALLOWED": False,
        "protocol_sha256": training.FAILED_HELDOUT_PROTOCOL_SHA,
        "trained_checkpoint_sha256": training.V2_TRAINED_CHECKPOINT_SHA,
        "initial_checkpoint_sha256": training.INITIAL_CHECKPOINT_SHA,
        "heldout_untouched": True,
        "records": [
            {
                "result": {"gates": {"episode_integrity": True}},
                "episode_integrity": True,
            }
            for _ in range(18)
        ],
        "decision": protocol["bindings"]["superseded_heldout_failed_gates"],
    }


def test_v3_protocol_binds_failed_heldout_and_exact_reward_repair():
    value = training.protocol()

    assert value["protocol"] == training.STAGE
    assert value["status"] == "PREREGISTERED_TRAINING_REPAIR"
    assert value["bindings"]["superseded_heldout_stage"] == training.FAILED_HELDOUT_STAGE
    assert value["bindings"]["superseded_heldout_manifest_sha256"] == (
        training.FAILED_HELDOUT_MANIFEST_SHA
    )
    assert value["bindings"]["superseded_heldout_protocol_sha256"] == (
        training.FAILED_HELDOUT_PROTOCOL_SHA
    )
    assert value["bindings"]["superseded_heldout_failed_gates"]["all"] is False
    assert value["bindings"]["superseded_heldout_failed_gates"][
        "trained_dense_strict_improvement"
    ] is False
    assert value["reward"] == {
        "low_level_priority_time_weight": 1.0,
        "low_level_soc_urgency_time_weight": 2.0,
        "low_level_soc_urgency_threshold": 0.65,
        "urgency_reward_config_exact": True,
    }
    assert value["bindings"]["reward_sources"] == training.REWARD_SOURCE_HASHES
    assert value["authorization"]["P3_QUALIFIED"] is False
    assert value["authorization"]["P4_ALLOWED"] is False
    assert value["authorization"]["checkpoint_promotable"] is False


def test_v3_schedule_is_fresh_and_disjoint_from_all_prior_seed_sets():
    value = training.protocol()
    entry = value["entry"]["seeds"]
    formal = value["formal"]["seeds"]
    heldout = value["heldout"]["seeds"]
    scheduled = entry + formal + heldout

    assert entry == [823001, 823002]
    assert formal == list(range(824001, 824013))
    assert heldout == list(range(825001, 825007))
    assert len(scheduled) == len(set(scheduled))
    assert not set(scheduled) & set(value["history"]["forbidden_seeds"])
    assert not set(heldout) & set(range(822001, 822007))
    assert value["heldout"]["untouched"] is True
    assert value["reservation"]["reserve_pipeline_and_heldout_before_first_reset"] is True


def test_v3_full_horizons_and_reward_values_are_applied_to_all_scenarios():
    value = training.protocol()
    expected = {"flow6": 1210.0, "flow12": 1570.0, "dense24": 1080.0}
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


def test_v3_entry_and_formal_start_from_original_with_fresh_optimizer():
    value = training.protocol()
    entry = training.stage_initialization_contract("entry", value)
    formal = training.stage_initialization_contract("formal", value)

    assert entry["checkpoint_sha256"] == training.INITIAL_CHECKPOINT_SHA
    assert formal["checkpoint_sha256"] == training.INITIAL_CHECKPOINT_SHA
    assert entry["fresh_optimizer"] is True
    assert formal["fresh_optimizer"] is True
    assert entry["entry_updates_discarded"] is True
    assert formal["formal_initialization_is_original"] is True
    assert training.INITIAL_CHECKPOINT_SHA != training.V2_TRAINED_CHECKPOINT_SHA
    assert value["bindings"]["entry_checkpoints_formal_initialization"] == "forbidden"


def test_v3_failed_heldout_manifest_validation_is_exact_and_fail_closed(tmp_path, monkeypatch):
    value = training.protocol()
    path = tmp_path / "failed-heldout.json"
    path.write_text(json.dumps(_failed_manifest(value)), encoding="utf-8")
    monkeypatch.setattr(training, "_sha256", lambda _: training.FAILED_HELDOUT_MANIFEST_SHA)

    report = training.validate_failed_heldout_manifest(path, value)
    assert report["PASS"] is False
    assert len(report["records"]) == 18

    altered = _failed_manifest(value)
    altered["decision"] = dict(altered["decision"])
    altered["decision"]["trained_dense_strict_improvement"] = True
    path.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(ValueError):
        training.validate_failed_heldout_manifest(path, value)

    altered = _failed_manifest(value)
    altered["records"][0].pop("result")
    path.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(ValueError):
        training.validate_failed_heldout_manifest(path, value)


def test_v3_completion_and_wait_performance_are_report_only():
    value = training.protocol()
    assert value["completion_reporting"]["mode"] == "report_only"
    assert value["completion_reporting"]["training_rollout_completion_gate"] is False
    assert "completion_floor_by_scenario" not in value["hard_gates"]

    rollout_source = inspect.getsource(training._rollout_gates)
    stage_source = inspect.getsource(training._training_stage)
    assert "completion_floor" not in rollout_source
    assert "all_requested_complete" not in rollout_source
    assert '"completion_gate": False' in rollout_source
    assert "priority_weighted_wait" not in rollout_source
    assert '"completion_report_only": True' in stage_source
    assert "if not all(bool(value) for value in gates.values())" in stage_source


def test_v3_urgency_config_gate_is_exact():
    value = training.protocol()
    config = training.scenario_config("dense24", value)
    assert training._urgency_reward_config_exact(config, value)
    changed = replace(config, low_level_soc_urgency_threshold=0.66)
    assert not training._urgency_reward_config_exact(changed, value)


def test_v3_source_hashes_include_runner_protocol_and_reward_sources():
    hashes = training.source_hashes()

    assert hashes["terminal_marl/configs/low_v4_2/p3_mixed_priority_soc_training_v3.yaml"] == (
        training.PROTOCOL_SHA
    )
    for path in training.REWARD_SOURCE_HASHES:
        assert hashes[path] == training.REWARD_SOURCE_HASHES[path]
    assert hashes["terminal_marl/core/merge_scheduler.py"] == training.SCHEDULER_SOURCE_SHA
    assert all(not key.startswith(("/", "\\")) and ":" not in key for key in hashes)
    assert all(len(value) == 64 for value in hashes.values())


def test_v3_formal_is_sequential_and_root_flags_cannot_promote():
    value = training.protocol()
    assert value["formal"]["sequential_updates"] is True
    assert value["formal"]["full_scenario_horizon"] is True
    assert value["formal"]["stochastic"] is True
    source = inspect.getsource(training.run)
    assert '"PPO_called": True' in source
    assert '"P3_QUALIFIED": False' in source
    assert '"P4_ALLOWED": False' in source
    assert '"checkpoint_promotable": False' in source
    assert '"formal_pass_authorizes_heldout_only": True' in source
