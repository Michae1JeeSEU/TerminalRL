import inspect
import json

import pytest

from terminal_marl.training import (
    evaluate_v4_2_p3_mixed_priority_soc_heldout_v2 as heldout,
)


def _valid_root() -> dict:
    return {
        "stage": heldout.TRAINING_STAGE,
        "PASS": True,
        "HELDOUT_EVALUATION_ALLOWED": True,
        "heldout_untouched": True,
        "P3_QUALIFIED": False,
        "P4_ALLOWED": False,
        "checkpoint_promotable": False,
        "formal_pass_authorizes_heldout_only": True,
        "protocol_sha256": heldout.TRAINING_PROTOCOL_SHA,
        "initial_checkpoint_sha256": heldout.INITIAL_CHECKPOINT_SHA,
        "urgency_reward_config": dict(heldout.REWARD_CONFIG),
        "entry": {"PASS": True},
        "formal": {
            "PASS": True,
            "final_checkpoint": "/tmp/checkpoint_12.pt",
            "final_checkpoint_sha256": heldout.TRAINED_CHECKPOINT_SHA,
        },
    }


def _valid_marker() -> dict:
    return {
        "stage": heldout.TRAINING_STAGE,
        "protocol_sha256": heldout.TRAINING_PROTOCOL_SHA,
        "initial_checkpoint_sha256": heldout.INITIAL_CHECKPOINT_SHA,
        "entry_seeds": heldout.TRAINING_ENTRY_SEEDS,
        "formal_seeds": heldout.TRAINING_FORMAL_SEEDS,
        "heldout_seeds": heldout.TRAINING_HELDOUT_SEEDS,
        "heldout_untouched": True,
        "reserved_before_first_reset": True,
        "urgency_reward_config": dict(heldout.REWARD_CONFIG),
    }


def _records():
    return [
        {
            "seed": seed,
            "scenario": scenario,
            "controller": controller,
            "episode_integrity": True,
        }
        for seed, scenario in zip(heldout.SEEDS, heldout.SCENARIOS)
        for controller in heldout.CONTROLLERS
    ]


def _aggregate_template():
    def row(completion, wait, low_soc):
        return {
            "completion_fraction": completion,
            "priority_weighted_wait_seconds": wait,
            "low_soc_mean_wait_seconds": low_soc,
        }

    return {
        "sparse": {
            "trained": row(1.0, 10.0, 10.0),
            "initial": row(1.0, 11.0, 11.0),
            "rule": row(1.0, 12.0, 12.0),
        },
        "dense": {
            "trained": row(0.8, 8.0, 9.0),
            "initial": row(0.8, 9.0, 10.0),
            "rule": row(0.7, 12.0, 13.0),
        },
        "overall": {
            "trained": row(0.9, 9.0, 9.0),
            "initial": row(0.9, 10.0, 10.0),
            "rule": row(0.8, 12.0, 12.0),
        },
    }


def test_v2_protocol_binds_v3_root_lineage_reward_and_pairing():
    value = heldout.protocol()

    assert value["protocol"] == heldout.STAGE
    assert value["status"] == "PREREGISTERED_HELDOUT"
    assert value["bindings"]["training_root_stage"] == heldout.TRAINING_STAGE
    assert value["bindings"]["training_root_manifest_sha256"] == (
        heldout.TRAINING_ROOT_MANIFEST_SHA
    )
    assert value["bindings"]["training_protocol_sha256"] == heldout.TRAINING_PROTOCOL_SHA
    assert value["bindings"]["trained_checkpoint_sha256"] == heldout.TRAINED_CHECKPOINT_SHA
    assert value["bindings"]["initial_checkpoint_sha256"] == heldout.INITIAL_CHECKPOINT_SHA
    assert value["bindings"]["reward_sources"] == heldout.REWARD_SOURCE_HASHES
    assert value["reward"] == {
        "low_level_priority_time_weight": 1.0,
        "low_level_soc_urgency_time_weight": 2.0,
        "low_level_soc_urgency_threshold": 0.65,
        "urgency_reward_config_exact": True,
    }
    assert value["heldout"]["seeds"] == list(range(825001, 825007))
    assert value["heldout"]["scenarios"] == [
        "flow6", "flow12", "dense24", "flow6", "flow12", "dense24",
    ]
    assert value["heldout"]["controllers"] == ["trained", "initial", "rule"]
    assert value["heldout"]["expected_records"] == 18


def test_v2_scenario_config_uses_v3_full_horizons_and_urgency():
    training_p = heldout.training_protocol()
    expected = {"flow6": 1210.0, "flow12": 1570.0, "dense24": 1080.0}
    for name, horizon in expected.items():
        config = heldout.scenario_config(name, training_p)
        assert config.max_simulation_seconds == horizon
        assert config.low_level_priority_time_weight == 1.0
        assert config.low_level_soc_urgency_time_weight == 2.0
        assert config.low_level_soc_urgency_threshold == 0.65
        assert heldout._scenario_horizon_exact(config, name)


def test_v2_root_validation_is_exact_and_fail_closed(tmp_path, monkeypatch):
    value = heldout.protocol()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_valid_root()), encoding="utf-8")
    monkeypatch.setattr(heldout, "_sha256", lambda _: heldout.TRAINING_ROOT_MANIFEST_SHA)
    assert heldout.validate_training_root_manifest(path, value)["PASS"] is True

    altered = _valid_root()
    altered["formal"]["PASS"] = False
    path.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(ValueError):
        heldout.validate_training_root_manifest(path, value)

    altered = _valid_root()
    altered["urgency_reward_config"] = dict(heldout.REWARD_CONFIG)
    altered["urgency_reward_config"]["low_level_soc_urgency_weight"] = 3.0
    path.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(ValueError):
        heldout.validate_training_root_manifest(path, value)


def test_v2_training_marker_validation_checks_new_seed_schedule_and_reward(tmp_path, monkeypatch):
    value = heldout.protocol()
    path = tmp_path / heldout.TRAINING_RESERVATION_MARKER
    path.write_text(json.dumps(_valid_marker()), encoding="utf-8")
    monkeypatch.setattr(heldout, "_sha256", lambda _: heldout.TRAINING_RESERVATION_MARKER_SHA)
    assert heldout.validate_training_reservation_marker(path, value)["heldout_untouched"] is True

    altered = _valid_marker()
    altered["formal_seeds"] = [824001]
    path.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(ValueError):
        heldout.validate_training_reservation_marker(path, value)


def test_v2_trained_checkpoint_payload_requires_v3_lineage_and_urgency():
    value = heldout.protocol()
    payload = {
        "stage": heldout.TRAINING_STAGE,
        "formal": True,
        "ppo_called": True,
        "protocol_sha256": heldout.TRAINING_PROTOCOL_SHA,
        "initial_checkpoint_sha256": heldout.INITIAL_CHECKPOINT_SHA,
        "checkpoint_promotable": False,
        "urgency_reward_config": dict(heldout.REWARD_CONFIG),
    }
    assert heldout.validate_trained_checkpoint_payload(payload, value)["formal"] is True
    payload["protocol_sha256"] = "wrong"
    with pytest.raises(ValueError):
        heldout.validate_trained_checkpoint_payload(payload, value)


def test_v2_decision_pairing_and_strict_dense_improvement():
    records = _records()
    aggregates = _aggregate_template()
    gates = heldout.decision_gates(records, aggregates, frozen_proposal_subgraph=True)
    assert gates["all_records"] is True
    assert gates["all_episode_integrity"] is True
    assert gates["trained_dense_strict_improvement"] is True
    assert gates["all"] is True

    equal = _aggregate_template()
    equal["dense"]["trained"]["priority_weighted_wait_seconds"] = 9.0
    equal_gates = heldout.decision_gates(records, equal, frozen_proposal_subgraph=True)
    assert equal_gates["trained_dense_strict_improvement"] is False
    assert equal_gates["all"] is False


def test_v2_decision_none_metrics_fail_closed_and_pairing_is_order_sensitive():
    records = _records()
    aggregates = _aggregate_template()
    aggregates["overall"]["trained"]["low_soc_mean_wait_seconds"] = None
    gates = heldout.decision_gates(records, aggregates, frozen_proposal_subgraph=True)
    assert gates["none_metrics_fail_closed"] is False
    assert gates["all"] is False

    swapped = list(records)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    gates = heldout.decision_gates(swapped, _aggregate_template(), frozen_proposal_subgraph=True)
    assert gates["all_records"] is False


def test_v2_episode_completion_is_report_only_and_source_hashes_cover_lineage():
    value = heldout.protocol()
    assert value["episode_gates"]["completion_report_only"] is True
    assert "completion_floor" not in value["episode_gates"]
    source = inspect.getsource(heldout.evaluate_episode)
    assert "completion_report_only" in source
    assert "urgency_reward_config_exact" in source

    hashes = heldout.source_hashes()
    required = {
        "terminal_marl/training/run_v4_2_p3_mixed_priority_soc_training_v2.py",
        "terminal_marl/configs/low_v4_2/p3_mixed_priority_soc_training_v2.yaml",
        "terminal_marl/training/evaluate_v4_2_p3_mixed_priority_soc_heldout_v1.py",
        "terminal_marl/training/run_v4_2_p3_mixed_priority_soc_training_v3.py",
        "terminal_marl/configs/low_v4_2/p3_mixed_priority_soc_training_v3.yaml",
        "terminal_marl/core/rewards.py",
        "terminal_marl/core/config.py",
        "terminal_marl/envs/parallel_env.py",
    }
    assert required <= set(hashes)
    assert all(len(item) == 64 for item in hashes.values())


def test_v2_output_authorization_flags_are_pass_only_and_p4_false():
    value = heldout.protocol()
    authorization = value["authorization"]
    assert authorization["heldout_evaluation_allowed_on_pass"] is True
    assert authorization["P3_QUALIFIED_on_pass"] is True
    assert authorization["checkpoint_promotable_on_pass"] is True
    assert authorization["P4_ALLOWED"] is False
    source = inspect.getsource(heldout.run)
    assert '"P3_QUALIFIED": passed' in source
    assert '"checkpoint_promotable": passed' in source
    assert '"P4_ALLOWED": False' in source
