import inspect
import json

import pytest

from terminal_marl.training import (
    evaluate_v4_2_p3_v3_early_stop_q3_v1 as q3,
)


def _valid_root() -> dict:
    return {
        "stage": q3.TRAINING_STAGE,
        "PASS": True,
        "HELDOUT_EVALUATION_ALLOWED": True,
        "heldout_untouched": True,
        "P3_QUALIFIED": False,
        "P4_ALLOWED": False,
        "checkpoint_promotable": False,
        "formal_pass_authorizes_heldout_only": True,
        "protocol_sha256": q3.TRAINING_PROTOCOL_SHA,
        "initial_checkpoint_sha256": q3.INITIAL_CHECKPOINT_SHA,
        "urgency_reward_config": dict(q3.REWARD_CONFIG),
        "entry": {"PASS": True},
        "formal": {
            "PASS": True,
            "final_checkpoint": "/tmp/checkpoint_12.pt",
            "final_checkpoint_sha256": q3.TRAINING_FINAL_CHECKPOINT_SHA,
        },
    }


def _frontier_record(index: int, sha: str) -> dict:
    selected = index == 5
    return {
        "index": index,
        "label": "checkpoint_5" if selected else ("original_k3" if index == 0 else f"checkpoint_{index}"),
        "seed": 825006,
        "scenario": "dense24",
        "checkpoint_sha256": sha,
        "safety_all_zero": True,
        "completion_fraction": 1.0,
        "completion_report_only": True,
        "exact_replay": True,
        "frozen_proposal_subgraph": True,
        "deterministic": True,
        "no_teacher": True,
        "no_labels": True,
        "no_updates": True,
        "urgency_reward_config_exact": True,
        "safety": {"SafetyInvariantError": 0, "collisions": 0},
    }


def _valid_frontier() -> dict:
    hashes = [f"{index:064x}" for index in range(1, 13)]
    hashes[4] = q3.TRAINED_CHECKPOINT_SHA
    records = [_frontier_record(0, q3.INITIAL_CHECKPOINT_SHA)]
    records.extend(_frontier_record(index, hashes[index - 1]) for index in range(1, 13))
    return {
        "stage": "V4_2_P3_V3_CHECKPOINT_SAFETY_FRONTIER_V1",
        "PASS": True,
        "diagnostic_only": True,
        "diagnostic_integrity": True,
        "exposed_seed_diagnostic_only": True,
        "all_candidates_completed": True,
        "candidate_count": 13,
        "expected_candidate_count": 13,
        "earliest_failed_checkpoint": 12,
        "last_safe_before_first_failure": 11,
        "failure_checkpoint_indices": [12],
        "safe_checkpoint_indices": list(range(12)),
        "seed": 825006,
        "scenario": "dense24",
        "horizon_seconds": 1080.0,
        "initial_checkpoint_sha256": q3.INITIAL_CHECKPOINT_SHA,
        "no_seed_consumption": True,
        "no_seed_reservation": True,
        "P3_QUALIFIED": False,
        "P4_ALLOWED": False,
        "checkpoint_promotable": False,
        "protocol_sha256": q3.FRONTIER_PROTOCOL_SHA,
        "failed_evidence": {
            "heldout_consumption_marker_sha256": q3.EXPOSED_CONSUMPTION_MARKER_SHA,
        },
        "checkpoint_sha256s": hashes,
        "records": records,
    }


def _records():
    return [
        {
            "seed": seed,
            "scenario": scenario,
            "controller": controller,
            "episode_integrity": True,
        }
        for seed, scenario in zip(q3.SEEDS, q3.SCENARIOS)
        for controller in q3.CONTROLLERS
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


def test_q3_protocol_is_frozen_and_fresh_schedule_is_ordered():
    value = q3.protocol()
    assert q3.PROTOCOL_SHA != "TO_BE_FROZEN"
    assert value["protocol"] == q3.STAGE
    assert value["status"] == "PREREGISTERED_FRESH_Q3_CONFIRMATION"
    assert value["q3"]["seeds"] == list(range(826001, 826007))
    assert value["q3"]["scenarios"] == [
        "flow6", "flow12", "dense24", "flow6", "flow12", "dense24",
    ]
    assert value["q3"]["expected_records"] == 18
    assert value["bindings"]["trained_checkpoint_sha256"] == q3.TRAINED_CHECKPOINT_SHA
    assert value["provenance"]["exposed_seed_diagnostic_only"] is True
    assert value["provenance"]["posthoc_selected_checkpoint"] == "checkpoint_5"


def test_frontier_selection_and_evidence_validation_fail_closed(tmp_path, monkeypatch):
    path = tmp_path / "frontier.json"
    path.write_text(json.dumps(_valid_frontier()), encoding="utf-8")
    value = q3.protocol()
    monkeypatch.setattr(q3, "_sha", lambda _: q3.FRONTIER_MANIFEST_SHA)
    report = q3.validate_frontier_manifest(path, value)
    assert report["records"][5]["checkpoint_sha256"] == q3.TRAINED_CHECKPOINT_SHA

    altered = _valid_frontier()
    altered["earliest_failed_checkpoint"] = 11
    path.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(ValueError):
        q3.validate_frontier_manifest(path, q3.protocol())


def test_training_root_and_selected_payload_are_exact_and_fail_closed(tmp_path, monkeypatch):
    value = q3.protocol()
    root_path = tmp_path / "manifest.json"
    root_path.write_text(json.dumps(_valid_root()), encoding="utf-8")
    monkeypatch.setattr(q3, "_sha", lambda _: q3.TRAINING_ROOT_MANIFEST_SHA)
    assert q3.validate_training_root_manifest(root_path, value)["PASS"] is True

    altered = _valid_root()
    altered["formal"]["final_checkpoint_sha256"] = q3.TRAINED_CHECKPOINT_SHA
    root_path.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(ValueError):
        q3.validate_training_root_manifest(root_path, value)

    payload = {
        "stage": q3.TRAINING_STAGE,
        "formal": True,
        "ppo_called": True,
        "protocol_sha256": q3.TRAINING_PROTOCOL_SHA,
        "initial_checkpoint_sha256": q3.INITIAL_CHECKPOINT_SHA,
        "checkpoint_promotable": False,
        "urgency_reward_config": dict(q3.REWARD_CONFIG),
    }
    assert q3.validate_trained_checkpoint_payload(payload, value)["formal"] is True
    payload["checkpoint_promotable"] = True
    with pytest.raises(ValueError):
        q3.validate_trained_checkpoint_payload(payload, value)


def test_scenario_horizons_reward_and_fresh_seed_overlap_rejection():
    training_p = q3.training_protocol()
    expected = {"flow6": 1210.0, "flow12": 1570.0, "dense24": 1080.0}
    for name, horizon in expected.items():
        config = q3.scenario_config(name, training_p)
        assert config.max_simulation_seconds == horizon
        assert config.low_level_priority_time_weight == 1.0
        assert config.low_level_soc_urgency_time_weight == 2.0
        assert config.low_level_soc_urgency_threshold == 0.65
    assert not set(q3.SEEDS) & set(q3.FORBIDDEN_SEEDS)
    with pytest.raises(ValueError):
        q3.validate_fresh_seed_schedule({"seeds": {826001}}, q3.protocol())
    with pytest.raises(ValueError):
        q3.validate_fresh_seed_schedule({"seeds": {826002}}, q3.protocol())


def test_q3_pairing_strict_dense_gate_and_none_fail_closed():
    records = _records()
    gates = q3.decision_gates(records, _aggregate_template(), frozen_proposal_subgraph=True)
    assert gates["all_records"] is True
    assert gates["trained_dense_strict_improvement"] is True
    assert gates["all"] is True

    equal = _aggregate_template()
    equal["dense"]["trained"]["priority_weighted_wait_seconds"] = 9.0
    equal_gates = q3.decision_gates(records, equal, frozen_proposal_subgraph=True)
    assert equal_gates["trained_dense_strict_improvement"] is False
    assert equal_gates["all"] is False

    none_value = _aggregate_template()
    none_value["overall"]["trained"]["low_soc_mean_wait_seconds"] = None
    none_gates = q3.decision_gates(records, none_value, frozen_proposal_subgraph=True)
    assert none_gates["none_metrics_fail_closed"] is False
    assert none_gates["all"] is False


def test_q3_episode_completion_is_report_only_and_source_hashes_cover_lineage():
    value = q3.protocol()
    assert value["q3"]["completion_report_only"] is True
    assert "completion_floor" not in value["episode_gates"]
    source = inspect.getsource(q3.evaluate_episode)
    assert "completion_report_only" in source
    hashes = q3.source_hashes()
    required = {
        "terminal_marl/training/evaluate_v4_2_p3_mixed_priority_soc_heldout_v2.py",
        "terminal_marl/training/evaluate_v4_2_p3_mixed_priority_soc_heldout_v1.py",
        "terminal_marl/training/run_v4_2_p3_mixed_priority_soc_training_v3.py",
        "terminal_marl/configs/low_v4_2/p3_mixed_priority_soc_training_v3.yaml",
        "terminal_marl/training/diagnose_v4_2_p3_v3_checkpoint_safety_frontier_v1.py",
        "terminal_marl/core/rewards.py",
        "terminal_marl/core/config.py",
        "terminal_marl/envs/parallel_env.py",
    }
    assert required <= set(hashes)
    assert all(len(item) == 64 for item in hashes.values())


def test_q3_reservation_precedes_first_reset_and_failure_cannot_promote():
    source = inspect.getsource(q3.run)
    assert source.index("reserve_once") < source.index("evaluate_episode")
    assert '"P3_QUALIFIED": passed' in source
    assert '"checkpoint_promotable": passed' in source
    assert '"P4_ALLOWED": False' in source


def test_q3_marker_payload_is_fresh_and_posthoc_provenance(tmp_path):
    payload = q3.reservation_payload(tmp_path / "result")
    assert payload["q3_seeds"] == list(q3.SEEDS)
    assert payload["fresh_q3"] is True
    assert payload["exposed_seed_diagnostic_only"] is True
    assert payload["reserved_before_first_reset"] is True
    assert payload["trained_checkpoint_sha256"] == q3.TRAINED_CHECKPOINT_SHA
