import copy
import inspect
import json
from pathlib import Path

import pytest

from terminal_marl.training import (
    evaluate_v4_2_p3_mixed_priority_soc_heldout_v1 as heldout,
)


def _row(wait: float, priority: int = 1, soc: float = 0.60) -> dict:
    return {
        "uid": 0,
        "base_priority": priority,
        "initial_soc": soc,
        "wait_seconds": wait,
        "request_to_touchdown_seconds": wait + 10.0,
    }


def _records() -> list[dict]:
    records = []
    for seed, scenario in zip(heldout.SEEDS, heldout.SCENARIOS):
        for controller in heldout.CONTROLLERS:
            if scenario == "dense24":
                requests = 24
                completed = 20
            else:
                requests = 6 if scenario == "flow6" else 12
                completed = requests
            baseline = 2.0 if controller == "initial" else 1.0
            if controller == "rule":
                baseline = 3.0
            rows = [
                _row(baseline, priority=1, soc=0.60),
                _row(baseline, priority=3, soc=0.72),
                _row(baseline, priority=5, soc=0.90),
            ]
            records.append({
                "seed": seed,
                "scenario": scenario,
                "controller": controller,
                "episode_integrity": True,
                "result": {
                    "requests": requests,
                    "completed": completed,
                },
                "lifecycle": {
                    "replay_exact": True,
                    "rows": rows,
                },
            })
    return records


def test_protocol_binds_audit_checkpoints_and_fresh_order():
    value = heldout.protocol()
    assert value["protocol"] == heldout.STAGE
    assert value["bindings"]["audit_manifest_sha256"] == heldout.AUDIT_MANIFEST_SHA
    assert value["bindings"]["audit_protocol_sha256"] == heldout.AUDIT_PROTOCOL_SHA
    assert value["bindings"]["trained_checkpoint_sha256"] == heldout.TRAINED_CHECKPOINT_SHA
    assert value["bindings"]["initial_checkpoint_sha256"] == heldout.INITIAL_CHECKPOINT_SHA
    assert value["heldout"]["seeds"] == list(heldout.SEEDS)
    assert value["heldout"]["scenarios"] == list(heldout.SCENARIOS)
    assert value["heldout"]["controllers"] == list(heldout.CONTROLLERS)
    assert value["heldout"]["expected_records"] == 18
    assert value["heldout"]["dense_horizon_seconds"] == 1080


def test_scenario_config_uses_full_frozen_horizons():
    value = heldout.training_protocol()
    assert heldout.scenario_config("flow6", value).max_simulation_seconds == 1210.0
    assert heldout.scenario_config("flow12", value).max_simulation_seconds == 1570.0
    dense = heldout.scenario_config("dense24", value)
    assert dense.max_simulation_seconds == 1080.0
    assert dense.reset_mode == "warm_start_rings"
    assert dense.warm_start_uavs == 12


def test_audit_manifest_validation_is_fail_closed(tmp_path, monkeypatch):
    value = heldout.protocol()
    path = tmp_path / "audit.json"
    report = {
        "stage": heldout.AUDIT_STAGE,
        "PASS": True,
        "HELDOUT_EVALUATION_ALLOWED": True,
        "heldout_untouched": True,
        "P4_ALLOWED": False,
        "checkpoint_promotable": False,
        "protocol_sha256": heldout.AUDIT_PROTOCOL_SHA,
        "training_protocol_sha256": heldout.TRAINING_PROTOCOL_SHA,
        "scheduler_source": heldout.SCHEDULER_SOURCE.as_posix(),
        "scheduler_source_sha256": heldout.SCHEDULER_SOURCE_SHA,
        "corrected_coverage": {
            "priority_coverage": True,
            "soc_in_bounds": True,
            "soc_range_observed": True,
        },
        "final_checkpoint": {
            "path": "/server/checkpoint_8.pt",
            "sha256": heldout.TRAINED_CHECKPOINT_SHA,
        },
    }
    path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(heldout, "_sha", lambda _: heldout.AUDIT_MANIFEST_SHA)
    assert heldout.validate_audit_manifest(path, value)["PASS"] is True
    report["checkpoint_promotable"] = True
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(heldout.HeldoutInputError):
        heldout.validate_audit_manifest(path, value)


def test_training_reservation_marker_validation_is_fail_closed(tmp_path, monkeypatch):
    value = heldout.protocol()
    path = tmp_path / heldout.TRAINING_RESERVATION_MARKER
    marker = {
        "stage": heldout.TRAINING_STAGE,
        "protocol_sha256": heldout.TRAINING_PROTOCOL_SHA,
        "pilot_manifest_sha256": heldout.PILOT_MANIFEST_SHA,
        "superseded_training_manifest_sha256": heldout.SUPERSEDED_TRAINING_MANIFEST_SHA,
        "candidate_checkpoint_sha256": heldout.INITIAL_CHECKPOINT_SHA,
        "entry_seeds": heldout.TRAINING_ENTRY_SEEDS,
        "formal_seeds": heldout.TRAINING_FORMAL_SEEDS,
        "heldout_seeds": heldout.TRAINING_HELDOUT_SEEDS,
        "heldout_untouched": True,
        "reserved_before_first_reset": True,
    }
    path.write_text(json.dumps(marker), encoding="utf-8")
    monkeypatch.setattr(
        heldout,
        "_sha",
        lambda source: heldout.TRAINING_RESERVATION_MARKER_SHA
        if Path(source) == path else heldout._sha256(source),
    )
    assert heldout.validate_training_reservation_marker(path, value)["heldout_untouched"]
    marker["heldout_untouched"] = False
    path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(heldout.HeldoutInputError):
        heldout.validate_training_reservation_marker(path, value)


def test_trained_checkpoint_payload_validation_is_fail_closed():
    value = heldout.protocol()
    payload = {
        "stage": heldout.TRAINING_STAGE,
        "formal": True,
        "ppo_called": True,
        "protocol_sha256": heldout.TRAINING_PROTOCOL_SHA,
        "initial_checkpoint_sha256": heldout.INITIAL_CHECKPOINT_SHA,
        "checkpoint_promotable": False,
    }
    assert heldout.validate_trained_checkpoint_payload(payload, value)["formal"] is True
    payload["ppo_called"] = False
    with pytest.raises(heldout.HeldoutInputError):
        heldout.validate_trained_checkpoint_payload(payload, value)


def test_aggregation_and_decision_require_strict_dense_improvement():
    records = _records()
    aggregates = heldout.aggregate_groups(records)
    assert aggregates["sparse"]["trained"]["completion_fraction"] == 1.0
    assert aggregates["dense"]["trained"]["completion_fraction"] == aggregates["dense"]["initial"]["completion_fraction"]
    assert aggregates["dense"]["trained"]["priority_weighted_wait_seconds"] < aggregates["dense"]["initial"]["priority_weighted_wait_seconds"]
    decision = heldout.decision_gates(
        records, aggregates, frozen_proposal_subgraph=True,
    )
    assert decision["trained_dense_strict_improvement"]
    assert decision["all"]


def test_none_metrics_fail_closed_and_do_not_pass_decision():
    records = _records()
    aggregates = heldout.aggregate_groups(records)
    broken = copy.deepcopy(aggregates)
    broken["dense"]["initial"]["priority_weighted_wait_seconds"] = None
    decision = heldout.decision_gates(
        records, broken, frozen_proposal_subgraph=True,
    )
    assert decision["none_metrics_fail_closed"] is False
    assert decision["all"] is False


def test_completion_is_not_an_episode_integrity_gate():
    source = inspect.getsource(heldout._episode_gates)
    assert "all_requests_complete" not in source
    assert "completion_fraction" not in source
    assert "lifecycle_replay_exact" in source


def test_source_hashes_are_repository_relative():
    hashes = heldout.source_hashes()
    assert "terminal_marl/training/evaluate_v4_2_p3_mixed_priority_soc_heldout_v1.py" in hashes
    assert "terminal_marl/training/run_v4_2_p3_mixed_priority_soc_training_v2.py" in hashes
    assert all(not key.startswith(("/", "\\")) and ":" not in key for key in hashes)
    assert all(len(value) == 64 for value in hashes.values())
