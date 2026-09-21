import inspect
import json
from types import SimpleNamespace

import pytest

from terminal_marl.training import (
    diagnose_v4_2_p3_v3_checkpoint_safety_frontier_v1 as diagnostic,
)


def _training_config():
    return diagnostic.scenario_config(
        diagnostic.SCENARIO, diagnostic.training_protocol()
    )


def test_frontier_protocol_binds_failure_evidence_and_checkpoint_order():
    value = diagnostic.protocol()

    assert value["protocol"] == diagnostic.STAGE
    assert value["status"] == "PREREGISTERED_EXPOSED_SEED_DIAGNOSTIC"
    assert value["exposed_seed_diagnostic_only"] is True
    bindings = value["bindings"]
    assert bindings["heldout_protocol_sha256"] == diagnostic.HELDOUT_PROTOCOL_SHA
    assert bindings["heldout_consumption_marker_sha256"] == diagnostic.HELDOUT_MARKER_SHA
    assert bindings["failed_summary_sha256"] == diagnostic.FAILED_SUMMARY_SHA
    assert bindings["failed_action_tape_sha256"] == diagnostic.FAILED_TAPE_SHA
    assert bindings["failed_safety_component_message"] == diagnostic.FAILED_SAFETY_MESSAGE
    assert value["candidates"]["indices"] == list(range(1, 13))
    assert value["candidates"]["sha256"] == list(diagnostic.CHECKPOINT_SHA256S)
    assert value["scenario"]["seed"] == 825006
    assert value["scenario"]["name"] == "dense24"


def test_frontier_protocol_rejects_protocol_sha_mismatch(monkeypatch):
    monkeypatch.setattr(diagnostic, "PROTOCOL_SHA", "0" * 64)
    with pytest.raises(ValueError):
        diagnostic.protocol()


def test_failed_evidence_validator_is_exact_and_fail_closed(tmp_path, monkeypatch):
    value = diagnostic.protocol()
    marker_path = tmp_path / diagnostic.HELDOUT_MARKER_NAME
    summary_path = tmp_path / "summary.json"
    tape_path = tmp_path / "tape.json"
    marker = {
        "stage": "V4_2_P3_MIXED_PRIORITY_SOC_HELDOUT_V2",
        "protocol_sha256": diagnostic.HELDOUT_PROTOCOL_SHA,
        "heldout_seeds": [825001, 825002, 825003, 825004, 825005, 825006],
        "scenarios": ["flow6", "flow12", "dense24", "flow6", "flow12", "dense24"],
        "controllers": ["trained", "initial", "rule"],
        "heldout_untouched": True,
        "reserved_before_first_reset": True,
        "one_time": True,
        "urgency_reward_config": dict(diagnostic.REWARD_CONFIG),
    }
    summary = {
        "seed": 825006,
        "safety": {"SafetyInvariantError": 1},
        "safety_error": diagnostic.FAILED_SAFETY_MESSAGE,
    }
    tape = {
        "episode_seed": 825006,
        "action_tape": [{
            "safety_event": {
                "kind": "safety_invariant_error",
                "message": diagnostic.FAILED_SAFETY_MESSAGE,
            }
        }],
    }
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    tape_path.write_text(json.dumps(tape), encoding="utf-8")

    def fake_sha(path):
        name = str(path)
        if name.endswith(diagnostic.HELDOUT_MARKER_NAME):
            return diagnostic.HELDOUT_MARKER_SHA
        if name.endswith("summary.json"):
            return diagnostic.FAILED_SUMMARY_SHA
        return diagnostic.FAILED_TAPE_SHA

    monkeypatch.setattr(diagnostic, "_sha", fake_sha)
    evidence = diagnostic.validate_failed_evidence(
        marker=marker_path, summary=summary_path, action_tape=tape_path, p=value,
    )
    assert evidence["summary"]["seed"] == 825006

    summary["safety_error"] = "changed"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError):
        diagnostic.validate_failed_evidence(
            marker=marker_path, summary=summary_path, action_tape=tape_path, p=value,
        )


def test_candidate_path_resolution_checks_all_hashes_and_order(tmp_path, monkeypatch):
    value = diagnostic.protocol()
    initial = tmp_path / "initial.pt"
    initial.write_bytes(b"initial")
    checkpoint_dir = tmp_path / "formal"
    checkpoint_dir.mkdir()
    for index in diagnostic.CHECKPOINT_INDICES:
        (checkpoint_dir / f"checkpoint_{index}.pt").write_bytes(str(index).encode())

    def fake_sha(path):
        path = str(path)
        if path.endswith("initial.pt"):
            return diagnostic.INITIAL_CHECKPOINT_SHA
        for index, expected in zip(diagnostic.CHECKPOINT_INDICES, diagnostic.CHECKPOINT_SHA256S):
            if path.endswith(f"checkpoint_{index}.pt"):
                return expected
        raise AssertionError(path)

    monkeypatch.setattr(diagnostic, "_sha", fake_sha)
    candidates = diagnostic.resolve_candidate_paths(checkpoint_dir, initial, value)
    assert [item["index"] for item in candidates] == list(range(13))
    assert candidates[0]["label"] == "original_k3"
    assert candidates[-1]["sha256"] == diagnostic.FINAL_CHECKPOINT_SHA


def test_trained_loader_lineage_is_fail_closed():
    payload = {
        "stage": "V4_2_P3_MIXED_PRIORITY_SOC_TRAINING_V3",
        "formal": True,
        "ppo_called": True,
        "protocol_sha256": diagnostic.TRAINING_PROTOCOL_SHA,
        "initial_checkpoint_sha256": diagnostic.INITIAL_CHECKPOINT_SHA,
        "checkpoint_promotable": False,
        "urgency_reward_config": dict(diagnostic.REWARD_CONFIG),
    }
    assert diagnostic.validate_trained_checkpoint_payload(payload)["formal"] is True
    payload["urgency_reward_config"]["low_level_soc_urgency_time_weight"] = 3.0
    with pytest.raises(ValueError):
        diagnostic.validate_trained_checkpoint_payload(payload)


def test_classifier_records_safety_error_shield_shortlist_and_completion():
    config = _training_config()
    failed = diagnostic.classify_summary({
        "safety": {
            "collisions": 0,
            "nmac": 0,
            "hard_separation_violations": 0,
            "uncertified_execution": 0,
            "SafetyInvariantError": 1,
        },
        "safety_error": diagnostic.FAILED_SAFETY_MESSAGE,
        "completed": 21,
        "seconds": 684.0,
        "steps": 684,
        "counts": {"actor_decisions": 100, "shield_actor": 5},
        "runtime_shortlist": {
            "runtime_shortlist_exhausted": 118,
            "runtime_shortlist_fallback_selected": 6,
        },
        "action_tape": "tape.json",
    }, config)
    assert failed["safety_error_present"] is True
    assert failed["safety_all_zero"] is False
    assert failed["completed"] == 21
    assert failed["seconds"] == 684.0
    assert failed["shield_fraction"] == 0.05
    assert failed["shortlist_exhausted"] == 118
    assert failed["shortlist_fallback_selected"] == 6
    assert failed["completion_fraction"] == 21 / 24

    safe = diagnostic.classify_result({
        "safety": {"collisions": 0, "SafetyInvariantError": 0},
        "safety_error": None,
        "completed": 24,
        "seconds": 1080.0,
        "counts": {"actor_decisions": 10, "shield_actor": 0},
        "runtime_shortlist": {},
    }, config)
    assert safe["safety_all_zero"] is True
    assert safe["safety_error_present"] is False


def test_frontier_summary_keeps_running_after_first_failure():
    records = [
        {"index": 0, "safety_all_zero": True, "safety_error_present": False},
        {"index": 1, "safety_all_zero": False, "safety_error_present": True},
        {"index": 2, "safety_all_zero": True, "safety_error_present": False},
        {"index": 3, "safety_all_zero": False, "safety_error_present": True},
    ]
    result = diagnostic.frontier_summary(records)
    assert result == {
        "earliest_failed_checkpoint": 1,
        "safe_checkpoint_indices": [0, 2],
        "failure_checkpoint_indices": [1, 3],
        "last_safe_before_first_failure": 0,
    }
    source = inspect.getsource(diagnostic.run)
    assert "for item in loaded" in source
    assert "except SafetyInvariantError" in source
    assert "break" not in source.split("for item in loaded", 1)[1].split(
        "failure_indices", 1
    )[0]


def test_frontier_is_diagnostic_only_and_uses_direct_rollout():
    value = diagnostic.protocol()
    assert value["authorization"]["PASS_used_for_promotion"] is False
    source = inspect.getsource(diagnostic.run)
    assert "student_rollout_v4_2.rollout(" in source
    assert "evaluate_v4_2_p2_v4" not in source
    assert '"checkpoint_promotable": False' in source
    assert '"P3_QUALIFIED": False' in source
    assert '"P4_ALLOWED": False' in source
    assert '"diagnostic_integrity": bool(diagnostic_integrity)' in source
    assert '"PASS": bool(diagnostic_integrity)' in source
