"""Contract tests for the balanced 24-pair proposal-support wrapper."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from terminal_marl.training import run_p3_proposal_support_balanced_20261001 as runner


def _summary(pair_id: int, *, complete: bool = True, env_seed: int | None = None,
             keyed_seed: int | None = None):
    spec = runner.PAIR_SPECS[pair_id - 1]
    control = {
        "status": "completed" if complete else "censored",
        "normal": 5 if complete else None, "emergency": 0, "failed": 0,
        "diverted": 0, "unresolved": 0, "generated": 24, "admitted": 24,
        "completed": 5 if complete else None, "resolved": 5 if complete else None,
        "wait_seconds": 1.0 if complete else 1.0,
        "request_elapsed_total_seconds": 2.0 if complete else None,
        "energy_total_consumed": 3.0 if complete else None,
        "raw_reward_excluding_rejection_penalty": 2.0 if complete else None,
        "release_rejection_penalty": -0.1 if complete else None,
        "complete_scene_metrics_known": complete,
    }
    reference = {
        "status": "completed", "normal": 4, "emergency": 0, "failed": 1,
        "diverted": 0, "unresolved": 0, "generated": 24, "admitted": 24,
        "completed": 4, "resolved": 5, "wait_seconds": 2.0,
        "request_elapsed_total_seconds": 3.0, "energy_total_consumed": 4.0,
        "raw_reward_excluding_rejection_penalty": 1.0, "release_rejection_penalty": -0.2,
        "complete_scene_metrics_known": True,
    }
    return {
        "pair": {
            "pair_id": pair_id,
            "env_seed": env_seed if env_seed is not None else spec["env_seed"],
            "keyed_policy_seed": keyed_seed if keyed_seed is not None else spec["keyed_policy_seed"],
        },
        "status": "PAIR_OK" if complete else "PAIR_CENSORED",
        "control_metrics": control, "reference_metrics": reference,
        "complete_scene_delta": {"complete_scene_delta_known": complete},
        "reference_control_checks": {"hard_safety_clear": True, "technical_ok": True},
        "new_failed_arrival_ids": [] if complete else None,
        "new_unresolved_arrival_ids": [] if complete else None,
        "new_diverted_arrival_ids": [] if complete else None,
    }


def _natural_arm(spec):
    ids = list(range(24))
    return {
        "status": "completed", "censored": False,
        "env_seed": spec["env_seed"], "keyed_policy_seed": spec["keyed_policy_seed"],
        "generated_requests": 24, "admitted_count": 24,
        "model_unchanged": True, "caller_rng_unchanged": True,
        "safety": {"safety_violation": False},
        "ledger": {
            "requested_ids": ids, "outcomes": {str(i): "completed" for i in ids},
            "conservation": {"requested_equals_resolved_union_unresolved": True},
            "terminal_alignment": {"valid": True},
        },
        "energy_ledger": {"coverage": {"valid": True}},
        "wait_integrity": {
            "all_request_wait_observed_finite": True,
            "external_queue_integral_observed_finite": True,
            "internal_holding_observed_finite": True,
            "elapsed_by_requested_id_observed_finite": True,
        },
    }


def test_registered_grid_is_six_by_four_and_imports_nine():
    assert [item["pair_id"] for item in runner.PAIR_SPECS] == list(range(1, 25))
    assert runner.IMPORTED_PAIR_IDS == {1, 5, 9, 13, 14, 15, 16, 17, 21}
    assert len({item["env_seed"] for item in runner.PAIR_SPECS}) == 6
    assert len({item["keyed_policy_seed"] for item in runner.PAIR_SPECS}) == 4


def test_reference_loader_checks_all_24_arms(tmp_path: Path):
    (tmp_path / "arms").mkdir()
    pair_statuses = {str(item["pair_id"]): {"legacy": "completed"} for item in runner.PAIR_SPECS}
    (tmp_path / "full.json").write_text(json.dumps({
        "status": "FULL_OK", "pair_count": 24,
        "pair_ids": list(range(1, 25)), "pair_statuses": pair_statuses,
    }), encoding="utf-8")
    for spec in runner.PAIR_SPECS:
        (tmp_path / "arms" / f"pair_{spec['pair_id']:04d}_legacy.json").write_text(
            json.dumps(_natural_arm(spec)), encoding="utf-8",
        )
    reference, arms = runner._reference_arms(tmp_path)
    assert reference["pair_count"] == 24
    assert len(reference["arms"]) == 24
    assert len(arms) == 24
    assert reference["arms"]["1"]["env_seed"] == 877101


def test_aggregate24_reports_balanced_business_criterion_and_profiles():
    summaries = [_summary(pair_id) for pair_id in range(1, 25)]
    review = runner._aggregate24(summaries)
    assert review["pair_count"] == 24
    assert review["all_24_natural"] is True
    assert review["normal_not_lower"] is True
    assert review["successful_not_lower"] is True
    assert review["wait_not_higher"] is True
    assert review["strict_business_gain"] is True
    assert review["criterion_pass"] is True
    assert review["emergency_total"] == 0.0
    assert review["diverted_total"] == 0.0
    assert review["unresolved_total"] == 0.0
    assert review["generated_total"] == 576.0
    assert review["admitted_total"] == 576.0
    assert review["resolved_total"] == 120.0
    assert review["raw_reward_excluding_rejection_penalty_total"] == 48.0
    assert review["release_rejection_penalty_total"] == pytest.approx(-2.4)
    profiles = runner._profiles(summaries)
    assert len(profiles) == 6
    assert all(item["pair_count"] == 4 and item["descriptive_only"] for item in profiles)


def test_censored_pair_keeps_unknown_aggregate_and_no_new_ids():
    summaries = [_summary(pair_id) for pair_id in range(1, 25)]
    summaries[15] = _summary(16, complete=False)
    review = runner._aggregate24(summaries)
    assert review["all_24_natural"] is False
    assert review["normal_total"] is None
    assert review["wait_total_seconds"] is None
    assert review["no_new_failure_or_unresolved_or_diverted"] is None
    assert review["criterion_pass"] is False


def test_full_fixture_serialization_preserves_imported_ids_and_unknowns(tmp_path: Path):
    payload = {
        "stage": runner.STAGE, "status": "FULL_OK", "pair_count": 24,
        "imported_pair_ids": sorted(runner.IMPORTED_PAIR_IDS),
        "balanced_business_review": {"criterion_pass": False, "normal_total": None},
    }
    path = tmp_path / "full.json"
    runner._write_json(path, payload)
    restored = json.loads(path.read_text(encoding="utf-8"))
    assert restored["pair_count"] == 24
    assert restored["imported_pair_ids"] == sorted(runner.IMPORTED_PAIR_IDS)
    assert restored["balanced_business_review"]["normal_total"] is None


def test_aggregate_requires_exactly_24_summaries():
    with pytest.raises(runner.BalancedProposalSupportError, match="all 24"):
        runner._aggregate24([_summary(1)])


def test_aggregate_rejects_duplicate_ids_technical_failure_and_unknown_issue_list():
    summaries = [_summary(pair_id) for pair_id in range(1, 25)]
    duplicate = list(summaries)
    duplicate[-1] = _summary(1)
    with pytest.raises(runner.BalancedProposalSupportError, match="unique pair IDs"):
        runner._aggregate24(duplicate)
    technical = [_summary(pair_id) for pair_id in range(1, 25)]
    technical[0]["reference_control_checks"]["technical_ok"] = False
    assert runner._aggregate24(technical)["criterion_pass"] is False
    unknown_issue = [_summary(pair_id) for pair_id in range(1, 25)]
    unknown_issue[0]["new_failed_arrival_ids"] = None
    review = runner._aggregate24(unknown_issue)
    assert review["new_issue_lists_known"] is False
    assert review["criterion_pass"] is False


def test_profiles_reject_missing_seed_group():
    with pytest.raises(runner.BalancedProposalSupportError, match="requires four"):
        runner._profiles([_summary(pair_id) for pair_id in range(1, 24)])


def test_v1_arm_content_must_match_immutable_summary():
    arm = {
        "status": "completed",
        "proposal_support_control": dict(runner.EXPECTED_CONTROL),
        "reference_control_checks": {"technical_ok": True},
        "generated_requests": 1, "admitted_count": 1, "completed_count": 1, "resolved_count": 1,
        "outcome_counts": {"normal": 1, "emergency": 0, "failed": 0, "diverted": 0, "unresolved": 0},
        "ledger": {"requested_ids": [0]},
        "energy_ledger": {
            "rows": {"0": {"arrival_id": 0, "energy_consumed": 1.0}},
            "unique_arrival_ids": [0], "coverage": {"requested_ids": [0], "valid": True},
        },
        "wait_metrics": {"request_elapsed_seconds_by_arrival": {"0": 0.0}},
        "wait_integrity": {"elapsed_by_requested_id_observed_finite": True},
    }
    actual = runner.support.gate._business_metrics(arm)
    summary = {
        "status": "PAIR_OK",
        "control_metrics": actual,
        "reference_control_checks": dict(arm["reference_control_checks"]),
    }
    runner._validate_v1_arm(summary, arm, 1)
    roundtrip = copy.deepcopy(summary)
    roundtrip["control_metrics"]["energy_total_consumed"] += 1e-12
    runner._validate_v1_arm(roundtrip, arm, 1)
    too_large = copy.deepcopy(summary)
    too_large["control_metrics"]["energy_total_consumed"] += 1e-6
    with pytest.raises(runner.BalancedProposalSupportError, match="arm metrics"):
        runner._validate_v1_arm(too_large, arm, 1)
    none_value = copy.deepcopy(summary)
    none_value["control_metrics"]["energy_total_consumed"] = None
    with pytest.raises(runner.BalancedProposalSupportError, match="arm metrics"):
        runner._validate_v1_arm(none_value, arm, 1)
    summary["control_metrics"]["status"] = "tampered"
    with pytest.raises(runner.BalancedProposalSupportError, match="arm metrics"):
        runner._validate_v1_arm(summary, arm, 1)
