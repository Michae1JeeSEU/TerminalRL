from __future__ import annotations

import json

import pytest

from terminal_marl.training import evaluate_p3_proposal_head_probe_v1 as runner


def _row(arrival_id: int, *, outcome: str, priority: int = 1, wait: float = 10.0) -> dict:
    return {
        "arrival_id": arrival_id,
        "uid": arrival_id,
        "base_priority": priority,
        "initial_soc": 0.8,
        "final_soc": 0.6,
        "request_time": float(arrival_id),
        "touchdown_time": float(arrival_id) + 20.0,
        "wait_seconds": wait,
        "outcome": outcome,
    }


def _life(rows: list[dict], *, exact: bool = True, scope: str = "all_requested_terminal_rows") -> dict:
    outcomes: dict[str, int] = {}
    for row in rows:
        outcomes[row["outcome"]] = outcomes.get(row["outcome"], 0) + 1
    return {
        "scope": scope,
        "replay_exact": exact,
        "all_requests_observed": True,
        "missing_request_count": 0,
        "requested_count": len(rows),
        "observed_terminal_count": len(rows),
        "row_count": len(rows),
        "outcomes": outcomes,
        "rows": rows,
    }


def _record(seed: int, scenario: str, rows: list[dict], *, full: bool = True) -> dict:
    life = _life(rows, exact=full, scope=(
        "all_requested_terminal_rows" if full else "successful_terminal_rows"
    ))
    return {
        "seed": seed,
        "scenario": scenario,
        "lifecycle": life,
        "normal": sum(row["outcome"] == "completed" for row in rows),
        "emergency": sum(row["outcome"] == "emergency_landed" for row in rows),
        "failed_energy": sum(row["outcome"] == "failed_energy" for row in rows),
        "other_failures": {},
        "unresolved": 0,
        "safety": {"collisions": 0},
    }


def _summary(*, normal: int, priority: float, low_soc: float, full: bool = True) -> dict:
    return {
        "normal": normal,
        "failed_energy": 0,
        "other": 0,
        "unresolved": 0,
        "safety_nonzero_episodes": 0,
        "full_request_complete": full,
        "priority_weighted_wait_seconds": priority,
        "low_soc_wait_seconds": low_soc,
    }


def test_split_records_keeps_registered_train_validation_seed_sets() -> None:
    records = [
        {"seed": seed, "scenario": runner.SEED_SCENARIO[seed]}
        for seed in sorted(runner.SEED_SCENARIO)
    ]
    split = runner._split_records(records)
    assert [row["seed"] for row in split["train"]] == [873101, 873102, 873103]
    assert [row["seed"] for row in split["validation"]] == [873104, 873105, 873106]
    assert len(split["overall"]) == 6

    with pytest.raises(runner.ProposalHeadEvaluationError, match="duplicate"):
        runner._split_records(records + [records[0]])


def test_normalize_record_rejects_success_only_legacy_lifecycle() -> None:
    legacy = _record(873101, "flow6", [_row(0, outcome="completed")], full=False)
    with pytest.raises(runner.ProposalHeadEvaluationError, match="success_only_legacy"):
        runner.normalize_record(
            legacy,
            expected_scenario="flow6",
            expected_seed=873101,
            expected_requests=1,
            strict=True,
        )


def test_normalize_invalid_lifecycle_preserves_observed_energy_failure() -> None:
    partial = _record(
        873103,
        "dense24",
        [_row(0, outcome="completed"), _row(1, outcome="failed_energy")],
        full=False,
    )
    normalized = runner.normalize_record(
        partial,
        expected_scenario="dense24",
        expected_seed=873103,
        expected_requests=3,
        strict=False,
    )
    assert normalized["lifecycle_valid"] is False
    assert normalized["outcome_counts"] == {
        "requests": 3,
        "normal": 1,
        "emergency": 0,
        "failed_energy": 1,
        "other": 0,
        "unresolved": 1,
    }


def test_safety_nonzero_ignores_routine_telemetry_and_counts_violations() -> None:
    record = {
        "summary": {
            "safety": {
                "collisions": 0,
                "nmac": 0,
                "hard_separation_violations": 0,
                "uncertified_execution": 0,
                "SafetyInvariantError": 0,
                "shield": 12,
                "runtime_shortlist_candidates_checked": 48,
            }
        }
    }
    assert runner._safety_nonzero(record) is False
    record["summary"]["safety"]["hard_separation_violations"] = 1
    assert runner._safety_nonzero(record) is True


def test_aggregate_preserves_failed_request_in_full_denominator() -> None:
    record = _record(
        873103,
        "dense24",
        [_row(0, outcome="completed"), _row(1, outcome="failed_energy")],
    )
    normalized = runner.normalize_record(
        record,
        expected_scenario="dense24",
        expected_seed=873103,
        expected_requests=2,
    )
    aggregate = runner._aggregate([normalized])
    assert aggregate["counts"] == {
        "requests": 2,
        "normal": 1,
        "emergency": 0,
        "failed_energy": 1,
        "other": 0,
        "unresolved": 0,
    }
    assert aggregate["full_request_complete"] is True
    assert aggregate["lifecycle"]["requested_count"] == 2


def test_compare_screen_requires_full_lifecycle_and_wait_limits() -> None:
    baseline = {
        split: _summary(normal=10, priority=100.0, low_soc=80.0)
        for split in ("train", "validation", "overall")
    }
    candidate = {
        split: _summary(normal=11, priority=102.0, low_soc=81.0)
        for split in ("train", "validation", "overall")
    }
    result = runner.compare_screen(
        baseline,
        candidate,
        {"qK3_coverage_improved": True},
    )
    assert result["development_screen"] is True
    assert result["split_checks"]["validation"]["screen_pass"] is True

    incomplete = {
        split: _summary(normal=11, priority=102.0, low_soc=81.0, full=False)
        for split in ("train", "validation", "overall")
    }
    failed = runner.compare_screen(
        baseline,
        incomplete,
        {"qK3_coverage_improved": True},
    )
    assert failed["development_screen"] is False

    over_limit = {
        split: _summary(normal=11, priority=106.0, low_soc=81.0)
        for split in ("train", "validation", "overall")
    }
    limited = runner.compare_screen(
        baseline,
        over_limit,
        {"qK3_coverage_improved": True},
    )
    assert limited["split_checks"]["validation"]["priority_wait"]["passed"] is False
    assert limited["development_screen"] is False


def test_fit_coverage_improvement_accepts_before_after_report() -> None:
    assert runner._fit_coverage_improved({
        "metrics_before": {"qK3_coverage": 0.2},
        "metrics_after": {"qK3_coverage": 0.3},
    }) is True
    assert runner._fit_coverage_improved({
        "metrics_before": {"qK3_coverage": 0.3},
        "metrics_after": {"qK3_coverage": 0.2},
    }) is False
    assert runner._fit_coverage_improved({
        "metrics_before": {"qK3_coverage": 0.2},
        "metrics_after": {"qK3_coverage": 0.2000005},
    }) is False
    assert runner._fit_coverage_improved({
        "metrics_before": {"qK3_coverage": 0.2},
        "metrics_after": {"qK3_coverage": 0.200001},
    }) is True


def _fit_report_fixture(tmp_path, *, effective_fit: object = True) -> tuple[dict, object]:
    fit_dir = tmp_path / "fit"
    fit_dir.mkdir(parents=True)
    checkpoint = fit_dir / "candidate.pt"
    checkpoint.write_bytes(b"candidate-checkpoint")
    report = {
        "status": "COMPLETED",
        "source_checkpoint_sha256": runner.SOURCE_CHECKPOINT_SHA256,
        "protocol_sha256": runner._protocol_sha256(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": runner._sha256(checkpoint),
        "effective_fit": effective_fit,
        "checkpoint_roundtrip_model_exact": True,
        "checkpoint_roundtrip_optimizer_exact": True,
        "invariants": {
            "trainable_parameter_names": ["proposal_head.weight", "proposal_head.bias"],
            "trainable_parameter_count": 2425,
            "nonproposal_state_bitwise_unchanged": True,
            "gate_logits_bitwise_unchanged": True,
            "all_nonproposal_parameters_frozen": True,
            "checkpoint_roundtrip_model_exact": True,
            "checkpoint_roundtrip_optimizer_exact": True,
        },
    }
    (fit_dir / "fit_report.json").write_text(json.dumps(report), encoding="utf-8")
    return report, fit_dir


def test_fit_binding_requires_effective_fit_and_exact_invariants(tmp_path) -> None:
    _, fit_dir = _fit_report_fixture(tmp_path)
    report, _, _ = runner._load_fit_binding(fit_dir)
    assert report["effective_fit"] is True

    missing_effective, missing_dir = _fit_report_fixture(tmp_path / "missing")
    del missing_effective["effective_fit"]
    (missing_dir / "fit_report.json").write_text(json.dumps(missing_effective), encoding="utf-8")
    with pytest.raises(runner.ProposalHeadEvaluationError, match="effective_fit"):
        runner._load_fit_binding(missing_dir)

    false_effective, false_dir = _fit_report_fixture(tmp_path / "false")
    false_effective["effective_fit"] = False
    (false_dir / "fit_report.json").write_text(json.dumps(false_effective), encoding="utf-8")
    with pytest.raises(runner.ProposalHeadEvaluationError, match="effective_fit"):
        runner._load_fit_binding(false_dir)
