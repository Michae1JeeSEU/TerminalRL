from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from terminal_marl.training import evaluate_p3_runtime_rank_repair as runner


def _row(arrival_id: int, *, outcome: str = "completed", wait: float = 10.0) -> dict:
    return {
        "arrival_id": arrival_id,
        "uid": arrival_id,
        "base_priority": 1,
        "initial_soc": 0.8,
        "final_soc": 0.6,
        "request_time": float(arrival_id),
        "touchdown_time": float(arrival_id) + wait,
        "wait_seconds": wait,
        "outcome": outcome,
    }


def _life(rows: list[dict], *, exact: bool = True) -> dict:
    outcomes: dict[str, int] = {}
    for row in rows:
        outcomes[row["outcome"]] = outcomes.get(row["outcome"], 0) + 1
    return {
        "scope": "all_requested_terminal_rows",
        "replay_exact": exact,
        "all_requests_observed": exact,
        "ledger_alignment": exact,
        "missing_request_count": 0 if exact else len(rows),
        "requested_count": len(rows),
        "observed_terminal_count": len(rows),
        "row_count": len(rows),
        "outcomes": outcomes,
        "rows": rows,
        "priority_weighted_wait_seconds": 10.0,
        "low_soc": {"mean_wait_seconds": 10.0, "count": len(rows)},
    }


def _record(seed: int, scenario: str, *, failed: bool = False) -> dict:
    requests = runner.EXPECTED_REQUESTS[scenario]
    rows = [_row(index, outcome="failed_energy" if failed and index == 0 else "completed")
            for index in range(requests)]
    life = _life(rows)
    return {
        "seed": seed,
        "scenario": scenario,
        "lifecycle": life,
        "summary": {"safety": {"collisions": 0}},
        "normal": requests - int(failed),
        "emergency": 0,
        "failed_energy": int(failed),
        "unresolved": 0,
        "other_failures": {},
        "safety": {"collisions": 0},
    }


def _baseline() -> list[dict]:
    return [_record(seed, runner.SEED_SCENARIO[seed]) for seed in runner.SEED_SCENARIO]


def test_cells_are_exactly_the_existing_six_development_cells() -> None:
    cells = runner._evaluation_cells()
    assert [(cell["seed"], cell["scenario"], cell["split"]) for cell in cells] == [
        (873101, "flow6", "train"),
        (873102, "flow12", "train"),
        (873103, "dense24", "train"),
        (873104, "flow6", "validation"),
        (873105, "flow12", "validation"),
        (873106, "dense24", "validation"),
    ]


def test_strict_no_regression_uses_exact_wait_limits() -> None:
    base = {name: {"normal": 5, "priority_weighted_wait_seconds": 100.0,
                   "low_soc_wait_seconds": 80.0} for name in ("validation", "overall")}
    candidate = {name: {"normal": 5, "priority_weighted_wait_seconds": 100.0,
                        "low_soc_wait_seconds": 80.0} for name in ("validation", "overall")}
    assert runner.no_regression_strict(base, candidate)["passed"]
    candidate["overall"]["priority_weighted_wait_seconds"] = 100.0001
    assert not runner.no_regression_strict(base, candidate)["passed"]


def _screen_summary(*, normal: int, priority: float, low_soc: float) -> dict:
    return {
        "normal": normal,
        "failed_energy": 0,
        "other": 0,
        "unresolved": 0,
        "safety_nonzero_episodes": 0,
        "safety_status_unknown_episodes": 0,
        "full_request_complete": True,
        "priority_weighted_wait_seconds": priority,
        "low_soc_wait_seconds": low_soc,
    }


def test_compare_screen_maps_nested_train_qk3_metrics() -> None:
    baseline = {
        split: _screen_summary(normal=10, priority=100.0, low_soc=80.0)
        for split in ("train", "validation", "overall")
    }
    candidate = {
        split: _screen_summary(normal=11, priority=102.0, low_soc=81.0)
        for split in ("train", "validation", "overall")
    }
    result = runner.compare_screen(
        baseline,
        candidate,
        {
            "metrics_before": {"train": {"qK3_mean": 0.20}},
            "metrics_after": {"train": {"qK3_mean": 0.200001}},
        },
    )
    assert result["coverage_improved"] is True
    assert result["development_screen"] is True


def test_nested_qk3_improvement_does_not_override_degraded_screen_gates() -> None:
    baseline = {
        split: _screen_summary(normal=10, priority=100.0, low_soc=80.0)
        for split in ("train", "validation", "overall")
    }
    candidate = {
        split: _screen_summary(normal=9, priority=106.0, low_soc=86.0)
        for split in ("train", "validation", "overall")
    }
    result = runner.compare_screen(
        baseline,
        candidate,
        {
            "metrics_before": {"train": {"qK3_mean": 0.20}},
            "metrics_after": {"train": {"qK3_mean": 0.30}},
        },
    )
    assert result["coverage_improved"] is True
    assert result["split_checks"]["validation"]["normal"]["passed"] is False
    assert result["split_checks"]["validation"]["priority_wait"]["passed"] is False
    assert result["development_screen"] is False


def test_normalization_retains_failed_request_in_denominator() -> None:
    raw = _record(873103, "dense24", failed=True)
    normalized = runner.normalize_record(
        raw, expected_scenario="dense24", expected_seed=873103,
        expected_requests=runner.EXPECTED_REQUESTS["dense24"],
    )
    assert normalized["outcome_counts"]["requests"] == 24
    assert normalized["outcome_counts"]["failed_energy"] == 1
    assert normalized["outcome_counts"]["normal"] == 23


def test_fit_binding_rejects_old_kl_stage(tmp_path: Path) -> None:
    fit = tmp_path / "fit"
    fit.mkdir()
    checkpoint = fit / "candidate.pt"
    checkpoint.write_bytes(b"candidate")
    report = {
        "stage": "P3_ADAPTIVE_KL_FORMAL_ADMISSION_V7",
        "status": "COMPLETED",
        "source_checkpoint_sha256": runner.SOURCE_CHECKPOINT_SHA256,
        "scope": "proposal_head_only",
        "checkpoint_restore_exact": True,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": runner._sha256(checkpoint),
    }
    (fit / "fit_report.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(runner.RuntimeRankEvaluationError, match="stage"):
        runner._load_fit_binding(fit)


def test_smoke_prefix_forces_screen_false_and_uses_one_cell(monkeypatch, tmp_path: Path) -> None:
    # This test exercises only the smoke guard; the full checkpoint loader is
    # replaced with small policy doubles.
    class Model:
        def state_dict(self):
            return {"proposal_head.weight": _Tensor(), "proposal_head.bias": _Tensor(),
                    "actor_encoder.weight": _Tensor()}

        def eval(self):
            return self

    class _Tensor:
        def detach(self): return self
        def cpu(self): return self
        def contiguous(self): return self
        def numpy(self): return b"tensor"

    source = SimpleNamespace(model=Model(), optimizer=SimpleNamespace(state={}), observation_dim=88)
    candidate = SimpleNamespace(model=Model(), optimizer=SimpleNamespace(state={}), observation_dim=88)
    # Old freeze validation requires an actual changed proposal tensor.  Use
    # test doubles for the delegated wrapper instead of an artificial model.
    monkeypatch.setattr(runner, "_load_fit_binding", lambda _: ({
        "stage": runner.STAGE, "status": "COMPLETED", "scope": "proposal_head_only",
        "source_checkpoint_sha256": runner.SOURCE_CHECKPOINT_SHA256,
        "checkpoint": "candidate.pt", "checkpoint_sha256": "candidate-sha",
        "checkpoint_restore_exact": True, "metrics_before": {}, "metrics_after": {},
    }, tmp_path / "candidate.pt", "fit-sha"))
    (tmp_path / "candidate.pt").write_bytes(b"candidate")
    monkeypatch.setattr(runner, "load_baseline_records", lambda: _baseline())
    monkeypatch.setattr(runner.formal_v7, "_protocol", lambda: {})
    config = SimpleNamespace(max_simulation_seconds=10.0, dt=5.0)
    monkeypatch.setattr(runner, "_build_eval_inputs", lambda _: ({"flow6": config, "flow12": config, "dense24": config}, {}))
    monkeypatch.setattr(runner, "SOURCE_CHECKPOINT", tmp_path / "source.pt")
    (tmp_path / "source.pt").write_bytes(b"source")
    monkeypatch.setattr(runner, "_sha256", lambda path: runner.SOURCE_CHECKPOINT_SHA256 if str(path).endswith("source.pt") else "candidate-sha")
    monkeypatch.setattr(runner, "_load_policy", lambda *_: (source, {}))
    monkeypatch.setattr(runner, "validate_candidate_freeze", lambda *_: {"proposal_parameter_changed": True})
    monkeypatch.setattr(runner, "_protocol_sha256", lambda: "protocol-sha")
    monkeypatch.setattr(runner, "_model_digest", lambda _: "stable")
    calls = []
    def fake_rollout(*args, **kwargs):
        calls.append({"seed": kwargs["seed"], "max_steps": kwargs["max_steps"],
                      "deterministic": kwargs["deterministic"],
                      "collect_labels": kwargs["collect_labels"]})
        return [], {"action_tape": "tape", "outcomes": {"completed": 6},
                    "safety": {"collisions": 0}}
    monkeypatch.setattr(runner.student_rollout_v4_2, "rollout", fake_rollout)
    monkeypatch.setattr(runner.terminal_lifecycle_metrics, "terminal_lifecycle_metrics",
                        lambda *args: _life([_row(i) for i in range(6)]))
    result = runner.run(fit_dir=tmp_path / "fit", output_dir=tmp_path / "out", smoke=True)
    assert result["records_processed"] == 1
    assert result["records_expected"] == 1
    assert result["development_screen"] is False
    assert result["comparison"]["development_screen"] is False
    assert result["no_regression_strict"]["passed"] is False
    assert calls == [{"seed": 873101, "max_steps": 2,
                      "deterministic": False, "collect_labels": False}]
