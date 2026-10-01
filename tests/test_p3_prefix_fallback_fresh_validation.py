from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from terminal_marl.training import validate_p3_prefix_fallback_20261001 as runner


def _metrics(*, complete: bool = True, normal: int = 20, wait: float | None = 10.0) -> dict:
    return {
        "status": "completed" if complete else "running",
        "technical_ok": True,
        "natural_terminal": complete,
        "censored": not complete,
        "complete_scene_metrics_known": complete,
        "generated": 24,
        "admitted": 24,
        "successful": normal if complete else None,
        "resolved": 24 if complete else None,
        "normal": normal,
        "emergency": 24 - normal if complete else None,
        "failed": 0,
        "diverted": 0,
        "unresolved": 0 if complete else None,
        "request_identity_ids": list(range(24)),
        "request_identity_exact_24": True,
        "wait_seconds": wait if complete else None,
        "request_elapsed_total_seconds": 100.0 if complete else None,
        "energy_total_consumed": 200.0 if complete else None,
        "raw_reward_excluding_rejection_penalty": 5.0 if complete else None,
        "release_rejection_penalty": -1.0 if complete else None,
        "hard_safety_clear": True,
        "error": None,
    }


def _record(arm: str, metrics: dict | None = None, *, ids: dict | None = None) -> dict:
    is_before = arm.startswith("before")
    is_prefix = arm.endswith("prefix")
    return {
        "arm": arm,
        "checkpoint_sha256": runner.BEFORE_CHECKPOINT_SHA256 if is_before else runner.AFTER_CHECKPOINT_SHA256,
        "config": {
            "action_schema_version": runner.ACTION_SCHEMA,
            "observation_schema_version": runner.OBSERVATION_SCHEMA,
        },
        "runtime_capture": {"available": True, "factory_calls": 1, "replay_factory_calls": 1},
        "prefix_fallback_control": (
            {"source_distributions_immutable": True, "sampler_aliases_restored": True, "runtime_aliases_restored": True}
            if is_prefix else None
        ),
        "status": "completed",
        "model_unchanged": True,
        "weights_unchanged": True,
        "caller_rng_unchanged": True,
        "observer_rng_unchanged": True,
        "teacher_called": False,
        "training_or_update_called": False,
        "safety_nonzero": False,
        "metrics": metrics or _metrics(),
        "_ids": ids or {"failed": [], "diverted": [], "unresolved": []},
    }


def _summary(cell_id: int, *, after_prefix: dict | None = None) -> dict:
    cell = dict(runner.CELL_SPECS[cell_id - 1])
    rows = {
        arm: _compact(arm, after_prefix if arm == "after_prefix" and after_prefix else None)
        for arm in runner.ARMS
    }
    return {
        "status": "PAIR_OK",
        "cell": cell,
        "arms": rows,
        "request_identity_exact_24": True,
        "technical_ok": True,
        "hard_safety_clear": True,
        "all_four_natural": all(row["metrics"]["complete_scene_metrics_known"] for row in rows.values()),
        "new_failed_arrival_ids_after_prefix_vs_after_legacy": [],
        "new_diverted_arrival_ids_after_prefix_vs_after_legacy": [],
        "new_unresolved_arrival_ids_after_prefix_vs_after_legacy": [],
        "new_failed_arrival_ids_after_prefix_vs_before_prefix": [],
        "new_diverted_arrival_ids_after_prefix_vs_before_prefix": [],
        "new_unresolved_arrival_ids_after_prefix_vs_before_prefix": [],
        "new_failed_arrival_ids_before_prefix_vs_before_legacy": [],
        "new_diverted_arrival_ids_before_prefix_vs_before_legacy": [],
        "new_unresolved_arrival_ids_before_prefix_vs_before_legacy": [],
    }


def _compact(arm: str, metrics: dict | None = None) -> dict:
    return {"arm": arm, "metrics": metrics or _metrics()}


def test_fresh_grid_and_schema_contract_are_independent_of_old_cells():
    assert [cell["env_seed"] for cell in runner.CELL_SPECS] == [886101, 886102, 886103, 886104, 886105, 886106]
    assert [cell["policy_seed"] for cell in runner.CELL_SPECS] == [886201, 886202, 886203, 886204, 886205, 886206]
    assert runner.ARMS == ("before_legacy", "after_legacy", "before_prefix", "after_prefix")
    contract = runner._schema_contract()
    assert "request_identity" in contract["required_metric_fields"]
    assert "release_rejection_penalty" in contract["required_metric_fields"]


def test_aggregate_keeps_censored_costs_unknown():
    rows = [_summary(index) for index in range(1, 7)]
    rows[2]["arms"]["after_prefix"]["metrics"] = _metrics(complete=False)
    aggregate = runner._aggregate(rows, "after_prefix")
    assert aggregate["known_complete"] is False
    assert aggregate["wait_seconds"] is None
    assert aggregate["energy_total_consumed"] is None
    assert aggregate["generated"] == 144.0


def test_business_review_uses_aggregate_screen_and_no_per_cell_cost_gate():
    rows = []
    for index in range(1, 7):
        row = _summary(index)
        row["arms"]["after_prefix"]["metrics"] = _metrics(normal=22, wait=9.0)
        rows.append(row)
    review = runner._business_review(rows)
    assert review["criterion_pass"] is True
    assert review["reference_comparisons"]["after_prefix_vs_after_legacy"]["normal_not_lower"] is True
    rows[0]["arms"]["after_prefix"]["metrics"]["wait_seconds"] = 50.0
    for row in rows[1:]:
        row["arms"]["after_prefix"]["metrics"]["wait_seconds"] = 0.0
    # The protocol keeps per-cell regressions visible without inventing a
    # per-cell hard cost gate; the aggregate remains mechanically screenable.
    assert runner._business_review(rows)["criterion_pass"] is True


def test_pair_summary_preserves_all_four_arm_delta_fields(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        runner,
        "_ids",
        lambda record, kinds: [value for kind in kinds for value in record.get("_ids", {}).get(kind, [])],
    )
    cell = dict(runner.CELL_SPECS[0])
    records = {arm: _record(arm) for arm in runner.ARMS}
    records["after_prefix"]["metrics"]["normal"] = 21
    summary = runner._pair_summary(
        cell,
        records,
        {arm: tmp_path / f"{arm}.json" for arm in runner.ARMS},
    )
    assert summary["status"] == "PAIR_OK"
    assert summary["after_prefix_minus_after_legacy"]["normal"] == 1.0
    assert summary["after_prefix_minus_before_prefix"]["normal"] == 1.0
    assert summary["request_identity_exact_24"] is True


def test_run_cell_retains_fatal_arm_artifact_and_stops_before_next_arm(monkeypatch, tmp_path: Path):
    calls: list[str] = []

    def fail_first(*, config, cell, arm, output):
        calls.append(arm)
        raise RuntimeError("native fixture failure")

    monkeypatch.setattr(runner, "_arm_record", fail_first)
    summary = runner._run_cell(tmp_path, object(), dict(runner.CELL_SPECS[0]))
    assert calls == ["before_legacy"]
    assert summary["status"] == "PAIR_TECHNICAL_ERROR"
    artifact = tmp_path / "arms" / "cell_01_before_legacy.json"
    assert artifact.is_file()
    assert json.loads(artifact.read_text(encoding="utf-8"))["error"]["type"] == "RuntimeError"
    assert not (tmp_path / "arms" / "cell_01_after_legacy.json").exists()


def test_pilot_reuse_rejects_mutated_arm(tmp_path: Path):
    (tmp_path / "preflight.json").write_text("{}\n", encoding="utf-8")
    arms_dir = tmp_path / "arms"
    pairs_dir = tmp_path / "pairs"
    arms_dir.mkdir()
    pairs_dir.mkdir()
    arm_paths = {}
    arm_hashes = {}
    for arm in runner.ARMS:
        path = arms_dir / f"cell_01_{arm}.json"
        path.write_text(json.dumps({"arm": arm}) + "\n", encoding="utf-8")
        arm_paths[arm] = str(path)
        arm_hashes[arm] = runner._sha256(path)
    pair = {
        "status": "PAIR_OK", "cell": {"cell_id": 1},
        "arms": {arm: {} for arm in runner.ARMS}, "technical_ok": True,
    }
    pair_path = pairs_dir / "cell_01.json"
    pair_path.write_text(json.dumps(pair) + "\n", encoding="utf-8")
    pilot = {
        "status": "PILOT_OK", "preflight_sha256": runner._sha256(tmp_path / "preflight.json"),
        "preflight_runner_sha256": "runner", "pilot_artifacts": {
            "pair_path": str(pair_path), "pair_sha256": runner._sha256(pair_path),
            "arm_paths": arm_paths, "arm_sha256": arm_hashes,
        },
    }
    (tmp_path / "pilot.json").write_text(json.dumps(pilot) + "\n", encoding="utf-8")
    with pytest.raises(runner.FreshValidationError, match="pilot arm after_legacy changed"):
        (arms_dir / "cell_01_after_legacy.json").write_text("changed\n", encoding="utf-8")
        runner._validate_pilot_reuse(tmp_path, {"runner_sha256": "runner"})


def test_parser_exposes_only_registered_phases():
    parser = runner.build_parser()
    assert parser.parse_args(["--phase", "preflight"]).phase == "preflight"
    with pytest.raises(SystemExit):
        parser.parse_args(["--phase", "train"])
