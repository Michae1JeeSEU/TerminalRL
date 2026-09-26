from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from terminal_marl.training import collect_p3_runtime_rank_repair as runner


class _Actor:
    def __init__(self, arrival_id: int):
        self.arrival_id = arrival_id


class _Env:
    def __init__(self, config=None):
        self.step_count = 0
        self.active = {1: _Actor(11), 2: _Actor(22)}

    def _observations(self):
        return {
            "1": np.full(88, float(self.step_count), dtype=np.float32),
            "2": np.full(88, float(self.step_count), dtype=np.float32),
        }

    def step(self, actions, proposal_shortlists=None):
        self.step_count += 1
        return self._observations(), {}, {}, {}, {}


def _row(step: int, uid: int, arrival_id: int) -> dict:
    return {
        "seed": 873101,
        "split": "train",
        "step": step,
        "uid": uid,
        "arrival_id": arrival_id,
        "observation": [float(step)] * 88,
        "source_proposal_logits": [0.0] * 25,
        "source_gate_logits": [0.0, 0.0],
        "certified_proposals": [4],
    }


def _branch_report() -> dict:
    branch = {"reward": 0.0, "next_value": 0.0, "decision": "safety_reject", "safety_counter_delta": {}}
    return {
        "wait": dict(branch),
        "releases": [dict(branch) for _ in range(25)],
        "runtime_singleton_branches": [dict(branch) for _ in range(25)],
        "runtime_certified_proposals": [7],
        "runtime_q_vs_approved_mass_abs_error": 0.0,
        "label_scope": runner.LABEL_SCOPE,
    }


def test_one_replay_captures_multiple_rows_and_preserves_other_actions(tmp_path, monkeypatch):
    tape = tmp_path / "tape.json"
    tape.write_text(json.dumps({"action_tape": [{}, {}]}), encoding="utf-8")
    calls = []

    def replay(env, path):
        env.step(
            {"1": np.full(3, 1.0, dtype=np.float32), "2": np.full(3, 2.0, dtype=np.float32)},
            {"2": (np.full(3, 7.0, dtype=np.float32),)},
        )
        env.step(
            {"1": np.full(3, 3.0, dtype=np.float32), "2": np.full(3, 4.0, dtype=np.float32)},
            {"2": (np.full(3, 8.0, dtype=np.float32),)},
        )
        return {"exact": True, "steps_replayed": 2}

    def evaluate(base, row, policy, actions, shortlists, branch_rng):
        calls.append((row["step"], actions, shortlists))
        assert set(actions) == {"1", "2"}
        assert "2" in shortlists
        result = _branch_report()
        result["branch_rng_restored"] = True
        return result

    monkeypatch.setattr(runner, "TerminalParallelEnv", _Env)
    monkeypatch.setattr(runner.SafetyTraceRecorder, "replay_action_tape", staticmethod(replay))
    monkeypatch.setattr(runner.runtime_diagnostic, "_evaluate_branches", evaluate)
    monkeypatch.setattr(runner.collector.evaluator, "_model_digest", lambda policy: "model")
    monkeypatch.setattr(runner.collector, "_summary_outcomes", lambda path: {})
    monkeypatch.setattr(runner.collector, "_replay_outcomes", lambda env: {})

    result = runner._collect_record(
        record={"seed": 873101, "split": "train", "scenario": "flow6", "action_tape": str(tape)},
        config=object(),
        source_policy={"policy": object(), "model_digest_before": "model"},
        target_rows=[_row(0, 1, 11), _row(1, 2, 22)],
    )

    assert result["replay_exact"] is True
    assert result["rows_observed"] == 2
    assert len(calls) == 2
    assert np.array_equal(calls[0][1]["2"], np.full(3, 2.0, dtype=np.float32))
    assert np.array_equal(calls[1][1]["1"], np.full(3, 3.0, dtype=np.float32))
    assert result["model_weights_unchanged"] is True


def test_runtime_label_replaces_c_and_preserves_catalogue_and_serializes():
    source = _row(3, 1, 11)
    labelled, summary = runner._label_row(
        source,
        {
            **_branch_report(),
            "runtime_certified_proposals": [7, 2, 7],
            "runtime_q_vs_approved_mass_abs_error": 1e-8,
        },
    )
    assert labelled["catalogue_certified_proposals"] == [4]
    assert labelled["certified_proposals"] == [2, 7]
    assert labelled["q_runtime_vs_actual_error"] == 1e-8
    assert labelled["runtime_branch_count"] == 51
    assert labelled["safety_counter_delta_nonzero_count"] == 0
    json.loads(json.dumps(runner._jsonable(labelled), allow_nan=False))
    assert summary["runtime_certified_proposals"] == [2, 7]


def test_nonzero_safety_counter_delta_rejects_runtime_label():
    report = _branch_report()
    report["runtime_singleton_branches"][3]["safety_counter_delta"] = {"nmac": 1}
    try:
        runner._label_row(_row(3, 1, 11), report)
    except runner.RuntimeRankCollectionError as exc:
        assert "safety counter" in str(exc)
    else:
        raise AssertionError("nonzero safety counter delta was accepted")


def test_missing_target_row_is_retained_as_failure(tmp_path, monkeypatch):
    tape = tmp_path / "tape.json"
    tape.write_text(json.dumps({"action_tape": [{}]}), encoding="utf-8")

    monkeypatch.setattr(runner, "TerminalParallelEnv", _Env)
    monkeypatch.setattr(
        runner.SafetyTraceRecorder,
        "replay_action_tape",
        staticmethod(lambda env, path: {"exact": True, "steps_replayed": 1}),
    )
    monkeypatch.setattr(runner.collector.evaluator, "_model_digest", lambda policy: "model")
    monkeypatch.setattr(runner.collector, "_summary_outcomes", lambda path: {})
    monkeypatch.setattr(runner.collector, "_replay_outcomes", lambda env: {})

    result = runner._collect_record(
        record={"seed": 873101, "split": "train", "scenario": "flow6", "action_tape": str(tape)},
        config=object(),
        source_policy={"policy": object(), "model_digest_before": "model"},
        target_rows=[_row(2, 1, 11)],
    )

    assert result["missing_rows"]
    assert result["rows_observed"] == 0
    assert runner._record_ok(result, smoke=False) is False


def test_smoke_manifest_and_rows_are_serialized(tmp_path, monkeypatch):
    collection = tmp_path / "collection"
    collection.mkdir()
    tape = tmp_path / "tape.json"
    tape.write_text(json.dumps({"action_tape": [{}]}), encoding="utf-8")
    row = _row(0, 1, 11)

    monkeypatch.setattr(runner.trainer, "_load_collection", lambda path: ([row], {
        "collection_dir": str(collection),
        "manifest": str(collection / "manifest.json"),
        "manifest_sha256": "manifest",
        "rows": str(collection / "rows.json"),
        "rows_sha256": "rows",
    }))
    monkeypatch.setattr(runner, "EXPECTED_RECORDS", 1)
    monkeypatch.setattr(
        runner.collector,
        "_load_source_inputs",
        lambda root: ({}, {"flow6": object()}, [{
            "seed": 873101,
            "split": "train",
            "scenario": "flow6",
            "action_tape": str(tape),
        }], tmp_path / "formal_manifest.json"),
    )
    monkeypatch.setattr(
        runner.collector,
        "_load_source_policy",
        lambda protocol, config: {"policy": object(), "model_digest_before": "model", "sha256": "checkpoint"},
    )
    monkeypatch.setattr(runner, "TerminalParallelEnv", _Env)
    monkeypatch.setattr(
        runner.SafetyTraceRecorder,
        "replay_action_tape",
        staticmethod(lambda env, path: (env.step({"1": np.zeros(3, dtype=np.float32)}, {}), {"exact": True})[1]),
    )
    monkeypatch.setattr(runner.runtime_diagnostic, "_evaluate_branches", lambda *args: _branch_report())
    monkeypatch.setattr(runner.collector.evaluator, "_model_digest", lambda policy: "model")

    output = tmp_path / "output"
    result = runner.collect(output_dir=output, collection_dir=collection, formal_root=tmp_path, smoke=True)

    assert result["status"] == "SMOKE_COMPLETED"
    assert result["runtime_labels"] is True
    assert result["training"] is False
    assert result["rows_count"] == 1
    assert result["records_count"] == 1
    rows = json.loads((output / "rows.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert rows[0]["certified_proposals"] == [7]
    assert manifest["source_collection_manifest"] == str(collection / "manifest.json")
