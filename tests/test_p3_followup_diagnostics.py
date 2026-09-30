"""Contract tests for the bounded P3 follow-up diagnostic entry point."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import numpy as np

from terminal_marl.training import run_p3_followup_diagnostics as runner


def _candidate(episode: int, step: int, uid: int) -> dict[str, int]:
    return {"episode": episode, "step": step, "uid": uid}


def test_selection_is_deterministic_without_replacement() -> None:
    candidates = [
        _candidate(episode=(index // 10) + 1, step=index, uid=index % 10)
        for index in range(48)
    ]
    first = runner._select_candidates(candidates)
    second = runner._select_candidates(candidates)

    assert first == second
    assert first["sample_size"] == runner.SELECTION_SIZE == 24
    selected = first["selected"]
    assert len(selected) == 24
    keys = [(item["episode"], item["step"], item["uid"]) for item in selected]
    assert len(set(keys)) == len(keys)
    assert [item["selection_order"] for item in selected] == list(range(1, 25))
    assert first["without_replacement"] is True


def test_four_shards_are_disjoint_and_reconstruct_registered_selection() -> None:
    selected = [{"selection_order": order, "episode": 1, "step": order, "uid": order}
                for order in range(1, runner.SELECTION_SIZE + 1)]
    shards = [
        runner._shard_selection(
            selected,
            shard_index=index,
            shard_count=4,
        )
        for index in range(4)
    ]

    shard_orders = [[item["selection_order"] for item in shard] for shard in shards]
    assert all(len(orders) == 6 for orders in shard_orders)
    assert all(orders == sorted(orders) for orders in shard_orders)
    assert len(set(order for orders in shard_orders for order in orders)) == runner.SELECTION_SIZE
    assert sorted(order for orders in shard_orders for order in orders) == list(range(1, 25))


def test_shard_arguments_are_bounded_and_diagnose_only() -> None:
    assert runner._validate_shard_args("diagnose", 0, 1) == (0, 1)
    assert runner._validate_shard_args("diagnose", 3, 4) == (3, 4)
    with pytest.raises(runner.FollowupDiagnosticError):
        runner._validate_shard_args("diagnose", -1, 4)
    with pytest.raises(runner.FollowupDiagnosticError):
        runner._validate_shard_args("diagnose", 4, 4)
    with pytest.raises(runner.FollowupDiagnosticError):
        runner._validate_shard_args("diagnose", 0, 5)
    with pytest.raises(runner.FollowupDiagnosticError):
        runner._validate_shard_args("pilot", 0, 4)
    with pytest.raises(runner.FollowupDiagnosticError):
        runner._validate_shard_args("replay", 1, 1)


def test_row_inputs_preserves_recorded_actions_and_shortlists() -> None:
    row = {
        "a_requested": {
            "4": [0.1, 0.2],
            "7": [0.3, 0.4],
        },
        "runtime_proposal_shortlists": {
            "4": [[0.0, 0.1], [0.2, 0.3]],
            "7": [[0.4, 0.5]],
        },
    }
    actions, shortlists = runner._row_inputs(row)

    assert sorted(actions) == ["4", "7"]
    assert sorted(shortlists) == ["4", "7"]
    assert len(shortlists["4"]) == 2
    assert len(shortlists["7"]) == 1
    assert actions["4"].tolist() == pytest.approx([0.1, 0.2])
    assert shortlists["4"][1].tolist() == pytest.approx([0.2, 0.3])


def test_loaded_manifest_contains_this_runner() -> None:
    manifest = runner._loaded_code_manifest()
    paths = {Path(item["path"]).resolve() for item in manifest["files"]}
    assert Path(runner.__file__).resolve() in paths
    own = next(item for item in manifest["files"] if Path(item["path"]).resolve() == Path(runner.__file__).resolve())
    assert len(own["sha256"]) == 64


def test_alignment_uses_one_target_row_and_optional_delta_q(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = object()
    after = object()
    seen: dict[str, object] = {}

    monkeypatch.setattr(runner, "_resolve", lambda path: Path("after.pt"))
    monkeypatch.setattr(runner, "_sha256", lambda path: runner.AFTER_CHECKPOINT_SHA256)
    monkeypatch.setattr(
        runner,
        "_artifact",
        lambda path, expected_sha256=None: {"sha256": expected_sha256},
    )
    monkeypatch.setattr(runner.rank_eval, "_load_policy", lambda path, config: (after, {"kind": "fake"}))
    monkeypatch.setattr(
        runner.pilot,
        "_model_digest",
        lambda policy: runner.SOURCE_WARMUP_MODEL_DIGEST if policy is before else runner.AFTER_MODEL_DIGEST,
    )

    def fake_alignment(before_policy, after_policy, world, *, delta_q=None):
        seen["before"] = before_policy
        seen["after"] = after_policy
        seen["world"] = np.asarray(world)
        seen["delta_q"] = delta_q
        return {
            "local_m": [0.25] if delta_q is not None else None,
            "delta_q": delta_q,
        }

    monkeypatch.setattr(runner.paired_gate_diagnostic, "gate_update_alignment", fake_alignment)
    result = runner._alignment_report(
        object(), before, np.zeros((1, runner.OBSERVATION_DIM), dtype=np.float32),
        target_uid=7, delta_q=1.25,
    )

    assert seen["before"] is before
    assert seen["after"] is after
    assert seen["world"].shape == (1, runner.OBSERVATION_DIM)
    assert seen["delta_q"] == pytest.approx(1.25)
    assert result["alignment_scope"]["target_uid"] == 7
    assert result["alignment"]["local_m"] == [0.25]


def test_branch_callback_persists_raw_report_and_progress(tmp_path: Path) -> None:
    callback = runner._paired_branch_callback(
        tmp_path,
        selection_order=3,
        total_branches=26,
        completed_states=2,
        expected_states=24,
        started=runner.time.monotonic(),
    )
    raw = {
        "branch": "release_4",
        "steps": 17,
        "termination_classification": "computation_budget_censored",
        "censored": True,
    }

    callback(raw, 5, 26)

    artifact = tmp_path / "branches" / "state_0003" / "branch_0005.json"
    assert json.loads(artifact.read_text()) == raw
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["phase"] == "paired_running"
    assert progress["selection_order"] == 3
    assert progress["current_branch"] == 5
    assert progress["total_branches"] == 26
    assert progress["branch_steps"] == 17
    assert progress["last_classification"] == "computation_budget_censored"
    assert progress["processed"] == 2
    assert progress["expected"] == 24
    assert progress["registered_selection_size"] == 24
    assert progress["processed_selection_size"] == 24
    assert progress["shard_index"] == 0
    assert progress["shard_count"] == 1


def test_pilot_fake_selected_state_wires_branch_callback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = {
        "selection_order": 1,
        "episode": 1,
        "step": 10,
        "uid": 4,
    }
    key = (1, 10, 4)
    source = {"policy": object(), "source_model_digest": "digest"}
    state = {
        "candidate": selected,
        "snapshot": object(),
        "episode": 1,
        "step": 10,
        "uid": 4,
        "target_world": np.zeros((1, runner.OBSERVATION_DIM), dtype=np.float32),
        "target_world_index": 0,
        "first_actions": {},
        "first_shortlists": {},
        "source_observation_hash": "obs",
        "source_action_tape_index": 10,
        "shadow_methods_restored": True,
    }
    monkeypatch.setattr(
        runner,
        "_load_selection",
        lambda output: {"selection": {"selected": [selected]}},
    )
    monkeypatch.setattr(
        runner,
        "_capture_selected_states",
        lambda output, preflight, selected_rows: (source, {key: state}),
    )
    monkeypatch.setattr(runner.pilot, "_model_digest", lambda policy: "digest")
    monkeypatch.setattr(
        runner.neutral_eval,
        "_build_neutral_inputs",
        lambda: ({}, {"dense24": object()}, {}),
    )
    monkeypatch.setattr(
        runner,
        "_alignment_report",
        lambda *args, **kwargs: {"alignment": "fake", "delta_q": kwargs.get("delta_q")},
    )
    callback_seen: dict[str, object] = {}

    def fake_paired(*args, branch_callback=None, **kwargs):
        callback_seen["callback"] = branch_callback
        assert callable(branch_callback)
        for index in range(1, 27):
            branch_callback(
                {
                    "branch": "wait" if index == 1 else f"release_{index - 2}",
                    "steps": index,
                    "termination_classification": "computation_budget_censored",
                },
                index,
                26,
            )
        return {"branches": [{} for _ in range(26)], "qualified_delta_Q": None}

    monkeypatch.setattr(runner.paired_gate_diagnostic, "paired_gate_value", fake_paired)
    result = runner._run_pilot(tmp_path, {})

    assert result["status"] == "COMPLETED"
    assert callable(callback_seen["callback"])
    artifacts = sorted((tmp_path / "branches" / "state_0001").glob("branch_*.json"))
    assert len(artifacts) == 26
    assert json.loads(artifacts[-1].read_text())["steps"] == 26


def test_preflight_writes_phase_contract_without_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = {
        "stage": runner.STAGE,
        "phase": "preflight",
        "status": "PREFLIGHT_OK",
        "PPO_ALLOWED": False,
        "runtime": {"loaded_code_manifest": {"count": 0}},
    }
    monkeypatch.setattr(runner, "_build_preflight", lambda: fake)

    output = tmp_path / "followup"
    result = runner.run(output, "preflight")

    assert result is fake
    assert json.loads((output / "preflight.json").read_text()) == fake
    assert json.loads((output / "summary.json").read_text())["replay_calls"] == 0
    progress = json.loads((output / "progress.json").read_text())
    assert progress["status"] == "PREFLIGHT_OK"
    assert not (output / "selection.json").exists()


def test_failure_is_persisted_with_no_training_scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail() -> dict[str, object]:
        raise RuntimeError("synthetic preflight failure")

    monkeypatch.setattr(runner, "_build_preflight", fail)
    output = tmp_path / "failed"
    result = runner.run(output, "preflight")

    assert result["status"] == "FAILED"
    assert result["scope"]["training_called"] is False
    report = json.loads((output / "report.json").read_text())
    assert report["error"]["message"] == "synthetic preflight failure"
    assert json.loads((output / "progress.json").read_text())["status"] == "FAILED"
