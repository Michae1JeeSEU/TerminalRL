"""Contract tests for the replay-only P3 gate anchor collector."""
from __future__ import annotations

import contextlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from terminal_marl.training import collect_p3_gate_anchor_states_20261001 as runner


def test_binary_fields_keep_unknown_separate_from_false() -> None:
    assert runner._binary_code(True) == (1, True)
    assert runner._binary_code(False) == (0, False)
    assert runner._binary_code(None) == (-1, None)
    assert runner._binary_code("0") == (-1, None)
    assert runner._proposal_code(None) == -2
    assert runner._proposal_code(-1) == -1
    assert runner._proposal_code(24) == 24
    assert runner._proposal_code(25) == -2


def test_npz_contract_rejects_wrong_shape_and_nonfinite(tmp_path: Path) -> None:
    kwargs = {
        "steps": np.asarray([0], dtype=np.int64),
        "uids": np.asarray([2], dtype=np.int64),
        "arrivals": np.asarray([2], dtype=np.int64),
        "actor_mask": np.asarray([1], dtype=np.int8),
        "requested_gate": np.asarray([1], dtype=np.int8),
        "requested_proposal": np.asarray([3], dtype=np.int8),
        "metadata": {"schema": "test"},
    }
    with pytest.raises(runner.AnchorCollectionError, match="88"):
        runner._save_npz(tmp_path / "bad.npz", world=np.zeros((1, 87), dtype=np.float32), **kwargs)
    bad = np.zeros((1, 88), dtype=np.float32)
    bad[0, 0] = np.nan
    with pytest.raises(runner.AnchorCollectionError, match="non-finite"):
        runner._save_npz(tmp_path / "bad_nan.npz", world=bad, **kwargs)


def test_npz_roundtrip_has_explicit_row_fields_and_unknown_codes(tmp_path: Path) -> None:
    path = tmp_path / "anchors.npz"
    sha = runner._save_npz(
        path,
        world=np.zeros((2, 88), dtype=np.float32),
        steps=np.asarray([0, 1], dtype=np.int64),
        uids=np.asarray([10, 2], dtype=np.int64),
        arrivals=np.asarray([110, 102], dtype=np.int64),
        actor_mask=np.asarray([1, -1], dtype=np.int8),
        requested_gate=np.asarray([0, -1], dtype=np.int8),
        requested_proposal=np.asarray([-1, -2], dtype=np.int8),
        metadata={"schema": "p3_gate_anchor_states_v1", "rows": 2},
    )
    assert len(sha) == 64
    with np.load(path, allow_pickle=False) as data:
        assert data["world"].shape == (2, 88)
        assert data["world"].dtype == np.float32
        assert data["row_index"].tolist() == [0, 1]
        assert data["uid"].tolist() == [10, 2]
        assert data["actor_mask"].tolist() == [1, -1]
        assert data["requested_gate"].tolist() == [0, -1]
        assert data["requested_proposal"].tolist() == [-1, -2]
        assert json.loads(str(data["metadata_json"].item()))["schema"] == "p3_gate_anchor_states_v1"


class _Actor:
    def __init__(self) -> None:
        self.uid = 7
        self.arrival_id = 101
        self.phase = SimpleNamespace(value="ring_outer")
        self.ring_role = SimpleNamespace(value="outer")
        self.soc = 0.8
        self.safety_retry_after = 0.0
        self.committed_plan = None


class _ReplayEnv:
    instances: list["_ReplayEnv"] = []

    def __init__(self, config: object) -> None:
        self.config = config
        self.step_count = 0
        self.now = 0.0
        self.active = {7: _Actor()}
        self.future_release_reservations = set()
        self.original_step_calls = 0
        self.steps_seen: list[int] = []
        self.__class__.instances.append(self)

    def _observations(self):
        if self.step_count == 0:
            return {"7": np.zeros(88, dtype=np.float32)}
        return {}

    def step(self, actions=None, proposal_shortlists=None):
        self.original_step_calls += 1
        self.steps_seen.append(self.step_count)
        self.step_count += 1
        self.now += 1.0
        self.active = {} if self.step_count >= 1 else self.active
        return {}, {"7": 0.0}, {}, {}, {"release_decisions": {}}


def _fake_item() -> dict:
    observations = {"7": np.zeros(88, dtype=np.float32)}
    return {
        "cell_id": 1,
        "full_sha256": runner.SOURCE_FULL_SHA256,
        "arm_sha256": "arm",
        "tape_sha256": "tape",
        "sidecar_sha256": "sidecar",
        "tape_path": "fake-tape",
        "sidecar": {"registrations": {}},
        "rows": [
            {
                "step": 0,
                "observation_hash": runner._digest(observations),
                "requested_gate": {"7": 1},
                "requested_proposal": {"7": -1},
            },
            {"step": 1, "observation_hash": runner._digest({}),
             "requested_gate": {}, "requested_proposal": {}},
        ],
        "arm": {"safety_nonzero": None, "binding_checks": None, "lifecycle_valid": None},
    }


def test_collect_cell_keeps_all_actor_rows_and_empty_tick(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = {"enter": 0, "exit": 0}

    @contextlib.contextmanager
    def fake_controller(env, sidecar):
        calls["enter"] += 1
        try:
            yield object()
        finally:
            calls["exit"] += 1

    def fake_replay(env, path):
        for _ in range(2):
            env.step({}, {})
        return {"exact": True, "steps_expected": 2, "steps_replayed": 2,
                "config_match": True, "reward_match": True}

    monkeypatch.setattr(runner.frozen, "_CapturedTerminalEnv", _ReplayEnv)
    monkeypatch.setattr(runner.feasible, "_controller_scope", fake_controller)
    monkeypatch.setattr(runner.SafetyTraceRecorder, "replay_action_tape", staticmethod(fake_replay))
    _ReplayEnv.instances.clear()
    result = runner._collect_cell(_fake_item(), SimpleNamespace(), tmp_path)
    assert result["replay_exact"] is True
    assert result["steps_replayed"] == 2
    assert len(result["rows"]) == 1
    assert result["rows"][0]["uid"] == 7
    assert len(result["empty_world_ticks"]) == 1
    assert result["counts"]["requested_gate"] == {"true": 1, "false": 0, "unknown": 0}
    assert result["counts"]["actor_mask"] == {"true": 1, "false": 0, "unknown": 0}
    assert result["controller_installed_and_restored"] is True
    assert result["caller_rng_unchanged_after_restore"] is True
    assert calls == {"enter": 1, "exit": 1}
    assert _ReplayEnv.instances[0].original_step_calls == 2
    with np.load(tmp_path / "anchors" / "cell_01.npz", allow_pickle=False) as data:
        assert data["world"].shape == (1, 88)
        assert data["step"].tolist() == [0]
        assert data["uid"].tolist() == [7]


class _MultiUidEnv:
    instances: list["_MultiUidEnv"] = []

    def __init__(self, config: object) -> None:
        self.config = config
        self.step_count = 0
        self.now = 0.0
        self.active = {
            2: SimpleNamespace(
                uid=2, arrival_id=102,
                phase=SimpleNamespace(value="ring_outer"),
                ring_role=SimpleNamespace(value="outer"),
                soc=0.8, safety_retry_after=0.0, committed_plan=object(),
            ),
            10: SimpleNamespace(
                uid=10, arrival_id=110,
                phase=SimpleNamespace(value="ring_outer"),
                ring_role=SimpleNamespace(value="outer"),
                soc=0.7, safety_retry_after=0.0, committed_plan=None,
            ),
        }
        self.future_release_reservations = set()
        self.original_step_calls = 0
        self.__class__.instances.append(self)

    def _observations(self):
        if self.step_count == 0:
            return {
                "2": np.full(88, 2.0, dtype=np.float32),
                "10": np.full(88, 10.0, dtype=np.float32),
            }
        return {}

    def step(self, actions=None, proposal_shortlists=None):
        self.original_step_calls += 1
        self.step_count += 1
        self.now += 1.0
        self.active = {}
        return {}, {}, {}, {}, {"release_decisions": {}}


def _multi_uid_item() -> dict:
    observations = {
        "2": np.full(88, 2.0, dtype=np.float32),
        "10": np.full(88, 10.0, dtype=np.float32),
    }
    return {
        "cell_id": 1,
        "full_sha256": runner.SOURCE_FULL_SHA256,
        "arm_sha256": "arm-multi",
        "tape_sha256": "tape-multi",
        "sidecar_sha256": "sidecar-multi",
        "tape_path": "fake-tape-multi",
        "sidecar": {"registrations": {}},
        "rows": [
            {
                "step": 0,
                "observation_hash": runner._digest(observations),
                "requested_gate": {"2": 0},
                "requested_proposal": {"2": -1},
            },
            {"step": 1, "observation_hash": runner._digest({}),
             "requested_gate": {}, "requested_proposal": {}},
        ],
        "arm": {"safety_nonzero": None, "binding_checks": None, "lifecycle_valid": None},
    }


def test_collect_cell_preserves_world_uid_order_and_unknown_mask_gate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    @contextlib.contextmanager
    def fake_controller(env, sidecar):
        yield object()

    def fake_replay(env, path):
        env.step({}, {})
        env.step({}, {})
        return {"exact": True, "steps_expected": 2, "steps_replayed": 2,
                "config_match": True, "reward_match": True}

    monkeypatch.setattr(runner.frozen, "_CapturedTerminalEnv", _MultiUidEnv)
    monkeypatch.setattr(runner.feasible, "_controller_scope", fake_controller)
    monkeypatch.setattr(runner.SafetyTraceRecorder, "replay_action_tape", staticmethod(fake_replay))
    _MultiUidEnv.instances.clear()
    initial_observations = {
        "2": np.full(88, 2.0, dtype=np.float32),
        "10": np.full(88, 10.0, dtype=np.float32),
    }
    result = runner._collect_cell(_multi_uid_item(), SimpleNamespace(), tmp_path)
    expected_uids = [int(uid) for uid in runner.matched._world_from_observations(
        initial_observations
    )[0]]
    assert expected_uids == [10, 2]
    assert [row["uid"] for row in result["rows"]] == expected_uids
    assert result["counts"]["requested_gate"] == {"true": 0, "false": 1, "unknown": 1}
    assert result["counts"]["actor_mask"] == {"true": 1, "false": 1, "unknown": 0}
    assert result["caller_rng_unchanged_after_restore"] is True
    with np.load(tmp_path / "anchors" / "cell_01.npz", allow_pickle=False) as data:
        assert data["uid"].tolist() == expected_uids
        assert data["world"].shape == (2, 88)


class _DuplicateUidEnv(_MultiUidEnv):
    def _observations(self):
        if self.step_count == 0:
            return {
                "2": np.full(88, 2.0, dtype=np.float32),
                2: np.full(88, 20.0, dtype=np.float32),
            }
        return {}


def test_collect_cell_rejects_duplicate_uid_key_without_npz_overwrite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    @contextlib.contextmanager
    def fake_controller(env, sidecar):
        yield object()

    def fake_replay(env, path):
        env.step({}, {})
        return {"exact": True, "steps_expected": 1, "steps_replayed": 1,
                "config_match": True, "reward_match": True}

    duplicate_observations = {
        "2": np.full(88, 2.0, dtype=np.float32),
        2: np.full(88, 20.0, dtype=np.float32),
    }
    item = _multi_uid_item()
    item["rows"] = [{
        "step": 0,
        "observation_hash": runner._digest(duplicate_observations),
        "requested_gate": {}, "requested_proposal": {},
    }]
    monkeypatch.setattr(runner.frozen, "_CapturedTerminalEnv", _DuplicateUidEnv)
    monkeypatch.setattr(runner.feasible, "_controller_scope", fake_controller)
    monkeypatch.setattr(runner.SafetyTraceRecorder, "replay_action_tape", staticmethod(fake_replay))
    with pytest.raises(runner.AnchorCollectionError, match="duplicate anchor key"):
        runner._collect_cell(item, SimpleNamespace(), tmp_path)
    assert (tmp_path / "cell_01_partial.json").is_file()
    assert not (tmp_path / "anchors" / "cell_01.npz").exists()
    assert _DuplicateUidEnv.instances[-1].step.__func__ is _MultiUidEnv.step


def test_collect_cell_restores_step_and_rng_on_replay_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    @contextlib.contextmanager
    def fake_controller(env, sidecar):
        yield object()

    def failing_replay(env, path):
        env.step({}, {})
        raise RuntimeError("synthetic replay failure")

    monkeypatch.setattr(runner.frozen, "_CapturedTerminalEnv", _ReplayEnv)
    monkeypatch.setattr(runner.feasible, "_controller_scope", fake_controller)
    monkeypatch.setattr(runner.SafetyTraceRecorder, "replay_action_tape", staticmethod(failing_replay))
    with pytest.raises(RuntimeError, match="synthetic replay failure"):
        runner._collect_cell(_fake_item(), SimpleNamespace(), tmp_path)
    env = _ReplayEnv.instances[-1]
    assert env.step.__func__ is _ReplayEnv.step
    partial = json.loads((tmp_path / "cell_01_partial.json").read_text(encoding="utf-8"))
    assert partial["status"] == "PARTIAL"
    assert partial["rows_recorded"] == 1


def test_preflight_output_is_no_environment_and_strict_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "stage": runner.STAGE, "phase": "preflight", "status": "PREFLIGHT_OK",
        "environment_created": False, "model_forward_used": False,
        "gradient_operations": 0, "optimizer_operations": 0, "teacher_calls": 0,
        "PPO_ALLOWED": False, "heldout_used": False, "teacher_used": False,
        "optimizer_used": False, "off_policy": True, "no_training": True,
        "runner_sha256": "runner", "protocol_sha256": "protocol", "config": {},
        "input_bindings": {"source_full_sha256": runner.SOURCE_FULL_SHA256,
                            "event_full": {"sha256": runner.EVENT_FULL_SHA256},
                            "cells": [], "expected_steps_total": runner.TOTAL_STEPS},
        "source_bindings": {},
    }
    monkeypatch.setattr(runner, "_build_preflight", lambda: payload)
    result = runner._run_phase("preflight", tmp_path / "preflight")
    assert result["status"] == "PREFLIGHT_OK"
    stored = json.loads((tmp_path / "preflight" / "preflight.json").read_text(encoding="utf-8"))
    assert stored["environment_created"] is False
    assert stored["PPO_ALLOWED"] is False
    assert json.loads((tmp_path / "preflight" / "progress.json").read_text(encoding="utf-8"))["status"] == "PREFLIGHT_OK"


def test_pilot_reuse_requires_immutable_cell_hash(tmp_path: Path) -> None:
    output = tmp_path / "out"
    (output / "cells").mkdir(parents=True)
    (output / "anchors").mkdir(parents=True)
    (output / "preflight.json").write_text("{}\n", encoding="utf-8")
    npz_path = output / "anchors" / "cell_01.npz"
    npz_path.write_bytes(b"anchor-npz")
    cell_path = output / "cells" / "cell_01.json"
    cell_path.write_text(json.dumps({
        "status": "COMPLETED", "replay_exact": True,
        "anchors_npz": {"path": str(npz_path), "sha256": runner._sha256(npz_path)},
    }) + "\n", encoding="utf-8")
    pilot = {
        "status": "PILOT_OK", "cell_ids": [1],
        "preflight_sha256": runner._sha256(output / "preflight.json"),
        "cell_sha256": runner._sha256(cell_path),
    }
    (output / "pilot.json").write_text(json.dumps(pilot) + "\n", encoding="utf-8")
    assert runner._validate_pilot_reuse(output, {"status": "PREFLIGHT_OK"})["cell_ids"] == [1]
    cell_path.write_text(json.dumps({
        "status": "COMPLETED", "replay_exact": False,
        "anchors_npz": {"path": str(npz_path), "sha256": runner._sha256(npz_path)},
    }) + "\n", encoding="utf-8")
    with pytest.raises(runner.AnchorCollectionError):
        runner._validate_pilot_reuse(output, {"status": "PREFLIGHT_OK"})
