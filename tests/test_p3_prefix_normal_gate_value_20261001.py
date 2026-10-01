import json
import copy
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from terminal_marl.core.merge_scheduler import LowLevelIntent
from terminal_marl.training import diagnose_p3_prefix_normal_gate_value_20261001 as runner


def _event(step, uid, arrival, soc, *, delta=999.0, status="NORMAL_ELIGIBLE"):
    return {
        "registration_key": f"{step}|{uid}",
        "step": step,
        "uid": uid,
        "status": status,
        "context": {"arrival_id": arrival, "soc": soc, "phase": "ring_inner"},
        "comparison": {"D": None, "local_m": None, "delta_p": delta},
    }


def test_selection_uses_fixed_event_contract_not_p_or_outcome():
    group = {"replay": {"events": [
        _event(308, 99, 15, 0.8, delta=-999.0),
        _event(307, 15, 15, runner.STATE_LOCKS[1]["soc"], delta=123.0),
    ]}}
    selected = runner._select_event(group, 1)
    assert selected["registration_key"] == "307|15"


def test_stable_order_is_sampled_first_then_logit_rank():
    logits = np.asarray([0.0, 4.0, 4.0, -1.0] + [0.0] * 21, dtype=float)
    assert runner._stable_order(3, logits)[:5] == [3, 1, 2, 0, 4]
    metadata = runner._metadata_for_order(3, logits)
    assert metadata["sampled_proposal"] == 3
    assert metadata["requested_k"] == 3
    assert len(metadata["source_logits"]) == 25


def test_branch_hooks_restore_both_paired_functions_on_exception():
    original_step = runner.paired._call_step
    original_request = runner.paired.keyed_request
    original_details = runner.paired._details_context

    def old_step(*args, **kwargs):
        return "old-step"

    def old_request(*args, **kwargs):
        return "old-request"

    runner.paired._call_step = old_step
    runner.paired.keyed_request = old_request
    runner.paired._details_context = lambda *_args, **_kwargs: {"old": True}
    old_details = runner.paired._details_context
    try:
        with runner._branch_hooks([0.0] * 25, 15):
            assert runner.paired._call_step is not old_step
            assert runner.paired.keyed_request is not old_request
            assert runner.paired._details_context is not old_details
            raise RuntimeError("synthetic branch failure")
    except RuntimeError:
        pass
    finally:
        runner.paired._call_step = original_step
        runner.paired.keyed_request = original_request
        runner.paired._details_context = original_details
    assert runner.paired._call_step is original_step
    assert runner.paired.keyed_request is original_request
    assert runner.paired._details_context is original_details


def test_preflight_serialization_and_validation_without_environment_or_forward(monkeypatch, tmp_path):
    class Config:
        pass

    monkeypatch.setattr(runner.support, "_build_inputs", lambda: (Config(), {}))
    monkeypatch.setattr(runner, "_config_snapshot", lambda config: {"fake": True})
    monkeypatch.setattr(runner, "_source_bindings", lambda: {
        "before": {"sha256": runner.BEFORE_SHA256},
        "after": {"sha256": runner.AFTER_SHA256},
    })
    cells = []
    for cell_id, lock in runner.STATE_LOCKS.items():
        cells.append({
            "cell_id": cell_id,
            "group_sha256": f"group-{cell_id}",
            "arm_sha256": f"arm-{cell_id}",
            "tape_sha256": f"tape-{cell_id}",
            "sidecar_sha256": f"sidecar-{cell_id}",
            "selected_event": {
                "registration_key": lock["key"], "step": lock["step"], "uid": lock["uid"],
                "context": {"arrival_id": lock["arrival_id"], "soc": lock["soc"]},
            },
        })
    monkeypatch.setattr(runner, "_load_inputs", lambda: ({
        "event_full_sha256": runner.EVENT_FULL_SHA256,
        "fresh_full_sha256": runner.FRESH_FULL_SHA256,
    }, cells))
    payload = runner._build_preflight()
    assert payload["environment_created"] is False
    assert payload["model_forward_used"] is False
    assert payload["branches_per_seed"] == 26
    assert payload["branches_per_state"] == 104
    assert payload["total_branches"] == 416
    assert payload["pilot_branches"] == 104
    assert payload["full_new_branches"] == 312
    path = tmp_path / "preflight.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    runner._validate_preflight(json.loads(path.read_text(encoding="utf-8")))
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["branches_per_state"] = 26
    with pytest.raises(runner.NormalGateValueError, match="branch budget changed"):
        runner._validate_preflight(tampered)


def test_controller_scope_restores_instance_alias_after_error():
    class Config:
        release_threshold = 0.5

    class UAV:
        uid = 15

    class Env:
        config = Config()
        now = 307.0
        step_count = 307

        def _runtime_shortlist_release(self, uav, intent, shortlist):
            return "native"

    env = Env()
    native = env._runtime_shortlist_release
    logits = [float(index) for index in range(25)]
    order = runner._stable_order(2, logits)
    metadata = runner._metadata_for_order(2, logits)
    action = runner.HierarchicalActionV42(1, 2)
    intent = LowLevelIntent.from_array(action.normalized)
    shortlist = tuple(
        LowLevelIntent.from_array(item)
        for item in runner.paired._shortlist_from_logits(2, logits)
    )
    try:
        with runner._controller_scope(env, {15: order}, {15: metadata}):
            result = env._runtime_shortlist_release(UAV(), intent, shortlist)
            assert result == "native"
            raise RuntimeError("synthetic controller failure")
    except RuntimeError:
        pass
    assert env._runtime_shortlist_release.__func__ is native.__func__


def test_snapshot_rebind_cleans_step_and_controller_closures_for_independent_branches():
    class Env:
        def __init__(self, name):
            self.name = name

        def step(self):
            return self.name

        def _runtime_shortlist_release(self):
            return self.name

    original = Env("original")
    snapshot = copy.deepcopy(original)
    snapshot.step = lambda: original.name
    snapshot._runtime_shortlist_release = lambda: original.name
    runner._rebind_native_snapshot_methods(snapshot)
    branch_a = copy.deepcopy(snapshot)
    branch_b = copy.deepcopy(snapshot)
    runner._rebind_native_snapshot_methods(branch_a)
    runner._rebind_native_snapshot_methods(branch_b)
    branch_a.name = "branch-a"
    branch_b.name = "branch-b"
    assert original.step() == "original"
    assert original._runtime_shortlist_release() == "original"
    assert snapshot.step() == "original"
    assert snapshot._runtime_shortlist_release() == "original"
    assert branch_a.step() == "branch-a"
    assert branch_b._runtime_shortlist_release() == "branch-b"


def test_details_context_once_shares_one_forward_and_restores_alias_on_tensor_error():
    class Model:
        def __init__(self):
            self.calls = 0

        def distributions(self, tensor):
            self.calls += 1
            return ("gate", "proposal", tensor.clone())

    class Policy:
        def __init__(self):
            self.model = Model()

    policy = Policy()

    def same_details(current_policy, _env):
        model = current_policy.model
        model.distributions(torch.zeros(2))
        model.distributions(torch.zeros(2))
        return {"ok": True}

    assert runner._details_context_once(policy, object(), details_fn=same_details) == {"ok": True}
    assert policy.model.calls == 1
    assert "distributions" not in policy.model.__dict__

    def changed_details(current_policy, _env):
        model = current_policy.model
        model.distributions(torch.zeros(2))
        model.distributions(torch.ones(2))
        return {"unreachable": True}

    with pytest.raises(runner.NormalGateValueError, match="different distribution tensor"):
        runner._details_context_once(policy, object(), details_fn=changed_details)
    assert "distributions" not in policy.model.__dict__


def test_branch_hooks_preserve_native_singleton_shortlist():
    class Config:
        release_threshold = 0.5

    class Env:
        config = Config()

        def _runtime_shortlist_release(self, _uav, _intent, _shortlist):
            return "native"

    env = Env()
    original_step = runner.paired._call_step
    seen = {}

    def capture_step(_env, actions, shortlists):
        seen["actions"] = actions
        seen["shortlists"] = shortlists
        return "step"

    runner.paired._call_step = capture_step
    try:
        action = runner.HierarchicalActionV42(runner.paired.WAIT_GATE, None).normalized.copy()
        singleton = tuple(
            LowLevelIntent.from_array(item)
            for item in runner.paired._shortlist_from_logits(2, [float(i) for i in range(25)])[:1]
        )
        with runner._branch_hooks([float(i) for i in range(25)], 15):
            assert runner.paired._call_step(env, {"15": action, "99": action}, {"99": singleton}) == "step"
    finally:
        runner.paired._call_step = original_step
    assert len(seen["shortlists"]["99"]) == 1


def test_public_prefix_has_three_entries_and_full_order_has_twenty_five():
    logits = np.linspace(-1.0, 1.0, 25)
    shortlist = runner.paired._shortlist_from_logits(7, logits)
    assert len(shortlist) == 3
    assert len(runner._stable_order(7, logits)) == 25


def test_full_reuses_immutable_cell_result_without_pilot_wrapper(monkeypatch, tmp_path):
    (tmp_path / "preflight.json").write_text("{}", encoding="utf-8")
    (tmp_path / "pilot.json").write_text("{}", encoding="utf-8")
    (tmp_path / "cell_01.json").write_text(
        json.dumps({"cell_id": 1, "status": "COMPLETED", "D": None}), encoding="utf-8"
    )
    monkeypatch.setattr(runner, "_validate_preflight", lambda _payload: None)
    monkeypatch.setattr(runner, "_validate_pilot_reuse", lambda *_args: {})
    monkeypatch.setattr(runner.support, "_build_inputs", lambda: (object(), {}))
    monkeypatch.setattr(
        runner,
        "_load_inputs",
        lambda: ({}, [{"cell_id": index} for index in runner.CELLS]),
    )
    monkeypatch.setattr(
        runner,
        "_run_state",
        lambda item, *_args: {"cell_id": int(item["cell_id"]), "status": "COMPLETED"},
    )
    result = runner.run_full(tmp_path)
    assert [row["cell_id"] for row in result["cells"]] == [1, 2, 3, 4]
    assert result["cells"][0]["status"] == "COMPLETED"


def test_pilot_reuse_rejects_changed_immutable_cell_file(tmp_path):
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(json.dumps({"cells": []}), encoding="utf-8")
    cell_path = tmp_path / "cell_01.json"
    cell = {"cell_id": 1, "status": "COMPLETED", "D": None}
    cell_path.write_text(json.dumps(cell, sort_keys=True), encoding="utf-8")
    pilot = {
        "stage": runner.STAGE,
        "phase": "pilot",
        "status": "PILOT_OK",
        "preflight_sha256": runner._sha256(preflight_path),
        "preflight_payload": {"cells": []},
        "cell": cell,
        "cell_01_sha256": runner._sha256(cell_path),
    }
    (tmp_path / "pilot.json").write_text(json.dumps(pilot, sort_keys=True), encoding="utf-8")
    runner._validate_pilot_reuse(tmp_path, {"cells": []})
    cell_path.write_text(json.dumps({**cell, "D": 1.0}, sort_keys=True), encoding="utf-8")
    with pytest.raises(runner.NormalGateValueError, match="pilot cell artifact hash changed"):
        runner._validate_pilot_reuse(tmp_path, {"cells": []})


def test_branch_progress_records_actual_phase_seed_and_proposal(tmp_path):
    callback = runner._branch_callback(tmp_path, 1, 307, time.perf_counter(), "pilot")
    callback({"terminal": True}, 1, runner.PILOT_BRANCHES)
    payload = json.loads((tmp_path / "branches" / "cell_01_001.json").read_text(encoding="utf-8"))
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert payload["phase"] == "pilot"
    assert payload["continuation_seed"] == runner.CONTINUATION_SEEDS[0]
    assert payload["proposal"] is None
    assert progress["phase"] == "pilot"
    assert progress["continuation_seed"] == runner.CONTINUATION_SEEDS[0]
    assert progress["proposal"] is None
