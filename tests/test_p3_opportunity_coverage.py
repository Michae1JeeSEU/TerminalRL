"""Contract tests for the passive P3 executable-opportunity diagnostic."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from terminal_marl.core.low_level_joint_action_v4_2 import HierarchicalActionV42, RELEASE_GATE, WAIT_GATE
from terminal_marl.core.config import SimConfig
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training import run_p3_opportunity_coverage_20260930 as runner
from terminal_marl.training import run_p3_task_neutral_pilot as canonical_digest
from terminal_marl.training import paired_gate_diagnostic as paired


def _actor(arrival: int, wait: float, *, phase: str = "ring_outer", role: str = "outer",
           soc: float = 0.7, retry: float = 0.0, committed=None):
    return SimpleNamespace(
        uid=int(arrival), arrival_id=int(arrival), internal_wait_time=float(wait),
        external_wait_time=0.0, phase=SimpleNamespace(value=phase),
        ring_role=SimpleNamespace(value=role), soc=float(soc), energy_remaining=700.0,
        energy_consumed=100.0, energy_capacity=1500.0,
        safety_retry_after=float(retry), emergency_retry_after=0.0,
        committed_plan=committed,
    )


def _env(*actors, now: float = 100.0, affected=()):
    return SimpleNamespace(
        active={int(item.uid): item for item in actors}, now=float(now), step_count=int(now),
        future_release_reservations={}, airspace=SimpleNamespace(affected_interfaces=lambda: set(affected)),
        service_uavs={}, completed_uavs=[], energy_outcomes=[], telemetry={}, conflict_events=[],
        admission=SimpleNamespace(queue=SimpleNamespace(items=[])),
        ledger=SimpleNamespace(requested=set(), outcomes={}), lz_manager=SimpleNamespace(zones=[]),
    )


def _context(uids, masks):
    count = len(uids)
    return {
        "uids": [str(uid) for uid in uids],
        "actor_mask": np.asarray(masks, dtype=bool),
        "world": np.zeros((count, 88), dtype=np.float32),
        "gate_logits": np.zeros((count, 2), dtype=np.float64),
        "proposal_logits": np.zeros((count, 25), dtype=np.float64),
        "gate_probabilities": np.tile(np.asarray([[0.6, 0.4]]), (count, 1)),
        "proposal_probabilities": np.tile(np.full((1, 25), 1.0 / 25.0), (count, 1)),
        "step": 100,
    }


def test_target_selection_uses_wait_tie_and_arrival13_override():
    env = _env(_actor(3, 4.0), _actor(5, 4.0), _actor(13, 1.0))
    context = _context([3, 5, 13], [True, True, True])
    uid, detail = runner._select_target(env, context, 877101)
    assert uid == 3
    assert detail["selection"]["arrival_id"] == 3

    uid, detail = runner._select_target(env, context, 877104)
    assert uid == 13
    assert detail["selection"]["arrival_id"] == 13


def test_eligibility_requires_legacy_mask_retry_and_native_geometry():
    env = _env(_actor(13, 2.0, retry=101.0), affected=("o2m",))
    context = _context([13], [True])
    row = runner._eligibility(env, context, 0, "13")
    assert row["eligible"] is False
    assert "safety_retry" in row["reasons"]
    assert "geometry_affected" in row["reasons"]
    env.now = 101.0
    env.airspace.affected_interfaces = lambda: set()
    row = runner._eligibility(env, context, 0, "13")
    assert row["eligible"] is True


def test_reference_layout_requires_all_registered_pairs(tmp_path: Path):
    root = tmp_path / "reference"
    (root / "arms").mkdir(parents=True)
    statuses = {str(index): {"legacy": "completed"} for index in range(1, 25)}
    (root / "full.json").write_text(json.dumps({
        "status": "FULL_OK", "pair_count": 24, "pair_ids": list(range(1, 25)),
        "pair_statuses": statuses,
    }), encoding="utf-8")
    for spec in runner.PAIR_SPECS:
        (root / "arms" / f"pair_{spec['pair_id']:04d}_legacy.json").write_text(json.dumps({
            "status": "completed", "censored": False,
            "env_seed": spec["env_seed"], "keyed_policy_seed": spec["keyed_policy_seed"],
        }), encoding="utf-8")
    layout = runner._reference_layout(root)
    assert layout["selected_pair_ids"] == [item["pair_id"] for item in runner.PAIR_SPECS]
    assert layout["arms"]["1"]["sha256"]


def test_shadow_branches_keep_true_q_mass_and_live_state(monkeypatch: pytest.MonkeyPatch):
    env = _env(_actor(13, 1.0))
    context = _context([13], [True])
    context["proposal_probabilities"][:] = 0.0
    context["proposal_probabilities"][0, 4] = 0.2
    context["proposal_probabilities"][0, 0] = 0.8
    context["gate_probabilities"][0] = [0.25, 0.75]
    request = {
        "actions": {"13": np.asarray([0.0, 0.0, 0.0], dtype=np.float32)},
        "shortlists": {},
    }
    policy = SimpleNamespace(model=nn.Linear(1, 1))

    def step(branch_env, actions, shortlists):
        proposal = None
        if shortlists:
            selected = np.asarray(shortlists["13"][0])
            proposal = next(
                index for index in range(25)
                if np.allclose(selected, HierarchicalActionV42(RELEASE_GATE, index).normalized)
            )
        decision = "approved" if proposal == 4 else "safety_reject"
        return {}, {13: 1.0 if proposal == 4 else 0.0}, {}, {}, {"release_decisions": {13: decision}}

    result = runner._shadow_branches(
        policy, env, context, request, 13, copy.deepcopy(env), step,
    )
    assert result["branch_count"] == 26
    assert result["approved_proposals"] == [4]
    assert result["C"] == pytest.approx(0.2)
    assert result["pC"] == pytest.approx(0.15)
    assert result["live_environment_unchanged"] is True
    assert all(item["status"] == "ok" for item in result["branches"])


def test_shadow_branch_error_makes_coverage_unknown():
    env = _env(_actor(13, 1.0))
    context = _context([13], [True])
    request = {"actions": {"13": np.zeros(3, dtype=np.float32)}, "shortlists": {}}
    policy = SimpleNamespace(model=nn.Linear(1, 1))

    def step(_env_value, actions, _shortlists):
        if np.asarray(actions["13"])[0] >= 0.5:
            raise RuntimeError("branch failure")
        return {}, {13: 0.0}, {}, {}, {"release_decisions": {13: None}}

    result = runner._shadow_branches(policy, env, context, request, 13, copy.deepcopy(env), step)
    assert result["C"] is None
    assert result["pC"] is None
    assert result["C_unknown_reason"]
    assert any(item["status"] == "error" for item in result["branches"])


def test_reference_compare_requires_terminal_rows_and_allows_reward_tolerance():
    reference = {
        "status": "completed", "request_identity": [{"arrival_id": 1}], "ticks": 5,
        "ledger": {"outcomes": {"1": "completed"}, "terminal_rows": [{"arrival_id": 1, "x": 2.0}]},
        "safety": {"safety_violation": False}, "model_unchanged": True,
        "caller_rng_unchanged": True, "wait_metrics": {"value": 2.0},
        "energy_ledger": {"value": 3.0}, "raw_reward_excluding_rejection_penalty": 1.0,
        "reward_total": 1.5, "release_rejection_penalty": -0.5,
    }
    observed = copy.deepcopy(reference)
    observed["raw_reward_excluding_rejection_penalty"] = 1.0 + 5e-10
    assert runner._compare_reference(observed, reference)["exact"] is True
    observed["ledger"]["terminal_rows"][0]["x"] = 3.0
    result = runner._compare_reference(observed, reference)
    assert result["exact"] is False
    assert "ledger_terminal_rows" in result["differences"]


def test_trace_records_target_probability_and_88d_world():
    env = _env(_actor(13, 1.0))
    context = _context([13], [True])
    request = {"actions": {"13": np.zeros(3, dtype=np.float32)}, "shortlists": {}}
    row = runner._trace_pre(env, context, request)
    assert row["target_arrival_id"] == 13
    assert row["p_release"] == pytest.approx(0.4)
    assert row["q_max"] == pytest.approx(1.0 / 25.0)
    assert len(row["target"]["resources"]["landing_zones"]) == 0


def test_manifest_filter_ignores_only_new_runner():
    manifest = {"files": [
        {"path": "/home/user/projects/terminal/terminal_marl/core/a.py", "sha256": "a", "module": "a"},
        {"path": "/home/user/projects/terminal/terminal_marl/training/run_p3_opportunity_coverage_20260930.py", "sha256": "new", "module": "new"},
    ]}
    result = runner._manifest_by_relative(manifest)
    assert "core/a.py" in result
    assert all("opportunity_coverage" not in key for key in result)
    assert result == runner._manifest_by_relative({
        "files": [{"path": "/home/user/projects/terminal/terminal_marl/core/a.py",
                    "sha256": "a", "module": "__main__"}],
    })


def test_real_terminal_env_runs_all_native_first_step_shadow_branches():
    config = SimConfig(
        action_schema_version=runner.frozen.ACTION_SCHEMA,
        observation_schema_version=runner.frozen.OBSERVATION_SCHEMA,
        runtime_proposal_shortlist_k=3,
        arrival_model="finite_cohort",
        total_requested_uavs=1,
        fixed_arrival_schedule=(0.0,),
        reset_mode="warm_start_rings",
        warm_start_uavs=1,
        initial_soc_mode="fixed",
        initial_soc=0.8,
        max_simulation_seconds=3.0,
    )
    env = TerminalParallelEnv(config)
    env.reset(877101)
    uid = next(iter(env.active))
    context = _context([str(uid)], [True])
    request = {
        "actions": {str(uid): HierarchicalActionV42(WAIT_GATE, None).normalized.copy()},
        "shortlists": {},
    }
    policy = SimpleNamespace(model=nn.Linear(1, 1))
    result = runner._shadow_branches(
        policy, env, context, request, int(uid), copy.deepcopy(env), paired._call_step,
    )
    assert result["branch_count"] == 26
    assert len(result["branches"]) == 26
    assert result["live_environment_unchanged"] is True
    assert all(item["model_unchanged"] for item in result["branches"])
    assert all(item["caller_rng_unchanged"] for item in result["branches"])
    assert all(item["status"] == "ok" for item in result["branches"])
    assert result["C"] is not None
    assert 0.0 <= result["C"] <= 1.0
    assert result["pC"] is not None
    assert 0.0 <= result["pC"] <= 1.0


def test_run_opportunity_arm_preserves_censored_status_and_writes_states(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(runner, "WINDOWS", ((0, 0), (1, 1), (2, 2)))
    env = _env(_actor(13, 1.0), now=0.0)
    policy = SimpleNamespace(model=nn.Linear(1, 1))
    calls = []

    def details(_policy, live_env):
        context = _context(["13"], [True])
        context["step"] = int(live_env.step_count)
        return context

    def keyed(_policy, live_env, keyed_seed):
        calls.append(int(live_env.step_count))
        return {
            "actions": {"13": HierarchicalActionV42(WAIT_GATE, None).normalized.copy()},
            "shortlists": {}, "rows": [], "step": int(live_env.step_count),
            "keyed_seed": int(keyed_seed), "keying": "arrival_id|absolute_tick|stream",
        }

    def step(_live_env, _actions, shortlists):
        decision = "approved" if shortlists else "not_requested"
        return {}, {13: 0.0}, {}, {}, {"release_decisions": {13: decision}}

    def fake_run_arm(_policy, _config, *, progress=None, **_kwargs):
        for _ in range(3):
            request = runner.frozen._keyed_request(policy, env, 9303101)
            runner.frozen.paired._call_step(env, request["actions"], request["shortlists"])
            env.step_count += 1
        return {"status": "censored", "censored": True, "model_unchanged": True,
                "caller_rng_unchanged": True}

    monkeypatch.setattr(runner.paired, "_details_context", details)
    monkeypatch.setattr(runner.frozen, "_keyed_request", keyed)
    monkeypatch.setattr(runner.frozen.paired, "_call_step", step)
    monkeypatch.setattr(runner.frozen, "_run_arm", fake_run_arm)
    reference = {"arm": {"status": "completed"}}
    result = runner._run_opportunity_arm(
        {"policy": policy}, SimpleNamespace(), runner.PAIR_SPECS[0], reference,
        output=tmp_path,
    )
    assert calls == [0, 1, 2]
    assert result["status"] == "censored"
    assert result["reference_exact"]["exact"] is None
    states = sorted((tmp_path / "states").glob("*.json"))
    assert len(states) == 3
    assert all(json.loads(path.read_text(encoding="utf-8"))["branch_count"] == 26 for path in states)


def test_pilot_reuse_rejects_pair_config_source_and_preflight_tampering(tmp_path: Path):
    config = SimpleNamespace(
        observation_schema_version="v2.1",
        action_schema_version="v4.2",
        runtime_proposal_shortlist_k=3,
        student_decision_mask_mode="legacy",
    )
    policy = SimpleNamespace(model=nn.Linear(2, 2))
    execution_digest = paired._model_digest(policy)
    canonical_model_digest = canonical_digest._model_digest(policy)
    assert execution_digest != canonical_model_digest
    source = {"policy": policy, "source_model_digest": canonical_model_digest}
    preflight = {
        "reference": {"full_sha256": "reference-sha"},
        "source_binding": {"source_model_digest": canonical_model_digest},
    }
    expected = dict(runner.PAIR_SPECS[0])
    config_snapshot = runner._config_snapshot(config)
    reference_exact = {"exact": True}
    pair = {
        "pair": expected,
        "status": "PAIR_OK",
        "selected_state_count": 0,
        "reference_exact": reference_exact,
        "source_model_digest": execution_digest,
        "config_snapshot": config_snapshot,
    }
    arm = {
        "status": "completed",
        "censored": False,
        "model_unchanged": True,
        "caller_rng_unchanged": True,
        "reference_exact": reference_exact,
    }
    output = tmp_path
    (output / "pairs").mkdir()
    (output / "arms").mkdir()
    (output / "pairs" / "pair_0001.json").write_text(json.dumps(pair), encoding="utf-8")
    (output / "arms" / "pair_0001_legacy.json").write_text(json.dumps(arm), encoding="utf-8")
    pilot = {
        "status": "PILOT_OK",
        "preflight_reference_sha256": preflight["reference"]["full_sha256"],
        "pair": pair,
    }

    # A V1 pair predates the explicit algorithm labels and remains reusable.
    runner._validate_pilot_reuse(pilot, preflight, config, source, output)

    pair.update(runner._digest_metadata(source))
    (output / "pairs" / "pair_0001.json").write_text(json.dumps(pair), encoding="utf-8")
    runner._validate_pilot_reuse(pilot, preflight, config, source, output)

    tampered = copy.deepcopy(pilot)
    tampered["pair"]["pair"]["pair_id"] = 5
    with pytest.raises(runner.OpportunityProtocolError):
        runner._validate_pilot_reuse(tampered, preflight, config, source, output)

    config_bad = copy.copy(config)
    config_bad.student_decision_mask_mode = "pre_action_forced_wait"
    with pytest.raises(runner.OpportunityProtocolError):
        runner._validate_pilot_reuse(pilot, preflight, config_bad, source, output)

    source_bad = {"source_model_digest": "different-source"}
    with pytest.raises(runner.OpportunityProtocolError):
        runner._validate_pilot_reuse(pilot, preflight, config, {**source, **source_bad}, output)

    weight_bad = copy.deepcopy(policy)
    with torch.no_grad():
        weight_bad.model.weight[0, 0].add_(1.0)
    with pytest.raises(runner.OpportunityProtocolError):
        runner._validate_pilot_reuse(
            pilot, preflight, config,
            {"policy": weight_bad, "source_model_digest": canonical_model_digest},
            output,
        )

    preflight_bad = {"reference": {"full_sha256": "different-reference"}}
    with pytest.raises(runner.OpportunityProtocolError):
        runner._validate_pilot_reuse(pilot, preflight_bad, config, source, output)
