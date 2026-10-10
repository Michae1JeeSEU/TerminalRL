from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from terminal_marl.training import diagnose_p3_ppo_credit_signal as credit
from terminal_marl.training import probe_p3_task_neutral_critic_context as probe


class _TinyCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = nn.Parameter(torch.tensor([0.25]))
        self.critic_encoder = nn.Linear(88, 4)
        self.critic_head = nn.Sequential(nn.Linear(8, 4), nn.Tanh(), nn.Linear(4, 1))
        torch.manual_seed(17)
        for parameter in self.parameters():
            if parameter is not self.actor:
                nn.init.uniform_(parameter, -0.05, 0.05)

    def critic(self, agent_sets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        encoded = self.critic_encoder(agent_sets)
        valid = mask.unsqueeze(-1)
        mean = (encoded * valid).sum(1) / valid.sum(1).clamp_min(1)
        maximum = encoded.masked_fill(~valid, float("-inf")).max(1).values
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        return self.critic_head(torch.cat((mean, maximum), -1)).squeeze(-1)


def _context_env() -> SimpleNamespace:
    return SimpleNamespace(
        now=90.0,
        config=SimpleNamespace(max_simulation_seconds=1800.0, total_requested_uavs=24),
        active={1: object(), 2: object()},
        admission=SimpleNamespace(queue=SimpleNamespace(items=[
            SimpleNamespace(soc=0.4, external_wait_time=30.0),
            SimpleNamespace(soc=0.8, external_wait_time=60.0),
        ])),
        service_uavs={3: object()},
        arrivals=SimpleNamespace(generated=7),
        lz_manager=SimpleNamespace(zones=[
            SimpleNamespace(service_remaining=60.0),
            SimpleNamespace(service_remaining=0.0),
        ]),
    )


def test_context_schema_reads_only_pre_step_state_and_empty_aggregates() -> None:
    values = probe.critic_context(_context_env())
    assert tuple(values) == probe.CONTEXT_FEATURE_NAMES
    assert values["sim_time_norm"] == pytest.approx(0.05)
    assert values["active_uav_count_norm"] == pytest.approx(2 / 24)
    assert values["admission_queue_count_norm"] == pytest.approx(2 / 24)
    assert values["service_uav_count_norm"] == pytest.approx(1 / 24)
    assert values["remaining_arrivals_norm"] == pytest.approx(17 / 24)
    assert values["queue_soc_mean"] == pytest.approx(0.6)
    assert values["queue_soc_min"] == pytest.approx(0.4)
    assert values["queue_external_wait_norm"] == pytest.approx(90 / (24 * 300))
    assert values["lz_service_remaining_sum_norm"] == pytest.approx(60 / (2 * 120))
    assert values["lz_service_remaining_max_norm"] == pytest.approx(0.5)

    empty = _context_env()
    empty.admission.queue.items = []
    empty.lz_manager.zones = []
    observed = probe.critic_context(empty)
    assert observed["queue_soc_mean"] == 0.0
    assert observed["queue_soc_min"] == 0.0
    assert observed["queue_external_wait_norm"] == 0.0
    assert observed["lz_service_remaining_sum_norm"] == 0.0
    assert observed["lz_service_remaining_max_norm"] == 0.0


def test_context_missing_required_field_and_nonfinite_are_rejected() -> None:
    missing = _context_env()
    missing.admission.queue.items[0] = SimpleNamespace(soc=0.4)
    with pytest.raises(probe.CriticContextProbeError, match="external_wait_time"):
        probe.critic_context(missing)
    nonfinite = _context_env()
    nonfinite.now = float("nan")
    with pytest.raises(probe.CriticContextProbeError, match="env.now"):
        probe.critic_context(nonfinite)


def test_expanded_head_zero_columns_preserve_initial_prediction_and_add_context() -> None:
    torch.manual_seed(3)
    base = _TinyCritic()
    expanded = probe.ContextExpandedCritic(base)
    worlds = [np.ones((2, 88), dtype=np.float32), np.zeros((0, 88), dtype=np.float32)]
    contexts = [np.zeros(probe.CONTEXT_DIM, dtype=np.float32), np.zeros(probe.CONTEXT_DIM, dtype=np.float32)]
    legacy = probe._legacy_predictions(base, worlds)
    initial = probe._context_predictions(expanded, worlds, contexts)
    np.testing.assert_allclose(initial, legacy, rtol=1e-6, atol=1e-6)
    first = next(module for module in expanded.critic_head.modules() if isinstance(module, nn.Linear))
    with torch.no_grad():
        first.weight[:, -probe.CONTEXT_DIM:] = 0.1
    changed = probe._context_predictions(
        expanded,
        [worlds[0], worlds[0]],
        [np.zeros(probe.CONTEXT_DIM, dtype=np.float32), np.ones(probe.CONTEXT_DIM, dtype=np.float32)],
    )
    assert changed[0] != pytest.approx(changed[1])


def test_metrics_report_normalized_and_raw_units() -> None:
    result = probe._metrics(np.asarray([0.01]), np.asarray([0.0]), reward_scale=0.01)
    assert result["mse"] == pytest.approx(0.0001)
    assert result["mae"] == pytest.approx(0.01)
    assert result["raw_mse"] == pytest.approx(1.0)
    assert result["raw_mae"] == pytest.approx(1.0)


def _tiny_episode(index: int, target: float) -> dict:
    return {
        "index": index, "split": "fit", "seed": index, "policy_seed": index,
        "worlds": [np.zeros((1, 88), dtype=np.float32)],
        "contexts": [np.zeros(probe.CONTEXT_DIM, dtype=np.float32)],
        "targets": np.asarray([target], dtype=np.float64),
        "actor_world_mask": np.asarray([True], dtype=bool),
    }


def test_fit_updates_critic_only_and_split_aggregation_has_no_leakage() -> None:
    torch.manual_seed(7)
    base = _TinyCritic()
    expanded = probe.ContextExpandedCritic(base)
    actor_before = base.actor.detach().clone()
    episodes = [_tiny_episode(1, 0.2), _tiny_episode(2, -0.2)]
    fit = probe._fit_models(base, expanded, episodes)
    assert fit["shared_permutations"] is True
    assert fit["initial_prediction_max_abs_diff"] <= 1e-6
    assert torch.equal(base.actor.detach(), actor_before)

    cache = {
        "episode_01": {
            "legacy_prediction": np.asarray([0.1]),
            "context_expanded_prediction": np.asarray([0.2]),
        },
        "episode_02": {
            "legacy_prediction": np.asarray([0.3]),
            "context_expanded_prediction": np.asarray([0.4]),
        },
    }
    aggregate = probe._aggregate_prediction_cache(cache, episodes[:1])
    assert aggregate["steps"] == 1
    assert aggregate["legacy"]["all_worlds"]["count"] == 1
    assert aggregate["context_expanded"]["all_worlds"]["count"] == 1
    assert aggregate["legacy"]["all_worlds"]["mae"] == pytest.approx(0.1)


def test_save_critic_is_diagnostic_only_and_excludes_actor(tmp_path: Path) -> None:
    base = _TinyCritic()
    path = tmp_path / "critic.pt"
    probe._save_critic(path, base, kind="legacy_mean_max")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["diagnostic_only"] is True
    assert all(str(name).startswith(probe.CRITIC_PREFIXES) for name in payload["state_dict"])
    assert "actor" not in payload["state_dict"]


def test_preflight_accepts_real_record_warmup_checkpoint_and_config_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.pt"
    source.write_bytes(b"source")
    warmup = tmp_path / "warmup.pt"
    warmup.write_bytes(b"warmup")
    tape = tmp_path / "tape.json"
    tape.write_text(json.dumps({"action_tape": [{"step": 0, "a_requested": {}}]}), encoding="utf-8")
    config = SimpleNamespace(total_requested_uavs=24)
    snapshot = {"profile": "dense24", "task_priority_enabled": False}
    records: list[Path] = []
    warmup_sha = probe._sha256(warmup)
    for index in range(1, 25):
        record_path = tmp_path / f"record-{index:02d}.json"
        record_path.write_text(json.dumps({
            "scenario": "dense24", "seed": 879100 + index,
            "policy_seed": 879200 + index, "requests": 24,
            "task_priority_enabled": False, "technical_ok": True, "error": None,
            "checkpoint_sha256": warmup_sha, "config": snapshot,
            "action_tape": str(tape),
        }), encoding="utf-8")
        records.append(record_path)
    monkeypatch.setattr(probe, "WARMUP_CHECKPOINT", warmup)
    monkeypatch.setattr(probe, "WARMUP_CHECKPOINT_SHA256", warmup_sha)
    monkeypatch.setattr(probe, "EXPECTED_TOTAL_STEPS", 24)
    monkeypatch.setattr(probe, "_synced_path", lambda path: Path(path))
    monkeypatch.setattr(probe, "_episode_path", lambda index: records[index - 1])
    monkeypatch.setattr(
        probe.neutral_eval,
        "_checkpoint_from_report",
        lambda checkpoint, expected_sha: (source, {"source": "fixture"}),
    )
    monkeypatch.setattr(
        probe.neutral_eval,
        "_build_neutral_inputs",
        lambda: ({}, {"dense24": config}, None),
    )
    monkeypatch.setattr(probe.neutral_eval, "_config_snapshot", lambda value: snapshot)
    result = probe._preflight()
    assert result["total_steps"] == 24
    assert len(result["cells"]) == 24
    assert result["cells"][0]["record"]["checkpoint_sha256"] == warmup_sha


def test_check_only_has_zero_environment_replay_and_optimizer_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = SimpleNamespace(model=_TinyCritic(), device=torch.device("cpu"))
    inputs = {
        "checkpoint": tmp_path / "candidate.pt",
        "warmup": tmp_path / "warmup.pt",
        "config": SimpleNamespace(),
        "total_steps": probe.EXPECTED_TOTAL_STEPS,
    }
    monkeypatch.setattr(probe, "_preflight", lambda: inputs)
    monkeypatch.setattr(probe.rank_eval, "_load_policy", lambda checkpoint, config: (policy, {}))
    monkeypatch.setattr(
        probe.pilot,
        "_load_initial_warmup",
        lambda *args, **kwargs: (inputs["warmup"], {"warmup_gamma": 1.0, "warmup_reward_scale": 0.01}),
    )
    report = probe.run(tmp_path / "check_only", check_only=True)
    assert report["status"] == "CHECK_ONLY_OK"
    assert report["scope"]["env_resets"] == 0
    assert report["scope"]["replay_calls"] == 0
    assert report["scope"]["optimizer_used"] is False
    assert report["scope"]["ppo_updates"] is False
    assert report["scope"]["policy_sampling"] is False


def test_fixture_runtime_writes_report_cache_and_critic_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = SimpleNamespace(model=_TinyCritic(), device=torch.device("cpu"))
    actor_before = policy.model.actor.detach().clone()
    world_a = np.zeros((1, 88), dtype=np.float32)
    world_b = np.ones((1, 88), dtype=np.float32)
    world_c = np.full((1, 88), 0.5, dtype=np.float32)
    trajectories = [
        {
            "index": 1, "split": "fit", "seed": 879101, "policy_seed": 879201,
            "worlds": [world_a, world_b],
            "contexts": [np.zeros(probe.CONTEXT_DIM, dtype=np.float32)] * 2,
            "targets": np.asarray([0.1, 0.2], dtype=np.float64),
            "actor_world_mask": np.asarray([True, False], dtype=bool),
        },
        {
            "index": 2, "split": "validation", "seed": 879102, "policy_seed": 879202,
            "worlds": [world_c],
            "contexts": [np.zeros(probe.CONTEXT_DIM, dtype=np.float32)],
            "targets": np.asarray([0.3], dtype=np.float64),
            "actor_world_mask": np.asarray([True], dtype=bool),
        },
    ]
    trajectory_reports = [
        {
            "index": 1, "replay_exact": True, "terminal": True,
            "ledger_alignment": True, "safety": {"required_counters_zero": True},
        },
        {
            "index": 2, "replay_exact": True, "terminal": True,
            "ledger_alignment": True, "safety": {"required_counters_zero": True},
        },
    ]
    inputs = {
        "checkpoint": tmp_path / "candidate.pt",
        "warmup": tmp_path / "warmup.pt",
        "config": SimpleNamespace(),
        "total_steps": 3,
        "cells": [{"index": 1}, {"index": 2}],
    }
    monkeypatch.setattr(probe, "EXPECTED_EPISODES", 2)
    monkeypatch.setattr(probe, "FIT_EPISODES", 1)
    monkeypatch.setattr(probe, "VALIDATION_EPISODES", 1)
    monkeypatch.setattr(probe, "EXPECTED_REQUESTS", 24)
    monkeypatch.setattr(probe, "EXPECTED_TOTAL_STEPS", 3)
    monkeypatch.setattr(probe, "EPOCHS", 2)
    monkeypatch.setattr(probe, "_preflight", lambda: inputs)
    monkeypatch.setattr(probe.rank_eval, "_load_policy", lambda checkpoint, config: (policy, {}))
    monkeypatch.setattr(
        probe.pilot,
        "_load_initial_warmup",
        lambda *args, **kwargs: (inputs["warmup"], {"warmup_gamma": 1.0, "warmup_reward_scale": 0.01}),
    )
    monkeypatch.setattr(
        probe,
        "_capture_trajectories",
        lambda inputs, policy, progress_callback=None: (trajectories, trajectory_reports),
    )

    output = tmp_path / "runtime"
    report = probe.run(output, check_only=False)

    assert report["status"] == "COMPLETED"
    assert report["scope"]["replay_calls"] == 2
    assert report["scope"]["fit_episodes"] == 1
    assert report["scope"]["validation_episodes"] == 1
    assert report["scope"]["optimizer_used"] is True
    assert report["checks"]["validation_excluded_from_optimizer"] is True
    assert report["checks"]["source_actor_model_bitwise_unchanged"] is True
    assert torch.equal(policy.model.actor.detach(), actor_before)
    for name in ("report.json", "summary.json", "replay_cache.pt", "legacy_critic.pt", "context_critic.pt"):
        assert (output / name).is_file()
    saved_report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    saved_summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert saved_report["status"] == saved_summary["status"] == "COMPLETED"
    cache = torch.load(output / "replay_cache.pt", map_location="cpu", weights_only=False)
    assert len(cache["episodes"]) == 2
    legacy_payload = torch.load(output / "legacy_critic.pt", map_location="cpu", weights_only=False)
    context_payload = torch.load(output / "context_critic.pt", map_location="cpu", weights_only=False)
    assert legacy_payload["diagnostic_only"] is True
    assert context_payload["diagnostic_only"] is True
    assert all(str(name).startswith(probe.CRITIC_PREFIXES) for name in legacy_payload["state_dict"])
    assert all(str(name).startswith(probe.CRITIC_PREFIXES) for name in context_payload["state_dict"])
    assert report["metrics"]["validation"]["before"]["steps"] == 1
    assert report["metrics"]["validation"]["after"]["steps"] == 1


def test_load_tape_default_limit_is_unchanged_and_explicit_probe_limit_accepts(tmp_path: Path) -> None:
    path = tmp_path / "long.json"
    path.write_text(json.dumps({"action_tape": [
        {"step": index, "a_requested": {}} for index in range(1001)
    ]}), encoding="utf-8")
    with pytest.raises(credit.PPOCreditSignalError, match="exceeds 1000"):
        credit._load_tape(path)
    _payload, rows = credit._load_tape(path, max_replay_steps=1800)
    assert len(rows) == 1001
    with pytest.raises(credit.PPOCreditSignalError, match="positive integer"):
        credit._load_tape(path, max_replay_steps=0)


def test_capture_replay_state_observer_is_pre_step_and_optional(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeActor:
        committed_plan = None
        phase = "ring_outer"
        arrival_id = 12

    class FakeEnv:
        def __init__(self, config):
            self.config = config
            self.step_count = 0
            self.now = 0.0
            self.active = {42: FakeActor()}
            self.future_release_reservations = {}

        def _observations(self):
            return {"42": np.zeros(88, dtype=np.float32)}

        def step(self, actions, proposal_shortlists=None):
            self.step_count += 1
            self.now += 1.0
            return self._observations(), {"42": 2.0}, {}, {}, {
                "terminal": True,
                "reward_components": {"42": {"time": 2.0, "total": 2.0}},
            }

    config = SimpleNamespace(dt=1.0, total_requested_uavs=24, max_simulation_seconds=10.0)
    tape = tmp_path / "fake_tape.json"
    tape.write_text("{}", encoding="utf-8")
    tape_rows = [{
        "step": 0,
        "a_requested": {"42": [0.0, 0.0, 0.0]},
        "requested_gate": {"42": 0},
        "requested_proposal": {"42": -1},
    }]
    monkeypatch.setattr(credit, "TerminalParallelEnv", FakeEnv)
    monkeypatch.setattr(credit.student_rollout_v4_2, "world_value", lambda policy, world: 3.0)
    monkeypatch.setattr(credit, "_policy_release_probabilities", lambda policy, world: np.asarray([0.4], dtype=np.float32))
    monkeypatch.setattr(credit.terminal_lifecycle_metrics, "_capture_terminal_rows", lambda env: ([], {}))
    monkeypatch.setattr(credit.terminal_credit, "_request_capture", lambda env: [])
    monkeypatch.setattr(credit, "_request_lifecycle", lambda *args: {
        "terminal": True, "all_requests_observed": True, "ledger_alignment": True,
        "request_rows": [],
    })

    def fake_replay(env, path):
        env.step({"42": np.zeros(3, dtype=np.float32)}, {})
        return {"exact": True, "steps_replayed": 1}

    monkeypatch.setattr(credit.SafetyTraceRecorder, "replay_action_tape", fake_replay)
    result = credit._capture_replay(
        object(), config, tape, tape_rows,
        state_observer=lambda env: {"time_before": env.now}, max_replay_steps=1800,
    )
    assert result["captures"][0]["state_context"] == {"time_before": 0.0}
