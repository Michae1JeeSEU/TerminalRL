from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from terminal_marl.visualization import console_policy
from terminal_marl.visualization import ppo_checkpoint_viewer as viewer


def _checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"console-policy-checkpoint")
    return path


def _config(path: Path, **overrides):
    value = {
        "mode": "checkpoint",
        "profile": "p2_initial_k3",
        "scenario": "flow6",
        "seed": 17,
        "low_checkpoint": str(path),
        "high_checkpoint": None,
        "start_paused": False,
        "max_steps": 1,
        "max_simulation_seconds": 2.0,
        "playback_speed": 0.0,
    }
    value.update(overrides)
    return value


def test_console_policy_routes_frozen_profile_and_reports_provenance(monkeypatch, tmp_path: Path):
    checkpoint = _checkpoint(tmp_path)
    calls = {}

    def fake_run(path, **kwargs):
        calls["path"] = path
        calls["kwargs"] = kwargs
        return {
            "profile": "p2_initial_k3",
            "scenario": "flow6",
            "seed": 17,
            "observation_dim": 88,
            "runtime_proposal_shortlist_k": 3,
            "deterministic": True,
            "weights_unchanged": True,
            "optimizer_loaded": False,
        }

    monkeypatch.setattr(console_policy._viewer, "run", fake_run)
    factory = lambda _env, _render_config: object()
    result = console_policy.run_checkpoint(_config(checkpoint), factory)

    assert calls["path"] == checkpoint.resolve()
    assert calls["kwargs"]["gui"] is False
    assert calls["kwargs"]["observer_factory"] is factory
    assert calls["kwargs"]["max_steps"] == 1
    assert calls["kwargs"]["max_simulation_seconds"] == 2.0
    assert result["checkpoint"] == str(checkpoint.resolve())
    assert result["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert result["shortlist_k"] == 3
    assert result["observation_dim"] == 88
    assert result["deterministic"] is True
    assert result["weights_unchanged"] is True
    assert result["optimizer_loaded"] is False


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"mode": "live"}, "mode='checkpoint'"),
        ({"profile": "unknown"}, "unsupported checkpoint profile"),
        ({"scenario": "p2_dense24", "profile": "checkpoint12"}, "unsupported scenario"),
        ({"deterministic": False}, "deterministic=True"),
        ({"unexpected": 1}, "unsupported checkpoint config fields"),
    ],
)
def test_console_policy_rejects_bad_profile_scenario_and_schema(tmp_path: Path, overrides, match):
    checkpoint = _checkpoint(tmp_path)
    with pytest.raises((TypeError, ValueError), match=match):
        console_policy.run_checkpoint(_config(checkpoint, **overrides), None)


def test_console_policy_rejects_missing_checkpoint(tmp_path: Path):
    with pytest.raises(ValueError, match="checkpoint does not exist"):
        console_policy.run_checkpoint(_config(tmp_path / "missing.pt"), None)


def test_ppo_viewer_observer_hook_preserves_request_and_closes_after_final_frame(monkeypatch):
    config = SimpleNamespace(
        dt=1.0,
        max_simulation_seconds=2.0,
        teacher_enabled=False,
        runtime_proposal_shortlist_k=3,
    )

    class FakeEnv:
        def __init__(self, _config):
            self.config = config
            self.active = {}
            self.arrivals = SimpleNamespace(exhausted=lambda: True)
            self.admission = SimpleNamespace(queue=SimpleNamespace(items=[]))
            self.service_uavs = {}
            self.completed_uavs = []
            self.telemetry = {}
            self.now = 0.0
            self.step_calls = []
            self.observer = None

        def reset(self, seed):
            self.seed = seed
            return {}, {}

        def set_step_observer(self, observer):
            self.observer = observer

        def step(self, requested, proposals):
            self.step_calls.append((requested, proposals))
            self.now += 1.0
            return {}, {}, {}, {}, {}

    class FakeObserver:
        stop_requested = False
        playback_speed = 0.0

        def __init__(self):
            self.ready_calls = 0
            self.finish_calls = []
            self.close_calls = 0

        def wait_ready(self):
            self.ready_calls += 1
            return True

        def finish(self, env, *, wait_for_close):
            self.finish_calls.append((env, wait_for_close))

        def close(self):
            self.close_calls += 1

        def __call__(self, env):
            self.callback_env = env

    policy = SimpleNamespace(model=SimpleNamespace(eval=lambda: None), observation_dim=88)
    requested = {"1": "WAIT"}
    proposals = {"1": (0, 1, 2)}
    calls = {"details": []}

    monkeypatch.setattr(viewer, "_p2_scene_config", lambda _scenario: config)
    monkeypatch.setattr(viewer, "_load_profile", lambda _path, _profile: policy)
    monkeypatch.setattr(viewer, "TerminalParallelEnv", FakeEnv)
    monkeypatch.setattr(viewer, "_student_request_details", lambda *args, **kwargs: (
        None, None, None, requested, None, proposals,
    ))
    monkeypatch.setattr(viewer, "_pace_live_rollout", lambda _observer, _dt, _started: 0.0)

    created = []

    def factory(env, render_config):
        created.append((env, render_config))
        return FakeObserver()

    result = viewer.run(
        "ignored.pt",
        profile="p2_initial_k3",
        scenario="flow6",
        seed=19,
        gui=False,
        observer_factory=factory,
        max_steps=3,
        max_simulation_seconds=1.0,
    )

    env = created[0][0]
    observer = env.observer
    assert env._console_step_limit == 1
    assert env.step_calls == [(requested, proposals)]
    assert observer.ready_calls == 1
    assert observer.finish_calls == [(env, False)]
    assert observer.close_calls == 1
    assert result["deterministic"] is True
    assert result["runtime_proposal_shortlist_k"] == 3


def test_ppo_viewer_rejects_gui_and_factory_together(monkeypatch):
    with pytest.raises(ValueError, match="mutually exclusive"):
        viewer.run("ignored.pt", gui=True, observer_factory=lambda *_args: None)


@pytest.mark.parametrize("kwargs", [{"max_steps": 0}, {"max_steps": True}, {"max_steps": 1.5}])
def test_ppo_viewer_rejects_invalid_step_cap(kwargs):
    with pytest.raises(ValueError, match="max_steps must be a positive integer"):
        viewer.run("ignored.pt", **kwargs)


@pytest.mark.parametrize("seconds", [0.0, -1.0, float("nan"), float("inf")])
def test_ppo_viewer_rejects_invalid_seconds_cap(seconds):
    with pytest.raises(ValueError, match="max_simulation_seconds must be a finite positive number"):
        viewer.run("ignored.pt", max_simulation_seconds=seconds)


def test_p3_task_neutral_routes_formal_config_and_caps_observer(monkeypatch, tmp_path: Path):
    checkpoint = _checkpoint(tmp_path)
    sim_config = SimpleNamespace(
        dt=1.0,
        max_simulation_seconds=4.0,
        runtime_proposal_shortlist_k=3,
        teacher_enabled=False,
        task_priority_enabled=False,
        low_level_priority_time_weight=0.0,
        observation_schema_version="phaseA_low_obs_v2_1_retry_state",
        action_schema_version="low_action_schema_v4_2_hierarchical",
        high_level_enabled=False,
        scheduler_mode="execution",
    )

    class FakeEnv:
        def __init__(self, config):
            self.config = config
            self.active = {1: object()}
            self.arrivals = SimpleNamespace(exhausted=lambda: False)
            self.admission = SimpleNamespace(queue=SimpleNamespace(items=[]))
            self.service_uavs = {}
            self.completed_uavs = []
            self.telemetry = {}
            self.now = 0.0
            self.step_count = 0
            self.step_calls = []
            self.observer = None

        def reset(self, seed):
            self.seed = seed
            self.now = 0.0
            self.step_count = 0
            return {"1": object()}, {}

        def set_step_observer(self, observer):
            self.observer = observer

        def step(self, requested, proposals):
            self.step_calls.append((requested, proposals))
            self.now += 1.0
            self.step_count += 1
            if self.observer is not None:
                self.observer(self)
            return {"1": object()}, {}, {}, {}, {}

    class FakeObserver:
        stop_requested = False
        paused = False
        playback_speed = 0.0

        def __init__(self):
            self.frames = 0
            self.finish_calls = []
            self.close_calls = 0

        def wait_ready(self):
            return True

        def __call__(self, _env):
            self.frames += 1

        def finish(self, env, *, wait_for_close):
            self.finish_calls.append((env, wait_for_close))

        def close(self):
            self.close_calls += 1

    policy = SimpleNamespace(
        model=SimpleNamespace(eval=lambda: None, state_dict=lambda: {}),
        observation_dim=88,
        optimizer=SimpleNamespace(state={}),
    )
    requested = {"1": "WAIT"}
    proposals = {"1": (0, 1, 2)}
    loader_calls = {}

    from terminal_marl.training import evaluate_p3_runtime_rank_repair as rank_eval
    from terminal_marl.training import evaluate_p3_task_neutral as neutral_eval

    monkeypatch.setattr(neutral_eval, "_build_neutral_inputs", lambda: ({}, {"flow6": sim_config}, {}))

    def load_policy(path, config):
        loader_calls["path"] = path
        loader_calls["config"] = config
        return policy, {"metadata": {"observation_dimension": 88}}

    monkeypatch.setattr(rank_eval, "_load_policy", load_policy)
    monkeypatch.setattr(viewer, "TerminalParallelEnv", FakeEnv)
    monkeypatch.setattr(viewer, "_student_request_details", lambda *args, **kwargs: (
        None, None, None, requested, None, proposals,
    ))
    monkeypatch.setattr(viewer, "_pace_live_rollout", lambda _observer, _dt, _started: 0.0)

    created = []

    def factory(env, _render_config):
        created.append(env)
        assert env._console_step_limit == 2
        assert env.now == 0.0
        return FakeObserver()

    result = console_policy.run_checkpoint(
        _config(
            checkpoint,
            profile="p3_task_neutral",
            scenario="flow6",
            deterministic=False,
            max_steps=3,
            max_simulation_seconds=2.0,
        ),
        factory,
    )

    env = created[0]
    observer = env.observer
    assert loader_calls["path"] == checkpoint.resolve()
    assert loader_calls["config"] is sim_config
    assert len(env.step_calls) == 2
    assert observer.frames == 2
    assert observer.finish_calls == [(env, False)]
    assert observer.close_calls == 1
    assert result["profile"] == "p3_task_neutral"
    assert result["deterministic"] is False
    assert result["policy_seed"] == 17
    assert result["shortlist_k"] == 3
    assert result["observation_dim"] == 88
    assert result["task_neutral"] is True
    assert result["diagnostic_step_limit"] == 2
    assert result["weights_unchanged"] is True
    assert result["optimizer_loaded"] is False


def test_p3_task_neutral_rejects_non_neutral_request_flags(tmp_path: Path):
    checkpoint = _checkpoint(tmp_path)
    with pytest.raises(ValueError, match="task-neutral request requires task_priority_enabled=False"):
        console_policy.run_checkpoint(
            _config(
                checkpoint,
                profile="p3_task_neutral",
                scenario="flow6",
                task_priority_enabled=True,
        ),
            None,
        )


def test_p3_task_neutral_defaults_to_stochastic_request_mode(tmp_path: Path):
    checkpoint = _checkpoint(tmp_path)
    values = console_policy._validate_config(
        _config(checkpoint, profile="p3_task_neutral", scenario="flow6")
    )
    assert values["deterministic"] is False


def test_p3_task_neutral_rejects_incompatible_checkpoint_flags():
    expected = SimpleNamespace(
        observation_schema_version="phaseA_low_obs_v2_1_retry_state",
        action_schema_version="low_action_schema_v4_2_hierarchical",
        high_level_enabled=False,
        scheduler_mode="execution",
        teacher_enabled=False,
    )
    with pytest.raises(ValueError, match="task_priority_enabled=False"):
        console_policy._validate_neutral_semantics(
            {"metadata": {"task_priority_enabled": True}},
            expected_config=expected,
        )
    console_policy._validate_neutral_semantics(
        {
            "metadata": {
                "neutral_config_snapshot": {
                    "task_priority_enabled": False,
                    "low_level_priority_time_weight": 0.0,
                    "observation_dimension": 88,
                },
            },
        },
        expected_config=expected,
    )
