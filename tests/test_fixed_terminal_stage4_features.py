from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import pytest
from terminal_marl.configs.fixed_terminal_stage1 import DEFAULT_CONFIG
from terminal_marl.core.fixed_terminal.scenarios import Request, Scenario
from terminal_marl.core.fixed_terminal.routes import build_plan
from terminal_marl.core.fixed_terminal.learning_features import FEATURE_NAMES, extract_features
from terminal_marl.envs.fixed_terminal_env import FixedTerminalEnv


def tiny(*requests, horizon=20):
    return FixedTerminalEnv(replace(DEFAULT_CONFIG, horizon=horizon), Scenario(4001, "test", "test", tuple(requests), 1, 2))


def test_features_future_identity_order_and_source_independent():
    env = tiny(Request("a", 0, .8, 0, .2), Request("b", 0, .6, 1, 2), Request("secret", 15, .05, None, 0))
    plans = (build_plan(env.aircraft["a"], 0, env.config),)
    before = deepcopy(env.aircraft)
    first = extract_features(env, plans)
    other = deepcopy(env)
    other.aircraft["secret"].soc = .99
    other.scenario = replace(other.scenario, seed=99, load="hidden", soc_profile="hidden")
    other.aircraft = dict(reversed(list(other.aircraft.items())))
    assert extract_features(other, plans) == first
    other.aircraft["renamed99999"] = other.aircraft.pop("a")
    other.aircraft["renamed99999"].aircraft_id = "renamed99999"
    renamed = (replace(plans[0], aircraft_id="renamed99999"),)
    assert extract_features(other, renamed) == first
    assert env.aircraft == before
    assert len(first) == len(FEATURE_NAMES) == 93
    assert not any("seed" in n or "rank" in n or "aircraft_id" in n or "profile" in n for n in FEATURE_NAMES)
    with pytest.raises(ValueError, match="observed"):
        extract_features(env, (replace(plans[0], aircraft_id="secret"),))


def test_empty_and_expired_resources_and_observed_reserve_features():
    env = tiny(Request("a", 0, .8, 0, 0))
    plan = build_plan(env.aircraft["a"], 0, env.config)
    row = dict(zip(FEATURE_NAMES, extract_features(env, (plan,))))
    assert row["candidate_plan_count"] == 1 and row["external_soc_mean"] == 0
    assert row["candidate_postroute_reserve_slack_min"] < .8-env.config.energy.reserve
    env.submit_batch((plan,))
    row = dict(zip(FEATURE_NAMES, extract_features(env, ())))
    assert row["reserved_pad_entry_count"] == 2
    while not env.done:
        env.step()
    row = dict(zip(FEATURE_NAMES, extract_features(env, ())))
    assert row["reserved_pad_entry_count"] == 0 and row["reserved_pad_earliest_end_remaining"] == 0
    assert row["scheduled_completion_remaining_max"] == 0
