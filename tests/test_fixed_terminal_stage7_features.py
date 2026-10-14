from copy import deepcopy
from terminal_marl.core.fixed_terminal.search_features import FEATURE_NAMES, extract_search_features
from terminal_marl.core.fixed_terminal.baselines import observed_clone
from terminal_marl.core.fixed_terminal.routes import build_plan
from dataclasses import replace
from terminal_marl.configs.fixed_terminal_stage1 import DEFAULT_CONFIG
from terminal_marl.envs.fixed_terminal_env import FixedTerminalEnv
from terminal_marl.core.fixed_terminal.scenarios import Request, Scenario


def tiny(*requests):
    return FixedTerminalEnv(replace(DEFAULT_CONFIG, horizon=12.), Scenario(7001, "low", "high", tuple(requests), 1, 2))


def test_future_invariance_and_frozen109_schema():
    env = tiny(Request("a", 0, .95, 0, .1), Request("future", 5., .5, None, .4))
    plans = (build_plan(env.aircraft["a"], env.time, env.config),)
    catalogue = {"a": plans}
    expected = extract_search_features(env, (), catalogue, 0, 8, 1)
    altered = deepcopy(env)
    altered.aircraft["future"].soc = .99
    altered.aircraft["future"].theta = 4.
    assert expected == extract_search_features(altered, (), catalogue, 0, 8, 1)
    assert expected == extract_search_features(observed_clone(env), (), catalogue, 0, 8, 1)
    assert len(FEATURE_NAMES) == len(expected) == 109
    assert not any(word in name for name in FEATURE_NAMES for word in ("seed", "label", "outcome", "root_id", "method"))
    selected = extract_search_features(env, plans, catalogue, 1, 8, 1)
    assert selected[FEATURE_NAMES.index("candidate_plan_count")] == 1
    assert selected[FEATURE_NAMES.index("search_remaining_catalogue_holders")] == 0
    assert selected[FEATURE_NAMES.index("search_partial_negative_energy")] < 0
