from terminal_marl.training.a621_observation_v2_sufficiency_audit import (
    BASE_NAMES, FLOW_INDEX, PPO_AUTHORIZATION, RESERVATION_INDEX, V2_NAMES, WINDOW_INDEX, _feature_key, classify,
)
from terminal_marl.envs.high_level_wrapper import DynamicHighLevelWrapper, RuleLowLevelAdapter
from terminal_marl.training.a611_causal_audit import DemandProfile
from terminal_marl.training.a613_capacity_reaudit import repair_config
from terminal_marl.training.a621_observation_v2_sufficiency_audit import observation_v2


def _rows(harmful_seeds=(0, 1, 2)):
    rows = []
    for seed in range(12):
        label = "HARMFUL" if seed in harmful_seeds else ("BENEFICIAL" if seed % 2 else "NEUTRAL")
        rows.append({"seed": seed, "outcome_label": label})
    return rows


def _classifier(v1, v2, harmful_recall=.7):
    return [
        {"feature_set": "v1_29d", "split": "leave_one_seed_out_aggregate", "macro_f1_present": v1, "permutation_macro_f1_mean": .3},
        {"feature_set": "v2_full", "split": "leave_one_seed_out_aggregate", "macro_f1_present": v2, "permutation_macro_f1_mean": .3, "harmful_heldout_seed_count": 3, "harmful_recall_mean": harmful_recall},
    ]


def test_a621_v2_has_named_nonoverlapping_blocks_and_keeps_v1_external():
    assert len(V2_NAMES) == 46 and len(BASE_NAMES) == 24
    assert set(FLOW_INDEX).isdisjoint(RESERVATION_INDEX) and set(RESERVATION_INDEX).isdisjoint(WINDOW_INDEX)
    assert max(WINDOW_INDEX) == len(V2_NAMES) - 1
    assert _feature_key("v2_full") == "v2" and _feature_key("v1_29d") == "v1_29d"


def test_a621_v2_is_online_and_read_only_at_a_real_pre_action_state():
    profile = DemandProfile("v2", ((0., 5., .1),), 5., 20.)
    wrapper = DynamicHighLevelWrapper(repair_config(profile, dynamic=True), RuleLowLevelAdapter()); wrapper.reset(0)
    before = wrapper.base.now; row = observation_v2(wrapper)
    assert row["online_only"] and row["source_time"] == before and len(row["values"]) == len(V2_NAMES)


def test_a621_requires_harmful_coverage_before_any_positive_conclusion():
    conflicts = [{"feature_set": "v1_29d", "nearest_conflict_rate": .6}, {"feature_set": "v2_full", "nearest_conflict_rate": .1}]
    assert classify(_rows((0,)), conflicts, _classifier(.3, .9)) == "INSUFFICIENT_HARMFUL_COVERAGE"


def test_a621_identifiable_gate_requires_v2_uplift_conflict_reduction_and_harmful_recall():
    conflicts = [{"feature_set": "v1_29d", "nearest_conflict_rate": .6}, {"feature_set": "v2_full", "nearest_conflict_rate": .1}]
    assert classify(_rows(), conflicts, _classifier(.4, .8)) == "V2_IDENTIFIABLE"


def test_a621_cannot_authorize_ppo():
    assert PPO_AUTHORIZATION == "STOP"
