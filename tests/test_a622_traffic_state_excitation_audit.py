from terminal_marl.training.a614_reconfiguration_timing_audit import PPO_AUTHORIZATION
from terminal_marl.training.a622_traffic_state_excitation_audit import _feature_excitation, _classification, regime_definitions
from terminal_marl.training.a621_observation_v2_sufficiency_audit import V2_NAMES


def _row(value):
    return {"v2": [float((value * (index % 7 + 1) + index) % (index % 5 + 2)) for index in range(len(V2_NAMES))]}


def test_a622_defines_formal_flow_empty_and_stress_only_warm_start_regimes():
    rows, _ = regime_definitions(.02)
    formal = [row for row in rows if row["class"] == "traffic_excited_formal"]
    stress = next(row for row in rows if row["class"] == "stress_diagnostic_only")
    assert {row["regime"] for row in formal} == {"formal_high", "formal_bursty", "formal_overload"}
    assert all(row["reset_mode"] == "flow_from_empty" for row in formal)
    assert stress["reset_mode"] == "warm_start_rings" and not stress["formal_value_evidence"]


def test_a622_feature_gate_detects_unexcited_and_excited_traffic_matrix():
    _, flat = _feature_excitation([_row(0) for _ in range(12)], "formal")
    _, varied = _feature_excitation([_row(index) for index in range(12)], "formal")
    assert not flat["pass"]
    assert varied["pass"]


def test_a622_feature_gate_blocks_classifier_classification():
    gate = {"pass": False}
    assert _classification(gate, [], [], []) == "EXPERIMENT_NOT_EXCITED"


def test_a622_cannot_authorize_ppo():
    assert PPO_AUTHORIZATION == "STOP"
