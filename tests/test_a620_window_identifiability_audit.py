import numpy as np

from terminal_marl.training.a620_window_identifiability_audit import (
    OBSERVATION_NAMES, PPO_AUTHORIZATION, WAITING_MATERIALITY, _label, classify,
)


def _row(seed, label):
    return {"seed": seed, "outcome_label": label, "observation_29d": [float(seed)] * 29}


def _classifier(macro, permutation):
    return [{"model": "logistic_linear", "split": "leave_one_seed_out_aggregate", "macro_f1_present": macro,
             "permutation_macro_f1_mean": permutation, "balanced_accuracy": macro}]


def test_a620_label_boundaries_are_frozen_at_50k_waiting_area():
    assert _label(-WAITING_MATERIALITY) == "BENEFICIAL"
    assert _label(WAITING_MATERIALITY) == "HARMFUL"
    assert _label(WAITING_MATERIALITY - 1.) == "NEUTRAL"


def test_a620_observation_schema_is_exactly_29_pre_action_features():
    assert len(OBSERVATION_NAMES) == 29
    assert "seed" not in OBSERVATION_NAMES and "time" not in OBSERVATION_NAMES


def test_a620_classification_requires_cross_seed_and_beneficial_harmful_coverage():
    sparse = [_row(0, "BENEFICIAL"), _row(1, "HARMFUL")] * 3
    assert classify(sparse, [], _classifier(.9, .1)) == "INSUFFICIENT_DATA"
    rows = [_row(seed, label) for seed in range(6) for label in ("BENEFICIAL", "HARMFUL", "NEUTRAL")]
    nearest = [{"label_conflict": False} for _ in rows]
    assert classify(rows, nearest, _classifier(.8, .2)) == "IDENTIFIABLE"


def test_a620_cannot_authorize_ppo():
    assert PPO_AUTHORIZATION == "STOP"
