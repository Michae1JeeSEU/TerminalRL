"""Output-closure regressions for the frozen V3 L1 short-pilot diagnostics."""
from __future__ import annotations

import json
from types import SimpleNamespace

from terminal_marl.algo.shared_ppo import SharedPPO
from terminal_marl.training import run_v3_l1_short_pilot as pilot
from terminal_marl.training.run_v3_l1_short_pilot import _records_metrics, _serializable
from terminal_marl.training.train_low_v2_ppo import _model_state, _rollout_l1, l1_full_teacher_config


def test_short_pilot_metrics_keep_three_head_and_queue_provenance():
    config = l1_full_teacher_config(medium_rate=.001575, reset_mode="warm_start_rings", demand_model="finite_cohort",
                                    warm_start_uavs=8, cohort_uavs=8, horizon_seconds=2.0)
    policy = SharedPPO(87, teacher_bc_weight=.05)
    rows, summary = _rollout_l1(_model_state(policy), config, 1001, 0, 0, max_steps=2)
    metrics = _records_metrics([(rows, summary)], _model_state(policy))
    assert metrics["environment"]["external_queue_mean"] is not None
    assert set(metrics["distribution"]["heads"]) == {"release", "speed", "merge_time"}
    release = metrics["distribution"]["heads"]["release"]
    assert set(release["requested"]) == {"WAIT", "RELEASE"}
    assert set(metrics["distribution"]["combined_corrections"]) == {
        "none", "release-only", "speed-only", "merge-only", "release+speed", "release+merge", "speed+merge", "all-three",
    }
    json.dumps(_serializable(metrics), allow_nan=False)


def test_validation_grouping_uses_episode_demand_model(monkeypatch):
    def metric(records, _state):
        return {"count": len(records)}
    monkeypatch.setattr(pilot, "_records_metrics", metric)
    records = [
        ([], SimpleNamespace(reset_mode="warm_start_rings", demand_model="finite_cohort")),
        ([], SimpleNamespace(reset_mode="flow_from_empty", demand_model="finite_cohort")),
        ([], SimpleNamespace(reset_mode="flow_from_empty", demand_model="poisson")),
    ]
    grouped = pilot._validation_metrics(records, {})
    assert grouped["warm_start"]["count"] == 1
    assert grouped["from_empty"]["count"] == 2
    assert grouped["from_empty_finite"]["count"] == grouped["poisson"]["count"] == 1
