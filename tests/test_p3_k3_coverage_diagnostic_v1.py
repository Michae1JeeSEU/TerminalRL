from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "terminal_marl" / "training" / "diagnose_p3_k3_coverage_v1.py"
    spec = importlib.util.spec_from_file_location("p3_k3_coverage_diagnostic_v1", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sample(module):
    labels = ("source_03c0", "run0_episode_0080", "run1_episode_0240", "run2_episode_0240")
    return {
        "status": "evaluated",
        "scenario": "flow6",
        "ring_role": "ring_outer",
        "soc_bucket": "gt_0.65",
        "priority": 1,
        "late_merge_off": {"certified_release_count": 1},
        "late_merge_on": {"certified_release_count": 2},
        "policies": {
            label: {
                "q_k3": 0.1,
                "q_k3_on": 0.2,
                "p_release": 0.3,
                "p_release_q_k3": 0.03,
                "p_release_q_k3_on": 0.06,
            }
            for label in labels
        },
    }


def test_replay_error_cannot_complete():
    module = _module()
    report = {
        "error": {"type": "ReplayError"},
        "exact": True,
        "outcomes_match": True,
        "rng_unchanged": True,
        "policy_weights_unchanged": True,
    }
    assert module._completion_status("full", [report]) == "FAILED"


def test_source_and_witness_aggregates_are_separate():
    module = _module()
    source = {"witness": False, "samples": [_sample(module)]}
    witness = {"witness": True, "samples": [_sample(module)]}
    aggregate = module._aggregate([source, witness])
    assert aggregate["source_aggregate"]["evaluated_states"] == 1
    assert aggregate["witness_aggregate"]["evaluated_states"] == 1
    assert aggregate["source_aggregate"]["groups"]
    assert aggregate["witness_aggregate"]["groups"]
