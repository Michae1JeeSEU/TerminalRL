from __future__ import annotations

import importlib
import json


MODULE = "terminal_marl.training.collect_p3_proposal_head_probe_v1"


def test_quantile_slots_are_32_and_use_protocol_denominator():
    collector = importlib.import_module(MODULE)
    steps = collector._target_steps(625)
    assert len(steps) == 32
    assert steps[0] == 0
    assert steps[-1] == 604
    assert steps == sorted(steps)


def test_source_split_is_fixed_and_disjoint():
    collector = importlib.import_module(MODULE)
    assert [collector._split_for_seed(seed) for seed in range(873101, 873107)] == [
        "train", "train", "train", "validation", "validation", "validation"
    ]


def test_failed_replay_cannot_be_reported_completed():
    collector = importlib.import_module(MODULE)
    report = {
        "replay_exact": False,
        "outcomes_match": True,
        "rng_unchanged": True,
        "model_weights_unchanged": True,
        "errors": [{"kind": "replay"}],
    }
    assert collector._manifest_status("full", [report] * 6, []) == "FAILED"


def test_row_contract_is_json_serializable():
    collector = importlib.import_module(MODULE)
    row = {
        "observation": [0.0] * 88,
        "source_proposal_logits": [0.0] * 25,
        "source_gate_logits": [0.0, 0.0],
        "certified_proposals": [0, 4, 24],
        "seed": 873101,
        "split": "train",
        "step": 0,
        "uid": 1,
        "arrival_id": 0,
        "soc": 0.8,
        "priority": 2,
        "phase": "ring_outer",
    }
    encoded = json.dumps(collector._jsonable(row), allow_nan=False)
    decoded = json.loads(encoded)
    assert len(decoded["observation"]) == 88
    assert set(decoded["certified_proposals"]) <= set(range(25))
