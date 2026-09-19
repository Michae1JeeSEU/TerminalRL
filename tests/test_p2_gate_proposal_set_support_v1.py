from __future__ import annotations

import inspect

import pytest

from terminal_marl.training import collect_v4_2_p2_gate_proposal_set_support_v1 as support
from terminal_marl.training.student_rollout_v4_2 import rollout


def _retained(index: int, *, split: str, interface: str, seed: int) -> dict:
    feasible = [index % 10, (index + 1) % 10]
    return {
        "proposal_set_support_split": split,
        "proposal_set_target": feasible,
        "proposal_set_signature": ",".join(str(value) for value in feasible),
        "proposal_set_size": len(feasible),
        "teacher_catalogue_complete": True,
        "teacher_proposal": feasible[0],
        "policy_top1_feasible": index % 4 != 0,
        "policy_top3_feasible_hit": index % 9 != 0,
        "interface": interface,
        "arrival_id": 6 + index % 6,
        "seed": seed,
        "decision_event_id": f"{split}:{index}",
        "observation_v2_1": [0.0] * 88,
    }


def test_protocol_freezes_teacher_release_catalogues_and_no_training():
    value = support.protocol()
    assert value["formal"]["train_seeds"] == list(support.TRAIN_SEEDS)
    assert value["formal"]["dev_seeds"] == list(support.DEV_SEEDS)
    assert set(support.TRAIN_SEEDS).isdisjoint(support.DEV_SEEDS)
    assert value["bindings"]["catalogue_scope"] == "actor_events_with_teacher_RELEASE"
    assert value["authorizations"]["always_false"]["PPO_ALLOWED"] is False


def test_rollout_rejects_unknown_catalogue_scope(tmp_path):
    with pytest.raises(ValueError, match="catalogue_scope"):
        rollout(
            None, support.attribution.diagnostic_config(), seed=1, max_steps=1,
            output=tmp_path / "invalid", deterministic=True, collect_labels=True,
            collect_diagnostics=True, collect_catalogue_diagnostics=True,
            catalogue_scope="unknown",
        )


def test_train_metrics_and_gates_accept_sufficient_set_support():
    rows = []
    for index in range(400):
        interface = "o2m" if index < 150 else ("m2i" if index < 280 else "i2f")
        rows.append(_retained(
            index, split="train", interface=interface,
            seed=support.TRAIN_SEEDS[index % len(support.TRAIN_SEEDS)],
        ))
    records = [
        {"split": "train", "seed": seed, "completed": 12, "gates": {"all": True}}
        for seed in support.TRAIN_SEEDS
    ]
    metrics = support.split_metrics(rows, "train", records, support.TRAIN_SEEDS)
    gates = support.split_gates(metrics, "train")
    assert metrics["rows"] == 400
    assert metrics["unique_feasible_sets"] == 10
    assert metrics["policy_top1_infeasible"] == 100
    assert metrics["policy_top3_miss"] >= 10
    assert gates["all"] is True


def test_cross_split_checks_detect_overlap():
    train = _retained(1, split="train", interface="o2m", seed=support.TRAIN_SEEDS[0])
    dev = _retained(2, split="dev", interface="o2m", seed=support.DEV_SEEDS[0])
    assert all(support.cross_split_checks([train, dev]).values())
    bad = {**dev, "seed": train["seed"], "decision_event_id": train["decision_event_id"]}
    assert support.cross_split_checks([train, bad]) == {
        "train_dev_seed_disjoint": False, "train_dev_event_disjoint": False,
    }


def test_collector_has_no_optimizer_or_repair_path():
    source = inspect.getsource(support)
    assert "optimizer.step(" not in source
    assert '"PROPOSAL_REPAIR_ALLOWED": False' in source
    assert '"PPO_ALLOWED": False' in source
    assert '"checkpoint_promotable": False' in source
