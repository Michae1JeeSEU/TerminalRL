from __future__ import annotations

import inspect

from terminal_marl.training import collect_v4_2_p2_gate_proposal_support_v1 as support


def _row(index: int, *, split: str = "train", interface: str = "o2m") -> dict:
    return {
        "actor_mask": True,
        "teacher_gate": 1,
        "teacher_proposal": index % 10,
        "requested_gate": 1,
        "requested_proposal": (index + 1) % 10,
        "interface": interface,
        "arrival_id": 6 + index % 6,
        "seed": support.TRAIN_SEEDS[index % len(support.TRAIN_SEEDS)] if split == "train"
        else support.DEV_SEEDS[index % len(support.DEV_SEEDS)],
        "decision_event_id": f"{split}:{index}",
        "observation_v2_1": [0.0] * 88,
    }


def test_protocol_freezes_fresh_flow12_split_and_no_training():
    value = support.protocol()
    assert value["formal"]["train_seeds"] == list(support.TRAIN_SEEDS)
    assert value["formal"]["dev_seeds"] == list(support.DEV_SEEDS)
    assert value["formal"]["scenario"] == "flow12"
    assert set(support.TRAIN_SEEDS).isdisjoint(support.DEV_SEEDS)
    assert value["bindings"]["retained_rows"] == "actor_events_with_teacher_RELEASE"
    assert value["authorizations"]["always_false"]["PPO_ALLOWED"] is False


def test_retain_proposal_rows_filters_teacher_wait_and_non_actor():
    good = _row(1)
    teacher_wait = {**_row(2), "teacher_gate": 0, "teacher_proposal": None}
    non_actor = {**_row(3), "actor_mask": False}
    retained = support.retain_proposal_rows([good, teacher_wait, non_actor], split="train")
    assert len(retained) == 1
    assert retained[0]["proposal_support_split"] == "train"
    assert retained[0]["proposal_target"] == good["teacher_proposal"]


def test_train_metrics_and_gates_pass_sufficient_balanced_support():
    rows = []
    for index in range(400):
        if index < 170:
            interface = "o2m"
        elif index < 290:
            interface = "m2i"
        else:
            interface = "i2f"
        rows.extend(support.retain_proposal_rows([_row(index, interface=interface)], split="train"))
    records = [
        {"split": "train", "seed": seed, "completed": 12, "gates": {"all": True}}
        for seed in support.TRAIN_SEEDS
    ]
    metrics = support.split_metrics(rows, "train", records, support.TRAIN_SEEDS)
    gates = support.split_gates(metrics, "train")
    assert metrics["rows"] == 400
    assert metrics["unique_teacher_proposals"] == 10
    assert metrics["late_arrival_fraction"] == 1.0
    assert gates["all"] is True


def test_cross_split_checks_detect_seed_or_event_overlap():
    train = support.retain_proposal_rows([_row(1)], split="train")[0]
    dev = support.retain_proposal_rows([_row(2, split="dev")], split="dev")[0]
    assert all(support.cross_split_checks([train, dev]).values())
    bad = {**dev, "seed": train["seed"], "decision_event_id": train["decision_event_id"]}
    checks = support.cross_split_checks([train, bad])
    assert checks == {"train_dev_seed_disjoint": False, "train_dev_event_disjoint": False}


def test_collector_has_no_optimizer_or_repair_path():
    source = inspect.getsource(support)
    assert "optimizer.step(" not in source
    assert '"PROPOSAL_REPAIR_ALLOWED": False' in source
    assert '"PPO_ALLOWED": False' in source
    assert '"checkpoint_promotable": False' in source

