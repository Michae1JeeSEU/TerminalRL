from __future__ import annotations

import numpy as np

import terminal_marl.training.collect_v4_2_p2_gate_support as runner


def _row(index: int, interface: str, gate: int, *, split: str = "train", seed: int | None = None):
    phase = {"o2m": "ring_outer", "m2i": "ring_middle", "i2f": "ring_inner"}[interface]
    return {
        "seed": 772001 if seed is None else seed,
        "phase": phase,
        "step": index,
        "uav_id": index % 12,
        "decision_event_id": f"event-{split}-{index}",
        "observation_v2_1": np.zeros(88, dtype=np.float32).tolist(),
        "actor_mask": True,
        "teacher_gate": gate,
        "teacher_proposal": 0 if gate else None,
        "requested_gate": gate,
        "support_split": split,
        "scenario": "flow6",
        "interface": interface,
    }


def _passing_rows(split: str, count: int, seeds: tuple[int, ...]):
    rows = []
    for index in range(count):
        interface = runner.INTERFACES[index % len(runner.INTERFACES)]
        gate = (index // len(runner.INTERFACES)) % 2
        rows.append(_row(index, interface, gate, split=split, seed=seeds[index % len(seeds)]))
    return rows


def _records(split: str, seeds: tuple[int, ...]):
    return [{"split": split, "seed": seed, "requested": 1, "completed": 1,
             "episode_execution_complete": True,
             "gates": {"replay": True, "label_only": True}} for seed in seeds]


def test_protocol_freezes_seed_groups_and_runner_hash():
    p = runner.protocol()
    assert tuple(p["formal"]["train_support_seeds"]) == runner.FORMAL_TRAIN_SEEDS
    assert tuple(p["formal"]["dev_support_seeds"]) == runner.FORMAL_DEV_SEEDS
    assert tuple(p["formal"]["future_qualification_seeds"]) == runner.FUTURE_QUALIFICATION_SEEDS
    groups = [set(runner.FORMAL_TRAIN_SEEDS), set(runner.FORMAL_DEV_SEEDS), set(runner.FUTURE_QUALIFICATION_SEEDS)]
    assert all(not (left & right) for index, left in enumerate(groups) for right in groups[index + 1:])
    assert runner._sha(runner.PROTOCOL) == runner.PROTOCOL_SHA


def test_support_config_is_label_only_and_student_authoritative():
    config = runner.support_config(6)
    assert config.experiment_mode == "fixed_teacher_heavy"
    assert config.scheduler_mode == "execution"
    assert config.teacher_enabled and config.teacher_mode == "teacher_label_only"
    assert config.teacher_release_label_enabled is False
    assert config.low_level_progress_reward_per_meter == 0.0
    assert config.total_requested_uavs == 6


def test_actor_row_filter_keeps_only_true_gate_interface_rows():
    raw = [
        _row(0, "o2m", 0),
        {**_row(1, "m2i", 1), "actor_mask": False},
        {**_row(2, "i2f", 0), "phase": "committed"},
    ]
    selected = runner.filter_actor_rows(raw, split="dev", scenario="flow8")
    assert len(selected) == 1
    assert selected[0]["support_split"] == "dev"
    assert selected[0]["scenario"] == "flow8"
    assert selected[0]["interface"] == "o2m"


def test_split_metrics_and_gates_pass_at_frozen_lower_bounds():
    rows = _passing_rows("train", 3000, runner.FORMAL_TRAIN_SEEDS)
    result = runner.support_gates(rows, "train", _records("train", runner.FORMAL_TRAIN_SEEDS), runner.FORMAL_TRAIN_SEEDS)
    assert result["metrics"]["rows"] == 3000
    assert result["gates"]["all"]


def test_split_gates_fail_when_interface_class_is_below_minimum():
    rows = _passing_rows("dev", 1800, runner.FORMAL_DEV_SEEDS)
    # Remove every first-interface RELEASE witness while retaining global
    # cardinality and enough RELEASE rows on the other two interfaces.
    for row in rows:
        if row["interface"] == "o2m":
            row["teacher_gate"] = 0
    result = runner.support_gates(rows, "dev", _records("dev", runner.FORMAL_DEV_SEEDS), runner.FORMAL_DEV_SEEDS)
    assert not result["gates"]["interface_class_minimums"]
    assert not result["gates"]["all"]


def test_split_gates_fail_when_any_episode_gate_fails():
    rows = _passing_rows("dev", 1800, runner.FORMAL_DEV_SEEDS)
    records = _records("dev", runner.FORMAL_DEV_SEEDS)
    records[0]["gates"]["replay"] = False
    result = runner.support_gates(rows, "dev", records, runner.FORMAL_DEV_SEEDS)
    assert not result["gates"]["all_episode_gates"]
    assert not result["gates"]["all"]


def test_formal_authorization_is_narrow_and_smoke_never_authorizes():
    passing = {"all": True}
    assert runner.formal_authorization(passing, passing) == {
        "SUPPORT_COLLECTION": "PASS",
        "SUPPORT_FROZEN": True,
        "LEARNABILITY_AUDIT_ALLOWED": True,
        "GATE_REPAIR_ALLOWED": False,
        "PPO_ALLOWED": False,
        "curriculum_promotion_allowed": False,
    }
    smoke = runner.formal_authorization(passing, passing, smoke=True)
    assert smoke["SUPPORT_COLLECTION"] == "FAIL"
    assert not smoke["SUPPORT_FROZEN"] and not smoke["LEARNABILITY_AUDIT_ALLOWED"]
    assert not smoke["PPO_ALLOWED"] and not smoke["curriculum_promotion_allowed"]


def test_binding_constants_match_protocol():
    p = runner.protocol()
    assert p["candidate_checkpoint_sha256"] == runner.CANDIDATE_SHA
    assert p["bindings"]["d1_dataset_sha256"] == runner.D1_DATASET_SHA
    assert p["bindings"]["d1_manifest_sha256"] == runner.D1_MANIFEST_SHA
    assert p["bindings"]["diagnostic_manifest_sha256"] == runner.DIAGNOSTIC_MANIFEST_SHA
