from __future__ import annotations

import numpy as np

import terminal_marl.training.collect_v4_2_p2_gate_support_v2 as runner


def _row(index: int, interface: str, gate: int, *, split: str = "train", seed: int | None = None):
    phase = {"o2m": "ring_outer", "m2i": "ring_middle", "i2f": "ring_inner"}[interface]
    return {
        "seed": 775001 if seed is None else seed,
        "phase": phase,
        "step": index,
        "uav_id": index % 12,
        "decision_event_id": f"event-v2-{split}-{index}",
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


def _records(split: str, seeds: tuple[int, ...], *, fraction_gate: bool = True):
    shield_actor_count = 3 if fraction_gate else 48
    return [{
        "split": split,
        "seed": seed,
        "requested": 1,
        "completed": 1,
        "episode_execution_complete": True,
        "shield_actor_count": shield_actor_count,
        "actor_decisions": 959,
        "shield_actor_fraction": shield_actor_count / 959,
        "shield_actor_fraction_within_inherited_limit": fraction_gate,
        "gates": {
            "replay": True,
            "label_only": True,
            "shield_actor_fraction_within_inherited_limit": fraction_gate,
        },
    } for seed in seeds]


def test_protocol_freezes_fresh_seed_groups_and_hash():
    p = runner.protocol()
    assert tuple(p["formal"]["train_support_seeds"]) == runner.FORMAL_TRAIN_SEEDS
    assert tuple(p["formal"]["dev_support_seeds"]) == runner.FORMAL_DEV_SEEDS
    assert tuple(p["formal"]["future_qualification_seeds"]) == runner.FUTURE_QUALIFICATION_SEEDS
    groups = [set(runner.FORMAL_TRAIN_SEEDS), set(runner.FORMAL_DEV_SEEDS), set(runner.FUTURE_QUALIFICATION_SEEDS)]
    assert all(not (left & right) for index, left in enumerate(groups) for right in groups[index + 1:])
    assert runner._sha(runner.PROTOCOL) == runner.PROTOCOL_SHA
    assert runner.PROTOCOL_SHA != "8e614001e2475c2e1821036b88aeea93aa14afec468320eb7fd5792afca82aac"


def test_protocol_has_only_the_frozen_gate_semantic_change():
    p = runner.protocol()
    assert p["episode_gate"]["shield_actor_fraction_max"] == 0.05
    assert p["episode_gate"]["inherited_from"] == "i3_v2_1_qualification.yaml"
    assert runner.INHERITED_I3_PROTOCOL_SHA == "21d1761860ba4636c811b9a7739629e12fafe12dbf2ee9bd92ae6decfbd4f387"
    assert p["bindings"]["v1_fail_manifest_sha256"] == runner.V1_FAIL_MANIFEST_SHA


def test_shield_actor_fraction_boundary_3_of_959_passes_and_over_five_percent_fails():
    assert runner.shield_actor_fraction(3, 959) == 3 / 959
    assert runner.shield_actor_fraction_gate(3, 959)
    assert not runner.shield_actor_fraction_gate(48, 959)


def test_actor_row_filter_is_reused_for_fresh_support():
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
    result = runner.support_gates(rows, "train", _records("train", runner.FORMAL_TRAIN_SEEDS),
                                  runner.FORMAL_TRAIN_SEEDS)
    assert result["metrics"]["rows"] == 3000
    assert result["gates"]["all_episode_shield_actor_fraction_within_inherited_limit"]
    assert result["gates"]["all"]


def test_episode_fraction_gate_fails_above_five_percent_without_changing_other_gates():
    rows = _passing_rows("dev", 1800, runner.FORMAL_DEV_SEEDS)
    records = _records("dev", runner.FORMAL_DEV_SEEDS, fraction_gate=False)
    result = runner.support_gates(rows, "dev", records, runner.FORMAL_DEV_SEEDS)
    assert not result["gates"]["all_episode_shield_actor_fraction_within_inherited_limit"]
    assert result["gates"]["schema_88d_finite"]
    assert result["gates"]["interface_class_minimums"]
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
    assert not smoke["GATE_REPAIR_ALLOWED"] and not smoke["PPO_ALLOWED"]
    assert not smoke["curriculum_promotion_allowed"]


def test_v1_binding_declaration_and_no_row_mixing():
    p = runner.protocol()
    assert p["bindings"]["v1_fail_manifest_sha256"] == (
        "3f376b68209dbd7a68db058b8344204138dae3ea107cd2678d2f4966312fd979"
    )
    assert runner.V1_FAIL_MANIFEST_SHA == p["bindings"]["v1_fail_manifest_sha256"]
    assert p["bindings"]["source_usage"] == "read_only_binding_no_rows_mixed_into_support"


def test_v1_episode_helper_reuse_changes_only_the_episode_gate(monkeypatch, tmp_path):
    def fake_v1_episode_record(*args, **kwargs):
        return {
            "counts": {"shield_actor": 48, "actor_decisions": 959},
            "gates": {"safety_all_zero": True, "shield_zero": False, "replay": True},
        }, []

    monkeypatch.setattr(runner._v1, "_episode_record", fake_v1_episode_record)
    summary, rows = runner._episode_record(
        None, split="train", seed=775001, scenario="flow6", output=tmp_path / "episode", max_steps=160,
    )
    assert rows == []
    assert summary["shield_actor_count"] == 48
    assert summary["actor_decisions"] == 959
    assert summary["shield_actor_fraction"] > 0.05
    assert not summary["shield_actor_fraction_within_inherited_limit"]
    assert "shield_zero" not in summary["gates"]
    assert summary["gates"]["safety_all_zero"]
    assert summary["gates"]["replay"]
    assert summary["teacher_replaced_behavior"] is False
    assert summary["v1_rows_mixed"] is False
