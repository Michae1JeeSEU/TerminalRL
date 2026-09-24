from __future__ import annotations

import copy

import numpy as np
import pytest

from terminal_marl.core.config import SimConfig
from terminal_marl.core.low_level_joint_action_v4_2 import ACTION_SCHEMA_V4_2
from terminal_marl.core.uav import FlightPhase
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training.v4_1_joint_teacher import teacher_joint_expert_v4_1


def _config(*, final_mode: str = "continuous_inward_v1", priority_guard: bool = False) -> SimConfig:
    values = {
        "action_schema_version": ACTION_SCHEMA_V4_2,
        "teacher_enabled": True,
        "teacher_mode": "teacher_label_only",
        "scheduler_mode": "execution",
        "experiment_mode": "fixed_teacher_heavy",
        "reset_mode": "warm_start_rings",
        "warm_start_uavs": 3,
        "warm_start_steady_motion": True,
        "fixed_arrival_schedule": (0.0,) * 3,
        "total_requested_uavs": 3,
        "max_simulation_seconds": 2.0,
        "final_approach_geometry_mode": final_mode,
        "final_queue_capacity": 6,
        "lz_assignment_mode": "nearest_available",
    }
    if priority_guard:
        values.update({
            "emergency_landing_enabled": True,
            "emergency_priority_guard_enabled": True,
            "energy_model_mode": "phase_motion_v2",
        })
    return SimConfig(**values)


def _fixture(*, final_mode: str = "continuous_inward_v1", priority_guard: bool = False):
    env = TerminalParallelEnv(_config(final_mode=final_mode, priority_guard=priority_guard))
    observations, _ = env.reset(3)
    uav = env.active[int(next(iter(observations)))]
    uav.phase = FlightPhase.RING_INNER
    uav.committed_plan = None
    uav.safety_retry_after = env.now
    uav.energy_remaining = 0.8 * uav.energy_capacity
    return env, uav


def _uav_state(uav):
    return {
        "phase": uav.phase,
        "position": uav.position.copy(),
        "velocity": uav.velocity.copy(),
        "energy_remaining": float(uav.energy_remaining),
        "committed_plan": uav.committed_plan,
        "safety_retry_after": float(uav.safety_retry_after),
        "assigned_lz": uav.assigned_lz,
        "merge_reservation_id": uav.merge_reservation_id,
    }


def _lz_state(env):
    return [
        (zone.lz_id, zone.status, zone.reserved_by, zone.reservation_start,
         zone.expected_release_at, zone.occupied_by)
        for zone in env.lz_manager.zones
    ]


def test_continuous_teacher_tries_ranked_pads_and_records_selected_pad(monkeypatch):
    env, uav = _fixture()
    before_uav = _uav_state(uav)
    before_lz = _lz_state(env)
    before_reservations = copy.deepcopy(env.scheduler.reservations)
    calls: list[int | None] = []
    ranked = tuple(env.lz_manager.rank_available_zones(uav, env.now))
    assert len(ranked) >= 2

    def candidate_plan(_uav, _candidate, _target, _snapshot, lz_id):
        calls.append(lz_id)
        if lz_id == ranked[0]:
            return object(), object(), "mock_rejected_pad"
        return object(), object(), None

    monkeypatch.setattr(env, "_candidate_plan", candidate_plan)
    selected, audit = teacher_joint_expert_v4_1(env, uav, audit_catalogue=False)

    assert selected.kind == "RELEASE"
    assert calls == [ranked[0], ranked[1]]
    assert audit[0]["reason"] == "certified"
    assert audit[0]["final_lz_selected_id"] == ranked[1]
    assert audit[0]["final_lz_pairs_checked"] == [
        {"lz_id": ranked[0], "feasible": False, "reason": "mock_rejected_pad"},
        {"lz_id": ranked[1], "feasible": True, "reason": "certified"},
    ]
    assert _uav_state(uav)["phase"] == before_uav["phase"]
    np.testing.assert_array_equal(_uav_state(uav)["position"], before_uav["position"])
    np.testing.assert_array_equal(_uav_state(uav)["velocity"], before_uav["velocity"])
    assert _uav_state(uav)["energy_remaining"] == before_uav["energy_remaining"]
    assert _lz_state(env) == before_lz
    assert env.scheduler.reservations == before_reservations


def test_priority_guard_blocks_all_release_candidates_without_candidate_plan(monkeypatch):
    env, uav = _fixture(priority_guard=True)
    candidate_calls = []
    monkeypatch.setattr(env, "_emergency_lz_priority_blocks", lambda _uav: True)

    def candidate_plan(*_args):
        candidate_calls.append(True)
        raise AssertionError("priority guard must precede candidate_plan")

    monkeypatch.setattr(env, "_candidate_plan", candidate_plan)
    selected, audit = teacher_joint_expert_v4_1(env, uav, audit_catalogue=True)

    assert selected.index == 0
    release_rows = [row for row in audit if row["decoded"]["kind"] == "RELEASE"]
    wait_rows = [row for row in audit if row["decoded"]["kind"] == "WAIT"]
    assert len(release_rows) == 25 and len(wait_rows) == 1
    assert {row["reason"] for row in release_rows} == {"emergency_lz_priority"}
    assert wait_rows[0]["reason"] == "wait_legal"
    assert candidate_calls == []


def test_legacy_brake_teacher_uses_one_rule_pad_without_pad_expansion(monkeypatch):
    env, uav = _fixture(final_mode="brake_bezier_v1")
    choose_calls: list[int] = []
    candidate_calls: list[int | None] = []
    original_choose = env.lz_manager.choose_rule_zone

    def choose_rule_zone(_uav, now):
        chosen = original_choose(_uav, now)
        choose_calls.append(chosen)
        return chosen

    monkeypatch.setattr(env.lz_manager, "choose_rule_zone", choose_rule_zone)
    monkeypatch.setattr(
        env,
        "_candidate_plan",
        lambda _uav, _candidate, _target, _snapshot, lz_id: (
            candidate_calls.append(lz_id) or (object(), object(), None)
        ),
    )
    selected, audit = teacher_joint_expert_v4_1(env, uav, audit_catalogue=False)

    assert selected.kind == "RELEASE"
    assert len(choose_calls) == 1
    assert len(candidate_calls) == 1 and candidate_calls == choose_calls
    assert "final_lz_pairs_checked" not in audit[0]
    assert "final_lz_selected_id" not in audit[0]


@pytest.mark.parametrize("mask", ["committed", "retry", "future"])
def test_release_context_masks_committed_retry_and_future_reservation(mask):
    env, uav = _fixture()
    if mask == "committed":
        uav.committed_plan = object()
    elif mask == "retry":
        uav.safety_retry_after = env.now + 1.0
    else:
        env.future_release_reservations[uav.uid] = {"forecast": True}

    candidate_calls = []
    env._candidate_plan = lambda *_args: candidate_calls.append(True)
    selected, audit = teacher_joint_expert_v4_1(env, uav, audit_catalogue=True)

    assert selected.index == 0
    assert candidate_calls == []
    assert {row["reason"] for row in audit if row["decoded"]["kind"] == "RELEASE"} == {"release_context_masked"}


def test_zero_soc_is_runtime_reason_before_final_pad_search(monkeypatch):
    env, uav = _fixture()
    uav.energy_remaining = 0.0
    rank_calls = []
    candidate_calls = []
    monkeypatch.setattr(env.lz_manager, "rank_available_zones",
                        lambda *_args: rank_calls.append(True) or [])
    monkeypatch.setattr(env, "_candidate_plan",
                        lambda *_args: candidate_calls.append(True))

    selected, audit = teacher_joint_expert_v4_1(env, uav, audit_catalogue=True)

    assert selected.index == 0
    assert {row["reason"] for row in audit if row["decoded"]["kind"] == "RELEASE"} == {"soc"}
    assert rank_calls == []
    assert candidate_calls == []
