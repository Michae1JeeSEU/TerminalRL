from types import SimpleNamespace

import numpy as np
import pytest

from terminal_marl.core.config import SimConfig, RingRole
from terminal_marl.core.uav import ArrivalRequest, MissionType, QueuedArrivalState, TerminalOutcome
from terminal_marl.envs.high_level_wrapper import DynamicHighLevelWrapper, FrozenLowLevelPolicy


def _config(**overrides) -> SimConfig:
    values = {
        "experiment_mode": "dynamic_rule",
        "high_level_enabled": True,
        "arrival_model": "finite_cohort",
        "total_requested_uavs": 1,
        "fixed_arrival_schedule": (10000.0,),
        "max_simulation_seconds": 20000.0,
        "high_level_decision_interval_seconds": 1.0,
        "geometry_min_dwell_seconds": 600.0,
        "geometry_transition_seconds": 0.0,
    }
    values.update(overrides)
    return SimConfig(**values)


def _wrapper(**overrides) -> DynamicHighLevelWrapper:
    wrapper = DynamicHighLevelWrapper(_config(**overrides))
    wrapper.reset(seed=0)
    return wrapper


def _zero_action() -> np.ndarray:
    return np.zeros(10, dtype=np.float32)


def _context(*, queue=0, occupancy=(0.0, 0.0, 0.0), final=0.0, lz=0.0) -> dict:
    return {
        "external_queue": queue,
        "occupancy_ratio": list(occupancy),
        "final_pressure": final,
        "lz_utilization": lz,
    }


def _queued(config: SimConfig, arrival_id: int) -> QueuedArrivalState:
    request = ArrivalRequest(
        arrival_id,
        0.0,
        0,
        MissionType.STANDARD,
        1,
        config.energy_capacity,
        1.0,
    )
    return QueuedArrivalState(request, request.energy_capacity)


def test_sustained_low_demand_opens_geometry_gate_but_dynamic_dwell_still_applies():
    wrapper = _wrapper()
    for _ in range(600):
        wrapper.step(_zero_action())

    assert wrapper.geometry_gate_reason() == "sustained_low_demand"
    assert wrapper.geometry_event_gate()

    shrink = np.asarray((-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), dtype=np.float32)
    _, _, _, _, first = wrapper.step(shrink)
    assert first["high_action"]["geometry_gate_reason"] == "sustained_low_demand"
    assert any(first["high_action"]["structural_mask"])

    _, _, _, _, second = wrapper.step(shrink)
    assert second["high_action"]["geometry_gate_reason"] == "sustained_low_demand"
    assert second["high_action"]["structural_mask"] == [False] * 6


def test_short_low_demand_fluctuation_does_not_open_gate(monkeypatch):
    wrapper = _wrapper(geometry_min_dwell_seconds=10.0)
    monkeypatch.setattr(wrapper, "_context", lambda: _context())
    wrapper.base.now = 0.0
    wrapper._update_low_demand_state()

    monkeypatch.setattr(wrapper, "_context", lambda: _context(queue=1))
    wrapper.base.now = 1.0
    wrapper._update_low_demand_state()
    assert wrapper._low_demand_since is None

    monkeypatch.setattr(wrapper, "_context", lambda: _context())
    wrapper.base.now = 2.0
    wrapper._update_low_demand_state()
    wrapper.base.now = 11.0
    assert wrapper.geometry_gate_reason() == "none"
    wrapper.base.now = 12.0
    assert wrapper.geometry_gate_reason() == "sustained_low_demand"


def test_external_queue_cost_uses_tick_uav_seconds():
    empty = _wrapper()
    empty.base.admission.step = lambda *args, **kwargs: ([], [])
    _, _, _, _, empty_info = empty.step(_zero_action())

    queued = _wrapper()
    queued.base.admission.step = lambda *args, **kwargs: ([], [])
    queued.base.admission.queue.items.extend([_queued(queued.config, 1), _queued(queued.config, 2)])
    _, _, _, _, queued_info = queued.step(_zero_action())

    assert empty_info["raw_cost_integrals"]["external_queue_uav_seconds"] == pytest.approx(0.0)
    assert queued_info["raw_cost_integrals"]["external_queue_uav_seconds"] == pytest.approx(2.0)
    assert queued_info["reward_components"]["external_queue"] == pytest.approx(-0.1)
    assert queued_info["reward_schema"] == "high_reward_v2_time_integrated"
    assert queued_info["footprint_definition"] == "normalized_sum_ring_disk_area_proxy"
    assert queued_info["high_action"]["footprint_definition"] == "normalized_sum_ring_disk_area_proxy"


def test_queue_history_is_charged_when_macro_endpoint_queue_matches(monkeypatch):
    config_overrides = {
        "high_level_decision_interval_seconds": 2.0,
        "dt": 1.0,
        "fixed_arrival_schedule": (10000.0,),
        "max_simulation_seconds": 20000.0,
    }
    empty = _wrapper(**config_overrides)
    empty.base.admission.step = lambda *args, **kwargs: ([], [])
    _, _, _, _, empty_info = empty.step(_zero_action())

    queued = _wrapper(**config_overrides)
    queued.base.admission.queue.items.append(_queued(queued.config, 1))
    calls = {"count": 0}

    def keep_then_clear(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            queued.base.admission.queue.items.clear()
        return [], []

    queued.base.admission.step = keep_then_clear
    _, _, _, _, queued_info = queued.step(_zero_action())

    assert len(empty.base.admission.queue.items) == 0
    assert len(queued.base.admission.queue.items) == 0
    assert empty_info["raw_cost_integrals"]["external_queue_uav_seconds"] == pytest.approx(0.0)
    assert queued_info["raw_cost_integrals"]["external_queue_uav_seconds"] == pytest.approx(2.0)
    assert empty_info["reward_components"]["external_queue"] == pytest.approx(0.0)
    assert queued_info["reward_components"]["external_queue"] == pytest.approx(-0.1)


def test_completed_request_delay_is_charged_once_and_unresolved_request_is_integrated():
    wrapper = _wrapper()
    wrapper.base.request_times = {1: 0.0, 2: 0.0}
    wrapper.base.resolution_times = {1: 1.0}
    wrapper.base.ledger.request(1)
    wrapper.base.ledger.request(2)
    wrapper.base.ledger.outcomes[1] = TerminalOutcome.COMPLETED
    wrapper.base.completed_uavs = [
        SimpleNamespace(arrival_id=1, completion_time=1.0, request_time=0.0, energy_consumed=0.0)
    ]

    _, _, _, _, first = wrapper.step(_zero_action())
    _, _, _, _, second = wrapper.step(_zero_action())

    assert first["reward_components"]["service_completion"] == pytest.approx(8.0)
    assert first["raw_cost_integrals"]["unresolved_request_seconds"] == pytest.approx(2.0)
    assert second["reward_components"]["service_completion"] == pytest.approx(0.0)
    assert second["raw_cost_integrals"]["unresolved_request_seconds"] == pytest.approx(1.0)


def test_energy_ledger_keeps_removed_objects_and_deduplicates_queue_to_active():
    wrapper = _wrapper()
    removed = SimpleNamespace(arrival_id=41, energy_consumed=2.0)
    wrapper.base.active = {41: removed}
    previous_refs = wrapper._energy_object_refs()
    wrapper.base.active.clear()
    removed.energy_consumed = 5.0
    wrapper.base.energy_outcomes.append({"arrival_id": 41, "energy_consumed": 3.0})
    wrapper._observe_energy(previous_refs)
    assert wrapper._energy_ledger[41] == pytest.approx(5.0)
    assert wrapper._energy_total() == pytest.approx(5.0)

    queued = _queued(wrapper.config, 42)
    queued.energy_consumed = 4.0
    wrapper.base.admission.queue.items = [queued]
    wrapper.base.active = {42: SimpleNamespace(arrival_id=42, energy_consumed=4.0)}
    wrapper._observe_energy([])
    assert wrapper._energy_ledger[42] == pytest.approx(4.0)

    wrapper.reset(seed=0)
    assert wrapper._energy_ledger == {}


def test_v4_2_frozen_adapter_fails_before_checkpoint_loading():
    config = _config(action_schema_version="low_action_schema_v4_2_hierarchical")
    with pytest.raises(ValueError, match=r"V4\.2.*gate\+K3"):
        FrozenLowLevelPolicy("missing-v4-2-checkpoint.pt", config)
