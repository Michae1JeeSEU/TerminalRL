from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from terminal_marl.core.config import RingRole, SimConfig
from terminal_marl.core.uav import (
    ArrivalRequest,
    MissionType,
    QueuedArrivalState,
    TerminalOutcome,
)
from terminal_marl.envs.high_level_wrapper import (
    HIGH_LEVEL_COST_DEFINITION,
    DynamicHighLevelWrapper,
)


class _EmptyLowLevel:
    def act(self, observations):
        del observations
        return {}


def _config(**overrides) -> SimConfig:
    values = {
        "experiment_mode": "dynamic_rule",
        "high_level_enabled": True,
        "arrival_model": "finite_cohort",
        "total_requested_uavs": 1,
        "fixed_arrival_schedule": (10000.0,),
        "max_simulation_seconds": 100.0,
        "high_level_decision_interval_seconds": 1.0,
        "geometry_min_dwell_seconds": 600.0,
        "geometry_transition_seconds": 0.0,
        "dt": 1.0,
    }
    values.update(overrides)
    return SimConfig(**values)


def _zero_action() -> np.ndarray:
    return np.zeros(10, dtype=np.float32)


def _stub_control(wrapper: DynamicHighLevelWrapper) -> None:
    """Keep the real wrapper accounting while freezing control side effects."""
    capacity = tuple(wrapper.base.airspace.capacity(role) for role in RingRole)

    def coordinate(action, *, geometry_gate, **kwargs):
        del geometry_gate, kwargs
        action_tuple = tuple(float(value) for value in np.asarray(action))
        update = SimpleNamespace(
            structural_mask=(False,) * 6,
            requested_radius_delta=(0.0,) * 3,
            projected_radius_delta=(0.0,) * 3,
            applied_radius_delta=(0.0,) * 3,
            capacity_before=capacity,
            capacity_after=capacity,
            structural_reason_codes=(),
            structural_applied=False,
        )
        return SimpleNamespace(
            update=update,
            requested_action=action_tuple,
            applied_action=action_tuple,
            outcome="no_op",
            execution_outcome="NO_OP",
            request_id=None,
            original_request_time=None,
            actual_application_time=None,
            application_delay_seconds=None,
            operational_effective=False,
            structural_effective=False,
            capacity_effective=False,
            reason_code="no_op",
            repair_speed_residuals={},
            repair_magnitude=0.0,
            projection_magnitude=0.0,
            deferred_until=None,
            safety_reason=None,
        )

    wrapper.coordinator.coordinate = coordinate
    wrapper.coordinator.advance = lambda: None
    wrapper.coordinator.apply_speed_repairs = lambda actions: actions


def _accounting_wrapper(*, interval: float = 1.0) -> DynamicHighLevelWrapper:
    wrapper = DynamicHighLevelWrapper(
        _config(high_level_decision_interval_seconds=interval),
        low_level=_EmptyLowLevel(),
    )
    wrapper.reset(seed=0)
    wrapper.base._observations = lambda: {}
    wrapper._done = lambda: False
    _stub_control(wrapper)
    return wrapper


def _request(config: SimConfig, arrival_id: int, mission: MissionType) -> ArrivalRequest:
    return ArrivalRequest(
        arrival_id=arrival_id,
        request_time=0.0,
        gate_id=0,
        mission_type=mission,
        base_priority=1,
        energy_capacity=config.energy_capacity,
        initial_soc=1.0,
    )


def _state(arrival_id: int, energy: float, phase: str) -> SimpleNamespace:
    return SimpleNamespace(
        arrival_id=arrival_id,
        energy_consumed=float(energy),
        phase=SimpleNamespace(value=phase),
    )


def _install_two_tick_timeline(wrapper: DynamicHighLevelWrapper) -> None:
    base = wrapper.base
    request = _request(wrapper.config, 1, MissionType.STANDARD)
    queued = QueuedArrivalState(request, request.energy_capacity)
    base.request_times = {1: 0.0}
    base.resolution_times = {}
    base.ledger.request(1)
    base.admission.queue.items = [queued]
    base.active = {}
    base.service_uavs = {}
    base.completed_uavs = []
    state: dict[str, SimpleNamespace] = {}

    def step(_actions=None, proposal_shortlists=None):
        del _actions, proposal_shortlists
        tick = int(round(base.now / base.config.dt))
        base.external_queue_uav_seconds += len(base.admission.queue.items) * base.config.dt
        if tick == 0:
            queued.energy_consumed = 2.0
            base.admission.queue.items.clear()
            state["uav"] = _state(1, 2.0, "ring_outer")
            base.active = {1: state["uav"]}
        elif tick == 1:
            state["uav"].energy_consumed = 3.0
            base.active.clear()
            base.completed_uavs.append(state["uav"])
            base.ledger.outcomes[1] = TerminalOutcome.COMPLETED
            base.resolution_times[1] = base.now + base.config.dt
            base.energy_outcomes.append(
                {"arrival_id": 1, "energy_consumed": 3.0, "outcome": "completed"}
            )
        else:
            raise AssertionError(f"unexpected deterministic tick {tick}")
        base.now += base.config.dt
        base.step_count += 1
        return {}, {}, {}, {}, {}

    base.step = step


def _run_two_tick_timeline(interval: float) -> dict[str, float]:
    wrapper = _accounting_wrapper(interval=interval)
    _install_two_tick_timeline(wrapper)
    calls = 2 if interval == 1.0 else 1
    infos = [wrapper.step(_zero_action())[4] for _ in range(calls)]
    return {
        "queue": sum(info["raw_cost_integrals"]["external_queue_uav_seconds"] for info in infos),
        "delay": sum(info["raw_cost_integrals"]["unresolved_request_seconds"] for info in infos),
        "energy": wrapper._energy_total(),
        "footprint": sum(info["raw_cost_integrals"]["footprint_seconds"] for info in infos),
        "service_completion": sum(
            info["reward_components"]["service_completion"] / 8.0 for info in infos
        ),
    }


def test_cost_metadata_defines_overlap_and_experimental_units():
    wrapper = _accounting_wrapper()
    _install_two_tick_timeline(wrapper)
    info = wrapper.step(_zero_action())[4]
    metadata = info["cost_definition"]

    assert metadata == HIGH_LEVEL_COST_DEFINITION
    assert "every requested arrival" in metadata["unresolved_request_seconds"]
    assert "external queue" in metadata["unresolved_request_seconds"]
    assert "overlaps unresolved_request_seconds" in metadata["external_queue_uav_seconds"]
    assert "not calibrated" in metadata["coefficient_units"]
    assert info["high_action"]["cost_definition"] == metadata


def test_macro_interval_split_conserves_integrated_costs_and_completion():
    split = _run_two_tick_timeline(1.0)
    unsplit = _run_two_tick_timeline(2.0)

    for key in ("queue", "delay", "energy", "footprint", "service_completion"):
        assert split[key] == pytest.approx(unsplit[key])
    assert split == pytest.approx(
        {
            "queue": 1.0,
            "delay": 2.0,
            "energy": 3.0,
            "footprint": 2.0,
            "service_completion": 1.0,
        }
    )


def _install_lifecycle_timeline(wrapper: DynamicHighLevelWrapper) -> None:
    base = wrapper.base
    requests = [
        _request(wrapper.config, 1, MissionType.STANDARD),
        _request(wrapper.config, 2, MissionType.EMERGENCY),
        _request(wrapper.config, 3, MissionType.STANDARD),
        _request(wrapper.config, 4, MissionType.CARGO),
    ]
    queued = [QueuedArrivalState(request, request.energy_capacity) for request in requests]
    base.request_times = {request.arrival_id: 0.0 for request in requests}
    base.resolution_times = {}
    for request in requests:
        base.ledger.request(request.arrival_id)
    base.admission.queue.items = queued[:]
    base.active = {}
    base.service_uavs = {}
    base.completed_uavs = []
    states: dict[int, SimpleNamespace] = {}

    def resolve(arrival_id: int, outcome: TerminalOutcome, energy: float) -> None:
        base.ledger.outcomes[arrival_id] = outcome
        base.resolution_times[arrival_id] = base.now + base.config.dt
        base.energy_outcomes.append(
            {
                "arrival_id": arrival_id,
                "energy_consumed": energy,
                "outcome": outcome.value,
            }
        )

    def step(_actions=None, proposal_shortlists=None):
        del _actions, proposal_shortlists
        tick = int(round(base.now / base.config.dt))
        base.external_queue_uav_seconds += len(base.admission.queue.items) * base.config.dt
        if tick == 0:
            for item, energy in zip(queued[2:], (1.0, 1.0)):
                item.energy_consumed = energy
            for item, energy in zip(queued[:2], (4.0, 4.0)):
                item.energy_consumed = energy
            base.admission.queue.items = queued[2:]
            states[1] = _state(1, 4.0, "ring_outer")
            states[2] = _state(2, 4.0, "ring_outer")
            base.active = {1: states[1], 2: states[2]}
        elif tick == 1:
            states[1].energy_consumed = 6.0
            states[2].energy_consumed = 7.0
            queued[2].energy_consumed = 2.0
            queued[3].energy_consumed = 2.0
            states[1].phase = SimpleNamespace(value="service")
            states[2].phase = SimpleNamespace(value="service")
            base.active.clear()
            base.service_uavs = {1: states[1], 2: states[2]}
        elif tick == 2:
            states[1].energy_consumed = 7.0
            states[2].energy_consumed = 9.0
            queued[2].energy_consumed = 3.0
            queued[3].energy_consumed = 2.0
            base.service_uavs.clear()
            base.admission.queue.items.clear()
            states[1].phase = SimpleNamespace(value="completed")
            states[2].phase = SimpleNamespace(value="completed")
            base.completed_uavs.extend((states[1], states[2]))
            resolve(1, TerminalOutcome.COMPLETED, 7.0)
            resolve(2, TerminalOutcome.EMERGENCY_LANDED, 9.0)
            resolve(3, TerminalOutcome.FAILED_ENERGY, 3.0)
            resolve(4, TerminalOutcome.DIVERTED_CAPACITY, 2.0)
        else:
            raise AssertionError(f"unexpected deterministic tick {tick}")
        base.now += base.config.dt
        base.step_count += 1
        return {}, {}, {}, {}, {}

    base.step = step


def test_queue_air_service_failure_and_diversion_keep_wait_and_energy_history():
    wrapper = _accounting_wrapper(interval=3.0)
    _install_lifecycle_timeline(wrapper)
    info = wrapper.step(_zero_action())[4]

    assert info["raw_cost_integrals"]["external_queue_uav_seconds"] == pytest.approx(8.0)
    assert info["raw_cost_integrals"]["unresolved_request_seconds"] == pytest.approx(12.0)
    assert wrapper._energy_ledger == {1: 7.0, 2: 9.0, 3: 3.0, 4: 2.0}
    assert wrapper._energy_total() == pytest.approx(21.0)
    assert info["reward_components"]["service_completion"] == pytest.approx(8.0)
    assert info["outcomes"] == {
        1: TerminalOutcome.COMPLETED,
        2: TerminalOutcome.EMERGENCY_LANDED,
        3: TerminalOutcome.FAILED_ENERGY,
        4: TerminalOutcome.DIVERTED_CAPACITY,
    }


def test_end_tick_resolution_and_post_resolution_energy_are_counted_once():
    wrapper = _accounting_wrapper()
    base = wrapper.base
    request = _request(wrapper.config, 9, MissionType.STANDARD)
    queued = QueuedArrivalState(request, request.energy_capacity)
    base.request_times = {9: 0.0}
    base.resolution_times = {}
    base.ledger.request(9)
    base.admission.queue.items = [queued]
    base.active = {}
    base.service_uavs = {}
    base.completed_uavs = []
    resolved = {"done": False}

    def step(_actions=None, proposal_shortlists=None):
        del _actions, proposal_shortlists
        base.external_queue_uav_seconds += len(base.admission.queue.items) * base.config.dt
        if not resolved["done"]:
            queued.energy_consumed = 5.0
            base.admission.queue.items.clear()
            state = _state(9, 5.0, "completed")
            base.completed_uavs.append(state)
            base.ledger.outcomes[9] = TerminalOutcome.COMPLETED
            base.resolution_times[9] = base.now + base.config.dt
            base.energy_outcomes.append(
                {"arrival_id": 9, "energy_consumed": 5.0, "outcome": "completed"}
            )
            resolved["done"] = True
        base.now += base.config.dt
        base.step_count += 1
        return {}, {}, {}, {}, {}

    base.step = step
    first = wrapper.step(_zero_action())[4]
    second = wrapper.step(_zero_action())[4]

    assert first["raw_cost_integrals"]["unresolved_request_seconds"] == pytest.approx(1.0)
    assert second["raw_cost_integrals"]["unresolved_request_seconds"] == pytest.approx(0.0)
    assert first["reward_components"]["energy"] == pytest.approx(-0.0025)
    assert second["reward_components"]["energy"] == pytest.approx(0.0)
    assert wrapper._energy_total() == pytest.approx(5.0)


def test_real_sustained_low_shrink_uses_coordinator_and_respects_transition_limits():
    config = _config(
        max_simulation_seconds=620.0,
        high_level_decision_interval_seconds=1.0,
        geometry_min_dwell_seconds=600.0,
        geometry_transition_seconds=5.0,
    )
    wrapper = DynamicHighLevelWrapper(config)
    wrapper.reset(seed=0)
    zero = _zero_action()
    for _ in range(600):
        wrapper.step(zero)

    assert wrapper.geometry_gate_reason() == "sustained_low_demand"
    shrink = np.asarray((-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), dtype=np.float32)
    _, _, _, _, first = wrapper.step(shrink)
    event = first["high_action"]
    assert event["geometry_gate_reason"] == "sustained_low_demand"
    assert any(event["structural_mask"])
    assert event["transition_active"]
    assert all(
        abs(delta) <= config.max_radius_change_per_update + 1e-9
        for delta in event["applied_radius_delta"]
    )

    _, _, _, _, second = wrapper.step(shrink)
    assert second["high_action"]["transition_active"]
    assert second["high_action"]["structural_mask"] == [False] * 6
