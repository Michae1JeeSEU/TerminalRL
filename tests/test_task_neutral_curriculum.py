from dataclasses import asdict

import numpy as np
import pytest

from terminal_marl.core.arrival_process import ArrivalProcess
from terminal_marl.core.config import SimConfig
from terminal_marl.core.observations import (
    A5_OBSERVATION_SCHEMA,
    LOW_OBS_V2,
    LOW_OBS_V2_1,
    LOW_OBS_V2_3,
)
from terminal_marl.core.priority import PriorityEngine
from terminal_marl.core.rewards import low_level_urgency_penalty
from terminal_marl.core.uav import (
    ArrivalRequest,
    MISSION_SERVICE_SECONDS,
    MissionType,
    QueuedArrivalState,
)
from terminal_marl.envs.parallel_env import TerminalParallelEnv


def _arrival_config(task_priority_enabled: bool) -> SimConfig:
    count = 12
    return SimConfig(
        arrival_model="finite_cohort",
        total_requested_uavs=count,
        fixed_arrival_schedule=tuple(float(index) for index in range(count)),
        task_priority_enabled=task_priority_enabled,
    )


def test_task_priority_flag_defaults_legacy_and_is_serialized() -> None:
    assert SimConfig().task_priority_enabled is True
    payload = asdict(SimConfig(task_priority_enabled=False))
    assert payload["task_priority_enabled"] is False


def test_task_priority_flag_requires_bool() -> None:
    with pytest.raises(ValueError, match="task_priority_enabled must be bool"):
        SimConfig(task_priority_enabled="false")


def test_neutral_arrivals_preserve_seeded_physical_stream_and_use_base_one() -> None:
    legacy = ArrivalProcess(_arrival_config(True), seed=20260928).requests_until(100.0)
    neutral = ArrivalProcess(_arrival_config(False), seed=20260928).requests_until(100.0)

    assert len(legacy) == len(neutral) == 12
    assert [
        (request.request_time, request.gate_id, request.mission_type,
         request.initial_soc, request.energy_capacity)
        for request in legacy
    ] == [
        (request.request_time, request.gate_id, request.mission_type,
         request.initial_soc, request.energy_capacity)
        for request in neutral
    ]
    assert [request.base_priority for request in neutral] == [1] * len(neutral)
    assert all(request.base_priority in {1, 3, 5} for request in legacy)
    assert [
        MISSION_SERVICE_SECONDS[request.mission_type] for request in legacy
    ] == [
        MISSION_SERVICE_SECONDS[request.mission_type] for request in neutral
    ]

    emergency_config = dict(
        arrival_model="finite_cohort",
        total_requested_uavs=1,
        fixed_arrival_schedule=(0.0,),
        mission_probabilities=(0.0, 0.0, 1.0, 0.0),
    )
    emergency_legacy = ArrivalProcess(SimConfig(**emergency_config), seed=19).requests_until(0.0)[0]
    emergency_neutral = ArrivalProcess(
        SimConfig(**emergency_config, task_priority_enabled=False), seed=19
    ).requests_until(0.0)[0]
    assert emergency_legacy.mission_type is MissionType.EMERGENCY
    assert emergency_neutral.mission_type is MissionType.EMERGENCY
    assert emergency_legacy.base_priority == 5
    assert emergency_neutral.base_priority == 1
    assert MISSION_SERVICE_SECONDS[emergency_neutral.mission_type] == pytest.approx(10.0)


def _queued(config: SimConfig, *, base_priority: int, soc: float, wait: float) -> QueuedArrivalState:
    request = ArrivalRequest(
        arrival_id=0,
        request_time=0.0,
        gate_id=0,
        mission_type=MissionType.CARGO,
        base_priority=base_priority,
        energy_capacity=config.energy_capacity,
        initial_soc=soc,
    )
    queued = QueuedArrivalState(request, config.energy_capacity * soc)
    queued.external_wait_time = wait
    return queued


def test_neutral_dynamic_priority_ignores_base_but_keeps_soc_and_wait_terms() -> None:
    config = SimConfig(task_priority_enabled=False)
    priority = PriorityEngine(config)

    neutral_base_one = _queued(config, base_priority=1, soc=0.5, wait=0.0)
    neutral_base_five = _queued(config, base_priority=5, soc=0.5, wait=0.0)
    legacy_priority = PriorityEngine(SimConfig())
    legacy_base_one = _queued(config, base_priority=1, soc=0.5, wait=0.0)
    legacy_base_five = _queued(config, base_priority=5, soc=0.5, wait=0.0)

    assert priority.compute(neutral_base_one) == pytest.approx(priority.compute(neutral_base_five))
    assert legacy_priority.compute(legacy_base_one) != pytest.approx(legacy_priority.compute(legacy_base_five))
    assert priority.compute(_queued(config, base_priority=1, soc=0.5, wait=0.0)) < priority.compute(
        _queued(config, base_priority=1, soc=0.5, wait=1200.0)
    )
    assert priority.compute(_queued(config, base_priority=1, soc=0.2, wait=0.0)) > priority.compute(
        _queued(config, base_priority=1, soc=0.5, wait=0.0)
    )


def test_environment_neutral_reward_removes_priority_urgency_and_keeps_soc_urgency() -> None:
    config = SimConfig(
        task_priority_enabled=False,
        fixed_arrival_schedule=(0.0,),
        total_requested_uavs=1,
        initial_soc_mode="fixed",
        initial_soc=0.5,
        low_level_priority_time_weight=0.6,
        low_level_soc_urgency_time_weight=0.8,
        low_level_soc_urgency_threshold=0.65,
        max_simulation_seconds=20.0,
    )
    env = TerminalParallelEnv(config)
    env.reset(7)
    env.step({})
    uav = next(iter(env.active.values()))
    soc_before = uav.soc
    uav.base_priority = 5
    _, rewards, _, _, info = env.step({})

    uid = str(uav.uid)
    components = info["reward_components"][uid]
    expected_soc_only = low_level_urgency_penalty(
        base_time_penalty=config.low_level_time_penalty_per_second,
        dt=config.dt,
        base_priority=uav.base_priority,
        soc=soc_before,
        priority_weight=0.0,
        soc_weight=config.low_level_soc_urgency_time_weight,
        soc_urgency_threshold=config.low_level_soc_urgency_threshold,
        soc_reserve=config.soc_reserve,
    )
    full_priority = low_level_urgency_penalty(
        base_time_penalty=config.low_level_time_penalty_per_second,
        dt=config.dt,
        base_priority=uav.base_priority,
        soc=soc_before,
        priority_weight=config.low_level_priority_time_weight,
        soc_weight=config.low_level_soc_urgency_time_weight,
        soc_urgency_threshold=config.low_level_soc_urgency_threshold,
        soc_reserve=config.soc_reserve,
    )
    assert components["urgency"] == pytest.approx(expected_soc_only)
    assert components["urgency"] != pytest.approx(full_priority)
    assert rewards[uid] == pytest.approx(components["total"])


def _observation_config(schema: str, *, task_priority_enabled: bool) -> SimConfig:
    values = dict(
        observation_schema_version=schema,
        task_priority_enabled=task_priority_enabled,
        reset_mode="warm_start_rings",
        warm_start_uavs=1,
        fixed_arrival_schedule=(0.0,),
        total_requested_uavs=1,
        initial_soc_mode="fixed",
        initial_soc=0.8,
        mission_probabilities=(0.0, 1.0, 0.0, 0.0),
        max_active_uavs=2,
    )
    if schema == "phaseA_low_obs_v2_3_source_bound_88d":
        values.update(
            experiment_mode="fixed_low_student",
            emergency_landing_enabled=True,
            neighbor_comm_range_m=320.0,
        )
    return SimConfig(**values)


@pytest.mark.parametrize(
    ("schema", "mission_feature"),
    (
        ("phaseA_low_obs_v1_41d", A5_OBSERVATION_SCHEMA.slices["mission"]),
        ("phaseA_low_obs_v2_topology_87d", LOW_OBS_V2.slices["mission"]),
        ("phaseA_low_obs_v2_1_retry_state", LOW_OBS_V2_1.slices["mission"]),
        ("phaseA_low_obs_v2_3_source_bound_88d", LOW_OBS_V2_3.slices["self_mission_type"]),
    ),
)
def test_neutral_observation_mission_token_preserves_schema_and_service_metadata(
    schema: str, mission_feature: slice,
) -> None:
    env = TerminalParallelEnv(_observation_config(schema, task_priority_enabled=False))
    observations, _ = env.reset(41)
    uav = next(iter(env.active.values()))
    assert uav.mission_type is MissionType.CARGO
    assert MISSION_SERVICE_SECONDS[uav.mission_type] == pytest.approx(120.0)
    expected_dimension = {
        "phaseA_low_obs_v1_41d": 41,
        "phaseA_low_obs_v2_topology_87d": 87,
        "phaseA_low_obs_v2_1_retry_state": 88,
        "phaseA_low_obs_v2_3_source_bound_88d": 88,
    }[schema]
    assert observations[str(uav.uid)].shape == (expected_dimension,)
    assert np.array_equal(observations[str(uav.uid)][mission_feature], np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32))


def test_legacy_observation_mission_token_remains_physical_mission() -> None:
    env = TerminalParallelEnv(_observation_config("phaseA_low_obs_v2_topology_87d", task_priority_enabled=True))
    observations, _ = env.reset(41)
    uav = next(iter(env.active.values()))
    assert uav.mission_type is MissionType.CARGO
    assert np.array_equal(
        observations[str(uav.uid)][LOW_OBS_V2.slices["mission"]],
        np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    )
