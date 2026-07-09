import numpy as np

from terminal_marl import ExperimentMode, SimConfig, TerminalParallelEnv


def test_environment_spawns_runtime_uav_with_schedule_fields() -> None:
    config = SimConfig(
        arrival_rate=1.0,
        arrival_model="deterministic_interval",
        max_active_uavs=8,
        safe_distance=1.0,
        max_steps=4,
    )
    env = TerminalParallelEnv(config, experiment_mode=ExperimentMode.DYNAMIC_RULE)
    env.reset(seed=4)
    initial = len(env.aircraft)
    for _ in range(2):
        actions = {agent: np.zeros(4, dtype=np.float32) for agent in env.agents}
        env.step(actions)
    assert len(env.aircraft) > initial
    newest = env.aircraft[-1]
    assert newest.spawn_time > 0
    assert newest.planned_nominal_landing_time > newest.spawn_time
    assert f"uav_{newest.uid}" in env.possible_agents
