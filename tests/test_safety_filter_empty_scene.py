from terminal_marl.core.config import SimConfig
from terminal_marl.envs.parallel_env import TerminalParallelEnv


def test_empty_scene_does_not_create_safety_overrides() -> None:
    config = SimConfig(
        experiment_mode="dynamic_hrl",
        action_mode="intention_residual",
        initial_aircraft_count=0,
        formation_rows=0,
        formation_cols=0,
        arrival_rate=0.0,
        max_steps=3,
    )
    env = TerminalParallelEnv(config)
    env.reset(seed=0)
    before = env.safety_override_count
    env.step({})
    assert env.safety_override_count - before == 0
    assert not env.last_release_approval
    assert not env.last_landing_approval

