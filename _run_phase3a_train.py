"""Run Phase 3a PPO training."""
from src.config import EnvConfig
from src.train_phase3a import train_phase3a

config = EnvConfig()
config.num_uavs = 0
config.max_uavs = 25
config.sim.max_steps = 400
config.uav_inflow_rate = 0.0

model, config = train_phase3a(config, n_background=10, total_timesteps=200_000)
print("Training complete. Model saved to logs/ppo_phase3a")
