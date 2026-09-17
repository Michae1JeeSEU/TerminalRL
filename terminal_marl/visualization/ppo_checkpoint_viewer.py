"""Local PyVista viewer using the exact 88D evaluation request path."""
import argparse
import time
from pathlib import Path
import torch
from ..algo.joint_ppo_v4_2 import JointPPOV42
from ..core.checkpoints import load_checkpoint
from ..envs.parallel_env import TerminalParallelEnv
from ..training.run_v4_2_v2_1_i1_v2 import _sha
from ..training.run_v4_2_v2_1_student_qualification import scenario_config, protocol
from ..training.student_rollout_v4_2 import student_request
from .rollout_viewer import _attach_viewer, _pace_live_rollout
from .config import VisualizationConfig

CHECKPOINT_SHA = "965c5438dfc42f9898a4059b1225527bebd94f048ae9dfe6253cb49fc0519198"


def run(checkpoint, *, scenario="flow6", seed=690003, gui=True, playback_speed=10., max_steps=None):
    if _sha(Path(checkpoint)) != CHECKPOINT_SHA:
        raise ValueError("expected independently evaluated checkpoint_12.pt")
    config = scenario_config(scenario, protocol())
    policy = JointPPOV42(88)
    load_checkpoint(checkpoint, policy, config, load_optimizer=False)
    policy.model.eval()
    torch.manual_seed(seed)
    env = TerminalParallelEnv(config)
    observations, _ = env.reset(seed)
    observer = None
    try:
        if gui:
            _, observer = _attach_viewer(env, VisualizationConfig(enabled=True, frame_stride=1,
                playback_speed=playback_speed, show_trajectories=True, show_labels=True))
            if not observer.wait_ready():
                raise RuntimeError(f"viewer failed: {observer.error}")
        tick = time.perf_counter()
        for _ in range(max_steps or int(config.max_simulation_seconds / config.dt)):
            if observer and observer.stop_requested:
                break
            requested = student_request(policy, env, observations, deterministic=True)[3]
            observations, *_ = env.step(requested)
            if env.arrivals.exhausted() and not env.active and not env.admission.queue.items and not env.service_uavs:
                break
            tick = _pace_live_rollout(observer, config.dt, tick)
        return {"scenario": scenario, "seed": seed, "seconds": env.now,
                "completed": len(env.completed_uavs), "telemetry": dict(env.telemetry),
                "teacher_enabled": config.teacher_enabled, "observation_dim": policy.observation_dim}
    finally:
        if observer:
            observer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scenario", choices=("warm3", "flow3", "flow6"), default="flow6")
    parser.add_argument("--seed", type=int, default=690003)
    parser.add_argument("--playback-speed", type=float, default=10.)
    parser.add_argument("--gui", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-steps", type=int)
    a = parser.parse_args()
    torch.set_num_threads(1)
    print(run(a.checkpoint, scenario=a.scenario, seed=a.seed, gui=a.gui,
              playback_speed=a.playback_speed, max_steps=a.max_steps))
