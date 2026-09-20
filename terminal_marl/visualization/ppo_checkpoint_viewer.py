"""Local PyVista viewer using the exact 88D evaluation request path.

The legacy ``checkpoint12`` profile remains available.  The two P2 profiles
recreate the frozen-geometry natural-flow setup and, importantly, pass the
same K=3 runtime proposal shortlist used by P2 qualification/evaluation into
``TerminalParallelEnv.step``.
"""
import argparse
import time
from dataclasses import replace
from pathlib import Path

import torch

from ..algo.joint_ppo_v4_2 import JointPPOV42
from ..core.checkpoints import load_checkpoint
from ..envs.parallel_env import TerminalParallelEnv
from ..training import evaluate_v4_2_p2_gate_effectiveness_v1 as p2_effectiveness
from ..training.evaluate_v4_2_p2_runtime_shortlist_heldout_v1 import (
    _load_trained as load_p2_trained,
    protocol as p2_heldout_protocol,
)
from ..training.run_v4_2_ppo_p2_resume_v4 import p2_config
from ..training.run_v4_2_v2_1_i1_v2 import _sha
from ..training.run_v4_2_v2_1_student_qualification import scenario_config, protocol
from ..training.student_rollout_v4_2 import _student_request_details
from .rollout_viewer import _attach_viewer, _pace_live_rollout
from .config import VisualizationConfig

CHECKPOINT12_SHA = "965c5438dfc42f9898a4059b1225527bebd94f048ae9dfe6253cb49fc0519198"
P2_INITIAL_K3_SHA = "4b8be66c100aaaeb5b5699dcce68a96dbe1237964e4c308e7b23b95c62fb8c8c"
P2_TRAINED_K3_SHA = "714d141f7b472a17321e95e9c4c5574fdc3bcaa0c2911494f67b5b1a9f27592e"
# Backward-compatible public name used by the checkpoint-12 evidence note.
CHECKPOINT_SHA = CHECKPOINT12_SHA

PROFILES = ("checkpoint12", "p2_initial_k3", "p2_trained_k3")
P2_SCENARIOS = ("flow6", "flow8", "flow12")
P2_DENSE_SCENARIO = "p2_dense24"


def _p2_dense_scene_config():
    """Out-of-distribution visual stress scene for the fixed P2 controller.

    Twelve separation-safe aircraft start in steady flight across all three
    rings (canonical 6/4/2 allocation).  Twelve more requests then arrive at
    15-second intervals.  This is deliberately denser than P2 training and is
    diagnostic only; it must not be reported as P2 qualification evidence.
    """
    warm_count = 12
    continuing_count = 12
    interval_seconds = 15.0
    schedule = (0.0,) * warm_count + tuple(
        interval_seconds * index for index in range(1, continuing_count + 1)
    )
    return replace(
        p2_config(len(schedule)),
        reset_mode="warm_start_rings",
        warm_start_uavs=warm_count,
        warm_start_steady_motion=True,
        fixed_arrival_schedule=schedule,
        total_requested_uavs=len(schedule),
        max_simulation_seconds=schedule[-1] + 900.0,
        runtime_proposal_shortlist_k=3,
    )


def _p2_scene_config(scenario: str):
    if scenario == P2_DENSE_SCENARIO:
        return _p2_dense_scene_config()
    if scenario not in P2_SCENARIOS:
        choices = (*P2_SCENARIOS, P2_DENSE_SCENARIO)
        raise ValueError(f"P2 profiles require one of {choices}, got {scenario!r}")
    return replace(
        p2_config(int(scenario.removeprefix("flow"))),
        runtime_proposal_shortlist_k=3,
    )


def _load_profile(checkpoint: Path, profile: str):
    if profile == "checkpoint12":
        if _sha(checkpoint) != CHECKPOINT12_SHA:
            raise ValueError("expected independently evaluated checkpoint_12.pt")
        policy = JointPPOV42(88)
        return policy
    if profile == "p2_initial_k3":
        if _sha(checkpoint) != P2_INITIAL_K3_SHA:
            raise ValueError("expected qualified P2 initial K=3 checkpoint")
        return p2_effectiveness.load_candidate_checkpoint(checkpoint)[0]
    if profile == "p2_trained_k3":
        if _sha(checkpoint) != P2_TRAINED_K3_SHA:
            raise ValueError("expected P2 formal trained K=3 checkpoint")
        return load_p2_trained(checkpoint, p2_heldout_protocol())[0]
    raise ValueError(f"unsupported checkpoint profile: {profile}")


def run(
    checkpoint, *, profile="checkpoint12", scenario="flow6", seed=690003,
    gui=True, playback_speed=10., max_steps=None,
):
    checkpoint = Path(checkpoint)
    config = (
        scenario_config(scenario, protocol())
        if profile == "checkpoint12"
        else _p2_scene_config(scenario)
    )
    policy = _load_profile(checkpoint, profile)
    if profile == "checkpoint12":
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
            details = _student_request_details(
                policy, env, observations, deterministic=True,
                shortlist_k=config.runtime_proposal_shortlist_k,
            )
            requested, proposal_shortlists = details[3], details[-1]
            observations, *_ = env.step(requested, proposal_shortlists)
            if env.arrivals.exhausted() and not env.active and not env.admission.queue.items and not env.service_uavs:
                break
            tick = _pace_live_rollout(observer, config.dt, tick)
        return {"profile": profile, "scenario": scenario, "seed": seed, "seconds": env.now,
                "completed": len(env.completed_uavs), "telemetry": dict(env.telemetry),
                "teacher_enabled": config.teacher_enabled, "observation_dim": policy.observation_dim,
                "runtime_proposal_shortlist_k": config.runtime_proposal_shortlist_k}
    finally:
        if observer:
            observer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES, default="checkpoint12")
    parser.add_argument(
        "--scenario", choices=(
            "warm3", "flow3", "flow6", "flow8", "flow12", P2_DENSE_SCENARIO,
        ),
        default="flow6",
    )
    parser.add_argument("--seed", type=int, default=690003)
    parser.add_argument("--playback-speed", type=float, default=10.)
    parser.add_argument("--gui", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-steps", type=int)
    a = parser.parse_args()
    torch.set_num_threads(1)
    print(run(a.checkpoint, profile=a.profile, scenario=a.scenario, seed=a.seed, gui=a.gui,
              playback_speed=a.playback_speed, max_steps=a.max_steps))
