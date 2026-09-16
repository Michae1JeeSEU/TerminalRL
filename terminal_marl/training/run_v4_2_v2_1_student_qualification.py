"""Independent student-only initial-curriculum qualification, without updates."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
import yaml

from ..algo.joint_ppo_v4_2 import JointPPOV42
from ..core.checkpoints import load_checkpoint
from ..core.safety_trace import SafetyTraceRecorder
from ..envs.parallel_env import TerminalParallelEnv
from .run_v4_2_hierarchical_s0_qualification import _config
from .run_v4_2_i2_dagger_preflight import _policy_tape_logprob_exact
from .run_v4_2_v2_1_formal_i0_r3 import _write
from .run_v4_2_v2_1_formal_i0_r4 import historical_registry
from .run_v4_2_v2_1_i1_v2 import _sha, reserve_once
from .student_rollout_v4_2 import OBS_SCHEMA, rollout

PROTOCOL = Path("terminal_marl/configs/low_v4_2/i3_v2_1_qualification.yaml")
PROTOCOL_SHA = "21d1761860ba4636c811b9a7739629e12fafe12dbf2ee9bd92ae6decfbd4f387"


def protocol():
    if _sha(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("student qualification protocol changed")
    return yaml.safe_load(PROTOCOL.read_text())


def scenario_config(name, p):
    n = 6 if name == "flow6" else 3
    natural = name.startswith("flow")
    if name not in p["scenarios"]:
        raise ValueError("scenario outside admission scope")
    schedule = tuple(p["first_arrival_seconds"] + i * p["arrival_interval_seconds"] for i in range(n)) if natural else (0.,) * n
    return replace(_config(), observation_schema_version=OBS_SCHEMA, experiment_mode="fixed_low_student",
        teacher_enabled=False, teacher_release_label_enabled=False, high_level_enabled=False,
        reset_mode="flow_from_empty" if natural else "warm_start_rings", warm_start_uavs=0 if natural else n,
        total_requested_uavs=n, fixed_arrival_schedule=schedule, max_simulation_seconds=p["horizon_seconds"])


def qualified_dagger(path, p):
    if _sha(path) != p["dagger_manifest_sha256"]:
        raise ValueError("DAgger manifest mismatch")
    report = json.loads(path.read_text())
    if report["DAGGER_R1"] != "PASS" or not report["I3_ALLOWED"] or not all(report["gates"].values()):
        raise ValueError("DAgger has not qualified")
    checkpoint = path.parent / Path(report["checkpoint"]).name
    if _sha(checkpoint) != p["checkpoint_sha256"]:
        raise ValueError("DAgger checkpoint mismatch")
    policy = JointPPOV42(88)
    payload = load_checkpoint(checkpoint, policy, scenario_config("warm3", p), load_optimizer=False)
    if payload.get("stage") != "V4_2_V2_1_DAGGER_R1" or payload.get("smoke") or payload.get("ppo_called"):
        raise ValueError("invalid student lineage")
    policy.model.eval()
    return policy


def episode_gates(summary, config, p):
    decisions = summary["counts"]["actor_decisions"]
    shield_fraction = summary["counts"]["shield_actor"] / max(1, decisions)
    return {
        "all_requested_complete": summary["completed"] == config.total_requested_uavs,
        "deadline": 0 < summary["seconds"] <= p["horizon_seconds"],
        "terminal_and_queue_drained": summary["terminal"] and bool(summary["queue"]) and summary["queue"][-1] == 0,
        "safety": not any(summary["safety"].values()),
        "replay": summary["replay_exact"] and summary["requested_logprob_exact"],
        "shield_burden": shield_fraction <= p["gates"]["shield_actor_fraction_max"],
        "student_decisions": decisions > 0,
    }


def run(output, dagger_path, history_root, *, smoke=False):
    p = protocol()
    policy = qualified_dagger(dagger_path, p)
    output.mkdir(parents=True, exist_ok=False)
    seeds = [p["smoke_seed"]] if smoke else p["seeds"]
    if not smoke:
        history = historical_registry(history_root, output)
        if set(seeds) & history["seeds"]:
            raise ValueError("qualification seeds already consumed")
        reserve_once(dagger_path.parent / "i3_qualification_consumed.json", {
            "protocol_sha256": PROTOCOL_SHA, "output": str(output), "seeds": seeds})
    _write(output / "pre_reset_registry.json", {"seeds": seeds, "protocol_sha256": PROTOCOL_SHA, "smoke": smoke})
    results = []
    for index, seed in enumerate(seeds):
        name = p["scenarios"][index % 3]
        behavior = p["behaviors"][index // 3]
        config = scenario_config(name, p)
        horizon = p["smoke_steps"] if smoke else int(p["horizon_seconds"] / config.dt)
        if smoke:
            config = replace(config, max_simulation_seconds=horizon * config.dt)
        _, summary = rollout(policy, config, seed=seed, max_steps=horizon,
            output=output / f"seed{seed}", deterministic=behavior == "deterministic", collect_labels=False)
        _write(output / "progress.json", {"phase": "REPLAY", "seed": seed, "finished": len(results), "total": len(seeds)})
        summary["replay_exact"] = SafetyTraceRecorder.replay_action_tape(TerminalParallelEnv(config), summary["action_tape"])["exact"]
        summary["requested_logprob_exact"] = _policy_tape_logprob_exact(policy, config, summary["action_tape"])
        summary.update(scenario=name, behavior=behavior, gates=episode_gates(summary, config, p),
            throughput_per_hour=3600 * summary["completed"] / max(config.dt, summary["seconds"]),
            completion_fraction=summary["completed"] / config.total_requested_uavs,
            shield_actor_fraction=summary["counts"]["shield_actor"] / max(1, summary["counts"]["actor_decisions"]))
        results.append(summary)
        _write(output / "progress.json", {"phase": "QUALIFICATION", "finished": len(results), "total": len(seeds), "results": results})
        if any(summary["safety"].values()):
            break
    gates = {"all_episodes": len(results) == len(seeds),
        "episode_gates": all(all(r["gates"].values()) for r in results),
        "both_gates": all(sum(r["counts"][key] for r in results) > 0 for key in ("WAIT", "RELEASE"))}
    passed = not smoke and all(gates.values())
    result = {"stage": "V4_2_V2_1_I3", "smoke": smoke, "I3": "PASS" if passed else "FAIL",
        "PPO_ADMISSION_REVIEW_ALLOWED": passed, "PPO_ALLOWED": False, "protocol_sha256": PROTOCOL_SHA,
        "dagger_manifest_sha256": _sha(dagger_path), "checkpoint_sha256": p["checkpoint_sha256"],
        "scope": p["admission_scope"], "full_static_geometry_qualification": False,
        "teacher_enabled": False, "training_called": False, "gates": gates, "results": results}
    _write(output / "manifest.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("output-dir", "dagger-manifest", "history-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    a = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(run(a.output_dir, a.dagger_manifest, a.history_root, smoke=a.smoke), indent=2), flush=True)
