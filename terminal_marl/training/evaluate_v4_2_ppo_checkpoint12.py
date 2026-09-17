"""Fresh paired checkpoint evaluation, with existing viewer snapshots."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from dataclasses import replace

import torch
import yaml

from ..algo.joint_ppo_v4_2 import JointPPOV42
from ..core.checkpoints import load_checkpoint
from ..core.safety_trace import SafetyTraceRecorder
from ..envs.parallel_env import TerminalParallelEnv
from ..visualization.snapshot_playback import SnapshotRecorder, write_playback
from .run_v4_2_v2_1_i1_v2 import _sha, reserve_once
from .run_v4_2_v2_1_formal_i0_r3 import _write
from .run_v4_2_v2_1_formal_i0_r4 import historical_registry
from .run_v4_2_v2_1_student_qualification import protocol as i3_protocol, scenario_config, episode_gates, qualified_dagger
from .run_v4_2_i2_dagger_preflight import _policy_tape_logprob_exact
from .student_rollout_v4_2 import rollout

PROTOCOL = Path("terminal_marl/configs/low_v4_2/ppo_checkpoint12_evaluation.yaml")
PROTOCOL_SHA = "c1def0f240f33882a2a04f0bcaa4a2178124a7141f9b194d6e5dc35afd65c1dc"


def run(output, training_manifest, dagger_manifest, history_root, *, smoke=False):
    if _sha(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("evaluation protocol changed")
    p = yaml.safe_load(PROTOCOL.read_text())
    if _sha(training_manifest) != p["training_manifest_sha256"]:
        raise ValueError("training evidence mismatch")
    training = json.loads(training_manifest.read_text())
    if not training["PASS"] or len(training["records"]) != 12:
        raise ValueError("training block did not pass")
    checkpoint = training_manifest.parent / "checkpoint_12.pt"
    if _sha(checkpoint) != p["checkpoint_sha256"]:
        raise ValueError("final checkpoint mismatch")
    base_protocol = i3_protocol()
    candidate = JointPPOV42(88)
    payload = load_checkpoint(checkpoint, candidate, scenario_config("warm3", base_protocol), load_optimizer=False)
    if payload.get("stage") != "V4_2_V2_1_PPO" or not payload.get("formal"):
        raise ValueError("not a formal PPO checkpoint")
    baseline = qualified_dagger(dagger_manifest, base_protocol)
    output.mkdir(parents=True, exist_ok=False)
    seeds = [p["smoke_seed"]] if smoke else p["seeds"]
    if not smoke:
        history = historical_registry(history_root, output)
        if set(seeds) & history["seeds"]:
            raise ValueError("evaluation seeds already consumed")
        reserve_once(training_manifest.parent / "checkpoint12_evaluation_consumed.json", {
            "protocol_sha256": PROTOCOL_SHA, "seeds": seeds, "output": str(output)})
    _write(output / "pre_reset_registry.json", {"seeds": seeds, "paired_baseline": True, "smoke": smoke})
    results = {"checkpoint12": [], "dagger": []}
    for index, seed in enumerate(seeds):
        scenario = p["scenarios"][index % 3]
        behavior = p["behaviors"][index // 3]
        config = scenario_config(scenario, base_protocol)
        horizon = p["smoke_steps"] if smoke else int(config.max_simulation_seconds / config.dt)
        if smoke:
            config = replace(config, max_simulation_seconds=horizon * config.dt)
        for name, policy in (("checkpoint12", candidate), ("dagger", baseline)):
            _, summary = rollout(policy, config, seed=seed, max_steps=horizon,
                output=output / name / f"seed{seed}", deterministic=behavior == "deterministic")
            env = TerminalParallelEnv(config)
            recorder = SnapshotRecorder(p["snapshot_stride"]) if name == "checkpoint12" and (smoke or seed == p["visualization_seed"]) else None
            if recorder:
                env.set_step_observer(recorder)
            summary["replay_exact"] = SafetyTraceRecorder.replay_action_tape(env, summary["action_tape"])["exact"]
            summary["requested_logprob_exact"] = _policy_tape_logprob_exact(policy, config, summary["action_tape"])
            summary.update(scenario=scenario, behavior=behavior, gates=episode_gates(summary, config, base_protocol),
                throughput_per_hour=3600 * summary["completed"] / max(config.dt, summary["seconds"]))
            if recorder:
                recorder.capture(env)
                write_playback(output / "validation_scene.html", recorder.frames,
                    f"checkpoint_12 · {scenario} · seed {seed} · {behavior}")
            results[name].append(summary)
            _write(output / "progress.json", {"completed_rollouts": sum(map(len, results.values())), "total_rollouts": 2 * len(seeds), "results": results})
    gates = {"all_candidate_episodes": len(results["checkpoint12"]) == len(seeds),
        "I3_episode_gates": all(all(r["gates"].values()) for r in results["checkpoint12"]),
        "both_gates": all(sum(r["counts"][key] for r in results["checkpoint12"]) > 0 for key in ("WAIT", "RELEASE"))}
    pairs = [{"seed": a["seed"], "scenario": a["scenario"], "behavior": a["behavior"],
              "candidate_seconds": a["seconds"], "dagger_seconds": b["seconds"],
              "seconds_delta": a["seconds"] - b["seconds"],
              "candidate_completed": a["completed"], "dagger_completed": b["completed"]}
             for a, b in zip(results["checkpoint12"], results["dagger"])]
    result = {"stage": "CHECKPOINT12_INDEPENDENT_EVALUATION", "smoke": smoke,
        "PASS": not smoke and all(gates.values()), "gates": gates, "results": results,
        "paired_comparison": pairs, "checkpoint_sha256": _sha(checkpoint), "protocol_sha256": PROTOCOL_SHA,
        "next_training_preparation_allowed": not smoke and all(gates.values()), "training_called": False}
    _write(output / "manifest.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("output-dir", "training-manifest", "dagger-manifest", "history-root"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    a = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(run(a.output_dir, a.training_manifest, a.dagger_manifest, a.history_root, smoke=a.smoke), indent=2))
