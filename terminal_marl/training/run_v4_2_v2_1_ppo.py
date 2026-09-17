"""Bounded student-request PPO entry with evidence-bound admission."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml

from ..algo.student_ppo_v4_2 import StudentPPOV42
from ..algo.joint_ppo_v4_2 import hierarchical_joint_logprob
from ..core.checkpoints import load_checkpoint, save_checkpoint
from ..core.safety_trace import SafetyTraceRecorder
from ..envs.parallel_env import TerminalParallelEnv
from .run_v4_2_v2_1_formal_i0_r3 import _write
from .run_v4_2_v2_1_formal_i0_r4 import historical_registry
from .run_v4_2_v2_1_i1_v2 import _sha
from .run_v4_2_v2_1_i2_preflight import qualified_student
from .run_v4_2_v2_1_student_qualification import protocol as i3_protocol, qualified_dagger, scenario_config
from .student_rollout_v4_2 import rollout

PROTOCOL = Path("terminal_marl/configs/low_v4_2/ppo_v2_1_initial_entry_v2.yaml")
PROTOCOL_SHA = "6f810662a10bf42b439bde233d166137bc8e3a7d84da60796677565e6d653864"


def protocol():
    if _sha(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("PPO protocol changed")
    return yaml.safe_load(PROTOCOL.read_text())


def source_hashes():
    paths = [Path(__file__), Path("terminal_marl/algo/student_ppo_v4_2.py"),
        Path("terminal_marl/algo/joint_ppo_v4_2.py"), Path("terminal_marl/training/student_rollout_v4_2.py")]
    return {path.name: _sha(path) for path in paths}


def review(output, i3_path, dagger_path, i1_path, r4_path):
    p = protocol()
    qualified_student(i1_path, r4_path)
    qualified_dagger(dagger_path, i3_protocol())
    if _sha(i3_path) != p["i3_manifest_sha256"]:
        raise ValueError("I3 evidence mismatch")
    report = json.loads(i3_path.read_text())
    gates = {"I3": report["I3"] == "PASS" and report["PPO_ADMISSION_REVIEW_ALLOWED"],
        "all_qualification_gates": all(report["gates"].values()) and len(report["results"]) == 6
            and all(all(r["gates"].values()) for r in report["results"]),
        "student_only": not report["teacher_enabled"] and not report["training_called"],
        "scope": report["scope"] == p["scope"],
        "checkpoint_lineage": report["checkpoint_sha256"] == i3_protocol()["checkpoint_sha256"]}
    output.mkdir(parents=True, exist_ok=False)
    result = {"stage": "PPO_ADMISSION_REVIEW", "gates": gates, "ENTRY_VALIDATION_ALLOWED": all(gates.values()),
        "PPO_ALLOWED": False, "i3_sha256": _sha(i3_path), "dagger_sha256": _sha(dagger_path),
        "checkpoint_sha256": report["checkpoint_sha256"], "scope": p["scope"],
        "protocol_sha256": PROTOCOL_SHA, "source_hashes": source_hashes()}
    _write(output / "review.json", result)
    return result


def validate_admission(report, *, formal):
    p = protocol()
    required = "PPO_ALLOWED" if formal else "ENTRY_VALIDATION_ALLOWED"
    if not report.get(required) or not report.get("gates") or not all(report["gates"].values()):
        raise PermissionError("PPO admission/entry validation has not passed")
    if (report.get("i3_sha256") != p["i3_manifest_sha256"] or report.get("protocol_sha256") != PROTOCOL_SHA
            or report.get("source_hashes") != source_hashes() or report.get("scope") != p["scope"]
            or report.get("checkpoint_sha256") != i3_protocol()["checkpoint_sha256"]):
        raise ValueError("admission evidence, scope or implementation changed")


def requested_logprob_error(policy, steps):
    maximum = 0.
    with torch.no_grad():
        for step in steps:
            mask = step.actor_mask
            if not mask.any():
                continue
            gd, pd = policy.model.distributions(torch.as_tensor(step.world[mask]))
            _, _, current = hierarchical_joint_logprob(gd, pd, torch.as_tensor(step.requested_gate[mask]),
                                                       torch.as_tensor(step.requested_proposal[mask]))
            maximum = max(maximum, float(np.abs(current.numpy() - step.requested_logprob[mask]).max()))
    return maximum


def run(output, admission_path, dagger_path, history_root, *, formal=False, episodes=12):
    p = protocol()
    admission = json.loads(admission_path.read_text())
    validate_admission(admission, formal=formal)
    qualified_dagger(dagger_path, i3_protocol())
    source = json.loads(dagger_path.read_text())
    checkpoint = dagger_path.parent / Path(source["checkpoint"]).name
    if formal and not 1 <= episodes <= p["formal_max_episodes"]:
        raise ValueError("formal episode budget outside registered block")
    seeds = list(range(p["formal_seed_start"], p["formal_seed_start"] + episodes)) if formal else p["verification_seeds"]
    scenarios = p["formal_scenarios"] if formal else p["verification_scenarios"]
    output.mkdir(parents=True, exist_ok=False)
    history = historical_registry(history_root, output)
    if set(seeds) & history["seeds"]:
        raise ValueError("PPO/verification seeds previously consumed")
    _write(output / "pre_reset_registry.json", {"seeds": seeds, "formal": formal, "protocol_sha256": PROTOCOL_SHA})
    policy = StudentPPOV42(88, learning_rate=p["learning_rate"])
    base = scenario_config("warm3", i3_protocol())
    load_checkpoint(checkpoint, policy, base, load_optimizer=False)
    np.random.seed(seeds[0])
    records, passed = [], True
    update_keys = ("minibatch_size", "gamma", "gae_lambda", "clip_ratio", "value_coef", "entropy_coef")
    for index, seed in enumerate(seeds):
        config = scenario_config(scenarios[index % len(scenarios)], i3_protocol())
        horizon = int(p["formal_horizon_seconds"] / config.dt) if formal else p["verification_steps"][index]
        config = replace(config, max_simulation_seconds=horizon * config.dt)
        steps, summary = rollout(policy, config, seed=seed, max_steps=horizon, output=output / f"seed{seed}")
        error = requested_logprob_error(policy, steps)
        replay = SafetyTraceRecorder.replay_action_tape(TerminalParallelEnv(config), summary["action_tape"])["exact"]
        gates = {"safety": not any(summary["safety"].values()), "requested_ratio_initial_one": error <= 1e-5,
                 "replay": replay, "stochastic_student": bool(steps) and all(s.stochastic_behavior and not s.teacher_used for s in steps),
                 "actor_decisions": sum(int(s.actor_mask.sum()) for s in steps) > 0}
        if formal:
            gates["completion"] = summary["terminal"] and summary["completed"] == config.total_requested_uavs
            gates["shield_burden"] = summary["counts"]["shield_actor"] / max(1, summary["counts"]["actor_decisions"]) <= .05
        update = None
        if all(gates.values()):
            update = policy.update(steps, training_authorized=True, epochs=p["update_epochs"], **{key: p[key] for key in update_keys})
            gates["finite_update"] = all(np.isfinite(v) for v in update.values()) and all(torch.isfinite(v).all() for v in policy.model.parameters())
            gates["bounded_KL"] = update["approx_kl"] <= p["maximum_approx_kl"]
            saved = output / f"checkpoint_{index + 1}.pt"
            save_checkpoint(saved, policy, config, stage="V4_2_V2_1_PPO" if formal else "V4_2_V2_1_PPO_ENTRY_TEST",
                            ppo_called=True, formal=formal, admission_sha256=_sha(admission_path), protocol_sha256=PROTOCOL_SHA)
            restored = StudentPPOV42(88, learning_rate=p["learning_rate"])
            load_checkpoint(saved, restored, config, load_optimizer=True)
            gates["checkpoint_reload"] = all(torch.equal(v, restored.model.state_dict()[key]) for key, v in policy.model.state_dict().items())
            # Next episode also exercises optimizer restoration; verification
            # checkpoints are never the initialization of a formal block.
            policy = restored
        records.append({"seed": seed, "summary": summary, "initial_logprob_error": error, "gates": gates, "update": update})
        _write(output / "progress.json", {"completed": len(records), "total": len(seeds), "records": records})
        if not all(gates.values()):
            passed = False
            break
    passed = passed and len(records) == len(seeds)
    result = {"stage": "FORMAL_PPO_BLOCK" if formal else "PPO_ENTRY_VALIDATION", "PASS": passed,
        "PPO_ALLOWED": not formal and passed, "gates": {"all_rollouts_and_updates": passed},
        "i3_sha256": p["i3_manifest_sha256"], "protocol_sha256": PROTOCOL_SHA,
        "source_hashes": source_hashes(), "scope": p["scope"], "checkpoint_sha256": admission["checkpoint_sha256"],
        "review_sha256": _sha(admission_path), "records": records,
        "formal_training_called": formal, "verification_checkpoints_eligible_for_formal": False}
    _write(output / ("manifest.json" if formal else "admission.json"), result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("review", "verify-entry", "train"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dagger-manifest", type=Path, required=True)
    for name in ("i3-manifest", "i1-qualification", "r4-manifest", "admission", "history-root"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--episodes", type=int, default=12)
    a = parser.parse_args()
    torch.set_num_threads(1)
    if a.mode == "review":
        if not all((a.i3_manifest, a.i1_qualification, a.r4_manifest)):
            parser.error("review requires I3, I1 and R4 manifests")
        result = review(a.output_dir, a.i3_manifest, a.dagger_manifest, a.i1_qualification, a.r4_manifest)
    else:
        if a.admission is None or a.history_root is None:
            parser.error("execution requires admission and history root")
        result = run(a.output_dir, a.admission, a.dagger_manifest, a.history_root, formal=a.mode == "train", episodes=a.episodes)
    print(json.dumps(result, indent=2), flush=True)
