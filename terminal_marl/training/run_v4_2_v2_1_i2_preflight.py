"""88D Safe DAgger preflight tied to qualified R4 / I1 V2 artifacts."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
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
from .run_v4_2_v2_1_i1_v2 import validate_r4, _sha, _rows
from .student_rollout_v4_2 import OBS_SCHEMA, rollout

PROTOCOL = Path("terminal_marl/configs/low_v4_2/i2_v2_1_preflight.yaml")
PROTOCOL_SHA = "dc5c9f1268fdddeff77f545c23e0c8f77000990393e1820bda43643bd72c0bfa"
I1_SHA = "0e3817ad4fc928f20ba8f6cff7cf7b61801507c557e62daa53aa845039cf87cc"


def qualified_student(i1_path: Path, r4_path: Path):
    validate_r4(r4_path)
    if _sha(i1_path) != I1_SHA:
        raise ValueError("I1 V2 qualification hash mismatch")
    report = json.loads(i1_path.read_text())
    if report["I1_V2"] != "PASS" or not report["I2_PREFLIGHT_ALLOWED"] or report["Q_EVALUATION_COUNT"] != 1:
        raise ValueError("I1 V2 has not qualified")
    path = i1_path.parent / Path(report["canonical_checkpoint"]).name
    if _sha(path) != report["canonical_checkpoint_sha256"]:
        raise ValueError("canonical checkpoint changed")
    config = replace(_config(), observation_schema_version=OBS_SCHEMA)
    policy = JointPPOV42(88)
    payload = load_checkpoint(path, policy, config, load_optimizer=False)
    if payload.get("stage") != "V4_2_V2_1_I1_V2" or not payload.get("fresh_actor") or payload.get("ppo_called"):
        raise ValueError("wrong checkpoint lineage")
    policy.model.eval()
    return policy, config, report


def run(output: Path, i1_path: Path, r4_path: Path, history_root: Path, *, smoke=False):
    if _sha(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("preflight protocol changed")
    p = yaml.safe_load(PROTOCOL.read_text())
    policy, config, i1 = qualified_student(i1_path, r4_path)
    seeds = p["smoke_seeds"] if smoke else p["seeds"]
    horizon = p["smoke_steps"] if smoke else p["max_steps"]
    n = p["smoke_uavs"] if smoke else p["warm_start_uavs"]
    config = replace(config, warm_start_uavs=n, total_requested_uavs=n,
                     fixed_arrival_schedule=(0.,) * n, max_simulation_seconds=horizon * config.dt)
    output.mkdir(parents=True, exist_ok=False)
    if not smoke:
        history = historical_registry(history_root, output)
        if set(seeds) & history["seeds"]:
            raise ValueError("preflight seeds previously consumed")
    _write(output / "pre_reset_registry.json", {"seeds": seeds, "protocol_sha256": PROTOCOL_SHA, "smoke": smoke})
    summaries, rows = [], []
    for seed in seeds:
        _, summary = rollout(policy, config, seed=seed, max_steps=horizon,
                             output=output / f"seed{seed}", collect_labels=True)
        episode_rows = _rows(output / f"seed{seed}" / "labels.jsonl")
        replay = SafetyTraceRecorder.replay_action_tape(TerminalParallelEnv(config), summary["action_tape"])
        summary["replay_exact"] = replay["exact"]
        summary["requested_logprob_exact"] = _policy_tape_logprob_exact(policy, config, summary["action_tape"])
        summaries.append(summary)
        rows.extend(episode_rows)
        _write(output / "progress.json", {"completed": len(summaries), "total": len(seeds), "summaries": summaries})
    gates = {
        "lineage": True,
        "complete_rollouts": all(s["steps"] == horizon for s in summaries),
        "label_provenance": bool(rows) and all(len(r["observation_v2_1"]) == 88
            and np.isfinite(r["requested_logprob"]) and r["source"] == "student_visited_label_only" for r in rows),
        "unique_decisions": len({r["decision_event_id"] for r in rows}) == len(rows),
        "exact_replay": all(s["replay_exact"] for s in summaries),
        "requested_logprob": all(s["requested_logprob_exact"] for s in summaries),
        "safety": all(not any(s["safety"].values()) for s in summaries),
    }
    result = {"stage": "V4_2_V2_1_I2_PREFLIGHT", "smoke": smoke, "gates": gates,
        "I2_PREFLIGHT": "PASS" if all(gates.values()) else "FAIL",
        "I2_ROUND1_ALLOWED": not smoke and all(gates.values()), "PPO_ALLOWED": False,
        "I1_qualification_sha256": I1_SHA, "protocol_sha256": PROTOCOL_SHA,
        "student_checkpoint_sha256": i1["canonical_checkpoint_sha256"],
        "labels": len(rows), "summaries": summaries, "retraining_called": False}
    _write(output / "manifest.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("output-dir", "i1-qualification", "r4-manifest", "history-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(run(args.output_dir, args.i1_qualification, args.r4_manifest,
                         args.history_root, smoke=args.smoke), indent=2), flush=True)
