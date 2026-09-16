"""One preregistered 88D DAgger round; never opens consumed Q2."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml

from ..algo.joint_ppo_v4_2 import hierarchical_bc_loss
from ..core.checkpoints import save_checkpoint
from ..core.safety_trace import SafetyTraceRecorder
from ..envs.parallel_env import TerminalParallelEnv
from .run_v4_2_dagger_round1 import _semantic_reason
from .run_v4_2_i2_dagger_preflight import _policy_tape_logprob_exact
from .run_v4_2_v2_1_formal_i0_r3 import _write
from .run_v4_2_v2_1_formal_i0_r4 import historical_registry
from .run_v4_2_v2_1_i1_v2 import _sha, _rows, protocols, _arrays
from .run_v4_2_v2_1_i1_multiseed_qualification import _evaluate_seed, _install_q_access_guard
from .run_v4_2_v2_1_i2_preflight import qualified_student
from .student_rollout_v4_2 import rollout

PROTOCOL = Path("terminal_marl/configs/low_v4_2/i2_v2_1_round1.yaml")
PROTOCOL_SHA = "2c86405b3e985dd5a9d2711d2bed21fe99c15ee1662a2f858ca3fc181f9380fa"
AUDIT_SHA = "b8bd5c5e926f38932575cfb1890ab089eb7f20dd968b482eb28355af06b15daa"


def semantic_rows(rows):
    memory, retained = {}, []
    for row in rows:
        key = (row["seed"], row["uav_id"], row["phase"])
        reason = _semantic_reason(memory, key, teacher_joint=row["teacher_joint_index"],
                                  observation=np.asarray(row["observation_v2_1"]), tick=row["step"])
        if reason is not None:
            retained.append({**row, "semantic_sampling_reason": reason})
    return retained


def fine_tune(policy, rows, protocol):
    torch.manual_seed(protocol["training_seed"])
    rng = np.random.RandomState(protocol["training_seed"])
    policy.optimizer = torch.optim.Adam(policy.model.parameters(), lr=protocol["learning_rate"])
    x, gates, proposals, _ = _arrays(rows)
    log = []
    for epoch in range(protocol["epochs"]):
        permutation = rng.permutation(len(x))
        losses = []
        for start in range(0, len(x), protocol["batch_size"]):
            ids = permutation[start:start + protocol["batch_size"]]
            gd, pd = policy.model.distributions(x[ids])
            loss, _ = hierarchical_bc_loss(gd.logits, pd.logits, gates[ids], proposals[ids])
            if not torch.isfinite(loss):
                raise ValueError("nonfinite DAgger loss")
            policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.model.parameters(), 1., error_if_nonfinite=True)
            policy.optimizer.step()
            losses.append(float(loss.detach()))
        log.append({"epoch": epoch + 1, "mean_loss": float(np.mean(losses)), "finite_gradients": True})
    return log


def collect(policy, base, output, seeds, sizes, horizon):
    summaries, rows = [], []
    for index, seed in enumerate(seeds):
        n = sizes[index % len(sizes)]
        config = replace(base, warm_start_uavs=n, total_requested_uavs=n,
                         fixed_arrival_schedule=(0.,) * n, max_simulation_seconds=horizon * base.dt)
        _, summary = rollout(policy, config, seed=seed, max_steps=horizon,
                             output=output / f"seed{seed}", collect_labels=True)
        rows.extend(_rows(output / f"seed{seed}" / "labels.jsonl"))
        summary["replay_exact"] = SafetyTraceRecorder.replay_action_tape(
            TerminalParallelEnv(config), summary["action_tape"])["exact"]
        summary["requested_logprob_exact"] = _policy_tape_logprob_exact(policy, config, summary["action_tape"])
        summaries.append(summary)
        _write(output / "progress.json", {"completed": len(summaries), "total": len(seeds), "summaries": summaries})
        if any(summary["safety"].values()):
            break
    return rows, summaries


def collection_gates(rows, summaries, seeds, horizon):
    return {"complete_episodes": len(summaries) == len(seeds) and all(s["steps"] == horizon or s["terminal"] for s in summaries),
        "unique_events": bool(rows) and len({r["decision_event_id"] for r in rows}) == len(rows),
        "schema": all(len(r["observation_v2_1"]) == 88 for r in rows),
        "safety": all(not any(s["safety"].values()) for s in summaries),
        "replay": bool(summaries) and all(s["replay_exact"] and s["requested_logprob_exact"] for s in summaries)}


def run(output, i1_path, r4_path, preflight_path, history_root, *, smoke=False):
    if _sha(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("round1 protocol changed")
    p = yaml.safe_load(PROTOCOL.read_text())
    if _sha(preflight_path) != p["preflight_manifest_sha256"]:
        raise ValueError("preflight lineage mismatch")
    preflight = json.loads(preflight_path.read_text())
    if not preflight["I2_ROUND1_ALLOWED"] or not all(preflight["gates"].values()):
        raise ValueError("preflight not qualified")
    policy, base, i1 = qualified_student(i1_path, r4_path)
    output.mkdir(parents=True, exist_ok=False)
    seeds = p["smoke_seeds"][:1] if smoke else p["collection_seeds"]
    validation_seeds = p["smoke_seeds"][1:] if smoke else p["validation_seeds"]
    sizes, horizon = ([1], 1) if smoke else (p["warm_start_sizes"], p["collection_steps"])
    if not smoke:
        history = historical_registry(history_root, output)
        if set(seeds + validation_seeds) & history["seeds"]:
            raise ValueError("round1 seeds previously consumed")
    _write(output / "pre_reset_registry.json", {"collection_seeds": seeds, "validation_seeds": validation_seeds, "protocol_sha256": PROTOCOL_SHA})
    raw, summaries = collect(policy, base, output / "collection", seeds, sizes, horizon)
    new_rows = semantic_rows(raw)
    gates = collection_gates(raw, summaries, seeds, horizon)
    gates["cardinality"] = smoke or p["minimum_new_events"] <= len(new_rows) <= p["maximum_new_events"]
    _write(output / "collection_manifest.json", {"gates": gates, "raw_rows": len(raw), "new_rows": len(new_rows), "summaries": summaries})
    result = {"stage": "V4_2_V2_1_DAGGER_R1", "smoke": smoke, "protocol_sha256": PROTOCOL_SHA,
        "preflight_manifest_sha256": _sha(preflight_path), "I1_qualification_sha256": _sha(i1_path),
        "parent_checkpoint_sha256": i1["canonical_checkpoint_sha256"], "gates": gates,
        "PPO_ALLOWED": False, "I3_ALLOWED": False, "DAGGER_R1": "FAIL"}
    if not all(gates.values()):
        _write(output / "manifest.json", result)
        return result
    parent = _rows(r4_path.parent / "training.jsonl")
    aggregate = parent + new_rows
    dataset = output / "D1.jsonl"
    with dataset.open("w") as handle:
        for row in aggregate:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    _write(output / "D1_manifest.json", {"parent_sha256": _sha(r4_path.parent / "training.jsonl"),
        "parent_rows": len(parent), "new_rows": len(new_rows), "aggregate_rows": len(aggregate),
        "dataset_sha256": _sha(dataset), "protocol_sha256": PROTOCOL_SHA, "smoke": smoke})
    audit_path = r4_path.parent / "internal_audit.jsonl"
    state = {"phase": "TRAINING", "q_access_during_training": False}
    _install_q_access_guard((audit_path,), state)
    try:
        train_protocol = {**p, "epochs": 1} if smoke else p
        log = fine_tune(policy, aggregate, train_protocol)
        checkpoint = output / "dagger_r1_final.pt"
        save_checkpoint(checkpoint, policy, base, stage="V4_2_V2_1_DAGGER_R1", ppo_called=False,
                        parent_checkpoint_sha256=i1["canonical_checkpoint_sha256"], protocol_sha256=PROTOCOL_SHA,
                        dataset_sha256=_sha(dataset), smoke=smoke)
        _write(output / "training_freeze.json", {"checkpoint_sha256": _sha(checkpoint), "training": log})
    finally:
        state["phase"] = "FINISHED"
    if _sha(audit_path) != AUDIT_SHA:
        raise ValueError("R4 retention audit changed")
    retention = _evaluate_seed(policy, _rows(audit_path), protocols()[0], all(r["finite_gradients"] for r in log))
    validation_rows, validation = collect(policy, base, output / "validation", validation_seeds,
                                          sizes, 1 if smoke else p["validation_steps"])
    vg = collection_gates(validation_rows, validation, validation_seeds, 1 if smoke else p["validation_steps"])
    gates.update({"validation_" + key: value for key, value in vg.items()})
    gates["retention"] = retention["PASS"]
    gates["both_gates"] = all(sum(s["counts"][name] for s in validation) > 0 for name in ("WAIT", "RELEASE"))
    result.update(DAGGER_R1="PASS" if all(gates.values()) else "FAIL", I3_ALLOWED=not smoke and all(gates.values()),
                  checkpoint=str(checkpoint), checkpoint_sha256=_sha(checkpoint), retention=retention,
                  validation=validation, new_rows=len(new_rows), dataset_sha256=_sha(dataset))
    _write(output / "manifest.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("output-dir", "i1-qualification", "r4-manifest", "preflight-manifest", "history-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    a = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(run(a.output_dir, a.i1_qualification, a.r4_manifest,
                         a.preflight_manifest, a.history_root, smoke=a.smoke), indent=2), flush=True)
