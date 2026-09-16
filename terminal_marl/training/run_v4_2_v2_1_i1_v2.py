"""Fresh Q2 freeze, then R4 multi-seed BC and a single Q2 evaluation.

Historical I1 scripts/protocols remain reproducible and unmodified.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml

from ..algo.joint_ppo_v4_2 import JointPPOV42, hierarchical_bc_loss
from ..core.checkpoints import save_checkpoint, load_checkpoint
from ..core.safety_trace import SafetyTraceRecorder
from ..envs.parallel_env import TerminalParallelEnv
from .run_v4_2_hierarchical_s0_qualification import _config as base_config
from .run_v4_2_v2_1_formal_i0 import _config, _sha
from .run_v4_2_v2_1_formal_i0_r import _aliasing, _provenance
from .run_v4_2_v2_1_formal_i0_r3 import FixedCollectionSchedule, R3Stratum, _collect_partition, _verify_protocol, _write
from .run_v4_2_v2_1_formal_i0_r4 import historical_registry
from .run_v4_2_v2_1_i1_q_cohort import STRATA as Q_STRATA, support, _identity_accumulator
from .run_v4_2_v2_1_i1_bc import _arrays, _rows, _metrics, _architecture_fingerprint
from .run_v4_2_v2_1_i1_multiseed_qualification import _evaluate_seed, _install_q_access_guard

PROTOCOL_PATH = Path("terminal_marl/configs/low_v4_2/i1_v2_1_qualification_protocol_v2.yaml")
Q_PROTOCOL_PATH = Path("terminal_marl/configs/low_v4_2/i1_v2_1_q2_cohort.yaml")
PROTOCOL_SHA = "208d79e9600d3439f57b6fac747bdd96434f041adf1ddd9823619da387845638"
Q_PROTOCOL_SHA = "1fbd0d0bca62303edc8ca76dc6b6693f8755ab8bc26e02fb1d04faab28b2509f"
R4_MANIFEST_SHA = "b4291ddd3540a5a048cb645fb6b741b5379ad15220e1be865ce6993ae6d6fb2d"


def protocols() -> tuple[dict, dict]:
    if _sha(PROTOCOL_PATH) != PROTOCOL_SHA or _sha(Q_PROTOCOL_PATH) != Q_PROTOCOL_SHA:
        raise ValueError("I1 V2/Q2 protocol changed after preregistration")
    return (yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8")),
            yaml.safe_load(Q_PROTOCOL_PATH.read_text(encoding="utf-8")))


def validate_r4(path: Path) -> dict:
    if _sha(path) != R4_MANIFEST_SHA:
        raise ValueError("qualified R4 manifest mismatch")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["I0_R4"] != "PASS" or not manifest["I1_PROTOCOL_ALLOWED"] or not all(manifest["gates"].values()):
        raise ValueError("R4 has not qualified")
    protocol, _ = protocols()
    dataset = path.parent / "training.jsonl"
    if _sha(dataset) != protocol["training_dataset_sha256"]:
        raise ValueError("R4 training dataset mismatch")
    return manifest


def collect_q2(output: Path, history_root: Path, r4_path: Path, *, smoke: bool = False) -> dict:
    validate_r4(r4_path)
    protocol, q_protocol = protocols()
    support_protocol, check = _verify_protocol()
    if not check["PASS"]:
        raise ValueError("inherited collector semantics changed")
    output.mkdir(parents=True, exist_ok=False)
    strata = tuple(R3Stratum(s.name, s.spec) for s in Q_STRATA)
    cycles = q_protocol["cycles"]
    if smoke:
        strata = tuple(replace(s, spec=replace(s.spec, max_steps=1)) for s in strata[:2])
        cycles = 1
        seeds = q_protocol["smoke_seeds"]
    else:
        seeds = list(range(q_protocol["seed_start"], q_protocol["seed_start"] + len(strata) * cycles))
    history = _identity_accumulator() if smoke else historical_registry(history_root, output)
    if set(seeds) & history["seeds"]:
        raise ValueError("Q2 seeds already consumed; no automatic replacement")
    _write(output / "pre_reset_registry.json", {"seeds": seeds, "built_before_reset": True,
        "historical_seed_count": len(history["seeds"]), "protocol_sha256": PROTOCOL_SHA,
        "q_protocol_sha256": Q_PROTOCOL_SHA, "smoke_only": smoke})
    plan = FixedCollectionSchedule(strata, cycles, q_protocol["maximum_rows"], "v2-1-i1-q2")
    rows, info, tapes = _collect_partition(output, "qualification", seeds, support_protocol, schedule=plan)
    dataset = output / "q2.jsonl"
    with dataset.open("w", encoding="utf-8") as handle:
        for row in rows:
            row["qualification_usage"] = "NEVER_TRAIN_NEVER_MODEL_SELECT_NEVER_THRESHOLD_TUNE"
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    replay = []
    for path, spec, name in tapes:
        exact = SafetyTraceRecorder.replay_action_tape(TerminalParallelEnv(_config(spec)), path)["exact"]
        replay.append({"stratum": name, "exact": bool(exact), "path": str(path)})
        _write(output / "replay_progress.json", {"completed": len(replay), "total": len(tapes), "exact": all(r["exact"] for r in replay)})
    independent = True
    for key, historic in (("seed", "seeds"), ("episode_id", "episode_ids"), ("lifecycle_event_id", "lifecycle_ids"), ("full_state_hash", "full_state_hashes")):
        independent &= not bool({r[key] for r in rows} & history[historic])
    independent &= len({r["decision_event_id"] for r in rows}) == len(rows)
    independent &= len({r["full_state_hash"] for r in rows}) == len(rows)
    opposite = any(history["observation_labels"].get(r["observation_hash"], set()) - {r["teacher_gate"]} for r in rows)
    provenance, aliasing, coverage = _provenance(rows), _aliasing(rows), support(rows)
    gates = {
        "fixed_schedule": info["episodes"] == len(seeds) and not info["fatal"],
        "cardinality": q_protocol["minimum_rows"] <= len(rows) <= q_protocol["maximum_rows"],
        "support": coverage["all_minima_PASS"],
        "schema": provenance["schema_provenance"] == "PASS" and provenance["retry_feature_provenance"] == "PASS",
        "teacher_determinism": all(r["teacher_joint_index"] == r["teacher_repeat_joint_index"] for r in rows),
        "replay": len(replay) == len(strata) and all(r["exact"] for r in replay),
        "safety": not any(info["safety"].values()),
        "independence": bool(independent and not smoke),
        "aliasing": not opposite and aliasing["observation_aliasing"] == "PASS",
    }
    result = {"stage": "V4_2_V2_1_Q2", "smoke_only": smoke,
        "protocol_sha256": PROTOCOL_SHA, "q_protocol_sha256": Q_PROTOCOL_SHA,
        "r4_manifest_sha256": R4_MANIFEST_SHA, "dataset_path": str(dataset), "dataset_sha256": _sha(dataset),
        "rows": len(rows), "collection": info, "support": coverage, "provenance": provenance,
        "replay": replay, "aliasing": aliasing, "freeze_gates": gates,
        "Q2_FROZEN": not smoke and all(gates.values()), "PPO_ALLOWED": False}
    _write(output / "manifest.json", result)
    return result


def fit(seed: int, arrays: tuple, protocol: dict) -> tuple[JointPPOV42, list[dict]]:
    """BC uses only supplied training arrays; no cohort or audit path exists here."""
    observations, gates, proposals, joint = arrays
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    policy = JointPPOV42(88, learning_rate=protocol["learning_rate"])
    log = []
    for epoch in range(protocol["epochs"]):
        permutation = rng.permutation(len(observations))
        gradient_norms = []
        for start in range(0, len(observations), protocol["batch_size"]):
            indices = permutation[start:start + protocol["batch_size"]]
            gate_dist, proposal_dist = policy.model.distributions(observations[indices])
            loss, _ = hierarchical_bc_loss(gate_dist.logits, proposal_dist.logits, gates[indices], proposals[indices])
            policy.optimizer.zero_grad()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(policy.model.parameters(), 1., error_if_nonfinite=True)
            policy.optimizer.step()
            gradient_norms.append(float(norm))
        log.append({"epoch": epoch + 1, "metrics": _metrics(policy, *arrays), "finite_gradients": bool(np.isfinite(gradient_norms).all())})
    return policy, log


def reserve_once(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def train_and_qualify(output: Path, r4_path: Path, q_root: Path) -> dict:
    validate_r4(r4_path)
    protocol, _ = protocols()
    q_manifest_path, q_path = q_root / "manifest.json", q_root / "q2.jsonl"
    q_manifest = json.loads(q_manifest_path.read_text(encoding="utf-8"))
    if not q_manifest["Q2_FROZEN"] or not all(q_manifest["freeze_gates"].values()) or q_manifest["protocol_sha256"] != PROTOCOL_SHA:
        raise ValueError("fresh Q2 has not frozen under this protocol")
    q_manifest_sha = _sha(q_manifest_path)
    output.mkdir(parents=True, exist_ok=False)
    reserve_once(q_root / "i1_attempt.json", {"output": str(output), "protocol_sha256": PROTOCOL_SHA})
    state = {"phase": "TRAINING", "q_access_during_training": False}
    _install_q_access_guard((q_path, q_manifest_path, r4_path.parent / "internal_audit.jsonl"), state)
    try:
        rows = _rows(r4_path.parent / "training.jsonl")
        if len(rows) != protocol["training_rows"]:
            raise ValueError("R4 training count mismatch")
        arrays = _arrays(rows)
        config = replace(base_config(), observation_schema_version=protocol["observation_schema"])
        frozen = {}
        for seed in protocol["fresh_actor_seeds"]:
            _write(output / "progress.json", {"phase": "TRAINING", "seed": seed, "finished_seeds": list(frozen)})
            policy, log = fit(seed, arrays, protocol)
            path = output / f"seed{seed}_final.pt"
            save_checkpoint(path, policy, config, stage="V4_2_V2_1_I1_V2", fresh_actor=True,
                ppo_called=False, seed=seed, final_epoch=protocol["epochs"],
                training_dataset_sha256=protocol["training_dataset_sha256"],
                qualification_protocol_sha256=PROTOCOL_SHA, architecture_fingerprint=_architecture_fingerprint(policy))
            _write(output / f"seed{seed}_training.json", log)
            frozen[str(seed)] = {"checkpoint": str(path), "sha256": _sha(path), "finite_gradients": all(r["finite_gradients"] for r in log)}
        _write(output / "training_freeze.json", {"checkpoints": frozen, "protocol_sha256": PROTOCOL_SHA,
            "Q_DATASET_ACCESS_DURING_TRAINING": state["q_access_during_training"]})
        if state["q_access_during_training"]:
            raise RuntimeError("Q2 access during training invalidates qualification")
        state["phase"] = "Q_EVALUATION"
        reserve_once(q_root / "evaluation_consumed.json", {"output": str(output), "training_freeze_sha256": _sha(output / "training_freeze.json")})
        if _sha(q_path) != q_manifest["dataset_sha256"] or _sha(q_manifest_path) != q_manifest_sha:
            raise ValueError("Q2 changed after freeze")
        q_rows = _rows(q_path)
        if len(q_rows) != q_manifest["rows"]:
            raise ValueError("Q2 row count mismatch")
        results = {}
        for seed, entry in frozen.items():
            if _sha(Path(entry["checkpoint"])) != entry["sha256"]:
                raise ValueError("frozen checkpoint changed")
            policy = JointPPOV42(88)
            load_checkpoint(entry["checkpoint"], policy, config, load_optimizer=False)
            results[seed] = _evaluate_seed(policy, q_rows, protocol, entry["finite_gradients"])
        passed = all(r["PASS"] for r in results.values())
        canonical = frozen[str(protocol["canonical_checkpoint_seed"])]
        aggregates = {}
        for name in ("WAIT_recall", "RELEASE_recall", "gate_macro_F1", "proposal_top1", "proposal_top3", "decoded_accuracy"):
            values = np.asarray([row[name] for row in results.values()])
            aggregates[name] = {"mean": float(values.mean()), "std": float(values.std()),
                                "min": float(values.min()), "max": float(values.max())}
        result = {"stage": "V4_2_V2_1_I1_V2", "I1_V2": "PASS" if passed else "FAIL",
            "observation_schema": protocol["observation_schema"], "observation_dim": 88,
            "action_schema": protocol["action_schema"], "aggregate": aggregates,
            "protocol_sha256": PROTOCOL_SHA, "r4_manifest_sha256": R4_MANIFEST_SHA,
            "Q2_manifest_sha256": q_manifest_sha, "Q_EVALUATION_COUNT": 1,
            "Q_DATASET_ACCESS_DURING_TRAINING": False, "checkpoints": frozen, "seed_results": results,
            "canonical_checkpoint": canonical["checkpoint"] if passed else None,
            "canonical_checkpoint_sha256": canonical["sha256"] if passed else None,
            "I2_PREFLIGHT_ALLOWED": passed, "PPO_ALLOWED": False}
        _write(output / "qualification.json", result)
        return result
    finally:
        state["phase"] = "FINISHED"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--r4-manifest", type=Path, required=True)
    parser.add_argument("--history-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    q = collect_q2(args.output_dir / "q2", args.history_root, args.r4_manifest, smoke=args.smoke)
    print(json.dumps({"Q2_FROZEN": q["Q2_FROZEN"], "rows": q["rows"], "gates": q["freeze_gates"]}), flush=True)
    if q["Q2_FROZEN"]:
        result = train_and_qualify(args.output_dir / "i1", args.r4_manifest, args.output_dir / "q2")
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
