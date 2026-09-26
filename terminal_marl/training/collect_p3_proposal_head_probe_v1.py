"""Collect frozen source proposal-head rows from existing V7 development tapes.

This is an inference-only collector.  It replays existing action tapes, observes
pre-action states, and evaluates the source proposal head on the stored 88D
observations.  It never supplies policy or Teacher actions to the live replay.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import tempfile
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from terminal_marl.core.safety_trace import SafetyTraceRecorder, _digest
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training import diagnose_p3_k3_coverage_v1 as diagnostic
from terminal_marl.training import evaluate_p3_adaptive_kl_v7 as evaluator


STAGE = "P3_PROPOSAL_HEAD_PROBE_V1"
PROTOCOL = Path("docs/evidence/v4_2/p3_proposal_head_probe_v1_20260926.md")
PROTOCOL_SHA256 = "9564af249f85c79c53d97dd5f742c2f0585c35695984bd5ebdb3865870656b18"
SOURCE_CHECKPOINT = Path(
    "/home/user/training_outputs/terminal/p3_gate_coverage_repair_v2_20260925/fit853401/"
    "p3_gate_coverage_repair_v2.pt"
)
SOURCE_CHECKPOINT_SHA256 = "03c0acb0d561219cbe1dc8dac88cd4df696ade4e4691ff0d4db1d4a72ed5e3ba"
FORMAL_ROOT = Path("/home/user/training_outputs/terminal/p3_adaptive_kl_formal_v7_20260926")
SOURCE_MANIFEST = FORMAL_ROOT / "run1" / "dev_before_training" / "manifest.json"
OUTPUT_ROOT = Path("/home/user/training_outputs/terminal/p3_proposal_head_probe_v1_20260926")
SMOKE_ROOT = Path("/home/user/training_outputs/terminal/p3_proposal_head_probe_v1_20260926_smoke")
SOURCE_SEEDS = tuple(range(873101, 873107))
TRAIN_SEEDS = tuple(range(873101, 873104))
VALIDATION_SEEDS = tuple(range(873104, 873107))
SLOT_COUNT = 32
OBSERVATION_DIM = 88


class CollectionError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CollectionError(message)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "value") and isinstance(value.value, str):
        return value.value
    return value


def _read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"JSON input missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rng_snapshot() -> tuple[Any, tuple[Any, ...], torch.Tensor]:
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def _rng_equal(before: tuple[Any, tuple[Any, ...], torch.Tensor], after: tuple[Any, tuple[Any, ...], torch.Tensor]) -> bool:
    return diagnostic._rng_equal(before, after)


def _target_steps(step_count: int) -> list[int]:
    _require(step_count > 0, "tape must contain at least one step")
    return [int(math.floor((step_count - 1) * index / SLOT_COUNT)) for index in range(SLOT_COUNT)]


def _split_for_seed(seed: int) -> str:
    if int(seed) in TRAIN_SEEDS:
        return "train"
    if int(seed) in VALIDATION_SEEDS:
        return "validation"
    raise CollectionError(f"unexpected source seed: {seed}")


def _resolve_path(value: Any, relative_to: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else relative_to / path


def _protocol_check() -> dict[str, Any]:
    _require(PROTOCOL.is_file(), f"probe protocol missing: {PROTOCOL}")
    _require(_sha256(PROTOCOL) == PROTOCOL_SHA256, "probe protocol SHA changed")
    text = PROTOCOL.read_text(encoding="utf-8")
    for required in ("P3 proposal-head probe V1", "32", "192", "REGISTERED_PENDING_SMOKE"):
        _require(required in text, f"probe protocol binding missing: {required}")
    return {
        "path": str(PROTOCOL),
        "sha256": PROTOCOL_SHA256,
        "stage": STAGE,
        "slot_count": SLOT_COUNT,
        "max_rows": 192,
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
    }


def _load_source_inputs(formal_root: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], Path]:
    protocol = evaluator._protocol()
    configs = evaluator._build_configs(protocol)
    manifest_path = formal_root / "run1" / "dev_before_training" / "manifest.json"
    manifest = _read_json(manifest_path)
    records = [dict(row) for row in manifest.get("records", []) if isinstance(row, Mapping)]
    _require(len(records) == 6, f"source manifest expected 6 records, got {len(records)}")
    records.sort(key=lambda row: int(row.get("seed", -1)))
    _require(sorted(int(row.get("seed", -1)) for row in records) == list(SOURCE_SEEDS), "source seed set changed")
    _require(
        [str(row.get("scenario")) for row in records]
        == ["flow6", "flow12", "dense24", "flow6", "flow12", "dense24"],
        "source scenario order changed",
    )
    for row in records:
        seed = int(row["seed"])
        expected_split = _split_for_seed(seed)
        _require(str(row.get("split", expected_split)) == expected_split, f"source split changed for seed {seed}")
        _require(str(row.get("scenario")) in {"flow6", "flow12", "dense24"}, "source scenario changed")
        tape = _resolve_path(row.get("action_tape"), manifest_path.parent)
        _require(tape.is_file(), f"source tape missing: {tape}")
        row["action_tape"] = str(tape)
        row["split"] = expected_split
    source = SOURCE_CHECKPOINT
    _require(source.is_file(), f"source checkpoint missing: {source}")
    _require(_sha256(source) == SOURCE_CHECKPOINT_SHA256, "source checkpoint SHA mismatch")
    return protocol, configs, records, manifest_path


def _load_source_policy(protocol: Mapping[str, Any], config: Any) -> dict[str, Any]:
    protocol_sha = evaluator._sha256(evaluator.PROTOCOL)
    learning_rate = float((protocol.get("optimizer") or {}).get("learning_rate", 3e-4))
    policy, payload, actual_sha = evaluator._load_eval_policy(
        SOURCE_CHECKPOINT,
        config=config,
        expected_checkpoint_sha256=SOURCE_CHECKPOINT_SHA256,
        protocol_sha256=protocol_sha,
        final_checkpoint=False,
        learning_rate=learning_rate,
    )
    policy.model.eval()
    return {
        "policy": policy,
        "payload": payload,
        "sha256": actual_sha,
        "model_digest_before": evaluator._model_digest(policy),
        "checkpoint": str(SOURCE_CHECKPOINT),
    }


def _row_step_inputs(row: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, tuple[np.ndarray, ...]]]:
    actions = {str(uid): np.asarray(action, dtype=np.float32) for uid, action in (row.get("a_requested") or {}).items()}
    shortlists = {
        str(uid): tuple(np.asarray(action, dtype=np.float32) for action in values)
        for uid, values in (row.get("runtime_proposal_shortlists") or {}).items()
    }
    return actions, shortlists


def _state_row(env: TerminalParallelEnv, uid: int, *, policy: Any, seed: int, split: str, step: int,
               target_step: int, slot_index: int) -> dict[str, Any]:
    actor = env.active.get(int(uid))
    _require(actor is not None, f"selected actor disappeared: uid={uid}")
    observations = env._observations()
    observation = observations.get(str(uid), observations.get(int(uid)))
    _require(observation is not None, f"observation missing for uid={uid}")
    observation = np.asarray(observation, dtype=np.float32)
    _require(observation.ndim == 1 and observation.size == OBSERVATION_DIM, "stored observation is not finite 88D")
    _require(bool(np.isfinite(observation).all()), "stored observation is not finite 88D")
    shadow = copy.deepcopy(env)
    diagnostic._restore_shadow_methods(shadow)
    shadow.config = replace(shadow.config, reconcile_late_merge_slots_enabled=False)
    live_digest_before = _digest(SafetyTraceRecorder._state(env))
    rng_before = _rng_snapshot()
    catalogue = diagnostic._catalogue(shadow, int(uid))
    forward = diagnostic._policy_forward(policy, observation, set(catalogue["certified_proposal_indices"]))
    live_digest_after = _digest(SafetyTraceRecorder._state(env))
    rng_after = _rng_snapshot()
    _require(live_digest_before == live_digest_after, "live environment changed during source probe")
    _require(_rng_equal(rng_before, rng_after), "RNG changed during source probe")
    state = diagnostic._state(env, int(uid))
    return {
        "observation": _jsonable(observation),
        "source_proposal_logits": forward["proposal_logits"],
        "source_gate_logits": forward["gate_logits"],
        "certified_proposals": sorted(int(value) for value in catalogue["certified_proposal_indices"]),
        "proposal_catalogue_indices": catalogue["proposal_catalogue_indices"],
        "proposal_mapping_complete": bool(catalogue["proposal_mapping_complete"]),
        "seed": int(seed),
        "split": str(split),
        "step": int(step),
        "target_step": int(target_step),
        "slot_index": int(slot_index),
        "uid": int(uid),
        "arrival_id": int(actor.arrival_id),
        "soc": float(actor.soc),
        "priority": int(actor.base_priority),
        "phase": getattr(actor.phase, "value", actor.phase),
        "ring_role": None if actor.ring_role is None else getattr(actor.ring_role, "value", actor.ring_role),
        "state": state,
        "state_probe_live_digest_before": live_digest_before,
        "state_probe_live_digest_after": live_digest_after,
        "state_probe_rng_unchanged": True,
        "teacher_state_only": True,
        "late_merge_slots_enabled": False,
    }


def _summary_outcomes(tape: Path) -> dict[str, Any]:
    summary = _read_json(tape.parent.parent / "summary.json")
    outcomes = summary.get("outcomes") or {}
    return {str(key): value for key, value in outcomes.items()} if isinstance(outcomes, Mapping) else {}


def _replay_outcomes(env: TerminalParallelEnv) -> dict[str, Any]:
    return {str(key): getattr(value, "value", value) for key, value in env.ledger.outcomes.items()}


def _collect_tape(*, record: Mapping[str, Any], config: Any, source_policy: Mapping[str, Any],
                  targets_override: list[int] | None = None, validate_outcomes: bool = True,
                  source_tape: Path | None = None) -> dict[str, Any]:
    tape = source_tape or Path(str(record["action_tape"]))
    payload = _read_json(tape)
    rows = list(payload.get("action_tape", []))
    _require(rows, f"empty action tape: {tape}")
    targets = list(targets_override) if targets_override is not None else _target_steps(len(rows))
    config_binding = diagnostic._config_binding(config, tape)
    env = TerminalParallelEnv(config)
    original_step = env.step
    pending = list(enumerate(targets))
    samples: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    replay_rng_before = _rng_snapshot()
    policy_digest_before = evaluator._model_digest(source_policy["policy"])

    def observed_step(*args: Any, **kwargs: Any) -> Any:
        step = int(env.step_count)
        if pending and step >= int(pending[0][1]):
            slot_index, target_step = pending[0]
            actors = diagnostic._eligible_actors(env)
            if actors:
                actor = actors[int(slot_index) % len(actors)]
                pending.pop(0)
                try:
                    samples.append(_state_row(
                        env,
                        int(actor.uid),
                        policy=source_policy["policy"],
                        seed=int(record["seed"]),
                        split=str(record["split"]),
                        step=step,
                        target_step=int(target_step),
                        slot_index=int(slot_index),
                    ))
                except Exception as exc:
                    error = {"slot_index": int(slot_index), "target_step": int(target_step),
                             "type": type(exc).__name__, "message": str(exc)}
                    errors.append(error)
                    samples.append({
                        "status": "error", "seed": int(record["seed"]), "split": str(record["split"]),
                        "step": step, "target_step": int(target_step), "slot_index": int(slot_index),
                        "uid": int(actor.uid), "error": error,
                    })
        return original_step(*args, **kwargs)

    env.step = observed_step  # type: ignore[method-assign]
    replay_error = None
    try:
        replay = SafetyTraceRecorder.replay_action_tape(env, tape)
    except Exception as exc:
        replay_error = {"type": type(exc).__name__, "message": str(exc)}
        replay = {"exact": False, "error": replay_error, "steps_replayed": 0, "steps_expected": len(rows)}
    finally:
        env.step = original_step  # type: ignore[method-assign]
    for slot_index, target_step in pending:
        missing.append({"slot_index": int(slot_index), "target_step": int(target_step), "status": "missing_runtime_actor"})
    replay_rng_after = _rng_snapshot()
    policy_digest_after = evaluator._model_digest(source_policy["policy"])
    exact = bool(replay.get("exact")) and replay_error is None
    outcomes_match = None
    if validate_outcomes and replay_error is None:
        try:
            outcomes_match = _summary_outcomes(tape) == _replay_outcomes(env)
        except Exception as exc:
            errors.append({"type": type(exc).__name__, "message": str(exc), "kind": "outcome_check"})
            outcomes_match = False
    if replay_error is not None:
        errors.append({**replay_error, "kind": "replay"})
    return {
        "scenario": str(record.get("scenario")),
        "seed": int(record["seed"]),
        "split": str(record["split"]),
        "action_tape": str(record["action_tape"]),
        "replayed_tape": str(tape),
        "action_tape_sha256": _sha256(tape),
        "steps_expected": len(rows),
        "steps_replayed": replay.get("steps_replayed"),
        "target_steps": targets,
        "config_binding": config_binding,
        "replay_exact": exact,
        "outcomes_match": outcomes_match,
        "rng_unchanged": _rng_equal(replay_rng_before, replay_rng_after),
        "model_weights_unchanged": policy_digest_before == policy_digest_after,
        "samples": sorted(samples, key=lambda row: (int(row.get("slot_index", 0)), int(row.get("step", 0)))),
        "missing_slots": missing,
        "errors": errors,
        "rows_observed": len([row for row in samples if row.get("status", "evaluated") == "evaluated"]),
    }


def _manifest_status(mode: str, reports: list[Mapping[str, Any]], rows: list[Mapping[str, Any]],
                     *, source_model_digest_unchanged: bool = True) -> str:
    if mode == "smoke":
        if len(reports) != 1:
            return "FAILED"
        report = reports[0]
        return "SMOKE_COMPLETED" if (
            report.get("replay_exact") is True
            and report.get("rng_unchanged") is True
            and report.get("model_weights_unchanged") is True
            and source_model_digest_unchanged
            and not report.get("errors")
            and len(report.get("samples", [])) == 1
            and not report.get("missing_slots")
            and report["samples"][0].get("proposal_mapping_complete") is True
            and set(report["samples"][0].get("proposal_catalogue_indices", [])) == set(range(25))
        ) else "FAILED"
    return "COMPLETED" if (
        len(reports) == 6
        and len(rows) <= 192
        and all(report.get("replay_exact") is True for report in reports)
        and all(report.get("outcomes_match") is True for report in reports)
        and all(report.get("rng_unchanged") is True for report in reports)
        and all(report.get("model_weights_unchanged") is True for report in reports)
        and source_model_digest_unchanged
        and all(not report.get("errors") for report in reports)
    ) else "FAILED"


def collect(*, output_dir: Path, formal_root: Path = FORMAL_ROOT, smoke: bool = False) -> dict[str, Any]:
    protocol_binding = _protocol_check()
    _require(not output_dir.exists(), f"refusing existing output: {output_dir}")
    protocol, configs, records, manifest_path = _load_source_inputs(formal_root)
    source_policy = _load_source_policy(protocol, configs[str(records[0]["scenario"])])
    output_dir.mkdir(parents=True, exist_ok=False)
    reports: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    mode = "smoke" if smoke else "full"
    _write_json(output_dir / "progress.json", {"stage": STAGE, "mode": mode, "status": "RUNNING", "processed": 0})
    try:
        selected = records[:1] if smoke else records
        for index, record in enumerate(selected):
            config = configs[str(record["scenario"])]
            tape = Path(str(record["action_tape"]))
            if smoke:
                payload = _read_json(tape)
                first_step = diagnostic._first_eligible_prefix_step(tape, config)
                prefix_payload = dict(payload)
                prefix_payload["action_tape"] = payload["action_tape"][: first_step + 1]
                with tempfile.TemporaryDirectory(prefix="p3_proposal_head_smoke_") as directory:
                    prefix_path = Path(directory) / "prefix.json"
                    _write_json(prefix_path, prefix_payload)
                    report = _collect_tape(
                        record=record,
                        config=config,
                        source_policy=source_policy,
                        targets_override=[0],
                        validate_outcomes=False,
                        source_tape=prefix_path,
                    )
                report["prefix_rows"] = first_step + 1
                report["first_sample_step"] = first_step
            else:
                report = _collect_tape(record=record, config=config, source_policy=source_policy)
            reports.append(report)
            for row in report.get("samples", []):
                if row.get("status", "evaluated") == "evaluated":
                    all_rows.append(row)
            _write_json(output_dir / "progress.json", {
                "stage": STAGE, "mode": mode, "status": "RUNNING", "processed": index + 1,
                "expected": len(selected), "rows": len(all_rows),
            })
        source_model_digest_after = evaluator._model_digest(source_policy["policy"])
        source_model_digest_unchanged = source_policy["model_digest_before"] == source_model_digest_after
        status = _manifest_status(
            mode,
            reports,
            all_rows,
            source_model_digest_unchanged=source_model_digest_unchanged,
        )
        result = {
            "stage": STAGE,
            "status": status,
            "mode": mode,
            "protocol": protocol_binding,
            "source_manifest": str(manifest_path),
            "source_manifest_sha256": _sha256(manifest_path),
            "source_checkpoint": str(SOURCE_CHECKPOINT),
            "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
            "source_model_digest_before": source_policy["model_digest_before"],
            "source_model_digest_after": source_model_digest_after,
            "source_model_digest_unchanged": source_model_digest_unchanged,
            "records_expected": len(selected),
            "rows_expected": 1 if smoke else 192,
            "rows_observed": len(all_rows),
            "reports": reports,
            "teacher_online": False,
            "policy_sampling": False,
            "policy_update": False,
            "optimizer_used": False,
            "late_merge_slots_enabled": False,
            "P3_QUALIFIED": False,
            "P4_ALLOWED": False,
            "PPO_ALLOWED": False,
        }
        _write_json(output_dir / "rows.json", all_rows)
        _write_json(output_dir / "manifest.json", result)
        _write_json(output_dir / "progress.json", {"stage": STAGE, "mode": mode, "status": status,
                                                     "processed": len(selected), "rows": len(all_rows)})
        print(json.dumps({"stage": STAGE, "status": status, "output": str(output_dir), "rows": len(all_rows)}, sort_keys=True), flush=True)
        return result
    except Exception as exc:
        result = {
            "stage": STAGE, "status": "FAILED", "mode": mode, "output": str(output_dir),
            "reports": reports, "rows_observed": len(all_rows),
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "P3_QUALIFIED": False, "P4_ALLOWED": False, "PPO_ALLOWED": False,
        }
        _write_json(output_dir / "rows.json", all_rows)
        _write_json(output_dir / "manifest.json", result)
        _write_json(output_dir / "progress.json", result)
        print(json.dumps(result, sort_keys=True), flush=True)
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--formal-root", type=Path, default=FORMAL_ROOT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = args.output_dir or (SMOKE_ROOT if args.smoke else OUTPUT_ROOT / "collection")
    result = collect(output_dir=output, formal_root=args.formal_root, smoke=bool(args.smoke))
    return 0 if result["status"] in {"SMOKE_COMPLETED", "COMPLETED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
