"""Bounded KL-recovery probe for the two stopped V6 formal runs.

This module is a diagnostic runner.  It restores one of the two stopped V6
checkpoints with its real Adam state, builds a fresh gate reference from the
fixed ``03c0`` source, and processes at most four diagnostic episodes.  It
does not resume the formal schedule and never changes the frozen V6 protocol.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from terminal_marl.algo.gate_reference_v4_2 import GateReference
from terminal_marl.training import run_v4_2_p3_gate_anchored_v6 as v6


STAGE = "P3_KL_RECOVERY_PROBE_V1"
RUNNER_MODULE = "terminal_marl.training.run_p3_kl_recovery_probe_v1"
OUTPUT_ROOT = Path("/home/user/training_outputs/terminal/p3_kl_recovery_probe_v1_20260926")
SOURCE_CHECKPOINT = Path(
    "/home/user/training_outputs/terminal/p3_gate_coverage_repair_v2_20260925/"
    "fit853401/p3_gate_coverage_repair_v2.pt"
)
SOURCE_CHECKPOINT_SHA256 = v6.SOURCE_CHECKPOINT_SHA256
SOURCE_CHECKPOINT_KIND = v6.SOURCE_CHECKPOINT_KIND
RUNS: dict[int, dict[str, Any]] = {
    0: {
        "checkpoint": Path(
            "/home/user/training_outputs/terminal/p3_gate_anchored_v6_20260925/"
            "formal/run0/checkpoints/checkpoint_episode_0079.pt"
        ),
        "checkpoint_sha256": "e6aab40d5dc33b06e9550c8d71058342e1b19d483439d24f2b2fbbf989457e31",
        "seeds": [860080, 860081, 860082, 860083],
        "scenarios": ["dense24", "flow6", "dense24", "flow12"],
    },
    2: {
        "checkpoint": Path(
            "/home/user/training_outputs/terminal/p3_gate_anchored_v6_20260925/"
            "formal/run2/checkpoints/checkpoint_episode_0048.pt"
        ),
        "checkpoint_sha256": "7b5c225d4cbcec9fc022797237c820e6d2240ab1a5018ba662700b9f32f02685",
        "seeds": [862049, 862050, 862051, 862052],
        "scenarios": ["flow6", "dense24", "flow12", "dense24"],
    },
}
DEV_SEEDS = [863101, 863102, 863103]
DEV_SCENARIOS = ["flow6", "flow12", "dense24"]
RECOVERY_STATUSES = frozenset({"UPDATED", "SKIPPED_KL", "HARD_FAILED"})


class ProbeInputError(ValueError):
    """Raised when a diagnostic input violates the frozen probe binding."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProbeInputError(message)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if hasattr(value, "__dict__"):
        return {str(key): item for key, item in vars(value).items()}
    if isinstance(value, tuple) and value and isinstance(value[0], str):
        result: dict[str, Any] = {"status": value[0]}
        if len(value) > 1:
            result["update"] = value[1]
        if len(value) > 2:
            result["diagnostics"] = value[2]
        return result
    raise ProbeInputError(f"recovery result must be a mapping-like value: {type(value).__name__}")


def _recovery_module() -> Any:
    return importlib.import_module("terminal_marl.training.p3_kl_recovery_v1")


def _protocol_context(
    run_index: int,
    reference_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _require(run_index in RUNS, "run-index must be 0 or 2")
    reference_options = {
        str(key): value for key, value in (reference_options or {}).items()
    }
    _require(
        set(reference_options).issubset({"soft_kl_target", "soft_kl_gain"}),
        "unsupported reference options",
    )
    protocol = v6._protocol()
    v6._validate_schedule(protocol)
    protocol_sha256 = v6._sha256(v6.PROTOCOL)
    configs, snapshots = v6._build_configs(protocol)
    source_policy, source_payload, trainable, source_sha = v6._load_source(
        SOURCE_CHECKPOINT, configs["dense24"], protocol,
    )
    old_rows, coverage_rows, weights, weight_report = v6._load_anchor_data(protocol)
    legacy_reference = v6._reference(
        source_policy, old_rows, coverage_rows, weights, protocol,
    )
    reference = legacy_reference
    if reference_options:
        # Rebuild directly from the fixed source policy and original merged
        # rows.  This avoids a second normalization pass over q/weights that
        # a dataclasses.replace call would perform.
        rows = [*old_rows, *coverage_rows]
        observations, _ = v6.coverage._rows_to_tensors(rows)
        optimizer = protocol["optimizer"]
        reference = GateReference.from_policy(
            source_policy,
            observations,
            weights,
            coefficient=float(optimizer["reference_gate_kl_coefficient"]),
            max_kl=float(optimizer["cumulative_reference_weighted_kl_hard_cap"]),
            **reference_options,
        )
        _require(
            v6._reference_digest(reference) == v6._reference_digest(legacy_reference),
            "soft reference changed the legacy q reference digest",
        )
    reference_digest = v6._reference_digest(reference)
    legacy_reference_digest = v6._reference_digest(legacy_reference)
    entry = RUNS[run_index]
    checkpoint = Path(entry["checkpoint"])
    _require(checkpoint.is_file(), f"stopped checkpoint missing: {checkpoint}")
    actual_checkpoint_sha = _sha256(checkpoint)
    _require(actual_checkpoint_sha == str(entry["checkpoint_sha256"]),
             f"stopped checkpoint SHA mismatch: {actual_checkpoint_sha}")
    restored = v6._restore_policy(checkpoint, configs["dense24"], protocol, reference)
    _require(bool(restored.optimizer.state), "stopped checkpoint Adam state is empty")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    _require(isinstance(payload, Mapping), "stopped checkpoint payload is not a mapping")
    _require(payload.get("source_checkpoint_sha256") == SOURCE_CHECKPOINT_SHA256,
             "stopped checkpoint source SHA binding changed")
    _require(payload.get("initial_checkpoint_sha256") == SOURCE_CHECKPOINT_SHA256,
             "stopped checkpoint initial source SHA binding changed")
    _require(payload.get("protocol_sha256") == protocol_sha256,
             "stopped checkpoint protocol SHA binding changed")
    _require(payload.get("reference_gate_digest") == reference_digest,
             "stopped checkpoint reference digest does not match 03c0 reference")
    initial_reference_metrics = reference.metrics(restored)
    initial_reference_kl = float(initial_reference_metrics.get("reference_kl", np.nan))
    _require(np.isfinite(initial_reference_kl) and
             initial_reference_kl <= float(reference.max_kl),
             "stopped checkpoint initial reference KL exceeds cap")
    model_tensors = list(restored.model.state_dict().values())
    optimizer_tensors = [value for value in restored.optimizer.state_dict().get("state", {}).values()
                         if isinstance(value, Mapping) for value in value.values()]
    _require(all(torch.isfinite(value).all().item() for value in model_tensors
                 if isinstance(value, torch.Tensor)),
             "stopped checkpoint model contains non-finite values")
    _require(all(torch.isfinite(value).all().item() for value in optimizer_tensors
                 if isinstance(value, torch.Tensor)),
             "stopped checkpoint Adam contains non-finite values")
    return {
        "run_index": run_index,
        "protocol": protocol,
        "protocol_sha256": protocol_sha256,
        "configs": configs,
        "config_snapshots": snapshots,
        "runtime_protocol": v6._runtime_protocol(protocol),
        "source_checkpoint": SOURCE_CHECKPOINT,
        "source_checkpoint_sha256": source_sha,
        "source_payload": source_payload,
        "trainable": trainable,
        "old_rows": old_rows,
        "coverage_rows": coverage_rows,
        "weight_report": weight_report,
        "reference": reference,
        "reference_digest": reference_digest,
        "legacy_reference_digest": legacy_reference_digest,
        "reference_options": dict(reference_options) if reference_options else None,
        "initial_reference_metrics": initial_reference_metrics,
        "base_checkpoint": checkpoint,
        "base_checkpoint_sha256": actual_checkpoint_sha,
        "policy": restored,
    }


def _check_only_payload(context: Mapping[str, Any], output: Path) -> dict[str, Any]:
    reference = context.get("reference")
    initial_metrics = context.get("initial_reference_metrics") or {}
    return {
        "stage": str(context.get("stage", STAGE)),
        "status": "CHECK_ONLY_OK",
        "check_only": True,
        "run_index": int(context["run_index"]),
        "output_created": False,
        "output_existed": bool(output.exists()),
        "output": str(output),
        "source_checkpoint": str(context["source_checkpoint"]),
        "source_checkpoint_sha256": context["source_checkpoint_sha256"],
        "source_checkpoint_kind": SOURCE_CHECKPOINT_KIND,
        "base_checkpoint": str(context["base_checkpoint"]),
        "base_checkpoint_sha256": context["base_checkpoint_sha256"],
        "protocol_sha256": context["protocol_sha256"],
        "reference_digest": context["reference_digest"],
        "reference_options": dict(context.get("reference_options") or {}),
        "reference_gate_coefficient": (
            float(getattr(reference, "coefficient"))
            if hasattr(reference, "coefficient") else None
        ),
        "reference_gate_max_kl": (
            float(getattr(reference, "max_kl"))
            if hasattr(reference, "max_kl") else None
        ),
        "effective_kl_coefficient": initial_metrics.get("effective_kl_coefficient"),
        "trainable_parameters": list(context["trainable"]),
        "fresh_adam": False,
        "restored_adam": bool(context["policy"].optimizer.state),
        "actual_config_snapshots": context["config_snapshots"],
        "heldout_executed": False,
        "formal_continuation": False,
        "P3_QUALIFIED": False,
        "P4_ALLOWED": False,
        "PPO_ALLOWED": False,
    }


def _dev_run(context: Mapping[str, Any], output: Path, *, checkpoint: Path,
             label: str, policy: Any, rng_state: Mapping[str, Any]) -> dict[str, Any]:
    v6._restore_rng_state(rng_state)
    return v6._dev_evaluate(
        policy=policy,
        configs=context["configs"],
        runtime_protocol=context["runtime_protocol"],
        v6=context["protocol"],
        output=output,
        accepted_episodes=0,
        checkpoint=checkpoint,
        seeds=DEV_SEEDS,
        scenarios=DEV_SCENARIOS,
        protocol_sha256=context["protocol_sha256"],
    ) | {"label": label}


def _call_recover_update(context: Mapping[str, Any], record: Mapping[str, Any]) -> dict[str, Any]:
    recovery = _recovery_module()
    function = getattr(recovery, "recover_update", None)
    _require(callable(function), "p3_kl_recovery_v1.recover_update is missing")
    kwargs = {
        "policy": context["policy"],
        "steps": record.get("_steps", []),
        "config": context["configs"][str(record["scenario"])],
        "tape_path": str(record.get("action_tape")),
        "runtime_protocol": context["runtime_protocol"],
        "reference": context["reference"],
        "old_rows": context["old_rows"],
        "coverage_rows": context["coverage_rows"],
    }
    try:
        result = function(**kwargs)
    except Exception as exc:
        return {
            "status": "HARD_FAILED",
            "exception": {"type": type(exc).__name__, "message": str(exc)},
        }
    result = _as_mapping(result)
    status = str(result.get("status", ""))
    _require(status in RECOVERY_STATUSES, f"unknown recovery status: {status!r}")
    return result


def _checkpoint(context: Mapping[str, Any], output: Path, episode_index: int,
                timestep: int, config: Any, *, ppo_called: bool,
                reference_options: Mapping[str, Any] | None = None) -> tuple[Any, bool, Path]:
    checkpoint = output / "checkpoints" / f"checkpoint_episode_{episode_index:04d}.pt"
    restored, exact = v6._save_and_restore(
        context["policy"], checkpoint, config, context["protocol"],
        context["reference"], episode_index=episode_index, timestep=timestep,
        trainable=context["trainable"], protocol_sha256=context["protocol_sha256"],
        formal=False, ppo_called=ppo_called,
    )
    if reference_options is not None:
        sidecar = checkpoint.with_name(checkpoint.name + ".reference.json")
        v6._write_json(sidecar, {
            "stage": str(context.get("stage", STAGE)),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint) if checkpoint.is_file() else None,
            "reference_digest": context["reference_digest"],
            "reference_gate_coefficient": float(context["reference"].coefficient),
            "reference_gate_max_kl": float(context["reference"].max_kl),
            "soft_kl_target": getattr(context["reference"], "soft_kl_target", None),
            "soft_kl_gain": float(getattr(context["reference"], "soft_kl_gain", 0.0)),
            "reference_options": dict(reference_options),
        })
    return restored, bool(exact), checkpoint


def _gate_adapter_snapshot(policy: Any, trainable: Sequence[str]) -> dict[str, torch.Tensor]:
    state = policy.model.state_dict()
    return {
        str(name): state[str(name)].detach().cpu().clone()
        for name in trainable
        if str(name).startswith("gate_adapter.") and str(name) in state
    }


def _gate_adapter_delta_norm(before: Mapping[str, torch.Tensor], policy: Any) -> float:
    state = policy.model.state_dict()
    squared = 0.0
    for name, previous in before.items():
        current = state.get(name)
        if current is None:
            continue
        delta = current.detach().cpu().to(dtype=torch.float64) - previous.to(dtype=torch.float64)
        squared += float(torch.sum(delta * delta).item())
    return float(np.sqrt(max(squared, 0.0)))


def _run(
    context: dict[str, Any],
    output: Path,
    entry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    entry = dict(entry or RUNS[int(context["run_index"])])
    seeds = list(entry["seeds"])
    scenarios = list(entry["scenarios"])
    _require(len(seeds) == len(scenarios), "probe seed/scenario plan length changed")
    stage = str(context.get("stage", STAGE))
    reference_options = context.get("reference_options")
    report_gate_delta = bool(context.get("report_gate_adapter_delta", False))
    diagnostic_replay_count = int(entry.get("diagnostic_replay_count", 0))
    _require(0 <= diagnostic_replay_count <= len(seeds),
             "diagnostic replay count is outside the episode plan")
    output.mkdir(parents=True, exist_ok=False)
    checkpoints_dir = output / "checkpoints"
    records: list[dict[str, Any]] = []
    processed = updated = skipped = hard_failed = 0
    timestep = 0
    stop_reason: str | None = None
    current_checkpoint = Path(context["base_checkpoint"])
    train_rng = v6._rng_state()
    dev_rng = v6._rng_state()
    before = _dev_run(context, output / "dev_before", checkpoint=current_checkpoint,
                      label="before", policy=context["policy"], rng_state=dev_rng)
    if before["hard_failure"] or not before["full_resolution"]:
        stop_reason = "before development hard failure"
    else:
        v6._restore_rng_state(train_rng)
    for episode_index, (seed, scenario) in enumerate(zip(seeds, scenarios), start=1):
        if stop_reason is not None:
            break
        config = context["configs"][scenario]
        horizon = int(round(float(config.max_simulation_seconds) / float(config.dt)))
        episode_started = time.perf_counter()
        record = admission_record = v6.admission._record_rollout(
            policy=context["policy"], config=config, p=context["runtime_protocol"],
            scenario=scenario, seed=int(seed), output=output / "episodes" / f"episode_{episode_index:04d}_{scenario}_{seed}",
            label=f"recovery_{context['run_index']}", deterministic=False,
            max_steps=horizon, require_full_resolution=False, checkpoint=current_checkpoint,
        )
        record["run_index"] = int(context["run_index"])
        record["episode_index"] = int(episode_index)
        record["diagnostic_seed"] = int(seed)
        record["scenario"] = scenario
        record["source_checkpoint_sha256"] = context["source_checkpoint_sha256"]
        record["base_checkpoint_sha256"] = _sha256(current_checkpoint)
        record["protocol_sha256"] = context["protocol_sha256"]
        record["old_formal_seed"] = bool(episode_index == 1)
        record["non_independent_diagnostic_replay"] = True
        if diagnostic_replay_count:
            record["seed_consumption"] = (
                "diagnostic_replay_reproduction"
                if episode_index <= diagnostic_replay_count
                else "previously_unconsumed_planned"
            )
            record["diagnostic_replay_reproduction"] = bool(
                episode_index <= diagnostic_replay_count
            )
            record["previously_unconsumed_planned"] = bool(
                episode_index > diagnostic_replay_count
            )
        if report_gate_delta:
            record["gate_adapter_parameter_delta_norm"] = 0.0
            record["accepted_minibatches"] = 0
            initial_metrics = context.get("initial_reference_metrics") or {}
            record["effective_kl_coefficient"] = initial_metrics.get(
                "effective_kl_coefficient"
            )
        record["timing_rollout_wall_seconds"] = float(time.perf_counter() - episode_started)
        hard_pass = bool(v6._training_pass(record))
        record["rollout_hard_pass"] = hard_pass
        processed += 1
        if not hard_pass:
            hard_failed += 1
            record["recovery"] = {"status": "HARD_FAILED", "reason": "rollout hard gate"}
            stop_reason = "rollout hard failure"
        else:
            gate_snapshot = (
                _gate_adapter_snapshot(context["policy"], context["trainable"])
                if report_gate_delta else None
            )
            recovery_result = _call_recover_update(context, record)
            record["recovery"] = recovery_result
            if report_gate_delta:
                record["gate_adapter_parameter_delta_norm"] = _gate_adapter_delta_norm(
                    gate_snapshot or {}, context["policy"]
                )
                update = recovery_result.get("update")
                record["accepted_minibatches"] = int(
                    update.get("accepted_updates", 0)
                    if isinstance(update, Mapping) else 0
                )
                metrics = recovery_result.get("update", {}).get(
                    "reference_gate_after", {}
                ) if isinstance(recovery_result.get("update"), Mapping) else {}
                if isinstance(metrics, Mapping):
                    record["effective_kl_coefficient"] = metrics.get(
                        "effective_kl_coefficient"
                    )
            status = str(recovery_result.get("status"))
            if status not in {"UPDATED", "SKIPPED_KL"}:
                hard_failed += 1
                stop_reason = "recovery hard failure"
            else:
                recovery_started = time.perf_counter()
                try:
                    restored, exact, checkpoint = _checkpoint(
                        context, output, episode_index, timestep + len(record.get("_steps", [])), config,
                        ppo_called=True,
                        **({"reference_options": reference_options}
                           if reference_options is not None else {}),
                    )
                    record["checkpoint_after"] = str(checkpoint)
                    record["checkpoint_after_sha256"] = _sha256(checkpoint)
                    record["checkpoint_restore_exact"] = bool(exact)
                    record["timing_recovery_checkpoint_wall_seconds"] = float(
                        time.perf_counter() - recovery_started
                    )
                    if exact:
                        context["policy"] = restored
                        current_checkpoint = checkpoint
                        timestep += len(record.get("_steps", []))
                        if status == "UPDATED":
                            updated += 1
                        else:
                            skipped += 1
                    else:
                        hard_failed += 1
                        stop_reason = "checkpoint restore integrity failure"
                except Exception as exc:
                    record["checkpoint_restore_exact"] = False
                    record["checkpoint_error"] = {"type": type(exc).__name__, "message": str(exc)}
                    record["timing_recovery_checkpoint_wall_seconds"] = float(
                        time.perf_counter() - recovery_started
                    )
                    hard_failed += 1
                    stop_reason = "checkpoint save or restore failure"
        record["timing_episode_wall_seconds"] = float(time.perf_counter() - episode_started)
        record["ppo_accepted"] = bool(
            record.get("recovery", {}).get("status") == "UPDATED"
            and record.get("checkpoint_restore_exact") is True
        )
        public = v6.formal_v5._public_record(record)
        public["recovery"] = v6._jsonable(record.get("recovery", {}))
        public.update({key: record[key] for key in (
            "run_index", "episode_index", "diagnostic_seed", "scenario", "source_checkpoint_sha256",
            "base_checkpoint_sha256", "protocol_sha256", "old_formal_seed",
            "non_independent_diagnostic_replay", "rollout_hard_pass", "checkpoint_restore_exact",
            "checkpoint_after", "checkpoint_after_sha256", "ppo_accepted",
            "seed_consumption", "diagnostic_replay_reproduction",
            "previously_unconsumed_planned", "gate_adapter_parameter_delta_norm",
            "accepted_minibatches", "effective_kl_coefficient",
        ) if key in record})
        records.append(public)
        v6._write_json(output / "progress.json", {
            "stage": stage, "run_index": context["run_index"], "processed_episodes": processed,
            "updated_episodes": updated, "skipped_kl_episodes": skipped,
            "hard_failed_episodes": hard_failed, "records": records,
            "stopped": stop_reason is not None, "stop_reason": stop_reason,
        })
        if stop_reason is not None:
            break
    after = None
    if stop_reason is None:
        after_policy = context["policy"]
        v6._restore_rng_state(dev_rng)
        after = _dev_run(context, output / "dev_after", checkpoint=current_checkpoint,
                         label="after", policy=after_policy, rng_state=dev_rng)
        if after["hard_failure"] or not after["full_resolution"]:
            stop_reason = "after development hard failure"
    reference = context.get("reference")
    manifest = {
        "stage": stage,
        "status": "COMPLETED" if stop_reason is None and processed == len(seeds) else "STOPPED",
        "run_index": context["run_index"],
        "output": str(output),
        "source_checkpoint": str(context["source_checkpoint"]),
        "source_checkpoint_sha256": context["source_checkpoint_sha256"],
        "source_checkpoint_kind": SOURCE_CHECKPOINT_KIND,
        "base_checkpoint": str(context["base_checkpoint"]),
        "base_checkpoint_sha256": context["base_checkpoint_sha256"],
        "protocol_sha256": context["protocol_sha256"],
        "reference_digest": context["reference_digest"],
        "reference_options": dict(reference_options or {}),
        "reference_gate_coefficient": (
            float(getattr(reference, "coefficient"))
            if hasattr(reference, "coefficient") else None
        ),
        "reference_gate_max_kl": (
            float(getattr(reference, "max_kl"))
            if hasattr(reference, "max_kl") else None
        ),
        "reference_soft_settings": {
            "soft_kl_target": getattr(reference, "soft_kl_target", None),
            "soft_kl_gain": float(getattr(reference, "soft_kl_gain", 0.0)),
        },
        "seeds": seeds, "scenarios": scenarios,
        "diagnostic_replay_count": diagnostic_replay_count,
        "seed_consumption_plan": [
            {
                "episode_index": index,
                "seed": int(seed),
                "scenario": scenario,
                "classification": (
                    "diagnostic_replay_reproduction"
                    if index <= diagnostic_replay_count
                    else "previously_unconsumed_planned"
                ) if diagnostic_replay_count else "standard_probe",
            }
            for index, (seed, scenario) in enumerate(zip(seeds, scenarios), start=1)
        ],
        "diagnostic_non_independent": True, "formal_continuation": False,
        "episodes_requested": len(seeds), "processed_episodes": processed,
        "updated_episodes": updated, "skipped_kl_episodes": skipped,
        "hard_failed_episodes": hard_failed, "accepted_episodes": 0,
        "records": records, "development_before": before, "development_after": after,
        "performance_comparison": v6._comparison(before, after) if after is not None else None,
        "stop_reason": stop_reason, "heldout_executed": False,
        "P3_QUALIFIED": False, "P4_ALLOWED": False, "PPO_ALLOWED": False,
        "checkpoints": [record.get("checkpoint_after") for record in records if record.get("checkpoint_after")],
        "wall_seconds": None,
    }
    v6._write_json(output / "manifest.json", manifest)
    return manifest


def run(*, run_index: int, output: str | Path, check_only: bool = False) -> dict[str, Any]:
    _require(run_index in RUNS, "run-index must be 0 or 2")
    target = Path(output)
    if check_only:
        context = _protocol_context(run_index)
        return _check_only_payload(context, target)
    _require(not target.exists(), f"output already exists: {target}")
    context = _protocol_context(run_index)
    return _run(context, target)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-index", type=int, choices=(0, 2), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(run_index=args.run_index, output=args.output, check_only=args.check_only)
    except (ProbeInputError, FileNotFoundError, ValueError) as exc:
        print(json.dumps({"stage": STAGE, "status": "ERROR", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(v6._jsonable(result), indent=2, sort_keys=True))
    return 0 if result.get("status") in {"CHECK_ONLY_OK", "COMPLETED", "STOPPED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
