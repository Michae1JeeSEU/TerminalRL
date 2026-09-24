"""Compare the frozen canonical initial policy with the repaired candidate.

This development-only runner keeps one current dense24 configuration and
reuses the existing full and smoke arm checks.  It does not qualify,
promote, train, or authorize PPO.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from terminal_marl.training import diagnose_trained_dense30_v2 as diagnostic
from terminal_marl.training import evaluate_random_soc_arbitration_repair_v1 as arbitration
from terminal_marl.training import evaluate_random_soc_gate_repair_v1 as base
from terminal_marl.training import train_random_soc_gate_repair_v1 as repair


STAGE = "RANDOM_SOC_CANONICAL_REFERENCE_DEVELOPMENT_V1"
SCENARIO = "dense24"
DEVELOPMENT_PROTOCOL = Path(
    "docs/evidence/v4_2/random_soc_canonical_reference_v1_20260925.md"
)
CANONICAL_INITIAL_CHECKPOINT_SHA256 = (
    "4b8be66c100aaaeb5b5699dcce68a96dbe1237964e4c308e7b23b95c62fb8c8c"
)
CANDIDATE_CHECKPOINT_SHA256 = (
    "e04e5b4a8f8365b52a41f66c96ad1f66a09a554ea64118fcab8040d64175d353"
)
SMOKE_SEED = 848000
FULL_SEEDS = (848101, 848102)
SMOKE_MAX_STEPS = 2

# These are integrity and safety checks from the existing P3 helper.  The
# performance field ``no_failed_outcomes`` is deliberately excluded so that an
# ordinary outcome/performance failure does not stop the registered pair.
_STOP_ON_FALSE_GATES = frozenset({
    "safety",
    "replay",
    "requested_likelihood",
    "initial_ratio",
    "student_only",
    "stochastic",
    "actor_decisions",
    "runtime_shortlist_contract",
    "shield_burden",
    "geometry",
    "urgency_reward_config_exact",
    "no_updates",
})


# Keep the runner coupled to the existing evaluator seams.  In particular,
# the full helper owns the P3 replay/lifecycle checks and the smoke helper owns
# the finite-observation and two-step checks.
_evaluate_arm = base._evaluate_arm
_smoke_arm = arbitration._smoke_arm
_new_arm_record = base._new_arm_record
_not_run_arm = base._not_run_arm
_loader_record = base._loader_record
_config_snapshot = base._config_snapshot
_aggregate = base._aggregate
_dev_selection_gates = base._dev_selection_gates
_build_config = base._build_config
_load_yaml_protocol = base._load_yaml_protocol
_jsonable = base._jsonable
_write_json = base._write_json


def _validate_seed(seed: int, *, smoke: bool) -> None:
    allowed = (SMOKE_SEED,) if smoke else FULL_SEEDS
    if int(seed) not in allowed:
        raise ValueError(
            f"seed must be one of {allowed} for {'smoke' if smoke else 'full'} evaluation"
        )


def _build_common_config(protocol: Mapping[str, Any]) -> Any:
    config = _build_config(SCENARIO, protocol)
    if str(getattr(config, "release_arbitration_mode", "")) != "downstream_priority":
        raise ValueError("canonical reference requires downstream_priority arbitration")
    if bool(getattr(config, "reconcile_late_merge_slots_enabled", False)):
        raise ValueError("canonical reference requires late-merge reconciliation disabled")
    if bool(getattr(config, "teacher_enabled", False)):
        raise ValueError("canonical reference requires Teacher disabled")
    if str(getattr(config, "scheduler_mode", "execution")) != "execution":
        raise ValueError("canonical reference requires execution scheduler mode")
    if int(getattr(config, "total_requested_uavs", -1)) != 24:
        raise ValueError("canonical reference requires dense24 request count")
    if float(getattr(config, "max_simulation_seconds", -1.0)) != 1800.0:
        raise ValueError("canonical reference requires 1800 second horizon")
    return config


def _load_canonical_checkpoint(checkpoint: Path, config: Any):
    loader = getattr(diagnostic, "load_canonical_initial_checkpoint", None)
    if loader is None:
        raise AttributeError("canonical checkpoint loader is unavailable")
    return loader(checkpoint, config)


def _load_candidate_checkpoint(checkpoint: Path, config: Any):
    return repair.load_repaired_checkpoint(
        checkpoint,
        config,
        expected_sha256=CANDIDATE_CHECKPOINT_SHA256,
    )


def _annotate_arm(
    record: dict[str, Any],
    *,
    label: str,
    checkpoint: Path,
    checkpoint_sha256: str,
    loader_identity: str,
) -> dict[str, Any]:
    record.update({
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": str(checkpoint_sha256),
        "loader_identity": loader_identity,
        "canonical_initial": label == "canonical_initial",
        "development_only": True,
        "P3_QUALIFIED": False,
        "P4_ALLOWED": False,
        "promotable": False,
        "checkpoint_promotable": False,
    })
    return record


def _integrity_stop_reason(record: Mapping[str, Any]) -> str | None:
    """Return a stop reason for explicit safety/integrity failures only."""

    if bool(record.get("safety_invariant_error")):
        return "safety invariant error"
    summary = record.get("summary")
    if isinstance(summary, Mapping):
        safety = summary.get("safety")
        if isinstance(safety, Mapping) and any(bool(value) for value in safety.values()):
            return "nonzero safety counter"
    gates = record.get("gates")
    if isinstance(gates, Mapping):
        failed = sorted(
            key for key in _STOP_ON_FALSE_GATES
            if key in gates and not bool(gates[key])
        )
        if failed:
            return "integrity gates failed: " + ",".join(failed)
    if "weights_unchanged" in record and record.get("weights_unchanged") is not True:
        return "policy weights changed"
    if "teacher_enabled" in record and bool(record.get("teacher_enabled")):
        return "Teacher enabled"
    if "label_rows" in record and int(record.get("label_rows", 0)) != 0:
        return "label rows were produced"
    if "PPO_ALLOWED" in record and bool(record.get("PPO_ALLOWED")):
        return "PPO_ALLOWED was true"
    lifecycle = record.get("lifecycle")
    if isinstance(lifecycle, Mapping) and lifecycle.get("replay_exact") is not True:
        return "lifecycle replay was not exact"
    return None


def _not_run(
    *, label: str, checkpoint: Path, config: Any, seed: int, reason: str,
    checkpoint_sha256: str, loader_identity: str,
) -> dict[str, Any]:
    return _annotate_arm(
        _not_run_arm(
            label=label,
            checkpoint=checkpoint,
            config=config,
            scenario=SCENARIO,
            seed=int(seed),
            reason=reason,
        ),
        label=label,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        loader_identity=loader_identity,
    )


def _run_arm(
    *,
    label: str,
    checkpoint: Path,
    config: Any,
    protocol: Mapping[str, Any],
    seed: int,
    output: Path,
    smoke: bool,
    checkpoint_sha256: str,
    loader_identity: str,
    checkpoint_loader: Any,
) -> tuple[dict[str, Any], bool, str | None]:
    if smoke:
        record, stop, reason = _smoke_arm(
            label=label,
            checkpoint=checkpoint,
            candidate_sha256=checkpoint_sha256,
            config=config,
            seed=int(seed),
            output=output,
            checkpoint_loader=checkpoint_loader,
        )
    else:
        record, stop, reason = _evaluate_arm(
            label=label,
            checkpoint=checkpoint,
            candidate_sha256=checkpoint_sha256,
            config=config,
            protocol=protocol,
            scenario=SCENARIO,
            seed=int(seed),
            output=output,
            checkpoint_loader=checkpoint_loader,
        )
    if not smoke and not stop:
        integrity_reason = _integrity_stop_reason(record)
        if integrity_reason is not None:
            record["integrity_failure"] = True
            record["status"] = "failed"
            record.setdefault("error", {
                "type": "CanonicalReferenceIntegrityError",
                "message": integrity_reason,
            })
            stop = True
            reason = integrity_reason
    record = _annotate_arm(
        record,
        label=label,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        loader_identity=loader_identity,
    )
    return record, bool(stop), reason


def run(
    *,
    canonical_checkpoint: str | Path,
    candidate_checkpoint: str | Path,
    output: str | Path,
    seed: int,
    smoke: bool = False,
) -> dict[str, Any]:
    """Run one fresh canonical-initial/candidate comparison pair."""

    _validate_seed(int(seed), smoke=bool(smoke))
    output_path = Path(output)
    if output_path.exists():
        raise FileExistsError(f"pair output must be fresh: {output_path}")

    protocol = _load_yaml_protocol()
    config = _build_common_config(protocol)
    config_snapshot = _config_snapshot(config)
    canonical_path = Path(canonical_checkpoint)
    candidate_path = Path(candidate_checkpoint)
    output_path.mkdir(parents=True, exist_ok=False)

    specs = (
        (
            "canonical_initial",
            canonical_path,
            CANONICAL_INITIAL_CHECKPOINT_SHA256,
            "canonical_initial",
            _load_canonical_checkpoint,
        ),
        (
            "repair_candidate",
            candidate_path,
            CANDIDATE_CHECKPOINT_SHA256,
            "repaired_candidate",
            _load_candidate_checkpoint,
        ),
    )
    arms: list[dict[str, Any]] = []
    stopped = False
    stop_reason: str | None = None
    for label, checkpoint, checkpoint_sha256, loader_identity, loader in specs:
        if stopped:
            arms.append(
                _not_run(
                    label=label,
                    checkpoint=checkpoint,
                    config=config,
                    seed=int(seed),
                    reason=stop_reason or "previous arm stopped subsequent arm",
                    checkpoint_sha256=checkpoint_sha256,
                    loader_identity=loader_identity,
                )
            )
            continue
        record, stopped, stop_reason = _run_arm(
            label=label,
            checkpoint=checkpoint,
            config=config,
            protocol=protocol,
            seed=int(seed),
            output=output_path / label,
            smoke=bool(smoke),
            checkpoint_sha256=checkpoint_sha256,
            loader_identity=loader_identity,
            checkpoint_loader=loader,
        )
        arms.append(record)

    errors = [
        record.get("error") or {
            "arm": record.get("label"),
            "reason": record.get("reason", "arm failed"),
        }
        for record in arms
        if record.get("status") in {"failed", "not_run"}
    ]
    safety_failure = any(bool(record.get("safety_invariant_error")) for record in arms)
    if smoke:
        aggregate = {
            "canonical_initial": {"smoke_partial": True},
            "repair_candidate": {"smoke_partial": True},
        }
        selection = {
            "development_only": True,
            "selection_report_only": True,
            "smoke_partial": True,
            "hard_gate_pass": False,
            "gates": {"smoke_partial": True, "hard_gate_pass": False},
            "PASS": False,
        }
        status = "SMOKE_FAILED" if errors or safety_failure else "SMOKE_COMPLETED"
    else:
        canonical_metrics, canonical_error = _aggregate(arms[0])
        candidate_metrics, candidate_error = _aggregate(arms[1])
        gates = _dev_selection_gates(
            canonical_metrics if canonical_metrics is not None else {},
            candidate_metrics if candidate_metrics is not None else {},
        )
        aggregate = {
            "canonical_initial": canonical_metrics,
            "repair_candidate": candidate_metrics,
            "canonical_initial_error": canonical_error,
            "repair_candidate_error": candidate_error,
        }
        selection = {
            "development_only": True,
            "selection_report_only": True,
            "canonical_initial_metrics": canonical_metrics,
            "repair_candidate_metrics": candidate_metrics,
            "canonical_initial_aggregate_error": canonical_error,
            "repair_candidate_aggregate_error": candidate_error,
            "gates": gates,
            "PASS": bool(all(gates.values())),
        }
        status = "FAILED" if errors or safety_failure else "COMPLETED"

    result = {
        "stage": STAGE,
        "status": status,
        "smoke": bool(smoke),
        "development_only": True,
        "P3_QUALIFIED": False,
        "PPO_ALLOWED": False,
        "P4_ALLOWED": False,
        "promotable": False,
        "checkpoint_promotable": False,
        "scenario": SCENARIO,
        "seed": int(seed),
        "canonical_checkpoint": str(canonical_path),
        "candidate_checkpoint": str(candidate_path),
        "canonical_checkpoint_sha256": CANONICAL_INITIAL_CHECKPOINT_SHA256,
        "candidate_checkpoint_sha256": CANDIDATE_CHECKPOINT_SHA256,
        "checkpoints": {
            "canonical_initial": {
                "checkpoint": str(canonical_path),
                "sha256": CANONICAL_INITIAL_CHECKPOINT_SHA256,
                "loader": "diagnostic.load_canonical_initial_checkpoint",
            },
            "repair_candidate": {
                "checkpoint": str(candidate_path),
                "sha256": CANDIDATE_CHECKPOINT_SHA256,
                "loader": "train_random_soc_gate_repair_v1.load_repaired_checkpoint",
            },
        },
        "development_protocol": str(DEVELOPMENT_PROTOCOL),
        "gate_reference_yaml": str(base.PROTOCOL),
        "horizon_steps": int(round(float(config.max_simulation_seconds) / float(config.dt))),
        "smoke_max_steps": SMOKE_MAX_STEPS if smoke else None,
        "actual_config_snapshot": config_snapshot,
        "actual_config_snapshots": {
            "canonical_initial": config_snapshot,
            "repair_candidate": config_snapshot,
        },
        "config_diff": {},
        "unique_config_diff": {},
        "aggregate": aggregate,
        "selection": selection,
        "selection_gates": selection["gates"],
        "records": arms,
        "arms": arms,
        "failure": {
            "safety_invariant_error": safety_failure,
            "arm_errors": errors,
        },
    }
    _write_json(output_path / "pair_summary.json", result)
    return result


evaluate_pair = run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-checkpoint", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    result = run(
        canonical_checkpoint=args.canonical_checkpoint,
        candidate_checkpoint=args.candidate_checkpoint,
        output=args.output,
        seed=int(args.seed),
        smoke=bool(args.smoke),
    )
    print(json.dumps(_jsonable(result), sort_keys=True))
    return 0 if result.get("status") in {"COMPLETED", "SMOKE_COMPLETED"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
