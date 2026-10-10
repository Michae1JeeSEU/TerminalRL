"""Run a bounded control-versus-critic-refresh task-neutral comparison."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from terminal_marl.training import evaluate_p3_task_neutral as neutral_eval
from terminal_marl.training import probe_p3_task_neutral_critic_context as probe
from terminal_marl.training import run_p3_task_neutral_critic_warmup as critic_warmup
from terminal_marl.training import run_p3_task_neutral_learning_curve as curve
from terminal_marl.training import run_p3_task_neutral_pilot as pilot


STAGE = "P3_TASK_NEUTRAL_CRITIC_REFRESH_COMPARISON_V1"
ROOT = Path(
    "/home/user/training_outputs/terminal/"
    "p3_task_neutral_critic_refresh_comparison_20260929"
)
PROBE_ROOT = Path(
    "/home/user/training_outputs/terminal/"
    "p3_task_neutral_critic_context_probe_20260929/full"
)
PROBE_REPORT = PROBE_ROOT / "report.json"
PROBE_LEGACY_CRITIC = PROBE_ROOT / "legacy_critic.pt"
PROBE_LEGACY_CRITIC_SHA256 = "9c8813ebb49fc5a54f4867c883266373f54b58997a4c4c7d2d6d4011fa8da873"
SOURCE_CHECKPOINT_SHA256 = pilot.EXPECTED_CANDIDATE_SHA256
A82_WARMUP = probe.WARMUP_CHECKPOINT
A82_WARMUP_SHA256 = probe.WARMUP_CHECKPOINT_SHA256
GAMMA = 1.0
GAE_LAMBDA = 1.0
REWARD_SCALE = 0.01
TRAINING_PERMUTATION_SEEDS = (881001, 881002)
TRAINING_ENV_SEEDS = tuple(range(881101, 881149))
TRAINING_POLICY_SEEDS = tuple(range(881201, 881249))
EVALUATION_BATCHES = (2,)
EPISODES_PER_BATCH = 24
MINIBATCH_SIZE = 128
FIT_EPISODES = 18
VALIDATION_EPISODES = 6


class CriticRefreshComparisonError(ValueError):
    """Raised when the fixed comparison binding or output contract is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CriticRefreshComparisonError(message)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"JSON input missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _resolve(path: str | Path) -> Path:
    return neutral_eval._synced_path(Path(path))


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_digest(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(str(name).encode("utf-8"))
        value = state[name]
        _require(isinstance(value, torch.Tensor), f"state {name} is not a tensor")
        _require(bool(torch.isfinite(value).all().item()), f"state {name} is non-finite")
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def comparison_profile(name: str, warmup_sha256: str) -> curve.LearningCurveProfile:
    """Return the fixed two-batch profile used by either comparison arm."""
    _require(name in {"control", "refresh"}, "comparison profile name is invalid")
    _require(len(TRAINING_ENV_SEEDS) == 48 and len(TRAINING_POLICY_SEEDS) == 48,
             "comparison seed matrix changed")
    return curve.LearningCurveProfile(
        name=f"critic_refresh_{name}",
        batch_count=2,
        episodes_per_batch=EPISODES_PER_BATCH,
        permutation_seeds=TRAINING_PERMUTATION_SEEDS,
        training_env_seeds=TRAINING_ENV_SEEDS,
        training_policy_seeds=TRAINING_POLICY_SEEDS,
        evaluation_batches=EVALUATION_BATCHES,
        minibatch_size=MINIBATCH_SIZE,
        required_episodic_return=True,
        required_reward_scale=REWARD_SCALE,
        required_warmup_sha256=str(warmup_sha256),
    )


def _probe_bundle(
    report_path: str | Path = PROBE_REPORT,
    critic_path: str | Path = PROBE_LEGACY_CRITIC,
) -> dict[str, Any]:
    report_file = _resolve(report_path)
    critic_file = _resolve(critic_path)
    report = _read_json(report_file)
    _require(report.get("status") == "COMPLETED", "critic context probe is not COMPLETED")
    binding = report.get("binding")
    _require(isinstance(binding, Mapping), "critic context probe binding is missing")
    _require(binding.get("source_checkpoint_sha256") == SOURCE_CHECKPOINT_SHA256,
             "critic context probe source checkpoint SHA mismatch")
    _require(binding.get("warmup_checkpoint_sha256") == A82_WARMUP_SHA256,
             "critic context probe warmup SHA mismatch")
    _require(int(binding.get("fit_episodes", -1)) == FIT_EPISODES,
             "critic context probe fit episode count mismatch")
    _require(int(binding.get("validation_episodes", -1)) == VALIDATION_EPISODES,
             "critic context probe validation episode count mismatch")
    _require(float(binding.get("gamma")) == GAMMA and float(binding.get("reward_scale")) == REWARD_SCALE,
             "critic context probe credit binding mismatch")
    checks = report.get("checks")
    _require(isinstance(checks, Mapping), "critic context probe checks are missing")
    required_checks = (
        "replay_exact", "terminal", "ledger_alignment", "safety_required_zero",
        "source_actor_model_bitwise_unchanged", "validation_excluded_from_optimizer",
        "expanded_initial_prediction_matches_legacy",
    )
    for name in required_checks:
        _require(checks.get(name) is True, f"critic context probe check failed: {name}")
    trajectory_reports = report.get("trajectory_reports")
    _require(isinstance(trajectory_reports, list) and len(trajectory_reports) == 24,
             "critic context probe trajectory reports are incomplete")
    _require(sum(str(row.get("split")) == "fit" for row in trajectory_reports) == FIT_EPISODES,
             "critic context probe fit trajectory split mismatch")
    _require(sum(str(row.get("split")) == "validation" for row in trajectory_reports) == VALIDATION_EPISODES,
             "critic context probe validation trajectory split mismatch")
    _require(critic_file.is_file(), f"critic context artifact missing: {critic_path}")
    actual_sha = _sha256(critic_file)
    _require(actual_sha == PROBE_LEGACY_CRITIC_SHA256,
             f"critic context artifact SHA mismatch: expected {PROBE_LEGACY_CRITIC_SHA256}, got {actual_sha}")
    critic_payload = torch.load(critic_file, map_location="cpu", weights_only=False)
    _require(isinstance(critic_payload, Mapping), "critic context artifact payload is invalid")
    _require(critic_payload.get("diagnostic_only") is True,
             "critic context artifact is not diagnostic-only")
    _require(critic_payload.get("critic_kind") == "legacy_mean_max",
             "critic context artifact kind mismatch")
    critic_state = critic_payload.get("state_dict")
    _require(isinstance(critic_state, Mapping) and critic_state, "critic context critic state is missing")
    critic_digest = _state_digest(critic_state)
    return {
        "report_path": report_file, "critic_path": critic_file,
        "report": report, "critic_payload": dict(critic_payload),
        "critic_state": dict(critic_state), "critic_sha256": actual_sha,
        "critic_input_hash": critic_digest,
    }


def _merge_refresh_warmup(
    base_payload: Mapping[str, Any], probe_bundle: Mapping[str, Any],
    *, config: Any, output_path: Path,
) -> dict[str, Any]:
    """Merge probe legacy critic parameters into the complete a82 payload."""
    base_state = base_payload.get("model_state")
    critic_state = probe_bundle.get("critic_state")
    _require(isinstance(base_state, Mapping) and isinstance(critic_state, Mapping),
             "warmup model states are missing")
    expected_critic = {
        str(name): value for name, value in base_state.items()
        if str(name).startswith(critic_warmup.CRITIC_PREFIXES)
    }
    _require(set(critic_state) == set(expected_critic), "refresh critic state keys mismatch")
    merged_state: dict[str, torch.Tensor] = {}
    for name, value in base_state.items():
        _require(isinstance(value, torch.Tensor), f"warmup state {name} is not a tensor")
        merged_state[str(name)] = value.detach().cpu().clone()
    for name, value in critic_state.items():
        _require(isinstance(value, torch.Tensor), f"refresh critic state {name} is not a tensor")
        _require(tuple(value.shape) == tuple(expected_critic[name].shape),
                 f"refresh critic state shape mismatch: {name}")
        _require(bool(torch.isfinite(value).all().item()), f"refresh critic state is non-finite: {name}")
        merged_state[str(name)] = value.detach().cpu().clone()
    for name in base_state:
        if not str(name).startswith(critic_warmup.CRITIC_PREFIXES):
            _require(torch.equal(base_state[name].detach().cpu(), merged_state[str(name)]),
                     f"refresh changed non-critic state: {name}")
    payload = copy.deepcopy(dict(base_payload))
    payload["model_state"] = merged_state
    payload["gamma"] = GAMMA
    payload["reward_scale"] = REWARD_SCALE
    probe_fit = probe_bundle["report"].get("fit")
    _require(isinstance(probe_fit, Mapping), "critic context probe fit report is missing")
    for key in ("epochs", "minibatch_size", "learning_rate", "permutation_seed", "grad_clip"):
        _require(key in probe_fit, f"critic context probe fit parameter missing: {key}")
    fit_parameters = {
        "gamma": GAMMA,
        "reward_scale": REWARD_SCALE,
        "epochs": int(probe_fit["epochs"]),
        "minibatch_size": int(probe_fit["minibatch_size"]),
        "learning_rate": float(probe_fit["learning_rate"]),
        "permutation_seed": int(probe_fit["permutation_seed"]),
        "grad_clip": float(probe_fit["grad_clip"]),
        "loss": "MSE",
        "target": "full_episode_mc",
        "early_select": False,
        "reward_changed": False,
        "gae_lambda_changed": False,
    }
    payload["fit_parameters"] = fit_parameters
    episode_metrics = probe_bundle["report"].get("episode_metrics")
    _require(isinstance(episode_metrics, Mapping), "critic context probe episode metrics are missing")
    before_cells = episode_metrics.get("before")
    after_cells = episode_metrics.get("after")
    _require(isinstance(before_cells, Mapping) and isinstance(after_cells, Mapping),
             "critic context probe per-cell metrics are missing")
    fit_before_cells = {
        key: dict(value.get("legacy") or {})
        for key, value in before_cells.items()
        if isinstance(value, Mapping) and str(value.get("split")) == "fit"
    }
    fit_after_cells = {
        key: dict(value.get("legacy") or {})
        for key, value in after_cells.items()
        if isinstance(value, Mapping) and str(value.get("split")) == "fit"
    }
    validation_before_cells = {
        key: dict(value.get("legacy") or {})
        for key, value in before_cells.items()
        if isinstance(value, Mapping) and str(value.get("split")) == "validation"
    }
    validation_after_cells = {
        key: dict(value.get("legacy") or {})
        for key, value in after_cells.items()
        if isinstance(value, Mapping) and str(value.get("split")) == "validation"
    }
    before_fit = probe_fit.get("before")
    after_fit = probe_fit.get("after")
    _require(isinstance(before_fit, Mapping) and isinstance(after_fit, Mapping),
             "critic context probe fit metrics are missing")
    _require(isinstance(before_fit.get("legacy"), Mapping) and isinstance(after_fit.get("legacy"), Mapping),
             "critic context probe legacy fit metrics are missing")
    payload["fit_metrics"] = {
        "before": dict(before_fit["legacy"]),
        "after": dict(after_fit["legacy"]),
        "before_per_cell": fit_before_cells,
        "after_per_cell": fit_after_cells,
        "validation_before_per_cell": validation_before_cells,
        "validation_after_per_cell": validation_after_cells,
    }
    reports = probe_bundle["report"].get("trajectory_reports")
    _require(isinstance(reports, list), "critic context probe trajectory reports are missing")
    payload["trajectory_reports"] = [dict(row) for row in reports]
    payload["origin_stage"] = probe_bundle["report"].get("stage")
    payload["origin_probe_report"] = str(probe_bundle["report_path"])
    payload["origin_probe_critic_checkpoint"] = str(probe_bundle["critic_path"])
    payload["origin_fit_episodes"] = FIT_EPISODES
    payload["origin_validation_episodes"] = VALIDATION_EPISODES
    payload["origin_critic_kind"] = "legacy_mean_max"
    payload["origin_critic_diagnostic_only"] = True
    payload["critic_input_hash"] = str(probe_bundle["critic_input_hash"])
    payload["critic_input_sha256"] = str(probe_bundle["critic_sha256"])
    payload["refresh_method"] = "probe_legacy_critic_into_a82_warmup"
    pilot._validate_initial_warmup_payload(
        payload, probe_bundle["policy"], config,
        expected_gamma=GAMMA, expected_reward_scale=REWARD_SCALE,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return payload


def _prepare_inputs(output: Path, config: Any) -> dict[str, Any]:
    resolved, policy, binding = pilot._load_binding(config)
    base_path, base_check = pilot._load_initial_warmup(
        _resolve(A82_WARMUP), policy, config,
        expected_gamma=GAMMA, expected_reward_scale=REWARD_SCALE,
    )
    _require(str(base_check.get("checkpoint_sha256", "")).lower() == A82_WARMUP_SHA256,
             "control warmup SHA mismatch")
    base_payload = torch.load(base_path, map_location="cpu", weights_only=False)
    _require(isinstance(base_payload, Mapping), "control warmup payload is invalid")
    probe_bundle = _probe_bundle()
    probe_bundle = dict(probe_bundle)
    probe_bundle["policy"] = policy
    refresh_path = output / "prepared" / "refresh_warmup.pt"
    refresh_payload = _merge_refresh_warmup(
        base_payload, probe_bundle, config=config, output_path=refresh_path,
    )
    refresh_sha = _sha256(refresh_path)
    _load_path, refresh_check = pilot._load_initial_warmup(
        refresh_path, policy, config,
        expected_gamma=GAMMA, expected_reward_scale=REWARD_SCALE,
    )
    _require(refresh_check.get("model_state_loaded") is True, "refresh warmup model was not loaded")
    return {
        "resolved_candidate": resolved, "candidate_binding": binding,
        "control_warmup": base_path, "control_warmup_check": base_check,
        "control_warmup_payload": dict(base_payload),
        "refresh_warmup": refresh_path, "refresh_warmup_sha256": refresh_sha,
        "refresh_warmup_check": refresh_check, "refresh_payload": refresh_payload,
        "probe": probe_bundle,
    }


def _arm_summary(name: str, output: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    evaluation = report.get("evaluation") if isinstance(report.get("evaluation"), Mapping) else {}
    training = report.get("training") if isinstance(report.get("training"), Mapping) else {}
    return {
        "name": name, "output_dir": str(output), "status": report.get("status"),
        "profile": report.get("profile"),
        "training_rollouts": training.get("rollouts", report.get("training_rollouts", 0)),
        "training_steps": training.get("steps", report.get("training_steps", 0)),
        "ppo_updates": training.get("updates", report.get("ppo_updates", 0)),
        "evaluation_rollouts": evaluation.get("new_rollouts", report.get("evaluation_rollouts", 0)),
        "evaluation_snapshots": len(evaluation.get("snapshots", [])),
        "evaluation_summary": [
            {
                "batch_index": item.get("batch_index"),
                "after_summary": item.get("after_summary"),
                "paired_differences": item.get("paired_differences", []),
                "evaluation_error": item.get("evaluation_error"),
            }
            for item in evaluation.get("snapshots", [])
            if isinstance(item, Mapping)
        ],
        "checks": {
            key: value for key, value in report.get("checks", {}).items()
            if key in {"batches_completed", "updates_completed", "evaluation_snapshots_completed",
                       "new_evaluations_completed", "optimizer_identity_unchanged", "source_q_fixed"}
        },
        "training_batches": int(training.get("batches_processed", 0) or 0),
    }


def run(output_dir: str | Path = ROOT, *, check_only: bool = False) -> dict[str, Any]:
    output = Path(output_dir)
    _require(not output.exists(), f"comparison output must be fresh: {output}")
    output.mkdir(parents=True, exist_ok=False)
    progress_path = output / "progress.json"
    report: dict[str, Any] = {
        "stage": STAGE, "status": "STARTING", "check_only": bool(check_only),
        "full_evaluation": False, "PPO_ALLOWED": False,
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "fixed_profile": {
            "training_env_seeds": list(TRAINING_ENV_SEEDS),
            "training_policy_seeds": list(TRAINING_POLICY_SEEDS),
            "permutation_seeds": list(TRAINING_PERMUTATION_SEEDS),
            "batch_count": 2, "episodes_per_batch": 24,
            "evaluation_batches": list(EVALUATION_BATCHES),
            "evaluation_cells": [dict(cell) for cell in curve.EVALUATION_CELLS],
            "minibatch_size": MINIBATCH_SIZE,
            "gamma": GAMMA, "gae_lambda": GAE_LAMBDA, "reward_scale": REWARD_SCALE,
        },
        "arms": {}, "prepared": {}, "checks": {}, "deviations": [], "unresolved": [],
        "artifacts": {"output_dir": str(output), "progress": str(progress_path)},
    }
    _write_json(progress_path, {"stage": STAGE, "status": "STARTING", "check_only": bool(check_only),
                                "phase": "preparing", "arms_completed": 0})
    try:
        configs, neutral_configs, _runtime = neutral_eval._build_neutral_inputs()
        del configs, _runtime
        config = neutral_configs["dense24"]
        _require(getattr(config, "task_priority_enabled") is False, "task-neutral flag is not false")
        _require(float(getattr(config, "low_level_priority_time_weight")) == 0.0,
                 "task-neutral priority reward weight is not zero")
        horizon = int(round(float(config.max_simulation_seconds) / float(config.dt)))
        _require(horizon == 1800, f"comparison horizon must be 1800, got {horizon}")
        prepared = _prepare_inputs(output, config)
        report["prepared"] = {
            "control_warmup": str(prepared["control_warmup"]),
            "control_warmup_sha256": A82_WARMUP_SHA256,
            "refresh_warmup": str(prepared["refresh_warmup"]),
            "refresh_warmup_sha256": prepared["refresh_warmup_sha256"],
            "refresh_warmup_check": dict(prepared["refresh_warmup_check"]),
            "probe_report": str(prepared["probe"]["report_path"]),
            "probe_legacy_critic": str(prepared["probe"]["critic_path"]),
            "probe_legacy_critic_sha256": prepared["probe"]["critic_sha256"],
            "critic_input_hash": prepared["probe"]["critic_input_hash"],
            "probe_fit_episodes": FIT_EPISODES,
            "probe_validation_episodes": VALIDATION_EPISODES,
        }
        report["artifacts"]["prepared_refresh_warmup"] = str(prepared["refresh_warmup"])
        if check_only:
            _write_json(progress_path, {"stage": STAGE, "status": "RUNNING", "check_only": True,
                                        "phase": "control_check_only", "arms_completed": 0})
        for index, (name, warmup_path, warmup_sha) in enumerate((
            ("control", prepared["control_warmup"], A82_WARMUP_SHA256),
            ("refresh", prepared["refresh_warmup"], prepared["refresh_warmup_sha256"]),
        ), start=1):
            arm_output = output / name
            profile = comparison_profile(name, warmup_sha)
            _write_json(progress_path, {"stage": STAGE, "status": "RUNNING",
                                        "check_only": bool(check_only),
                                        "phase": f"{name}_running", "arms_completed": index - 1})
            arm_report = curve.run(
                output_dir=arm_output, check_only=bool(check_only),
                episodic_return=True, initial_warmup_checkpoint=warmup_path,
                reward_scale=REWARD_SCALE, profile_override=profile,
            )
            if check_only:
                arm_checks = arm_report.get("checks")
                _require(isinstance(arm_checks, Mapping), f"{name} check-only checks are missing")
                _require(arm_checks.get("runtime_rollouts") == 0,
                         f"{name} check-only executed runtime rollouts")
                _require(arm_checks.get("update_calls") == 0,
                         f"{name} check-only executed PPO updates")
                _require(arm_checks.get("evaluation_rollouts") == 0,
                         f"{name} check-only executed evaluation rollouts")
            report["arms"][name] = _arm_summary(name, arm_output, arm_report)
            report["arms"][name]["profile_spec"] = profile.as_dict()
            _write_json(progress_path, {"stage": STAGE,
                                        "status": "RUNNING" if index == 1 else "COMPLETED",
                                        "check_only": bool(check_only),
                                        "phase": f"{name}_complete", "arms_completed": index})
            _require(arm_report.get("status") in {"CHECK_ONLY_OK", "COMPLETED"},
                     f"{name} arm failed: {arm_report.get('unresolved')}")
        report["checks"] = {
            "profiles_fixed": True,
            "evaluation_cells_fixed": [dict(cell) for cell in curve.EVALUATION_CELLS],
            "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
            "control_warmup_sha256": A82_WARMUP_SHA256,
            "refresh_warmup_sha256": prepared["refresh_warmup_sha256"],
            "probe_legacy_critic_sha256": prepared["probe"]["critic_sha256"],
            "refresh_noncritic_state_preserved": True,
            "refresh_payload_validated_by_pilot": True,
            "training_rollouts_per_arm": 48,
            "evaluation_rollouts_per_arm": 4,
            "check_only_runtime_zero": bool(check_only),
        }
        report.update({
            "status": "CHECK_ONLY_OK" if check_only else "COMPLETED",
            "full_evaluation": not check_only,
            "technical_ok": True,
        })
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        report.update({"status": "FAILED", "full_evaluation": False,
                       "unresolved": [str(exc)],
                       "error": {"type": type(exc).__name__, "message": str(exc)}})
        _write_json(progress_path, {"stage": STAGE, "status": "FAILED", "check_only": bool(check_only),
                                    "phase": "failed", "arms_completed": len(report["arms"]),
                                    "error": str(exc)})
    _write_json(output / "report.json", report)
    _write_json(output / "summary.json", {
        "stage": STAGE, "status": report.get("status"), "check_only": bool(check_only),
        "prepared": report.get("prepared"), "arms": report.get("arms"),
        "checks": report.get("checks"), "unresolved": report.get("unresolved"),
    })
    if report.get("status") in {"COMPLETED", "CHECK_ONLY_OK"}:
        _write_json(progress_path, {"stage": STAGE, "status": report["status"],
                                    "check_only": bool(check_only), "phase": "complete", "arms_completed": 2})
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument("--check-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args.output_dir, check_only=bool(args.check_only))
    print(json.dumps(_jsonable({"stage": STAGE, "status": result.get("status"),
                                "output_dir": str(args.output_dir)}), sort_keys=True), flush=True)
    return 0 if result.get("status") in {"COMPLETED", "CHECK_ONLY_OK"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
