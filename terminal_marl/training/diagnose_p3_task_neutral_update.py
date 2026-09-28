"""Replay the existing task-neutral pilot episode-03 tape for update analysis.

The diagnostic uses the episode-02 checkpoint for the policy/value snapshot that
produced the frozen action tape, then evaluates the final pilot checkpoint on
the exact same captured worlds.  It does not sample actions or call an
optimizer; the simulator replay is performed exactly once.
"""
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from terminal_marl.algo import student_ppo_v4_2 as ppo
from terminal_marl.training import diagnose_p3_ppo_credit_signal as credit
from terminal_marl.training import evaluate_p3_runtime_rank_repair as rank_eval
from terminal_marl.training import evaluate_p3_task_neutral as neutral_eval


STAGE = "P3_TASK_NEUTRAL_UPDATE_DIAGNOSTIC_V1"
PILOT_ROOT = Path(
    "/home/user/training_outputs/terminal/"
    "p3_task_neutral_ppo_pilot_20260928/full"
)
EPISODE_RECORD = PILOT_ROOT / "training/episode_03/record.json"
BEFORE_CHECKPOINT = PILOT_ROOT / "training/episode_02/pilot_after_update.pt"
AFTER_CHECKPOINT = PILOT_ROOT / "pilot.pt"
SCENARIO = "dense24"
SEED = 873103
EXPECTED_REQUESTS = 24
EXPECTED_STEPS = 729
GAMMA = 0.99
MC_GAMMA_1 = 1.0
GAE_LAMBDA = 0.95
OBSERVATION_DIM = 88


class TaskNeutralUpdateError(ValueError):
    """Raised when the fixed replay or output contract is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TaskNeutralUpdateError(message)


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


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"JSON input missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskNeutralUpdateError(f"invalid JSON input: {path}") from exc
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _synced_path(path: Path) -> Path:
    """Resolve a server output through the documented local mirror."""

    return neutral_eval._synced_path(Path(path))


def _sha256(path: Path) -> str:
    return rank_eval._sha256(path)


def _model_digest(policy: Any) -> str:
    return rank_eval._model_digest(policy)


def _preflight() -> dict[str, Any]:
    record_path = _synced_path(EPISODE_RECORD)
    record = _read_json(record_path)
    _require(record.get("scenario") == SCENARIO, "episode-03 record scenario is not dense24")
    _require(int(record.get("seed", -1)) == SEED, "episode-03 record seed mismatch")
    _require(int(record.get("requests", -1)) == EXPECTED_REQUESTS, "episode-03 request count mismatch")
    summary = record.get("summary") if isinstance(record.get("summary"), Mapping) else {}
    expected_steps = int(summary.get("steps", EXPECTED_STEPS))
    _require(expected_steps == EXPECTED_STEPS, f"episode-03 step count mismatch: {expected_steps}")

    tape_value = record.get("action_tape")
    _require(isinstance(tape_value, str) and tape_value, "episode-03 action tape path missing")
    tape_path = _synced_path(Path(tape_value))
    before_path = _synced_path(BEFORE_CHECKPOINT)
    after_path = _synced_path(AFTER_CHECKPOINT)
    for label, path in (("action tape", tape_path), ("before checkpoint", before_path), ("after checkpoint", after_path)):
        _require(path.is_file(), f"{label} missing: {path}")
    _payload, tape_rows = credit._load_tape(tape_path)
    _require(len(tape_rows) == expected_steps, f"tape rows mismatch: {len(tape_rows)} != {expected_steps}")
    _require(tape_rows[0].get("step") == 0 and tape_rows[-1].get("step") == expected_steps - 1,
             "action tape step bounds mismatch")
    _configs, neutral_configs, _runtime = neutral_eval._build_neutral_inputs()
    config = neutral_configs[SCENARIO]
    _require(getattr(config, "task_priority_enabled", None) is False, "dense24 config is not neutral")
    _require(float(getattr(config, "low_level_priority_time_weight", 1.0)) == 0.0,
             "dense24 priority weight is not neutral")
    return {
        "record": record,
        "record_path": record_path,
        "tape_path": tape_path,
        "tape_rows": tape_rows,
        "before_path": before_path,
        "after_path": after_path,
        "config": config,
        "expected_steps": expected_steps,
        "before_sha256": _sha256(before_path),
        "after_sha256": _sha256(after_path),
    }


def _after_probability_map(policy: Any, capture: Mapping[str, Any]) -> dict[str, float]:
    world = np.asarray(capture["world"], dtype=np.float32)
    probabilities = credit._policy_release_probabilities(policy, world)
    uids = [str(uid) for uid in capture.get("uids", ())]
    _require(probabilities.shape == (len(uids),), "after probabilities are not aligned")
    return {uid: float(probability) for uid, probability in zip(uids, probabilities, strict=True)}


def _map_value(mapping: Mapping[Any, Any] | None, uid: Any, default: Any = None) -> Any:
    if not isinstance(mapping, Mapping):
        return default
    if uid in mapping:
        return mapping[uid]
    text = str(uid)
    if text in mapping:
        return mapping[text]
    return default


def _compact_row(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {key: row.get(key) for key in (
        "step", "uid", "arrival_id", "phase", "p_before", "p_after", "p_delta",
    )}


def _analyze_captures(captures: Sequence[Mapping[str, Any]], after_policy: Any) -> tuple[dict[str, Any], list[dict[str, Any]], list[ppo.StudentStepV42]]:
    ordered = [dict(row) for row in sorted(captures, key=lambda item: int(item["step"]))]
    _require([int(row["step"]) for row in ordered] == list(range(len(ordered))),
             "captured steps are not contiguous")
    steps = [credit._step_from_capture(row) for row in ordered]
    actor_mask = np.asarray([bool(np.asarray(step.actor_mask, dtype=bool).any()) for step in steps], dtype=bool)
    raw_advantage, raw_return = ppo.team_gae(steps, gamma=GAMMA, gae_lambda=GAE_LAMBDA)
    normalized = credit.actor_step_normalize(raw_advantage, actor_mask)
    rewards = np.asarray([float(step.reward) for step in steps], dtype=np.float64)
    _require(np.isfinite(rewards).all(), "captured rewards are non-finite")
    mc99 = credit.mc_returns(rewards, GAMMA)
    mc1 = credit.mc_returns(rewards, MC_GAMMA_1)

    actor_rows: list[dict[str, Any]] = []
    for index, (capture, step) in enumerate(zip(ordered, steps, strict=True)):
        after_probabilities = _after_probability_map(after_policy, capture)
        before_probabilities = capture.get("release_probability_by_uid") or {}
        uids = [str(uid) for uid in capture.get("uids", ())]
        for agent_index, uid in enumerate(uids):
            if not bool(step.actor_mask[agent_index]):
                continue
            p_before = _map_value(before_probabilities, uid)
            p_after = after_probabilities.get(uid)
            _require(p_before is not None and p_after is not None, f"probability missing for step={index}, uid={uid}")
            _require(math.isfinite(float(p_before)) and math.isfinite(float(p_after)),
                     f"non-finite probability for step={index}, uid={uid}")
            actor_rows.append({
                "step": int(capture["step"]),
                "uid": int(uid),
                "arrival_id": _map_value(capture.get("arrival_by_uid"), uid),
                "phase": _map_value(capture.get("phase_by_uid"), uid),
                "requested_gate": int(step.requested_gate[agent_index]),
                "proposal": int(step.requested_proposal[agent_index]),
                "p_before": float(p_before),
                "p_after": float(p_after),
                "p_delta": float(p_after - float(p_before)),
                "value": float(step.value),
                "nextvalue": float(step.next_value),
                "reward": float(step.reward),
                "rawadv": float(raw_advantage[index]),
                "normadv": float(normalized["values"][index]),
                "mc99": float(mc99[index]),
                "mc1": float(mc1[index]),
            })

    changed = [row for row in actor_rows if row["p_delta"] != 0.0]
    earliest = min(changed, key=lambda row: (int(row["step"]), int(row["uid"]))) if changed else None
    largest = max(changed, key=lambda row: (abs(float(row["p_delta"])), -int(row["step"]), -int(row["uid"]))) if changed else None
    arrival_summaries = {
        str(arrival_id): {
            "count": sum(row.get("arrival_id") == arrival_id for row in actor_rows),
            "rows": [row for row in actor_rows if row.get("arrival_id") == arrival_id],
        }
        for arrival_id in (20, 21)
    }
    analysis = {
        "steps": len(steps),
        "actor_step_count": int(actor_mask.sum()),
        "actor_row_count": len(actor_rows),
        "value_policy": "before_episode02_checkpoint",
        "gae": {
            "gamma": GAMMA,
            "lambda": GAE_LAMBDA,
            "raw_return_first": float(raw_return[0]) if len(raw_return) else None,
            "raw_return_last": float(raw_return[-1]) if len(raw_return) else None,
            "actor_normalization": normalized.get("available", False),
            "actor_normalization_count": normalized.get("actor_count", 0),
            "actor_normalization_mean": normalized.get("mean"),
            "actor_normalization_std": normalized.get("std"),
        },
        "mc_returns": {
            "gamma_0_99_first": float(mc99[0]) if len(mc99) else None,
            "gamma_0_99_last": float(mc99[-1]) if len(mc99) else None,
            "gamma_1_0_first": float(mc1[0]) if len(mc1) else None,
            "gamma_1_0_last": float(mc1[-1]) if len(mc1) else None,
        },
        "arrival_summaries": arrival_summaries,
        "earliest_probability_difference": _compact_row(earliest),
        "largest_probability_difference": _compact_row(largest),
    }
    return analysis, actor_rows, steps


def _save_replay_steps(path: Path, captures: Sequence[Mapping[str, Any]], *, tape_path: Path) -> None:
    steps: list[dict[str, Any]] = []
    for capture in captures:
        steps.append({
            "step": int(capture["step"]),
            "uids": [str(uid) for uid in capture.get("uids", ())],
            "world": torch.as_tensor(capture["world"], dtype=torch.float32),
            "requested_gate": torch.as_tensor(capture["requested_gate"], dtype=torch.int64),
            "requested_proposal": torch.as_tensor(capture["requested_proposal"], dtype=torch.int64),
            "gate_logprob": torch.as_tensor(capture["gate_logprob"], dtype=torch.float32),
            "proposal_logprob": torch.as_tensor(capture["proposal_logprob"], dtype=torch.float32),
            "actor_mask": torch.as_tensor(capture["actor_mask"], dtype=torch.bool),
            "reward": float(capture["reward"]),
            "value": float(capture["value"]),
            "next_value": float(capture["next_value"]),
            "terminated": bool(capture["terminated"]),
            "episode_end": bool(capture["episode_end"]),
        })
    torch.save({
        "format": "p3_task_neutral_replay_steps_v1",
        "source_tape": str(tape_path),
        "capture_policy": "before_episode02_checkpoint",
        "steps": steps,
    }, path)


def _progress(path: Path, *, status: str, processed: int, expected: int, error: str | None = None) -> None:
    value: dict[str, Any] = {
        "stage": STAGE,
        "status": status,
        "processed_steps": int(processed),
        "expected_steps": int(expected),
        "replay_calls": 0 if processed == 0 else 1,
        "PPO_ALLOWED": False,
        "optimizer_used": False,
    }
    if error:
        value["error"] = error
    _write_json(path, value)


def run(output_dir: str | Path, *, check_only: bool = False) -> dict[str, Any]:
    output = Path(output_dir)
    _require(not output.exists(), f"output directory must be fresh: {output}")
    output.mkdir(parents=True, exist_ok=False)
    _progress(output / "progress.json", status="STARTING", processed=0, expected=EXPECTED_STEPS)
    processed = 0
    try:
        inputs = _preflight()
        preflight = {
            "record": str(inputs["record_path"]),
            "action_tape": str(inputs["tape_path"]),
            "before_checkpoint": str(inputs["before_path"]),
            "after_checkpoint": str(inputs["after_path"]),
            "steps": inputs["expected_steps"],
            "requests": EXPECTED_REQUESTS,
            "before_checkpoint_sha256": inputs["before_sha256"],
            "after_checkpoint_sha256": inputs["after_sha256"],
        }
        if check_only:
            report = {
                "stage": STAGE,
                "status": "CHECK_ONLY_OK",
                "check_only": True,
                "preflight": preflight,
                "scope": {"replay_calls": 0, "optimizer_used": False, "env_reset": False},
                "artifacts": {"progress": str(output / "progress.json"), "report": str(output / "report.json")},
            }
            _write_json(output / "report.json", report)
            _progress(output / "progress.json", status="CHECK_ONLY_OK", processed=0, expected=inputs["expected_steps"])
            return report

        config = inputs["config"]
        before_policy, _before_payload = rank_eval._load_policy(inputs["before_path"], config)
        after_policy, _after_payload = rank_eval._load_policy(inputs["after_path"], config)
        before_digest = _model_digest(before_policy)
        after_digest = _model_digest(after_policy)

        def progress_callback(count: int) -> None:
            nonlocal processed
            processed = int(count)
            _progress(output / "progress.json", status="RUNNING", processed=processed, expected=inputs["expected_steps"])

        replay = credit._capture_replay(
            before_policy,
            config,
            inputs["tape_path"],
            inputs["tape_rows"],
            progress_callback=progress_callback,
        )
        captures = replay["captures"]
        processed = len(captures)
        _require(replay["replay"].get("exact") is True, "action tape replay was not exact")
        _require(int(replay["replay"].get("steps_replayed", -1)) == inputs["expected_steps"],
                 "replay step count mismatch")
        lifecycle = replay.get("lifecycle") or {}
        _require(lifecycle.get("terminal") is True, "replay did not reach terminal state")
        _require(lifecycle.get("all_requests_observed") is True and
                 len(lifecycle.get("request_rows", [])) == EXPECTED_REQUESTS,
                 "replay did not observe exactly 24 requests")
        analysis, actor_rows, _steps = _analyze_captures(captures, after_policy)
        _save_replay_steps(output / "replay_steps.pt", captures, tape_path=inputs["tape_path"])
        _write_json(output / "actor_rows.json", {"rows": actor_rows})
        before_after_digest = _model_digest(before_policy)
        after_after_digest = _model_digest(after_policy)
        model_checks = {
            "before_unchanged": before_digest == before_after_digest,
            "after_unchanged": after_digest == after_after_digest,
            "before_digest_before": before_digest,
            "before_digest_after": before_after_digest,
            "after_digest_before": after_digest,
            "after_digest_after": after_after_digest,
        }
        _require(model_checks["before_unchanged"] and model_checks["after_unchanged"],
                 "policy model changed during diagnostic")
        report = {
            "stage": STAGE,
            "status": "COMPLETED",
            "check_only": False,
            "preflight": preflight,
            "checks": {
                "replay_exact": True,
                "steps_expected": inputs["expected_steps"],
                "steps_replayed": int(replay["replay"].get("steps_replayed", -1)),
                "captures": len(captures),
                "requests": EXPECTED_REQUESTS,
                "terminal": bool(lifecycle.get("terminal")),
                "request_rows": len(lifecycle.get("request_rows", [])),
                "model": model_checks,
            },
            "analysis": analysis,
            "artifacts": {
                "progress": str(output / "progress.json"),
                "report": str(output / "report.json"),
                "actor_rows": str(output / "actor_rows.json"),
                "replay_steps": str(output / "replay_steps.pt"),
            },
            "scope": {
                "one_existing_action_tape": True,
                "capture_replay_calls": 1,
                "policy_sampling": False,
                "sampling_note": "The fixed tape replay does not resample actions; conditional proposal draws in the original stochastic collection can branch, so same seed alone does not identify a particular actor.",
                "optimizer_used": False,
                "teacher_used": False,
                "PPO_ALLOWED": False,
            },
        }
        _write_json(output / "report.json", report)
        _progress(output / "progress.json", status="COMPLETED", processed=processed, expected=inputs["expected_steps"])
        return report
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        report = {
            "stage": STAGE,
            "status": "FAILED",
            "check_only": bool(check_only),
            "processed_steps": processed,
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "scope": {"optimizer_used": False, "PPO_ALLOWED": False},
            "artifacts": {"progress": str(output / "progress.json"), "report": str(output / "report.json")},
        }
        _write_json(output / "report.json", report)
        _progress(output / "progress.json", status="FAILED", processed=processed, expected=EXPECTED_STEPS, error=str(exc))
        return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    result = run(args.output_dir, check_only=bool(args.check_only))
    print(json.dumps(_jsonable({"stage": STAGE, "status": result.get("status"), "output_dir": str(args.output_dir)}), sort_keys=True), flush=True)
    return 0 if result.get("status") in {"COMPLETED", "CHECK_ONLY_OK"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
