"""Bounded replay and mechanism diagnostic for the refresh energy failure.

The three phases are deliberately separate.  ``preflight`` reads the fixed
record, tape, checkpoint metadata, and neutral configuration without creating
an environment.  ``replay`` performs one exact source-tape replay and
registers three deterministic eligible states.  ``diagnose`` replays the tape
again, deep-copies only those states, and runs the existing paired WAIT/all
RELEASE diagnostic.  This module does not train, sample an optimizer, use a
teacher, or consume held-out data.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from terminal_marl.training import diagnose_p3_ppo_credit_signal as credit
from terminal_marl.training import evaluate_p3_runtime_rank_repair as rank_eval
from terminal_marl.training import evaluate_p3_task_neutral as neutral_eval
from terminal_marl.training import paired_gate_diagnostic
from terminal_marl.training import run_p3_followup_diagnostics as runner_utils


STAGE = "P3_FAILURE_MECHANISM_20260930_V1"
PPO_ALLOWED = False
ROOT = Path(
    "/home/user/training_outputs/terminal/"
    "p3_task_neutral_critic_refresh_comparison_20260929/full/refresh/training"
)
RECORD = ROOT / "batch_02/training/episode_24/record.json"
TAPE = ROOT / "batch_02/training/episode_24/rollout/action_tapes/safety_trace_seed_881148_step_983.json"
SOURCE_CHECKPOINT = ROOT / "batch_01/policy.pt"
AFTER_CHECKPOINT = ROOT / "batch_02/policy.pt"
AFTER_SUMMARY = ROOT / "batch_02/summary.json"
SOURCE_SHA256 = "1637a15d73184617a1e386cd29cbcdbe29c0679b1414f5fb413ec7a9fbef3805"
TAPE_SHA256 = "7bf68b8159e9d6038744735618067f0344a5a64218798bea2b93260c955212e3"
RECORD_SHA256 = "34e3bff462dba2e28571804fbcb530961d805740aa1e6b90fc39508bed3d60e1"
AFTER_SHA256 = "370344b6523f250bd6816bbfd6ef51bd065c4aea1352b21d8910adc1275e942c"
SOURCE_MODEL_DIGEST = "861cdc5f86e5e782a92b9a7cff9874836d3c3069397b7bcf8dddfbe0b8283e5c"
TARGET_UID = 57
TARGET_ARRIVAL_ID = 14
TARGET_FAILURE_STEP = 892
EXPECTED_STEPS = 984
EXPECTED_REQUESTS = 24
CONTINUATION_SEEDS = (9303201, 9303202, 9303203, 9303204)
BRANCH_MAX_STEPS = 1800
QUANTILES = (0.25, 0.50, 0.75)


class FailureMechanismError(ValueError):
    """Raised when a fixed failure-mechanism input/output contract is unmet."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FailureMechanismError(message)


def _read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"JSON input missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, Mapping), f"JSON object required: {path}")
    return dict(value)


def _resolve(path: Path) -> Path:
    return Path(neutral_eval._synced_path(path))


def _hash_artifact(path: Path, expected: str | None = None) -> dict[str, Any]:
    item = runner_utils._artifact(path, expected_sha256=expected)
    return dict(item)


def _failure_row(record: Mapping[str, Any]) -> dict[str, Any]:
    lifecycle = record.get("lifecycle", {})
    rows = lifecycle.get("rows", []) if isinstance(lifecycle, Mapping) else []
    _require(isinstance(rows, list), "record lifecycle rows are missing")
    matches = [row for row in rows if int(row.get("arrival_id", -1)) == TARGET_ARRIVAL_ID]
    _require(len(matches) == 1, "target failure arrival row is not unique")
    row = dict(matches[0])
    _require(int(row.get("uid", -1)) == TARGET_UID, "target failure UID mismatch")
    _require(str(row.get("outcome")) == "failed_energy", "target failure outcome mismatch")
    _require(int(float(row.get("terminal_time", -1))) == TARGET_FAILURE_STEP,
             "target failure terminal step mismatch")
    return row


def _strict_config() -> tuple[Any, dict[str, Any]]:
    # Keep this exact source of the neutral dense24 configuration in the
    # protocol; it prevents accidental use of a legacy or warmup config.
    _configs, neutral_configs, _runtime = neutral_eval._build_neutral_inputs()
    _require("dense24" in neutral_configs, "neutral dense24 config is missing")
    config = neutral_configs["dense24"]
    _require(int(getattr(config, "total_requested_uavs", -1)) == EXPECTED_REQUESTS,
             "dense24 config request count mismatch")
    _require(getattr(config, "task_priority_enabled", False) is False,
             "dense24 config is not task-neutral")
    _require(math.isclose(float(getattr(config, "low_level_priority_time_weight", 1.0)), 0.0,
                          rel_tol=0.0, abs_tol=1e-12),
             "dense24 priority weight is not zero")
    return config, runner_utils._config_snapshot(config)


def _load_fixed_inputs() -> dict[str, Any]:
    record_path = _resolve(RECORD)
    tape_path = _resolve(TAPE)
    source_path = _resolve(SOURCE_CHECKPOINT)
    after_path = _resolve(AFTER_CHECKPOINT)
    summary_path = _resolve(AFTER_SUMMARY)
    record = _read_json(record_path)
    _require(int(record.get("seed", -1)) == 881148, "refresh env seed mismatch")
    _require(int(record.get("policy_seed", -1)) == 881248, "refresh policy seed mismatch")
    _require(str(record.get("scenario")) == "dense24", "refresh scenario mismatch")
    _require(int(record.get("requests", -1)) == EXPECTED_REQUESTS, "refresh request count mismatch")
    _require(int(record.get("rollout_summary", {}).get("steps", -1)) == EXPECTED_STEPS,
             "refresh rollout step count mismatch")
    _require(str(record.get("checkpoint_sha256", "")).lower() == SOURCE_SHA256,
             "record producer checkpoint SHA mismatch")
    failure = _failure_row(record)
    tape_payload, tape_rows = credit._load_tape(tape_path)
    _require(len(tape_rows) == EXPECTED_STEPS, "refresh tape step count mismatch")
    _require(str(record.get("action_tape", "")).endswith("safety_trace_seed_881148_step_983.json"),
             "record action tape binding mismatch")
    summary = _read_json(summary_path)
    roundtrip = summary.get("checkpoint_roundtrip")
    _require(isinstance(roundtrip, Mapping), "after summary checkpoint_roundtrip is missing")
    _require(str(roundtrip.get("checkpoint_sha256", "")).lower() == AFTER_SHA256,
             "after summary checkpoint SHA mismatch")
    after_digest = str(roundtrip.get("model_digest", ""))
    _require(after_digest, "after summary model digest is missing")
    return {
        "record_path": record_path,
        "record": record,
        "failure": failure,
        "tape_path": tape_path,
        "tape_payload": tape_payload,
        "tape_rows": tape_rows,
        "source_path": source_path,
        "after_path": after_path,
        "after_summary_path": summary_path,
        "after_digest": after_digest,
    }


def _source_policy(config: Any, source_path: Path) -> tuple[Any, str, dict[str, Any]]:
    _require(source_path == _resolve(SOURCE_CHECKPOINT), "source checkpoint path is not batch_01")
    policy, payload = rank_eval._load_policy(source_path, config)
    digest = str(rank_eval._model_digest(policy))
    _require(digest == SOURCE_MODEL_DIGEST, "source full-checkpoint model digest mismatch")
    return policy, digest, dict(payload) if isinstance(payload, Mapping) else {}


def _public_candidate(capture: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    context = capture.get("state_context", {})
    marker = context.get("failure_target", {}) if isinstance(context, Mapping) else {}
    _require(isinstance(marker, Mapping), "failure observer marker is missing")
    actions, shortlists = runner_utils._row_inputs(row)
    _require(str(TARGET_UID) in actions, "target action is missing from eligible tape row")
    return {
        "episode": 24,
        "step": int(capture["step"]),
        "uid": TARGET_UID,
        "arrival_id": TARGET_ARRIVAL_ID,
        "phase": marker.get("phase"),
        "soc": marker.get("soc"),
        "energy_remaining": marker.get("energy_remaining"),
        "world_index": int(marker.get("world_index", -1)),
        "observation_hash": str(row.get("observation_hash", "")),
        "source_action_tape_index": row.get("action_tape_index"),
        "first_actions": runner_utils._jsonable(actions),
        "runtime_proposal_shortlists": runner_utils._jsonable(shortlists),
        "target_actor_mask": True,
        "selection_rule": "sorted_episode_step_uid_quantile_floor",
    }


def _eligible_points(captures: Sequence[Mapping[str, Any]], tape_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for capture, row in zip(captures, tape_rows, strict=True):
        context = capture.get("state_context", {})
        marker = context.get("failure_target", {}) if isinstance(context, Mapping) else {}
        if not isinstance(marker, Mapping):
            continue
        if not bool(marker.get("active")) or not bool(marker.get("actor_mask")):
            continue
        if int(marker.get("arrival_id", -1)) != TARGET_ARRIVAL_ID:
            continue
        if int(marker.get("uid", -1)) != TARGET_UID:
            continue
        points.append(_public_candidate(capture, row))
    ordered = sorted(points, key=lambda item: (int(item["episode"]), int(item["step"]), int(item["uid"])))
    keys = [(int(item["episode"]), int(item["step"]), int(item["uid"])) for item in ordered]
    _require(len(keys) == len(set(keys)), "eligible failure occurrences are not unique")
    return ordered


def _select_points(points: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = [dict(item) for item in sorted(points, key=lambda item: (int(item["episode"]), int(item["step"]), int(item["uid"]))) ]
    _require(len(ordered) >= 3, f"eligible failure occurrences are insufficient: {len(ordered)}")
    selected_indices = [int(math.floor((len(ordered) - 1) * q)) for q in QUANTILES]
    _require(len(set(selected_indices)) == 3, "registered quantile indices are not unique")
    selected = []
    for order, (quantile, index) in enumerate(zip(QUANTILES, selected_indices, strict=True), start=1):
        item = dict(ordered[index])
        item.update({"selection_order": order, "eligible_index": index, "quantile": quantile})
        selected.append(item)
    return {
        "selection_rule": "sorted_episode_step_uid_quantile_floor",
        "quantiles": list(QUANTILES),
        "eligible_total": len(ordered),
        "selected": selected,
    }


def _validate_lifecycle(lifecycle: Mapping[str, Any]) -> dict[str, Any]:
    _require(bool(lifecycle.get("terminal")), "source replay did not terminate")
    _require(bool(lifecycle.get("all_requests_observed")), "source replay requests incomplete")
    _require(bool(lifecycle.get("ledger_alignment")), "source replay ledger is not aligned")
    request_rows = lifecycle.get("request_rows", [])
    _require(isinstance(request_rows, list) and len(request_rows) == EXPECTED_REQUESTS,
             "source replay must retain all 24 request inventory rows")
    request_ids = {int(row.get("arrival_id", -1)) for row in request_rows}
    _require(request_ids == set(range(EXPECTED_REQUESTS)),
             "source replay request inventory IDs are incomplete")
    rows = lifecycle.get("rows", [])
    _require(isinstance(rows, list) and len(rows) == EXPECTED_REQUESTS,
             "source replay must retain all 24 terminal rows")
    terminal_ids = {int(row.get("arrival_id", -1)) for row in rows}
    _require(terminal_ids == set(range(EXPECTED_REQUESTS)),
             "source replay terminal IDs are incomplete")
    target = [row for row in rows if int(row.get("arrival_id", -1)) == TARGET_ARRIVAL_ID]
    _require(len(target) == 1 and str(target[0].get("outcome")) == "failed_energy",
             "source replay natural failed_energy row was not retained")
    _require(int(float(target[0].get("terminal_time", -1))) == TARGET_FAILURE_STEP,
             "source replay failure terminal step mismatch")
    return {
        "request_count": len(request_rows),
        "request_rows": [dict(row) for row in request_rows],
        "terminal_rows": [dict(row) for row in rows],
        "failed_energy_rows": [dict(row) for row in rows if str(row.get("outcome")) == "failed_energy"],
        "all_rows_retained": True,
    }


def _failure_observer(policy: Any):
    def observe(env: Any) -> dict[str, Any]:
        light = runner_utils._light_state_observer(env)
        context = paired_gate_diagnostic._details_context(policy, env)
        target_index = next((i for i, uid in enumerate(context["uids"]) if int(uid) == TARGET_UID), None)
        actor = light.get("actors", {}).get(str(TARGET_UID), {})
        marker = {
            "uid": TARGET_UID,
            "arrival_id": actor.get("arrival_id"),
            "active": target_index is not None and str(TARGET_UID) in light.get("actors", {}),
            "actor_mask": False if target_index is None else bool(context["actor_mask"][target_index]),
            "phase": actor.get("phase"),
            "soc": actor.get("soc"),
            "energy_remaining": actor.get("energy_remaining"),
            "world_index": None if target_index is None else int(target_index),
        }
        light["failure_target"] = marker
        return light
    return observe


def _progress(output: Path, phase: str, status: str, processed: int = 0, **extra: Any) -> None:
    payload = {
        "stage": STAGE,
        "phase": phase,
        "status": status,
        "processed": int(processed),
        "expected": EXPECTED_STEPS,
        "processed_steps": int(processed),
        "expected_steps": EXPECTED_STEPS,
        **extra,
    }
    runner_utils._atomic_write_json(output / "progress.json", payload)


def _write_failed_phase(output: Path, phase: str, error: BaseException) -> None:
    report = {
        "stage": STAGE,
        "phase": phase,
        "status": "FAILED",
        "PPO_ALLOWED": PPO_ALLOWED,
        "error": {"type": type(error).__name__, "message": str(error)},
    }
    runner_utils._atomic_write_json(output / "report.json", report)
    _progress(output, phase, "FAILED", error=report["error"])


def _build_preflight() -> dict[str, Any]:
    config, config_snapshot = _strict_config()
    inputs = _load_fixed_inputs()
    source_policy, source_digest, source_payload = _source_policy(config, inputs["source_path"])
    del source_policy
    return {
        "stage": STAGE,
        "phase": "preflight",
        "status": "PREFLIGHT_OK",
        "PPO_ALLOWED": PPO_ALLOWED,
        "runtime_versions": runner_utils._runtime_versions(),
        "config_snapshot": config_snapshot,
        "loaded_sourcefiles": runner_utils._loaded_code_manifest(),
        "source": {
            "checkpoint": _hash_artifact(inputs["source_path"], SOURCE_SHA256),
            "model_digest": source_digest,
            "loader": "evaluate_p3_runtime_rank_repair._load_policy(full_checkpoint)",
            "payload_keys": sorted(source_payload),
        },
        "after_alignment": {
            "checkpoint": _hash_artifact(inputs["after_path"], AFTER_SHA256),
            "summary": _hash_artifact(inputs["after_summary_path"]),
            "model_digest_from_summary": inputs["after_digest"],
            "causal_scope": "alignment_only; after batch_02 update is not claimed to cause the earlier batch_02 rollout failure",
        },
        "failure_input": {
            "record": _hash_artifact(inputs["record_path"], RECORD_SHA256),
            "tape": _hash_artifact(inputs["tape_path"], TAPE_SHA256),
            "env_seed": 881148,
            "policy_seed": 881248,
            "steps": EXPECTED_STEPS,
            "requests": EXPECTED_REQUESTS,
            "failure": inputs["failure"],
        },
        "scope": {
            "warmup_loaded": False,
            "after_continuation_used": False,
            "optimizer_calls": 0,
            "teacher_used": False,
            "heldout_used": False,
            "population_estimate": False,
        },
    }


def _load_preflight(output: Path) -> dict[str, Any]:
    value = _read_json(output / "preflight.json")
    _require(value.get("status") == "PREFLIGHT_OK", "preflight artifact is not complete")
    return value


def _run_replay(output: Path, preflight: Mapping[str, Any]) -> dict[str, Any]:
    config, config_snapshot = _strict_config()
    _require(dict(preflight.get("config_snapshot", {})) == config_snapshot, "resolved config differs from preflight")
    inputs = _load_fixed_inputs()
    policy, before_digest, _payload = _source_policy(config, inputs["source_path"])
    _progress(output, "replay", "RUNNING", 0)
    replay = credit._capture_replay(
        policy, config, inputs["tape_path"], inputs["tape_rows"],
        progress_callback=lambda count: _progress(output, "replay", "RUNNING", int(count)),
        state_observer=_failure_observer(policy),
    )
    runner_utils._atomic_write_json(
        output / "replay_raw_lifecycle.json",
        {"stage": STAGE, "phase": "replay", "lifecycle": replay.get("lifecycle", {})},
    )
    _require(bool(replay.get("replay", {}).get("exact")), "refresh replay was not exact")
    _require(int(replay.get("replay", {}).get("steps_replayed", -1)) == EXPECTED_STEPS,
             "refresh replay step count mismatch")
    lifecycle = _validate_lifecycle(replay.get("lifecycle", {}))
    points = _eligible_points(replay["captures"], inputs["tape_rows"])
    selection = _select_points(points)
    after_digest = str(rank_eval._model_digest(policy))
    _require(after_digest == before_digest, "source model changed during replay")
    report = {
        "stage": STAGE,
        "phase": "replay",
        "status": "COMPLETED",
        "PPO_ALLOWED": PPO_ALLOWED,
        "source_model_digest_before": before_digest,
        "source_model_digest_after": after_digest,
        "source_model_unchanged": True,
        "replay": {"exact": True, "steps": EXPECTED_STEPS, "requests": EXPECTED_REQUESTS},
        "lifecycle": lifecycle,
        "eligible": {"count": len(points), "failure_target_only": True},
        "scope": {"natural_failure_retained": True, "all_24_requests_retained": True,
                  "optimizer_calls": 0, "teacher_used": False, "heldout_used": False},
    }
    runner_utils._atomic_write_json(output / "selected.json", {"stage": STAGE, "phase": "replay", "selection": selection})
    runner_utils._atomic_write_json(output / "replay.json", report)
    _progress(output, "replay", "COMPLETED", EXPECTED_STEPS, selected=3)
    return report


def _capture_selected_states(policy: Any, config: Any, inputs: Mapping[str, Any], selected: Sequence[Mapping[str, Any]], output: Path) -> dict[int, dict[str, Any]]:
    wanted = {int(item["step"]): dict(item) for item in selected}
    states: dict[int, dict[str, Any]] = {}

    def observe(env: Any) -> dict[str, Any]:
        step = int(getattr(env, "step_count", -1))
        if step in wanted:
            snapshot = copy.deepcopy(env)
            paired_gate_diagnostic.restore_diagnostic_snapshot(snapshot)
            states[step] = {"snapshot": snapshot, "candidate": wanted[step]}
        return _failure_observer(policy)(env)

    replay = credit._capture_replay(
        policy, config, inputs["tape_path"], inputs["tape_rows"],
        progress_callback=lambda count: _progress(output, "selected_replay", "RUNNING", int(count)),
        state_observer=observe,
    )
    _require(bool(replay.get("replay", {}).get("exact")), "selected-state replay was not exact")
    _require(len(states) == len(selected), "selected state deep-copy count mismatch")
    rows_by_step = {int(row.get("step", index)): row for index, row in enumerate(inputs["tape_rows"])}
    captures_by_step = {int(row["step"]): row for row in replay["captures"]}
    for step, state in states.items():
        row = rows_by_step[step]
        actions, shortlists = runner_utils._row_inputs(row)
        capture = captures_by_step[step]
        uid_index = [int(uid) for uid in capture["uids"]].index(TARGET_UID)
        state.update({
            "first_actions": actions,
            "first_shortlists": shortlists,
            "target_world": np.asarray(capture["world"], dtype=np.float32).copy(),
            "target_world_row": int(uid_index),
            "source_observation_hash": str(row.get("observation_hash", "")),
        })
    return states


def _run_diagnose(output: Path, preflight: Mapping[str, Any]) -> dict[str, Any]:
    config, _snapshot = _strict_config()
    inputs = _load_fixed_inputs()
    selected_payload = _read_json(output / "selected.json")
    selected = list(selected_payload.get("selection", {}).get("selected", []))
    _require(len(selected) == 3, "diagnose requires exactly three registered states")
    policy, source_digest, _payload = _source_policy(config, inputs["source_path"])
    states = _capture_selected_states(policy, config, inputs, selected, output)
    after_policy, after_payload = rank_eval._load_policy(inputs["after_path"], config)
    after_digest = str(rank_eval._model_digest(after_policy))
    _require(after_digest == inputs["after_digest"], "after checkpoint digest differs from summary")
    reports: list[str] = []
    for index, item in enumerate(selected, start=1):
        step = int(item["step"])
        state = states[step]
        started = time.monotonic()
        callback = runner_utils._paired_branch_callback(
            output, selection_order=index, total_branches=26 * len(CONTINUATION_SEEDS),
            completed_states=index - 1, expected_states=len(selected), started=started,
            registered_selection_size=3,
        )
        paired = paired_gate_diagnostic.paired_gate_value(
            policy, state["snapshot"], state["first_actions"], state["first_shortlists"],
            TARGET_UID, CONTINUATION_SEEDS, BRANCH_MAX_STEPS, branch_callback=callback,
        )
        _require(bool(paired.get("model_unchanged")), "source model changed in paired diagnostic")
        _require(bool(paired.get("env_snapshot_unchanged")), "source environment snapshot changed in paired diagnostic")
        _require(bool(paired.get("rng_unchanged")), "caller RNG changed in paired diagnostic")
        delta_q = paired.get("qualified_delta_Q")
        target_world = np.asarray(state["target_world"], dtype=np.float32)[int(state["target_world_row"]):int(state["target_world_row"]) + 1]
        alignment = paired_gate_diagnostic.gate_update_alignment(
            policy, after_policy, target_world, delta_q=delta_q,
        )
        state_report = {
            "stage": STAGE, "phase": "diagnose", "status": "COMPLETED",
            "state": item, "paired": paired,
            "mechanism_metrics": {"D": delta_q, "C": paired.get("C_i"),
                                   "p": paired.get("p_release"), "m": alignment.get("local_m")},
            "alignment": {"source_model_digest": source_digest, "after_model_digest": after_digest,
                          "report": alignment,
                          "scope": "target-row probability alignment only; no causal claim"},
            "scope": {"continuation_seeds": list(CONTINUATION_SEEDS),
                      "branch_count": 26 * len(CONTINUATION_SEEDS),
                      "max_steps": BRANCH_MAX_STEPS,
                      "model_unchanged": bool(paired.get("model_unchanged")),
                      "env_snapshot_unchanged": bool(paired.get("env_snapshot_unchanged")),
                      "rng_unchanged": bool(paired.get("rng_unchanged")),
                      "retain_all_failed_censored_error": True,
                      "optimizer_calls": 0, "teacher_used": False, "heldout_used": False,
                      "population_estimate": False},
        }
        path = output / "diagnose" / f"state_{index:02d}.json"
        runner_utils._atomic_write_json(path, state_report)
        reports.append(str(path))
    after_source_digest = str(rank_eval._model_digest(policy))
    _require(after_source_digest == source_digest, "source model changed during diagnose")
    report = {
        "stage": STAGE, "phase": "diagnose", "status": "COMPLETED", "PPO_ALLOWED": PPO_ALLOWED,
        "selected_states": 3, "state_reports": reports,
        "source_model_unchanged": True,
        "after_alignment_model_digest": after_digest,
        "scope": {"independent_failure_mechanism_sample": True,
                   "not_merged_with_24_random_actor_samples": True,
                   "after_update_not_causal_for_batch_02_failure": True,
                   "optimizer_calls": 0, "teacher_used": False, "heldout_used": False},
    }
    runner_utils._atomic_write_json(output / "diagnose.json", report)
    _progress(output, "diagnose", "COMPLETED", 3)
    return report


def run_phase(phase: str, output_dir: Path) -> dict[str, Any]:
    output = Path(output_dir)
    _require(phase in {"preflight", "replay", "diagnose"}, f"unknown phase: {phase}")
    output.mkdir(parents=True, exist_ok=True)
    try:
        if phase == "preflight":
            report = _build_preflight()
            runner_utils._atomic_write_json(output / "preflight.json", report)
            runner_utils._atomic_write_json(output / "summary.json", {"stage": STAGE, "phase": phase, "status": report["status"]})
            return report
        preflight = _load_preflight(output)
        if phase == "replay":
            return _run_replay(output, preflight)
        return _run_diagnose(output, preflight)
    except Exception as error:
        _write_failed_phase(output, phase, error)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "replay", "diagnose"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    run_phase(args.phase, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
