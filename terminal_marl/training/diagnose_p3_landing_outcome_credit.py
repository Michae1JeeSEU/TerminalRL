"""Bounded landing-outcome opportunity probe for the three train dense24 emergencies.

This diagnostic keeps the calibrated 03c0 source policy frozen.  It first probes
WAIT source-tape prestates, obtains runtime-certified proposal sets from the
existing 25-singleton branch diagnostic, and only then runs the registered
full-cycle terminal-credit continuation on a small final selection.  It is a
local diagnostic artifact; it does not fit, update, or promote a policy.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from terminal_marl.core.safety_trace import SafetyTraceRecorder
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training import collect_p3_proposal_head_probe_v1 as collector
from terminal_marl.training import collect_p3_runtime_rank_repair as runtime_collector
from terminal_marl.training import diagnose_p3_action_credit as action_credit
from terminal_marl.training import diagnose_p3_terminal_credit as terminal_credit
from terminal_marl.training import evaluate_p3_proposal_head_probe_v1 as baseline_eval


STAGE = "P3_LANDING_OUTCOME_CREDIT_V1"
LOCAL_WINDOW_MODE = "P3_LANDING_OUTCOME_CREDIT_LOCAL_WINDOW_V1"
SOURCE_SEED = 873103
SOURCE_SCENARIO = "dense24"
SOURCE_SPLIT = "train"
TARGET_ARRIVAL_IDS = (12, 13, 21)
TARGET_PHASES = frozenset({"ring_outer", "ring_middle", "ring_inner"})
LOCAL_WINDOW_ARRIVAL_ID = 12
LOCAL_WINDOW_UID = 42
LOCAL_WINDOW_STEPS = (100, 101, 102, 103, 104)
MAX_PROBE_ROWS_PER_ARRIVAL = 40
MAX_FINAL_ROWS_PER_ARRIVAL = 3
REPEAT_COUNT = 3
SMOKE_REPEAT_COUNT = 1
SMOKE_STEP = 526
SMOKE_UID = 309
SMOKE_ARRIVAL_ID = 21
OBSERVATION_DIM = 88
PROPOSAL_COUNT = 25
DEFAULT_COLLECTION_DIR = Path(
    "/home/user/training_outputs/terminal/"
    "p3_runtime_rank_repair_20260926/collection"
)
DEFAULT_FORMAL_ROOT = collector.FORMAL_ROOT
DEFAULT_BASELINE_RECORD_ROOT = baseline_eval.BASELINE_RECORD_ROOT


class LandingOutcomeCreditError(ValueError):
    """Raised when the bounded landing-outcome probe input is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LandingOutcomeCreditError(message)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "value") and isinstance(value.value, str):
        return value.value
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _read_json(path: str | Path) -> Any:
    target = Path(path)
    _require(target.is_file(), f"JSON input missing: {target}")
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LandingOutcomeCreditError(f"invalid JSON input: {target}") from exc


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _policy_object(value: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get("policy", value)
    return getattr(value, "policy", value)


def _model_digest(value: Any) -> str:
    return str(collector.evaluator._model_digest(_policy_object(value)))


def _state_digest(env: Any) -> str:
    return str(collector._digest(SafetyTraceRecorder._state(env)))


def _enum_value(value: Any) -> str:
    raw = getattr(value, "value", value)
    text = str(raw)
    return text.rsplit(".", 1)[-1].lower()


def _window_metadata(*, local_window: bool, smoke: bool) -> dict[str, Any]:
    if not local_window:
        return {
            "local_window": False,
            "window_steps": [],
            "mode": "legacy_selection",
            "training_input_eligible": False,
        }
    steps = [LOCAL_WINDOW_STEPS[2]] if smoke else list(LOCAL_WINDOW_STEPS)
    return {
        "local_window": True,
        "window_steps": steps,
        "mode": LOCAL_WINDOW_MODE,
        "training_input_eligible": not smoke,
    }


def _uid_value(value: Any, uid: int, *, required: bool = False) -> Any:
    if isinstance(value, Mapping):
        if str(uid) in value:
            return value[str(uid)]
        if uid in value:
            return value[uid]
    if required:
        raise LandingOutcomeCreditError(f"source tape action field missing uid={uid}")
    return None


def _source_action_metadata(tape_row: Mapping[str, Any], uid: int) -> dict[str, Any]:
    action_field = None
    action = None
    for name in ("executed_mapped_action", "applied_action"):
        value = _uid_value(tape_row.get(name), uid)
        if value is not None:
            action_field = name
            action = value
            break
    _require(action_field is not None, f"source tape has no executed/applied action for uid={uid}")
    requested_gate = _uid_value(tape_row.get("requested_gate"), uid, required=True)
    requested_proposal = _uid_value(tape_row.get("requested_proposal"), uid)
    decoded = _uid_value(tape_row.get("decoded_action"), uid)
    vector = np.asarray(action, dtype=np.float32)
    _require(vector.shape == (3,) and bool(np.isfinite(vector).all()), f"source action vector is invalid for uid={uid}")
    executed_gate = "WAIT" if float(vector[0]) == 0.0 else "RELEASE" if float(vector[0]) == 1.0 else None
    _require(executed_gate in {"WAIT", "RELEASE"}, f"source action gate is not explicit for uid={uid}")
    return {
        "action_field": action_field,
        "action": _jsonable(action),
        "requested_gate": int(requested_gate),
        "requested_proposal": None if requested_proposal is None else int(requested_proposal),
        "executed_gate": executed_gate,
        "applied_gate": executed_gate,
        "decoded_action": _jsonable(decoded),
        "gate_logprob": _uid_value(tape_row.get("gate_logprob"), uid),
        "proposal_logprob": _uid_value(tape_row.get("proposal_logprob"), uid),
    }


def _light_probe_row(
    *,
    env: Any,
    actor: Any,
    record: Mapping[str, Any],
    source_policy: Any,
    step: int,
    source_action_metadata: Mapping[str, Any],
    source_observation_hash: Any,
) -> dict[str, Any]:
    uid = int(actor.uid)
    observations = env._observations()
    observation = observations.get(str(uid), observations.get(uid))
    _require(observation is not None, f"observation missing for uid={uid}")
    value = np.asarray(observation, dtype=np.float32)
    _require(value.shape == (OBSERVATION_DIM,), f"observation is not 88D for uid={uid}")
    _require(bool(np.isfinite(value).all()), f"observation is not finite for uid={uid}")
    policy = _policy_object(source_policy)
    rng_before = terminal_credit._rng_snapshot()
    try:
        with torch.no_grad():
            gate_distribution, proposal_distribution = policy.model.distributions(
                torch.as_tensor(value, dtype=torch.float32).reshape(1, OBSERVATION_DIM)
            )
        gate_logits = np.asarray(gate_distribution.logits[0].detach().cpu(), dtype=np.float32)
        proposal_logits = np.asarray(proposal_distribution.logits[0].detach().cpu(), dtype=np.float32)
    finally:
        terminal_credit._restore_rng(rng_before)
    _require(gate_logits.shape == (2,), f"source gate logits are not 2D for uid={uid}")
    _require(proposal_logits.shape == (PROPOSAL_COUNT,), f"source proposal logits are not 25D for uid={uid}")
    return {
        "seed": int(record["seed"]),
        "split": str(record["split"]),
        "scenario": str(record.get("scenario", SOURCE_SCENARIO)),
        "step": int(step),
        "uid": uid,
        "arrival_id": int(actor.arrival_id),
        "observation": value.copy(),
        "source_gate_logits": gate_logits.tolist(),
        "source_proposal_logits": proposal_logits.tolist(),
        "source_observation_hash": source_observation_hash,
        "source_action_metadata": dict(source_action_metadata),
        "soc": float(actor.soc),
        "priority": int(actor.base_priority),
        "phase": _enum_value(actor.phase),
        "ring_role": None if getattr(actor, "ring_role", None) is None else _enum_value(actor.ring_role),
        "teacher_state_only": True,
    }


def _source_lifecycle_rows(
    baseline_record_root: str | Path,
    *,
    seed: int = SOURCE_SEED,
) -> list[dict[str, Any]]:
    path = Path(baseline_record_root) / f"{SOURCE_SCENARIO}_{int(seed)}.json"
    payload = _read_json(path)
    metrics = payload.get("full_metrics") if isinstance(payload, Mapping) else None
    rows = metrics.get("rows") if isinstance(metrics, Mapping) else None
    _require(isinstance(rows, list) and rows, "source full lifecycle rows are missing")
    by_arrival = {int(row["arrival_id"]): dict(row) for row in rows if isinstance(row, Mapping)}
    for arrival_id in TARGET_ARRIVAL_IDS:
        row = by_arrival.get(int(arrival_id))
        _require(row is not None, f"source lifecycle target missing: arrival_id={arrival_id}")
        _require(
            bool(row.get("emergency_landing")) or str(row.get("outcome")) == "emergency_landed",
            f"source lifecycle target is not emergency: arrival_id={arrival_id}",
        )
    return [by_arrival[int(arrival_id)] for arrival_id in TARGET_ARRIVAL_IDS]


def _uniform_rows(rows: Sequence[Mapping[str, Any]], limit: int = MAX_PROBE_ROWS_PER_ARRIVAL) -> list[dict[str, Any]]:
    ordered = sorted((dict(row) for row in rows), key=lambda row: (int(row["step"]), int(row["uid"])))
    if len(ordered) <= int(limit):
        return ordered
    _require(int(limit) >= 2, "uniform row limit must be at least two")
    indices = [int(round(index * (len(ordered) - 1) / (int(limit) - 1))) for index in range(int(limit))]
    unique = list(dict.fromkeys(indices))
    if len(unique) < int(limit):
        for index in range(len(ordered)):
            if index not in unique:
                unique.append(index)
            if len(unique) == int(limit):
                break
    unique.sort()
    return [ordered[index] for index in unique]


def _probe_source_tape(
    *,
    record: Mapping[str, Any],
    config: Any,
    source_policy: Mapping[str, Any],
    target_arrivals: Sequence[int] = TARGET_ARRIVAL_IDS,
    smoke: bool = False,
    local_window: bool = False,
) -> dict[str, Any]:
    tape = Path(str(record["action_tape"]))
    payload = _read_json(tape)
    tape_rows = list(payload.get("action_tape", payload.get("records", [])))
    _require(tape_rows, f"source action tape is empty: {tape}")
    tape_by_step = {int(row["step"]): row for row in tape_rows if isinstance(row, Mapping)}
    target_set = {int(value) for value in target_arrivals}
    environment = TerminalParallelEnv(config)
    original_step = environment.step
    parent_rng = terminal_credit._rng_snapshot()
    model_before = _model_digest(source_policy)
    candidates: dict[tuple[int, int, int], dict[str, Any]] = {}

    def observed_step(*args: Any, **kwargs: Any) -> Any:
        step = int(environment.step_count)
        source_tape_row = tape_by_step.get(step)
        if source_tape_row is not None:
            actors = sorted(
                (
                    actor for actor in getattr(environment, "active", {}).values()
                    if int(getattr(actor, "arrival_id", -1)) in target_set
                    and _enum_value(getattr(actor, "phase", "")) in TARGET_PHASES
                ),
                key=lambda actor: (int(getattr(actor, "arrival_id")), int(getattr(actor, "uid"))),
            )
            for actor in actors:
                uid = int(actor.uid)
                metadata = _source_action_metadata(source_tape_row, uid)
                if local_window:
                    expected_step = LOCAL_WINDOW_STEPS[2] if smoke else None
                    if smoke:
                        keep = (
                            step == expected_step
                            and uid == LOCAL_WINDOW_UID
                            and int(actor.arrival_id) == LOCAL_WINDOW_ARRIVAL_ID
                        )
                    else:
                        keep = (
                            step in LOCAL_WINDOW_STEPS
                            and uid == LOCAL_WINDOW_UID
                            and int(actor.arrival_id) == LOCAL_WINDOW_ARRIVAL_ID
                        )
                    if not keep:
                        continue
                else:
                    if not smoke and metadata["executed_gate"] != "WAIT":
                        continue
                    if smoke and not (
                        step == SMOKE_STEP
                        and uid == SMOKE_UID
                        and int(actor.arrival_id) == SMOKE_ARRIVAL_ID
                    ):
                        continue
                row = _light_probe_row(
                    env=environment,
                    actor=actor,
                    record=record,
                    source_policy=source_policy,
                    step=step,
                    source_action_metadata=metadata,
                    source_observation_hash=source_tape_row.get("observation_hash"),
                )
                key = (step, uid, int(actor.arrival_id))
                candidates[key] = row
        return original_step(*args, **kwargs)

    environment.step = observed_step  # type: ignore[method-assign]
    replay_error: dict[str, Any] | None = None
    try:
        try:
            replay = SafetyTraceRecorder.replay_action_tape(environment, tape)
        except Exception as exc:
            replay_error = {"type": type(exc).__name__, "message": str(exc)}
            replay = {"exact": False, "steps_replayed": 0, "steps_expected": len(tape_rows)}
    finally:
        environment.step = original_step  # type: ignore[method-assign]
        terminal_credit._restore_rng(parent_rng)
    _require(replay_error is None and replay.get("exact") is True, f"source replay was not exact: {replay_error or replay}")
    model_after = _model_digest(source_policy)
    rng_unchanged = terminal_credit._rng_equal(parent_rng, terminal_credit._rng_snapshot())
    _require(model_before == model_after, "source probe changed source policy")
    _require(rng_unchanged, "source probe changed parent RNG")
    outcomes_match = collector._summary_outcomes(tape) == collector._replay_outcomes(environment)
    _require(outcomes_match, "source replay outcomes differ from source tape")
    rows = sorted(candidates.values(), key=lambda row: (int(row["arrival_id"]), int(row["step"]), int(row["uid"])))
    if local_window:
        expected_steps = {LOCAL_WINDOW_STEPS[2]} if smoke else set(LOCAL_WINDOW_STEPS)
        actual_steps = {
            int(row["step"])
            for row in rows
            if int(row["uid"]) == LOCAL_WINDOW_UID
            and int(row["arrival_id"]) == LOCAL_WINDOW_ARRIVAL_ID
        }
        _require(
            actual_steps == expected_steps and len(rows) == len(expected_steps),
            f"local window expected steps {sorted(expected_steps)}, got {sorted(actual_steps)}",
        )
        _require(
            all(str(row.get("phase")) in TARGET_PHASES for row in rows),
            "local window selected a non-ring phase",
        )
    elif smoke:
        _require(len(rows) == 1, f"smoke probe expected one fixed row, got {len(rows)}")
    else:
        _require(rows, "source replay found no target WAIT prestates")
    return {
        "record": dict(record),
        "action_tape": str(tape),
        "replay": {key: value for key, value in replay.items() if key != "steps"},
        "rows": rows,
        "checks": {
            "replay_exact": True,
            "source_policy_unchanged": model_before == model_after,
            "rng_unchanged": rng_unchanged,
            "outcomes_match": outcomes_match,
        },
    }


def _runtime_row(
    *,
    row: Mapping[str, Any],
    snapshot: Any,
    capture: Mapping[str, Any],
    source_policy: Mapping[str, Any],
    branch_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    parent_rng = terminal_credit._rng_snapshot()
    before_model = _model_digest(source_policy)
    catalogue_row = collector._state_row(
        snapshot,
        int(row["uid"]),
        policy=_policy_object(source_policy),
        seed=int(row["seed"]),
        split=str(row["split"]),
        step=int(row["step"]),
        target_step=int(row["step"]),
        slot_index=0,
    )
    light_observation = np.asarray(row["observation"], dtype=np.float32)
    catalogue_observation = np.asarray(catalogue_row["observation"], dtype=np.float32)
    _require(
        np.array_equal(light_observation, catalogue_observation),
        "selected observation changed before catalogue construction",
    )
    for key in ("source_gate_logits", "source_proposal_logits"):
        _require(
            np.array_equal(
                np.asarray(row[key], dtype=np.float32),
                np.asarray(catalogue_row[key], dtype=np.float32),
            ),
            f"selected {key} changed before catalogue construction",
        )
    evaluation_row = dict(catalogue_row)
    for key in (
        "observation",
        "source_gate_logits",
        "source_proposal_logits",
        "source_observation_hash",
        "source_action_metadata",
        "scenario",
        "seed",
        "split",
        "step",
        "uid",
        "arrival_id",
        "soc",
        "priority",
        "phase",
        "ring_role",
        "teacher_state_only",
    ):
        if key in row:
            evaluation_row[key] = row[key]
    terminal_credit._restore_rng(parent_rng)
    evaluation_row["proposal_probabilities"] = runtime_collector._softmax_probabilities(
        evaluation_row["source_proposal_logits"]
    )
    before_state = _state_digest(snapshot)
    branch_rng = terminal_credit._seeded_rng(int(branch_seed))
    branch_report = runtime_collector.runtime_diagnostic._evaluate_branches(
        snapshot,
        evaluation_row,
        source_policy,
        capture["actions"],
        capture["shortlists"],
        branch_rng,
    )
    terminal_credit._restore_rng(parent_rng)
    after_state = _state_digest(snapshot)
    after_model = _model_digest(source_policy)
    rng_unchanged = terminal_credit._rng_equal(parent_rng, terminal_credit._rng_snapshot())
    _require(before_state == after_state, "runtime C diagnostic changed selected snapshot")
    _require(before_model == after_model, "runtime C diagnostic changed source policy")
    _require(rng_unchanged, "runtime C diagnostic changed parent RNG")
    _require(branch_report.get("branch_rng_restored") is True, "runtime C branch RNG was not restored")
    labelled, summary = runtime_collector._label_row(evaluation_row, branch_report)
    runtime_c = sorted(int(value) for value in labelled.get("certified_proposals", ()))
    labelled["catalogue_certified_proposals"] = sorted(
        int(value) for value in labelled.get("catalogue_certified_proposals", ())
    )
    labelled["certified_proposals"] = runtime_c
    labelled["q_k3"] = float(branch_report["q_k3_runtime_labels"])
    labelled["source_release_probability"] = float(
        torch.softmax(torch.as_tensor(labelled["source_gate_logits"], dtype=torch.float32), dim=-1)[1]
    )
    labelled["runtime_probe_technical_checks"] = {
        "base_state_unchanged": before_state == after_state,
        "source_policy_unchanged": before_model == after_model,
        "parent_rng_unchanged": rng_unchanged,
        "branch_rng_restored": branch_report.get("branch_rng_restored") is True,
        "runtime_branch_count": int(summary["runtime_branch_count"]),
        "safety_counter_delta_nonzero_count": int(summary["safety_counter_delta_nonzero_count"]),
        "live_state_digest_before": branch_report.get("live_state_digest_before"),
        "live_state_digest_after": branch_report.get("live_state_digest_after"),
        "model_digest_before": branch_report.get("model_digest_before"),
        "model_digest_after": branch_report.get("model_digest_after"),
    }
    runtime_report = {
        "identity": {
            "seed": int(row["seed"]),
            "step": int(row["step"]),
            "uid": int(row["uid"]),
            "arrival_id": int(row["arrival_id"]),
        },
        "branch_seed": int(branch_seed),
        "branch_report": branch_report,
        "summary": summary,
        "technical_checks": labelled["runtime_probe_technical_checks"],
    }
    return labelled, runtime_report


def _runtime_probe(
    *,
    probe_rows: Sequence[Mapping[str, Any]],
    replay: Mapping[str, Any],
    source_policy: Mapping[str, Any],
    output_dir: Path,
    smoke: bool,
    progress: Callable[[Mapping[str, Any]], None],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    report_dir = output_dir / "runtime_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    for index, source_row in enumerate(probe_rows):
        key = terminal_credit._identity(source_row)
        snapshot = replay["snapshots"].get(key)
        capture = replay["captures"].get(key)
        _require(snapshot is not None and capture is not None, f"selected prestate missing: {key}")
        labelled, report = _runtime_row(
            row=source_row,
            snapshot=snapshot,
            capture=capture,
            source_policy=source_policy,
            branch_seed=20260928 + index,
        )
        labelled["runtime_report"] = str(report_dir / f"probe_{index:03d}.json")
        report["row"] = labelled
        _write_json(report_dir / f"probe_{index:03d}.json", report)
        rows.append(labelled)
        reports.append(report)
        _write_json(output_dir / "runtime_rows.json", {
            "stage": STAGE,
            "status": "RUNNING",
            "rows": rows,
            "processed_rows": len(rows),
            "expected_rows": len(probe_rows),
        })
        progress({
            "phase": "runtime_c",
            "status": "RUNNING",
            "runtime_rows_processed": len(rows),
            "runtime_rows_expected": len(probe_rows),
            "last_identity": report["identity"],
        })
    _require(rows, "runtime C produced no rows")
    return rows, {"reports": reports, "count": len(rows)}


def _score_value(row: Mapping[str, Any], kind: str) -> float:
    probability = float(row["source_release_probability"])
    q_k3 = float(row["q_k3"])
    if kind == "max_withheld":
        return (1.0 - probability) * q_k3
    if kind == "max_proposal_miss":
        return probability * (1.0 - q_k3)
    raise LandingOutcomeCreditError(f"unknown final selection score: {kind}")


def _select_final_states(
    runtime_rows: Sequence[Mapping[str, Any]],
    *,
    smoke: bool = False,
    local_window: bool = False,
) -> list[dict[str, Any]]:
    if local_window:
        expected_steps = {LOCAL_WINDOW_STEPS[2]} if smoke else set(LOCAL_WINDOW_STEPS)
        selected = sorted(
            (
                dict(row)
                for row in runtime_rows
                if int(row["seed"]) == SOURCE_SEED
                and str(row["scenario"]) == SOURCE_SCENARIO
                and str(row["split"]) == SOURCE_SPLIT
                and int(row["arrival_id"]) == LOCAL_WINDOW_ARRIVAL_ID
                and int(row["uid"]) == LOCAL_WINDOW_UID
                and int(row["step"]) in expected_steps
            ),
            key=lambda row: int(row["step"]),
        )
        actual_steps = {int(row["step"]) for row in selected}
        _require(
            len(selected) == len(expected_steps) and actual_steps == expected_steps,
            f"local window final selection expected steps {sorted(expected_steps)}, got {sorted(actual_steps)}",
        )
        for row in selected:
            row["final_selection_reason"] = f"local_window_step_{int(row['step'])}"
        return selected
    if smoke:
        selected = [
            dict(row) for row in runtime_rows
            if int(row["step"]) == SMOKE_STEP
            and int(row["uid"]) == SMOKE_UID
            and int(row["arrival_id"]) == SMOKE_ARRIVAL_ID
        ]
        _require(len(selected) == 1, "smoke final selection did not retain fixed row")
        selected[0]["final_selection_reason"] = "smoke_fixed_runtime_c"
        return selected
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in runtime_rows:
        if row.get("certified_proposals"):
            grouped.setdefault(int(row["arrival_id"]), []).append(dict(row))
    selected: list[dict[str, Any]] = []
    for arrival_id in TARGET_ARRIVAL_IDS:
        candidates = sorted(grouped.get(int(arrival_id), ()), key=lambda row: (int(row["step"]), int(row["uid"])))
        chosen: list[dict[str, Any]] = []
        for reason in ("max_withheld", "max_proposal_miss", "lowest_soc"):
            available = [
                row for row in candidates
                if all(
                    int(row["step"]) != int(previous["step"])
                    and abs(int(row["step"]) - int(previous["step"])) >= 10
                    for previous in chosen
                )
            ]
            if not available:
                continue
            if reason == "lowest_soc":
                pick = sorted(
                    available,
                    key=lambda row: (float(row["soc"]), int(row["step"]), int(row["uid"])),
                )[0]
            else:
                pick = sorted(
                    available,
                    key=lambda row: (-_score_value(row, reason), int(row["step"]), int(row["uid"])),
                )[0]
            pick["final_selection_reason"] = reason
            chosen.append(pick)
        selected.extend(chosen)
    _require(len(selected) <= len(TARGET_ARRIVAL_IDS) * MAX_FINAL_ROWS_PER_ARRIVAL, "final selection exceeds bound")
    return sorted(selected, key=lambda row: (int(row["arrival_id"]), int(row["step"]), int(row["uid"])))


def _outcomes(branch: Mapping[str, Any]) -> Mapping[str, Any]:
    lifecycle = branch.get("lifecycle")
    _require(isinstance(lifecycle, Mapping), "branch lifecycle is missing")
    values = lifecycle.get("outcomes")
    _require(isinstance(values, Mapping), "branch lifecycle outcomes are missing")
    return values


def _failed_count(branch: Mapping[str, Any]) -> int:
    return sum(int(value) for key, value in _outcomes(branch).items() if str(key) not in {"completed", "emergency_landed"})


def _normal_count(branch: Mapping[str, Any]) -> int:
    return int(_outcomes(branch).get("completed", 0))


def _target_normal(branch: Mapping[str, Any]) -> int:
    row = branch.get("target_terminal_row")
    if not isinstance(row, Mapping):
        return 0
    return int(str(row.get("outcome")) == "completed")


def _wait_metrics(branch: Mapping[str, Any]) -> dict[str, Any]:
    lifecycle = branch.get("lifecycle")
    lifecycle = lifecycle if isinstance(lifecycle, Mapping) else {}
    low = lifecycle.get("low_soc")
    low = low if isinstance(low, Mapping) else {}
    return {
        "priority_weighted_wait_seconds": lifecycle.get("priority_weighted_wait_seconds"),
        "mean_wait_seconds": lifecycle.get("mean_wait_seconds"),
        "low_soc_count": int(low.get("count", 0) or 0),
        "low_soc_mean_wait_seconds": low.get("mean_wait_seconds"),
    }


def _comparison_record(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    technical_ok: bool,
) -> dict[str, Any]:
    comparison = terminal_credit._compare_pair(baseline, candidate)
    baseline_normal = _normal_count(baseline)
    candidate_normal = _normal_count(candidate)
    baseline_failed = _failed_count(baseline)
    candidate_failed = _failed_count(candidate)
    return {
        "normal_delta": candidate_normal - baseline_normal,
        "failed_delta": candidate_failed - baseline_failed,
        "target_normal_delta": _target_normal(candidate) - _target_normal(baseline),
        "baseline_normal": baseline_normal,
        "candidate_normal": candidate_normal,
        "baseline_failed": baseline_failed,
        "candidate_failed": candidate_failed,
        "baseline_wait": _wait_metrics(baseline),
        "candidate_wait": _wait_metrics(candidate),
        "comparison": comparison,
        "technical_ok": bool(technical_ok),
    }


def _normal_opportunity(
    comparisons: Sequence[Mapping[str, Any]],
    *,
    technical_ok: bool | None = None,
) -> dict[str, Any]:
    normal = [float(item.get("normal_delta", 0.0)) for item in comparisons]
    failed = [float(item.get("failed_delta", 0.0)) for item in comparisons]
    technical = all(bool(item.get("technical_ok")) for item in comparisons)
    if technical_ok is not None:
        technical = technical and bool(technical_ok)
    repeat_count_valid = len(normal) == REPEAT_COUNT
    eligible = repeat_count_valid and bool(normal) and all(value >= 0.0 for value in normal) and any(
        value > 0.0 for value in normal
    ) and all(value <= 0.0 for value in failed) and technical
    return {
        "normal_opportunity": eligible,
        "technical_ok": technical,
        "normal_deltas": normal,
        "failed_deltas": failed,
        "mean_normal_gain": float(sum(normal) / len(normal)) if normal else 0.0,
        "repeat_count": len(normal),
        "repeat_count_valid": repeat_count_valid,
    }


def _state_opportunities(
    reports: Sequence[Mapping[str, Any]],
    row: Mapping[str, Any],
) -> dict[str, Any]:
    gate_release: list[dict[str, Any]] = []
    gate_wait: list[dict[str, Any]] = []
    proposal: list[dict[str, Any]] = []
    for report in reports:
        branches = report.get("branches")
        if not isinstance(branches, Mapping):
            continue
        technical_ok = bool(report.get("acceptance") is True)
        gate_release.append(
            _comparison_record(branches["wait"], branches["source_release"], technical_ok=technical_ok)
        )
        gate_wait.append(
            _comparison_record(branches["source_release"], branches["wait"], technical_ok=technical_ok)
        )
        proposal.append(
            _comparison_record(
                branches["source_release"],
                branches["runtime_certified_release"],
                technical_ok=technical_ok,
            )
        )
    gate_release_opportunity = _normal_opportunity(gate_release)
    gate_wait_opportunity = _normal_opportunity(gate_wait)
    proposal_opportunity = _normal_opportunity(proposal)
    p_release = float(row["source_release_probability"])
    release_gap = (1.0 - p_release) * float(gate_release_opportunity["mean_normal_gain"])
    wait_gap = p_release * float(gate_wait_opportunity["mean_normal_gain"])
    release_valid = bool(gate_release_opportunity["normal_opportunity"])
    wait_valid = bool(gate_wait_opportunity["normal_opportunity"])
    if release_valid and not wait_valid:
        gate_direction = "release"
        gate_gap = release_gap
    elif wait_valid and not release_valid:
        gate_direction = "wait"
        gate_gap = wait_gap
    else:
        gate_direction = "uncertain"
        gate_gap = 0.0
    return {
        "gate": {
            "direction": gate_direction,
            "release": {
                "comparisons": gate_release,
                "normal_opportunity": gate_release_opportunity,
                "source_policy_gap": release_gap if release_valid else 0.0,
            },
            "wait": {
                "comparisons": gate_wait,
                "normal_opportunity": gate_wait_opportunity,
                "source_policy_gap": wait_gap if wait_valid else 0.0,
            },
            "source_release_probability": p_release,
            "source_policy_gap": gate_gap,
            "training_label_eligible": bool(gate_direction == "release" and release_valid),
        },
        "proposal": {
            "comparisons": proposal,
            "normal_opportunity": proposal_opportunity,
            "direction": "release" if proposal_opportunity["normal_opportunity"] else "uncertain",
        },
        "local_mc_note": "local three-repeat MC estimate does not represent generalization",
    }


def _write_continuation_state(
    *,
    output_dir: Path,
    row: Mapping[str, Any],
    snapshot: Any,
    capture: Mapping[str, Any],
    source_policy: Mapping[str, Any],
    state_index: int,
    repeat_count: int,
) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    state_dir = output_dir / "states" / f"state_{state_index:03d}"
    for repeat in range(int(repeat_count)):
        report = terminal_credit._state_report(
            state_index=state_index,
            row=row,
            snapshot=snapshot,
            capture=capture,
            policy=_policy_object(source_policy),
            config=row["_config"],
            repeat=repeat,
            horizon=int(row["_horizon"]),
        )
        report["acceptance"] = bool(terminal_credit._state_acceptance(report))
        report["state_index"] = int(state_index)
        report["repeat"] = int(repeat)
        report["final_selection_reason"] = str(row.get("final_selection_reason"))
        for branch_name, branch in report.get("branches", {}).items():
            _write_json(state_dir / f"rep_{repeat}" / f"{branch_name}.json", branch)
        compact = copy.deepcopy(report)
        compact["branches"] = {
            branch_name: terminal_credit._branch_summary(
                branch,
                state_dir / f"rep_{repeat}" / f"{branch_name}.json",
            )
            for branch_name, branch in report.get("branches", {}).items()
        }
        _write_json(state_dir / f"rep_{repeat}" / "report.json", compact)
        reports.append(report)
        if not report["acceptance"]:
            raise LandingOutcomeCreditError(
                f"continuation state failed acceptance: state={state_index} repeat={repeat}"
            )
    return {"row": dict(row), "reports": reports}


def _summary(
    *,
    state_results: Sequence[Mapping[str, Any]],
    runtime_rows: Sequence[Mapping[str, Any]],
    selection_path: Path,
    local_window: bool = False,
    smoke: bool = False,
) -> dict[str, Any]:
    gate_states: list[dict[str, Any]] = []
    proposal_states: list[dict[str, Any]] = []
    state_summaries: list[dict[str, Any]] = []
    runtime_branch_runs = 0
    for row in runtime_rows:
        checks = row.get("runtime_probe_technical_checks")
        _require(isinstance(checks, Mapping), "runtime row technical checks are missing")
        _require("runtime_branch_count" in checks, "runtime row branch count is missing")
        runtime_branch_runs += int(checks["runtime_branch_count"])
    continuation_logical_branches = 0
    continuation_actual_runs = 0
    reused_source_branch_count = 0
    for result in state_results:
        row = result["row"]
        reports = result["reports"]
        for report in reports:
            branches = report.get("branches")
            _require(isinstance(branches, Mapping), "continuation branch reports are missing")
            continuation_logical_branches += len(branches)
            continuation_actual_runs += sum(
                int(branch.get("reused_source_branch") is not True)
                for branch in branches.values()
                if isinstance(branch, Mapping)
            )
            reused_source_branch_count += sum(
                int(branch.get("reused_source_branch") is True)
                for branch in branches.values()
                if isinstance(branch, Mapping)
            )
        opportunities = _state_opportunities(reports, row)
        state_entry = {
            "state_index": int(reports[0]["state_index"]),
            "repeat_count": len(reports),
            "seed": int(row["seed"]),
            "step": int(row["step"]),
            "uid": int(row["uid"]),
            "arrival_id": int(row["arrival_id"]),
            "final_selection_reason": row.get("final_selection_reason"),
            "gate": opportunities["gate"],
            "proposal": opportunities["proposal"],
            "local_mc_note": opportunities["local_mc_note"],
        }
        gate_states.append(state_entry["gate"] | {"state_index": state_entry["state_index"]})
        proposal_states.append(state_entry["proposal"] | {"state_index": state_entry["state_index"]})
        state_summaries.append(state_entry)
    metadata = _window_metadata(local_window=local_window, smoke=smoke) if local_window else {}
    return {
        "stage": STAGE,
        "status": "COMPLETED",
        **metadata,
        "selection": str(selection_path),
        "runtime_rows_count": len(runtime_rows),
        "runtime_branch_runs": runtime_branch_runs,
        "final_state_count": len(state_results),
        "continuation_logical_branches": continuation_logical_branches,
        "continuation_actual_runs": continuation_actual_runs,
        "reused_source_branch_count": reused_source_branch_count,
        "gate": {"states": gate_states},
        "proposal": {"states": proposal_states},
        "states": state_summaries,
        "local_mc_note": "local three-repeat MC estimate does not represent generalization",
    }


def run(
    *,
    output_dir: str | Path,
    collection_dir: str | Path = DEFAULT_COLLECTION_DIR,
    formal_root: str | Path = DEFAULT_FORMAL_ROOT,
    baseline_record_root: str | Path = DEFAULT_BASELINE_RECORD_ROOT,
    smoke: bool = False,
    local_window: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir)
    _require(not output.exists(), f"refusing existing output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    progress_path = output / "progress.json"
    window_metadata = _window_metadata(local_window=local_window, smoke=smoke) if local_window else {}
    target_arrivals = (LOCAL_WINDOW_ARRIVAL_ID,) if local_window else TARGET_ARRIVAL_IDS

    def progress(value: Mapping[str, Any]) -> None:
        payload = {"stage": STAGE, "status": "RUNNING", **dict(value)}
        _write_json(progress_path, payload)

    progress({
        "phase": "initial",
        **window_metadata,
        "probe_rows_processed": 0,
        "runtime_rows_processed": 0,
        "states_processed": 0,
    })
    reports: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    state_results: list[dict[str, Any]] = []
    try:
        source_lifecycle = _source_lifecycle_rows(baseline_record_root)
        lifecycle_by_arrival = {int(row["arrival_id"]): row for row in source_lifecycle}
        protocol, configs, source_records, formal_manifest = collector._load_source_inputs(Path(formal_root))
        source_record = next(
            (dict(record) for record in source_records if int(record.get("seed", -1)) == SOURCE_SEED),
            None,
        )
        _require(source_record is not None, f"source record missing: {SOURCE_SEED}")
        _require(str(source_record.get("scenario")) == SOURCE_SCENARIO, "source scenario changed")
        _require(str(source_record.get("split")) == SOURCE_SPLIT, "source split changed")
        source_policy = collector._load_source_policy(protocol, configs[SOURCE_SCENARIO])
        probe = _probe_source_tape(
            record=source_record,
            config=configs[SOURCE_SCENARIO],
            source_policy=source_policy,
            target_arrivals=target_arrivals,
            smoke=smoke,
            local_window=local_window,
        )
        if local_window:
            probe_rows = list(probe["rows"])
        elif smoke:
            probe_rows = probe["rows"]
        else:
            probe_rows = [
                row for arrival_id in TARGET_ARRIVAL_IDS
                for row in _uniform_rows(
                    [candidate for candidate in probe["rows"] if int(candidate["arrival_id"]) == int(arrival_id)],
                    MAX_PROBE_ROWS_PER_ARRIVAL,
                )
            ]
        _require(probe_rows, "probe selection is empty")
        _write_json(output / "probe_selection.json", {
            "stage": STAGE,
            **window_metadata,
            "source_seed": SOURCE_SEED,
            "target_arrival_ids": list(target_arrivals),
            "smoke": bool(smoke),
            "fixture_not_full_selection": bool(smoke),
            "source_lifecycle_targets": source_lifecycle,
            "source_replay": probe["replay"],
            "checks": probe["checks"],
            "rows": probe_rows,
        })
        progress({
            "phase": "probe_selection",
            "probe_rows_processed": len(probe_rows),
            "probe_rows_expected": len(probe_rows),
            "runtime_rows_processed": 0,
            "states_processed": 0,
        })

        parent_rng = terminal_credit._rng_snapshot()
        model_before = _model_digest(source_policy)
        replay = terminal_credit._replay_seed(
            record=source_record,
            config=configs[SOURCE_SCENARIO],
            target_rows=probe_rows,
        )
        model_after = _model_digest(source_policy)
        rng_unchanged = terminal_credit._rng_equal(parent_rng, terminal_credit._rng_snapshot())
        _require(replay["replay"].get("exact") is True, "selected-point exact replay failed")
        _require(model_before == model_after, "selected-point replay changed source policy")
        _require(rng_unchanged, "selected-point replay changed parent RNG")
        runtime_rows, runtime_report = _runtime_probe(
            probe_rows=probe_rows,
            replay=replay,
            source_policy=source_policy,
            output_dir=output,
            smoke=smoke,
            progress=progress,
        )
        _write_json(output / "runtime_rows.json", {
            "stage": STAGE,
            "status": "COMPLETED",
            "rows": runtime_rows,
            "processed_rows": len(runtime_rows),
            "technical_checks": runtime_report,
        })
        final_rows = _select_final_states(runtime_rows, smoke=smoke, local_window=local_window)
        selection_payload = {
            "stage": STAGE,
            **window_metadata,
            "source_seed": SOURCE_SEED,
            "target_arrival_ids": list(target_arrivals),
            "smoke": bool(smoke),
            "fixture_not_full_selection": bool(smoke),
            "probe_selection": str(output / "probe_selection.json"),
            "runtime_rows": str(output / "runtime_rows.json"),
            "selected_rows": final_rows,
            "selected_count": len(final_rows),
            "selection_bound": (
                len(window_metadata["window_steps"])
                if local_window
                else len(TARGET_ARRIVAL_IDS) * MAX_FINAL_ROWS_PER_ARRIVAL
            ),
            "selection_algorithm": (
                ["local_window_step_order"]
                if local_window
                else ["max_withheld", "max_proposal_miss", "lowest_soc"]
            ),
        }
        _write_json(output / "selection.json", selection_payload)
        progress({
            "phase": "final_selection",
            "probe_rows_processed": len(probe_rows),
            "runtime_rows_processed": len(runtime_rows),
            "final_rows_selected": len(final_rows),
            "states_processed": 0,
        })
        snapshots = replay["snapshots"]
        captures = replay["captures"]
        horizon = int(round(float(configs[SOURCE_SCENARIO].max_simulation_seconds) / float(configs[SOURCE_SCENARIO].dt)))
        repeat_count = SMOKE_REPEAT_COUNT if smoke else REPEAT_COUNT
        for state_index, selected_row in enumerate(final_rows):
            key = terminal_credit._identity(selected_row)
            snapshot = snapshots.get(key)
            capture = captures.get(key)
            _require(snapshot is not None and capture is not None, f"final selected snapshot missing: {key}")
            row_for_state = dict(selected_row)
            row_for_state["_config"] = configs[SOURCE_SCENARIO]
            row_for_state["_horizon"] = horizon
            progress({
                "phase": "continuation",
                "branch_status": "RUNNING",
                "state_index": state_index,
                "repeat_count": repeat_count,
                "branch_names": list(terminal_credit.BRANCH_NAMES),
                "runtime_rows_processed": len(runtime_rows),
                "states_processed": state_index,
            })
            result = _write_continuation_state(
                output_dir=output,
                row=row_for_state,
                snapshot=snapshot,
                capture=capture,
                source_policy=source_policy,
                state_index=state_index,
                repeat_count=repeat_count,
            )
            state_results.append(result)
            progress({
                "phase": "continuation",
                "branch_status": "COMPLETED",
                "state_index": state_index,
                "repeat_count": repeat_count,
                "branch_names": list(terminal_credit.BRANCH_NAMES),
                "runtime_rows_processed": len(runtime_rows),
                "states_processed": len(state_results),
            })
        summary = _summary(
            state_results=state_results,
            runtime_rows=runtime_rows,
            selection_path=output / "selection.json",
            local_window=local_window,
            smoke=smoke,
        )
        _write_json(output / "summary.json", summary)
        report = {
            "stage": STAGE,
            **window_metadata,
            "status": "SMOKE_COMPLETED" if smoke else "COMPLETED",
            "output_dir": str(output),
            "source_seed": SOURCE_SEED,
            "source_scenario": SOURCE_SCENARIO,
            "source_split": SOURCE_SPLIT,
            "fixture_not_full_selection": bool(smoke),
            "source_checkpoint_sha256": collector.SOURCE_CHECKPOINT_SHA256,
            "source_formal_manifest": str(formal_manifest),
            "source_collection_dir": str(collection_dir),
            "baseline_record_root": str(baseline_record_root),
            "probe_selection": str(output / "probe_selection.json"),
            "runtime_rows": str(output / "runtime_rows.json"),
            "selection": str(output / "selection.json"),
            "summary": str(output / "summary.json"),
            "probe_rows_count": len(probe_rows),
            "runtime_rows_count": len(runtime_rows),
            "runtime_branch_runs": int(summary["runtime_branch_runs"]),
            "final_state_count": len(state_results),
            "continuation_logical_branches": int(summary["continuation_logical_branches"]),
            "continuation_actual_runs": int(summary["continuation_actual_runs"]),
            "reused_source_branch_count": int(summary["reused_source_branch_count"]),
            "repeat_count": repeat_count,
            "training": False,
            "policy_sampling": bool(state_results),
            "policy_update": False,
            "PPO_ALLOWED": False,
            "source_lifecycle_targets": lifecycle_by_arrival,
        }
        _write_json(output / "report.json", report)
        _write_json(progress_path, {
            "stage": STAGE,
            "status": report["status"],
            "phase": "completed",
            "probe_rows_processed": len(probe_rows),
            "runtime_rows_processed": len(runtime_rows),
            "states_processed": len(state_results),
        })
        return report
    except BaseException as exc:
        failure = {
            "stage": STAGE,
            **window_metadata,
            "status": "FAILED",
            "output_dir": str(output),
            "source_seed": SOURCE_SEED,
            "source_scenario": SOURCE_SCENARIO,
            "target_arrival_ids": list(target_arrivals),
            "probe_rows_count": 0,
            "runtime_rows_count": len(runtime_rows),
            "states_processed": len(state_results),
            "training": False,
            "policy_sampling": bool(state_results),
            "policy_update": False,
            "PPO_ALLOWED": False,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        _write_json(output / "report.json", failure)
        _write_json(progress_path, {"stage": STAGE, "status": "FAILED", "error": failure["error"]})
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--collection-dir", type=Path, default=DEFAULT_COLLECTION_DIR)
    parser.add_argument("--formal-root", type=Path, default=DEFAULT_FORMAL_ROOT)
    parser.add_argument("--baseline-record-root", type=Path, default=DEFAULT_BASELINE_RECORD_ROOT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--local-window", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(
            output_dir=args.output_dir,
            collection_dir=args.collection_dir,
            formal_root=args.formal_root,
            baseline_record_root=args.baseline_record_root,
            smoke=bool(args.smoke),
            local_window=bool(args.local_window),
        )
    except BaseException as exc:
        print(json.dumps({"stage": STAGE, "status": "FAILED", "error": {"type": type(exc).__name__, "message": str(exc)}}, sort_keys=True), flush=True)
        return 1
    print(json.dumps(_jsonable(result), sort_keys=True), flush=True)
    return 0 if result.get("status") in {"COMPLETED", "SMOKE_COMPLETED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
