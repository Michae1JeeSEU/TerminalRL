"""Re-label the frozen proposal collection with execution-context K3 labels.

The source rows and source tapes are read-only inputs.  Each registered seed is
replayed once, and the requested actions and proposal shortlist passed to the
live ``env.step`` are captured before evaluating the WAIT/K3/singleton branch
diagnostic.  The diagnostic deep-copies the live environment for every branch,
so the source replay itself is never changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from terminal_marl.core.safety_trace import SafetyTraceRecorder
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training import collect_p3_proposal_head_probe_v1 as collector
from terminal_marl.training import diagnose_p3_action_credit as runtime_diagnostic
from terminal_marl.training import train_p3_proposal_head_probe_v1 as trainer


STAGE = "P3_RUNTIME_RANK_REPAIR_COLLECTION"
DEFAULT_COLLECTION_DIR = Path(
    "/home/user/training_outputs/terminal/"
    "p3_proposal_head_probe_v1_20260926/collection"
)
FORMAL_ROOT = collector.FORMAL_ROOT
OBSERVATION_DIM = 88
PROPOSAL_COUNT = 25
EXPECTED_ROWS = 164
EXPECTED_RECORDS = 6
EXPECTED_SPLIT_COUNTS = {"train": 82, "validation": 82}
LABEL_SCOPE = "same_prestate_fixed_other_tape_actions_runtime_singleton"


class RuntimeRankCollectionError(ValueError):
    """Raised when the source collection or replay contract is not met."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeRankCollectionError(message)


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


def _read_json(path: Path) -> Any:
    _require(path.is_file(), f"JSON input missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeRankCollectionError(f"invalid JSON: {path}") from exc


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rng_snapshot() -> tuple[Any, tuple[Any, ...], torch.Tensor]:
    return random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone()


def _restore_rng(snapshot: tuple[Any, tuple[Any, ...], torch.Tensor]) -> None:
    random.setstate(snapshot[0])
    np.random.set_state(snapshot[1])
    torch.random.set_rng_state(snapshot[2].clone())


def _rng_equal(left: tuple[Any, tuple[Any, ...], torch.Tensor], right: tuple[Any, tuple[Any, ...], torch.Tensor]) -> bool:
    return bool(runtime_diagnostic._rng_equal(left, right))


def _identity(row: Mapping[str, Any]) -> tuple[str, int, int, int, int]:
    return (
        str(row.get("split", "")),
        int(row["seed"]),
        int(row["step"]),
        int(row["uid"]),
        int(row["arrival_id"]),
    )


def _step_inputs(
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, tuple[np.ndarray, ...]]]:
    raw_actions = args[0] if args else kwargs.get("actions", {})
    raw_shortlists = (
        kwargs.get("proposal_shortlists")
        if "proposal_shortlists" in kwargs
        else (args[1] if len(args) > 1 else {})
    )
    actions = {
        str(uid): np.asarray(action, dtype=np.float32).copy()
        for uid, action in (raw_actions or {}).items()
    }
    shortlists = {
        str(uid): tuple(np.asarray(action, dtype=np.float32).copy() for action in values)
        for uid, values in (raw_shortlists or {}).items()
    }
    return actions, shortlists


def _observation_for(env: Any, uid: int) -> np.ndarray:
    observations = env._observations()
    observation = observations.get(str(uid), observations.get(uid))
    _require(observation is not None, f"observation missing for uid={uid}")
    value = np.asarray(observation, dtype=np.float32)
    _require(value.shape == (OBSERVATION_DIM,), f"observation is not 88D for uid={uid}")
    _require(bool(np.isfinite(value).all()), f"observation is not finite for uid={uid}")
    return value.copy()


def _validate_live_target(env: Any, row: Mapping[str, Any]) -> np.ndarray:
    uid = int(row["uid"])
    actor = getattr(env, "active", {}).get(uid)
    if actor is None:
        actor = getattr(env, "active", {}).get(str(uid))
    _require(actor is not None, f"target uid is not active: {uid}")
    _require(int(getattr(actor, "arrival_id")) == int(row["arrival_id"]),
             f"target arrival_id mismatch for uid={uid} step={row['step']}")
    observation = _observation_for(env, uid)
    expected = np.asarray(row["observation"], dtype=np.float32)
    _require(expected.shape == (OBSERVATION_DIM,), f"source observation is not 88D for uid={uid}")
    _require(np.array_equal(observation, expected),
             f"target observation mismatch for uid={uid} step={row['step']}")
    return observation


def _policy_object(source_policy: Any) -> Any:
    if isinstance(source_policy, Mapping):
        return source_policy.get("policy", source_policy)
    return getattr(source_policy, "policy", source_policy)


def _model_digest(source_policy: Any, *, use_recorded_before: bool = False) -> str | None:
    if use_recorded_before and isinstance(source_policy, Mapping):
        value = source_policy.get("model_digest_before")
        if value is not None:
            return str(value)
    try:
        return str(collector.evaluator._model_digest(_policy_object(source_policy)))
    except Exception:
        return None


def _softmax_probabilities(logits: Sequence[float]) -> list[float]:
    tensor = torch.as_tensor(list(logits), dtype=torch.float32)
    _require(tensor.shape == (PROPOSAL_COUNT,), "source proposal logits must be 25D")
    _require(bool(torch.isfinite(tensor).all().item()), "source proposal logits are not finite")
    return [float(value) for value in torch.softmax(tensor, dim=-1).cpu().tolist()]


def _runtime_values(branch_report: Mapping[str, Any]) -> tuple[list[int], float, str]:
    runtime = branch_report.get("runtime_certified_proposals")
    _require(runtime is not None, "runtime_certified_proposals is missing")
    _require(isinstance(runtime, (list, tuple, set, frozenset)), "runtime C is not a sequence")
    runtime_c = sorted({int(value) for value in runtime})
    _require(all(0 <= value < PROPOSAL_COUNT for value in runtime_c), "runtime C index out of range")
    error = branch_report.get("runtime_q_vs_approved_mass_abs_error")
    _require(error is not None, "runtime q error is missing")
    error_value = float(error)
    _require(math.isfinite(error_value) and error_value >= 0.0, "runtime q error is invalid")
    scope = str(branch_report.get("label_scope"))
    _require(scope == LABEL_SCOPE, f"runtime label scope changed: {scope}")
    return runtime_c, error_value, scope


def _runtime_label_hash(row: Mapping[str, Any], runtime_c: Sequence[int], error: float, scope: str) -> str:
    payload = {
        "arrival_id": int(row["arrival_id"]),
        "catalogue_certified_proposals": sorted(int(value) for value in row.get("certified_proposals", [])),
        "q_runtime_vs_actual_error": float(error),
        "runtime_certified_proposals": sorted(int(value) for value in runtime_c),
        "scope": str(scope),
        "seed": int(row["seed"]),
        "split": str(row["split"]),
        "step": int(row["step"]),
        "uid": int(row["uid"]),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _branch_safety_counter_deltas(branches: Sequence[Any]) -> list[dict[str, Any]]:
    nonzero_deltas: list[dict[str, Any]] = []
    for index, branch in enumerate(branches):
        _require(isinstance(branch, Mapping), f"runtime branch {index} is not an object")
        _require("decision" in branch, f"runtime branch {index} decision is missing")
        for field in ("reward", "next_value"):
            value = branch.get(field)
            _require(value is not None and math.isfinite(float(value)), f"runtime branch {index} {field} is not finite")
        delta = branch.get("safety_counter_delta")
        _require(isinstance(delta, Mapping), f"runtime branch {index} safety_counter_delta is missing")
        for name, value in delta.items():
            numeric = float(value)
            _require(math.isfinite(numeric), f"runtime branch {index} safety counter is not finite")
            if numeric != 0.0:
                nonzero_deltas.append({"branch_index": index, "counter": str(name), "delta": numeric})
    return nonzero_deltas


def _label_row(row: Mapping[str, Any], branch_report: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    wait = branch_report.get("wait")
    releases = branch_report.get("releases")
    singleton = branch_report.get("runtime_singleton_branches")
    _require(isinstance(wait, Mapping), "runtime WAIT branch is missing")
    _require(isinstance(releases, Sequence) and not isinstance(releases, (str, bytes)) and len(releases) == PROPOSAL_COUNT,
             "runtime K3 branch count is not 25")
    _require(isinstance(singleton, Sequence) and not isinstance(singleton, (str, bytes)) and len(singleton) == PROPOSAL_COUNT,
             "runtime singleton branch count is not 25")
    branches = [wait, *releases, *singleton]
    nonzero_deltas = _branch_safety_counter_deltas(branches)
    _require(not nonzero_deltas, "runtime branch safety counter delta is nonzero")
    decision_counts: dict[str, int] = {}
    for branch in branches:
        name = str(branch.get("decision"))
        decision_counts[name] = decision_counts.get(name, 0) + 1
    catalogue = sorted({int(value) for value in row.get("certified_proposals", [])})
    runtime_c, error, scope = _runtime_values(branch_report)
    labelled = dict(row)
    labelled["catalogue_certified_proposals"] = catalogue
    labelled["certified_proposals"] = runtime_c
    labelled["q_runtime_vs_actual_error"] = error
    labelled["label_scope"] = scope
    label_hash = _runtime_label_hash(row, runtime_c, error, scope)
    labelled["runtime_label_hash"] = label_hash
    labelled["runtime_branch_count"] = len(branches)
    labelled["safety_counter_delta_nonzero_count"] = len(nonzero_deltas)
    labelled["runtime_branch_decision_counts"] = decision_counts
    summary = {
        "identity": _identity(row),
        "catalogue_certified_proposals": catalogue,
        "runtime_certified_proposals": runtime_c,
        "q_runtime_vs_actual_error": error,
        "label_scope": scope,
        "runtime_label_hash": label_hash,
        "runtime_branch_count": len(branches),
        "safety_counter_delta_nonzero_count": len(nonzero_deltas),
        "runtime_branch_decision_counts": decision_counts,
        "branch_invariants": {
            key: branch_report.get(key)
            for key in (
                "live_state_digest_before",
                "live_state_digest_after",
                "model_digest_before",
                "model_digest_after",
                "branch_rng_restored",
                "actual_k3_approved_mass",
                "runtime_q_vs_approved_mass_abs_error",
            )
            if key in branch_report
        },
    }
    return labelled, summary


def _prefix_tape(tape: Path, step: int, directory: Path) -> Path:
    payload = _read_json(tape)
    rows = list(payload.get("action_tape", payload.get("records", [])))
    _require(0 <= int(step) < len(rows), f"selected step outside tape: {step}")
    prefix = dict(payload)
    prefix["action_tape"] = rows[: int(step) + 1]
    path = directory / "prefix.json"
    _write_json_atomic(path, prefix)
    return path


def _collect_record(
    *,
    record: Mapping[str, Any],
    config: Any,
    source_policy: Any,
    target_rows: Sequence[Mapping[str, Any]],
    smoke: bool = False,
    source_tape: Path | None = None,
    on_labelled_row: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    tape = source_tape or Path(str(record["action_tape"]))
    payload = _read_json(tape)
    tape_rows = list(payload.get("action_tape", payload.get("records", [])))
    _require(tape_rows, f"empty action tape: {tape}")
    by_step: dict[int, list[dict[str, Any]]] = {}
    for source_row in target_rows:
        by_step.setdefault(int(source_row["step"]), []).append(dict(source_row))

    environment = TerminalParallelEnv(config)
    original_step = environment.step
    attempted: set[tuple[str, int, int, int, int]] = set()
    labelled: dict[tuple[str, int, int, int, int], dict[str, Any]] = {}
    row_summaries: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    captures: list[dict[str, Any]] = []
    safety_counter_nonzero_count = 0
    replay_before = _rng_snapshot()
    model_before = _model_digest(source_policy, use_recorded_before=True)

    def observed_step(*args: Any, **kwargs: Any) -> Any:
        step = int(getattr(environment, "step_count", -1))
        rows_here = by_step.get(step, [])
        if rows_here:
            actions, shortlists = _step_inputs(args, kwargs)
            branch_rng = _rng_snapshot()
            capture = {
                "step": step,
                "args": _jsonable(args),
                "kwargs": _jsonable(kwargs),
                "actions": _jsonable(actions),
                "proposal_shortlists": _jsonable(shortlists),
                "rows": [],
            }
            for source_row in rows_here:
                key = _identity(source_row)
                attempted.add(key)
                row_capture = {"identity": key, "status": "ERROR"}
                branch_report: Mapping[str, Any] | None = None
                try:
                    _validate_live_target(environment, source_row)
                    evaluation_row = dict(source_row)
                    evaluation_row["proposal_probabilities"] = _softmax_probabilities(
                        source_row["source_proposal_logits"]
                    )
                    branch_report = runtime_diagnostic._evaluate_branches(
                        environment,
                        evaluation_row,
                        source_policy,
                        actions,
                        shortlists,
                        branch_rng,
                    )
                    labelled_row, summary = _label_row(source_row, branch_report)
                    labelled[key] = labelled_row
                    row_summaries.append(summary)
                    row_capture.update({"status": "OK", **summary})
                    if on_labelled_row is not None:
                        on_labelled_row(labelled_row)
                except Exception as exc:
                    nonzero_count = 0
                    if isinstance(branch_report, Mapping):
                        try:
                            candidate_branches = [
                                branch_report.get("wait"),
                                *(branch_report.get("releases") or []),
                                *(branch_report.get("runtime_singleton_branches") or []),
                            ]
                            nonzero_count = len(_branch_safety_counter_deltas(candidate_branches))
                        except Exception:
                            nonzero_count = 0
                    safety_counter_nonzero_count += nonzero_count
                    error = {
                        "kind": "runtime_label",
                        "identity": key,
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "safety_counter_delta_nonzero_count": nonzero_count,
                    }
                    errors.append(error)
                    row_capture["error"] = error
                    row_capture["safety_counter_delta_nonzero_count"] = nonzero_count
                finally:
                    _restore_rng(branch_rng)
                capture["rows"].append(row_capture)
            captures.append(capture)
        return original_step(*args, **kwargs)

    environment.step = observed_step  # type: ignore[method-assign]
    replay_error: dict[str, Any] | None = None
    try:
        try:
            replay = SafetyTraceRecorder.replay_action_tape(environment, tape)
        except Exception as exc:
            replay_error = {"kind": "replay", "type": type(exc).__name__, "message": str(exc)}
            replay = {"exact": False, "steps_replayed": 0, "steps_expected": len(tape_rows)}
    finally:
        environment.step = original_step  # type: ignore[method-assign]
    replay_after = _rng_snapshot()
    model_after = _model_digest(source_policy)
    if replay_error is not None:
        errors.append(replay_error)

    missing_rows = [
        {"identity": _identity(source_row), "status": "missing_runtime_row"}
        for source_row in target_rows
        if _identity(source_row) not in attempted
    ]
    outcomes_match: bool | None = None
    if not smoke and replay_error is None:
        try:
            outcomes_match = (
                collector._summary_outcomes(Path(str(record["action_tape"])))
                == collector._replay_outcomes(environment)
            )
            if not outcomes_match:
                errors.append({"kind": "outcome_check", "message": "source outcomes differ from replay"})
        except Exception as exc:
            outcomes_match = False
            errors.append({"kind": "outcome_check", "type": type(exc).__name__, "message": str(exc)})

    return {
        "seed": int(record["seed"]),
        "scenario": str(record["scenario"]),
        "split": str(record["split"]),
        "action_tape": str(record["action_tape"]),
        "action_tape_sha256": _sha256(tape),
        "steps_expected": len(tape_rows),
        "steps_replayed": replay.get("steps_replayed"),
        "replay_exact": bool(replay.get("exact")) and replay_error is None,
        "outcomes_match": outcomes_match,
        "rng_unchanged": _rng_equal(replay_before, replay_after),
        "model_weights_unchanged": model_before is not None and model_after is not None and model_before == model_after,
        "safety_counter_delta_nonzero_count": safety_counter_nonzero_count + sum(
            int(summary.get("safety_counter_delta_nonzero_count", 0))
            for summary in row_summaries
        ),
        "source_model_digest_before": model_before,
        "source_model_digest_after": model_after,
        "target_rows": len(target_rows),
        "rows_observed": len(labelled),
        "rows_attempted": len(attempted),
        "missing_rows": missing_rows,
        "row_summaries": row_summaries,
        "step_inputs": captures,
        "errors": errors,
    }


def _record_ok(report: Mapping[str, Any], *, smoke: bool) -> bool:
    return (
        report.get("replay_exact") is True
        and (smoke or report.get("outcomes_match") is True)
        and report.get("rng_unchanged") is True
        and report.get("model_weights_unchanged") is True
        and int(report.get("safety_counter_delta_nonzero_count", 0)) == 0
        and not report.get("missing_rows")
        and not report.get("errors")
        and int(report.get("rows_observed", 0)) == int(report.get("target_rows", -1))
    )


def _write_progress(output_dir: Path, *, mode: str, status: str, processed_rows: int,
                    processed_records: int, expected_rows: int, expected_records: int) -> None:
    _write_json_atomic(
        output_dir / "progress.json",
        {
            "stage": STAGE,
            "mode": mode,
            "status": status,
            "processed_rows": int(processed_rows),
            "processed_records": int(processed_records),
            "expected_rows": int(expected_rows),
            "expected_records": int(expected_records),
        },
    )


def _manifest_result(
    *,
    mode: str,
    status: str,
    output_dir: Path,
    source_collection_dir: Path,
    source_collection_report: Mapping[str, Any],
    formal_root: Path,
    formal_manifest: Path,
    source_checkpoint_sha256: str | None,
    records: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    runtime_q_error_summary: Mapping[str, Any] | None = None,
    error: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    split_counts = {
        "train": sum(str(row.get("split")) == "train" for row in rows),
        "validation": sum(str(row.get("split")) == "validation" for row in rows),
    }
    result: dict[str, Any] = {
        "stage": STAGE,
        "status": status,
        "mode": mode,
        "output_dir": str(output_dir),
        "records": list(records),
        "records_count": len(records),
        "records_expected": 1 if mode == "smoke" else EXPECTED_RECORDS,
        "rows_count": len(rows),
        "rows_expected": 1 if mode == "smoke" else EXPECTED_ROWS,
        "split_counts": split_counts,
        "source_collection_dir": str(source_collection_dir),
        "source_collection_manifest": source_collection_report.get("manifest"),
        "source_collection_manifest_sha256": source_collection_report.get("manifest_sha256"),
        "source_collection_rows": source_collection_report.get("rows"),
        "source_collection_rows_sha256": source_collection_report.get("rows_sha256"),
        "source_formal_root": str(formal_root),
        "source_formal_manifest": str(formal_manifest),
        "source_checkpoint_sha256": source_checkpoint_sha256 or collector.SOURCE_CHECKPOINT_SHA256,
        "runtime_labels": True,
        "training": False,
        "policy_sampling": False,
        "policy_update": False,
        "partial": not (mode == "full" and status == "COMPLETED"),
        "runtime_q_error_summary": dict(runtime_q_error_summary or {"count": 0, "values": []}),
        "PPO_ALLOWED": False,
        "P3_QUALIFIED": False,
        "P4_ALLOWED": False,
    }
    if error is not None:
        result["error"] = dict(error)
    return result


def collect(
    *,
    output_dir: Path,
    collection_dir: Path = DEFAULT_COLLECTION_DIR,
    formal_root: Path = FORMAL_ROOT,
    smoke: bool = False,
) -> dict[str, Any]:
    """Run the one-replay-per-seed runtime label repair collection."""
    output_dir = Path(output_dir)
    collection_dir = Path(collection_dir)
    formal_root = Path(formal_root)
    _require(not output_dir.exists(), f"refusing existing output directory: {output_dir}")

    source_rows, collection_report = trainer._load_collection(collection_dir)
    source_rows = [dict(row) for row in source_rows]
    if not smoke:
        _require(len(source_rows) == EXPECTED_ROWS, f"source collection expected {EXPECTED_ROWS} rows, got {len(source_rows)}")
        _require(
            {split: sum(str(row.get("split")) == split for row in source_rows) for split in EXPECTED_SPLIT_COUNTS}
            == EXPECTED_SPLIT_COUNTS,
            f"source split counts changed: { {split: sum(str(row.get('split')) == split for row in source_rows) for split in EXPECTED_SPLIT_COUNTS} }",
        )
    protocol, configs, source_records, formal_manifest = collector._load_source_inputs(formal_root)
    source_records = [dict(record) for record in source_records]
    _require(len(source_records) == EXPECTED_RECORDS, f"source records expected {EXPECTED_RECORDS}, got {len(source_records)}")
    records_by_seed = {int(record["seed"]): record for record in source_records}
    _require(len(records_by_seed) == len(source_records), "source record seeds are duplicated")
    for row in source_rows:
        _require(int(row["seed"]) in records_by_seed, f"source row seed has no tape record: {row['seed']}")

    output_dir.mkdir(parents=True, exist_ok=False)
    mode = "smoke" if smoke else "full"
    _write_progress(
        output_dir,
        mode=mode,
        status="RUNNING",
        processed_rows=0,
        processed_records=0,
        expected_rows=1 if smoke else EXPECTED_ROWS,
        expected_records=1 if smoke else EXPECTED_RECORDS,
    )
    labelled_by_identity: dict[tuple[str, int, int, int, int], dict[str, Any]] = {}
    reports: list[dict[str, Any]] = []
    processed_rows = 0
    source_checkpoint_sha256: str | None = None
    selected_rows = source_rows[:1] if smoke else source_rows
    selected_records = (
        [records_by_seed[int(selected_rows[0]["seed"])]]
        if smoke
        else list(source_records)
    )

    def persist_rows() -> None:
        ordered = [labelled_by_identity[_identity(row)] for row in selected_rows if _identity(row) in labelled_by_identity]
        _write_json_atomic(output_dir / "rows.json", ordered)

    def q_error_summary() -> dict[str, Any]:
        values = sorted(float(row["q_runtime_vs_actual_error"]) for row in labelled_by_identity.values())
        if not values:
            return {"count": 0, "values": []}
        mid = values[len(values) // 2]
        p95 = values[min(len(values) - 1, int(math.ceil(0.95 * len(values)) - 1))]
        return {
            "count": len(values),
            "min": values[0],
            "max": values[-1],
            "mean": float(sum(values) / len(values)),
            "p50": mid,
            "p95": p95,
            "values": values,
        }

    persist_rows()
    try:
        policies: dict[str, Any] = {}
        for record_index, record in enumerate(selected_records, start=1):
            record_rows = [row for row in selected_rows if int(row["seed"]) == int(record["seed"])]
            if not record_rows:
                continue
            scenario = str(record["scenario"])
            if scenario not in policies:
                policies[scenario] = collector._load_source_policy(protocol, configs[scenario])
                if isinstance(policies[scenario], Mapping):
                    source_checkpoint_sha256 = str(
                        policies[scenario].get("sha256", policies[scenario].get("source_checkpoint_sha256", ""))
                    ) or source_checkpoint_sha256
            source_tape = Path(str(record["action_tape"]))
            temporary: tempfile.TemporaryDirectory[str] | None = None
            replay_tape = source_tape
            if smoke:
                temporary = tempfile.TemporaryDirectory(prefix="p3_runtime_rank_smoke_")
                replay_tape = _prefix_tape(source_tape, int(record_rows[0]["step"]), Path(temporary.name))

            def on_labelled_row(labelled_row: dict[str, Any]) -> None:
                nonlocal processed_rows
                labelled_by_identity[_identity(labelled_row)] = labelled_row
                processed_rows = len(labelled_by_identity)
                persist_rows()
                _write_progress(
                    output_dir,
                    mode=mode,
                    status="RUNNING",
                    processed_rows=processed_rows,
                    processed_records=len(reports),
                    expected_rows=1 if smoke else EXPECTED_ROWS,
                    expected_records=1 if smoke else EXPECTED_RECORDS,
                )

            try:
                report = _collect_record(
                    record=record,
                    config=configs[scenario],
                    source_policy=policies[scenario],
                    target_rows=record_rows,
                    smoke=smoke,
                    source_tape=replay_tape,
                    on_labelled_row=on_labelled_row,
                )
            finally:
                if temporary is not None:
                    temporary.cleanup()
            reports.append(report)
            _write_json_atomic(output_dir / f"record_{int(record['seed'])}.json", report)
            persist_rows()
            _write_progress(
                output_dir,
                mode=mode,
                status="RUNNING",
                processed_rows=processed_rows,
                processed_records=len(reports),
                expected_rows=1 if smoke else EXPECTED_ROWS,
                expected_records=1 if smoke else EXPECTED_RECORDS,
            )

        status = "SMOKE_COMPLETED" if smoke else "COMPLETED"
        if len(reports) != (1 if smoke else EXPECTED_RECORDS) or not all(_record_ok(report, smoke=smoke) for report in reports):
            status = "FAILED"
        final_rows = [labelled_by_identity[_identity(row)] for row in selected_rows if _identity(row) in labelled_by_identity]
        result = _manifest_result(
            mode=mode,
            status=status,
            output_dir=output_dir,
            source_collection_dir=collection_dir,
            source_collection_report=collection_report,
            formal_root=formal_root,
            formal_manifest=Path(formal_manifest),
            source_checkpoint_sha256=source_checkpoint_sha256,
            records=reports,
            rows=final_rows,
            runtime_q_error_summary=q_error_summary(),
        )
        _write_json_atomic(output_dir / "rows.json", final_rows)
        _write_json_atomic(output_dir / "manifest.json", result)
        _write_progress(
            output_dir,
            mode=mode,
            status=status,
            processed_rows=len(final_rows),
            processed_records=len(reports),
            expected_rows=1 if smoke else EXPECTED_ROWS,
            expected_records=1 if smoke else EXPECTED_RECORDS,
        )
        return result
    except Exception as exc:
        final_rows = [labelled_by_identity[_identity(row)] for row in selected_rows if _identity(row) in labelled_by_identity]
        result = _manifest_result(
            mode=mode,
            status="FAILED",
            output_dir=output_dir,
            source_collection_dir=collection_dir,
            source_collection_report=collection_report,
            formal_root=formal_root,
            formal_manifest=Path(formal_manifest),
            source_checkpoint_sha256=source_checkpoint_sha256,
            records=reports,
            rows=final_rows,
            runtime_q_error_summary=q_error_summary(),
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        _write_json_atomic(output_dir / "rows.json", final_rows)
        _write_json_atomic(output_dir / "manifest.json", result)
        _write_progress(
            output_dir,
            mode=mode,
            status="FAILED",
            processed_rows=len(final_rows),
            processed_records=len(reports),
            expected_rows=1 if smoke else EXPECTED_ROWS,
            expected_records=1 if smoke else EXPECTED_RECORDS,
        )
        return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--collection-dir", type=Path, default=DEFAULT_COLLECTION_DIR)
    parser.add_argument("--formal-root", type=Path, default=FORMAL_ROOT)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = collect(
        output_dir=args.output_dir,
        collection_dir=args.collection_dir,
        formal_root=args.formal_root,
        smoke=bool(args.smoke),
    )
    print(json.dumps(_jsonable(result), sort_keys=True), flush=True)
    return 0 if result["status"] in {"COMPLETED", "SMOKE_COMPLETED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
