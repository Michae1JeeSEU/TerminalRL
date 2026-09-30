"""Matched full-cycle gate-value diagnostic on four registered tape states.

This diagnostic restores pre-action states from completed after-legacy action
tapes, fixes every non-target first action to that tape, and evaluates the
target WAIT plus 25 RELEASE branches under the frozen before policy.  It is an
off-policy diagnostic only: no new episode, optimizer, teacher, PPO, or
held-out input is used.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from contextlib import contextmanager
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from terminal_marl.core.safety_trace import SafetyTraceRecorder
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training import diagnose_p3_matched_gate_shift_20261001 as matched
from terminal_marl.training import evaluate_p3_runtime_rank_repair as rank_eval
from terminal_marl.training import paired_gate_diagnostic as paired
from terminal_marl.training import run_p3_followup_diagnostics as followup
from terminal_marl.training import run_p3_forced_wait_frozen_20260930 as frozen_wait
from terminal_marl.training import run_p3_trained_proposal_support_20261001 as trained_support


STAGE = "P3_TRAINED_GATE_VALUE_20261001_V1"
PPO_ALLOWED = False
SCENARIO = "dense24"
OBSERVATION_DIM = 88
PROPOSAL_COUNT = 25
SHORTLIST_K = 3
MAX_STEPS = 1800
CONTINUATION_SEEDS = (9303201, 9303202, 9303203, 9303204)
TARGET_UIDS = (3, 2, 2, 3)
TARGET_ARRIVAL_IDS = (3, 2, 2, 3)
TARGET_STEPS = (168, 202, 198, 187)
MATCHED_FULL_SHA256 = "237bc7393e0dc831204b59aaf29aa99ea55dc142ce7ab3cc9fbc4effe8c4ec3e"
NATIVE_FULL_SHA256 = "3f65446f49474452d31c3e1498c8019f04f4250cad3e8594c75107a123525919"
BEFORE_SHA256 = trained_support.BEFORE_CHECKPOINT_SHA256
AFTER_SHA256 = trained_support.AFTER_CHECKPOINT_SHA256
WARMUP_SHA256 = trained_support.WARMUP_CHECKPOINT_SHA256
# The matched four-cell groups/NPZs live under the matched diagnostic output;
# the registered action tapes and native full binding remain in the support
# run consumed by ``matched._load_arm_inputs``.
INPUT_ROOT = matched.OUTPUT_ROOT
OUTPUT_ROOT = Path(
    "/home/user/training_outputs/terminal/"
    "p3_followup_20260930/trained_gate_value_20261001_v2"
)
PROTOCOL = Path(__file__).resolve().parents[2] / "docs/evidence/v4_2/p3_trained_gate_value_protocol_20261001.md"


class TrainedGateValueError(ValueError):
    """Raised for an input, replay, model, or branch integrity mismatch."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TrainedGateValueError(message)


def _jsonable(value: Any) -> Any:
    return matched._jsonable(value)


def _canonical_jsonable(value: Any) -> Any:
    """Compare live Path-rich bindings with their JSON round-trip form."""
    return _jsonable(value)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _synced(path: str | Path) -> Path:
    return matched._synced(Path(path))


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any], *, refuse_existing: bool = False) -> None:
    if refuse_existing and path.exists():
        raise TrainedGateValueError(f"refusing existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{__import__('os').getpid()}.tmp")
    temporary.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _rng_snapshot() -> Any:
    return trained_support._rng_snapshot()


def _restore_rng(value: Any) -> None:
    trained_support._restore_rng(value)


def _rng_equal(left: Any, right: Any) -> bool:
    return bool(trained_support._rng_equal(left, right))


def _group_path(cell_id: int) -> Path:
    return _synced(INPUT_ROOT / f"cell_{int(cell_id):02d}.json")


def _load_npz_state(path: Path, cell_id: int) -> dict[str, Any]:
    required = {"world", "step", "uid", "arrival_id", "phase", "soc", "actor_mask", "retry_blocked",
                "before_gate_probs", "after_gate_probs", "proposal_q_before", "proposal_q_after", "metadata_json"}
    with np.load(path, allow_pickle=False) as data:
        _require(required <= set(data.files), f"cell {cell_id} NPZ fields are incomplete")
        world = np.asarray(data["world"])
        _require(world.ndim == 2 and world.shape[1] == OBSERVATION_DIM, f"cell {cell_id} NPZ world is not 88D")
        metadata = json.loads(str(data["metadata_json"].item()))
        _require(metadata.get("format") == "matched_gate_shift_v1", f"cell {cell_id} NPZ format changed")
        _require(metadata.get("reward_scale_applied") is False, f"cell {cell_id} NPZ reward scale changed")
        candidates: list[int] = []
        for index in range(len(data["step"])):
            phase = str(data["phase"][index])
            soc = float(data["soc"][index])
            if (bool(data["actor_mask"][index]) and phase == "ring_inner"
                    and 0.4 < soc <= 0.5 and not bool(data["retry_blocked"][index])):
                candidates.append(index)
        _require(candidates, f"cell {cell_id} has no registered selection candidate")
        first = min(candidates, key=lambda index: (int(data["step"][index]), int(index)))
        lock = {
            1: (168, 3, 3), 2: (202, 2, 2), 3: (198, 2, 2), 4: (187, 3, 3),
        }[int(cell_id)]
        observed = (int(data["step"][first]), int(data["uid"][first]), int(data["arrival_id"][first]))
        _require(observed == lock, f"cell {cell_id} registered state changed: {observed} != {lock}")
        return {
            "cell_id": int(cell_id), "npz_row": int(first), "step": observed[0],
            "uid": observed[1], "arrival_id": observed[2], "phase": str(data["phase"][first]),
            "soc": float(data["soc"][first]), "metadata": metadata,
        }


def _load_input_bundle() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    full_path = _synced(INPUT_ROOT / "full.json")
    _require(full_path.is_file(), f"matched full input missing: {full_path}")
    _require(_sha256(full_path) == MATCHED_FULL_SHA256, "matched full SHA changed")
    full = _read_json(full_path)
    _require(full.get("status") == "FULL_OK" and len(full.get("cells", [])) == 4,
             "matched full input is not the registered four-cell completion")
    native_full = _synced(trained_support.OUTPUT_DIR / "full.json")
    _require(native_full.is_file() and _sha256(native_full) == NATIVE_FULL_SHA256,
             "native proposal-support full binding changed")
    source_items = matched._load_arm_inputs()
    _require(len(source_items) == 4, "matched source arm count changed")
    cells: list[dict[str, Any]] = []
    for source in source_items:
        cell = dict(source["cell"])
        cell_id = int(cell["cell_id"])
        group_path = _group_path(cell_id)
        group = _read_json(group_path)
        _require(group.get("status") == "COMPLETED", f"matched group {cell_id} is incomplete")
        group_cell = group.get("cell")
        _require(group_cell == cell, f"matched group {cell_id} cell binding changed")
        artifacts = group.get("artifacts") if isinstance(group.get("artifacts"), Mapping) else {}
        npz_value = artifacts.get("matched_states_npz")
        npz_path = _synced(Path(str(npz_value)))
        _require(npz_path.is_file(), f"matched cell {cell_id} NPZ missing")
        _require(str(artifacts.get("matched_states_npz_sha256")) == _sha256(npz_path),
                 f"matched cell {cell_id} NPZ SHA changed")
        state = _load_npz_state(npz_path, cell_id)
        full_cell = next((item for item in full["cells"] if int(item.get("cell", {}).get("cell_id", -1)) == cell_id), None)
        _require(isinstance(full_cell, Mapping), f"matched full cell {cell_id} is missing")
        _require(full_cell.get("action_tape_sha256") == source["tape_sha256"], f"cell {cell_id} tape binding changed")
        cells.append({**source, "group_path": group_path, "group_sha256": _sha256(group_path),
                      "group": group, "npz_path": npz_path, "npz_sha256": _sha256(npz_path),
                      "selection": state})
    return {
        "matched_full_path": full_path, "matched_full_sha256": _sha256(full_path),
        "native_full_path": native_full, "native_full_sha256": _sha256(native_full),
    }, cells


def _load_policies(config: Any) -> tuple[Any, Any, dict[str, Any]]:
    caller = _rng_snapshot()
    try:
        before, before_binding = matched._policy_binding(config, trained_support.BEFORE_CHECKPOINT, BEFORE_SHA256, "before")
        after, after_binding = matched._policy_binding(config, trained_support.AFTER_CHECKPOINT, AFTER_SHA256, "after")
        freeze = matched._frozen_binding(before, after, config)
    finally:
        _restore_rng(caller)
    _require(_rng_equal(caller, _rng_snapshot()), "policy loading did not restore caller RNG")
    return before, after, {"before": before_binding, "after": after_binding, "freeze": freeze}


@contextmanager
def _arrival_keyed_continuation_scope():
    """Temporarily use the registered arrival-id keyed continuation sampler."""
    original = paired.keyed_request

    def adapter(policy: Any, env: Any) -> Any:
        keyed_seed = getattr(env, "_paired_seed", None)
        _require(keyed_seed is not None, "paired continuation seed is missing")
        payload = frozen_wait._keyed_request(policy, env, int(keyed_seed))
        return paired.KeyedRequest(payload["actions"], payload["shortlists"], dict(payload))

    paired.keyed_request = adapter
    try:
        yield
    finally:
        paired.keyed_request = original


def _build_preflight() -> dict[str, Any]:
    config, _ = trained_support._build_inputs()
    bundle, cells = _load_input_bundle()
    before, after, policies = _load_policies(config)
    del before, after
    return {
        "stage": STAGE, "phase": "preflight", "status": "PREFLIGHT_OK", "PPO_ALLOWED": False,
        "scope": {"env_created": False, "replay_calls": 0, "policy_sampling": False,
                   "optimizer_used": False, "teacher_used": False, "heldout_used": False},
        "matched_binding": bundle,
        "cells": [
            {"cell": item["cell"], "group_path": str(item["group_path"]), "group_sha256": item["group_sha256"],
             "npz_path": str(item["npz_path"]), "npz_sha256": item["npz_sha256"],
             "tape_path": str(item["tape_path"]), "tape_sha256": item["tape_sha256"],
             "selection": item["selection"]}
            for item in cells
        ],
        "source": {"before_sha256": BEFORE_SHA256, "after_sha256": AFTER_SHA256,
                   "warmup_sha256": WARMUP_SHA256, "policies": policies,
                   "continuation_seeds": list(CONTINUATION_SEEDS), "max_steps": MAX_STEPS,
                   "observation_dim": OBSERVATION_DIM, "shortlist_k": SHORTLIST_K,
                   "keying": "arrival_id|absolute_tick|stream"},
        "config_snapshot": trained_support._config_snapshot(config),
        "runner_sha256": _sha256(Path(__file__)),
        "protocol_sha256": _sha256(PROTOCOL) if PROTOCOL.is_file() else None,
        "qualification": "mechanism_state_diagnostic; not a performance cohort or qualification",
    }


def _validate_preflight(payload: Mapping[str, Any]) -> None:
    _require(payload.get("status") == "PREFLIGHT_OK" and payload.get("stage") == STAGE,
             "trained gate-value preflight is not successful")
    _require(payload.get("runner_sha256") == _sha256(Path(__file__)), "runner changed after preflight")
    _require(payload.get("protocol_sha256") == (_sha256(PROTOCOL) if PROTOCOL.is_file() else None),
             "protocol changed after preflight")
    config, _ = trained_support._build_inputs()
    _require(payload.get("config_snapshot") == trained_support._config_snapshot(config), "config changed after preflight")
    bundle, cells = _load_input_bundle()
    stored_binding = payload.get("matched_binding")
    _require(_canonical_jsonable(stored_binding) == _canonical_jsonable(bundle),
             "matched/native full binding changed")
    live = _build_preflight()
    _require(payload.get("cells") == live.get("cells"), "cell, group, NPZ, or selection binding changed")
    _require(payload.get("source") == live.get("source"), "source or continuation binding changed")
    del cells


def _phase_inputs(output: Path) -> tuple[dict[str, Any], Any, list[dict[str, Any]], Any, Any]:
    preflight = _read_json(output / "preflight.json")
    _validate_preflight(preflight)
    config, _ = trained_support._build_inputs()
    _bundle, cells = _load_input_bundle()
    before, after, _policies = _load_policies(config)
    return preflight, config, cells, before, after


def _npz_step_rows(path: Path, step: int) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        indices = [int(i) for i, value in enumerate(data["step"]) if int(value) == int(step)]
        _require(indices, f"NPZ step {step} is missing")
        return {
            "indices": indices,
            "uids": [int(data["uid"][i]) for i in indices],
            "arrival_ids": [int(data["arrival_id"][i]) for i in indices],
            "world": np.asarray(data["world"][indices], dtype=np.float32).copy(),
            "actor_mask": np.asarray(data["actor_mask"][indices], dtype=bool).copy(),
            "before_gate_probs": np.asarray(data["before_gate_probs"][indices], dtype=np.float64).copy(),
            "after_gate_probs": np.asarray(data["after_gate_probs"][indices], dtype=np.float64).copy(),
            "proposal_q_before": np.asarray(data["proposal_q_before"][indices], dtype=np.float64).copy(),
            "proposal_q_after": np.asarray(data["proposal_q_after"][indices], dtype=np.float64).copy(),
        }


def _arrivals_exhausted(env: Any) -> bool:
    arrivals = getattr(env, "arrivals", None)
    value = getattr(arrivals, "exhausted", False)
    return bool(value() if callable(value) else value)


def _capture_state(item: Mapping[str, Any], config: Any, before_policy: Any, output: Path,
                   phase: str) -> dict[str, Any]:
    selection = dict(item["selection"])
    step = int(selection["step"])
    tape_rows = list(item["tape_rows"])
    _require(step < len(tape_rows), f"cell {item['cell']['cell_id']} selected step is outside tape")
    npz_rows = _npz_step_rows(Path(item["npz_path"]), step)
    rng_before = _rng_snapshot()
    environment = TerminalParallelEnv(config)
    if not _rng_equal(rng_before, _rng_snapshot()):
        _restore_rng(rng_before)
        raise TrainedGateValueError("environment construction changed caller RNG")
    original_step = environment.step
    captured: dict[str, Any] = {}

    def observed_step(*args: Any, **kwargs: Any) -> Any:
        current = int(getattr(environment, "step_count", -1))
        if current == step and not captured:
            source_row = tape_rows[step]
            observations = environment._observations()
            uids, world = matched._world_from_observations(observations)
            _require(matched._digest({key: value for key, value in sorted(observations.items())}) == source_row.get("observation_hash"),
                     f"cell {item['cell']['cell_id']} selected observation hash changed")
            _require(uids == [str(value) for value in npz_rows["uids"]],
                     f"cell {item['cell']['cell_id']} native UID order differs from NPZ")
            _require(np.array_equal(world, npz_rows["world"]),
                     f"cell {item['cell']['cell_id']} selected 88D world differs from NPZ")
            details = paired._details_context(before_policy, environment)
            _require(details["uids"] == uids, "source actor UID order changed")
            _require(np.array_equal(np.asarray(details["world"], dtype=np.float32), world),
                     "source details world differs from replay world")
            _require(np.array_equal(np.asarray(details["actor_mask"], dtype=bool), npz_rows["actor_mask"]),
                     "source actor mask differs from matched NPZ")
            arrivals = getattr(environment, "arrivals", None)
            arrivals_generated = int(getattr(arrivals, "generated", -1))
            arrivals_exhausted = _arrivals_exhausted(environment)
            actor = getattr(environment, "active", {}).get(int(selection["uid"]))
            _require(actor is not None and int(getattr(actor, "arrival_id", -1)) == int(selection["arrival_id"]),
                     "selected UID/arrival identity changed")
            snapshot = copy.deepcopy(environment)
            paired.restore_diagnostic_snapshot(snapshot)
            actions, shortlists = followup._row_inputs(source_row)
            captured.update({
                "snapshot": snapshot, "uids": uids, "world": world.copy(), "details": details,
                "actions": actions, "shortlists": shortlists, "source_row": dict(source_row),
                "arrival_prefix": {"generated": arrivals_generated, "exhausted": arrivals_exhausted,
                                   "future_arrivals_pending": not arrivals_exhausted},
                "rng_before_step": _rng_snapshot(),
            })
        return original_step(*args, **kwargs)

    environment.step = observed_step  # type: ignore[method-assign]
    replay_rng_after = rng_before
    replay_error: BaseException | None = None
    try:
        replay = SafetyTraceRecorder.replay_action_tape(environment, item["tape_path"])
    except BaseException as error:
        replay_error = error
    finally:
        replay_rng_after = _rng_snapshot()
        environment.step = original_step  # type: ignore[method-assign]
        _restore_rng(rng_before)
    _require(_rng_equal(rng_before, replay_rng_after),
             f"cell {item['cell']['cell_id']} replay/observer changed caller RNG")
    if replay_error is not None:
        raise replay_error
    _require(replay.get("exact") is True and int(replay.get("steps_replayed", -1)) == len(tape_rows),
             f"cell {item['cell']['cell_id']} action-tape replay was not exact")
    _require(captured, f"cell {item['cell']['cell_id']} registered pre-state was not observed")
    selected_check = next((row for row in replay.get("steps", []) if int(row.get("step", -1)) == step), None)
    _require(isinstance(selected_check, Mapping) and selected_check.get("post_state_match") is True
             and selected_check.get("requested_category_match") is True,
             f"cell {item['cell']['cell_id']} selected post-state/action check failed")
    return {
        "selection": selection, "snapshot": captured["snapshot"], "uids": captured["uids"],
        "world": captured["world"], "details": captured["details"], "first_actions": captured["actions"],
        "first_shortlists": captured["shortlists"], "source_row": captured["source_row"],
        "arrival_prefix": captured["arrival_prefix"],
        "replay": {key: value for key, value in replay.items() if key != "steps"},
        "selected_replay_check": dict(selected_check), "npz_step_rows": npz_rows,
        "caller_rng_unchanged": _rng_equal(rng_before, replay_rng_after),
    }


def _first_actions_unchanged(paired_report: Mapping[str, Any], base_actions: Mapping[str, Any],
                             base_shortlists: Mapping[str, Any], target_uid: int) -> bool:
    for branch in paired_report.get("branches", []):
        first = branch.get("first_step", {})
        actions = first.get("actions", {})
        shortlists = first.get("proposal_shortlists", {})
        expected_action_keys = {str(uid) for uid in base_actions if str(uid) != str(target_uid)}
        expected_shortlist_keys = {str(uid) for uid in base_shortlists if str(uid) != str(target_uid)}
        actual_action_keys = {str(uid) for uid in actions if str(uid) != str(target_uid)}
        actual_shortlist_keys = {str(uid) for uid in shortlists if str(uid) != str(target_uid)}
        if actual_action_keys != expected_action_keys or actual_shortlist_keys != expected_shortlist_keys:
            return False
        for uid, action in base_actions.items():
            if str(uid) == str(target_uid):
                continue
            if str(uid) not in actions or not np.array_equal(np.asarray(actions[str(uid)]), np.asarray(action)):
                return False
        for uid, shortlist in base_shortlists.items():
            if str(uid) == str(target_uid):
                continue
            got = shortlists.get(str(uid), ())
            if len(got) != len(shortlist):
                return False
            if any(not np.array_equal(np.asarray(left), np.asarray(right)) for left, right in zip(got, shortlist, strict=True)):
                return False
    return True


def _branch_natural(branch: Mapping[str, Any]) -> bool:
    return bool(branch.get("terminal") is True and not branch.get("censored") and not branch.get("error")
                and not branch.get("safety_violation") and not branch.get("safety_invariant_error"))


def _energy_total(branch: Mapping[str, Any]) -> float | None:
    values: dict[str, float] = {}
    rows = branch.get("energy_outcomes", [])
    if not isinstance(rows, Sequence):
        return None
    for row in rows:
        if not isinstance(row, Mapping) or row.get("arrival_id") is None or row.get("energy_consumed") is None:
            continue
        value = float(row["energy_consumed"])
        if not math.isfinite(value):
            return None
        values[str(int(row["arrival_id"]))] = value
    return float(sum(values.values())) if values else None


def _business_branch_metrics(branch: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return complete branch business fields without filling missing values."""
    if not _branch_natural(branch):
        return None
    elapsed = branch.get("request_elapsed_integral_seconds")
    if elapsed is None or not math.isfinite(float(elapsed)):
        return None
    wait = branch.get("wait_seconds", branch.get("request_wait_integral_seconds"))
    if wait is not None and not math.isfinite(float(wait)):
        return None
    counts = {name: int((branch.get("outcome_counts") or {}).get(name, 0))
              for name in ("normal", "emergency", "failed", "diverted", "unresolved")}
    return {"outcomes": counts, "request_elapsed_integral_seconds": float(elapsed),
            "wait_seconds": None if wait is None else float(wait),
            "energy_total_consumed": _energy_total(branch)}


def _weighted_business_metrics(wait: Mapping[str, Any], releases: Sequence[Mapping[str, Any]],
                               q: np.ndarray) -> dict[str, Any] | None:
    wait_metrics = _business_branch_metrics(wait)
    release_metrics = [_business_branch_metrics(row) for row in releases]
    if wait_metrics is None or any(row is None for row in release_metrics):
        return None
    typed_release = [row for row in release_metrics if row is not None]
    outcomes = {
        name: float(np.dot(q, np.asarray([row["outcomes"][name] for row in typed_release], dtype=np.float64)))
        for name in ("normal", "emergency", "failed", "diverted", "unresolved")
    }
    elapsed = float(np.dot(q, np.asarray(
        [row["request_elapsed_integral_seconds"] for row in typed_release], dtype=np.float64)))
    energy_values = [row["energy_total_consumed"] for row in typed_release]
    energy = None if any(value is None for value in energy_values) else float(np.dot(q, np.asarray(energy_values, dtype=np.float64)))
    wait_values = [row["wait_seconds"] for row in typed_release]
    release_wait = None if any(value is None for value in wait_values) else float(np.dot(q, np.asarray(wait_values, dtype=np.float64)))
    return {"wait": wait_metrics, "release_q_weighted": {
        "outcomes": outcomes, "request_elapsed_integral_seconds": elapsed,
        "wait_seconds": release_wait, "energy_total_consumed": energy,
    }}


def _summarize_paired(paired_report: Mapping[str, Any]) -> dict[str, Any]:
    q = np.asarray(paired_report.get("proposal_q", ()), dtype=np.float64)
    _require(q.shape == (PROPOSAL_COUNT,) and np.isfinite(q).all() and float(q.sum()) > 0.0,
             "paired source proposal q is invalid")
    q = q / float(q.sum())
    waits = list(paired_report.get("wait", ()))
    releases = list(paired_report.get("releases", ()))
    seeds = [int(value) for value in paired_report.get("continuation_seeds", CONTINUATION_SEEDS)]
    _require(len(waits) == len(seeds) and len(releases) == len(seeds) * PROPOSAL_COUNT,
             "paired branch cardinality changed")
    per_seed: list[dict[str, Any]] = []
    all_branches = waits + releases
    outcome_total: Counter[str] = Counter()
    branch_rows: list[dict[str, Any]] = []
    for branch in all_branches:
        for key, value in (branch.get("outcome_counts") or {}).items():
            outcome_total[str(key)] += int(value)
        resolved_energy = _energy_total(branch)
        branch_rows.append({
            "branch": branch.get("branch"), "proposal": branch.get("proposal"),
            "terminal": branch.get("terminal"), "censored": branch.get("censored"),
            "termination_classification": branch.get("termination_classification"),
            "error": branch.get("error"), "safety_violation": branch.get("safety_violation"),
            "outcome_counts": branch.get("outcome_counts"),
            "request_elapsed_integral_seconds": branch.get("request_elapsed_integral_seconds"),
            "resolved_energy_total_consumed": resolved_energy,
            "energy_total_consumed": resolved_energy if _branch_natural(branch) else None,
        })
    for index, seed in enumerate(seeds):
        wait = waits[index]
        rows = releases[index * PROPOSAL_COUNT:(index + 1) * PROPOSAL_COUNT]
        valid = _branch_natural(wait) and all(_branch_natural(row) for row in rows)
        d_value = None
        c_value = None
        if rows and all(isinstance(row.get("first_step"), Mapping) for row in rows):
            if all(not row.get("error") for row in rows):
                approvals = [bool(row.get("first_step", {}).get("actual_immediate_approval", False)) for row in rows]
                c_value = float(np.dot(q, np.asarray(approvals, dtype=np.float64)))
        if valid:
            release = float(np.dot(q, np.asarray([float(row["undiscounted_return"]) for row in rows], dtype=np.float64)))
            d_value = release - float(wait["undiscounted_return"])
        business = _weighted_business_metrics(wait, rows, q) if valid else None
        per_seed.append({"seed": seed, "complete": bool(valid), "C": c_value, "D": d_value,
                         "wait_return": wait.get("undiscounted_return"),
                         "release_returns": [row.get("undiscounted_return") for row in rows],
                         "business_metrics": business})
    d_values = [float(row["D"]) for row in per_seed if row["D"] is not None]
    complete = bool(d_values and len(d_values) == len(per_seed)
                   and paired_report.get("terminal_complete") is True
                   and paired_report.get("external_common_prefix", {}).get("valid") is True)
    d_mean = float(np.mean(d_values)) if complete else None
    helper_d = paired_report.get("qualified_delta_Q")
    if complete and helper_d is not None:
        _require(math.isclose(d_mean, float(helper_d), rel_tol=0.0, abs_tol=1e-8),
                 "qualified D differs from paired helper full-return value")
    return {
        "proposal_q": q.tolist(), "per_seed": per_seed, "D": d_mean,
        "C": (float(np.mean([row["C"] for row in per_seed]))
              if all(row["C"] is not None for row in per_seed) else None),
        "branch_metrics": branch_rows, "alternative_branch_outcome_counts": dict(outcome_total),
        "outcome_counts_all_branches": dict(outcome_total),
        "all_branches_natural": complete, "qualified": complete,
    }


def _run_state_impl(item: Mapping[str, Any], config: Any, before_policy: Any, after_policy: Any,
                    output: Path, phase: str) -> dict[str, Any]:
    state_rng = _rng_snapshot()
    before_digest = rank_eval._model_digest(before_policy)
    after_digest = rank_eval._model_digest(after_policy)
    captured = _capture_state(item, config, before_policy, output, phase)
    snapshot = captured["snapshot"]
    target_uid = int(captured["selection"]["uid"])
    context = captured["details"]
    target_index, _ = paired._context_target(context, target_uid)
    q = np.asarray(context["proposal_probabilities"][target_index], dtype=np.float64)
    logits = np.asarray(context["proposal_logits"][target_index], dtype=np.float64)
    _require(q.shape == (PROPOSAL_COUNT,) and np.isfinite(q).all(), "source q is invalid")
    world = np.asarray(captured["world"], dtype=np.float32)
    alignment = paired.gate_update_alignment(before_policy, after_policy, world, delta_q=None)
    _require(alignment.get("proposal_probability_unchanged") is True, "before/after proposal q changed")
    _require(alignment.get("model_unchanged") is True and alignment.get("rng_unchanged") is True,
             "gate alignment changed model or caller RNG")
    _require(_rng_equal(state_rng, _rng_snapshot()),
             "capture/alignment changed caller RNG before branch evaluation")
    _require(before_digest == rank_eval._model_digest(before_policy),
             "before model digest changed before branch evaluation")
    _require(after_digest == rank_eval._model_digest(after_policy),
             "after model digest changed before branch evaluation")
    p_before = float(alignment["before_gate_release_probability"][target_index])
    p_after = float(alignment["after_gate_release_probability"][target_index])
    npz_rows = captured["npz_step_rows"]
    _require(np.allclose(np.asarray(context["proposal_probabilities"], dtype=np.float64), npz_rows["proposal_q_before"], atol=1e-7, rtol=0.0),
             "source q differs from matched NPZ")
    _require(np.allclose(np.asarray(alignment["before_gate_release_probability"]), npz_rows["before_gate_probs"][:, paired.RELEASE_GATE], atol=1e-7, rtol=0.0),
             "source gate probabilities differ from matched NPZ")
    _require(np.allclose(np.asarray(alignment["after_gate_release_probability"]), npz_rows["after_gate_probs"][:, paired.RELEASE_GATE], atol=1e-7, rtol=0.0),
             "after gate probabilities differ from matched NPZ")
    branch_started = time.perf_counter()

    def branch_callback(report: Mapping[str, Any], completed: int, total: int) -> None:
        branch_dir = output / "branches"
        branch_dir.mkdir(parents=True, exist_ok=True)
        index = int(completed) - 1
        seed_index, proposal_index = divmod(index, PROPOSAL_COUNT + 1)
        proposal = None if proposal_index == 0 else proposal_index - 1
        branch_payload = {
            "stage": STAGE, "phase": phase, "cell_id": int(item["cell"]["cell_id"]),
            "branch_index": int(completed), "continuation_seed": int(CONTINUATION_SEEDS[seed_index]),
            "proposal": proposal, "elapsed_wall_seconds": float(time.perf_counter() - branch_started),
            "report": dict(report),
        }
        _write_json(branch_dir / f"cell_{int(item['cell']['cell_id']):02d}_{int(completed):03d}.json", branch_payload,
                    refuse_existing=True)
        _write_json(
            output / "progress.json",
            {"stage": STAGE, "phase": phase, "status": "RUNNING", "cell_id": int(item["cell"]["cell_id"]),
             "branches_completed": int(completed), "branches_expected": int(total),
             "continuation_seed": int(CONTINUATION_SEEDS[seed_index]), "proposal": proposal,
             "start_step": int(captured["selection"]["step"]),
             "elapsed_wall_seconds": float(time.perf_counter() - branch_started),
            "native_policy_sampler_used": False, "keyed_continuation_sampling": True,
             "prefix_sampling": False, "keying": "arrival_id|absolute_tick|stream", "PPO_ALLOWED": False},
        )
        if report.get("error") or report.get("safety_violation") or report.get("safety_invariant_error"):
            raise TrainedGateValueError(
                f"cell {item['cell']['cell_id']} branch {completed} has technical or hard-safety error"
            )
    original_keyed_request = paired.keyed_request
    try:
        with _arrival_keyed_continuation_scope():
            paired_report = paired.paired_gate_value(
                before_policy, snapshot, captured["first_actions"], captured["first_shortlists"],
                target_uid, CONTINUATION_SEEDS, MAX_STEPS, branch_callback=branch_callback,
            )
    finally:
        _require(paired.keyed_request is original_keyed_request,
                 "continuation request hook was not restored")
    _require(paired_report.get("model_unchanged") is True and paired_report.get("env_snapshot_unchanged") is True
             and paired_report.get("rng_unchanged") is True, "paired diagnostic integrity changed")
    if any(branch.get("safety_violation") or branch.get("safety_invariant_error") or branch.get("error")
           for branch in paired_report.get("branches", [])):
        raise TrainedGateValueError(f"cell {item['cell']['cell_id']} contains a technical or hard-safety branch failure")
    summary = _summarize_paired(paired_report)
    _require(_first_actions_unchanged(paired_report, captured["first_actions"], captured["first_shortlists"], target_uid),
             "non-target first action or shortlist changed")
    d_value = summary["D"]
    delta_p = float(p_after - p_before)
    _require(before_digest == rank_eval._model_digest(before_policy), "before model digest changed")
    _require(after_digest == rank_eval._model_digest(after_policy), "after model digest changed")
    return {
        "stage": STAGE, "phase": phase, "status": "COMPLETED",
        "cell": dict(item["cell"]), "selection": captured["selection"],
        "input_binding": {"group_path": str(item["group_path"]), "group_sha256": item["group_sha256"],
                           "npz_path": str(item["npz_path"]), "npz_sha256": item["npz_sha256"],
                           "tape_path": str(item["tape_path"]), "tape_sha256": item["tape_sha256"]},
        "replay": captured["replay"], "selected_replay_check": captured["selected_replay_check"],
        "source_policy": {"model_digest": before_digest, "model_unchanged": before_digest == rank_eval._model_digest(before_policy)},
        "after_policy": {"model_digest": after_digest, "model_unchanged": after_digest == rank_eval._model_digest(after_policy)},
        "p_before": p_before, "p_after": p_after, "delta_p": delta_p,
        "alignment": {
            "linear_approximation": alignment.get("linear_approximation", [])[target_index],
            "actual_probability_delta": alignment.get("actual_probability_delta", [])[target_index],
            "exact_linear_residual": alignment.get("exact_linear_residual", [])[target_index],
            "model_unchanged": alignment.get("model_unchanged"),
            "rng_unchanged": alignment.get("rng_unchanged"),
        },
        "proposal_q_before": q.tolist(), "proposal_logits_before": logits.tolist(),
        "C": summary["C"], "D": d_value, "local_m": None if d_value is None else float(delta_p * d_value),
        "p_before_times_C": None if summary["C"] is None else float(p_before * summary["C"]),
        "p_after_times_C": None if summary["C"] is None else float(p_after * summary["C"]),
        "per_seed": summary["per_seed"], "branch_metrics": summary["branch_metrics"],
        "outcome_counts_all_branches": summary["outcome_counts_all_branches"],
        "all_branches_natural": summary["all_branches_natural"], "qualified_D": summary["qualified"],
        "external_common_prefix": paired_report.get("external_common_prefix"),
        "paired": paired_report,
        "invariants": {
            "first_actions_other_uids_fixed": True, "continuation_policy": "before_3fe7_frozen",
            "native_policy_sampler_used": False, "keyed_continuation_sampling": True,
            "prefix_sampling": False, "keying": "arrival_id|absolute_tick|stream",
            "optimizer_used": False, "teacher_used": False,
            "PPO_ALLOWED": False, "caller_rng_unchanged": captured["caller_rng_unchanged"] and paired_report.get("rng_unchanged") is True,
            "source_model_unchanged": before_digest == rank_eval._model_digest(before_policy),
            "after_model_unchanged": after_digest == rank_eval._model_digest(after_policy),
            "future_arrivals_required": bool(captured["arrival_prefix"]["future_arrivals_pending"]),
            "mechanism_state_not_population_sample": True,
        },
    }


def _run_state(item: Mapping[str, Any], config: Any, before_policy: Any, after_policy: Any,
               output: Path, phase: str) -> dict[str, Any]:
    caller_rng = _rng_snapshot()
    try:
        result = _run_state_impl(item, config, before_policy, after_policy, output, phase)
        after = _rng_snapshot()
        _require(_rng_equal(caller_rng, after), "state diagnostic changed caller RNG")
        return result
    except BaseException as error:
        after = _rng_snapshot()
        _restore_rng(caller_rng)
        if not _rng_equal(caller_rng, after):
            raise TrainedGateValueError("state diagnostic changed caller RNG on failure") from error
        raise
    finally:
        _restore_rng(caller_rng)


def _write_failure(output: Path, phase: str, error: BaseException) -> None:
    if output.exists():
        error_path = output / "error.json"
        if not error_path.exists():
            _write_json(error_path, {"stage": STAGE, "phase": phase, "status": "FAILED",
                                     "error": {"type": type(error).__name__, "message": str(error)}})
        _write_json(output / "progress.json", {"stage": STAGE, "phase": phase, "status": "FAILED",
                                                "error": {"type": type(error).__name__, "message": str(error)}})


def run_preflight(output: Path) -> dict[str, Any]:
    _require(not output.exists(), f"output directory must be fresh: {output}")
    output.mkdir(parents=True, exist_ok=False)
    try:
        payload = _build_preflight()
        _write_json(output / "preflight.json", payload, refuse_existing=True)
        _write_json(output / "progress.json", {"stage": STAGE, "phase": "preflight", "status": "PREFLIGHT_OK",
                                                "environment_created": False, "PPO_ALLOWED": False})
        return payload
    except BaseException as error:
        _write_failure(output, "preflight", error)
        raise


def run_pilot(output: Path) -> dict[str, Any]:
    try:
        preflight, config, cells, before, after = _phase_inputs(output)
        _require(not (output / "pilot.json").exists(), "pilot artifact already exists")
        _write_json(output / "progress.json", {"stage": STAGE, "phase": "pilot", "status": "RUNNING",
                                                "cell_id": 1, "branches_expected": 104})
        state = _run_state(cells[0], config, before, after, output, "pilot")
        _write_json(output / "state_01.json", state, refuse_existing=True)
        payload = {"stage": STAGE, "phase": "pilot", "status": "PILOT_OK", "PPO_ALLOWED": False,
                   "preflight_sha256": _sha256(output / "preflight.json"),
                   "state_sha256": _sha256(output / "state_01.json"), "state": state,
                   "scope": {"branch_count": 104, "mechanism_states": 1,
                             "no_new_performance_episode": True}}
        _write_json(output / "pilot.json", payload, refuse_existing=True)
        _write_json(output / "progress.json", {"stage": STAGE, "phase": "pilot", "status": "PILOT_OK",
                                                "cell_id": 1, "branches_completed": 104, "branches_expected": 104})
        return payload
    except BaseException as error:
        _write_failure(output, "pilot", error)
        raise


def _validate_pilot(output: Path, preflight: Mapping[str, Any]) -> dict[str, Any]:
    pilot = _read_json(output / "pilot.json")
    _require(pilot.get("status") == "PILOT_OK", "pilot is not reusable")
    _require(pilot.get("preflight_sha256") == _sha256(output / "preflight.json"), "pilot preflight changed")
    state_path = output / "state_01.json"
    _require(state_path.is_file() and pilot.get("state_sha256") == _sha256(state_path), "pilot state changed")
    _require(pilot.get("state", {}).get("cell", {}).get("cell_id") == 1, "pilot state identity changed")
    _require(preflight.get("status") == "PREFLIGHT_OK", "preflight changed")
    return pilot


def run_full(output: Path) -> dict[str, Any]:
    try:
        preflight, config, cells, before, after = _phase_inputs(output)
        pilot = _validate_pilot(output, preflight)
        _require(not (output / "full.json").exists(), "full artifact already exists")
        states = [dict(pilot["state"])]
        for item in cells[1:]:
            cell_id = int(item["cell"]["cell_id"])
            _write_json(output / "progress.json", {"stage": STAGE, "phase": "full", "status": "RUNNING",
                                                    "cell_id": cell_id, "branches_expected": 104})
            state = _run_state(item, config, before, after, output, "full")
            _write_json(output / f"state_{cell_id:02d}.json", state, refuse_existing=True)
            states.append(state)
        payload = {"stage": STAGE, "phase": "full", "status": "FULL_OK", "PPO_ALLOWED": False,
                   "pilot_reused": True, "state_count": len(states), "states": states,
                   "scope": {"branch_count": len(states) * 104, "mechanism_states": 4,
                             "no_new_performance_episode": True, "population_estimate": False},
                   "qualification": "mechanism_state_diagnostic; no performance or PPO qualification"}
        _write_json(output / "full.json", payload, refuse_existing=True)
        _write_json(output / "progress.json", {"stage": STAGE, "phase": "full", "status": "FULL_OK",
                                                "state_count": len(states), "branches_completed": len(states) * 104})
        return payload
    except BaseException as error:
        _write_failure(output, "full", error)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "pilot", "full"), required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.phase == "preflight":
            result = run_preflight(args.output_dir)
        elif args.phase == "pilot":
            result = run_pilot(args.output_dir)
        else:
            result = run_full(args.output_dir)
    except Exception as error:
        print(json.dumps({"stage": STAGE, "status": "FAILED", "error": str(error)}, sort_keys=True), flush=True)
        return 2
    print(json.dumps({"stage": STAGE, "status": result.get("status"), "PPO_ALLOWED": False}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
