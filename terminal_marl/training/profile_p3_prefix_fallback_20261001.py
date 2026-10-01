"""Passively profile the frozen P3 prefix-fallback tapes.

This module deliberately consumes existing ActionTape and sidecar artifacts.
It does not call the student rollout, update model parameters, or execute an
actor probe's returned actions.  The only live transitions are exact replays
of the named immutable tapes.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MethodType
from typing import Any, Iterator

import numpy as np

from terminal_marl.core.safety_trace import SafetyTraceRecorder
from terminal_marl.training import evaluate_p3_runtime_rank_repair as rank_eval
from terminal_marl.training import run_p3_before_prefix_fallback_20261001 as before_prefix
from terminal_marl.training import run_p3_forced_wait_frozen_20260930 as frozen
from terminal_marl.training import run_p3_prefix_fallback_20261001 as prefix
from terminal_marl.training import run_p3_trained_proposal_support_20261001 as support
from terminal_marl.training import student_rollout_v4_2 as student
from terminal_marl.training import terminal_lifecycle_metrics as lifecycle
from terminal_marl.training.p3_prefix_fallback_control import prefix_fallback_controller


STAGE = "P3_PREFIX_FALLBACK_PROFILE_20261001_V1"
PPO_ALLOWED = False
HELDOUT_USED = False
TEACHER_USED = False
OPTIMIZER_USED = False
NO_TRAINING = True
MAX_STEPS = int(prefix.MAX_STEPS)
OUTPUT_DIR = Path(
    "/home/user/training_outputs/terminal/"
    "p3_followup_20260930/prefix_fallback_profile_20261001"
)
PROTOCOL = Path(__file__).resolve().parents[2] / (
    "docs/evidence/v4_2/p3_prefix_fallback_profile_protocol_20261001.md"
)

NATIVE_FULL_SHA256 = "3f65446f49474452d31c3e1498c8019f04f4250cad3e8594c75107a123525919"
BEFORE_PREFIX_FULL_SHA256 = "b4ee3a191c26911edf94fd24e1da4077bd2a0962e10cd5507174b52a8af73028"
AFTER_PREFIX_FULL_SHA256 = "020312813fc2c90a445bd3fe30e94e6739797151fe46efb65f760809a9361c50"

NATIVE_ROOT = prefix.BASELINE_ROOT
BEFORE_PREFIX_ROOT = before_prefix.OUTPUT_DIR
AFTER_PREFIX_ROOT = before_prefix.AFTER_PREFIX_BASELINE_ROOT

ARMS = (
    "before_legacy",
    "after_legacy",
    "before_prefix_fallback",
    "after_prefix_fallback",
)
PREFIX_ARMS = {"before_prefix_fallback", "after_prefix_fallback"}
ARM_SPECS: dict[str, dict[str, Any]] = {
    "before_legacy": {
        "root": NATIVE_ROOT,
        "full_sha256": NATIVE_FULL_SHA256,
        "checkpoint": support.BEFORE_CHECKPOINT,
        "checkpoint_sha256": support.BEFORE_CHECKPOINT_SHA256,
        "controller": False,
        "source_arm": "before_legacy",
    },
    "after_legacy": {
        "root": NATIVE_ROOT,
        "full_sha256": NATIVE_FULL_SHA256,
        "checkpoint": support.AFTER_CHECKPOINT,
        "checkpoint_sha256": support.AFTER_CHECKPOINT_SHA256,
        "controller": False,
        "source_arm": "after_legacy",
    },
    "before_prefix_fallback": {
        "root": BEFORE_PREFIX_ROOT,
        "full_sha256": BEFORE_PREFIX_FULL_SHA256,
        "checkpoint": support.BEFORE_CHECKPOINT,
        "checkpoint_sha256": support.BEFORE_CHECKPOINT_SHA256,
        "controller": True,
        "source_arm": "before_prefix_fallback",
    },
    "after_prefix_fallback": {
        "root": AFTER_PREFIX_ROOT,
        "full_sha256": AFTER_PREFIX_FULL_SHA256,
        "checkpoint": support.AFTER_CHECKPOINT,
        "checkpoint_sha256": support.AFTER_CHECKPOINT_SHA256,
        "controller": True,
        "source_arm": "after_prefix_fallback",
    },
}

DEPENDENCY_FILES = {
    **dict(prefix.DEPENDENCY_FILES),
    "before_prefix_wrapper": Path(before_prefix.__file__).resolve(),
    "profile_runner": Path(__file__).resolve(),
}


class PrefixFallbackProfileError(ValueError):
    """Raised when a frozen profile contract is not satisfied."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PrefixFallbackProfileError(message)


def _jsonable(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    return value


def _read_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    _require(target.is_file(), f"JSON input missing: {target}")
    value = json.loads(target.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object required: {target}")
    return dict(value)


def _write_json(path: Path, value: Mapping[str, Any], *, refuse_existing: bool = False) -> None:
    if refuse_existing and path.exists():
        raise PrefixFallbackProfileError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sync_path(path: str | Path) -> Path:
    return support._sync_path(path)


def _rng_snapshot() -> Any:
    return support._rng_snapshot()


def _restore_rng(value: Any) -> None:
    support._restore_rng(value)


def _rng_equal(left: Any, right: Any) -> bool:
    return support._rng_equal(left, right)


def _runtime_dependency_shas() -> dict[str, str]:
    values: dict[str, str] = {}
    for name, path in DEPENDENCY_FILES.items():
        resolved = _sync_path(path)
        _require(resolved.is_file(), f"runtime dependency missing: {path}")
        values[name] = _sha256(resolved)
    return values


def _build_inputs() -> tuple[Any, dict[str, Any]]:
    return prefix._build_inputs()


def _config_snapshot(config: Any) -> dict[str, Any]:
    return prefix._config_snapshot(config)


def _checkpoint(path: Path, expected_sha256: str) -> Path:
    resolved = _sync_path(path)
    _require(resolved.is_file(), f"checkpoint missing: {path}")
    actual = _sha256(resolved)
    _require(actual.lower() == expected_sha256.lower(), f"checkpoint SHA mismatch: {actual}")
    return resolved


def _cell_by_id(cell_id: int) -> dict[str, Any]:
    for cell in prefix.CELL_SPECS:
        if int(cell["cell_id"]) == int(cell_id):
            return dict(cell)
    raise PrefixFallbackProfileError(f"unknown cell id: {cell_id}")


def _sidecar_path(spec: Mapping[str, Any], cell_id: int, arm: str) -> Path:
    # Resolve the concrete file after joining the server path.  The shared
    # Windows mirror resolver maps files, not directory roots.
    return _sync_path(Path(spec["root"]) / "sidecars" / f"cell_{int(cell_id):02d}_{arm}_orders.json")


def _arm_path(spec: Mapping[str, Any], cell_id: int, arm: str) -> Path:
    return _sync_path(Path(spec["root"]) / "arms" / f"cell_{int(cell_id):02d}_{arm}.json")


def _full_path(spec: Mapping[str, Any]) -> Path:
    return _sync_path(Path(spec["root"]) / "full.json")


def _load_inputs() -> tuple[dict[str, Any], dict[tuple[int, str], dict[str, Any]]]:
    """Validate the three immutable full summaries and all 16 named arms."""
    groups: dict[str, dict[str, Any]] = {
        "native": {"root": NATIVE_ROOT, "sha256": NATIVE_FULL_SHA256},
        "before_prefix": {"root": BEFORE_PREFIX_ROOT, "sha256": BEFORE_PREFIX_FULL_SHA256},
        "after_prefix": {"root": AFTER_PREFIX_ROOT, "sha256": AFTER_PREFIX_FULL_SHA256},
    }
    full_records: dict[str, Any] = {}
    for name, group in groups.items():
        full_path = _full_path(group)
        _require(full_path.is_file(), f"immutable full summary missing: {full_path}")
        actual = _sha256(full_path)
        _require(actual.lower() == str(group["sha256"]).lower(), f"immutable full SHA changed: {actual}")
        full = _read_json(full_path)
        _require(full.get("status") == "FULL_OK", f"immutable full is not FULL_OK: {name}")
        _require(int(full.get("pair_count", -1)) == len(prefix.CELL_SPECS), f"immutable full cell count changed: {name}")
        full_records[name] = {
            "path": str(full_path),
            "sha256": actual,
            "status": full.get("status"),
            "pair_count": int(full.get("pair_count", -1)),
        }

    arms: dict[tuple[int, str], dict[str, Any]] = {}
    manifest_rows: list[dict[str, Any]] = []
    group_for_arm = {
        "before_legacy": "native", "after_legacy": "native",
        "before_prefix_fallback": "before_prefix", "after_prefix_fallback": "after_prefix",
    }
    for cell in prefix.CELL_SPECS:
        cell_id = int(cell["cell_id"])
        for arm in ARMS:
            spec = ARM_SPECS[arm]
            arm_path = _arm_path(spec, cell_id, arm)
            arm_record = _read_json(arm_path)
            _require(arm_record.get("cell") == dict(cell), f"arm cell identity changed: {arm}/{cell_id}")
            _require(int(arm_record.get("seed", -1)) == int(cell["env_seed"]), f"arm environment seed changed: {arm}/{cell_id}")
            _require(int(arm_record.get("policy_seed", -1)) == int(cell["policy_seed"]), f"arm policy seed changed: {arm}/{cell_id}")
            tape_value = arm_record.get("action_tape")
            _require(isinstance(tape_value, str) and tape_value, f"arm action tape missing: {arm}/{cell_id}")
            tape_path = _sync_path(tape_value)
            _require(tape_path.is_file(), f"arm action tape missing: {tape_path}")
            sidecar = None
            sidecar_sha = None
            if arm in PREFIX_ARMS:
                sidecar_path = _sidecar_path(spec, cell_id, arm)
                _require(sidecar_path.is_file(), f"prefix sidecar missing: {sidecar_path}")
                sidecar = _read_json(sidecar_path)
                _require(sidecar.get("cell", {}).get("cell_id") == cell_id, f"sidecar cell changed: {arm}/{cell_id}")
                _require(sidecar.get("arm") == arm, f"sidecar arm changed: {arm}/{cell_id}")
                _require(isinstance(sidecar.get("registrations"), Mapping), f"sidecar registrations missing: {arm}/{cell_id}")
                for registration_key, registration in sidecar["registrations"].items():
                    _require(isinstance(registration, Mapping), f"sidecar registration is not an object: {arm}/{cell_id}/{registration_key}")
                    _require(isinstance(registration.get("order"), Sequence) and not isinstance(registration.get("order"), (str, bytes)),
                             f"sidecar order missing: {arm}/{cell_id}/{registration_key}")
                    _require(len(registration["order"]) == prefix.CONTROLLER_MAX_PROPOSALS,
                             f"sidecar order width changed: {arm}/{cell_id}/{registration_key}")
                    _require(set(int(value) for value in registration["order"]) == set(range(prefix.CONTROLLER_MAX_PROPOSALS)),
                             f"sidecar order is not a complete catalogue: {arm}/{cell_id}/{registration_key}")
                    _require(math.isfinite(float(registration.get("env_now"))),
                             f"sidecar env_now missing: {arm}/{cell_id}/{registration_key}")
                sidecar_sha = _sha256(sidecar_path)
            row = {
                "arm": arm,
                "cell": dict(cell),
                "source_group": group_for_arm[arm],
                "status": arm_record.get("status"),
                "arm_path": str(arm_path),
                "arm_sha256": _sha256(arm_path),
                "action_tape": str(tape_path),
                "action_tape_sha256": _sha256(tape_path),
                "sidecar": None if sidecar is None else str(_sidecar_path(spec, cell_id, arm)),
                "sidecar_sha256": sidecar_sha,
                "checkpoint": str(spec["checkpoint"]),
                "checkpoint_sha256": str(spec["checkpoint_sha256"]),
            }
            arms[(cell_id, arm)] = {"manifest": row, "record": arm_record, "sidecar_payload": sidecar}
            manifest_rows.append(row)
    return {"fulls": full_records, "arms": manifest_rows}, arms


def _build_preflight() -> dict[str, Any]:
    config, _ = _build_inputs()
    _require(PROTOCOL.is_file(), f"profile protocol missing: {PROTOCOL}")
    inputs, _arms = _load_inputs()
    return {
        "stage": STAGE,
        "phase": "preflight",
        "status": "PREFLIGHT_OK",
        "PPO_ALLOWED": PPO_ALLOWED,
        "heldout_used": HELDOUT_USED,
        "teacher_used": TEACHER_USED,
        "optimizer_used": OPTIMIZER_USED,
        "no_training": NO_TRAINING,
        "config_snapshot": _config_snapshot(config),
        "schema": {
            "observation": prefix.OBSERVATION_SCHEMA,
            "action": prefix.ACTION_SCHEMA,
            "observation_dim": 88,
        },
        "cells": [dict(cell) for cell in prefix.CELL_SPECS],
        "arms": list(ARMS),
        "requests_per_cell": 24,
        "runtime": {
            "replay": "SafetyTraceRecorder.replay_action_tape",
            "lifecycle": "terminal_lifecycle_metrics.lifecycle helpers on one replay env",
            "actor_probe": "student._student_request_details; outputs discarded; RNG restored",
            "max_steps": MAX_STEPS,
            "timing_clock": "time.perf_counter_ns",
            "threads": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        },
        "inputs": inputs,
        "protocol_sha256": _sha256(PROTOCOL),
        "dependency_shas": _runtime_dependency_shas(),
        "runner_sha256": _sha256(__file__),
        "qualification": "passive_measurement_only; no_training; no_heldout",
    }


def _validate_preflight(payload: Mapping[str, Any]) -> None:
    _require(payload.get("stage") == STAGE and payload.get("status") == "PREFLIGHT_OK", "profile preflight is not successful")
    for key, expected in (("PPO_ALLOWED", False), ("heldout_used", False), ("teacher_used", False), ("optimizer_used", False), ("no_training", True)):
        _require(payload.get(key) is expected, f"profile flag changed: {key}")
    _require(payload.get("runner_sha256") == _sha256(__file__), "profile runner changed after preflight")
    _require(payload.get("protocol_sha256") == _sha256(PROTOCOL), "profile protocol changed after preflight")
    _require(payload.get("dependency_shas") == _runtime_dependency_shas(), "profile dependency changed after preflight")
    config, _ = _build_inputs()
    _require(payload.get("config_snapshot") == _config_snapshot(config), "profile config changed after preflight")
    _require(payload.get("cells") == [dict(cell) for cell in prefix.CELL_SPECS], "profile cell grid changed after preflight")
    _require(payload.get("arms") == list(ARMS), "profile arm grid changed after preflight")
    inputs, _ = _load_inputs()
    _require(payload.get("inputs") == inputs, "immutable profile inputs changed after preflight")


def _phase_inputs(output: Path) -> tuple[dict[str, Any], Any, dict[tuple[int, str], dict[str, Any]]]:
    payload = _read_json(output / "preflight.json")
    _validate_preflight(payload)
    config, _ = _build_inputs()
    _inputs, arms = _load_inputs()
    return payload, config, arms


def _timing_summary(values_ns: Sequence[int]) -> dict[str, Any]:
    if not values_ns:
        return {"count": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None, "max_ms": None}
    values = np.asarray(values_ns, dtype=np.float64) / 1_000_000.0
    return {
        "count": int(values.size),
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "max_ms": float(values.max()),
    }


def _set_instance_method(obj: Any, name: str, function: Any) -> tuple[bool, Any]:
    had = name in getattr(obj, "__dict__", {})
    previous = getattr(obj, "__dict__", {}).get(name)
    setattr(obj, name, MethodType(function, obj))
    return had, previous


def _restore_instance_method(obj: Any, name: str, had: bool, previous: Any) -> None:
    if had:
        setattr(obj, name, previous)
    else:
        try:
            delattr(obj, name)
        except AttributeError:
            pass


class _PassiveObserver:
    def __init__(self, env: Any, policy: Any, arm: str, progress_path: Path | None = None, cell_id: int | None = None) -> None:
        self.env = env
        self.policy = policy
        self.arm = arm
        self.progress_path = progress_path
        self.cell_id = cell_id
        self.release_calls = 0
        self.candidate_plan_calls = 0
        self.landing_zone_pair_checks = 0
        self.suffix_plan_pairs = 0
        self.suffix_proposal_checks = 0
        self.approvals_rank_gt3 = 0
        self.zero_check_calls = 0
        self.release_records: list[dict[str, Any]] = []
        self._current_release: dict[str, Any] | None = None
        self.actor_request_ns: list[int] = []
        self.private_release_ns: list[int] = []
        self.controller_release_ns: list[int] = []
        self.candidate_plan_ns: list[int] = []
        self.env_step_ns: list[int] = []
        self.observer_overhead_ns: list[int] = []
        self.actor_forward_counts: list[int] = []
        self.actor_observation_counts: list[int] = []
        # ``SafetyTraceRecorder.replay_action_tape`` validates action/state
        # fields, but it does not validate rewards.  Keep the native step
        # return and reward component stream so the profile can make that
        # comparison explicitly.
        self.step_reward_totals: list[float] = []
        self.step_reward_components: dict[str, float] = {}
        self.step_reward_component_rows = 0
        self.step_infos: list[dict[str, Any]] = []
        self.rng_restored = True
        self.probe_errors: list[dict[str, Any]] = []
        self.sidecar_order_lookups = 0

    @staticmethod
    def _rank(candidate: Any) -> int | None:
        kind = str(getattr(candidate, "kind", ""))
        marker = "runtime_shortlist_rank_"
        if not kind.startswith(marker):
            return None
        try:
            return int(kind[len(marker):])
        except ValueError:
            return None

    def _probe_actor(self, observations: Mapping[str, Any]) -> None:
        active_count = len(observations)
        self.actor_observation_counts.append(active_count)
        if not observations:
            self.actor_forward_counts.append(0)
            return
        snapshot = _rng_snapshot()
        calls = 0
        original_distributions = getattr(self.policy.model, "distributions")

        def counted_distributions(*args: Any, **kwargs: Any):
            nonlocal calls
            calls += 1
            return original_distributions(*args, **kwargs)

        had = "distributions" in getattr(self.policy.model, "__dict__", {})
        previous = getattr(self.policy.model, "__dict__", {}).get("distributions")
        setattr(self.policy.model, "distributions", counted_distributions)
        started = time.perf_counter_ns()
        error: BaseException | None = None
        try:
            student._student_request_details(
                self.policy, self.env, dict(observations), deterministic=False, shortlist_k=3,
            )
        except BaseException as exc:  # retained in arm output; phase caller decides hard-stop
            error = exc
            self.probe_errors.append({"type": type(exc).__name__, "message": str(exc)})
        finally:
            elapsed = time.perf_counter_ns() - started
            if had:
                setattr(self.policy.model, "distributions", previous)
            else:
                try:
                    delattr(self.policy.model, "distributions")
                except AttributeError:
                    pass
            self._restore_probe_rng(snapshot)
            after_restore = _rng_snapshot()
        self.actor_request_ns.append(int(elapsed))
        self.actor_forward_counts.append(int(calls))
        if error is not None:
            raise error
        if calls != 1:
            self.probe_errors.append({"type": "ForwardCountError", "message": f"nonempty actor probe used {calls} distribution forwards"})
            raise PrefixFallbackProfileError(f"nonempty actor probe used {calls} distribution forwards")
        if not _rng_equal(snapshot, after_restore):
            self.rng_restored = False
            raise PrefixFallbackProfileError("actor probe RNG was not restored")

    @staticmethod
    def _restore_probe_rng(snapshot: Any) -> None:
        _restore_rng(snapshot)

    def wrap_step(self, original_step: Any):
        observer = self

        def stepped(_self: Any, *args: Any, **kwargs: Any):
            observations = _self._observations()
            observer._probe_actor(observations)
            started = time.perf_counter_ns()
            result = original_step(*args, **kwargs)
            observer.env_step_ns.append(int(time.perf_counter_ns() - started))
            # TerminalParallelEnv returns (observations, rewards, terminated,
            # truncated, info).  Preserve the returned team reward and the
            # per-UAV component mapping used by student_rollout.
            if isinstance(result, tuple) and len(result) >= 5:
                rewards = result[1]
                if isinstance(rewards, Mapping):
                    observer.step_reward_totals.append(float(sum(float(value) for value in rewards.values())))
                else:
                    observer.step_reward_totals.append(float(rewards))
                info = result[4]
                if isinstance(info, Mapping):
                    observer.step_infos.append(dict(info))
                    components = info.get("reward_components")
                    if isinstance(components, Mapping):
                        observer.step_reward_component_rows += len(components)
                        for values in components.values():
                            if not isinstance(values, Mapping):
                                continue
                            for name, value in values.items():
                                if str(name) == "total":
                                    continue
                                try:
                                    observer.step_reward_components[str(name)] = (
                                        observer.step_reward_components.get(str(name), 0.0) + float(value)
                                    )
                                except (TypeError, ValueError):
                                    continue
            step = int(getattr(_self, "step_count", 0) or 0)
            if observer.progress_path is not None and (step == 1 or step % 50 == 0):
                captured = getattr(_self, "_captured_request_rows", None)
                _write_json(observer.progress_path, {
                    "stage": STAGE,
                    "phase": "arm",
                    "status": "RUNNING",
                    "arm": observer.arm,
                    "cell_id": observer.cell_id,
                    "absolute_step": step,
                    "env_now": float(getattr(_self, "now", float("nan"))),
                    "active_observation_count": len(observations),
                    "generated_request_rows": len(captured) if isinstance(captured, list) else None,
                    "native_forward_count": int(sum(observer.actor_forward_counts)),
                    "release_calls": int(observer.release_calls),
                })
            return result

        return stepped

    def wrap_release(self, original_release: Any):
        observer = self

        def released(_self: Any, uav: Any, intent: Any, shortlist: Any):
            before = int(getattr(_self, "telemetry", {}).get("runtime_shortlist_candidates_checked", 0) or 0)
            row = {
                "uid": int(getattr(uav, "uid", -1)),
                "step": int(getattr(_self, "step_count", -1)),
                "env_now": float(getattr(_self, "now", float("nan"))),
                "shortlist_size": len(shortlist),
                "candidate_reasons": defaultdict(list),
            }
            observer.release_calls += 1
            observer._current_release = row
            started = time.perf_counter_ns()
            error: BaseException | None = None
            decision = None
            try:
                decision = original_release(uav, intent, shortlist)
                return decision
            except BaseException as exc:
                error = exc
                raise
            finally:
                elapsed = int(time.perf_counter_ns() - started)
                observer.controller_release_ns.append(elapsed) if observer.arm in PREFIX_ARMS else observer.private_release_ns.append(elapsed)
                after = int(getattr(_self, "telemetry", {}).get("runtime_shortlist_candidates_checked", 0) or 0)
                telemetry_row = getattr(_self, "last_release_telemetry", {}).get(int(getattr(uav, "uid", -1)), {})
                row.update({
                    "decision": decision,
                    "error": None if error is None else {"type": type(error).__name__, "message": str(error)},
                    "runtime_candidates_checked_delta": after - before,
                    "runtime_candidates_checked_row": telemetry_row.get("runtime_shortlist_candidates_checked") if isinstance(telemetry_row, Mapping) else None,
                "runtime_shortlist_rank": telemetry_row.get("runtime_shortlist_rank") if isinstance(telemetry_row, Mapping) else None,
                    "decision_history": list(telemetry_row.get("decision_history", [])) if isinstance(telemetry_row, Mapping) else [],
                    "candidate_reasons": {str(key): list(value) for key, value in row["candidate_reasons"].items()},
                    "candidate_plan_calls": sum(len(value) for value in row["candidate_reasons"].values()),
                })
                if not row["candidate_plan_calls"]:
                    observer.zero_check_calls += 1
                rank = row.get("runtime_shortlist_rank")
                decision_value = getattr(row.get("decision"), "value", row.get("decision"))
                if isinstance(rank, int) and rank > 3 and str(decision_value).lower() == "approved" and int(row.get("runtime_candidates_checked_delta", 0) or 0) > 0:
                    observer.approvals_rank_gt3 += 1
                # The suffix proposal count is the number of candidates
                # actually checked beyond the native K=3 prefix.  It is a
                # per-release telemetry delta, independent of the final rank
                # (which may be stale on an early return).
                if observer.arm in PREFIX_ARMS:
                    observer.suffix_proposal_checks += max(
                        int(row.get("runtime_candidates_checked_delta", 0) or 0) - 3,
                        0,
                    )
                observer.release_records.append(row)
                observer._current_release = None

        return released

    def wrap_native_release(self, original_release: Any):
        observer = self

        def native_release(*args: Any, **kwargs: Any):
            started = time.perf_counter_ns()
            try:
                return original_release(*args, **kwargs)
            finally:
                observer.private_release_ns.append(int(time.perf_counter_ns() - started))

        return native_release

    def wrap_candidate_plan(self, original_plan: Any):
        observer = self

        def planned(_self: Any, *args: Any, **kwargs: Any):
            candidate = args[1] if len(args) > 1 else kwargs.get("candidate")
            lz_id = args[4] if len(args) > 4 else kwargs.get("lz_id")
            rank = observer._rank(candidate)
            started = time.perf_counter_ns()
            result = original_plan(*args, **kwargs)
            observer.candidate_plan_ns.append(int(time.perf_counter_ns() - started))
            observer.candidate_plan_calls += 1
            if lz_id is not None:
                observer.landing_zone_pair_checks += 1
            if rank is not None and rank > 3:
                observer.suffix_plan_pairs += 1
            reason = result[2] if isinstance(result, tuple) and len(result) >= 3 else None
            current = observer._current_release
            if current is not None and rank is not None:
                current["candidate_reasons"].setdefault(str(rank), []).append(reason)
                current.setdefault("candidate_lz_ids", {}).setdefault(str(rank), []).append(lz_id)
            return result

        return planned


def _lifecycle_from_env(config: Any, env: Any, captured: list[dict[str, Any]], first_seen: Mapping[int, Mapping[str, Any]], replay: Mapping[str, Any]) -> dict[str, Any]:
    rows = lifecycle._merge_terminal_rows(captured, lifecycle._completed_rows(env, first_seen))
    ledger = getattr(env, "ledger", None)
    expected_ids = set(range(int(config.total_requested_uavs)))
    requested_ids = {int(value) for value in getattr(ledger, "requested", set())}
    missing_ids = sorted(expected_ids - {int(row["arrival_id"]) for row in rows})
    unexpected_ids = sorted({int(row["arrival_id"]) for row in rows} - expected_ids)
    ledger_values = {int(key): str(lifecycle._enum_value(value)) for key, value in getattr(ledger, "outcomes", {}).items()}
    row_values = {int(row["arrival_id"]): str(row.get("outcome")) for row in rows}
    _require(not unexpected_ids, f"replay has unexpected arrival IDs: {unexpected_ids}")
    _require(not (set(ledger_values) - set(row_values)), "replay ledger has rows missing from lifecycle")
    _require(not (set(row_values) - set(ledger_values)), "replay lifecycle has rows missing from ledger")
    _require(all(row_values[key] == value for key, value in ledger_values.items()), "replay ledger outcome mismatch")
    summary = lifecycle.summarize_terminal_rows(
        rows, low_soc_threshold=float(getattr(config, "soc_warning", 0.35)),
        requested_count=int(config.total_requested_uavs), missing_arrival_ids=missing_ids, enforce_unique=True,
    )
    summary.update({
        "replay_exact": replay.get("exact") is True,
        "steps_expected": replay.get("steps_expected"),
        "steps_replayed": replay.get("steps_replayed"),
        "requested_arrival_ids": sorted(expected_ids),
        "observed_arrival_ids": sorted(row_values),
        "unexpected_arrival_ids": unexpected_ids,
        "ledger_requested_ids": sorted(requested_ids),
        "ledger_outcomes_without_rows": sorted(set(ledger_values) - set(row_values)),
        "ledger_unresolved_requested_ids": sorted(requested_ids - set(ledger_values)),
        "rows_without_ledger_outcome": sorted(set(row_values) - set(ledger_values)),
        "all_requests_observed": not missing_ids,
        "ledger_alignment": not missing_ids and not unexpected_ids and requested_ids == expected_ids,
    })
    return summary


def _compare_metrics(original: Mapping[str, Any], observed: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "generated", "admitted", "successful", "resolved", "normal", "emergency",
        "failed", "diverted", "unresolved", "censored", "complete_scene_metrics_known",
        "hard_safety_clear", "wait_seconds", "request_elapsed_total_seconds", "energy_total_consumed",
    )
    values: dict[str, Any] = {}
    equal = True
    for key in keys:
        left = original.get(key)
        right = observed.get(key)
        same = (left == right) if not isinstance(left, (float, int)) or not isinstance(right, (float, int)) else (
            left is None and right is None or left is not None and right is not None and math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-8)
        )
        values[key] = {"original": left, "observed": right, "match": bool(same)}
        equal = equal and bool(same)
    return {"all_match": equal, "fields": values}


def _compare_rewards(
    arm_record: Mapping[str, Any],
    observer: _PassiveObserver | None,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare the actual env.step reward stream with the saved arm summary.

    ActionTape replay does not include reward validation.  This comparison is
    therefore kept separate from ``replay_exact`` and leaves unavailable
    values as ``None``.
    """
    summary = arm_record.get("summary") if isinstance(arm_record.get("summary"), Mapping) else {}
    original_metrics = arm_record.get("metrics") if isinstance(arm_record.get("metrics"), Mapping) else {}
    observed_total = None if observer is None or not observer.step_reward_totals else float(sum(observer.step_reward_totals))
    observed_components = None if observer is None else {
        str(key): float(value) for key, value in observer.step_reward_components.items()
    }
    original_total = summary.get("reward_total")
    if original_total is None:
        original_total = original_metrics.get("reward_total_observed", original_metrics.get("reward_total"))
    original_components = summary.get("reward_component_totals")
    if not isinstance(original_components, Mapping):
        original_components = None
    fields: dict[str, Any] = {}

    def compare(left: Any, right: Any) -> bool | None:
        if left is None or right is None:
            return None
        try:
            return bool(math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-8))
        except (TypeError, ValueError):
            return bool(left == right)

    total_match = compare(original_total, observed_total)
    fields["reward_total"] = {
        "original": original_total,
        "observed": observed_total,
        "match": total_match,
    }
    component_match: bool | None = None
    if isinstance(original_components, Mapping) and observed_components is not None:
        names = sorted({str(key) for key in original_components} | set(observed_components))
        component_fields = {}
        component_states = []
        for name in names:
            state = compare(original_components.get(name), observed_components.get(name))
            component_fields[name] = {
                "original": original_components.get(name),
                "observed": observed_components.get(name),
                "match": state,
            }
            if state is not None:
                component_states.append(state)
        component_match = all(component_states) if component_states else None
        fields["reward_components"] = component_fields
    else:
        fields["reward_components"] = None

    original_penalty = original_metrics.get("release_rejection_penalty_observed", original_metrics.get("release_rejection_penalty"))
    observed_penalty = runtime.get("release_rejection_penalty")
    penalty_match = compare(original_penalty, observed_penalty)
    fields["release_rejection_penalty"] = {
        "original": original_penalty,
        "observed": observed_penalty,
        "match": penalty_match,
    }
    required = []
    if original_total is not None:
        required.append(total_match)
    if original_components is not None:
        required.append(component_match)
    if original_penalty is not None:
        required.append(penalty_match)
    coverage_known = bool(required) and all(state is not None for state in required)
    return {
        "available": bool(required),
        "coverage_known": coverage_known,
        "requires_match": bool(required),
        "all_match": bool(all(required)) if coverage_known else None,
        "fields": fields,
        "observed_step_count": 0 if observer is None else len(observer.step_reward_totals),
        "observed_component_rows": 0 if observer is None else int(observer.step_reward_component_rows),
    }


def _safety_from_runtime(runtime: Mapping[str, Any], replay: Mapping[str, Any] | None = None) -> dict[str, Any]:
    telemetry = runtime.get("telemetry") if isinstance(runtime.get("telemetry"), Mapping) else {}
    keys = ("collisions", "nmac", "hard_separation_violations", "uncertified_execution", "SafetyInvariantError")
    result = {key: telemetry.get(key) for key in keys}
    if isinstance(replay, Mapping):
        steps = replay.get("steps")
        if isinstance(steps, Sequence):
            result["SafetyInvariantError"] = sum(bool(row.get("blocked")) for row in steps if isinstance(row, Mapping))
    return result


def _safety_nonzero_value(snapshot: Mapping[str, Any]) -> bool | None:
    values = [snapshot.get(key) for key in snapshot]
    if not values or any(value is None for value in values):
        return None
    try:
        return any(int(value) != 0 for value in values)
    except (TypeError, ValueError):
        return None


def _exact_zero(value: Any) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) == 0.0
    except (TypeError, ValueError):
        return False


def _safety_provenance(
    arm_record: Mapping[str, Any],
    replay: Mapping[str, Any],
    runtime: Mapping[str, Any],
    manifest: Mapping[str, Any],
    spec: Mapping[str, Any],
    tape_sha256: str | None,
    checkpoint_sha256: str | None,
) -> dict[str, Any]:
    """Keep raw safety fields separate from a narrowly derived replay value."""
    raw = _safety_from_runtime(runtime, replay)
    summary = arm_record.get("summary") if isinstance(arm_record.get("summary"), Mapping) else {}
    source_safety = summary.get("safety") if isinstance(summary.get("safety"), Mapping) else {}
    source_metrics = arm_record.get("metrics") if isinstance(arm_record.get("metrics"), Mapping) else {}
    source_hard_clear = source_metrics.get("hard_safety_clear") is True
    steps = replay.get("steps") if isinstance(replay, Mapping) else None
    blocked_count = (
        sum(bool(row.get("blocked")) for row in steps if isinstance(row, Mapping))
        if isinstance(steps, Sequence) else None
    )
    exact_steps = bool(
        isinstance(replay, Mapping)
        and replay.get("steps_expected") is not None
        and replay.get("steps_expected") == replay.get("steps_replayed")
        and int(replay.get("steps_expected", 0) or 0) > 0
    )
    source_certified = bool(
        source_hard_clear
        and all(_exact_zero(source_safety.get(key)) for key in (
            "collisions", "nmac", "hard_separation_violations",
            "SafetyInvariantError", "uncertified_execution",
        ))
    )
    replay_primitives_zero = all(
        _exact_zero(raw.get(key))
        for key in ("collisions", "nmac", "hard_separation_violations")
    )
    replay_error_zero = blocked_count == 0
    input_hashes_match = bool(
        tape_sha256 is not None
        and tape_sha256 == manifest.get("action_tape_sha256")
        and checkpoint_sha256 is not None
        and checkpoint_sha256 == str(spec.get("checkpoint_sha256"))
    )
    conditions = {
        "source_uncertified_execution_zero": _exact_zero(source_safety.get("uncertified_execution")),
        "source_certified": source_certified,
        "runtime_uncertified_execution_missing": raw.get("uncertified_execution") is None,
        "source_hard_safety_clear": source_hard_clear,
        "replay_exact": replay.get("exact") is True,
        "replay_config_match": replay.get("config_match") is True,
        "replay_reset_state_match": replay.get("reset_state_match") is True,
        "replay_steps_exact": exact_steps,
        "replay_steps_unblocked": replay_error_zero,
        "replay_safety_invariant_error_zero": replay_error_zero,
        "replay_collision_nmac_separation_zero": replay_primitives_zero,
        "input_hashes_match": input_hashes_match,
    }
    derived = all(conditions.values())
    shared = dict(raw)
    if derived:
        shared["uncertified_execution"] = 0
    positive = any(
        _exact_zero(value) is False and value is not None
        for value in (raw.get("collisions"), raw.get("nmac"), raw.get("hard_separation_violations"), raw.get("SafetyInvariantError"), raw.get("uncertified_execution"))
    )
    provenance_valid = all(shared.get(key) is not None for key in (
        "collisions", "nmac", "hard_separation_violations", "SafetyInvariantError", "uncertified_execution",
    ))
    return {
        "raw": raw,
        "shared": shared,
        "derived_uncertified_execution": bool(derived),
        "derivation_label": "exact_replay_of_frozen_certified_source" if derived else None,
        "derived_field": "uncertified_execution" if derived else None,
        "derived_value": 0 if derived else None,
        "observed_positive": bool(positive),
        "provenance_valid": bool(provenance_valid),
        "blocked_count": blocked_count,
        "source_safety": dict(source_safety),
        "conditions": conditions,
    }


def _install_profile_observers(env: Any, observer: _PassiveObserver, controller: Any | None) -> tuple[list[tuple[Any, str, bool, Any]], Any | None]:
    restores: list[tuple[Any, str, bool, Any]] = []
    for name, function in (
        ("_runtime_shortlist_release", observer.wrap_release),
        ("_candidate_plan", observer.wrap_candidate_plan),
    ):
        original = getattr(env, name)
        had, previous = _set_instance_method(env, name, function(original))
        restores.append((env, name, had, previous))
    native_restore = None
    if controller is not None:
        original_native = controller._original
        controller._original = observer.wrap_native_release(original_native)

        def restore_native() -> None:
            controller._original = original_native

        native_restore = restore_native
    return restores, native_restore


def _restore_profile_observers(restores: Sequence[tuple[Any, str, bool, Any]], native_restore: Any | None) -> None:
    if native_restore is not None:
        native_restore()
    for obj, name, had, previous in reversed(restores):
        _restore_instance_method(obj, name, had, previous)


def _profile_arm(config: Any, item: Mapping[str, Any], output: Path) -> dict[str, Any]:
    manifest = item["manifest"]
    arm = str(manifest["arm"])
    cell_id = int(manifest["cell"]["cell_id"])
    spec = ARM_SPECS[arm]
    arm_record = item["record"]
    tape = Path(str(manifest["action_tape"]))
    checkpoint = _checkpoint(Path(spec["checkpoint"]), str(spec["checkpoint_sha256"]))
    input_tape_sha256 = _sha256(tape) if tape.is_file() else None
    checkpoint_sha256 = _sha256(checkpoint) if checkpoint.is_file() else None
    caller_before = _rng_snapshot()
    policy = None
    model_before = None
    env = None
    observer = None
    replay: Mapping[str, Any] = {}
    lifecycle_summary: Mapping[str, Any] = {}
    runtime: Mapping[str, Any] = {}
    error: dict[str, Any] | None = None
    controller_context = None
    controller = None
    restores: list[tuple[Any, str, bool, Any]] = []
    native_restore = None
    capture_rows: list[dict[str, Any]] = []
    first_seen: dict[int, dict[str, Any]] = {}
    original_step = None
    original_resolve = None
    wall_started = time.perf_counter_ns()
    try:
        policy, _payload = rank_eval._load_policy(checkpoint, config)
        model_before = rank_eval._model_digest(policy)
        env = frozen._CapturedTerminalEnv(config)
        observer = _PassiveObserver(env, policy, arm, output / "progress.json", cell_id)
        if bool(spec["controller"]):
            sidecar = item.get("sidecar_payload")
            _require(isinstance(sidecar, Mapping), f"prefix sidecar payload missing: {arm}/{cell_id}")
            audit = {"registrations": dict(sidecar.get("registrations", {}))}

            def order_provider(current_env: Any, _uav: Any, uid: int):
                observer.sidecar_order_lookups += 1
                return prefix._registration_lookup(audit, current_env, uid)["order"]

            def metadata_provider(current_env: Any, _uav: Any, uid: int):
                return prefix._registration_lookup(audit, current_env, uid)["metadata"]

            controller_context = prefix_fallback_controller(
                env, order_provider, source_metadata=metadata_provider,
            )
            controller = controller_context.__enter__()
        restores, native_restore = _install_profile_observers(env, observer, controller)
        original_step = env.step
        original_resolve = env._resolve
        capture_rows, first_seen = lifecycle._capture_terminal_rows(env)
        capture_step = env.step
        env.step = MethodType(observer.wrap_step(capture_step), env)
        replay = SafetyTraceRecorder.replay_action_tape(env, tape)
        _require(isinstance(replay, Mapping), "replay result is not a mapping")
        lifecycle_summary = _lifecycle_from_env(config, env, capture_rows, first_seen, replay)
        runtime = support._live_runtime_snapshot({"factory_calls": 1, "env": env}, config)
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        if env is not None:
            if original_step is not None:
                env.step = original_step
            if original_resolve is not None:
                env._resolve = original_resolve
        _restore_profile_observers(restores, native_restore)
        if controller_context is not None:
            controller_context.__exit__(None, None, None)
        _restore_rng(caller_before)

    model_after = None if policy is None else rank_eval._model_digest(policy)
    caller_after = _rng_snapshot()
    caller_rng_unchanged = _rng_equal(caller_before, caller_after)
    if error is None and observer is not None and not observer.rng_restored:
        error = {"type": "RNGRestorationError", "message": "actor probe RNG restoration check failed"}
    safety_provenance = _safety_provenance(
        arm_record, replay, runtime, manifest, spec, input_tape_sha256, checkpoint_sha256,
    )
    replay_metrics: dict[str, Any] = {}
    if error is None and observer is not None:
        replay_safety = safety_provenance["shared"]
        replay_safety_nonzero = _safety_nonzero_value(replay_safety)
        replay_record = {
            "status": arm_record.get("status"),
            "summary": arm_record.get("summary"),
            "lifecycle": dict(lifecycle_summary),
            "runtime_capture": dict(runtime),
            "requests": int(config.total_requested_uavs),
            "outcomes": dict(lifecycle_summary.get("outcomes", {})),
            "weights_unchanged": model_before == model_after,
            "model_unchanged": model_before == model_after,
            "teacher_called": False,
            "training_or_update_called": False,
            "safety": replay_safety,
            "safety_nonzero": replay_safety_nonzero,
            "error": None,
        }
        try:
            replay_metrics = support._record_metrics(replay_record)
        except BaseException as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
    reward_comparison = _compare_rewards(arm_record, observer, runtime) if observer is not None else {
        "available": False, "coverage_known": False, "requires_match": False,
        "all_match": None, "fields": {},
        "observed_step_count": 0, "observed_component_rows": 0,
    }
    counters = None if observer is None else {
        "release_calls": int(observer.release_calls),
        "candidate_plan_calls": int(observer.candidate_plan_calls),
        "proposal_checks_total": int(sum(
            int(row.get("runtime_candidates_checked_delta", 0) or 0)
            for row in observer.release_records
        )),
        "landing_zone_pair_checks": int(observer.landing_zone_pair_checks),
        "suffix_plan_pairs": int(observer.suffix_plan_pairs),
        "suffix_proposal_checks": int(observer.suffix_proposal_checks),
        "approvals_rank_gt3": int(observer.approvals_rank_gt3),
        "zero_check_calls": int(observer.zero_check_calls),
        "release_records": observer.release_records,
        "first_three_all_energy_blocked_calls": int(sum(
            1 for row in observer.release_records
            if all(
                row.get("candidate_reasons", {}).get(str(rank))
                and all(reason == "emergency_energy_guard" for reason in row["candidate_reasons"][str(rank)])
                for rank in (1, 2, 3)
            )
        )),
        "full_mixed_candidate_set_calls": int(sum(
            1 for row in observer.release_records
            if any(int(rank) > 3 for rank in row.get("candidate_reasons", {}))
            and any(
                reason != "emergency_energy_guard"
                for reasons in row.get("candidate_reasons", {}).values()
                for reason in reasons
            )
        )),
    }
    timings = None if observer is None else {
        "actor_request": _timing_summary(observer.actor_request_ns),
        "private_release": _timing_summary(observer.private_release_ns),
        "controller_release_total": _timing_summary(observer.controller_release_ns),
        "candidate_plan": _timing_summary(observer.candidate_plan_ns),
        "env_step": _timing_summary(observer.env_step_ns),
        "observer_overhead": _timing_summary(observer.observer_overhead_ns),
        "denominators": {
            "actor_request": "nonempty replay observations probed; probe actions discarded",
            "private_release": "actual _runtime_shortlist_release calls; native underlying call when prefix controller is installed",
            "controller_release_total": "actual prefix controller invocation including expansion and native call",
            "candidate_plan": "actual _candidate_plan calls",
            "env_step": "original env.step body excluding actor probe",
        },
        "cpu_threads": {"OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS")},
    }
    original_metrics = arm_record.get("metrics") if isinstance(arm_record.get("metrics"), Mapping) else {}
    comparison = _compare_metrics(original_metrics, replay_metrics) if replay_metrics else {"all_match": False, "fields": {}}
    binding = {
        "model_unchanged": model_before is not None and model_before == model_after,
        "caller_rng_unchanged": caller_rng_unchanged,
        "replay_exact": replay.get("exact") is True if isinstance(replay, Mapping) else False,
        "replay_steps_exact": (
            isinstance(replay, Mapping)
            and replay.get("steps_expected") == replay.get("steps_replayed")
        ),
        "lifecycle_valid": bool(lifecycle_summary.get("ledger_alignment")) if lifecycle_summary else False,
        "input_action_tape_sha256_match": input_tape_sha256 == str(manifest.get("action_tape_sha256")),
        "checkpoint_sha256_match": checkpoint_sha256 == str(spec["checkpoint_sha256"]),
        "controller_sidecar_order_lookup": (
            not bool(spec["controller"]) or (observer is not None and observer.sidecar_order_lookups > 0)
        ),
        "reward_binding_match": (
            not reward_comparison.get("requires_match", False)
            or reward_comparison.get("all_match") is True
        ),
        "safety_provenance_valid": bool(safety_provenance.get("provenance_valid")),
        "safety_observed_or_derived_clear": bool(
            safety_provenance.get("provenance_valid")
            and not safety_provenance.get("observed_positive")
        ),
    }
    reward_ok = (
        not reward_comparison.get("requires_match", False)
        or reward_comparison.get("all_match") is True
    )
    status = "completed" if error is None and all(binding.values()) and comparison["all_match"] and reward_ok else "failed"
    return {
        "stage": STAGE,
        "phase": "arm",
        "status": status,
        "arm": arm,
        "cell": dict(manifest["cell"]),
        "source": manifest,
        "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha256, "expected_sha256": str(spec["checkpoint_sha256"])},
        "artifact_bindings": {
            "input_action_tape_sha256": input_tape_sha256,
            "input_action_tape_expected_sha256": manifest.get("action_tape_sha256"),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_expected_sha256": str(spec["checkpoint_sha256"]),
        },
        "replay": dict(replay),
        "lifecycle": dict(lifecycle_summary),
        "runtime_capture": dict(runtime),
        "metrics": replay_metrics,
        "original_metrics": dict(original_metrics),
        "comparison": comparison,
        "reward_comparison": reward_comparison,
        "safety_provenance": safety_provenance,
        "reward_stream": None if observer is None else {
            "step_count": len(observer.step_reward_totals),
            "reward_total_observed": float(sum(observer.step_reward_totals)) if observer.step_reward_totals else None,
            "reward_component_totals_observed": dict(observer.step_reward_components),
            "component_rows": int(observer.step_reward_component_rows),
        },
        "passive_counters": counters,
        "timing": timings,
        "actor_probe": None if observer is None else {
            "forward_count_total": int(sum(observer.actor_forward_counts)),
            "forward_counts": observer.actor_forward_counts,
            "active_observation_counts": observer.actor_observation_counts,
            "outputs_executed": False,
            "rng_restored": bool(observer.rng_restored),
            "errors": observer.probe_errors,
        },
        "binding_checks": binding,
        "model_digest_before": model_before,
        "model_digest_after": model_after,
        "caller_rng_unchanged": caller_rng_unchanged,
        "wall_time_seconds": (time.perf_counter_ns() - wall_started) / 1_000_000_000.0,
        "error": error,
        "PPO_ALLOWED": PPO_ALLOWED,
        "teacher_called": False,
        "training_or_update_called": False,
    }


def _arm_output_path(output: Path, cell_id: int, arm: str) -> Path:
    return output / "arms" / f"cell_{int(cell_id):02d}_{arm}.json"


def _write_progress(output: Path, payload: Mapping[str, Any]) -> None:
    _write_json(output / "progress.json", dict(payload))


def _profile_one(output: Path, config: Any, item: Mapping[str, Any], phase: str) -> dict[str, Any]:
    cell_id = int(item["manifest"]["cell"]["cell_id"])
    arm = str(item["manifest"]["arm"])
    _write_progress(output, {"stage": STAGE, "phase": phase, "status": "RUNNING", "cell_id": cell_id, "arm": arm})
    result = _profile_arm(config, item, output)
    _write_json(_arm_output_path(output, cell_id, arm), result, refuse_existing=True)
    return result


def _validate_arm_result(result: Mapping[str, Any]) -> None:
    _require(result.get("status") in {"completed", "failed"}, "arm result status missing")
    _require(isinstance(result.get("binding_checks"), Mapping), "arm binding checks missing")
    _require(isinstance(result.get("passive_counters"), Mapping), "arm counters missing")
    _require(isinstance(result.get("timing"), Mapping), "arm timing missing")
    _require(result.get("error") is None or result.get("status") == "failed", "arm error/status mismatch")


def _arm_result_qualifies(result: Mapping[str, Any]) -> bool:
    """Return the mechanical continuation gate for the next frozen arm."""
    binding = result.get("binding_checks")
    return bool(
        result.get("status") == "completed"
        and isinstance(binding, Mapping)
        and all(value is True for value in binding.values())
        and result.get("comparison", {}).get("all_match") is True
        and (
            not result.get("reward_comparison", {}).get("requires_match", False)
            or result.get("reward_comparison", {}).get("all_match") is True
        )
    )


def _run_arm_sequence(
    output: Path,
    config: Any,
    selected: Sequence[Mapping[str, Any]],
    phase: str,
) -> list[dict[str, Any]]:
    """Run selected arms in order and stop immediately after a failed arm."""
    results: list[dict[str, Any]] = []
    for item in selected:
        result = _profile_one(output, config, item, phase)
        results.append(result)
        if not _arm_result_qualifies(result):
            break
    return results


def _pilot_payload(output: Path, preflight: Mapping[str, Any], results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected_count = 2
    completed = len(results) == expected_count and all(item.get("status") == "completed" for item in results)
    exact = all(bool(item.get("binding_checks", {}).get("replay_exact")) for item in results)
    binding_ok = len(results) == expected_count and all(_arm_result_qualifies(item) for item in results)
    status = "PILOT_OK" if completed and exact and binding_ok else "PILOT_FAILED"
    return {
        "stage": STAGE, "phase": "pilot", "status": status,
        "preflight_sha256": _sha256(output / "preflight.json"),
        "arms": [
            {"arm": item.get("arm"), "cell_id": item.get("cell", {}).get("cell_id"),
             "path": str(_arm_output_path(output, int(item["cell"]["cell_id"]), str(item["arm"]))),
             "sha256": _sha256(_arm_output_path(output, int(item["cell"]["cell_id"]), str(item["arm"])))}
            for item in results
        ],
        "completed": completed,
        "expected_arm_count": expected_count,
        "completed_arm_count": len(results),
        "exact_replay": exact,
        "binding_checks_pass": binding_ok,
        "PPO_ALLOWED": PPO_ALLOWED,
    }


def run_preflight(output: Path) -> dict[str, Any]:
    _require(not output.exists(), f"profile output already exists: {output}")
    payload = _build_preflight()
    _write_json(output / "preflight.json", payload, refuse_existing=True)
    _write_progress(output, {"stage": STAGE, "phase": "preflight", "status": "PREFLIGHT_OK"})
    return payload


def _validate_pilot_reuse(output: Path, preflight: Mapping[str, Any]) -> list[dict[str, Any]]:
    preflight_path = output / "preflight.json"
    _require(preflight_path.is_file(), "pilot preflight missing")
    pilot = _read_json(output / "pilot.json")
    _require(pilot.get("preflight_sha256") == _sha256(preflight_path), "pilot preflight changed")
    _require(pilot.get("status") == "PILOT_OK", "pilot is not complete")
    items: list[dict[str, Any]] = []
    for cell_id, arm in ((1, "before_legacy"), (1, "before_prefix_fallback")):
        path = _arm_output_path(output, cell_id, arm)
        _require(path.is_file(), f"pilot arm missing: {path}")
        row = _read_json(path)
        _validate_arm_result(row)
        _require(row.get("status") == "completed", f"pilot arm is not completed: {path}")
        _require(bool(row.get("binding_checks", {}).get("replay_exact")), f"pilot arm replay is not exact: {path}")
        _require(bool(row.get("comparison", {}).get("all_match")), f"pilot arm metrics do not match: {path}")
        _require(row.get("reward_comparison", {}).get("all_match") is not False, f"pilot arm rewards do not match: {path}")
        items.append(row)
    return items


def run_pilot(output: Path) -> dict[str, Any]:
    preflight, config, arms = _phase_inputs(output)
    _require(not (output / "pilot.json").exists(), "pilot artifact already exists")
    selected = [arms[(1, "before_legacy")], arms[(1, "before_prefix_fallback")]]
    results = _run_arm_sequence(output, config, selected, "pilot")
    payload = _pilot_payload(output, preflight, results)
    _write_json(output / "pilot.json", payload, refuse_existing=True)
    _write_progress(output, payload)
    return payload


def run_full(output: Path) -> dict[str, Any]:
    preflight, config, arms = _phase_inputs(output)
    _require(not (output / "full.json").exists(), "full artifact already exists")
    pilot = _read_json(output / "pilot.json")
    _require(pilot.get("status") == "PILOT_OK", "pilot is not accepted for full replay")
    pilot_rows = _validate_pilot_reuse(output, preflight)
    results: list[dict[str, Any]] = list(pilot_rows)
    for cell in prefix.CELL_SPECS:
        for arm in ARMS:
            if int(cell["cell_id"]) == 1 and arm in {"before_legacy", "before_prefix_fallback"}:
                continue
            result = _profile_one(output, config, arms[(int(cell["cell_id"]), arm)], "full")
            results.append(result)
            if not _arm_result_qualifies(result):
                break
        if results and not _arm_result_qualifies(results[-1]):
            break
    status = "FULL_OK" if len(results) == 16 and all(row.get("status") == "completed" for row in results) else "FULL_FAILED"
    payload = {
        "stage": STAGE, "phase": "full", "status": status,
        "preflight_sha256": _sha256(output / "preflight.json"),
        "pilot_sha256": _sha256(output / "pilot.json"),
        "arm_count": len(results),
        "arms": [
            {"cell_id": int(row["cell"]["cell_id"]), "arm": row["arm"],
             "path": str(_arm_output_path(output, int(row["cell"]["cell_id"]), str(row["arm"]))),
             "sha256": _sha256(_arm_output_path(output, int(row["cell"]["cell_id"]), str(row["arm"]))),
             "status": row.get("status")}
            for row in results
        ],
        "PPO_ALLOWED": PPO_ALLOWED,
    }
    _write_json(output / "full.json", payload, refuse_existing=True)
    _write_progress(output, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "pilot", "full"), required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.phase == "preflight":
            run_preflight(args.output_dir)
        elif args.phase == "pilot":
            run_pilot(args.output_dir)
        else:
            run_full(args.output_dir)
    except BaseException as exc:
        try:
            _write_json(args.output_dir / "error.json", {
                "stage": STAGE, "phase": args.phase, "status": "FAILED",
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }, refuse_existing=True)
        finally:
            raise
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
