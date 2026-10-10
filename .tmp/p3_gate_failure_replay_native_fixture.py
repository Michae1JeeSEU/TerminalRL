"""One-shot native equivalence fixture for the gate-failure replay observer.

This temporary check uses the real frozen arm runner, environment, trajectory
generator, and safety path.  It compares an uninstrumented legacy arm with the
same arm under the replay observer.  The details/request hooks only provide
the known single-actor fixture policy inputs from the existing fixture.
"""
from __future__ import annotations

import copy
import json
import math
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import nn

from terminal_marl.core.config import SimConfig
from terminal_marl.training import paired_gate_diagnostic as paired
from terminal_marl.training import run_p3_forced_wait_frozen_20260930 as frozen
from terminal_marl.training import run_p3_gate_failure_replay_20261001 as replay
from terminal_marl.training import run_p3_gate_release_control_20260930 as control


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "p3_gate_failure_replay_native_fixture_v3.json"
ENV_SEED = 877101
KEYED_SEED = 9303101
MAX_SIMULATION_SECONDS = 300.0
INITIAL_SOC = 0.001


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if hasattr(value, "value") and isinstance(value.value, str):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _nonfinite_paths(value: Any, path: str = "$") -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            rows.extend(_nonfinite_paths(item, f"{path}.{key}"))
        return rows
    if isinstance(value, (list, tuple, set, frozenset)):
        for index, item in enumerate(value):
            rows.extend(_nonfinite_paths(item, f"{path}[{index}]"))
        return rows
    if isinstance(value, np.ndarray):
        return _nonfinite_paths(value.tolist(), path)
    if isinstance(value, torch.Tensor):
        return _nonfinite_paths(value.detach().cpu().tolist(), path)
    if isinstance(value, np.generic):
        return _nonfinite_paths(value.item(), path)
    if isinstance(value, float) and not math.isfinite(value):
        return [{"path": path, "value": repr(value)}]
    return rows


def _dynamic_details(_policy: Any, env: Any) -> dict[str, Any]:
    """Known single-actor fixture details with genuine active UIDs."""
    uids = [str(uid) for uid in sorted(getattr(env, "active", {}))]
    count = len(uids)
    actor_mask = np.asarray(
        [bool(frozen._legacy_actor(env, uid)) for uid in uids], dtype=bool
    )
    world = np.zeros((count, frozen.OBSERVATION_DIM), dtype=np.float32)
    gate_probabilities = np.tile(np.asarray([[1.0, 0.0]], dtype=np.float64), (count, 1))
    proposal_probabilities = np.zeros((count, 25), dtype=np.float64)
    gate_logits = np.tile(np.asarray([[0.0, -1.0]], dtype=np.float64), (count, 1))
    proposal_logits = np.zeros((count, 25), dtype=np.float64)
    if count:
        proposal_probabilities[:, 12] = 1.0
        proposal_logits[:, 12] = 1.0
    return {
        "uids": uids,
        "world": world,
        "actor_mask": actor_mask,
        "gate_logits": gate_logits,
        "proposal_logits": proposal_logits,
        "gate_probabilities": gate_probabilities,
        "proposal_probabilities": proposal_probabilities,
        "step": int(getattr(env, "step_count", 0)),
    }


def _fixture_config() -> Any:
    if frozen.MAX_STEPS != 1800 or control.MAX_STEPS != 1800:
        raise RuntimeError("registered 1800-step cap changed")
    return SimConfig(
        action_schema_version=frozen.ACTION_SCHEMA,
        observation_schema_version=frozen.OBSERVATION_SCHEMA,
        runtime_proposal_shortlist_k=3,
        arrival_model="finite_cohort",
        total_requested_uavs=1,
        fixed_arrival_schedule=(0.0,),
        reset_mode="warm_start_rings",
        warm_start_uavs=1,
        initial_soc_mode="fixed",
        initial_soc=INITIAL_SOC,
        max_simulation_seconds=MAX_SIMULATION_SECONDS,
    )


def _run_arm(policy: Any, config: Any, trace: Any | None = None) -> dict[str, Any]:
    original_details = paired._details_context
    original_keyed = frozen._keyed_request
    try:
        paired._details_context = _dynamic_details
        frozen._keyed_request = control._control_keyed_request
        if trace is None:
            return frozen._run_arm(
                policy, config, env_seed=ENV_SEED, keyed_seed=KEYED_SEED,
                mode="legacy",
            )
        with replay._instrument_runtime(trace):
            return frozen._run_arm(
                policy, config, env_seed=ENV_SEED, keyed_seed=KEYED_SEED,
                mode="legacy",
            )
    finally:
        paired._details_context = original_details
        frozen._keyed_request = original_keyed


def _arm_summary(arm: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": arm.get("status"),
        "stop_reason": arm.get("stop_reason"),
        "censored": arm.get("censored"),
        "ticks": arm.get("ticks"),
        "simulation_seconds": arm.get("simulation_seconds"),
        "terminal_outcome": arm.get("ledger", {}).get("outcomes"),
        "outcome_counts": arm.get("outcome_counts"),
        "request_identity_count": arm.get("request_identity_count"),
        "request_identity_unique": arm.get("request_identity_unique"),
        "model_digest_before": arm.get("model_digest_before"),
        "model_digest_after": arm.get("model_digest_after"),
        "model_unchanged": arm.get("model_unchanged"),
        "caller_rng_unchanged": arm.get("caller_rng_unchanged"),
        "safety": arm.get("safety"),
    }


def _run_once() -> dict[str, Any]:
    config = _fixture_config()
    base_policy = SimpleNamespace(model=nn.Linear(1, 1))
    baseline_policy = copy.deepcopy(base_policy)
    observed_policy = copy.deepcopy(base_policy)

    baseline = _run_arm(baseline_policy, config)
    trace = replay._TraceCollector([0])
    observed = _run_arm(observed_policy, config, trace)

    baseline_view = replay._comparison_view(baseline, control=False)
    observed_view = replay._comparison_view(observed, control=False)
    baseline_view.pop("elapsed_seconds", None)
    observed_view.pop("elapsed_seconds", None)
    mismatches: list[dict[str, Any]] = []
    replay._compare_values(baseline_view, observed_view, "arm", mismatches)

    trace_payload = trace.payload(
        arm_mode="legacy", env_seed=ENV_SEED, keyed_seed=KEYED_SEED,
    )
    ticks = int(observed.get("ticks", -1))
    target_states = [
        row.get("targets", {}).get("0", {})
        for row in trace.steps
    ]
    target_state_observed = any(item.get("before", {}).get("state") is not None
                                or item.get("after", {}).get("state") is not None
                                for item in target_states)
    steps_equal_ticks = len(trace.steps) == ticks
    raw_nonfinite_paths = _nonfinite_paths(trace_payload)
    trace_export = replay._jsonable(trace_payload)
    json.dumps(trace_export, allow_nan=False)
    serialized_baseline = _jsonable(baseline)
    serialized_observed = _jsonable(observed)
    json.dumps(serialized_baseline, allow_nan=False)
    json.dumps(serialized_observed, allow_nan=False)

    baseline_summary = _arm_summary(baseline)
    observed_summary = _arm_summary(observed)
    comparison = {
        "excluded_fields": ["elapsed_seconds"],
        "exact_equivalence_excluding_walltime": not mismatches,
        "mismatches": mismatches,
        "model_equivalence": (
            baseline.get("model_digest_before") == observed.get("model_digest_before")
            and baseline.get("model_digest_after") == observed.get("model_digest_after")
            and baseline.get("model_unchanged") is True
            and observed.get("model_unchanged") is True
        ),
        "rng_equivalence": (
            baseline.get("caller_rng_unchanged") is True
            and observed.get("caller_rng_unchanged") is True
        ),
        "safety_equivalence": baseline.get("safety") == observed.get("safety"),
        "terminal_outcome_equivalence": (
            baseline.get("ledger", {}).get("outcomes")
            == observed.get("ledger", {}).get("outcomes")
        ),
    }
    trace_checks = {
        "summary": trace_payload["summary"],
        "step_records": len(trace.steps),
        "arm_ticks": ticks,
        "steps_equal_ticks": steps_equal_ticks,
        "target0_fieldstate_observed": target_state_observed,
        "raw_nonfinite_count": len(raw_nonfinite_paths),
        "raw_nonfinite_paths": raw_nonfinite_paths,
        "json_allow_nan_false": True,
    }
    checks = {
        "native_equivalence_assertions": bool(not mismatches),
        "trace_assertions": bool(steps_equal_ticks and target_state_observed),
        "model_rng_safety_terminal_assertions": bool(
            baseline.get("model_unchanged") is True
            and observed.get("model_unchanged") is True
            and baseline.get("caller_rng_unchanged") is True
            and observed.get("caller_rng_unchanged") is True
            and baseline.get("safety", {}).get("safety_violation") is False
            and observed.get("safety", {}).get("safety_violation") is False
            and baseline.get("ledger", {}).get("outcomes")
            == observed.get("ledger", {}).get("outcomes")
        ),
    }
    pre_serialization = {
        "status": "PRE_SERIALIZATION",
        "fixture": {
            "env_seed": ENV_SEED,
            "keyed_policy_seed": KEYED_SEED,
            "initial_soc": INITIAL_SOC,
            "max_simulation_seconds": MAX_SIMULATION_SECONDS,
            "registered_max_steps": 1800,
        },
        "baseline": baseline_summary,
        "observed": observed_summary,
        "comparison": comparison,
        "trace": trace_checks,
        "checks": checks,
    }
    OUTPUT.write_text(
        json.dumps(_jsonable(pre_serialization), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "fixture": {
            "env_seed": ENV_SEED,
            "keyed_policy_seed": KEYED_SEED,
            "initial_soc": INITIAL_SOC,
            "max_simulation_seconds": MAX_SIMULATION_SECONDS,
            "registered_max_steps": 1800,
            "observation_schema": frozen.OBSERVATION_SCHEMA,
            "observation_dimension": frozen.OBSERVATION_DIM,
            "policy": "SimpleNamespace(model=nn.Linear(1,1)); no checkpoint/Teacher/PPO",
            "native_arm": "frozen._run_arm(mode=legacy)",
            "details_hook": "dynamic active UIDs + frozen._legacy_actor + 88D zero world + proposal12=1",
            "request_hook": "run_p3_gate_release_control_20260930._control_keyed_request",
        },
        "baseline": baseline_summary,
        "observed": observed_summary,
        "comparison": comparison,
        "trace": trace_checks,
        "checks": checks,
        "raw": {
            "baseline": serialized_baseline,
            "observed": serialized_observed,
            "trace": trace_export,
        },
    }


def main() -> int:
    if OUTPUT.exists():
        raise RuntimeError(f"refusing second fixture attempt: {OUTPUT}")
    try:
        payload = _run_once()
        if not payload["comparison"]["exact_equivalence_excluding_walltime"]:
            raise AssertionError(json.dumps(payload["comparison"], sort_keys=True))
        if not payload["trace"]["steps_equal_ticks"]:
            raise AssertionError("trace step count differs from native arm ticks")
        if not payload["trace"]["target0_fieldstate_observed"]:
            raise AssertionError("trace did not observe target arrival 0 state")
        OUTPUT.write_text(
            json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload["checks"], sort_keys=True))
        return 0
    except BaseException:
        existing: dict[str, Any] = {}
        if OUTPUT.exists():
            try:
                loaded = json.loads(OUTPUT.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    existing = loaded
            except (OSError, json.JSONDecodeError):
                existing = {}
        failure = existing
        failure["status"] = "fixture_exception"
        failure["traceback"] = traceback.format_exc()
        OUTPUT.write_text(json.dumps(failure, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps(failure, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
