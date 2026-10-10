"""One-shot real-environment fixture for the frozen gate-control request path.

This file is temporary evidence code.  It runs one real single-actor
``_CapturedTerminalEnv`` through the frozen ``_run_arm`` with a local dummy
policy and a temporary details/request hook.  It does not write project or
registered training outputs.
"""
from __future__ import annotations

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
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training import paired_gate_diagnostic as paired
from terminal_marl.training import run_p3_forced_wait_frozen_20260930 as frozen
from terminal_marl.training import run_p3_gate_release_control_20260930 as runner


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "p3_gate_control_natural_fixture_projectvenv.json"
LOG = ROOT / "p3_gate_control_natural_fixture_projectvenv.log"
ENV_SEED = 877101
KEYED_SEED = 9303101


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


def _dynamic_details(_policy: Any, env: Any) -> dict[str, Any]:
    """Return a genuine active-UID view with deterministic unit distributions."""

    uids = [str(uid) for uid in sorted(getattr(env, "active", {}))]
    count = len(uids)
    actor_mask = np.asarray(
        [bool(frozen._legacy_actor(env, uid)) for uid in uids], dtype=bool
    )
    world = np.zeros((count, frozen.OBSERVATION_DIM), dtype=np.float32)
    gate_probabilities = np.tile(np.asarray([[1.0, 0.0]], dtype=np.float64), (count, 1))
    proposal_probabilities = np.zeros((count, 25), dtype=np.float64)
    if count:
        proposal_probabilities[:, 12] = 1.0
    gate_logits = np.tile(np.asarray([[0.0, -1.0]], dtype=np.float64), (count, 1))
    proposal_logits = np.zeros((count, 25), dtype=np.float64)
    if count:
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


def _energy_row_checks(arm: dict[str, Any]) -> dict[str, Any]:
    energy = arm.get("energy_ledger", {})
    rows = energy.get("rows", {}) if isinstance(energy, dict) else {}
    row_values = []
    for key, row in rows.items() if isinstance(rows, dict) else ():
        row_values.append({
            "arrival_id": str(key),
            "keys": sorted(row) if isinstance(row, dict) else [],
            "energy_consumed": row.get("energy_consumed") if isinstance(row, dict) else None,
            "energy_remaining": row.get("energy_remaining") if isinstance(row, dict) else None,
            "outcome": row.get("outcome") if isinstance(row, dict) else None,
        })
    total_consumed = sum(
        float(item["energy_consumed"])
        for item in row_values
        if isinstance(item.get("energy_consumed"), (int, float))
        and math.isfinite(float(item["energy_consumed"]))
    )
    return {
        "rows_count": len(row_values),
        "rows": row_values,
        "energy_consumed_sum": float(total_consumed),
        "coverage": energy.get("coverage") if isinstance(energy, dict) else None,
    }


def _run_once() -> dict[str, Any]:
    if frozen.MAX_STEPS != 1800 or runner.MAX_STEPS != 1800:
        raise RuntimeError("registered 1800-step cap changed")
    config = SimConfig(
        action_schema_version=frozen.ACTION_SCHEMA,
        observation_schema_version=frozen.OBSERVATION_SCHEMA,
        runtime_proposal_shortlist_k=3,
        arrival_model="finite_cohort",
        total_requested_uavs=1,
        fixed_arrival_schedule=(0.0,),
        reset_mode="warm_start_rings",
        warm_start_uavs=1,
        initial_soc_mode="fixed",
        initial_soc=0.001,
        max_simulation_seconds=300.0,
    )
    policy = SimpleNamespace(model=nn.Linear(1, 1))
    original_details = paired._details_context
    original_keyed = frozen._keyed_request
    try:
        paired._details_context = _dynamic_details
        frozen._keyed_request = runner._control_keyed_request
        arm = frozen._run_arm(
            policy,
            config,
            env_seed=ENV_SEED,
            keyed_seed=KEYED_SEED,
            mode="legacy",
        )
    finally:
        paired._details_context = original_details
        frozen._keyed_request = original_keyed
    serialized = _jsonable(arm)
    if not isinstance(serialized, dict):
        raise RuntimeError("arm result did not serialize to a JSON object")
    return {
        "fixture": {
            "env_class": f"{TerminalParallelEnv.__module__}.{TerminalParallelEnv.__name__}",
            "runner_env_class": f"{frozen._CapturedTerminalEnv.__module__}.{frozen._CapturedTerminalEnv.__name__}",
            "env_seed": ENV_SEED,
            "keyed_seed": KEYED_SEED,
            "initial_soc": 0.001,
            "max_simulation_seconds": 300.0,
            "registered_max_steps": 1800,
            "policy": "SimpleNamespace(model=nn.Linear(1,1)); no checkpoint/Teacher/PPO",
            "details_hook": "dynamic active UIDs + frozen._legacy_actor + 88D zero world + WAIT=1/proposal12=1",
            "request_hook": "run_p3_gate_release_control_20260930._control_keyed_request",
        },
        "observed": {
            "status": arm.get("status"),
            "stop_reason": arm.get("stop_reason"),
            "censored": arm.get("censored"),
            "ticks": arm.get("ticks"),
            "simulation_seconds": arm.get("simulation_seconds"),
            "ledger_requested_ids": arm.get("ledger", {}).get("requested_ids"),
            "ledger_outcomes": arm.get("ledger", {}).get("outcomes"),
            "ledger_unresolved_ids": arm.get("ledger", {}).get("unresolved_ids"),
            "request_identity_count": arm.get("request_identity_count"),
            "request_identity_unique": arm.get("request_identity_unique"),
            "model_unchanged": arm.get("model_unchanged"),
            "caller_rng_unchanged": arm.get("caller_rng_unchanged"),
            "safety": arm.get("safety"),
            "energy": _energy_row_checks(arm),
            "request_elapsed_seconds_by_arrival": arm.get("wait_metrics", {}).get(
                "request_elapsed_seconds_by_arrival"
            ),
            "wait_integrity": arm.get("wait_integrity"),
            "json_serialization": True,
        },
        "raw_arm": serialized,
    }


def main() -> int:
    if OUTPUT.exists():
        raise RuntimeError(f"refusing second fixture attempt: {OUTPUT}")
    try:
        payload = _run_once()
        OUTPUT.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload["observed"], sort_keys=True))
        return 0
    except BaseException:
        failure = {"status": "fixture_exception", "traceback": traceback.format_exc()}
        OUTPUT.write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(failure, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
