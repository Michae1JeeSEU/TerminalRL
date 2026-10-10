"""Bounded read-only probe for emergency candidates rejected by the slack gate."""

from __future__ import annotations

import importlib.util
import inspect
import json
import math
import sys
import time
import types
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OLD_COORDINATOR = ROOT / "results" / "_landing_flow_v5.py"
OUTPUT = ROOT / "results" / "flow_v5_missed_emergency_probe.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SEED = 841301
MAX_STEPS = 450
TARGET_UIDS = (3, 4, 11)
CHECK_FROM = 260.0
CHECK_EVERY = 5.0
EMERGENCY_DURATIONS = (28.0, 36.0, 48.0, 60.0, 90.0)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _load_old_coordinator() -> str:
    if not OLD_COORDINATOR.exists():
        raise FileNotFoundError(OLD_COORDINATOR)
    # The package __init__ eagerly imports the live environment.  Install a
    # namespace-only package shell so the historical coordinator can be
    # loaded before that environment import, as required by this probe.
    package = sys.modules.get("terminal_marl")
    if package is None:
        package = types.ModuleType("terminal_marl")
        package.__path__ = [str(ROOT / "terminal_marl")]
        package.__package__ = "terminal_marl"
        sys.modules["terminal_marl"] = package
    module_name = "terminal_marl.core.landing_flow"
    spec = importlib.util.spec_from_file_location(module_name, OLD_COORDINATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load coordinator spec: {OLD_COORDINATOR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    source = inspect.getsourcefile(module.LandingFlowCoordinator)
    if source is None or Path(source).resolve() != OLD_COORDINATOR.resolve():
        raise RuntimeError(f"coordinator source mismatch: {source}")
    return str(Path(source).resolve())


def _current_release_reason(env, uid: int) -> dict[str, Any]:
    decision = env.last_release_decisions.get(uid)
    row = env.last_release_telemetry.get(uid)
    if not isinstance(row, Mapping):
        return {
            "decision": decision,
            "reason": None if decision is None else str(decision),
            "primary_blocker": None,
            "secondary_constraints": [],
        }
    secondary = row.get("secondary_constraints", ())
    if not isinstance(secondary, (list, tuple, set)):
        secondary = [secondary]
    secondary = [str(item) for item in secondary if item is not None]
    reason = secondary[-1] if secondary else (None if decision is None else str(decision))
    return {
        "decision": decision,
        "reason": reason,
        "primary_blocker": row.get("primary_blocker"),
        "secondary_constraints": secondary,
    }


def _earliest_normal_energy_guard(env, info: Mapping[str, Any], current):
    if current is not None:
        return current
    rows = info.get("release_intent_telemetry", {})
    decisions = info.get("release_decisions", {})
    if not isinstance(rows, Mapping):
        return None
    for raw_uid, row in rows.items():
        if not isinstance(row, Mapping) or not row.get("flow_coordination"):
            continue
        reasons = row.get("flow_candidate_rejection_reasons", ())
        if not isinstance(reasons, (list, tuple, set)):
            continue
        if "emergency_energy_guard" not in {str(item) for item in reasons}:
            continue
        uid = int(raw_uid)
        decision = decisions.get(str(raw_uid), decisions.get(uid))
        return {
            "uid": uid,
            "time": float(env.now),
            "decision": decision,
            "reason": "emergency_energy_guard",
            "candidate_rejection_reasons": [str(item) for item in reasons],
        }
    return None


def _candidate_witnesses(
    env,
    probe_safety,
    uid: int,
    witness_counts: dict[int, int],
    witnesses: list[dict[str, Any]],
) -> None:
    if witness_counts[uid] >= 3:
        return
    uav = env.active.get(uid)
    if uav is None or uav.committed_plan is not None or uav.ring_role is None:
        return
    if float(uav.soc) > float(env.config.emergency_trigger_soc) + 1e-12:
        return
    if float(env.now) < CHECK_FROM - 1e-9:
        return
    offset = (float(env.now) - CHECK_FROM) / CHECK_EVERY
    if abs(offset - round(offset)) > 1e-8:
        return

    coordinator = env.flow_coordinator
    zones = tuple(env.lz_manager.rank_available_zones(uav, env.now))
    if not zones:
        return
    committed = coordinator._committed_plans()
    snapshot = coordinator._snapshot()
    slack_limit = (
        2.0 * float(env.airspace.headways["i2f"]) + 5.0
    ) * float(env.config.power_reference)
    remaining = float(uav.energy_remaining)
    for duration in EMERGENCY_DURATIONS:
        if witness_counts[uid] >= 3:
            break
        for lz_id in zones:
            if witness_counts[uid] >= 3:
                break
            try:
                plan = coordinator.trajectory.final_approach(
                    uav,
                    env.lz_manager.zones[lz_id],
                    env.airspace,
                    env.now,
                    duration,
                    emergency=True,
                )
            except (RuntimeError, ValueError, KeyError, AssertionError):
                continue
            required = float(env.energy.emergency_required_energy(uav, plan))
            if remaining + 1e-9 < required:
                continue
            safety = probe_safety.filter_plan(plan, committed, snapshot)
            if not safety.approved:
                continue
            slack = remaining - required
            if slack <= slack_limit + 1e-9:
                continue
            release_reason = _current_release_reason(env, uid)
            record = {
                "now": float(env.now),
                "uid": int(uid),
                "soc": float(uav.soc),
                "role": uav.ring_role.value,
                "free_compatible_pads": [int(item) for item in zones],
                "pad": int(lz_id),
                "duration": float(duration),
                "energy_remaining": remaining,
                "required": required,
                "slack": slack,
                "current_limit": float(slack_limit),
                "route_end": float(plan.end_time),
                "current_release_reason": release_reason,
            }
            witnesses.append(record)
            witness_counts[uid] += 1
            print(
                json.dumps(_jsonable(record), sort_keys=True, separators=(",", ":")),
                flush=True,
            )


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {OUTPUT}")
    coordinator_source = _load_old_coordinator()

    from terminal_marl.core.safety_filter import SafetyFilter
    from terminal_marl.envs.parallel_env import TerminalParallelEnv
    from terminal_marl.training.diagnose_landing_flow import _build_diagnostic_config

    config = _build_diagnostic_config(True)
    env = TerminalParallelEnv(config)
    env.reset(seed=SEED)
    if env.flow_coordinator is None:
        raise RuntimeError("coordinated flow coordinator was not installed")
    probe_safety = SafetyFilter(env.config)
    started = time.perf_counter()
    witnesses: list[dict[str, Any]] = []
    witness_counts = {uid: 0 for uid in TARGET_UIDS}
    earliest_guard = None
    steps = 0
    terminal = False

    while steps < MAX_STEPS:
        all_three = all(value >= 3 for value in witness_counts.values())
        if all_three:
            break
        for uid in TARGET_UIDS:
            _candidate_witnesses(env, probe_safety, uid, witness_counts, witnesses)
        actions = {
            str(uid): np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
            for uid in env.active
        }
        _, _, _terminated, _truncated, info = env.step(actions)
        steps += 1
        earliest_guard = _earliest_normal_energy_guard(env, info, earliest_guard)
        terminal = bool(
            env.arrivals.exhausted()
            and not env.active
            and not env.admission.queue.items
            and not env.service_uavs
        )
        if terminal:
            break

    result = {
        "protocol": {
            "seed": SEED,
            "max_steps": MAX_STEPS,
            "coordinated": True,
            "config_rule_flow_coordination_enabled": bool(config.rule_flow_coordination_enabled),
            "initial_soc_min": float(config.initial_soc_min),
            "initial_soc_max": float(config.initial_soc_max),
            "emergency_trigger_soc": float(config.emergency_trigger_soc),
            "coordinator_source": coordinator_source,
            "candidate_execution": False,
            "safety_filter_source": "separate SafetyFilter(config)",
            "check_start": CHECK_FROM,
            "check_interval": CHECK_EVERY,
            "durations": list(EMERGENCY_DURATIONS),
        },
        "stop": {
            "steps": int(steps),
            "sim_time": float(env.now),
            "terminal": terminal,
            "wall_seconds": float(time.perf_counter() - started),
            "witness_counts": {str(uid): int(value) for uid, value in witness_counts.items()},
        },
        "witnesses": witnesses[:27],
        "earliest_normal_energy_guard_rejection": earliest_guard,
    }
    text = json.dumps(_jsonable(result), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    if len(text.encode("utf-8")) > 50 * 1024:
        raise RuntimeError("probe output exceeds 50KB")
    OUTPUT.write_text(text + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "bytes": len((text + "\n").encode("utf-8")),
                "steps": steps,
                "sim_time": env.now,
                "witness_counts": witness_counts,
                "earliest_normal_energy_guard_rejection": earliest_guard,
            },
            default=_jsonable,
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
