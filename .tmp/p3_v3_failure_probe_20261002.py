"""Bounded read-only reproduction probe for the first V3 matrix arm.

This module instruments only method entry/exception boundaries in the
existing generator and time parameterizer.  It preserves all returns,
exceptions, candidates, and RNG calls, and runs the exact first registered
source arm for at most fourteen environment steps.
"""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import re
import sys
import traceback
from typing import Any

import numpy as np

from terminal_marl.core.trajectory_generator import TrajectoryGenerator
from terminal_marl.core.trajectory_time_parameterization import (
    TrajectoryTimeParameterizer,
)
from terminal_marl.scripts import evaluate_p3_foundation_repair_20261002 as evaluator


PROJECT_ROOT = Path("/home/user/projects/terminal")
MANIFEST_PATH = PROJECT_ROOT / ".tmp" / "p3_v3_matrix_source_binding_20261002.json"
OUTPUT_DIR = Path(
    "/home/user/training_outputs/terminal/trajectory_v3_repair_20261002/failure_probe"
)
OUTPUT_JSON = OUTPUT_DIR / "failure_probe.json"
OUTPUT_CONTRACT_PREFLIGHT = OUTPUT_DIR / "output_contract_preflight.json"
MAX_STEPS = 14
FIRST_SPEC = {
    "cell_id": "cell_1",
    "env_seed": 877101,
    "policy_seed": 878201,
    "arm_id": "legacy88",
    "kind": "source",
    "schema": "88",
    "execution": "legacy",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    try:
        return evaluator._jsonable(value)
    except Exception:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return repr(value)


def _compact(value: Any, *, limit: int = 100) -> Any:
    if isinstance(value, np.ndarray):
        return _compact(value.tolist(), limit=limit)
    if isinstance(value, dict):
        return {str(key): _compact(item, limit=limit) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        values = list(value)
        if len(values) > limit:
            return [_compact(item, limit=limit) for item in values[:limit]] + [
                f"... truncated {len(values) - limit} items"
            ]
        return [_compact(item, limit=limit) for item in values]
    return _jsonable(value)


def _finite_attr(obj: Any, name: str) -> Any:
    value = getattr(obj, name, None)
    return _compact(value)


def _uav_state(uav: Any) -> dict[str, Any]:
    return {
        "uav_id": _finite_attr(uav, "uav_id"),
        "uid": _finite_attr(uav, "uid"),
        "phase": _finite_attr(uav, "phase"),
        "ring_role": _finite_attr(uav, "ring_role"),
        "position": _finite_attr(uav, "position"),
        "velocity": _finite_attr(uav, "velocity"),
        "acceleration": _finite_attr(uav, "acceleration"),
        "speed": _finite_attr(uav, "speed"),
        "ring_progress": _finite_attr(uav, "ring_progress"),
        "soc": _finite_attr(uav, "soc"),
        "energy": _finite_attr(uav, "energy"),
        "energy_remaining": _finite_attr(uav, "energy_remaining"),
        "energy_used": _finite_attr(uav, "energy_used"),
        "battery_energy": _finite_attr(uav, "battery_energy"),
        "energy_capacity": _finite_attr(uav, "energy_capacity"),
    }


def _lz_state(lz: Any) -> dict[str, Any]:
    return {
        "lz_id": _finite_attr(lz, "lz_id"),
        "position": _finite_attr(lz, "position"),
    }


def _ring_state(airspace: Any, uav: Any) -> dict[str, Any] | None:
    try:
        ring = airspace.actual_ring(uav.ring_role)
        return {
            "radius": _finite_attr(ring, "radius"),
            "altitude": _finite_attr(ring, "altitude"),
            "role": _finite_attr(uav, "ring_role"),
        }
    except Exception as exc:
        return {"snapshot_error": f"{type(exc).__name__}: {exc}"}


def _constraints_state(constraints: Any) -> dict[str, Any] | None:
    if constraints is None:
        return None
    return {
        "end_horizontal_speed": _finite_attr(constraints, "end_horizontal_speed"),
        "tangent_yaw": _finite_attr(constraints, "tangent_yaw"),
        "end_direction": _finite_attr(constraints, "end_direction"),
        "end_curvature_vector": _finite_attr(constraints, "end_curvature_vector"),
    }


def _source_binding() -> dict[str, Any]:
    if not MANIFEST_PATH.is_file():
        raise RuntimeError(f"source binding manifest missing: {MANIFEST_PATH}")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    paths = manifest.get("source_paths")
    expected = manifest.get("server_sha256")
    if not isinstance(paths, list) or not isinstance(expected, dict):
        raise RuntimeError("source binding manifest has no complete server hash map")
    observed: dict[str, str] = {}
    mismatches: dict[str, dict[str, str]] = {}
    for relative in paths:
        path = PROJECT_ROOT / str(relative)
        if not path.is_file():
            raise RuntimeError(f"source file missing: {path}")
        actual = _sha256(path)
        observed[str(relative)] = actual
        if actual.lower() != str(expected.get(str(relative), "")).lower():
            mismatches[str(relative)] = {
                "expected": str(expected.get(str(relative), "")),
                "actual": actual,
            }
    if mismatches:
        raise RuntimeError(f"source hash mismatch: {mismatches}")
    checkpoint = manifest.get("checkpoint", {})
    checkpoint_path = Path(str(checkpoint.get("path", "")))
    if not checkpoint_path.is_file():
        raise RuntimeError(f"checkpoint missing: {checkpoint_path}")
    checkpoint_sha = _sha256(checkpoint_path)
    if checkpoint_sha.lower() != str(checkpoint.get("expected_sha256", "")).lower():
        raise RuntimeError(
            "checkpoint hash mismatch: "
            f"expected {checkpoint.get('expected_sha256')}, actual {checkpoint_sha}"
        )
    if not bool(manifest.get("source_hashes_match")):
        raise RuntimeError("source binding manifest is not marked as matching")
    return {
        "manifest": str(MANIFEST_PATH),
        "source_paths": [str(path) for path in paths],
        "source_sha256": observed,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
    }


def _validate_first_spec() -> dict[str, Any]:
    specs = evaluator._arm_specs()
    if not specs or dict(specs[0]) != FIRST_SPEC:
        raise RuntimeError(f"first registered spec changed: {specs[0] if specs else None}")
    return dict(specs[0])


def _minimal_serialization_check(binding: dict[str, Any], spec: dict[str, Any]) -> None:
    payload = {
        "stage": "P3_V3_FAILURE_PROBE_20261002",
        "status": "CHECK_ONLY_READY",
        "source_binding": binding,
        "first_spec": spec,
        "max_steps": MAX_STEPS,
        "generator_failure_count": 0,
        "parameterizer_failure_count": 0,
    }
    json.dumps(payload, allow_nan=False, sort_keys=True)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


def _output_contract_preflight(binding: dict[str, Any], spec: dict[str, Any]) -> None:
    """Write/read one exclusive nested JSON payload without touching an env."""
    if OUTPUT_CONTRACT_PREFLIGHT.exists():
        raise RuntimeError(f"output-contract preflight already exists: {OUTPUT_CONTRACT_PREFLIGHT}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = _jsonable({
        "stage": "P3_V3_FAILURE_PROBE_OUTPUT_CONTRACT_20261002",
        "source_checkpoint_sha256": binding["checkpoint_sha256"],
        "first_spec": spec,
        "representative": {
            "vector": np.asarray((1.25, -2.5, 3.75), dtype=float),
            "nested": {"times": np.asarray((0.0, 0.2, 1.0), dtype=float)},
        },
        "environment_steps": 0,
    })
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with OUTPUT_CONTRACT_PREFLIGHT.open("x", encoding="utf-8") as handle:
        handle.write(serialized)
    roundtrip = json.loads(OUTPUT_CONTRACT_PREFLIGHT.read_text(encoding="utf-8"))
    if roundtrip != payload:
        raise RuntimeError("output-contract preflight JSON roundtrip mismatch")
    if roundtrip.get("source_checkpoint_sha256") != binding["checkpoint_sha256"]:
        raise RuntimeError("output-contract preflight source SHA mismatch")
    if roundtrip.get("environment_steps") != 0:
        raise RuntimeError("output-contract preflight changed environment step count")
    print(json.dumps({
        "status": "OUTPUT_CONTRACT_PREFLIGHT_OK",
        "path": str(OUTPUT_CONTRACT_PREFLIGHT),
        "source_checkpoint_sha256": binding["checkpoint_sha256"],
        "environment_steps": 0,
    }, sort_keys=True), flush=True)


def _run_probe(binding: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    if OUTPUT_JSON.exists():
        raise RuntimeError(f"probe output already exists: {OUTPUT_JSON}")
    if OUTPUT_DIR.exists():
        if not OUTPUT_DIR.is_dir():
            raise RuntimeError(f"probe output path is not a directory: {OUTPUT_DIR}")

    generator_failures: deque[dict[str, Any]] = deque(maxlen=100)
    parameterizer_failures: deque[dict[str, Any]] = deque(maxlen=100)
    original_generator = TrajectoryGenerator._v3_continuous_inward_landing_points
    original_parameterize = TrajectoryTimeParameterizer.parameterize

    def wrapped_generator(self, uav, lz, airspace, now, *args, **kwargs):
        try:
            return original_generator(self, uav, lz, airspace, now, *args, **kwargs)
        except Exception as exc:
            generator_failures.append({
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
                "caller_stack": traceback.format_stack(),
                "now": _compact(now),
                "uav": _uav_state(uav),
                "ring": _ring_state(airspace, uav),
                "lz": _lz_state(lz),
                "kwargs": _compact(kwargs),
            })
            raise

    def wrapped_parameterize(self, path, *args, **kwargs):
        try:
            return original_parameterize(self, path, *args, **kwargs)
        except Exception as exc:
            constraints = kwargs.get("constraints")
            parameterizer_failures.append({
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
                "path_points": _compact(getattr(path, "points", None)),
                "start_time": _compact(kwargs.get("start_time")),
                "initial_velocity": _compact(kwargs.get("initial_velocity")),
                "initial_acceleration": _compact(kwargs.get("initial_acceleration")),
                "target_horizontal_speed": _compact(kwargs.get("target_horizontal_speed")),
                "constraints": _constraints_state(constraints),
            })
            raise

    TrajectoryGenerator._v3_continuous_inward_landing_points = wrapped_generator
    TrajectoryTimeParameterizer.parameterize = wrapped_parameterize
    print(
        json.dumps({
            "status": "PROBE_START",
            "first_spec": spec,
            "max_steps": MAX_STEPS,
            "source_checkpoint_sha256": binding["checkpoint_sha256"],
        }, sort_keys=True),
        flush=True,
    )
    arm: dict[str, Any]
    try:
        base = evaluator._base_config()
        arm = evaluator._run_arm(spec, base, max_steps=MAX_STEPS)
    finally:
        TrajectoryGenerator._v3_continuous_inward_landing_points = original_generator
        TrajectoryTimeParameterizer.parameterize = original_parameterize

    compact_arm = {
        "status": arm.get("status"),
        "stop_reason": arm.get("stop_reason"),
        "error": arm.get("error"),
        "progress_length": len(arm.get("progress", [])),
        "progress": arm.get("progress", []),
        "generated_requests": arm.get("generated_requests"),
        "request_count_contract": arm.get("request_count_contract"),
        "safety": arm.get("safety"),
        "validity": arm.get("validity"),
    }
    payload = {
        "stage": "P3_V3_FAILURE_PROBE_20261002",
        "status": (
            "FAILURE_OBSERVED"
            if generator_failures or parameterizer_failures or arm.get("error") is not None
            else "NO_FAILURE_WITHIN_MAX_STEPS"
        ),
        "source_binding": binding,
        "first_spec": spec,
        "max_steps": MAX_STEPS,
        "arm": compact_arm,
        "generator_failures": list(generator_failures),
        "parameterizer_failures": list(parameterizer_failures),
        "failure_event_limits": {
            "generator_failures_max": 100,
            "parameterizer_failures_max": 100,
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSON.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "status": payload["status"],
        "output": str(OUTPUT_JSON),
        "arm_status": arm.get("status"),
        "arm_error": arm.get("error"),
        "progress_length": len(arm.get("progress", [])),
        "generator_failures": len(generator_failures),
        "parameterizer_failures": len(parameterizer_failures),
    }, sort_keys=True), flush=True)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output-contract-preflight", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if sum(bool(value) for value in (args.check_only, args.output_contract_preflight, args.run)) != 1:
        parser.error("choose exactly one of --check-only, --output-contract-preflight, or --run")
    try:
        binding = _source_binding()
        spec = _validate_first_spec()
        if args.check_only:
            _minimal_serialization_check(binding, spec)
            return 0
        if args.output_contract_preflight:
            _output_contract_preflight(binding, spec)
            return 0
        _run_probe(binding, spec)
        return 0
    except Exception as exc:
        print(json.dumps({
            "stage": "P3_V3_FAILURE_PROBE_20261002",
            "status": "ERROR",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }, indent=2, sort_keys=True), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
