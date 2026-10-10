from __future__ import annotations

import hashlib
import importlib.util
import json
import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from terminal_marl.core.config import RingRole
from terminal_marl.core.dynamic_airspace import DynamicAirspace
from terminal_marl.core.trajectory_generator import TrajectoryGenerator


TEST_PATH = ROOT / "tests" / "test_trajectory_generator_v3.py"
OUTPUT = Path(
    "/home/user/training_outputs/terminal/trajectory_v3_repair_20261002/"
    "batch_reference_before.json"
)
SOURCE_PATHS = (
    ROOT / "terminal_marl" / "core" / "trajectory_generator.py",
    ROOT / "terminal_marl" / "core" / "trajectory_time_parameterization.py",
    ROOT / "terminal_marl" / "core" / "vehicle_dynamics.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_fixture_module():
    spec = importlib.util.spec_from_file_location("test_trajectory_generator_v3_reference", TEST_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load fixture module: {TEST_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _jsonable(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "value") and isinstance(value.value, str):
        return value.value
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _plan_record(name: str, plan, generator) -> dict:
    metadata = _jsonable(dict(plan.metadata))
    departure_keys = {
        key: value
        for key, value in metadata.items()
        if any(token in key for token in ("departure", "scheduled", "timing", "speed", "merge"))
    }
    return {
        "mode": name,
        "plan_id": str(plan.plan_id),
        "schema": metadata.get("trajectory_schema_version"),
        "trajectory_hash": str(plan.trajectory_hash),
        "start_time": float(plan.start_time),
        "end_time": float(plan.end_time),
        "duration": float(plan.end_time - plan.start_time),
        "metadata": metadata,
        "departure_metadata": departure_keys,
        "points": [
            {
                "t": float(point.t),
                "position": _jsonable(point.position),
                "velocity": _jsonable(point.velocity),
                "acceleration": _jsonable(point.acceleration),
            }
            for point in plan.points
        ],
        "segments": [
            {
                "start_time": float(segment.start_time),
                "end_time": float(segment.end_time),
                "duration": float(segment.duration),
                "bounds": _jsonable(segment.bounds()),
            }
            for segment in plan.polynomial_segments
        ],
        "validate_kinematics_errors": _jsonable(generator.validate_kinematics(plan)),
    }


def main(output: Path = OUTPUT) -> int:
    if output.exists():
        raise RuntimeError(f"refusing to overwrite existing artifact: {output}")
    before = {str(path.relative_to(ROOT)): _sha256(path) for path in SOURCE_PATHS}
    fixture = _load_fixture_module()
    config = fixture._config()
    airspace = DynamicAirspace(config)
    generator = TrajectoryGenerator(config)
    outer = airspace.actual_ring(RingRole.OUTER)
    inbound_uav = SimpleNamespace(
        uid=2,
        position=fixture.np.asarray((800.0, 0.0, 110.0)),
        velocity=fixture.np.asarray((-5.0, 0.0, 0.0)),
        acceleration=fixture.np.zeros(3),
        speed=5.0,
        ring_progress=None,
        ring_role=None,
        committed_plan=None,
    )
    inbound = generator.inbound(inbound_uav, SimpleNamespace(angle=0.0), airspace, 0.0)
    holder = fixture._uav(airspace, RingRole.OUTER, uid=3)
    transfer = generator.ring_transfer(
        holder, RingRole.OUTER, RingRole.MIDDLE, fixture._reservation(120.0, 1.0), airspace, 0.0,
    )
    emergency = generator.emergency_landing(holder, fixture._lz(), airspace, 0.0)
    brake_config = fixture._config(final_approach_geometry_mode="brake_bezier_v1")
    brake_airspace = DynamicAirspace(brake_config)
    brake_generator = TrajectoryGenerator(brake_config)
    brake_uav = fixture._uav(brake_airspace, RingRole.INNER, speed=10.0)
    brake = brake_generator.final_approach(
        brake_uav, fixture._lz(), fixture._reservation(0.0), 0.0, airspace=brake_airspace,
    )
    plans = (
        ("inbound", inbound, generator),
        ("ring_transfer", transfer, generator),
        ("emergency_landing", emergency, generator),
        ("brake_bezier_final_approach", brake, brake_generator),
    )
    records = [_plan_record(name, plan, plan_generator) for name, plan, plan_generator in plans]
    after = {str(path.relative_to(ROOT)): _sha256(path) for path in SOURCE_PATHS}
    if before != after:
        raise RuntimeError(f"caller source hashes changed during execution: before={before} after={after}")
    payload = {
        "status": "REFERENCE_CAPTURED",
        "fixture_source": str(TEST_PATH),
        "source_hashes_before": before,
        "source_hashes_after": after,
        "plans": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "plan_count": len(records),
        "validate_errors": {record["mode"]: record["validate_kinematics_errors"] for record in records},
        "source_hashes": before,
        "artifact": str(output),
        "artifact_sha256": _sha256(output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "--outputpath", dest="output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    started = time.monotonic()
    code = main(args.output)
    print(json.dumps({"runtime_seconds": time.monotonic() - started}, sort_keys=True))
    raise SystemExit(code)
