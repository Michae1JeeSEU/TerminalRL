"""Inject the one changed method only in a separate test process, never R2."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

workspace = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(workspace))
from terminal_marl.core.trajectory_generator import TrajectoryGenerator

folder = Path(__file__).resolve().parent
binding = json.loads((workspace / ".tmp/p3_v3_r2_matrix_source_binding_20261002.json").read_text())

def source_hashes():
    return {name: hashlib.sha256((workspace / name).read_bytes()).hexdigest()
            for name in binding["source_paths"]}

before = source_hashes()
if before != binding["local_sha256"]:
    raise RuntimeError("running R2 sources differ from frozen binding")
candidate_path = folder / "trajectory_generator_candidate.py"
manifest = json.loads((folder / "candidate_manifest.json").read_text())
if hashlib.sha256(candidate_path.read_bytes()).hexdigest() != manifest["candidate_sha256"]:
    raise RuntimeError("candidate hash mismatch")
name = "terminal_marl.core._r2_ring_budget_regression_candidate"
spec = importlib.util.spec_from_file_location(name, candidate_path)
candidate = importlib.util.module_from_spec(spec)
sys.modules[name] = candidate
spec.loader.exec_module(candidate)
TrajectoryGenerator._v3_ring_transition_parameters = candidate.TrajectoryGenerator._v3_ring_transition_parameters
if Path(TrajectoryGenerator._v3_ring_transition_parameters.__code__.co_filename) != candidate_path:
    raise RuntimeError("candidate method was not installed in the test process")
print(json.dumps({"candidate_method_file": str(candidate_path), "source_matches_before": len(before)}), flush=True)
code = pytest.main([
    "tests/test_trajectory_generator_v3.py",
    "tests/test_v3_energy_reachability_failures.py",
    "tests/test_energy_reachability.py",
    "tests/test_emergency_energy_v2.py",
    "tests/test_warm_start_energy_reachability.py",
    str(folder / "test_canonical.py"), "-q",
])
after = source_hashes()
if before != after:
    raise RuntimeError("R2 sources changed during isolated tests")
print(json.dumps({"pytest_exit_code": int(code), "source_matches_after": len(after),
                  "running_R2_sources_unchanged": before == after}), flush=True)
raise SystemExit(int(code))
