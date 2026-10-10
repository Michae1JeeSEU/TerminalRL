"""Create an isolated, single-change generator candidate; leave R2 sources intact."""
from pathlib import Path
import difflib
import hashlib
import json

workspace = Path(__file__).resolve().parents[2]
folder = Path(__file__).resolve().parent
relative = "terminal_marl/core/trajectory_generator.py"
source = workspace / relative
original = source.read_bytes().decode("utf-8")
old = 'raise RuntimeError("ring_hold_tangential_budget_unavailable")'
new = 'raise TrajectoryInfeasibleError("ring_hold_tangential_budget_unavailable")'
if original.count(old) != 1:
    raise RuntimeError("expected exactly one frozen ring-budget rejection")
candidate = original.replace(old, new)
(folder / "trajectory_generator_candidate.py").write_bytes(candidate.encode("utf-8"))
patch = "".join(difflib.unified_diff(
    original.splitlines(keepends=True), candidate.splitlines(keepends=True),
    fromfile="a/" + relative, tofile="b/" + relative,
))
(folder / "repair.patch").write_bytes(patch.encode("utf-8"))
payload = {
    "purpose": "isolated candidate, not applied to running R2",
    "source": relative,
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "candidate_sha256": hashlib.sha256((folder / "trajectory_generator_candidate.py").read_bytes()).hexdigest(),
    "patch_sha256": hashlib.sha256((folder / "repair.patch").read_bytes()).hexdigest(),
}
(folder / "candidate_manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
