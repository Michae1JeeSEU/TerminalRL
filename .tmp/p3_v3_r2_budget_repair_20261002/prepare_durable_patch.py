"""Save the exact tested change and its canonical regression tests for later apply."""
import difflib
import hashlib
from pathlib import Path

workspace = Path(__file__).resolve().parents[2]
folder = Path(__file__).resolve().parent
source = folder / "test_ring_budget_infeasibility.py"
test = source.read_text(encoding="utf-8")
for line in ("import importlib.util\n", "import os\n", "from pathlib import Path\n", "import sys\n"):
    test = test.replace(line, "")
begin = test.index("@pytest.fixture\ndef generator_class():")
end = test.index("\n\ndef components(", begin)
test = test[:begin] + "@pytest.fixture\ndef generator_class():\n    return TrajectoryGenerator\n" + test[end:]
(folder / "test_canonical.py").write_text(test, encoding="utf-8")
patch = (folder / "repair.patch").read_bytes().decode("utf-8")
new_path = "tests/test_v3_ring_budget_infeasibility.py"
patch += "".join(difflib.unified_diff(
    [], test.splitlines(keepends=True), fromfile="/dev/null", tofile="b/" + new_path,
))
target = workspace / "docs/evidence/v4_2/patches/v3_r2_ring_budget_20261002.patch"
target.parent.mkdir(parents=True, exist_ok=True)
target.write_bytes(patch.encode("utf-8"))
print(target.relative_to(workspace))
print("patch_sha256=" + hashlib.sha256(target.read_bytes()).hexdigest())
print("canonical_test_sha256=" + hashlib.sha256((folder / "test_canonical.py").read_bytes()).hexdigest())
