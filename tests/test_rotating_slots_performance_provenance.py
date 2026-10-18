"""Bounded P0 checks; catalogue IDs describe scope, not completed qualification."""
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

from terminal_marl.core.rotating_slots.performance.contract import (
    ContractInconsistent, preflight,
)
from terminal_marl.core.rotating_slots.performance.provenance import (
    AMENDMENT_PATH, PHYSICAL_PATH, R0_COMMIT, R0_POLICY_ID, R0_SOURCE_SHA256,
    ProvenanceError, digest, export_reference, implementation_fingerprint,
    require_same_semantics, run_reference, semantic_fingerprint,
    verify_deployment, verify_reference,
)

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "rotating_slots_performance_contract_v1_0"
REFERENCE = ROOT / "artifacts/rotating_slots/performance_reference_r0"


@pytest.fixture(scope="module")
def reference():
    export_reference(ROOT, REFERENCE)
    return REFERENCE


@pytest.fixture
def copied_source(reference, tmp_path):
    source = tmp_path / "source"
    shutil.copytree(reference, source)
    return source


def test_a01_r0_export_exact_git_bytes_and_isolated_child(reference, monkeypatch):
    manifest = verify_reference(ROOT, reference)
    assert manifest["source_sha256"] == R0_SOURCE_SHA256
    assert manifest["reference_commit"] == R0_COMMIT
    from terminal_marl.core.rotating_slots import world
    monkeypatch.setattr(world, "_performance_parent_cache_sentinel", True, raising=False)
    monkeypatch.setenv("PYTHONPATH", str(ROOT))
    result = run_reference(reference, code=(
        "import sys\nfrom terminal_marl.core.rotating_slots import world\n"
        "w=world.RotatingSlotWorld()\nw.advance(.4)\n"
        "query_result={'time':w.time,'parent_cache_present':hasattr(world,'_performance_parent_cache_sentinel'),"
        "'performance_loaded':any(k.startswith('terminal_marl.core.rotating_slots.performance') for k in sys.modules)}"
    ))
    assert result["policy_id"] == R0_POLICY_ID
    assert result["query_result"] == {"time": .4, "parent_cache_present": False, "performance_loaded": False}
    assert result["pid"] != __import__("os").getpid()
    assert all(Path(p).is_relative_to(reference.resolve()) for p in result["imported_module_paths"].values())


def test_a01_mixed_import_and_changed_reference_rejected(reference, copied_source):
    with pytest.raises(ProvenanceError, match="REFERENCE_IMPORT_OUTSIDE_ROOT"):
        run_reference(reference, code=(
            "import sys,types\nm=types.ModuleType('terminal_marl.injected')\n"
            f"m.__file__={str(ROOT / 'terminal_marl/__init__.py')!r}\n"
            "sys.modules['terminal_marl.injected']=m"
        ))
    target = copied_source / "terminal_marl/core/rotating_slots/world.py"
    target.write_bytes(target.read_bytes() + b"\n# changed\n")
    with pytest.raises(ProvenanceError, match="REFERENCE_BYTES_MISMATCH"):
        run_reference(copied_source)


def test_a02_policy_and_amendment_separate_from_config(copied_source):
    original = semantic_fingerprint(copied_source)
    legacy = semantic_fingerprint(copied_source, policy_id="legacy_f67")
    assert original["config_hash"] == legacy["config_hash"]
    assert original["effective_rules_hash"] != legacy["effective_rules_hash"]
    with pytest.raises(ProvenanceError, match="SEMANTIC_DRIFT"):
        require_same_semantics(original, legacy)
    amendment = copied_source / AMENDMENT_PATH
    amendment.write_text(amendment.read_text(encoding="utf8") + "\nChanged admission rule\n", encoding="utf8")
    with pytest.raises(ContractInconsistent, match="CONTRACT_INCONSISTENT"):
        preflight(PACKAGE, copied_source)


def test_a03_recursive_helpers_wrappers_native_and_imports(tmp_path):
    for relative in ("terminal_marl/core/rotating_slots/performance/future/helper.py",
                     "scripts/server/future_wrapper.py", "terminal_marl/future/kernel.pyd"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"first")
    dependency = tmp_path / "venv/Lib/site-packages/external.py"
    dependency.parent.mkdir(parents=True)
    dependency.write_bytes(b"dependency")
    wrapper = tmp_path / "caller.py"
    wrapper.write_bytes(b"wrapper")
    modules = {"external": SimpleNamespace(__file__=str(dependency)),
               "arbitrary_wrapper_name": SimpleNamespace(__file__=str(wrapper))}
    before = implementation_fingerprint(tmp_path, modules=modules)
    assert len(before["actual_source_files_sha256"]) == 4
    for relative in list(before["actual_source_files_sha256"]):
        path = tmp_path / relative
        original = path.read_bytes()
        path.write_bytes(original + b"changed")
        assert implementation_fingerprint(tmp_path, modules=modules)["implementation_sha256"] != before["implementation_sha256"]
        path.write_bytes(original)
    with pytest.raises(ProvenanceError, match="SOURCE_OUTSIDE_ROOT"):
        implementation_fingerprint(tmp_path, modules={"terminal_marl.bad": SimpleNamespace(__file__=__file__)})


@pytest.mark.parametrize("old,new", [
    ("hard_separation_m: 80.0", "hard_separation_m: 79.0"),
    ("physics_sample_dt_s: 0.2", "physics_sample_dt_s: 0.3"),
    ("trigger_absolute_soc: 0.35", "trigger_absolute_soc: 0.34"),
    ("terminal_visit_peak_duration_budget_s: 120.0", "terminal_visit_peak_duration_budget_s: 121.0"),
    ("induced_root_bisections: 80", "induced_root_bisections: 79"),
])
def test_a04_physical_and_numeric_yaml_mutations_rejected(copied_source, old, new):
    target = copied_source / PHYSICAL_PATH
    text = target.read_text(encoding="utf8")
    assert old in text
    target.write_text(text.replace(old, new, 1), encoding="utf8")
    with pytest.raises(ContractInconsistent, match="CONTRACT_INCONSISTENT"):
        preflight(PACKAGE, copied_source)


@pytest.mark.parametrize("filename,old,new", [
    ("energy.py", "bisections=80", "bisections=79"),
    ("energy.py", "range(80)", "range(79)"),
    ("energy.py", "depth>=12", "depth>=11"),
    ("energy.py", "depth>=26", "depth>=25"),
    ("energy.py", "1e-5,2e-10", "1e-4,2e-10"),
    ("energy.py", "max(1.,rhs)*1e-12", "max(1.,rhs)*1e-11"),
    ("energy.py", "np.maximum(1.,rhs)*1e-12", "np.maximum(1.,rhs)*1e-11"),
    ("safety.py", "max_depth=32", "max_depth=31"),
    ("safety.py", "depth>=24", "depth>=23"),
    ("safety.py", "abs(a.end_time-b.start_time)>1e-9", "abs(a.end_time-b.start_time)>1e-8"),
    ("safety.py", "(('p',1e-7),('v',1e-7),('a',1e-7))", "(('p',1e-6),('v',1e-7),('a',1e-7))"),
    ("safety.py", "_close(capture.p,(p[0],p[1],15.),1e-6)", "_close(capture.p,(p[0],p[1],15.),1e-5)"),
    ("trajectory.py", "range(185,301,5)", "range(185,301,10)"),
    ("world.py", "REFERENCE_PHASE_BINS = 24", "REFERENCE_PHASE_BINS = 23"),
    ("world.py", "physics_dt=.2", "physics_dt=.3"),
    ("world.py", "soc <= .35", "soc <= .34"),
    ("world.py", "policy_id=POLICY_ID)", "policy_id='legacy_f67')"),
])
def test_a04_actual_hardcoded_policy_mutations_rejected(copied_source, filename, old, new):
    target = copied_source / "terminal_marl/core/rotating_slots" / filename
    text = target.read_text(encoding="utf8")
    assert old in text
    target.write_text(text.replace(old, new, 1), encoding="utf8")
    with pytest.raises(ContractInconsistent, match="CONTRACT_INCONSISTENT"):
        preflight(PACKAGE, copied_source)


def test_a04_implementation_only_helper_does_not_change_semantics(copied_source):
    before = semantic_fingerprint(copied_source)
    helper = copied_source / "terminal_marl/core/rotating_slots/future_helper.py"
    helper.write_text("VALUE = 123\n", encoding="utf8")
    require_same_semantics(before, semantic_fingerprint(copied_source))
    preflight(PACKAGE, copied_source)


@pytest.mark.parametrize("filename", [
    "rotating_slots_simulation_performance_contract_v1_0.md",
    "performance_acceptance_config_v1_0.yaml", "performance_acceptance_catalog_v1_0.json",
])
def test_preflight_rejects_normative_package_discrepancy(copied_source, tmp_path, filename):
    package = tmp_path / "package"
    shutil.copytree(PACKAGE, package)
    target = package / filename
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ContractInconsistent, match="CONTRACT_INCONSISTENT"):
        preflight(package, copied_source)


def test_a05_preflight_and_reference_leave_history_read_only(copied_source):
    history = copied_source / "results/legacy60/results.json"
    history.parent.mkdir(parents=True)
    history.write_bytes(b'{"code_commit":"f67b5c8","status":"PASS"}')
    before = sha256(history.read_bytes()).hexdigest()
    package_before = {p.name: sha256(p.read_bytes()).hexdigest() for p in PACKAGE.iterdir() if p.is_file()}
    preflight(PACKAGE, copied_source)
    semantic_fingerprint(copied_source)
    assert sha256(history.read_bytes()).hexdigest() == before
    assert {p.name: sha256(p.read_bytes()).hexdigest() for p in PACKAGE.iterdir() if p.is_file()} == package_before


def test_a06_deployment_label_requires_verified_actual_bytes(monkeypatch):
    monkeypatch.setenv("TERMINAL_CODE_COMMIT", R0_COMMIT)
    with pytest.raises(ProvenanceError, match="DEPLOYMENT_MANIFEST_REQUIRED"):
        verify_deployment(ROOT, None)
    expected = dict(implementation_fingerprint(ROOT), implementation_commit=R0_COMMIT)
    assert verify_deployment(ROOT, expected)["implementation_commit"] == R0_COMMIT
    with pytest.raises(ProvenanceError, match="DEPLOYMENT_COMMIT_MISMATCH"):
        verify_deployment(ROOT, expected, supplied_commit="f67b5c8462433ba701191c493a37e4a8e472334e")
    expected["actual_source_files_sha256"] = dict(expected["actual_source_files_sha256"])
    key = next(iter(expected["actual_source_files_sha256"]))
    expected["actual_source_files_sha256"][key] = "0" * 64
    expected["implementation_sha256"] = digest(expected["actual_source_files_sha256"])
    with pytest.raises(ProvenanceError, match="DEPLOYMENT_BYTES_MISMATCH"):
        verify_deployment(ROOT, expected)


def test_r0_and_current_eol_provenance_remains_explicit(reference):
    reference_semantics, current = semantic_fingerprint(reference), semantic_fingerprint(ROOT)
    assert reference_semantics["config_hash"] == current["config_hash"]
    assert reference_semantics["physical_config_bytes_sha256"] == sha256((reference / PHYSICAL_PATH).read_bytes()).hexdigest()
    assert current["physical_config_bytes_sha256"] == sha256((ROOT / PHYSICAL_PATH).read_bytes()).hexdigest()
    require_same_semantics(reference_semantics, current)
    result = preflight(PACKAGE, ROOT, reference_semantics)
    assert result["performance_status"] == "NOT_QUALIFIED"
    assert result["human_status"] == "HUMAN_PENDING"
