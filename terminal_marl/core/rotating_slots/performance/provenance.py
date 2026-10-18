"""Byte-verified deployment identities and a separate, fixed R0 interpreter.

All Git operations here are explicit local export operations. Deployment checks
do not invoke Git and never infer identity from TERMINAL_CODE_COMMIT alone.
"""
from __future__ import annotations

import ast
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

R0_COMMIT = "1c5afd37c891a0ed4776f64367fc7264db3c1c54"
R0_POLICY_ID = "upstream_capacity_only_pad_arrival_v1_20261018"
PHYSICAL_PATH = "new10.16/fixed_terminal_rotating_slots_stage1_config_v1_0.yaml"
AMENDMENT_PATH = "docs/rotating_slots_stage1/amendment_20261018.md"
REFERENCE_PATHS = ("terminal_marl/__init__.py", "terminal_marl/core/__init__.py",
                   "terminal_marl/core/rotating_slots", "new10.16", AMENDMENT_PATH)
REFERENCE_MANIFEST = ".reference_manifest.json"
# SHA256 of the canonical path -> SHA256 map generated from fixed Git blobs.
# This value is filled from the R0 Git object, never from the working tree.
R0_SOURCE_SHA256 = "cd76b4c14b0a48d6d2a71ec7c63ce77010e6ea019b545807635b926eb2984891"
SOURCE_SUFFIXES = {".py", ".pyx", ".pxd", ".pyd", ".so", ".dll", ".c", ".cc", ".cpp", ".h"}


class ProvenanceError(ValueError):
    """A declared identity did not match observed source bytes or semantics."""


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _relative(root, path):
    path = Path(path).resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        raise ProvenanceError(f"SOURCE_OUTSIDE_ROOT: {path}") from None


def _source_files(root, extra_paths=(), modules=None):
    """Conservative tree coverage plus every observed project import/entrypoint.

    Recursive discovery intentionally includes future helpers, wrappers, workers
    and native products, rather than enumerating today's module names.
    """
    paths = set()
    for directory in (root / "terminal_marl", root / "scripts"):
        if directory.exists():
            paths.update(p.resolve() for p in directory.rglob("*")
                         if p.is_file() and "__pycache__" not in p.parts
                         and p.suffix.lower() in SOURCE_SUFFIXES)
    for directory in (root / "new10.16", root / "rotating_slots_performance_contract_v1_0"):
        if directory.exists():
            paths.update(p.resolve() for p in directory.rglob("*") if p.is_file()
                         and "__pycache__" not in p.parts and p.suffix not in {".pyc", ".pyo"})
    if (root / AMENDMENT_PATH).is_file():
        paths.add((root / AMENDMENT_PATH).resolve())
    for path in extra_paths:
        path = Path(path)
        path = path if path.is_absolute() else root / path
        if not path.is_file():
            raise ProvenanceError(f"SOURCE_MISSING: {path}")
        paths.add(path.resolve())
    imported = {}
    for name, module in (sys.modules if modules is None else modules).copy().items():
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if name == "terminal_marl" or name.startswith("terminal_marl."):
            _relative(root, path)  # Reject a mixed checkout even when root has a copy.
            imported[name] = str(path)
            paths.add(path)
        elif path.is_relative_to(root) and not any(
                part.lower() in {"venv", ".venv", "env", "site-packages", "__pycache__"}
                for part in path.relative_to(root).parts):
            # Includes a supplied CLI/test wrapper loaded under an arbitrary name.
            paths.add(path)
            imported[name] = str(path)
    hashes = {_relative(root, path): sha256(path.read_bytes()).hexdigest()
              for path in sorted(paths)}
    return hashes, imported


def implementation_fingerprint(root, extra_paths=(), modules=None):
    """Return actual byte identity; pass CLI/test/native entrypoints in extra_paths.

    Invoke again at completion to include modules imported after startup. Paths
    outside the project must be supplied as separate dependency provenance, not
    silently assigned this checkout's identity.
    """
    root = Path(root).resolve()
    hashes, imported = _source_files(root, extra_paths, modules)
    return {"schema": "ft_rs_implementation_manifest_v1_0",
            "actual_source_files_sha256": hashes,
            "implementation_sha256": digest(hashes),
            "imported_module_paths": imported,
            "extra_paths": [_relative(root, root / Path(p)) for p in extra_paths]}


def verify_deployment(root, expected_manifest, supplied_commit=None):
    """Verify a trusted launcher manifest and optional commit label without Git.

    A SHA-shaped label is evidence only when the caller supplies the expected
    manifest generated locally for that deployment. This is a consistency check,
    not cryptographic authentication of an untrusted manifest.
    """
    if not expected_manifest or not expected_manifest.get("implementation_commit"):
        raise ProvenanceError("DEPLOYMENT_MANIFEST_REQUIRED")
    label = supplied_commit or os.environ.get("TERMINAL_CODE_COMMIT")
    if label and label != expected_manifest["implementation_commit"]:
        raise ProvenanceError("DEPLOYMENT_COMMIT_MISMATCH")
    actual = implementation_fingerprint(root, expected_manifest.get("extra_paths", ()))
    expected = expected_manifest.get("actual_source_files_sha256")
    if not expected or digest(expected) != expected_manifest.get("implementation_sha256"):
        raise ProvenanceError("DEPLOYMENT_MANIFEST_INVALID")
    if actual["actual_source_files_sha256"] != expected:
        raise ProvenanceError("DEPLOYMENT_BYTES_MISMATCH")
    return dict(actual, implementation_commit=expected_manifest["implementation_commit"],
                supplied_code_commit=label, manifest_sha256=digest(expected_manifest))


def _git_export(repo_root):
    """Read fixed Git contents in one local archive and verify each blob id."""
    command = ["git", "-c", "core.autocrlf=false", "archive", "--format=tar", R0_COMMIT, *REFERENCE_PATHS]
    raw = subprocess.run(command, cwd=repo_root, check=True, capture_output=True).stdout
    entries = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
        for member in archive.getmembers():
            if member.isdir():
                continue
            if not member.isfile() or Path(member.name).is_absolute() or ".." in Path(member.name).parts:
                raise ProvenanceError("REFERENCE_UNSAFE_ARCHIVE_MEMBER")
            entries[member.name] = archive.extractfile(member).read()
    tree = subprocess.run(["git", "ls-tree", "-r", R0_COMMIT, "--", *REFERENCE_PATHS],
                          cwd=repo_root, check=True, capture_output=True, text=True).stdout
    blobs = {}
    for line in tree.splitlines():
        header, path = line.split("\t", 1)
        mode, kind, blob = header.split()
        if kind != "blob" or mode not in {"100644", "100755"}:
            raise ProvenanceError("REFERENCE_UNSUPPORTED_GIT_ENTRY")
        blobs[path] = blob
    if set(entries) != set(blobs):
        raise ProvenanceError("REFERENCE_TREE_MISMATCH")
    from hashlib import sha1
    for path, data in entries.items():
        if sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() != blobs[path]:
            raise ProvenanceError(f"REFERENCE_GIT_BLOB_MISMATCH: {path}")
    hashes = {path: sha256(data).hexdigest() for path, data in sorted(entries.items())}
    manifest = {"schema": "ft_rs_r0_source_export_v1_0", "reference_commit": R0_COMMIT,
                "policy_id": R0_POLICY_ID, "actual_source_files_sha256": hashes,
                "source_sha256": digest(hashes), "git_blobs": blobs}
    return entries, manifest


def _verify_export(reference_root, expected_manifest=None):
    root = Path(reference_root).resolve()
    manifest = json.loads((root / REFERENCE_MANIFEST).read_text(encoding="utf8"))
    hashes = manifest.get("actual_source_files_sha256", {})
    if (manifest.get("reference_commit") != R0_COMMIT or manifest.get("policy_id") != R0_POLICY_ID
            or digest(hashes) != R0_SOURCE_SHA256 or manifest.get("source_sha256") != R0_SOURCE_SHA256):
        raise ProvenanceError("REFERENCE_IDENTITY_MISMATCH")
    if expected_manifest is not None and manifest != expected_manifest:
        raise ProvenanceError("REFERENCE_MANIFEST_MISMATCH")
    found = {p.relative_to(root).as_posix(): sha256(p.read_bytes()).hexdigest()
             for p in root.rglob("*") if p.is_file() and p.name != REFERENCE_MANIFEST}
    if found != hashes:
        raise ProvenanceError("REFERENCE_BYTES_MISMATCH")
    return manifest


def export_reference(repo_root, destination):
    """Export exact R0 locally. Never overwrite an existing export or source tree."""
    destination = Path(destination).resolve()
    entries, manifest = _git_export(Path(repo_root).resolve())
    if manifest["source_sha256"] != R0_SOURCE_SHA256:
        raise ProvenanceError("REFERENCE_FIXED_GIT_DIGEST_MISMATCH")
    if destination.exists():
        return _verify_export(destination, manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".r0-export-") as temporary:
        temporary = Path(temporary)
        for path, data in entries.items():
            target = temporary / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        (temporary / REFERENCE_MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf8")
        temporary.rename(destination)
    return _verify_export(destination, manifest)


def verify_reference(repo_root, reference_root):
    """Cross-check export bytes with fixed local Git contents, without writing."""
    _, manifest = _git_export(Path(repo_root).resolve())
    return _verify_export(reference_root, manifest)


_REFERENCE_CHILD = r'''
import contextlib, hashlib, importlib, io, json, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
request = json.loads(sys.argv[2])
sys.dont_write_bytecode = True
sys.path.insert(0, str(root))
manifest = json.loads((root / ".reference_manifest.json").read_text(encoding="utf8"))
def verify():
    files = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in root.rglob("*") if p.is_file() and p.name != ".reference_manifest.json"}
    if files != manifest["actual_source_files_sha256"]:
        raise RuntimeError("REFERENCE_BYTES_MISMATCH")
verify()
captured = io.StringIO()
namespace = {"reference_root": root}
with contextlib.redirect_stdout(captured):
    for name in request["modules"]:
        importlib.import_module(name)
    if request["code"]:
        exec(request["code"], namespace)
from terminal_marl.core.rotating_slots.config import load_config
from terminal_marl.core.rotating_slots.world import POLICY_ID
config = load_config()
paths = {}
for name, module in list(sys.modules.items()):
    if name == "terminal_marl" or name.startswith("terminal_marl."):
        path = pathlib.Path(module.__file__).resolve()
        if not path.is_relative_to(root):
            raise RuntimeError("REFERENCE_IMPORT_OUTSIDE_ROOT: " + str(path))
        if path.relative_to(root).as_posix() not in manifest["actual_source_files_sha256"]:
            raise RuntimeError("REFERENCE_IMPORT_UNVERIFIED: " + str(path))
        paths[name] = str(path)
verify()
print(json.dumps({"reference_commit": manifest["reference_commit"], "policy_id": POLICY_ID,
                  "source_sha256": manifest["source_sha256"], "imported_module_paths": paths,
                  "config_hash": config.config_hash, "profile_hash": config.profile_hash,
                  "query_result": namespace.get("query_result"), "captured_stdout": captured.getvalue(),
                  "pid": __import__("os").getpid()}, allow_nan=False))
'''


def run_reference(reference_root, modules=("terminal_marl.core.rotating_slots.world",
                                         "terminal_marl.core.rotating_slots.driver"),
                  code=None, timeout=60, python=None):
    """Execute bounded query code in a fresh -I/-B R0 child; return JSON evidence.

    Query code writes JSON-compatible ``query_result``. It must not install a new
    kernel or mutate the export. Parent caches and PYTHONPATH are not inherited.
    This is an oracle process boundary, not a sandbox for hostile query code.
    """
    root = Path(reference_root).resolve()
    manifest = _verify_export(root)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    result = subprocess.run([str(python or sys.executable), "-I", "-B", "-c", _REFERENCE_CHILD,
                             str(root), json.dumps({"modules": list(modules), "code": code})],
                            cwd=root, env=env, capture_output=True, text=True, timeout=timeout)
    _verify_export(root, manifest)
    if result.returncode:
        raise ProvenanceError(f"REFERENCE_CHILD_FAILED: {result.returncode}\n{result.stderr}")
    evidence = json.loads(result.stdout)
    if evidence["reference_commit"] != R0_COMMIT or evidence["policy_id"] != R0_POLICY_ID:
        raise ProvenanceError("REFERENCE_CHILD_IDENTITY_MISMATCH")
    evidence["actual_source_files_sha256"] = manifest["actual_source_files_sha256"]
    evidence["manifest_sha256"] = digest(manifest)
    evidence["physical_config_bytes_sha256"] = manifest["actual_source_files_sha256"][PHYSICAL_PATH]
    return evidence


def _functions(root, filename, names):
    tree = ast.parse((root / "terminal_marl/core/rotating_slots" / filename).read_text(encoding="utf8"))
    result = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            # Locate class methods rather than similarly named facade methods.
            result[node.name] = node
    if set(result) != set(names):
        raise ProvenanceError(f"SEMANTIC_SOURCE_MISSING: {filename}")
    return result


def _default(function, name):
    defaults = dict(zip([a.arg for a in function.args.args][-len(function.args.defaults):],
                        function.args.defaults))
    if name not in defaults:
        raise ProvenanceError(f"UNRECOGNIZED_NUMERICAL_POLICY: default {function.name}.{name}")
    return ast.literal_eval(defaults[name])


def _limit(function, expression, operator):
    matches = [ast.literal_eval(node.comparators[0]) for node in ast.walk(function)
               if isinstance(node, ast.Compare) and len(node.ops) == 1
               and ast.unparse(node.left) == expression
               and type(node.ops[0]).__name__ == operator
               and isinstance(node.comparators[0], (ast.Constant, ast.UnaryOp))]
    if len(matches) != 1:
        raise ProvenanceError(f"UNRECOGNIZED_NUMERICAL_POLICY: {function.name}:{expression}:{operator}")
    return matches[0]


def _scaled_residual(function, scale_function):
    expressions = []
    for node in ast.walk(function):
        if isinstance(node, ast.Compare) and len(node.comparators) == 1:
            right = node.comparators[0]
            if (isinstance(right, ast.BinOp) and isinstance(right.op, ast.Mult)
                    and isinstance(right.left, ast.Call)
                    and ast.unparse(right.left.func) == scale_function
                    and isinstance(right.right, ast.Constant)):
                expressions.append(ast.dump(node, include_attributes=False))
    if len(expressions) != 1:
        raise ProvenanceError(f"UNRECOGNIZED_NUMERICAL_POLICY: {function.name} branch residual")
    return expressions[0]


def _metadata_tolerances(function):
    expected = {"SEGMENT_TIME_GAP", "C2_DISCONTINUITY", "SOURCE_STATE_MISMATCH",
                "TARGET_STATE_MISMATCH", "ATOMIC_DESCENT_REQUIRED", "CAPTURE_STATE_MISMATCH",
                "TOUCHDOWN_MISMATCH", "ENTRY_STATE_MISMATCH"}
    guards = {}
    for node in ast.walk(function):
        if isinstance(node, ast.If):
            for statement in node.body:
                if not isinstance(statement, ast.Return) or not isinstance(statement.value, ast.Call):
                    continue
                call = statement.value
                if (isinstance(call.func, ast.Name) and call.func.id == "ValidationResult" and len(call.args) >= 3
                        and isinstance(call.args[2], ast.Constant) and call.args[2].value in expected):
                    guards[call.args[2].value] = ast.dump(node.test, include_attributes=False)
    # ENTRY_STATE_MISMATCH uses a precomputed list of comparisons; include its
    # named append expression rather than hashing unrelated function literals.
    entry_matches = [ast.dump(n.args[0], include_attributes=False) for n in ast.walk(function)
                     if isinstance(n, ast.Call) and ast.unparse(n.func) == "matches.append"]
    eps = [ast.literal_eval(n.value) for n in ast.walk(function) if isinstance(n, ast.Assign)
           and any(isinstance(t, ast.Name) and t.id == "eps" for t in n.targets)]
    if set(guards) != expected or len(entry_matches) != 1 or len(eps) != 1:
        raise ProvenanceError("UNRECOGNIZED_NUMERICAL_POLICY: metadata tolerances")
    return {"guards": guards, "entry_matches": entry_matches[0], "algebraic_endpoint_eps": eps[0]}


def _numerical_projection(root):
    energy = _functions(root, "energy.py", ("_root", "_power_array", "_quadrature_cells", "integrate_energy", "_peak_measure"))
    safety = _functions(root, "safety.py", ("validate_plan", "validate_separation", "certified_minimum_distance"))
    world = _functions(root, "world.py", ("__init__", "_advance_energy", "_update_emergency"))
    vector_roots = [ast.literal_eval(n.iter.args[0]) for n in ast.walk(energy["_power_array"])
                    if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "_"
                    and isinstance(n.iter, ast.Call) and isinstance(n.iter.func, ast.Name)
                    and n.iter.func.id == "range" and len(n.iter.args) == 1]
    quadrature_tolerances = [ast.literal_eval(n.args[0]) for n in ast.walk(energy["_quadrature_cells"])
                            if isinstance(n, ast.Call) and ast.unparse(n.func) == "np.maximum"]
    relative_tolerances = [ast.literal_eval(n.args[1].left) for n in ast.walk(energy["_quadrature_cells"])
                           if isinstance(n, ast.Call) and ast.unparse(n.func) == "np.maximum"
                           and isinstance(n.args[1], ast.BinOp) and isinstance(n.args[1].op, ast.Mult)]
    alerts = [ast.dump(n.value, include_attributes=False) for n in ast.walk(world["_update_emergency"])
              if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "alert" for t in n.targets)]
    hold_offsets = [ast.literal_eval(n.right) for n in ast.walk(world["_update_emergency"])
                    if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add)
                    and isinstance(n.left, ast.Name) and n.left.id == "t" and isinstance(n.right, ast.Constant)]
    world_tree = ast.parse((root / "terminal_marl/core/rotating_slots/world.py").read_text(encoding="utf8"))
    state_thresholds = []
    for node in ast.walk(world_tree):
        if isinstance(node, ast.Compare) and any(
                isinstance(part, ast.Constant) and type(part.value) in (int, float)
                for value in node.comparators for part in ast.walk(value)):
            names = {part.id for part in ast.walk(node.left) if isinstance(part, ast.Name)}
            attributes = {part.attr for part in ast.walk(node.left) if isinstance(part, ast.Attribute)}
            if names & {"soc", "peak", "qhold"} or attributes & {"soc", "peak_seconds", "peak_used_s"}:
                state_thresholds.append(ast.dump(node, include_attributes=False))
    if len(vector_roots) != 1 or len(quadrature_tolerances) != 1 or len(relative_tolerances) != 1 or len(alerts) != 1 or len(hold_offsets) != 2:
        raise ProvenanceError("UNRECOGNIZED_NUMERICAL_POLICY: root/quadrature/emergency expression")
    return {
        "root_bisections_default": _default(energy["_root"], "bisections"),
        "vector_root_bisections": vector_roots[0],
        "scalar_root_branch_residual": _scaled_residual(energy["_root"], "max"),
        "vector_root_branch_residual": _scaled_residual(energy["_power_array"], "np.maximum"),
        "quadrature_depth": _limit(energy["_quadrature_cells"], "depth", "GtE"),
        "quadrature_atol_j": quadrature_tolerances[0],
        "quadrature_rtol": relative_tolerances[0],
        "integration_max_dt_default": _default(energy["integrate_energy"], "max_dt"),
        "peak_depth": _limit(energy["_peak_measure"], "depth", "GtE"),
        "peak_min_interval": _limit(energy["_peak_measure"], "b - a", "Lt"),
        "plan_depth": _limit(safety["validate_plan"], "depth", "GtE"),
        "plan_min_interval": _limit(safety["validate_plan"], "b - a", "LtE"),
        "plan_endpoint_metadata_tolerances": _metadata_tolerances(safety["validate_plan"]),
        "separation_depth": _default(safety["validate_separation"], "max_depth"),
        "separation_min_interval": _default(safety["validate_separation"], "min_interval"),
        "minimum_distance_tolerance": _default(safety["certified_minimum_distance"], "tolerance_m"),
        "minimum_distance_budget": _default(safety["certified_minimum_distance"], "max_intervals"),
        "minimum_distance_depth": _limit(safety["certified_minimum_distance"], "depth", "GtE"),
        "minimum_distance_min_interval": _limit(safety["certified_minimum_distance"], "b - a", "LtE"),
        "physics_dt_default": _default(world["__init__"], "physics_dt"),
        "scheduler_period_default": _default(world["__init__"], "scheduler_period"),
        "emergency_alert_expression": alerts[0], "holding_offsets_s": hold_offsets,
        "world_soc_and_peak_thresholds": sorted(state_thresholds),
    }


def semantic_fingerprint(root, *, policy_id=None, assertions=None,
                         candidate_policy=None, numerical_policy=None):
    """Return separate frozen-rule/config/candidate/numerical identities.

    Explicit policy dictionaries are for negative fixtures and reviewed future
    adapters; callers must compare with the frozen R0 fingerprint before use.
    Function ASTs conservatively freeze policy/control flow. Named numerical
    expressions/defaults freeze hardcoded budgets without hashing helper bytes.
    This metadata alone is not a behavioral-equivalence qualification.
    """
    import yaml
    root = Path(root).resolve()
    physical_bytes = (root / PHYSICAL_PATH).read_bytes()
    config = yaml.safe_load(physical_bytes)
    world = ast.parse((root / "terminal_marl/core/rotating_slots/world.py").read_text(encoding="utf8"))
    source_policy = next(ast.literal_eval(n.value) for n in world.body if isinstance(n, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == "POLICY_ID" for t in n.targets))
    policy_id = policy_id if policy_id is not None else source_policy
    rules = _functions(root, "world.py", ("_protect_times", "_outer_capacity", "_wait_upstream", "reroute", "admit"))
    effective_rules = {"policy_id": policy_id, "source_policy_id": source_policy,
                       "amendment_text_sha256": sha256((root / AMENDMENT_PATH).read_text(encoding="utf8").encode()).hexdigest(),
                       "rule_functions": {k: ast.dump(v, include_attributes=False) for k, v in rules.items()},
                       "assertions": assertions or {"admission": "upstream_wait_when_not_proven_capacity_full",
                                                    "pad_interval": "descent_start_to_touchdown_plus_service"}}
    if candidate_policy is None:
        candidates = {}
        for filename, names in (("trajectory.py", ("_search", "generate_transfer", "generate_inbound", "generate_final")),
                                ("world.py", ("_canonical_reference_duration", "_reference", "_try_final", "_freeflow_screen")),
                                ("driver.py", ("__call__",))):
            candidates[filename] = {k: ast.dump(v, include_attributes=False)
                                    for k, v in _functions(root, filename, names).items()}
        phase_bins = [ast.literal_eval(n.value) for n in ast.walk(world) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "REFERENCE_PHASE_BINS" for t in n.targets)]
        candidate_policy = {"functions": candidates, "reference_phase_bins": phase_bins}
    if numerical_policy is None:
        numerical_policy = _numerical_projection(root)
    result = {"schema": "ft_rs_semantic_fingerprint_v1_0", "reference_commit": R0_COMMIT, "policy_id": policy_id,
              "physical_config_bytes_sha256": sha256(physical_bytes).hexdigest(),
              "config_hash": digest(config), "vehicle_profile_hash": digest(config["vehicle"]),
              "energy_profile_hash": digest(config["energy"]),
              "profile_hash": digest({k: config[k] for k in ("vehicle", "energy", "coordinates")}),
              "effective_rules_hash": digest(effective_rules),
              "candidate_policy_hash": digest(candidate_policy),
              "numerical_policy_hash": digest(numerical_policy)}
    result["semantic_sha256"] = digest({k: v for k, v in result.items() if k != "physical_config_bytes_sha256"})
    return result


def require_same_semantics(reference, current):
    """Reject any semantic metadata drift; never accept a matching YAML alone."""
    # Git LF and a Windows CRLF checkout have different raw-byte provenance,
    # while the parsed physical configuration and effective rules are identical.
    changed = sorted(k for k in set(reference) | set(current)
                     if k != "physical_config_bytes_sha256" and reference.get(k) != current.get(k))
    if changed:
        raise ProvenanceError("SEMANTIC_DRIFT: " + ", ".join(changed))
    return current
