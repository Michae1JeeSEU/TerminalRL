"""Executable catalogue and evidence archive. Enumeration never grants a pass."""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
CATALOGUE = ROOT/"new10.16/acceptance_catalog_v1_0.json"
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def limit_threads():
    for key in THREAD_VARIABLES: os.environ[key] = "1"
    try:
        from threadpoolctl import threadpool_limits
        global _THREAD_LIMITER
        _THREAD_LIMITER = threadpool_limits(limits=1)
    except ImportError:
        #CLI setsenvironment beforeNumPy import; nooptionaldependency needed.
        pass


def cpu_model():
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"): return line.split(":", 1)[1].strip()
        except OSError: pass
    return platform.processor() or "UNAVAILABLE"


def plain(value):
    if hasattr(value, "to_dict"): return plain(value.to_dict())
    if isinstance(value, Mapping): return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [plain(v) for v in value]
    if isinstance(value, Path): return str(value.resolve())
    if isinstance(value, float) and not math.isfinite(value): return None
    if hasattr(value, "item"): return plain(value.item())
    return value


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".writing")
    temporary.write_text(json.dumps(plain(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
    temporary.replace(path)


def catalogue():
    result = json.loads(CATALOGUE.read_text(encoding="utf8"))["cases"]
    ids = [x["id"] for x in result]
    if len(ids) != 94 or len(set(ids)) != 94: raise ValueError("invalid normative catalogue")
    return result


def provenance(command):
    from .config import load_config
    supplied = os.environ.get("TERMINAL_CODE_COMMIT")
    #The server is an executionenvironment only. Itscommit identity must be
    #supplied by the local launcher; noGit command runs there.
    if supplied:
        commit_value, dirty = supplied, None
    elif sys.platform == "win32":
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
        tracked = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=False)
        commit_value, dirty = commit.stdout.strip() if commit.returncode == 0 else "UNAVAILABLE", bool(tracked.stdout.strip())
    else:
        commit_value, dirty = "UNAVAILABLE_SET_TERMINAL_CODE_COMMIT", None
    files = sorted((ROOT/"terminal_marl/core/rotating_slots").glob("*.py"))
    files += sorted((ROOT/"terminal_marl/scripts").glob("*stage1*.py"))
    files += [ROOT/"terminal_marl/visualization/rotating_slots_replay.py", CATALOGUE,
        ROOT/"new10.16/fixed_terminal_rotating_slots_stage1_operating_contract_v1_0.md",
        ROOT/"new10.16/fixed_terminal_rotating_slots_stage1_config_v1_0.yaml", ROOT/"new10.16/energy_reference_v1_0.py"]
    hashes = {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in files if p.exists()}
    digest = sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    try:
        from threadpoolctl import threadpool_info
        native_threads = threadpool_info()
    except ImportError: native_threads = "UNAVAILABLE"
    return {"code_commit": commit_value,
        "working_tree_dirty": dirty, "source_files_sha256": hashes,
        "code_sha256": digest, "config_hash": load_config().config_hash,
        "command": command, "cwd": str(Path.cwd()), "python": sys.version, "executable": sys.executable,
        "supplied_code_commit": supplied, "diagnostic_report": os.environ.get("TERMINAL_DIAGNOSTIC_REPORT"),
        "native_threadpools": native_threads,
        "platform": platform.platform(), "cpu_model": cpu_model(), "logical_cpus": os.cpu_count(),
        "pid": os.getpid(), "threads": {key: os.environ.get(key) for key in THREAD_VARIABLES},
        "started_at": datetime.now(timezone.utc).isoformat()}


@dataclass
class Evidence:
    case_id: str
    output: Path
    provenance: dict
    seed: int = 11
    actions: list = field(default_factory=list)
    observations: list = field(default_factory=list)
    assertions: list = field(default_factory=list)
    worlds: list = field(default_factory=list)
    plans: list = field(default_factory=list)
    attempted_plans: list = field(default_factory=list)
    replay_paths: list = field(default_factory=list)
    _saved_worlds: int = 0

    def action(self, operation, inputs):
        self.actions.append({"sequence": len(self.actions), "operation": operation, "inputs": plain(inputs)})

    def observe(self, name, value):
        self.observations.append({"name": name, "actual": plain(value)})

    def check(self, condition, assertion, expected, actual):
        passed = bool(condition)
        self.assertions.append({"id": f"{self.case_id}-{len(self.assertions)+1:03d}", "assertion": assertion,
            "expected": plain(expected), "actual": plain(actual), "passed": passed})
        if not passed: raise AssertionError(assertion)

    def save_replays(self):
        from .replay import ReplayBundle
        for i in range(self._saved_worlds, len(self.worlds)):
            root = self.output/f"replay_{i:02d}"
            bundle = ReplayBundle.write(root, world=self.worlds[i], code_commit=self.provenance["code_commit"],
                seed=self.seed, command=self.provenance["command"], assertions=self.assertions)
            self.replay_paths.append(str(bundle.path))
        self._saved_worlds = len(self.worlds)
        if not self.worlds and not self.replay_paths:
            record = analytic_record(self.plans)
            bundle = ReplayBundle.write(self.output/"analytic_replay", record=record,
                code_commit=self.provenance["code_commit"], seed=self.seed,
                command=self.provenance["command"], assertions=self.assertions)
            self.replay_paths.append(str(bundle.path))
        return self.replay_paths


def analytic_record(plans, *, duration=None):
    """Explicit analyticfixture record; no flight outcome is manufactured.

    Telemetry for arbitrary negative/safetyfixtures is deliberately absent.
    Stable positivefixture power and SOC follow the actual constantpower model.
    """
    from . import make_stable_plan, evaluate_power, load_config
    if not plans: plans = [make_stable_plan("baseline", "outer", 0, end_time=.4)]
    #Only one plan per identity belongs in a synthetic frame.
    unique = {}
    for p in plans: unique[p.aircraft_id] = p
    plans = list(unique.values())
    start = min(p.start_time for p in plans)
    end = duration if duration is not None else max(p.end_time for p in plans)
    #Physicscases retain representativeanalytic samples;3lap scenario passes
    #its requestedfull duration and keeps0.2s resolution.
    if duration is None: end = min(end, start+.4)
    times = [start+i*.2 for i in range(int((end-start)/.2)+1)]
    if not times or times[-1] < end-1e-10: times.append(end)
    states = []
    for t in times:
        aircraft = []
        for p in plans:
            st = p.state(t)
            stable = p.kind == "STABLE"
            power = evaluate_power(st.v, st.a) if stable else None
            soc = .9-(t-start)*power.battery_w/(85*252*3600) if stable else .9
            aircraft.append({"id": p.aircraft_id, "status": "STABLE" if stable else "ANALYTIC_FIXTURE",
                "soc": soc, "p": st.p, "v": st.v, "a": st.a, "jerk": st.jerk,
                "ring": p.target_ring if stable else None, "slot": p.target_slot if stable else None,
                "service_class": "standard", "trajectory_hash": p.trajectory_hash,
                "battery_power_w": power.battery_w if power else None, "emergency": {"declared": False},
                "fixture_telemetry": "constant_power_analytic" if stable else "not_physical_execution"})
        states.append({"time": t, "aircraft": aircraft, "slots": [], "pads": [], "metrics": {}})
    return {"config_hash": load_config().config_hash, "sample_dt_s": .2, "manifest": {"seed": 11, "requests": []},
        "states": states, "trajectories": {p.trajectory_hash: p.to_dict() for p in plans},
        "events": [], "metrics": {"fixture_only": True}, "rejections": []}


AMENDMENT_SOURCE = "docs/rotating_slots_stage1/amendment_20261018.md"
AMENDED_CRITERIA = {
    "C05": ("Capacity-only reroute; otherwise external waiting",
            "37 permanently occupied outer slots with no recorded release; middle and inner capacity remains empty",
            "Only proven OUTER_CAPACITY reroutes; other unresolved requests remain upstream without a physical aircraft or resampling"),
    "C13": ("Emergency pad protection follows the descent interval",
            "Current ground service ends after the candidate emergency final's protected descent interval begins; reject the overlapping reservation",
            "Departure may precede service release; overlapping protected arrival or any trajectory/energy violation is rejected without shortening service"),
    "F08": ("Early emergency departure before a future pad release",
            "All pads initially busy but released before the chosen final reaches the protected descent interval",
            "A certified rescue may depart immediately and use the pad after release; retain full service, safety, energy and peak checks"),
    "F09": ("Unresolved emergency without an invalid departure-time deadline proof",
            "No candidate passes the bounded rescue search; retain the physical waiting aircraft and follow it to the actual failure/censoring outcome",
            "Do not emit the obsolete PAD_DEADLINE_MISSED proof or claim global infeasibility; preserve EMERGENCY_WAITING until an actual disposition"),
}


def _execution_case(case):
    """Overlay only execution criteria; catalogue() remains the frozen original."""
    from .world import POLICY_ID
    result = dict(case, policy_id=POLICY_ID, effective_criteria_source=AMENDMENT_SOURCE)
    if case["id"] in AMENDED_CRITERIA:
        for key, value in zip(("title", "fixture_requirement", "expected"), AMENDED_CRITERIA[case["id"]]):
            result["legacy_catalogue_"+key] = case[key]
            result[key] = value
    return result


def _case_policy_fields(case):
    return {k: v for k, v in case.items() if k in ("policy_id", "effective_criteria_source") or k.startswith("legacy_catalogue_")}


def _job(args):
    case, output, command = args
    limit_threads()
    from .fixtures import FixtureBlocked, execute_case
    root = Path(output)/case["id"]
    root.mkdir(parents=True, exist_ok=False)
    info = provenance(command)
    e = Evidence(case["id"], root, info)
    started = time.perf_counter(); status = "NOT_RUN"; reason = ""
    try:
        if case["id"] == "V12":
            status, reason = "HUMAN_PENDING", "researcher observation and explicit acceptance required"
        else:
            execute_case(case["id"], e)
            if not e.assertions: raise AssertionError("fixture executed no assertions")
            status = "PASS"
    except FixtureBlocked as exc:
        status, reason = "BLOCKED", str(exc)
    except Exception as exc:
        status, reason = "FAIL", f"{type(exc).__name__}: {exc}"
        (root/"traceback.log").write_text(traceback.format_exc(), encoding="utf8")
    try:
        e.save_replays()
    except Exception as exc:
        status, reason = "FAIL", reason+f"; replay_archive: {type(exc).__name__}: {exc}"
        (root/"replay_error.log").write_text(traceback.format_exc(), encoding="utf8")
    write_json(root/"fixture.json", {"seed": e.seed, "actions": e.actions, "attempted_plans": e.attempted_plans,
                                   **_case_policy_fields(case)})
    write_json(root/"assertions.json", e.assertions)
    write_json(root/"observations.json", e.observations)
    write_json(root/"environment.json", info)
    (root/"execution.log").write_text("\n".join(json.dumps(plain(x), ensure_ascii=False) for x in e.actions)+"\n"+status+": "+reason+"\n", encoding="utf8")
    result = {"id": case["id"], "title": case["title"], "fixture_requirement": case["fixture_requirement"],
        "expected": case["expected"], "status": status, "reason": reason, "seed": e.seed,
        "elapsed_wall_seconds": time.perf_counter()-started, "actual": e.observations,
        "assertion_count": len(e.assertions), "assertions_path": str((root/"assertions.json").resolve()),
        "log_path": str((root/"execution.log").resolve()), "fixture_path": str((root/"fixture.json").resolve()),
        "environment_path": str((root/"environment.json").resolve()), "replay_paths": e.replay_paths,
        "code_sha256": info["code_sha256"], "config_hash": info["config_hash"], "code_commit": info["code_commit"],
        "command": command, "researcher_visual_acceptance": "PENDING" if case["id"].startswith("V") else "NOT_APPLICABLE"}
    result.update(_case_policy_fields(case))
    write_json(root/"result.json", result)
    return result


def task_coverage(expected, rows):
    actual = [row["id"] for row in rows]; counts = Counter(actual)
    return {"expected_count": len(expected), "actual_count": len(actual),
            "missing": sorted(set(expected)-set(actual)), "unexpected": sorted(set(actual)-set(expected)),
            "duplicates": sorted(k for k, n in counts.items() if n > 1),
            "failures": [r["id"] for r in rows if r.get("status") in ("FAIL", "FAILED", "BLOCKED")],
            "pending": [r["id"] for r in rows if r.get("status") not in ("PASS", "COMPLETED", "FAIL", "FAILED", "BLOCKED")]}


def run_catalogue(output, *, cases=None, workers=1, command=None):
    if workers < 1: raise ValueError("workers must bepositive")
    root = Path(output).resolve(); root.mkdir(parents=True, exist_ok=True)
    if (root/"results.json").exists(): raise FileExistsError("use a freshoutputdirectory")
    all_cases = [_execution_case(r) for r in catalogue()]; ids = [r["id"] for r in all_cases]
    selected = set(ids if cases is None else cases)
    if len(selected-set(ids)): raise ValueError("unknown caseIDs: "+str(sorted(selected-set(ids))))
    requested = [r for r in all_cases if r["id"] in selected]
    command = command or subprocess.list2cmdline([sys.executable, "-m", "terminal_marl.scripts.run_stage1_tests", "--output", str(root), "--workers", str(workers), "--cases", ",".join(r["id"] for r in requested)])
    rows = []
    def progress():
        write_json(root/"progress.json", {"schema": "FT-RS-S1-1.0-progress", "completed": len(rows),
            "total": len(requested), "updated_at": datetime.now(timezone.utc).isoformat(),
            "case_status": {r["id"]: r["status"] for r in rows}, "coverage": task_coverage([c["id"] for c in requested], rows)})
    progress()
    jobs = [(r, str(root), command) for r in requested]
    if workers == 1:
        for job in jobs: rows.append(_job(job)); progress()
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending = {pool.submit(_job, job): job[0]["id"] for job in jobs}
            for future in as_completed(pending):
                try: rows.append(future.result())
                except Exception as exc:
                    rows.append({"id": pending[future], "status": "FAIL", "reason": f"worker_exception: {type(exc).__name__}: {exc}"})
                progress()
    coverage = task_coverage([r["id"] for r in requested], rows)
    by_id = {r["id"]: r for r in rows}
    results = [by_id.get(r["id"], {**r, "status": "NOT_RUN", "reason": "not selected inthis invocation"}) for r in all_cases]
    automatic = [r for r in results if r["id"][0] in "ABCDEFG"]
    visual = [r for r in results if r["id"].startswith("V") and r["id"] != "V12"]
    result = {"schema": "FT-RS-S1-1.0-qualification-results", "catalogue_is_results": False,
        "cases": results, "coverage": coverage,
        "policy_id": all_cases[0]["policy_id"], "effective_criteria_source": AMENDMENT_SOURCE,
        "automatic_status": "AUTOMATION_COMPLETE" if all(r["status"] == "PASS" for r in automatic) else "IMPLEMENTATION_IN_PROGRESS",
        "visualization_integration_status": "ADAPTER_CHECKS_COMPLETE" if all(r["status"] == "PASS" for r in visual) else "NOT_COMPLETE",
        "researcher_visual_acceptance": "PENDING", "stage1_status": "NOT_ACCEPTED",
        "counts": dict(Counter(r["status"] for r in results)), "workers": workers, "command": command}
    write_json(root/"results.json", result)
    return result


def merge_reports(paths, output):
    """Strictcrosshost aggregation; only executedrows contribute evidence."""
    from .config import load_config
    canonical = load_config().config_hash
    normative = catalogue(); known = {r["id"] for r in normative}
    observed, hashes, commits, policies = {}, set(), set(), set()
    sources = []
    for path in paths:
        source = Path(path).resolve()
        data = json.loads(source.read_text(encoding="utf8"))
        if data.get("schema") != "FT-RS-S1-1.0-qualification-results":
            raise ValueError("notan executablequalificationreport: "+str(source))
        sources.append({"path": str(source), "sha256": sha256(source.read_bytes()).hexdigest()})
        seen = set()
        for row in data["cases"]:
            identity = row["id"]
            if identity not in known or identity in seen: raise ValueError("invalid/duplicatecaseID insource: "+identity)
            seen.add(identity)
            if row["status"] == "NOT_RUN": continue
            if row.get("config_hash") != canonical: raise ValueError("canonicalconfigurationhash mismatch: "+identity)
            if not row.get("code_sha256") or not row.get("code_commit"): raise ValueError("missingcodeprovenance: "+identity)
            hashes.add(row["code_sha256"]); commits.add(row["code_commit"])
            policies.add(row.get("policy_id"))
            if identity in observed and observed[identity] != row: raise ValueError("conflictingexecutedcase: "+identity)
            if identity == "V12" and row["status"] != "HUMAN_PENDING": raise ValueError("V12 cannot be automaticallyapproved")
            observed[identity] = row
    if len(hashes) > 1 or len(commits) > 1: raise ValueError("codehash/commit mismatch across executedreports")
    if len(policies) > 1: raise ValueError("policy mismatch across executed reports")
    policy = next(iter(policies), None)
    if policy is not None:
        from .world import POLICY_ID
        if policy != POLICY_ID: raise ValueError("unsupported execution policy")
        normative = [_execution_case(r) for r in normative]
    rows = [observed.get(r["id"], {**r, "status": "NOT_RUN", "reason": "noexecutedevidence inmergedreports"}) for r in normative]
    result = {"schema": "FT-RS-S1-1.0-qualification-results", "catalogue_is_results": False,
        "cases": rows, "sources": sources, "coverage": task_coverage([r["id"] for r in normative], list(observed.values())),
        "counts": dict(Counter(r["status"] for r in rows)),
        "automatic_status": "AUTOMATION_COMPLETE" if all(r["status"] == "PASS" for r in rows if r["id"][0] in "ABCDEFG") else "IMPLEMENTATION_IN_PROGRESS",
        "visualization_integration_status": "ADAPTER_CHECKS_COMPLETE" if all(r["status"] == "PASS" for r in rows if r["id"].startswith("V") and r["id"] != "V12") else "NOT_COMPLETE",
        "researcher_visual_acceptance": "PENDING", "stage1_status": "NOT_ACCEPTED",
        "code_sha256": next(iter(hashes), None), "code_commit": next(iter(commits), None), "config_hash": canonical}
    if policy is not None:
        result.update(policy_id=policy, effective_criteria_source=AMENDMENT_SOURCE)
    target = Path(output).resolve(); target.mkdir(parents=True, exist_ok=True)
    if (target/"results.json").exists(): raise FileExistsError("refusingtooverwrite mergedreport")
    write_json(target/"results.json", result)
    return result


def audit_replay(path, output):
    """Delegate the readonly independentepisodeaudit toits dedicatedmodule."""
    from .audit import audit_replay as independent_audit
    target = Path(output).resolve()
    target.mkdir(parents=True, exist_ok=True)
    return independent_audit(path, output=target/"audit.json")
