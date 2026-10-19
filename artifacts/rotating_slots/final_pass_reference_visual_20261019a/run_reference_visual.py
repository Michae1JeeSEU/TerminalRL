"""One-off R0 scenario material extraction and common-viewer comparison."""
import sys
sys.dont_write_bytecode = True
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "1"
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import time
import traceback

PROJECT = Path(__file__).resolve().parents[3]
OWNED = Path(__file__).resolve().parent
REFERENCE = PROJECT / "artifacts/rotating_slots/performance_reference_r0"
OPT = PROJECT / "artifacts/rotating_slots/final_pass_regression_local_20261019a"
OLD_R0 = PROJECT / "artifacts/rotating_slots/final_pass_reference_regression_local_20261019a"
CASES = {"V01": "threshold_crossing", "V02": "three_laps_controls",
         "V04": "full_lifecycle", "V06": "six_pads", "V07": None, "V08": None}
REUSE = {"V07": ("F01",), "V08": ("F09", "F08")}
MEMBERS = ("config.yaml", "states.jsonl", "events.jsonl", "rejections.jsonl",
           "trajectories.json", "manifest.json", "metrics.json")
TIMEOUT = 1800

QUERY = r'''
import os, sys, traceback
from pathlib import Path
os.environ['TERMINAL_CODE_COMMIT'] = '1c5afd37c891a0ed4776f64367fc7264db3c1c54'
from terminal_marl.core.rotating_slots import fixtures, qualification
qualification.limit_threads()
root = Path(params['case_output'])
info = qualification.provenance(params['command'])
e = qualification.Evidence(params['case'], root, info)
result = {'case': params['case'], 'scenario': params['scenario'], 'status': 'SCENARIO_EXCEPTION',
          'replay_paths': [], 'error': None, 'pid': os.getpid(),
          'python_flags': {'isolated': bool(sys.flags.isolated), 'dontwrite': sys.dont_write_bytecode},
          'orig_argv': sys.orig_argv, 'fixture_module': fixtures.__file__,
          'qualification_module': qualification.__file__}
try:
    fixtures.execute_scenario(params['scenario'], e)
    if not e.assertions or not all(a['passed'] for a in e.assertions):
        raise AssertionError('scenario assertions absent or unsuccessful')
    paths = e.save_replays()
    if not paths: raise AssertionError('scenario produced no replay')
    result.update(status='R0_SCENARIO_MATERIAL', replay_paths=paths)
except Exception as exc:
    result['error'] = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
result.update(assertions=e.assertions, actions=e.actions, observations=e.observations,
              world_outcomes=[{'time': w.time, 'failed': w.failed, 'metrics': w.metrics()} for w in e.worlds])
qualification.write_json(root/'environment.json', info)
qualification.write_json(root/'assertions.json', e.assertions)
qualification.write_json(root/'scenario_result.json', result)
query_result = qualification.plain(result)
'''


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")


def file_hash(path):
    h = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bundle_hashes(path):
    return {p.name: file_hash(p) for p in sorted(path.iterdir()) if p.is_file()}


def generate(case, output, provenance):
    """Thread-pool work is only the existing isolated subprocess helper."""
    directory = output / case
    directory.mkdir(exist_ok=False)
    params = dict(case=case, scenario=CASES[case], case_output=str(directory),
                  command=subprocess.list2cmdline(sys.orig_argv))
    code = "params = " + repr(params) + "\n" + QUERY
    argv = [sys.executable, "-I", "-B", "-c", provenance._REFERENCE_CHILD,
            str(REFERENCE), json.dumps({"modules": [], "code": code})]
    write(directory / "child_argv.json", argv)
    start = time.monotonic()
    record = dict(case=case, mode="fresh_R0_scenario", started_utc=utc(), timeout_s=TIMEOUT,
                  status="IMPLEMENTATION_OR_SCENARIO_ERROR", child_numeric_exit=None)
    try:
        evidence = provenance.run_reference(REFERENCE, modules=(), code=code, timeout=TIMEOUT,
                                            python=sys.executable)
        write(directory / "reference_evidence.json", evidence)
        (directory / "captured_stdout.log").write_text(evidence["captured_stdout"], encoding="utf8")
        # run_reference returns only after testing the actual subprocess returncode == 0.
        record.update(child_numeric_exit=0, exit_source="run_reference checked actual returncode",
                      status=evidence["query_result"]["status"], result=evidence["query_result"])
        assert evidence["source_sha256"] == provenance.R0_SOURCE_SHA256
        assert len(evidence["actual_source_files_sha256"]) == 24
        assert evidence["query_result"]["python_flags"] == {"isolated": True, "dontwrite": True}
    except Exception as exc:
        record.update(status="IMPLEMENTATION_OR_SCENARIO_ERROR",
                      error=dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
        match = re.match(r"REFERENCE_CHILD_FAILED: (-?\d+)\n", str(exc))
        if match:
            record["child_numeric_exit"] = int(match.group(1))
        if isinstance(exc, subprocess.TimeoutExpired):
            record["status"] = "REFERENCE_QUERY_TIMEOUT"
            for name in ("stdout", "stderr"):
                value = getattr(exc, name, None)
                if value is not None:
                    (directory / ("timeout_" + name + ".log")).write_bytes(value if isinstance(value, bytes) else value.encode())
    record.update(ended_utc=utc(), wall_s=time.monotonic()-start)
    if record["child_numeric_exit"] is not None:
        (directory / "child_numericexitcode.txt").write_text(str(record["child_numeric_exit"]) + "\n")
    write(directory / "process.json", record)
    return record


def reuse(case, output, manifest):
    directory = output / case
    directory.mkdir(exist_ok=False)
    paths, source_records = [], []
    for source_case in REUSE[case]:
        source = OLD_R0 / source_case
        result = json.loads((source / "result.json").read_bytes())
        environment = json.loads((source / "environment.json").read_bytes())
        assert result["status"] == "PASS" and result["code_commit"] == manifest["reference_commit"]
        assert environment["code_commit"] == manifest["reference_commit"]
        for name, digest in environment["source_files_sha256"].items():
            assert manifest["actual_source_files_sha256"].get(name) == digest, name
        assertions = json.loads((source / "assertions.json").read_bytes())
        assert assertions and all(a["passed"] for a in assertions)
        paths.extend(result["replay_paths"])
        source_records.append(dict(case=source_case, result_path=str(source / "result.json"),
                                   result_sha256=file_hash(source / "result.json"), assertion_count=len(assertions)))
    result = dict(replay_paths=paths, status="R0_SCENARIO_MATERIAL", reused_cases=source_records)
    record = dict(case=case, mode="read_only_existing_R0_archive", status="R0_SCENARIO_MATERIAL",
                  child_numeric_exit=None, no_new_child=True, result=result)
    write(directory / "process.json", record)
    return record


def inspect_and_compare(record, output):
    from terminal_marl.core.rotating_slots.replay import ReplayBundle
    from terminal_marl.visualization.rotating_slots_replay import ReplayController
    case = record["case"]
    opt_result = json.loads((OPT / case / "result.json").read_bytes())
    assert opt_result["status"] == "PASS"
    reference_paths = [Path(p).resolve() for p in record["result"]["replay_paths"]]
    optimized_paths = [Path(p).resolve() for p in opt_result["replay_paths"]]
    comparison = dict(case=case, R0_paths=list(map(str, reference_paths)), OPT_paths=list(map(str, optimized_paths)),
                      path_counts_equal=len(reference_paths)==len(optimized_paths), bundles=[], files=[], differences=[])
    write(output / case / "comparison.json", comparison)
    if not comparison["path_counts_equal"]:
        comparison["differences"].append(dict(kind="replay_path_count", R0=len(reference_paths), OPT=len(optimized_paths)))
    for index, (r0, opt) in enumerate(zip(reference_paths, optimized_paths)):
        if record["mode"] == "fresh_R0_scenario":
            assert r0.relative_to(output / case) == opt.relative_to(OPT / case), "unexpected relative replay mapping"
        for side, path in (("R0", r0), ("OPT", opt)):
            before = bundle_hashes(path)
            archive = ReplayBundle.validate(path)
            bundle = ReplayBundle.load(path)
            controller = ReplayController(bundle)
            point = min(archive.time_bounds[1], archive.time_bounds[0]+.137)
            snapshot = controller.seek(point)
            controller.seek(archive.time_bounds[1]); controller.jump_event(-1); controller.jump_event(1)
            assert controller.seek(point) == snapshot, "seek/event round-trip changed snapshot"
            assert bundle_hashes(path) == before, "read-only viewer changed archive"
            comparison["bundles"].append(dict(side=side, path=str(path), file_sha256=before,
                state_count=archive.state_count, time_bounds=list(archive.time_bounds), seek_time=point,
                validate=True, snapshot_roundtrip_equal=True, archive_unchanged=True))
            write(output / case / "comparison.json", comparison)
        for member in MEMBERS:
            a, b = r0 / member, opt / member
            first_difference, offset = None, 0
            with a.open("rb") as left, b.open("rb") as right:
                while True:
                    x, y = left.read(1024*1024), right.read(1024*1024)
                    if x != y and first_difference is None:
                        first_difference = offset + next((i for i, pair in enumerate(zip(x, y)) if pair[0] != pair[1]), min(len(x),len(y)))
                    if not x and not y: break
                    offset += max(len(x), len(y))
            row = dict(replay_index=index, member=member, R0_path=str(a), OPT_path=str(b),
                       R0_sha256=file_hash(a), OPT_sha256=file_hash(b), R0_bytes=a.stat().st_size,
                       OPT_bytes=b.stat().st_size, bytes_equal=first_difference is None,
                       first_differing_byte=first_difference)
            comparison["files"].append(row)
            if first_difference is not None: comparison["differences"].append(row)
            write(output / case / "comparison.json", comparison)
    comparison["all_equal"] = comparison["path_counts_equal"] and not comparison["differences"]
    write(output / case / "comparison.json", comparison)
    record["comparison_equal"] = comparison["all_equal"]
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new child directory of this script's artifact directory")
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=list(CASES))
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    if not sys.flags.isolated or not sys.flags.dont_write_bytecode:
        parser.error("parent requires Python -I -B")
    output = args.output.resolve()
    if not output.is_relative_to(OWNED) or output == OWNED or output.exists():
        parser.error("output must be a fresh child of the owned artifact directory")
    if len(set(args.cases)) != len(args.cases): parser.error("duplicate cases")
    selected = [case for case in CASES if case in args.cases]
    sys.path.insert(0, str(PROJECT))
    from terminal_marl.core.rotating_slots.performance import provenance
    output.mkdir(parents=True, exist_ok=False)
    started = utc()
    before = provenance.implementation_fingerprint(PROJECT, extra_paths=(str(Path(__file__)),), modules={})
    write(output / "source_before.json", before)
    records, error, source_unchanged, reference_unchanged = [], None, False, False
    write(output / "invocation.json", dict(argv=sys.orig_argv, cases=selected, workers=args.workers,
          timeout_per_child_s=TIMEOUT, started_utc=started, reference=str(REFERENCE), opt_root=str(OPT),
          reuse_root=str(OLD_R0), material_aliases={"V03":"V02", "V11":"V02", "V05":"V01", "V09":"V01", "V10":"V04"}))
    try:
        manifest = provenance._verify_export(REFERENCE)
        assert len(manifest["actual_source_files_sha256"]) == 24
        write(output / "reference_before.json", manifest)
        fresh = [case for case in selected if CASES[case] is not None]
        # Only a fixed wave of at most two subprocesses is active; no kernel pools.
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for offset in range(0, len(fresh), args.workers):
                jobs = [pool.submit(generate, case, output, provenance) for case in fresh[offset:offset+args.workers]]
                wave = [job.result() for job in jobs]
                for record in wave:
                    records.append(record)
                    if record["status"] != "R0_SCENARIO_MATERIAL": raise RuntimeError("scenario/child did not complete; see process.json")
                    inspect_and_compare(record, output)
                    if not record["comparison_equal"]: raise RuntimeError("SEMANTIC_DIFFERENCE; see comparison.json")
        for case in selected:
            if case not in REUSE: continue
            record = reuse(case, output, manifest); records.append(record)
            inspect_and_compare(record, output)
            if not record["comparison_equal"]: raise RuntimeError("SEMANTIC_DIFFERENCE; no regeneration of reused archives")
    except Exception as exc:
        error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
    finally:
        try:
            final_reference = provenance._verify_export(REFERENCE)
            write(output / "reference_after.json", final_reference)
            reference_unchanged = (output / "reference_before.json").is_file() and final_reference == json.loads((output / "reference_before.json").read_bytes())
        except Exception as exc:
            write(output / "reference_after_error.json", dict(type=type(exc).__name__, message=str(exc)))
        after = provenance.implementation_fingerprint(PROJECT, extra_paths=(str(Path(__file__)),), modules={})
        write(output / "source_after.json", after)
        source_unchanged = before == after
    success = (error is None and source_unchanged and reference_unchanged and len(records)==len(selected)
               and all(r.get("comparison_equal") is True for r in records))
    summary = dict(status="R0_SCENARIO_MATERIAL" if success else "MATERIAL_PREPARATION_ERROR", success=success,
        started_utc=started, ended_utc=utc(), selected_cases=selected, records=records, error=error,
        source_unchanged=source_unchanged, reference_unchanged=reference_unchanged,
        original_V_failures_unchanged=True, human_status="HUMAN_PENDING", numeric_exit=0 if success else 1)
    write(output / "summary.json", summary)
    (output / "numericexitcode.txt").write_text(str(summary["numeric_exit"]) + "\n")
    print(json.dumps({k:v for k,v in summary.items() if k != "records"}, ensure_ascii=False))
    return summary["numeric_exit"]


if __name__ == "__main__":
    raise SystemExit(main())
