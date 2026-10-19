"""Fixed local recovery: original22 + two persisted run-A rows + missing16."""
import sys
sys.dont_write_bytecode = True
import os
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
for _key in THREADS:
    os.environ[_key] = "1"
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

PROJECT = Path("D:/7_codespace/terminal").resolve()
OWNED = Path(__file__).resolve().parent
REFERENCE = PROJECT / "artifacts/rotating_slots/performance_reference_r0"
OLD = PROJECT / "artifacts/rotating_slots/final_pass_reference_regression_local_20261019a"
BENCHMARK = OLD / "G09/benchmark"
R0 = "1c5afd37c891a0ed4776f64367fc7264db3c1c54"
COMMAND = "run_stage1_tests G09 actualbenchmark"
TIMEOUT = 1800
RUN_A = OWNED / "run_20261019a"
REUSED_IDS = ("benchmark_n60_stable_r2", "benchmark_n60_stable_r3")

# Requests (including all 40 job dictionaries) travel by file, not a long argv.
PREFIX = r'''
import sys, os, json, math
from pathlib import Path
from datetime import datetime, timezone
params = json.loads(Path(REQUEST_PATH).read_bytes())
os.environ['TERMINAL_CODE_COMMIT'] = '1c5afd37c891a0ed4776f64367fc7264db3c1c54'
Path(params['evidence_dir'], 'child_start.json').write_text(json.dumps({
    'pid': os.getpid(), 'UTC': datetime.now(timezone.utc).isoformat(),
    'orig_argv': sys.orig_argv, 'isolated': bool(sys.flags.isolated),
    'dontwrite': sys.dont_write_bytecode,
    'native_threads': {k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS')}}, indent=2))
from terminal_marl.core.rotating_slots import diagnostics, qualification
from terminal_marl.core.rotating_slots.replay import ReplayBundle
qualification.limit_threads()
'''

VALIDATE = r'''
from hashlib import sha256
def checked(row, expected, identity, allowed_root):
    assert row['id'] == expected['id'] and row['status'] == 'COMPLETED', row['id']
    assert json.dumps(row['job'],sort_keys=True) == json.dumps(expected,sort_keys=True), row['id']
    for key in ('code_commit','code_sha256','config_hash','source_files_sha256'):
        assert row['provenance'][key] == identity[key], (row['id'],key)
    path = Path(row['replay_path']).resolve()
    assert path.is_relative_to(Path(allowed_root).resolve())
    assert path == (Path(expected['output'])/expected['id']/'replay').resolve()
    archive = ReplayBundle.validate(path)
    assert archive.index['code_commit'] == identity['code_commit']
    assert archive.index['config_hash'] == identity['config_hash']
    assert json.loads((path/'metrics.json').read_bytes()) == row['metrics'], 'ARCHIVE_METRICS_MISMATCH'
    for key in ('wall_seconds','cpu_seconds'):
        assert type(row[key]) in (int,float) and math.isfinite(row[key]) and row[key] > 0
    inflight = expected['mode'] == 'inflight_attempt'
    assert row['actual_airborne_count'] == expected['count'] + int(inflight)
    assert row['metrics']['physical_airborne_count'] == row['actual_airborne_count']
    assert row['inflight_admitted'] is inflight and row['metrics']['conservation']
    assert row['state_count'] == archive.state_count and archive.time_bounds[1] == row['simulated_seconds']
    return {'id':row['id'],'replay_path':str(path),'state_count':archive.state_count,
            'wall_seconds':row['wall_seconds'],'cpu_seconds':row['cpu_seconds'],
            'actual_airborne_count':row['actual_airborne_count'],'inflight_admitted':row['inflight_admitted']}
'''

INSPECT = VALIDATE + r'''
identity = qualification.provenance('run_stage1_tests G09 actualbenchmark')
assert identity['code_commit'] == params['reference_commit']
manifest = json.loads((reference_root/'.reference_manifest.json').read_bytes())
for name,digest in identity['source_files_sha256'].items():
    assert manifest['actual_source_files_sha256'].get(name) == digest, name
expected = {j['id']:j for j in params['expected']}
found = sorted(Path(params['old_benchmark']).glob('*/result.json'))
assert len(found) == 22, len(found)
rows, seen = [], set()
for path in found:
    assert path.resolve().is_relative_to(Path(params['old_benchmark']).resolve())
    row = json.loads(path.read_bytes())
    assert row['id'] == path.parent.name and row['id'] in expected and row['id'] not in seen
    seen.add(row['id'])
    item = checked(row,expected[row['id']],identity,params['old_benchmark'])
    item.update(result_path=str(path.resolve()),result_sha256=sha256(path.read_bytes()).hexdigest())
    rows.append(item)
original_rows = rows
run_a = Path(params['run_a']).resolve()
a_summary = json.loads((run_a/'summary.json').read_bytes())
assert a_summary['status']=='RECOVERY_ERROR' and a_summary['numeric_exit']==1
assert a_summary['unchanged']=={'source':True,'reference':True,'old_inputs':True}
for prefix in ('source','reference','old_inputs'):
    assert json.loads((run_a/(prefix+'_before.json')).read_bytes()) == json.loads((run_a/(prefix+'_after.json')).read_bytes())
assert json.loads((run_a/'reference_after.json').read_bytes()) == manifest
a_map = json.loads((run_a/'source_after.json').read_bytes())['actual_source_files_sha256']
assert set(a_map)==set(params['current_source_map'])
for name,digest in a_map.items():
    if name != params['driver_relative_path']: assert params['current_source_map'][name]==digest, name
found_a = sorted((run_a/'jobs').glob('*/result.json'))
assert {p.parent.name for p in found_a}==set(params['reused_ids']) and len(found_a)==2
reused_rows, metric_checks = [], []
for path in found_a:
    row=json.loads(path.read_bytes()); name=row['id']
    assert name==path.parent.name and name in params['reused_ids'] and name not in seen
    request_dir=run_a/'children'/name
    request=json.loads((request_dir/'request.json').read_bytes())
    invocation=json.loads((request_dir/'invocation.json').read_bytes())
    process=json.loads((request_dir/'process.json').read_bytes())
    startup=json.loads((request_dir/'child_start.json').read_bytes())
    expected_a=dict(expected[name],output=str(run_a/'jobs'))
    assert request['job']==expected_a and Path(request['evidence_dir']).resolve()==request_dir.resolve()
    assert invocation['name']==process['name']==name and process['child_numeric_exit']==1 and process['success'] is False
    assert (request_dir/'numericexitcode.txt').read_text().strip()=='1'
    assert process['error']['message'].startswith('REFERENCE_CHILD_FAILED: 1\n')
    assert 'line 30, in checked' in process['error']['message'] and 'AssertionError' in process['error']['message']
    assert startup['isolated'] and startup['dontwrite'] and type(startup['pid']) is int and startup['pid']>0
    assert all(v=='1' for v in startup['native_threads'].values()) and len(startup['native_threads'])==6
    assert startup['orig_argv'][1:]==invocation['argv'][1:]
    assert Path(invocation['argv'][-2]).resolve()==reference_root.resolve()
    old_query=json.loads(invocation['argv'][-1])
    assert old_query['modules']==[]
    assert old_query['code'].splitlines()[29].strip()=="assert json.loads((path/'metrics.json').read_bytes()) == row['metrics']"
    for key in ('code_commit','code_sha256','config_hash','source_files_sha256'):
        assert request['identity'][key]==identity[key]
    item=checked(row,expected_a,identity,str(run_a/'jobs'))
    item.update(result_path=str(path.resolve()),result_sha256=sha256(path.read_bytes()).hexdigest(),
        input_origin='run_A_persisted_COMPLETED',previous_child_pid=startup['pid'],previous_child_numeric_exit=1,
        previous_error_stage='post-job in-memory metrics versus serialized archive assertion',
        parent_export_postcheck_completed=True,child_post_import_audit_completed=False,
        current_strict_R0_revalidation=True)
    reused_rows.append(item);seen.add(name)
    # Reproduce only the serialization boundary using actual persisted metrics; no World/job execution.
    archived_metrics=json.loads((Path(row['replay_path'])/'metrics.json').read_bytes())
    memory_metrics=dict(row['metrics'])
    memory_metrics['pad_busy_seconds']={int(k):v for k,v in row['metrics']['pad_busy_seconds'].items()}
    assert memory_metrics != archived_metrics
    assert qualification.plain(memory_metrics)==archived_metrics
    negative=json.loads(json.dumps(row))
    negative['metrics']['physical_airborne_count']+=1
    rejected=False
    try: checked(negative,expected_a,identity,str(run_a/'jobs'))
    except AssertionError as exc:
        assert str(exc)=='ARCHIVE_METRICS_MISMATCH';rejected=True
    assert rejected
    metric_checks.append({'id':name,'persisted_archive_metrics_equal':True,
        'int_pad_key_direct_equality':False,'original_plain_equality':True,
        'modified_actual_metric_rejected':True,'modified_metric':'physical_airborne_count'})
missing = [j for j in params['expected'] if j['id'] not in seen]
assert len(missing)==16 and len(seen)==24
query_result = {'identity':identity,'verified_original':original_rows,'original_count':22,
    'verified_reused':reused_rows,'reused_count':2,'validated_count':24,'missing':missing,
    'missing_count':16,'normalization_checks':metric_checks,'run_A_still_error':True,
    'current_production_sources_same_except_recovery_driver':True,'read_only':True}
'''

JOB = VALIDATE + r'''
raw = diagnostics._diagnostic_job(params['job'])
row = json.loads((Path(params['job']['output'])/params['job']['id']/'result.json').read_bytes())
assert row == qualification.plain(raw), 'PERSISTED_RESULT_DIFFERS_FROM_ORIGINAL_PLAIN'
identity = qualification.provenance('run_stage1_tests G09 actualbenchmark')
for key in ('code_commit','code_sha256','config_hash','source_files_sha256'):
    assert identity[key] == params['identity'][key], key
validation = checked(row,params['job'],identity,params['job']['output'])
query_result = {'row':row,'validation':validation,'persisted_equals_original_plain':True}
'''

CONSUME = r'''
os.environ['TERMINAL_DIAGNOSTIC_REPORT'] = params['report']
result = qualification.run_catalogue(params['case_output'],cases=['G09'],workers=1,
    command='R0 G09 consume G09_BENCHMARK_ONLY_RECOVERY report')
row = next(r for r in result['cases'] if r['id']=='G09')
assert row['status']=='PASS', row
query_result = {'G09':row,'report':params['report'],'old_run_still_interrupted':True}
'''


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")
    temporary.replace(path)


def digest_file(path):
    value = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024),b""): value.update(chunk)
    return value.hexdigest()


def expected_jobs(output):
    return [dict(id=f"benchmark_n{n}_{mode}_r{repeat}",kind="benchmark",count=n,mode=mode,
        repeat=repeat,duration=.2,output=str(output),command=COMMAND)
        for n in (10,30,60,83) for mode in ("stable","inflight_attempt") for repeat in range(5)]


def old_snapshot():
    """Hash only the requested 91 case results and completed 22 archives."""
    catalogue = json.loads((REFERENCE/"new10.16/acceptance_catalog_v1_0.json").read_bytes())["cases"]
    expected = {c["id"] for c in catalogue} - {"G04","G05","G09"}
    results = sorted(OLD.glob("*/result.json"))
    assert {p.parent.name for p in results} == expected and len(results) == 91
    files = set(results)
    benchmark_results = sorted(BENCHMARK.glob("*/result.json"))
    assert len(benchmark_results) == 22, "expected exactly 22 old result files"
    for result in benchmark_results:
        assert result.resolve().is_relative_to(BENCHMARK.resolve())
        row = json.loads(result.read_bytes())
        archive = Path(row["replay_path"]).resolve()
        assert archive == (BENCHMARK/result.parent.name/"replay").resolve()
        assert archive.is_relative_to(OLD.resolve())
        files.add(result)
        files.update(p for p in archive.iterdir() if p.is_file())
    hashes = {}
    for path in sorted(files):
        path = path.resolve()
        assert path.is_relative_to(OLD.resolve()), path
        hashes[path.relative_to(OLD).as_posix()] = digest_file(path)
    return dict(root=str(OLD),old_case_result_count=91,old_benchmark_result_count=22,files_sha256=hashes)


def reused_snapshot():
    names = ("summary.json", "source_before.json", "source_after.json", "reference_before.json",
             "reference_after.json", "old_inputs_before.json", "old_inputs_after.json", "invocation.json")
    files = {RUN_A/name for name in names}
    found = sorted((RUN_A/"jobs").glob("*/result.json"))
    assert {p.parent.name for p in found}==set(REUSED_IDS) and len(found)==2
    for result in found:
        name=result.parent.name
        archive=Path(json.loads(result.read_bytes())["replay_path"]).resolve()
        assert archive==(RUN_A/"jobs"/name/"replay").resolve()
        assert archive.is_relative_to(RUN_A.resolve())
        files.add(result);files.update(p for p in archive.iterdir() if p.is_file())
        child=RUN_A/"children"/name
        files.update(child/n for n in ("request.json","invocation.json","process.json","child_start.json","numericexitcode.txt"))
    hashes={}
    for path in sorted(files):
        path=path.resolve();assert path.is_relative_to(RUN_A.resolve()),path
        hashes[path.relative_to(RUN_A).as_posix()]=digest_file(path)
    return dict(root=str(RUN_A),reused_completed_count=2,previous_run_numeric_exit=1,files_sha256=hashes)


def reference_call(name, body, params, output, provenance):
    directory = output/"children"/name
    directory.mkdir(parents=True,exist_ok=False)
    params = dict(params,evidence_dir=str(directory))
    request = directory/"request.json"; write(request,params)
    code = "REQUEST_PATH = " + repr(str(request)) + "\n" + PREFIX + body
    argv = [sys.executable,"-I","-B","-c",provenance._REFERENCE_CHILD,str(REFERENCE),
            json.dumps({"modules":[],"code":code})]
    assert len(subprocess.list2cmdline(argv)) < 30000, "child command too long"
    record = dict(name=name,started_utc=utc(),timeout_s=TIMEOUT,argv=argv,child_numeric_exit=None,success=False)
    write(directory/"invocation.json",record)
    started = time.monotonic()
    try:
        evidence = provenance.run_reference(REFERENCE,modules=(),code=code,timeout=TIMEOUT,python=sys.executable)
        write(directory/"reference_evidence.json",evidence)
        write(directory/"query_result.json",evidence["query_result"])
        (directory/"captured_stdout.log").write_text(evidence["captured_stdout"],encoding="utf8")
        record.update(child_numeric_exit=0,exit_source="run_reference verified actual returncode",success=True,
                      child_pid=evidence["pid"],query_result=evidence["query_result"])
    except Exception as exc:
        record["error"] = dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
        found = re.match(r"REFERENCE_CHILD_FAILED: (-?\d+)\n",str(exc))
        if found: record["child_numeric_exit"] = int(found.group(1))
        if isinstance(exc,subprocess.TimeoutExpired):
            for key in ("stdout","stderr"):
                value = getattr(exc,key,None)
                if value is not None:
                    (directory/("timeout_"+key+".log")).write_bytes(value if isinstance(value,bytes) else value.encode())
    record.update(ended_utc=utc(),elapsed_wall_s=time.monotonic()-started)
    if record["child_numeric_exit"] is not None:
        (directory/"numericexitcode.txt").write_text(str(record["child_numeric_exit"])+"\n")
    write(directory/"process.json",record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True,type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inspect-only",action="store_true")
    mode.add_argument("--run-missing",action="store_true")
    parser.add_argument("--workers",type=int,choices=(1,2),default=2)
    args = parser.parse_args()
    if not sys.flags.isolated or not sys.flags.dont_write_bytecode: parser.error("parent requires -I -B")
    output = args.output.resolve()
    if output == OWNED or not output.is_relative_to(OWNED) or output.exists(): parser.error("fresh owned child output required")
    assert OWNED == PROJECT/"artifacts/rotating_slots/final_pass_reference_g09_recovery_20261019a"
    sys.path.insert(0,str(PROJECT))
    from terminal_marl.core.rotating_slots.performance import provenance
    output.mkdir(parents=True,exist_ok=False)
    start, started_utc = time.monotonic(), utc()
    before = provenance.implementation_fingerprint(PROJECT,extra_paths=(str(Path(__file__)),),modules={})
    write(output/"source_before.json",before)
    records, error, plan, completed = [], None, None, 0
    unchanged = dict(source=False,reference=False,old_inputs=False,reused_inputs=False)
    write(output/"invocation.json",dict(argv=sys.orig_argv,mode="inspect-only" if args.inspect_only else "run-missing",
          workers=args.workers,started_utc=started_utc,old_root=str(OLD),reused_root=str(RUN_A),reference=str(REFERENCE)))
    def progress(status):
        write(output/"progress.json",dict(status=status,UTC=utc(),original_completed=22,reused_completed=2,new_completed=completed,
              fixed_total=40,missing_total=16,elapsed_wall_s=time.monotonic()-start))
    try:
        manifest = provenance._verify_export(REFERENCE)
        assert manifest["reference_commit"]==R0 and len(manifest["actual_source_files_sha256"])==24
        write(output/"reference_before.json",manifest)
        snapshot = old_snapshot(); write(output/"old_inputs_before.json",snapshot)
        write(output/"reused_inputs_before.json",reused_snapshot())
        progress("VALIDATING_EXISTING_READ_ONLY")
        inspected = reference_call("inspect",INSPECT,dict(expected=expected_jobs(BENCHMARK),old_benchmark=str(BENCHMARK),reference_commit=R0,
            run_a=str(RUN_A),reused_ids=REUSED_IDS,current_source_map=before["actual_source_files_sha256"],
            driver_relative_path=Path(__file__).resolve().relative_to(PROJECT).as_posix()),output,provenance)
        records.append(inspected)
        if not inspected["success"]: raise RuntimeError("R0 readonly inspection failed; preserve raw child error")
        plan = inspected["query_result"]
        write(output/"plan.json",plan)
        progress("INSPECTED_22_ORIGINAL_2_REUSED_16_MISSING")
        if args.run_missing:
            new_jobs = [dict(job,output=str(output/"jobs")) for job in plan["missing"]]
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for offset in range(0,len(new_jobs),args.workers):
                    wave = new_jobs[offset:offset+args.workers]
                    progress("RUNNING_FIXED_MISSING_WAVE")
                    futures = [pool.submit(reference_call,j["id"],JOB,dict(job=j,identity=plan["identity"]),output,provenance) for j in wave]
                    results = [future.result() for future in futures]
                    records.extend(results)
                    completed += sum(r["success"] for r in results)
                    progress("MISSING_WAVE_FINISHED")
                    if not all(r["success"] for r in results): raise RuntimeError("missing job failed; no next wave or automatic retry")
            old_by_id = {r["id"]:Path(r["result_path"]) for r in plan["verified_original"]}
            old_by_id.update({r["id"]:Path(r["result_path"]) for r in plan["verified_reused"]})
            new_by_id = {j["id"]:output/"jobs"/j["id"]/"result.json" for j in new_jobs}
            rows = [json.loads((old_by_id.get(j["id"]) or new_by_id[j["id"]]).read_bytes()) for j in expected_jobs(BENCHMARK)]
            report = dict(schema="FT-RS-S1-1.0-diagnostics",protocol="FROZEN_FORMAL_MATRIX",
                scope="G09_BENCHMARK_ONLY_RECOVERY",contains_original_traffic_matrix=False,
                original_diagnostics_complete=False,old_run_status="INTERRUPTED_WITHOUT_FINAL_EXIT",
                old_root=str(OLD),original_count=22,reused_root=str(RUN_A),reused_completed=2,
                reused_run_A_status="RECOVERY_ERROR",new_count=16,jobs=rows)
            write(output/"diagnostics.json",report)
            progress("CONSUMING_G09_WITH_ORIGINAL_R0")
            checked_case = reference_call("check_G09",CONSUME,dict(report=str(output/"diagnostics.json"),case_output=str(output/"check_G09")),output,provenance)
            records.append(checked_case)
            if not checked_case["success"]: raise RuntimeError("original R0 G09 consumption failed")
    except Exception as exc:
        error = dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    finally:
        for name, action, saved in (
            ("source",lambda:provenance.implementation_fingerprint(PROJECT,extra_paths=(str(Path(__file__)),),modules={}),"source_before.json"),
            ("reference",lambda:provenance._verify_export(REFERENCE),"reference_before.json"),
            ("old_inputs",old_snapshot,"old_inputs_before.json"),
            ("reused_inputs",reused_snapshot,"reused_inputs_before.json")):
            try:
                after = action(); write(output/(name+"_after.json"),after)
                unchanged[name] = (output/saved).is_file() and after==json.loads((output/saved).read_bytes())
            except Exception as exc:
                write(output/(name+"_after_error.json"),dict(type=type(exc).__name__,message=str(exc)))
    success = error is None and plan is not None and all(unchanged.values())
    if args.run_missing: success = success and completed==16 and records[-1]["success"] and records[-1]["name"]=="check_G09"
    status = ("INSPECTED_22_ORIGINAL_2_REUSED_16_MISSING" if args.inspect_only else "G09_RECOVERY_CHECKED") if success else "RECOVERY_ERROR"
    summary = dict(status=status,success=success,mode="inspect-only" if args.inspect_only else "run-missing",
        started_utc=started_utc,ended_utc=utc(),elapsed_wall_s=time.monotonic()-start,new_completed=completed,
        original_completed=22 if plan else None,reused_completed=2 if plan else None,
        validated_existing=24 if plan else None,missing_count=16 if plan else None,unchanged=unchanged,error=error,
        child_processes=[{k:r.get(k) for k in ("name","child_pid","child_numeric_exit","success")} for r in records],
        old_run_still_interrupted=True,run_A_still_RECOVERY_ERROR=True,original_V_failures_preserved=True,no_hard_timeout_full_tree_CPU_claim=True,
        human_status="HUMAN_PENDING",numeric_exit=0 if success else 1)
    write(output/"summary.json",summary); progress(status)
    (output/"numericexitcode.txt").write_text(str(summary["numeric_exit"])+"\n")
    print(json.dumps(summary,ensure_ascii=False)); return summary["numeric_exit"]


if __name__ == "__main__":
    raise SystemExit(main())
