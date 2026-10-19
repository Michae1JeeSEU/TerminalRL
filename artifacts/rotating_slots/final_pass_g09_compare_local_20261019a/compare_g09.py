"""Read-only fixed G09 job comparison; not a formal performance qualification."""
import sys
sys.dont_write_bytecode = True
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics
import subprocess
import time
import traceback

PROJECT = Path("D:/7_codespace/terminal").resolve()
OWNED = Path(__file__).resolve().parent
ARTIFACTS = PROJECT/"artifacts/rotating_slots"
REFERENCE = ARTIFACTS/"performance_reference_r0"
R0_OLD = ARTIFACTS/"final_pass_reference_regression_local_20261019a/G09/benchmark"
RECOVERY = ARTIFACTS/"final_pass_reference_g09_recovery_20261019a"
R0_A = RECOVERY/"run_20261019a/jobs"
R0_B = RECOVERY/"run_20261019b"
OPT = ARTIFACTS/"final_pass_regression_local_20261019a"
OPT_JOBS = OPT/"G09/benchmark"
R0_COMMIT = "1c5afd37c891a0ed4776f64367fc7264db3c1c54"
R0_SHA = "cd76b4c14b0a48d6d2a71ec7c63ce77010e6ea019b545807635b926eb2984891"
OPT_COMMIT = "afb61940886add31e472738bf9ba654199c9fa1c"
COMMAND = "run_stage1_tests G09 actualbenchmark"
MEMBERS = ("config.yaml", "states.jsonl", "events.jsonl", "rejections.jsonl",
           "trajectories.json", "manifest.json", "metrics.json")
LABEL = "diagnostic_profiled_observation_not_formal_speedup"

QUERY = r'''
import json, math
from pathlib import Path
params=json.loads(Path(REQUEST_PATH).read_bytes())
from terminal_marl.core.rotating_slots.replay import ReplayBundle
validations=[]
for pair in params['pairs']:
    for side in ('R0','OPT'):
        item=pair[side]; row=json.loads(Path(item['result_path']).read_bytes())
        where=side+':'+pair['id']
        def require(condition,field):
            if not condition: raise ValueError(where+': '+field)
        require(row['provenance']==item['provenance'],'persisted provenance changed')
        require(row['status']=='COMPLETED','not COMPLETED')
        path=Path(row['replay_path']).resolve()
        require(path==Path(item['replay_path']).resolve(),'replay path changed')
        archive=ReplayBundle.validate(path)
        require(archive.index['code_commit']==item['commit'],'archive commit')
        require(archive.index['config_hash']==row['provenance']['config_hash'],'archive config hash')
        require(json.loads((path/'metrics.json').read_bytes())==row['metrics'],'archive/result metrics mismatch')
        expected=pair['protocol']; inflight=expected['mode']=='inflight_attempt'
        require(row['actual_airborne_count']==expected['count']+int(inflight),'actual airborne count')
        require(row['metrics']['physical_airborne_count']==row['actual_airborne_count'],'metric airborne count')
        require(row['inflight_admitted'] is inflight,'inflight admitted')
        require(row['metrics']['conservation'] is True,'conservation')
        require(row['simulated_seconds']==expected['duration'],'actual simulated duration')
        require(row['state_count']==archive.state_count,'state count')
        require(archive.time_bounds[1]==row['simulated_seconds'],'recorded end time')
        for field in ('wall_seconds','cpu_seconds'):
            require(type(row[field]) in (int,float) and math.isfinite(row[field]) and row[field]>0,field)
        require(row['measurement_instrumentation']['profiler']=='cProfile','original profiler')
        require(row['measurement_instrumentation']['profiler_enabled_during_run'] is True,'profiled original path')
        validations.append({'side':side,'id':pair['id'],'replay_path':str(path),'validated':True,
            'state_count':archive.state_count,'time_bounds':list(archive.time_bounds),
            'actual_airborne_count':row['actual_airborne_count'],'inflight_admitted':row['inflight_admitted'],
            'simulated_seconds':row['simulated_seconds'],'metrics_equal':True,
            'wall_seconds':row['wall_seconds'],'cpu_seconds':row['cpu_seconds']})
query_result={'validations':validations,'read_only':True}
'''


class NotReady(Exception):
    pass


def require(value, message):
    if not value: raise ValueError(message)


def write(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")


def file_hash(path):
    h=sha256()
    with path.open("rb") as stream:
        for part in iter(lambda:stream.read(1024*1024),b""): h.update(part)
    return h.hexdigest()


def expected():
    return [dict(id=f"benchmark_n{n}_{mode}_r{r}",kind="benchmark",count=n,mode=mode,repeat=r,
                 duration=.2,command=COMMAND)
            for n in (10,30,60,83) for mode in ("stable","inflight_attempt") for r in range(5)]


def protocol(row, specification):
    require(row["id"]==specification["id"],"row id differs: "+str(row.get("id")))
    for name,value in specification.items():
        require(json.dumps(row["job"].get(name))==json.dumps(value),f"{row['id']}: protocol differs at {name}")
    require(set(row["job"])==set(specification)|{"output"},row["id"]+": unexpected job fields")


def ordered_ids(rows, ids):
    actual=[r["id"] for r in rows]
    require(len(actual)==len(set(actual)),"duplicate job ID")
    require(actual==ids,"missing/unexpected/out-of-order job IDs")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-check",action="store_true")
    mode.add_argument("--compare",action="store_true")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if not sys.flags.isolated or not sys.flags.dont_write_bytecode: parser.error("parent requires -I -B")
    output=args.output.resolve()
    if output==OWNED or not output.is_relative_to(OWNED) or output.exists(): parser.error("fresh owned child output required")
    sys.path.insert(0,str(PROJECT))
    from terminal_marl.core.rotating_slots.performance import provenance
    output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic(); start_utc=datetime.now(timezone.utc).isoformat()
    inputs={}; report=dict(mode="self-check" if args.self_check else "compare",status="CHECKING",pairs=[],
        validations=[],member_comparisons=[],differences=[],groups=[],negative_checks=[],
        measurement_label=LABEL,human_status="HUMAN_PENDING",formal_40_comparison_executed=False)
    write(output/"invocation.json",dict(argv=sys.orig_argv,started_utc=start_utc,script_sha256=file_hash(Path(__file__))))
    reference_before=None; state="ERROR"; exit_code=1

    def track(path, allowed):
        path=Path(path).resolve();allowed=Path(allowed).resolve()
        require(path.is_relative_to(allowed),"input outside declared root: "+str(path))
        require(path.is_file(),"missing input: "+str(path))
        h=file_hash(path)
        require(str(path) not in inputs or inputs[str(path)]==h,"input changed while reading: "+str(path))
        inputs[str(path)]=h
        return path

    def load(path, allowed):
        return json.loads(track(path,allowed).read_bytes())

    def ready(path, allowed):
        if not Path(path).is_file(): raise NotReady("required terminal evidence absent: "+str(path))
        return load(path,allowed)

    def source_checks():
        preflight=load(OPT/"preflight.json",OPT)
        require(preflight["head"]==OPT_COMMIT,"OPT preflight source label differs")
        argv=load(OPT/"argv.json",OPT)["argv"]
        require(argv[argv.index("--code-commit")+1]==OPT_COMMIT,"OPT argv source label differs")
        rows=load(OPT/"core_hashes_before.json",OPT); seen=set()
        require(len(rows)==preflight["core_sourcefile_count"]==96,"OPT recorded source count differs")
        for item in rows:
            path=track(PROJECT/Path(item["path"]),PROJECT)
            require(str(path) not in seen,"duplicate OPT source entry");seen.add(str(path))
            require(inputs[str(path)]==item["sha256"].lower(),"OPT source bytes differ: "+str(path))
        actual={str(p.resolve()) for p in (PROJECT/"terminal_marl/core").rglob("*.py")}
        require(actual==seen,"OPT recorded/current core inventory differs")
        recorded_after=load(OPT/"core_hashes_after.json",OPT)
        recorded_comparison=load(OPT/"core_hashes_comparison.json",OPT)
        require(recorded_comparison['total']==recorded_comparison['equal_count']==96 and recorded_comparison['different_count']==0,"OPT historical after source comparison differs")
        require(len(recorded_after['entries'])==96,"OPT historical after source count")
        after_paths=set()
        for item in recorded_after['entries']:
            path=track(PROJECT/Path(item['path']),PROJECT)
            require(str(path) not in after_paths,"duplicate OPT after source entry");after_paths.add(str(path))
            require(Path(item['resolved_path']).resolve()==path,"OPT after-map path root differs")
            require(item['before_sha256'].lower()==item['after_sha256'].lower()==inputs[str(path)],"OPT recorded after/current source differs: "+str(path))
        require(after_paths==seen,"OPT before/after source inventories differ")
        report["OPT_source_identity"]=dict(commit=OPT_COMMIT,recorded_core_files=96,current_core_bytes_equal=True,
            historical_after_map_equal=True,historical_after_observed_utc=recorded_after['observed_utc'],
            scope="recorded before/after and actual comparison-time files checked; actual files checked again after comparison")

    try:
        reference_before=provenance._verify_export(REFERENCE)
        require(reference_before["reference_commit"]==R0_COMMIT and reference_before["source_sha256"]==R0_SHA,"fixed R0 identity")
        require(len(reference_before["actual_source_files_sha256"])==24,"fixed R0 member count")
        write(output/"reference_before.json",reference_before)
        source_checks()
        specifications=expected(); report_rows=None; provenance_expected={}
        if args.self_check:
            specifications=specifications[:1]
        else:
            # These gates read real completion evidence; they never start or resume a run.
            opt_exit=OPT/"numeric_exit_code.txt"
            if not opt_exit.is_file() or track(opt_exit,OPT).read_text().strip()!="0": raise NotReady("OPT actual exit0 unavailable")
            observed=ready(OPT/"completion_observed.json",OPT)
            if observed.get('numeric_exit')!=0 or any(p['alive'] for p in observed['pids']): raise NotReady("OPT observed terminal state unavailable")
            require(observed['observation_is_not_precise_completion_utc'] is True,"OPT observed-time interpretation")
            result=ready(OPT/"results.json",OPT)
            catalogue=json.loads((REFERENCE/"new10.16/acceptance_catalog_v1_0.json").read_bytes())["cases"]
            require({c['id'] for c in result['cases']}=={c['id'] for c in catalogue} and len(result['cases'])==94,"OPT 94-result coverage")
            if any(c['status'] not in ('PASS','HUMAN_PENDING') for c in result['cases']): raise NotReady("OPT catalogue has nonterminal/non-success row")
            require([c['id'] for c in result['cases'] if c['status']=='HUMAN_PENDING']==['V12'],"OPT human-pending identity")
            coverage=result['coverage']
            require(coverage['expected_count']==coverage['actual_count']==94 and not any(coverage[k] for k in ('missing','unexpected','duplicates','failures')),"OPT root coverage differs")
            for case in result['cases']:
                require(load(OPT/case['id']/"result.json",OPT)==case,"OPT per-case/root result mismatch: "+case['id'])
            if load(OPT/"G09/result.json",OPT)['status']!='PASS': raise NotReady("OPT G09 not PASS")
            opt_bench=ready(OPT_JOBS/"benchmark_results.json",OPT)
            ordered_ids(opt_bench['jobs'],[s['id'] for s in specifications])
            if any(r['status']!='COMPLETED' for r in opt_bench['jobs']): raise NotReady("OPT40 not complete")
            b=ready(R0_B/"summary.json",RECOVERY)
            if b.get('status')!='G09_RECOVERY_CHECKED' or b.get('success') is not True or b.get('numeric_exit')!=0: raise NotReady("R0 B recovery not complete")
            require((b['original_completed'],b['reused_completed'],b['new_completed'])==(22,2,16),"R0 B origin counts")
            require(all(b['unchanged'].values()),"R0 B recorded source/input change")
            for name in ('source','reference','old_inputs','reused_inputs'):
                require(ready(R0_B/(name+'_before.json'),RECOVERY)==ready(R0_B/(name+'_after.json'),RECOVERY),"R0 B before/after differs: "+name)
            for path in (R0_B/'numericexitcode.txt',RECOVERY/'launch_20261019b.child_numeric_exit.txt',RECOVERY/'launch_20261019b.supervisor_numeric_exit.txt'):
                if not path.is_file() or track(path,RECOVERY).read_text().strip()!='0': raise NotReady("R0 B actual numeric exit0 unavailable: "+str(path))
            completion=ready(RECOVERY/'launch_20261019b.completion.json',RECOVERY)
            if completion.get('child_numeric_exit')!=0 or completion.get('supervisor_numeric_exit')!=0 or completion.get('exception'): raise NotReady("R0 B supervisor not successful")
            if ready(R0_B/'check_G09/G09/result.json',RECOVERY)['status']!='PASS': raise NotReady("R0 original G09 consumer not PASS")
            process=ready(R0_B/'children/check_G09/process.json',RECOVERY)
            if process.get('success') is not True or process.get('child_numeric_exit')!=0: raise NotReady("R0 final consumer child exit0 unavailable")
            consume=ready(R0_B/'children/check_G09/reference_evidence.json',RECOVERY)
            require(consume['reference_commit']==R0_COMMIT and consume['source_sha256']==R0_SHA,"R0 consumer source identity")
            require(all(Path(p).resolve().is_relative_to(REFERENCE) for p in consume['imported_module_paths'].values()),"R0 consumer mixed import")
            recovered=ready(R0_B/'diagnostics.json',RECOVERY)
            require(recovered['scope']=='G09_BENCHMARK_ONLY_RECOVERY' and recovered['contains_original_traffic_matrix'] is False,"R0 B report scope")
            report_rows=recovered['jobs'];ordered_ids(report_rows,[s['id'] for s in specifications])
            report['terminal_evidence']=dict(OPT_exit=0,OPT_case_count=94,OPT_G09='PASS',R0_B_exit=0,
                OPT_terminal_observed_utc=observed['observed_utc'],OPT_precise_completion_utc=None,
                R0_original_consumer='PASS',R0_origins={'original':22,'reused_A':2,'new_B':16})

        for index,specification in enumerate(specifications):
            id=specification['id']
            r0_root=R0_OLD if index<22 else R0_A if index<24 else R0_B/'jobs'
            pair=dict(id=id,protocol=specification)
            for side,job_root,commit in (('R0',r0_root,R0_COMMIT),('OPT',OPT_JOBS,OPT_COMMIT)):
                path=job_root/id/'result.json';allowed=R0_OLD if job_root==R0_OLD else RECOVERY if side=='R0' else OPT
                row=load(path,allowed);protocol(row,specification)
                require(row['status']=='COMPLETED',side+':'+id+': not COMPLETED')
                require(row['job']['output']==str(job_root),side+':'+id+': unexpected output root')
                if side=='R0' and report_rows is not None: require(row==report_rows[index],id+': R0 report differs from actual result')
                if side=='OPT' and not args.self_check: require(row==opt_bench['jobs'][index],id+': OPT benchmark report differs from actual result')
                identity=row['provenance'];require(identity['code_commit']==commit,side+':'+id+': source label')
                require(sha256(json.dumps(identity['source_files_sha256'],sort_keys=True).encode()).hexdigest()==identity['code_sha256'],side+':'+id+': provenance map digest')
                for name,h in identity['source_files_sha256'].items():
                    root=REFERENCE if side=='R0' else PROJECT
                    actual_path=track(root/Path(name),root)
                    require(inputs[str(actual_path)]==h,side+':'+id+': actual source differs: '+name)
                projection={k:identity[k] for k in ('code_commit','code_sha256','config_hash','source_files_sha256')}
                if side in provenance_expected: require(provenance_expected[side]==projection,side+': source/config varies between jobs')
                else: provenance_expected[side]=projection
                archive=Path(row['replay_path']).resolve();require(archive==(job_root/id/'replay').resolve(),side+':'+id+': archive path')
                replay_index=load(archive/'replay.json',job_root)
                for member in replay_index['files']:
                    require(Path(member).name==member,'unsafe replay member name')
                    track(archive/member,job_root)
                pair[side]=dict(result_path=str(path),replay_path=str(archive),commit=commit,provenance=identity,
                    actual_output=row['job']['output'],wall_seconds=row['wall_seconds'],cpu_seconds=row['cpu_seconds'])
            require(pair['R0']['provenance']['config_hash']==pair['OPT']['provenance']['config_hash'],id+': config identity differs')
            report['pairs'].append(pair)

        if args.self_check:
            real=load(Path(report['pairs'][0]['R0']['result_path']),R0_OLD)
            altered=json.loads(json.dumps(real));altered['job']['duration']=.25
            for name,action in (('changed_protocol',lambda:protocol(altered,specifications[0])),
                                ('missing_ID',lambda:ordered_ids([], [real['id']])),
                                ('duplicate_ID',lambda:ordered_ids([real,real],[real['id']]))):
                rejected=False
                try: action()
                except ValueError: rejected=True
                require(rejected,'self-check negative did not reject: '+name)
                report['negative_checks'].append(dict(name=name,rejected=True,in_memory_only=True))

        for pair in report['pairs']:
            for member in MEMBERS:
                a=Path(pair['R0']['replay_path'])/member;b=Path(pair['OPT']['replay_path'])/member
                difference=None;offset=0
                with a.open('rb') as left,b.open('rb') as right:
                    while True:
                        x,y=left.read(1024*1024),right.read(1024*1024)
                        if x!=y and difference is None:
                            difference=offset+next((i for i,two in enumerate(zip(x,y)) if two[0]!=two[1]),min(len(x),len(y)))
                        if not x and not y: break
                        offset+=max(len(x),len(y))
                item=dict(id=pair['id'],member=member,R0_path=str(a),OPT_path=str(b),R0_sha256=file_hash(a),
                    OPT_sha256=file_hash(b),R0_bytes=a.stat().st_size,OPT_bytes=b.stat().st_size,
                    bytes_equal=difference is None,first_differing_byte=difference)
                report['member_comparisons'].append(item)
                if difference is not None:report['differences'].append(item)
        write(output/'comparison.json',report)
        request=output/'validation_request.json';write(request,{'pairs':report['pairs']})
        code='REQUEST_PATH = '+repr(str(request))+'\n'+QUERY
        child_argv=[sys.executable,'-I','-B','-c',provenance._REFERENCE_CHILD,str(REFERENCE),json.dumps({'modules':[],'code':code})]
        require(len(subprocess.list2cmdline(child_argv))<30000,'validation argv too long')
        write(output/'reference_argv.json',child_argv)
        evidence=provenance.run_reference(REFERENCE,modules=(),code=code,timeout=1800,python=sys.executable)
        write(output/'reference_evidence.json',evidence)
        require(evidence['source_sha256']==R0_SHA and evidence['reference_commit']==R0_COMMIT,'validation child R0 identity')
        report['validations']=evidence['query_result']['validations']
        require(len(report['validations'])==2*len(specifications),'validation count')
        for n in (10,30,60,83):
            for mode in ('stable','inflight_attempt'):
                selected=[pair for pair in report['pairs'] if pair['protocol']['count']==n and pair['protocol']['mode']==mode]
                if not selected:continue
                if not args.self_check:require(len(selected)==5,'group repeat count')
                group=dict(count=n,mode=mode,sample_size=len(selected),measurement_label=LABEL)
                for side in ('R0','OPT'):
                    group[side]={}
                    for metric in ('wall_seconds','cpu_seconds'):
                        raw=[p[side][metric] for p in selected];ordered=sorted(raw)
                        group[side][metric]=dict(raw=raw,median=statistics.median(raw),p95=ordered[math.ceil(.95*len(raw))-1],p95_definition='nearest_rank')
                group['observed_R0_over_OPT_median_wall_ratio']=group['R0']['wall_seconds']['median']/group['OPT']['wall_seconds']['median']
                report['groups'].append(group)
        report['formal_40_comparison_executed']=not args.self_check
        state='SELF_CHECK_RECORDED' if args.self_check else 'COMPARISON_RECORDED'
        exit_code=0 if not report['differences'] else 1
    except NotReady as exc:
        state='NOT_READY';exit_code=2;report['not_ready_reason']=str(exc)
    except Exception as exc:
        state='COMPARISON_ERROR';exit_code=1
        report['differences'].append(dict(kind='integrity_protocol_or_execution_error',type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
    finally:
        write(output/'input_hashes_before.json',inputs)
        after={path:file_hash(Path(path)) if Path(path).is_file() else None for path in inputs}
        write(output/'input_hashes_after.json',after)
        report['inputs_unchanged']=inputs==after
        report['OPT_source_final_after_comparison_checked']=report['inputs_unchanged'] and report.get('OPT_source_identity',{}).get('historical_after_map_equal',False)
        try:
            reference_after=provenance._verify_export(REFERENCE);write(output/'reference_after.json',reference_after)
            report['R0_export_unchanged']=reference_before==reference_after
        except Exception as exc:
            report['R0_export_unchanged']=False;report['reference_after_error']=str(exc)
        if not report['inputs_unchanged'] or not report['R0_export_unchanged']:
            state='COMPARISON_ERROR';exit_code=1
            report['differences'].append(dict(kind='input_or_reference_changed'))
    report.update(status=state,numeric_exit=exit_code,started_utc=start_utc,ended_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_wall_s=time.monotonic()-started,pair_count=len(report['pairs']),validation_count=len(report['validations']),
        member_count=len(report['member_comparisons']),qualification='NOT_QUALIFIED',
        limitations=['original cProfile on both sides','.2sim-second jobs','concurrent contention may differ',
            'R0 original22 worker history plus fresh A/B18 versus OPT reused worker',
            'not six-window cold-cache or full-traffic qualification'])
    write(output/'comparison.json',report);(output/'numericexitcode.txt').write_text(str(exit_code)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('pairs','validations','member_comparisons','groups')},ensure_ascii=False))
    return exit_code


if __name__=='__main__':
    raise SystemExit(main())
