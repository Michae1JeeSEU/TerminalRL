"""Explicit four-parent A4 launcher. No retry, recovery, or ledger mutation."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path("/home/user/projects/terminal")
PYTHON = "/home/user/envs/terminal_env/bin/python"
OUTPUT = Path("/home/user/training_outputs/terminal/rs_selective_quality_a4_20261030")
FREEZE = PROJECT/"artifacts/rs_select_quality_a4_20261030/comparison_source_freeze_exportfix1.json"
COMMIT = "b80ca3aee7d7ba6495640f4a3b02f72b026fabbb"
SOURCE_HASH = "1fd5c9a19919891b75f2158b42e291e63d8ef9f9f41400b8e9f2cbd487f7ed03"
FREEZE_SHA256 = "6283b8714cbcb4c24aaa3ace24eea87ed03cabb499f7bda9d3e3eb867052b6f0"
PROFILE = "stitched_continuation_v1"
THREAD_ENV = {"OMP_NUM_THREADS":"1", "OPENBLAS_NUM_THREADS":"1", "MKL_NUM_THREADS":"1",
              "NUMEXPR_NUM_THREADS":"1", "VECLIB_MAXIMUM_THREADS":"1"}
SPECS = (("s3r23","FULL_H",16), ("s3r23","RULE_LOCAL_K8",17),
         ("s3r24","FULL_H",18), ("s3r24","RULE_LOCAL_K8",19))
COMMON_SHA256 = {
    "common_initial/s3r23/boundary_summary.json":"21fe42b26780685209d46b5defc2b52aa1974c851e0f41a0e2148536f704a07e",
    "common_initial/s3r23/common.checkpoint.json":"0f67e40052c6d80ba35efebaf1e47fb113a1aca7046332d20b0c44f8cc29efc7",
    "common_initial/s3r23/initializer_callback_records.json":"01ac3f833e0952769e7fea913d3e4e57233ee6a3de2ccefd6c1888c11c933ce6",
    "common_initial/s3r23/initializer_services.json":"f8af68608be6ce154ada027eaf7bfdec78ba2aa5e6e91f630994c9bea2fb3ac2",
    "common_initial/s3r23/initializer_tasks.json":"7b61da0a3d4e2680379b97e3e36de180057b88203cfb35561952c0f199272132",
    "common_initial/s3r24/boundary_summary.json":"f4ee1fb053708c99f0a378e67433ae576ab4e5aeffe0beddcdcf614a3766103c",
    "common_initial/s3r24/common.checkpoint.json":"255f3ce5a3c45b01d3fb59ce5069ce39dec73b420735759ecdd6ce510f5eda77",
    "common_initial/s3r24/initializer_callback_records.json":"ffba76cb445ce3c4fa4c0e4a1287e3fe617fbaf8305a0921b34d9046b3987bbe",
    "common_initial/s3r24/initializer_services.json":"a16bbfa302885d6025ad24788ba33c19e3cd84213ed8454a8141df12b66df919",
    "common_initial/s3r24/initializer_tasks.json":"655538d3d625a41ae8a8dfe5761b5fe4dba6c39eeddb8b4c0e7056fe9b16c26a",
}


def require(value, message):
    if not value: raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open("x", encoding="utf8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())


def environment(project):
    env = dict(os.environ, **THREAD_ENV)
    env["PYTHONPATH"] = str(project)+":/home/user/envs/terminal_stage2_v1/lib/python3.12/site-packages"
    require(len(THREAD_ENV)==5 and all(env[key]=="1" for key in THREAD_ENV), "five literal thread settings required")
    return env


def job_id(phase, root, method):
    return "comparison_"+phase+"_"+root+"_"+method


def preflight(phase, project, output, freeze, base, a4, d):
    require(phase in ("arm","audit"), "explicit arm or audit phase required")
    require(output.is_dir() and (output/"repair_budget_ledger.json").is_file(), "existing shared A4 ledger required")
    require(sha(freeze)==FREEZE_SHA256, "approved freeze bytes differ")
    source=load(freeze)
    require(source["commit"]==COMMIT and source["source_hash"]==SOURCE_HASH, "approved b80 export-fix source identity differs")
    base.verify_sources(source)
    common={name:dict(exists=(output/name).is_file(),sha256=sha(output/name)) for name in COMMON_SHA256}
    require(all(row["exists"] and row["sha256"]==COMMON_SHA256[name] for name,row in common.items()),
            "original common files changed")
    for root in ("s3r23","s3r24"):
        summary=load(output/"common_initial"/root/"boundary_summary.json")
        require(summary["status"]=="COMMON_INITIAL_READY" and summary["source_binding"]["root"]==root,
                "common readiness/root differs")
    resource=d.campaign.resource_probe()
    require(set(range(16,20))<=set(resource.get("free_physical_cores",())), "physical CPUs16-19 must all be free")
    require(resource.get("memory_available_bytes",0)>0, "available memory evidence required")
    with a4.ledger_transaction(output,write=False) as ledger:
        require(all(ledger.get(k)==v for k,v in a4.POLICY.items()), "A4 ledger policy differs")
        require(ledger.get("comparison_source_hash",SOURCE_HASH)==SOURCE_HASH and
                ledger.get("comparison_evaluation_profile",PROFILE)==PROFILE, "comparison ledger binding differs")
        remaining=ledger["cap_child_wall_s"]-a4.old.active_elapsed(ledger)
        require(remaining>0, "shared child-wall budget exhausted")
        for root,method,cpu in SPECS:
            require(job_id(phase,root,method) not in ledger["jobs"], "job already attempted; primary recovery required")
    for root,method,cpu in SPECS:
        home=output/"arms"/root/method
        directory=home/"evaluation"/root/method
        if phase=="arm": require(not home.exists(), "formal arm output already exists")
        else:
            require((directory/"result.json").is_file(), "audit requires saved arm result")
            binding=load(home/"resolved_run_binding.json")
            require(binding["source_hash"]==SOURCE_HASH and binding["evaluation_profile"]==PROFILE,
                    "saved arm source/profile differs")
            require(not (directory/"audit_result.json").exists() and
                    not (directory/"comparison_audit.started.json").exists(), "audit already attempted")
        require(not (output/"logs"/(job_id(phase,root,method)+".log")).exists(), "bounded child log already exists")
        require(not (output/"logs"/f"LAUNCHER_A4_{phase}_{root}_{method}.log").exists(), "launcher log already exists")
    for suffix in ("started","finished"):
        require(not (output/"receipts"/f"LAUNCHER_A4_{phase}.{suffix}.json").exists(), "launcher receipt already exists")
    for root,method,cpu in SPECS:
        require(not (output/"receipts"/f"LAUNCHER_A4_{phase}_{root}_{method}.started.json").exists(),
                "per-job launcher receipt already exists")
    return dict(source_commit=COMMIT,source_hash=SOURCE_HASH,freeze_sha256=FREEZE_SHA256,
                common_files=common,resource=resource,remaining_child_wall_s=remaining)


def launch(phase, *, project=PROJECT, output=OUTPUT, freeze=FREEZE, base, a4, d, popen=subprocess.Popen):
    env=environment(project)
    evidence=preflight(phase,project,output,freeze,base,a4,d)
    (output/"logs").mkdir(exist_ok=True); (output/"receipts").mkdir(exist_ok=True)
    receipt_home=output/"receipts"; rows=[]; processes=[]; launch_error=None
    with ExitStack() as stack:
        # Reserve every launcher log and aggregate receipt before spawning any parent.
        started_stream=stack.enter_context((receipt_home/f"LAUNCHER_A4_{phase}.started.json").open("x",encoding="utf8"))
        logs=[stack.enter_context((output/"logs"/f"LAUNCHER_A4_{phase}_{root}_{method}.log").open("x",encoding="utf8"))
              for root,method,cpu in SPECS]
        try:
            for (root,method,cpu),stream in zip(SPECS,logs):
                command=[PYTHON,str(project/"scripts/stage3/run_selective_common_comparison_a4.py"),
                    "--stage",phase,"--evaluation-profile",PROFILE,"--root",root,"--method",method,
                    "--cpu",str(cpu),"--output",str(output),"--source-freeze",str(freeze)]
                process=popen(command,cwd=str(project),env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                row=dict(root=root,method=method,cpu=cpu,job_id=job_id(phase,root,method),parent_pid=process.pid,
                         command=command,launcher_log=str(stream.name),started_epoch=time.time())
                rows.append(row); processes.append(process)
                write_new(receipt_home/f"LAUNCHER_A4_{phase}_{root}_{method}.started.json",row)
                print(json.dumps(dict(event="PARENT_STARTED",**row)),flush=True)
        except Exception as error:
            launch_error=repr(error)
        started=dict(phase=phase,launcher_pid=os.getpid(),evidence=evidence,parents=rows,launch_error=launch_error,
                     environment={k:env[k] for k in (*THREAD_ENV,"PYTHONPATH")})
        json.dump(started,started_stream,sort_keys=True,indent=2,allow_nan=False)
        started_stream.flush(); os.fsync(started_stream.fileno())
        print(json.dumps(dict(event="ALL_PARENTS_STARTED",phase=phase,pids=[p.pid for p in processes],launch_error=launch_error)),flush=True)
        for row,process in zip(rows,processes): row["parent_exit_code"]=process.wait()
    with a4.ledger_transaction(output,write=False) as ledger:
        for row in rows:
            child=dict(ledger["jobs"].get(row["job_id"],{}))
            row.update(child_exit_code=child.get("exit_code"),child_pid=child.get("child_pid"),
                       child_status=child.get("status"),child_stop_reason=child.get("stop_reason"))
            directory=output/"arms"/row["root"]/row["method"]/"evaluation"/row["root"]/row["method"]
            result_path=directory/("audit_result.json" if phase=="audit" else "result.json")
            result=load(result_path) if result_path.is_file() else {}
            row.update(result_path=str(result_path),arm_status=result.get("status"),
                       full_a20_covered=result.get("full_a20_covered"))
            row["success"]=(row["parent_exit_code"]==0 and row["child_exit_code"]==0 and
                row["child_pid"] is not None and row["child_status"]=="completed" and
                row["arm_status"]==("PASS" if phase=="audit" else "COMPLETE") and
                (phase!="audit" or row["full_a20_covered"] is True))
    finished=dict(phase=phase,launcher_pid=os.getpid(),parents=rows,launch_error=launch_error,
                  success=launch_error is None and len(rows)==4 and all(row["success"] for row in rows),finished_epoch=time.time())
    write_new(receipt_home/f"LAUNCHER_A4_{phase}.finished.json",finished)
    print(json.dumps(finished,sort_keys=True),flush=True)
    return finished


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("arm","audit")); args=parser.parse_args(argv)
    os.environ.update(environment(PROJECT))
    require(all(os.environ[key]=="1" for key in THREAD_ENV), "thread environment must precede dependency imports")
    sys.path[:0]=[str(PROJECT),"/home/user/envs/terminal_stage2_v1/lib/python3.12/site-packages"]
    from scripts.stage3 import run_selective_reoptimization_a as base
    from scripts.stage3 import run_selective_quality_gap_a4 as a4
    d=base.dependencies(PROJECT)
    result=launch(args.phase,base=base,a4=a4,d=d)
    return 0 if result["success"] else 1


if __name__=="__main__": sys.exit(main())
