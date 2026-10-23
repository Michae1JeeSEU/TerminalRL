"""Explicit, resumable bounded Stage3 campaign. Importing never starts work."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import contextlib
import threading
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED

from ..stage2_campaign import atomic_json, canonical, digest, load_json

ROOT = Path(__file__).resolve().parents[4]
PACKAGE = ROOT / "new10.16/rotating_slots_stage3_learning_v1_0"
PHASES = {"preflight_tests":900., "data_and_labels":7200., "training":1800., "evaluation":3600., "archive":900.}
THREAD_ENV = {k:"1" for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS")}


def contract_config():
    import yaml
    return yaml.safe_load((PACKAGE/"stage3_learning_config_v1_0.yaml").read_text(encoding="utf-8"))


def utc_now(): return datetime.now(timezone.utc).isoformat()


class BudgetLimited(RuntimeError): pass


class CampaignLedger:
    """Persist epoch plus explicit human pauses; a restart never grants new time.

    Interrupted active intervals include downtime. Only pause/resume explicitly
    excludes human editing; external measured tests are credited once by ID.
    """
    def __init__(self, path, *, clock=time.time):
        self.path=Path(path); self.clock=clock;self.lock=threading.RLock()
        if self.path.exists(): self.data=load_json(self.path)
        else:
            cfg=contract_config()["compute"]
            self.data=dict(schema="stage3_campaign_ledger_v1",started_epoch=clock(),elapsed_floor_s=0.,
                total_s=cfg["campaign_wall_seconds"],compute_stop_s=cfg["compute_stop_seconds"],
                phase_caps=cfg["phase_caps_seconds"],phase_used={},external_credits={},jobs={},
                paused_at=None,pause_intervals=[],active_phase=None)
            self.persist()
    @property
    def elapsed(self):
        now=self.data.get("paused_at") or self.clock()
        excluded=sum(end-start for start,end in self.data["pause_intervals"])
        credit=sum(v["wall_s"] for v in self.data["external_credits"].values())
        return max(self.data["elapsed_floor_s"],now-self.data["started_epoch"]-excluded+credit)
    def phase_used(self, phase):
        value=self.data["phase_used"].get(phase,0.)
        active=self.data.get("active_phase")
        if active and active["phase"]==phase: value+=max(0.,self.clock()-active["started_epoch"])
        return value
    def remaining(self, phase):
        return max(0.,min(self.data["phase_caps"][phase]-self.phase_used(phase),
            (self.data["total_s"] if phase=="archive" else self.data["compute_stop_s"])-self.elapsed))
    def persist(self):
        with self.lock:
            self.data["elapsed_floor_s"]=self.elapsed
            atomic_json(self.path,self.data)
    def credit(self, credit_id, wall_s, phase="preflight_tests"):
        if not math.isfinite(wall_s) or wall_s<0: raise ValueError("invalid external wall credit")
        prior=self.data["external_credits"].get(credit_id)
        value=dict(phase=phase,wall_s=float(wall_s))
        if prior and prior!=value: raise ValueError("external credit ID already differs")
        if not prior:
            self.data["external_credits"][credit_id]=value
            self.data["phase_used"][phase]=self.data["phase_used"].get(phase,0.)+wall_s
        self.persist()
    def pause(self):
        if self.data.get("active_phase"): raise ValueError("cannot pause an active subprocess phase")
        self.persist()
        if self.data["paused_at"] is None: self.data["paused_at"]=self.clock()
        self.persist()
    def resume(self):
        if self.data["paused_at"] is not None:
            self.data["pause_intervals"].append([self.data["paused_at"],self.clock()]);self.data["paused_at"]=None
        self.persist()
    @contextmanager
    def phase(self, name):
        self.resume()
        active=self.data.get("active_phase")
        if active:
            old=active["phase"]
            self.data["phase_used"][old]=self.data["phase_used"].get(old,0.)+max(0.,self.clock()-active["started_epoch"])
            self.data["active_phase"]=None;self.persist()
        if self.remaining(name)<=0: raise BudgetLimited(name+" budget exhausted")
        started=self.clock();self.data["active_phase"]=dict(phase=name,started_epoch=started);self.persist()
        try: yield
        finally:
            self.data["phase_used"][name]=self.data["phase_used"].get(name,0.)+max(0.,self.clock()-started)
            self.data["active_phase"]=None;self.persist()


def source_files():
    from ..config import load_service_profile
    files=list((ROOT/"terminal_marl/core/rotating_slots").rglob("*.py"))
    files+=list((ROOT/"scripts/stage3").glob("*.py"))
    files+=list((ROOT/"tests").glob("test_rotating_slots_stage3*.py"))
    files+=list((ROOT/"configs/stage3").glob("*.yaml"))
    files+=list((ROOT/"terminal_marl/configs").glob("fixed_terminal_rotating_slots*.yaml"))
    files += [Path(load_service_profile(profile).path) for profile in ("baseline","service_half_v1")]
    files+=list(PACKAGE.glob("*"))
    files+=[ROOT/"terminal_marl/visualization/rotating_slots_replay.py",ROOT/"terminal_marl/visualization/rotating_slots_controller.py"]
    return sorted(set(p for p in files if p.is_file()))


def freeze_local(output, seed_scan_paths=()):
    """Only this explicit LOCAL command executes Git. Server execution uses hashes."""
    from .data import freeze_split,scan_used_seeds
    import yaml
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    if (output/"source_freeze.json").exists(): raise ValueError("source freeze already exists; never overwrite campaign identity")
    def git(*args): return subprocess.check_output(["git",*args],cwd=ROOT,stderr=subprocess.DEVNULL).decode("utf-8",errors="replace")
    hashes={p.relative_to(ROOT).as_posix():sha256(p.read_bytes()).hexdigest() for p in source_files()}
    diff=git("diff","--binary","HEAD","--","terminal_marl","scripts/stage3","configs/stage3","tests/test_rotating_slots_stage3*")
    value=dict(source_commit=git("rev-parse","HEAD").strip(),workspace_status=git("status","--short"),
        workspace_diff_hash=sha256(diff.encode()).hexdigest(),source_files=hashes,source_hash=digest(hashes),
        branch=git("branch","--show-current").strip(),created_at=utc_now())
    # Automatic collision discovery reads only identities/protocol metadata, never results.
    discovered=[]
    for base in (ROOT/"artifacts",ROOT/"configs",ROOT/"docs"):
        if base.exists():
            for pattern in ("**/manifest.json","**/protocol.json","**/split_manifest.json","**/campaign.json"):
                discovered.extend(base.glob(pattern))
    discovered.extend(Path(p) for p in seed_scan_paths)
    used,scanned=scan_used_seeds(discovered)
    plan=yaml.safe_load((PACKAGE/"stage3_dataset_plan_v1_0.yaml").read_text(encoding="utf-8"))
    split=freeze_split(plan,used,scan_inputs=scanned)
    atomic_json(output/"split_manifest.json",split)
    (output/"workspace.diff").write_text(diff,encoding="utf-8")
    atomic_json(output/"source_freeze.json",value)
    return value


def verify_sources(frozen):
    for relative,expected in frozen["source_files"].items():
        path=(ROOT/relative).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError("frozen source hash mismatch: "+relative)
    if digest(frozen["source_files"])!=frozen["source_hash"]: raise ValueError("source aggregate mismatch")


def resource_probe():
    """Actual dependency/thread/topology facts, without installing or guessing."""
    versions={}
    for package in ("numpy","scipy","PyYAML","ortools","torch","pytest","psutil"):
        try: versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package]=None
    result=dict(python=platform.python_version(),executable=sys.executable,platform=platform.platform(),
        versions=versions,thread_environment={k:os.environ.get(k) for k in THREAD_ENV},observed_at=utc_now())
    try:
        import psutil
        from ..stage2_campaign import physical_cores
        per_cpu=psutil.cpu_percent(interval=1.,percpu=True)
        occupied=set();processes=[]
        for process in psutil.process_iter(["pid","name","cmdline","cpu_percent"]):
            try:
                command=" ".join(process.info.get("cmdline") or [])
                if process.pid==os.getpid() or not any(x in command for x in ("run_rotating_slots","stage2","stage3","equal_linear","pytest")): continue
                affinity=process.cpu_affinity()
                # Only narrow pinned jobs reserve explicit cores. Full-affinity idle shells do not.
                if len(affinity)<len(per_cpu): occupied.update(affinity)
                processes.append(dict(pid=process.pid,name=process.info["name"],affinity=affinity,command=command[:500]))
            except (psutil.NoSuchProcess,psutil.AccessDenied): pass
        cores=physical_cores(); siblings={}
        for cpu in range(len(per_cpu)):
            path=Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
            try: siblings[cpu]=path.read_text().strip()
            except OSError: siblings[cpu]=str(cpu)
        occupied_groups={siblings.get(cpu) for cpu in occupied}
        free=[cpu for cpu in cores if siblings.get(cpu) not in occupied_groups and per_cpu[cpu]<20.]
        result.update(cpu_load_percent=per_cpu,physical_cores=cores,occupied_cores=sorted(occupied),
            free_physical_cores=free,memory_available_bytes=psutil.virtual_memory().available,
            related_processes=processes,load_average=list(os.getloadavg()) if hasattr(os,"getloadavg") else None)
    except ImportError:
        # Linux stdlib fallback: no dependency installation is needed for a probe.
        from ..stage2_campaign import physical_cores
        def cpu_ticks():
            rows={}
            for line in Path("/proc/stat").read_text().splitlines():
                fields=line.split()
                if fields and fields[0].startswith("cpu") and fields[0][3:].isdigit():
                    values=list(map(int,fields[1:]));rows[int(fields[0][3:])]=(sum(values[:8]),values[3]+(values[4] if len(values)>4 else 0))
            return rows
        if Path("/proc/stat").exists():
            first=cpu_ticks();time.sleep(1.);second=cpu_ticks()
            loads={cpu:100.*(1.-(idle-first[cpu][1])/max(1,total-first[cpu][0])) for cpu,(total,idle) in second.items()}
            allowed=set(os.sched_getaffinity(0));occupied=set();processes=[]
            def parse_list(value):
                cpus=set()
                for part in value.split(","):
                    limits=part.split("-");cpus.update(range(int(limits[0]),int(limits[-1])+1))
                return cpus
            for path in Path("/proc").iterdir():
                if not path.name.isdigit() or int(path.name)==os.getpid():continue
                try:
                    command=(path/"cmdline").read_bytes().replace(b"\0",b" ").decode(errors="replace")
                    if not any(x in command for x in ("run_rotating_slots","stage2","stage3","equal_linear","pytest")):continue
                    status=(path/"status").read_text()
                    line=next(s for s in status.splitlines() if s.startswith("Cpus_allowed_list:"))
                    affinity=parse_list(line.split(":",1)[1].strip())
                    if affinity and len(affinity)<len(allowed):occupied.update(affinity)
                    processes.append(dict(pid=int(path.name),affinity=sorted(affinity),command=command[:500]))
                except (OSError,StopIteration,ValueError):pass
            def siblings(cpu):
                try:return Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list").read_text().strip()
                except OSError:return str(cpu)
            blocked={siblings(cpu) for cpu in occupied}
            cores=physical_cores();free=[cpu for cpu in cores if siblings(cpu) not in blocked and loads.get(cpu,100.)<20.]
            mem={line.split(":")[0]:line.split(":")[1].strip() for line in Path("/proc/meminfo").read_text().splitlines()}
            result.update(resource_status="LINUX_PROC_OBSERVED",cpu_load_percent=loads,physical_cores=cores,
                free_physical_cores=free,occupied_cores=sorted(occupied),related_processes=processes,
                memory_available_bytes=int(mem["MemAvailable"].split()[0])*1024,load_average=list(os.getloadavg()))
        else:result.update(free_physical_cores=[],resource_status="TOPOLOGY_UNAVAILABLE")
    return result


def preflight(output, *, probe_only=False):
    from ..config import load_service_profile
    from ..stage2_planning import candidate_policy_id,objective_schema_id,PlanningBudget,HORIZON
    from ..world import STAGE2_POLICY_ID
    output=Path(output)
    frozen=load_json(output/"source_freeze.json");verify_sources(frozen)
    environment=resource_probe()
    atomic_json(output/"environment_probe.json",environment)
    if probe_only: return environment
    if not environment.get("free_physical_cores"): raise ValueError("no verified free physical core; inspect resource probe")
    missing=[x for x in ("numpy","scipy","PyYAML","ortools","torch","pytest") if environment["versions"].get(x) is None]
    if missing: raise ValueError("required dependencies missing: "+", ".join(missing))
    if any(environment["thread_environment"].get(k)!="1" for k in THREAD_ENV): raise ValueError("numerical thread environment must equal1 before Python starts")
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    environment["torch_threads"]=torch.get_num_threads();environment["torch_interop_threads"]=torch.get_num_interop_threads()
    from ortools.sat.python import cp_model
    probe_model=cp_model.CpModel();probe_model.AddBoolOr([probe_model.NewBoolVar("stage3_probe")])
    probe_solver=cp_model.CpSolver();probe_solver.parameters.num_search_workers=1
    environment["cp_probe_status"]=probe_solver.StatusName(probe_solver.Solve(probe_model))
    if environment["cp_probe_status"]!="OPTIMAL":raise ValueError("CP environment probe failed")
    split=load_json(output/"split_manifest.json");cfg=contract_config();config=load_service_profile("service_half_v1")
    hashes=frozen["source_files"]
    def family(*names): return digest({k:v for k,v in hashes.items() if any(k.endswith(n) for n in names)})
    fp=dict(source_commit=frozen["source_commit"],workspace_diff_hash=frozen["workspace_diff_hash"],
        physical_config_hash=config.config_hash,numerical_backend_id=digest(dict(sources=family("energy.py","safety.py","trajectory.py"),versions=environment["versions"])),
        admission_policy_id="shared_fifo_admission:"+family("stage2_campaign.py"),emergency_policy_id="rescue_need_v7:"+family("rescue_rules.py","world.py"),
        failure_policy_id=STAGE2_POLICY_ID+":"+family("world.py"),proposal_universe_id=candidate_policy_id("downstream_service_v8"),
        descriptor_schema_id=cfg["proposal"]["descriptor_schema_id"],feature_schema_id=cfg["features"]["schema_id"],
        feature_cost_policy_id=cfg["features"]["cost_policy_id"],objective_id=objective_schema_id("downstream_service_v8"),
        tail_estimator_id="downstream_service_v8:"+family("stage2_downstream.py"),evaluation_protocol_id=digest(cfg["evaluation"]),
        behavior_policy_id="offline_reference_stratified_probe_v1",planning_budget_profile="quality_v5",label_policy_id=cfg["labeling"]["policy_id"],
        model_id="NO_MODEL",split_manifest_id=split["split_manifest_id"],hardware_profile_id=digest(environment),cache_regime="FRESH_PROCESS_COLD_PER_ARM")
    binding=dict(record_type="RunBinding",contract_id=cfg["contract_id"],fingerprint=fp,
        source_hash=frozen["source_hash"],source_files=hashes,contract_hash=sha256((PACKAGE/cfg["contract_file"]).read_bytes()).hexdigest(),
        inputs=dict(split_manifest="split_manifest.json",physical_config_path=str(config.path),physical_config_sha256=config.raw_hash),
        environment=environment,resolved_budgets=cfg["compute"],planning_budget=PlanningBudget.for_profile("quality_v5").settings(),
        horizon_s=HORIZON,created_at=utc_now(),evaluation_cpu=environment["free_physical_cores"][0])
    if (output/"run_binding.generated.json").exists():
        prior=load_json(output/"run_binding.generated.json")
        # Probe timestamps/utilization change; scientific source/thread/numerical identity cannot.
        if prior["source_hash"]!=binding["source_hash"] or prior["fingerprint"]["split_manifest_id"]!=fp["split_manifest_id"]:
            raise ValueError("existing binding incompatible")
        return prior
    import yaml
    atomic_json(output/"run_binding.generated.json",binding)
    (output/"resolved_config.generated.yaml").write_text(yaml.safe_dump(dict(cfg,actual_binding=fp),sort_keys=False),encoding="utf-8")
    return binding


def run_subprocess(ledger, phase, job_id, command, directory, *, cap=None,parallel=False):
    """Fresh process, absolute hard timeout and immutable completed job receipts."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    with ledger.lock:prior=ledger.data["jobs"].get(job_id)
    if prior and prior["status"]=="completed": return prior
    allowance=ledger.remaining(phase)
    if cap is not None: allowance=min(allowance,cap)
    if allowance<=0: raise BudgetLimited(job_id)
    receipt=dict(job_id=job_id,status="running",command=list(map(str,command)),started_at=utc_now(),timeout_s=allowance)
    with ledger.lock:
        ledger.data["jobs"][job_id]=dict(receipt);ledger.persist()
    started=time.perf_counter()
    parent_cpu_started=time.process_time()
    try:
        import resource
        previous_usage=None if parallel else resource.getrusage(resource.RUSAGE_CHILDREN)
    except ImportError:previous_usage=None
    env=dict(os.environ,**THREAD_ENV)
    with (directory/(job_id.replace(":","_")+".log")).open("ab") as log:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        with ledger.lock:
            receipt["pid"]=process.pid;ledger.data["jobs"][job_id]=dict(receipt);ledger.persist()
        try:
            code=process.wait(timeout=allowance)
            receipt.update(status="completed" if code==0 else "failed",exit_code=code)
        except subprocess.TimeoutExpired:
            process.kill();process.wait();receipt.update(status="BUDGET_LIMITED",exit_code=process.returncode)
    receipt.update(wall_s=time.perf_counter()-started,finished_at=utc_now())
    receipt["parent_cpu_s"]=None if parallel else time.process_time()-parent_cpu_started
    if previous_usage is not None:
        usage=resource.getrusage(resource.RUSAGE_CHILDREN)
        receipt["child_process_tree_cpu_s"]=usage.ru_utime+usage.ru_stime-previous_usage.ru_utime-previous_usage.ru_stime
        receipt["total_process_tree_cpu_s"]=receipt["parent_cpu_s"]+receipt["child_process_tree_cpu_s"]
    else:receipt["cpu_accounting_status"]="PARALLEL_PHASE_AGGREGATE" if parallel else "CHILD_RUSAGE_UNAVAILABLE"
    with ledger.lock:
        ledger.data["jobs"][job_id]=dict(receipt);ledger.persist()
    return receipt


def job_command(output, kind, **payload):
    path=Path(output)/"jobs"/(kind+"_"+str(payload.get("case_id",payload.get("root_id","once"))))
    path=path.with_suffix(".json");atomic_json(path,dict(kind=kind,output=str(Path(output).resolve()),**payload))
    return [sys.executable,"-m","terminal_marl.core.rotating_slots.stage3_learning.campaign","--job",str(path)]


def run_root_jobs(output,ledger,kind,roots):
    """At most two independent single-threaded children, distinct free cores."""
    output=Path(output);probe=resource_probe();free=probe.get("free_physical_cores",[])
    if not free:raise ValueError("root job resource probe found no free physical core")
    workers=min(contract_config()["compute"]["max_parallel_processes"],len(free))
    phase="data_and_labels";started=time.perf_counter();parent_started=time.process_time()
    try:
        import resource
        before=resource.getrusage(resource.RUSAGE_CHILDREN)
    except ImportError:before=None
    receipts=[];items=iter(enumerate(roots));pending={};failed=False
    def launch(pool,cpu):
        try:index,root=next(items)
        except StopIteration:return False
        command=job_command(output,kind,root_id=root["root_id"],cpu=cpu)
        future=pool.submit(run_subprocess,ledger,phase,kind+":"+root["root_id"],command,output/"logs",
            cap=contract_config()["labeling"]["per_snapshot_total_wall_seconds"] if kind=="label" else None,parallel=True)
        pending[future]=(root,cpu);return True
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for cpu in free[:workers]:launch(pool,cpu)
        while pending:
            done,_=wait(pending,return_when=FIRST_COMPLETED)
            released=[]
            for future in done:
                root,cpu=pending.pop(future);released.append(cpu);receipt=future.result();receipts.append(dict(receipt,root_id=root["root_id"]))
                if receipt["status"]!="completed":failed=True
            if not failed:
                for cpu in released:launch(pool,cpu)
    observation=dict(kind=kind,workers=workers,cores=free[:workers],wall_s=time.perf_counter()-started,
        parent_cpu_s=time.process_time()-parent_started,resource_probe=probe,
        parallelism_reason="TWO_AVAILABLE_PHYSICAL_CORES" if workers==2 else "ONLY_ONE_VERIFIED_FREE_PHYSICAL_CORE")
    if before is not None:
        after=resource.getrusage(resource.RUSAGE_CHILDREN)
        observation["child_process_tree_cpu_s"]=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime
    with ledger.lock:
        ledger.data.setdefault("parallel_batches",[]).append(observation);ledger.persist()
    atomic_json(output/(kind+"_receipts.json"),dict(jobs=receipts,observation=observation))
    return receipts


def plot_training_history(output):
    """Render only actual completed epochs; no synthetic curve for absent training."""
    from .data import read_jsonl
    output=Path(output);directory=output/"model";card_path=directory/"model_card.json"
    card=load_json(card_path) if card_path.exists() else {}
    history_path=directory/"training_history.jsonl";curve=directory/"training_curve.png"
    value=dict(model_card_path=str(card_path),training_history_path=str(history_path),
        weights_present=(directory/"model.pt").is_file(),curve_present=curve.is_file())
    if not card.get("formal_training_started"):
        value.update(status="SKIPPED_DATA_INSUFFICIENT" if card.get("status")=="DATA_INSUFFICIENT" else "NOT_RUN",
            reason="No formal completed training epochs; no curve or weights generated by archive")
    elif not history_path.exists():value.update(status="HISTORY_MISSING",reason="Training started but no epoch history is present")
    else:
        try:
            history=read_jsonl(history_path)
            if not history:raise ValueError("no completed epochs")
            for row in history:
                if not all(isinstance(row.get(key),(int,float)) and math.isfinite(row[key]) for key in ("epoch","train_loss","validation_rank_loss")):
                    raise ValueError("history has missing/nonfinite epoch loss")
            if any(b["epoch"]<=a["epoch"] for a,b in zip(history,history[1:])):raise ValueError("epoch history is not strictly ordered")
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            figure=Figure(figsize=(7.,5.));FigureCanvasAgg(figure)
            axes=figure.subplots(2,1,sharex=True)
            axes[0].plot([row["epoch"] for row in history],[row["train_loss"] for row in history],marker="o",markersize=3,color="#2365a4")
            axes[0].set_ylabel("Train combined loss");axes[0].set_title("Stage3 observed training history")
            axes[1].plot([row["epoch"] for row in history],[row["validation_rank_loss"] for row in history],marker="o",markersize=3,color="#b05a28")
            axes[1].set_ylabel("Validation rank loss");axes[1].set_xlabel("Completed epoch")
            for axis in axes:axis.grid(alpha=.25)
            figure.tight_layout()
            temporary=curve.with_name(curve.name+".partial");figure.savefig(temporary,format="png",dpi=160);os.replace(temporary,curve)
            value.update(status="RENDERED_ACTUAL_HISTORY",curve_path=str(curve),curve_present=True,epochs=len(history),
                history_sha256=sha256(history_path.read_bytes()).hexdigest(),curve_sha256=sha256(curve.read_bytes()).hexdigest())
        except (OSError,ValueError,ImportError,KeyError) as exc:
            value.update(status="CURVE_NOT_RENDERED",reason=str(exc),error_type=type(exc).__name__)
    atomic_json(output/"training_curve_status.json",value);return value


def archive_report(output, ledger):
    output=Path(output)
    def load(name): return load_json(output/name) if (output/name).exists() else {}
    training=load("model/model_card.json"); same=load("same_state_comparison.json"); closed=load("closed_loop_comparison.json");signal=load("heldout_signal.json")
    checks=load("targeted_checks.json")
    curve=plot_training_history(output)
    statuses=dict(infrastructure="INFRASTRUCTURE_READY" if checks.get("all_automated_pass") else "IMPLEMENTATION_NOT_FULLY_VERIFIED",
        signal=signal.get("signal_status",training.get("learnable_signal_status","LEARNABLE_SIGNAL_NOT_ESTABLISHED")),
        online="ONLINE_BENEFIT_NOT_ESTABLISHED",training=training.get("status","NOT_RUN"),
        evaluation=closed.get("status",same.get("status","NOT_RUN")),human_visual_review="HUMAN_PENDING")
    atomic_json(output/"final_status.json",statuses)
    text="# Stage3 bounded pilot\n\n"+"\n".join(f"- {k}: {v}" for k,v in statuses.items())
    text+="\n\nElapsed campaign wall: %.3fs.\n\n"%ledger.elapsed
    text+="Full online quality and timing are reported in same_state_comparison.json and closed_loop_comparison.json when present. Accuracy alone is not an online benefit claim. No automatic quality-degradation tolerance is authorized.\n"
    text+="\nReplay entry: use ReplayController(path) and RotatingSlotsReplayViewer(controller).run() from terminal_marl.visualization.rotating_slots_replay; visual acceptance remains HUMAN_PENDING.\n"
    text+="\nTraining artifacts: model/model_card.json, model/training_history.jsonl; curve status="+curve["status"]+"."
    if curve.get("curve_path"):text+=" Curve: model/training_curve.png (actual completed epochs)."
    text+="\nLatency quantiles use actual callback wall_s / dispatch_wall_s with nearest-rank P50/P95/max, explicit watchdog/timeouts and deadline misses. Coverage summaries separate proposed, attempted, single-valid and certified supported descriptors by source ring and construction stage.\n"
    text+="\nCPU: "+json.dumps({k:{f:r.get(f) for f in ("parent_cpu_s","child_process_tree_cpu_s","total_process_tree_cpu_s")} for k,r in ledger.data["jobs"].items()})+"\n"
    (output/"FINAL_REPORT.md").write_text(text,encoding="utf-8")
    ledger.persist();return statuses


def execute_job(payload):
    output=Path(payload["output"]);kind=payload["kind"]
    frozen=load_json(output/"source_freeze.json");verify_sources(frozen)
    binding=load_json(output/"run_binding.generated.json")
    cpu=payload.get("cpu",binding["evaluation_cpu"])
    if hasattr(os,"sched_setaffinity"): os.sched_setaffinity(0,{cpu})
    if kind=="collect":
        from .data import make_root
        roots=load_json(output/"split_manifest.json")["roots"]
        make_root(next(r for r in roots if r["root_id"]==payload["root_id"]),output,binding)
    elif kind=="label":
        from .labeling import label_snapshot
        label_snapshot(output,payload["root_id"],binding)
    elif kind in ("same_state","closed_loop"):
        from .evaluation import evaluate_job
        evaluate_job(payload,binding)
    else: raise ValueError("unknown job kind")


def build_training_dataset(output):
    from .data import read_jsonl,auxiliary_labels
    output=Path(output);contexts=[];pairs=[];index=[]
    roots=load_json(output/"split_manifest.json")["roots"]
    for root in roots:
        directory=output/"labels"/root["root_id"]
        if not (directory/"labels.json").exists():continue
        if root["split"] not in ("train","validation"):
            index.append(dict(root_id=root["root_id"],split=root["split"],status="SEALED_NOT_READ",labels_path=str(directory/"labels.json")))
            continue
        labels=load_json(directory/"labels.json")
        index.append(dict(root_id=root["root_id"],split=root["split"],status=labels["status"],labels_path=str(directory/"labels.json")))
        raw=output/"raw"/root["root_id"]
        features=read_jsonl(raw/"features.jsonl");facts=labels.get("facts",[])
        descriptors={d["descriptor_id"]:d for d in read_jsonl(raw/"descriptors.jsonl")}
        by_context={}
        for feature in features:
            did=feature["descriptor_id"]
            if did not in descriptors:continue
            context_id=descriptors[did]["source_node_id"]
            by_context.setdefault(context_id,{})[did]=feature["features"]
        for context_id,encoded in by_context.items():
            local_facts=[f for f in facts if f["descriptor_id"] in encoded]
            contexts.append(dict(context_id=context_id,ancestry_root_id=root["root_id"],split=root["split"],
                fingerprint=labels["fingerprint"],features=list(encoded.values()),facts=local_facts,
                validity_labels=auxiliary_labels(local_facts)))
        pairs.extend(labels["preference_pairs"])
    result=dict(schema="stage3_training_dataset_v1",fingerprint=load_json(output/"run_binding.generated.json")["fingerprint"],contexts=contexts,preference_pairs=pairs)
    atomic_json(output/"training_dataset.json",result);atomic_json(output/"dataset_index.json",dict(roots=index,heldout_sealed=True))
    return result


def _main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",nargs="?",choices=("freeze","probe","preflight","tests","collect","label","train","evaluate","archive","all","pause","resume","credit"))
    parser.add_argument("--output",type=Path)
    parser.add_argument("--job",type=Path)
    parser.add_argument("--seed-scan",action="append",default=[])
    parser.add_argument("--credit-id");parser.add_argument("--credit-wall",type=float,default=0.)
    parser.add_argument("--checks-xml",type=Path,action="append",default=[])
    parser.add_argument("--root-id",action="append",default=[],help="Label only named frozen root(s), e.g. s3r01 serialization pilot")
    args=parser.parse_args(argv)
    if args.job: execute_job(load_json(args.job));return 0
    if args.output is None or args.stage is None: parser.error("stage and --output required")
    if args.root_id and args.stage!="label":parser.error("--root-id is only valid for the label stage")
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    if args.stage=="freeze": freeze_local(output,args.seed_scan);return 0
    ledger=CampaignLedger(output/"budget_ledger.json")
    if args.credit_id:ledger.credit(args.credit_id,args.credit_wall)
    if args.stage=="credit":
        if not args.credit_id: parser.error("credit requires --credit-id and --credit-wall")
        return 0
    if args.stage=="pause":ledger.pause();return 0
    if args.stage=="resume":ledger.resume();return 0
    stages=["preflight","tests","collect","label","train","evaluate","archive"] if args.stage=="all" else [args.stage]
    try:
        for stage in stages:
            phase={"probe":"preflight_tests","preflight":"preflight_tests","tests":"preflight_tests","collect":"data_and_labels","label":"data_and_labels","train":"training","evaluate":"evaluation","archive":"archive"}[stage]
            with ledger.phase(phase):
                if stage in ("probe","preflight"):preflight(output,probe_only=stage=="probe")
                elif stage=="tests":
                    from .evaluation import run_targeted_checks
                    checks=run_targeted_checks(output,ledger,external_xmls=args.checks_xml)
                    if not checks["all_automated_pass"]:raise ValueError("targeted checks failed or required coverage missing")
                elif stage=="collect":
                    require_checks(output)
                    from .data import validate_independent_initial_states
                    roots=load_json(output/"split_manifest.json")["roots"]
                    receipts=run_root_jobs(output,ledger,"collect",roots)
                    completed={receipt["root_id"] for receipt in receipts if receipt["status"]=="completed"}
                    rows=[load_json(output/"roots"/root["root_id"]/"root.json") for root in roots if root["root_id"] in completed]
                    validate_independent_initial_states(rows)
                    atomic_json(output/"raw_snapshot_index.json",dict(roots=rows,complete=len(rows)==24))
                    if any(r["status"]=="failed" for r in receipts):raise RuntimeError("collection child failed")
                    if len(rows)!=24:raise BudgetLimited("collection interrupted")
                elif stage=="label":
                    require_checks(output)
                    roots=load_json(output/"raw_snapshot_index.json")["roots"]
                    if args.root_id:
                        missing=set(args.root_id)-{root["root_id"] for root in roots}
                        if missing:raise ValueError("unknown frozen root IDs: "+", ".join(sorted(missing)))
                        roots=[root for root in roots if root["root_id"] in set(args.root_id)]
                    receipts=run_root_jobs(output,ledger,"label",roots)
                    build_training_dataset(output)
                    if any(r["status"]=="failed" for r in receipts):raise RuntimeError("labeling child failed")
                    if any(r["status"]!="completed" for r in receipts):raise BudgetLimited("labeling interrupted")
                elif stage=="train":
                    require_checks(output)
                    if (output/"heldout_access.json").exists():raise ValueError("formal training prohibited after heldout access")
                    dataset=build_training_dataset(output)
                    command=[sys.executable,str(ROOT/"scripts/stage3/train.py"),"--dataset",str(output/"training_dataset.json"),"--output-dir",str(output/"model"),"--fingerprint",str(output/"run_binding.generated.json"),"--config",str(PACKAGE/"stage3_learning_config_v1_0.yaml"),"--deadline",str(time.time()+ledger.remaining(phase))]
                    receipt=run_subprocess(ledger,phase,"formal_training",command,output/"logs")
                    if receipt["status"]=="failed":raise RuntimeError("training child failed")
                    if receipt["status"]!="completed":raise BudgetLimited("training interrupted: "+receipt["status"])
                elif stage=="evaluate":
                    from .evaluation import run_evaluation
                    run_evaluation(output,ledger)
                elif stage=="archive":archive_report(output,ledger)
    except BudgetLimited as exc:
        atomic_json(output/"budget_stop.json",dict(status="BUDGET_LIMITED",reason=str(exc),elapsed_s=ledger.elapsed))
        archive_report(output,ledger);return 3
    except Exception as exc:
        atomic_json(output/"campaign_error.json",dict(status="FAILED",error_type=type(exc).__name__,error=str(exc),elapsed_s=ledger.elapsed))
        archive_report(output,ledger);return 2
    return 0


def require_checks(output):
    path=Path(output)/"targeted_checks.json"
    if not path.exists() or not load_json(path).get("all_automated_pass"):
        raise ValueError("successful targeted checks required before compute")
    if not (Path(output)/"run_binding.generated.json").exists():raise ValueError("server preflight binding required")


@contextmanager
def campaign_lock(output):
    """OS-released single runner lock; a crashed process leaves no stale owner."""
    directory=Path(output);directory.mkdir(parents=True,exist_ok=True)
    with (directory/"campaign.lock").open("a+b") as stream:
        stream.seek(0);stream.write(b"0");stream.flush();stream.seek(0)
        try:
            if os.name=="nt":
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as exc:raise ValueError("another campaign runner holds this output lock") from exc
        try:yield
        finally:
            if os.name=="nt":
                stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument("--output",type=Path);parser.add_argument("--job")
    args,_=parser.parse_known_args(argv)
    if args.output is not None and args.job is None:
        with campaign_lock(args.output):return _main(argv)
    return _main(argv)


if __name__=="__main__": raise SystemExit(main())
