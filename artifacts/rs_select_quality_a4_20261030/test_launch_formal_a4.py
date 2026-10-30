"""Temporary fixtures only: never start a subprocess or a world."""
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from scripts.stage3 import run_selective_reoptimization_a as base
from scripts.stage3 import run_selective_quality_gap_a4 as a4

PATH=Path(__file__).with_name("launch_formal_a4.py")
spec=importlib.util.spec_from_file_location("formal_a4_launcher_fixture",PATH)
launcher=importlib.util.module_from_spec(spec); spec.loader.exec_module(launcher)


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True),encoding="utf8")


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    project=tmp_path/"project"; output=tmp_path/"output"; output.mkdir()
    driver=project/"scripts/stage3/run_selective_reoptimization_a.py"
    driver.parent.mkdir(parents=True); driver.write_text("# frozen fixture\n")
    monkeypatch.setattr(base,"PROJECT",project); monkeypatch.setattr(base,"DRIVER",driver)
    files={driver.relative_to(project).as_posix():base.file_hash(driver)}
    source=dict(commit=launcher.COMMIT,source_hash=base.digest(files),files=files)
    freeze=project/"freeze.json"; save(freeze,source)
    monkeypatch.setattr(launcher,"SOURCE_HASH",source["source_hash"])
    monkeypatch.setattr(launcher,"FREEZE_SHA256",launcher.sha(freeze))
    common={}
    for name in launcher.COMMON_SHA256:
        root=name.split("/")[1]
        value=dict(status="COMMON_INITIAL_READY",source_binding=dict(root=root)) if name.endswith("boundary_summary.json") else {}
        save(output/name,value); common[name]=launcher.sha(output/name)
    monkeypatch.setattr(launcher,"COMMON_SHA256",common)
    ledger=dict(a4.POLICY,settled_child_wall_s=100.,jobs={})
    save(output/"repair_budget_ledger.json",ledger)
    resource=dict(free_physical_cores=[16,17,18,19],memory_available_bytes=1000000000)
    d=NS(campaign=NS(resource_probe=lambda:resource))
    return NS(project=project,output=output,freeze=freeze,d=d,resource=resource,
              kwargs=dict(project=project,output=output,freeze=freeze,base=base,a4=a4,d=d))


@pytest.mark.parametrize("phase",("arm","audit"))
@pytest.mark.parametrize("child_exit",(0,7))
def test_four_async_starts_before_wait_and_parent_zero_does_not_hide_child_failure(fixture,phase,child_exit):
    f=fixture; events=[]; starts=[]
    if phase=="audit":
        for root,method,cpu in launcher.SPECS:
            home=f.output/"arms"/root/method
            save(home/"resolved_run_binding.json",dict(source_hash=launcher.SOURCE_HASH,evaluation_profile=launcher.PROFILE))
            save(home/"evaluation"/root/method/"result.json",dict(status="COMPLETE"))
    class FakePopen:
        def __init__(self,command,**kwargs):
            self.pid=4000+len(starts); self.command=command; starts.append(self); events.append("start")
            assert command[0]=="/home/user/envs/terminal_env/bin/python"
            assert kwargs["cwd"]==str(f.project) and kwargs["start_new_session"] is True
            assert all(kwargs["env"][key]=="1" for key in launcher.THREAD_ENV)
            assert kwargs["env"]["PYTHONPATH"]==str(f.project)+":/home/user/envs/terminal_stage2_v1/lib/python3.12/site-packages"
            root=command[command.index("--root")+1]; method=command[command.index("--method")+1]
            expected=f.output/"logs"/f"LAUNCHER_A4_{phase}_{root}_{method}.log"
            assert Path(kwargs["stdout"].name)==expected
            assert expected.name!=launcher.job_id(phase,root,method)+".log"
        def wait(self):
            assert len(starts)==4 and events[:4]==["start"]*4
            receipt=launcher.load(f.output/"receipts"/f"LAUNCHER_A4_{phase}.started.json")
            assert len(receipt["parents"])==4
            events.append("wait")
            root=self.command[self.command.index("--root")+1]; method=self.command[self.command.index("--method")+1]
            ledger=launcher.load(f.output/"repair_budget_ledger.json")
            ledger["jobs"][launcher.job_id(phase,root,method)]=dict(exit_code=child_exit,child_pid=self.pid+100,
                status="completed" if child_exit==0 else "failed")
            save(f.output/"repair_budget_ledger.json",ledger)
            path=f.output/"arms"/root/method/"evaluation"/root/method/("audit_result.json" if phase=="audit" else "result.json")
            save(path,dict(status="PASS" if phase=="audit" else "COMPLETE",full_a20_covered=True))
            return 0
    before={name:launcher.sha(f.output/name) for name in launcher.COMMON_SHA256}
    result=launcher.launch(phase,popen=FakePopen,**f.kwargs)
    assert events==["start"]*4+["wait"]*4
    assert result["success"]==(child_exit==0)
    assert all(row["parent_exit_code"]==0 and row["child_exit_code"]==child_exit for row in result["parents"])
    started=launcher.load(f.output/"receipts"/f"LAUNCHER_A4_{phase}.started.json")
    assert len(started["evidence"]["common_files"])==10 and len(started["environment"])==6
    assert before=={name:launcher.sha(f.output/name) for name in before}
    json.dumps(result,allow_nan=False)
    with pytest.raises(ValueError,match="already attempted|already exists"):
        launcher.launch(phase,popen=lambda *a,**k:pytest.fail("duplicate launch"),**f.kwargs)


@pytest.mark.parametrize("collision",("launcher_log","bounded_log","ledger","arm","receipt"))
def test_preflight_rejects_existing_evidence_before_any_spawn(fixture,collision):
    f=fixture; root,method,cpu=launcher.SPECS[-1]
    if collision=="launcher_log": path=f.output/"logs"/f"LAUNCHER_A4_arm_{root}_{method}.log"
    elif collision=="bounded_log": path=f.output/"logs"/(launcher.job_id("arm",root,method)+".log")
    elif collision=="arm": path=f.output/"arms"/root/method/"old.json"
    elif collision=="receipt": path=f.output/"receipts"/"LAUNCHER_A4_arm.started.json"
    else:
        path=f.output/"repair_budget_ledger.json"; ledger=launcher.load(path)
        ledger["jobs"][launcher.job_id("arm",root,method)]=dict(status="failed",child_pid=None)
        save(path,ledger)
    if collision!="ledger": save(path,{"original":"immutable"})
    old=path.read_bytes()
    with pytest.raises(ValueError):
        launcher.launch("arm",popen=lambda *a,**k:pytest.fail("preflight spawned"),**f.kwargs)
    assert path.read_bytes()==old


@pytest.mark.parametrize("mismatch",("freeze_bytes","commit","source","common","cpu","memory"))
def test_preflight_rejects_identity_and_resource_mismatch(fixture,mismatch,monkeypatch):
    f=fixture
    if mismatch in ("freeze_bytes","commit","source"):
        source=launcher.load(f.freeze); source["commit" if mismatch=="commit" else "source_hash"]="changed"
        save(f.freeze,source)
        if mismatch!="freeze_bytes": monkeypatch.setattr(launcher,"FREEZE_SHA256",launcher.sha(f.freeze))
    elif mismatch=="common": save(f.output/next(iter(launcher.COMMON_SHA256)),{})
    elif mismatch=="cpu": f.resource["free_physical_cores"]=[16,17,18]
    else: f.resource["memory_available_bytes"]=0
    with pytest.raises(ValueError):
        launcher.launch("arm",popen=lambda *a,**k:pytest.fail("invalid preflight spawned"),**f.kwargs)


def test_fixed_exportfix_freeze_and_exact_default_paths_are_preserved():
    freeze=PATH.with_name("comparison_source_freeze_exportfix1.json")
    source=launcher.load(freeze)
    assert launcher.sha(freeze)==launcher.FREEZE_SHA256
    assert source["commit"]==launcher.COMMIT and source["source_hash"]==launcher.SOURCE_HASH
    assert launcher.PROJECT.as_posix()=="/home/user/projects/terminal"
    assert launcher.OUTPUT.as_posix()=="/home/user/training_outputs/terminal/rs_selective_quality_a4_20261030"
    assert [row[2] for row in launcher.SPECS]==[16,17,18,19]
