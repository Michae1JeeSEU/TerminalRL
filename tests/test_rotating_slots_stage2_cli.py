"""CLI exit status follows the archived campaign state after run/resume."""
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.server import run_rotating_slots_stage2 as cli
from terminal_marl.core.rotating_slots import stage2_campaign as campaign


@pytest.mark.parametrize("action",("run","resume"))
@pytest.mark.parametrize("status",("INVALID_STOP","COMPLETED_3600","BUDGET_LIMITED"))
def test_run_resume_exit_follows_persisted_status_after_output(monkeypatch,action,status):
    output=Path("archived-campaign")
    result={"report":{"archived":True},"status":"RETURN_VALUE_IS_NOT_AUTHORITATIVE"}
    observed=[]
    stream=io.StringIO()

    def run_campaign(path,*,target):
        assert path==output and target==60.
        observed.append("archived_and_returned")
        return result

    def load_json(path):
        assert path==output/"campaign.json"
        assert observed==["archived_and_returned"]
        assert json.loads(stream.getvalue())==result
        observed.append("persisted_status_read")
        return {"status":status}

    monkeypatch.setattr(campaign,"run_campaign",run_campaign)
    monkeypatch.setattr(campaign,"load_json",load_json)
    monkeypatch.setattr("sys.argv",["stage2",action,"--output",str(output),"--target","60"])
    with redirect_stdout(stream):
        if status=="INVALID_STOP":
            with pytest.raises(SystemExit) as error: cli.main()
            assert error.value.code==1
        else:
            assert cli.main() in (None,0)
    assert observed==["archived_and_returned","persisted_status_read"]
    assert json.loads(stream.getvalue())==result


@pytest.mark.parametrize("action",("status","report"))
def test_invalid_stop_status_and_report_queries_still_succeed(monkeypatch,action,capsys):
    persisted={"status":"INVALID_STOP","archived":True}
    monkeypatch.setattr(campaign,"load_json",lambda path:persisted)
    monkeypatch.setattr(campaign,"common_boundary",lambda path:60.)
    monkeypatch.setattr(campaign,"CampaignBudget",lambda path:SimpleNamespace(elapsed=12.))
    monkeypatch.setattr(campaign,"report_campaign",lambda path:{"campaign":persisted,"report_complete":True})
    monkeypatch.setattr("sys.argv",["stage2",action,"--output","archived-campaign"])
    assert cli.main() in (None,0)
    result=json.loads(capsys.readouterr().out)
    assert result["campaign"]==persisted
    if action=="status":
        assert result["common_completed_time_s"]==60. and result["elapsed_s"]==12.
    else:
        assert result["report_complete"]


def test_prepare_keeps_existing_output_and_exit_semantics(monkeypatch,capsys):
    result={"status":"PREPARED","archived":True}
    monkeypatch.setattr(campaign,"prepare_campaign",lambda *args,**kwargs:result)
    def unexpected_read(path): raise AssertionError("prepare does not query campaign status for an exit code")
    monkeypatch.setattr(campaign,"load_json",unexpected_read)
    monkeypatch.setattr("sys.argv",["stage2","prepare","--output","archived-campaign"])
    assert cli.main() in (None,0)
    assert json.loads(capsys.readouterr().out)==result
