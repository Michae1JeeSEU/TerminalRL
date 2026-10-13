import json
from dataclasses import asdict, replace
from terminal_marl.configs.fixed_terminal_stage1 import DEFAULT_CONFIG
from terminal_marl.configs.fixed_terminal_stage2 import DEFAULT_SCHEDULER_CONFIG
from terminal_marl.scripts.evaluate_fixed_terminal_stage2 import run, aggregate, main


def test_runner_serialization_label_full_counts_and_max_delay(tmp_path):
    config = replace(DEFAULT_CONFIG, horizon=10)
    result = run(tmp_path/"fcfs", "fcfs", config=config)
    assert result["scheduler"] == "fcfs"
    frozen = json.loads((tmp_path/"fcfs"/"frozen_input.json").read_text())
    assert frozen["scheduler"] == "fcfs" and frozen["scheduler_config"] == asdict(DEFAULT_SCHEDULER_CONFIG)
    assert "selected_candidates_jointly_committed" in frozen["assumptions"]["normal_domain"]
    assert result["assumptions"] == frozen["assumptions"]
    assert "UNKNOWN_never_authorizes_emergency" in result["assumptions"]["emergency_domain"]
    assert "normal_domain_infeasibility_certificate_required" in result["assumptions"]["emergency_domain"]
    for filename in ("requests.jsonl", "events.jsonl", "decisions.jsonl"):
        rows = [json.loads(line) for line in (tmp_path/"fcfs"/filename).read_text().splitlines()]
        assert rows
    assert sum(result["summary"]["outcomes"].values()) == result["summary"]["request_count"]
    normal = [r for r in result["requests"] if r["outcome"] == "normal_landed"]
    assert result["summary"]["normal_delay_max_seconds"] == max((r["delay_seconds"] for r in normal), default=None)
    rows = aggregate(tmp_path, tmp_path/"table")
    assert len(rows) == 1 and rows[0]["method"] == "fcfs"
    assert (tmp_path/"table.csv").is_file() and (tmp_path/"table.json").is_file()


def test_stage2_defaults_do_not_change_stage1_physics():
    from terminal_marl.configs import fixed_terminal_stage2
    assert not hasattr(fixed_terminal_stage2, "TerminalConfig")
    assert asdict(DEFAULT_CONFIG)["rings"] == ({"radius": 12., "height": 6.}, {"radius": 20., "height": 10.}, {"radius": 28., "height": 14.})
    assert [DEFAULT_CONFIG.operational_capacity(i) for i in range(3)] == [6, 8, 9]
