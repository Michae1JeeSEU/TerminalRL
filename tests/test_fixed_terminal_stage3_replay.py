import json
from dataclasses import replace
import pytest
from terminal_marl.configs.fixed_terminal_stage1 import DEFAULT_CONFIG
from terminal_marl.core.fixed_terminal.value_analysis import replay_episode
from terminal_marl.scripts.evaluate_fixed_terminal_stage2 import run


def test_replay_before_decision_and_full_request_ledger(tmp_path):
    source = tmp_path/"episode"
    original = run(source, "fcfs", config=replace(DEFAULT_CONFIG, horizon=10))
    initial = replay_episode(source, 0)
    assert all(a.plan is None for a in initial.aircraft.values())
    before_five = replay_episode(source, 5)
    assert before_five.time == 5 and any(a.plan for a in before_five.aircraft.values())
    assert all(e["time"] < 5 or e["event"] in ("failed_energy", "normal_landed", "emergency_landed") for e in before_five.events)
    complete = replay_episode(source)
    assert complete.report()["requests"] == original["requests"]


def test_replay_rejects_corrupt_plan_and_frozen_configuration(tmp_path):
    source = tmp_path/"episode"
    run(source, "fcfs", config=replace(DEFAULT_CONFIG, horizon=2))
    path = source/"decisions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["normal_plans"][0]["completion"] += 1
    path.write_text("\n".join(json.dumps(row) for row in rows)+"\n")
    with pytest.raises(ValueError, match="completion"):
        replay_episode(source)
    frozen = source/"frozen_input.json"
    data = json.loads(frozen.read_text())
    data["config"]["energy"]["hold"] *= 2
    frozen.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="mismatch"):
        replay_episode(source)
