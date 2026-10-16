from pathlib import Path

from terminal_marl.core.rotating_slots.extra_fixtures import (cumulative_peak, actual_cumulative_peak,
    mixed_conservation, capacity_above_slots)
from terminal_marl.core.rotating_slots.qualification import Evidence
from terminal_marl.core.rotating_slots.fixtures import FixtureBlocked
import pytest


def evidence(case,tmp_path):
    return Evidence(case,tmp_path,{"code_commit":"test-fixture","command":"pytest"})


def test_cumulative_peak_uses_actual_candidate_and_preserves_prior(tmp_path):
    e=evidence("E07",tmp_path)
    assert cumulative_peak(e)
    assert all(a["passed"] for a in e.assertions)
    assert e.worlds[0].rejections[-1]["reason"]=="PEAK_BUDGET"
    assert e.worlds[1].aircraft["peak"].peak_seconds>60.


def test_actual_cumulative_peak_spans_executed_transfer_and_next_candidate(tmp_path):
    e=evidence("E07-actual",tmp_path)
    assert actual_cumulative_peak(e)
    assert all(a["passed"] for a in e.assertions)
    assert e.worlds[0].aircraft["actual_peak"].peak_seconds>0
    assert e.worlds[0].rejections[-1]["reason"]=="PEAK_BUDGET"


def test_mixed_outcomes_are_one_executed_world(tmp_path):
    e=evidence("E14",tmp_path)
    assert mixed_conservation(e)
    assert len(e.worlds)==1 and e.worlds[0].failed
    assert all(a["passed"] for a in e.assertions)


def test_capacity_search_budget_is_explicit_and_cannot_award_pass(tmp_path):
    e=evidence("A05",tmp_path)
    with pytest.raises(FixtureBlocked,match="bounded chain search exhausted"):
        capacity_above_slots(e,max_seconds=0.)
    assert not e.assertions
