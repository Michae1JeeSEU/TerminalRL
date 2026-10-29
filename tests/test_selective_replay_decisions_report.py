"""Saved-evidence semantics; no simulator or solver dependency."""
import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "replay_report", Path(__file__).parents[1] / "scripts/stage3/report_selective_replay_decisions.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def test_uid_is_sorted_complete_states_string_ids():
    states = [{"aircraft": [{"id": "initial_inner_00"}, {"id": "initial_ground_pad0"}]},
              {"aircraft": [{"id": "req_000010"}, {"id": "req_000002"}]}]
    assert report.uid_mapping(states) == {
        "initial_ground_pad0": 0, "initial_inner_00": 1, "req_000002": 2, "req_000010": 3}


def test_equal_time_uses_event_order_and_half_open_completion():
    actions = [{"start_event_index": 3, "start_time": 20.0, "end_time": 60.0,
                "completion_event_index": 12}]
    assert report.active_at(actions, 20.0, 4) == actions
    assert report.active_at(actions, 20.0, 2) == []
    assert report.active_at(actions, 60.0, 13) == []
    assert report.active_at(actions, 59.0, 13) == []


def test_expanded_attempt_can_return_previous_scope():
    data = {"selective_replanning": {"actual_k": 1,
            "calls": [{"scope": {"replan_ids": ["a"]}},
                      {"scope": {"replan_ids": ["a", "b"]}}],
            "scope": {"actual_k": 1, "replan_ids": ["a"]}}}
    selected = report.selection(data)
    assert selected["initial_ids"] == ["a"]
    assert selected["expanded_ids"] == ["b"]
    assert selected["returned_ids"] == ["a"]
    assert selected["returned_scope_size"] == 1


def fixture(status):
    task = {"snapshot_time": 0., "effective_time": 20., "status": status}
    record = {"time": 20., "aircraft_choices": [{"aircraft_id": "a", "selected_kind": "MOVE",
                                               "selected_pad_ids": [3]}],
              "selective_replanning": {"actual_k": 1, "calls": [{"scope": {"replan_ids": ["a"]}}],
                                       "scope": {"actual_k": 1, "replan_ids": ["a"]}}}
    event = {"type": "normal_task_started", "time": 0., "details": task}
    states = [{"time": 0., "aircraft": [{"id": "a"}]}]
    return report.extract("test", [record], [task], [event], states, {})


def test_rejected_move_and_predicted_pad_are_not_executed():
    result = fixture("REJECTED")
    row = result["rounds"][0]
    assert row["actual_actions"] == []
    assert row["aircraft_decisions"][0]["selected_kind"] == "MOVE"
    assert row["aircraft_decisions"][0]["outcome"] == "NO_NEW_ACTION"
    assert "P3" not in report.decisions_text(row)


def test_pending_has_no_fabricated_delivery_or_action():
    result = fixture("REVALIDATING")
    row = result["rounds"][0]
    assert row["finished_time"] is None
    assert row["actual_delivery_time"] is None
    assert row["aircraft_decisions"][0]["outcome"] == "TASK_PENDING"
    assert row["actual_actions"] == []


def test_previous_action_maintained_without_new_submission():
    assert report.execution_outcome("COMMITTED", [], [{"kind": "TRANSFER"}]) == "MAINTAIN_PREVIOUS_ACTION"
    assert report.execution_outcome("COMMITTED", [], []) == "NO_NEW_ACTION"
