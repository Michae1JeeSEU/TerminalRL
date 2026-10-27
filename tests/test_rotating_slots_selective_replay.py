"""Selection annotations are pure saved-data views, never maneuver authority."""
from copy import deepcopy
from types import SimpleNamespace

from terminal_marl.visualization.rotating_slots_controller import ReplayController


def controller():
    callback=dict(time=20.,planning_timing=dict(snapshot_time=0.,effective_time=20.),
        actual_delivery_status="COMMITTED",actual_delivery_time=20.,
        selective_replanning=dict(method="RULE_LOCAL_K8",scope=dict(replan_ids=["a"],fixed_background_ids=["b"],mandatory_ids=["a"],objective_frame_id="frame"),
            background_blockers=[dict(source="a",target="b",reason="RESOURCE_CONFLICT")],relations=[],ages={}))
    value=object.__new__(ReplayController)
    value.time=10.;value.bundle=SimpleNamespace(data=dict(decisions=[callback],trajectories=dict(h="authoritative")))
    return value


def test_X02_scope_labels_blockers_countdown_and_no_physics_change():
    value=controller();before=deepcopy(value.bundle.data)
    overlay=value.selective_overlay()
    assert overlay["labels"]=={"a":"M/S","b":"B"}
    assert overlay["countdown_s"]==10.
    assert overlay["status"]=="AWAITING_RECORDED_DELIVERY"
    assert overlay["blockers"][0]["target"]=="b"
    assert overlay["human_review"]=="HUMAN_PENDING"
    overlay["selected_ids"].append("invalid")
    assert value.bundle.data==before
    value.time=20.
    assert value.selective_overlay()["status"]=="COMMITTED"


def test_selection_from_future_snapshot_is_hidden_and_old_replay_supported():
    value=controller();value.time=-1.
    assert value.selective_overlay() is None
    value.bundle.data["decisions"]=[];value.time=20.
    assert value.selective_overlay() is None


def test_unknown_frame_without_scope_has_no_fabricated_groups():
    value=controller()
    value.bundle.data["decisions"][0]["selective_replanning"].update(scope=None,status="LOCAL_SEARCH_INCOMPLETE")
    view=value.selective_overlay()
    assert view["selected_ids"]==[] and view["background_ids"]==[] and view["objective_frame_id"] is None
