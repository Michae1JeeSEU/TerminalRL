"""V7 separates proved rescue need, reserved opportunities and actual flight."""
from dataclasses import replace

import pytest

from terminal_marl.core.rotating_slots.rescue_rules import (
    RESCUE_PROFILE_V6, RESCUE_PROFILE_V7, assess_rescue,
)
from terminal_marl.core.rotating_slots.stage2_planning import CandidateProvider, PlanningNode
from terminal_marl.core.rotating_slots.trajectory import generate_final, generate_transfer
from terminal_marl.core.rotating_slots.world import RotatingSlotWorld, STAGE2_POLICY_ID
from terminal_marl.core.rotating_slots.replay import ReplayBundle
from terminal_marl.visualization.rotating_slots_replay import ReplayController, _recorded_rescue_status


def world():
    w = RotatingSlotWorld(experiment_policy_id=STAGE2_POLICY_ID)
    w.rescue_rule_profile = RESCUE_PROFILE_V7
    w.auto_emergency = False
    return w


def final(w, key, *, emergency=False, pad=5, start=0.):
    a = w.aircraft[key]
    p = generate_final(key, a.ring, a.slot, pad, start, emergency=emergency,
                       source_version=a.version, config=w.config).plan
    assert p is not None
    return p


def test_v7_need_does_not_require_an_opportunity_but_default_v6_is_unchanged():
    inputs = dict(normal_energy_lower_bound=.003, rescue_feasible=False)
    assert not assess_rescue(.24, .001, **inputs).rescue_required
    assert not assess_rescue(.24, .001, profile=RESCUE_PROFILE_V6, **inputs).rescue_required
    result = assess_rescue(.24, .001, profile=RESCUE_PROFILE_V7, **inputs)
    assert result.rescue_required and result.normal_status == "INFEASIBLE"
    assert result.rescue_reason == "NORMAL_RESERVE_IMPOSSIBLE"
    assert result.rescue_slack_seconds is None


def test_unknown_rescue_declares_waiting_once_and_keeps_unrelated_normal_resources(monkeypatch):
    w = world()
    w.seed_hold("need", "outer", 18, soc=.24)
    w.seed_hold("normal", "inner", 0, soc=.85)
    searched = []
    monkeypatch.setattr(w, "_search_rescue_opportunity", lambda a: searched.append(a.id))
    before = (w.slots.reservations, w.pads.reservations)
    w._update_emergency()
    assert w.aircraft["need"].emergency.declared  # first detection, before search
    w.handle_emergencies()
    w.time = 5.
    w._update_emergency()
    w.handle_emergencies()
    e = w.aircraft["need"].emergency
    assert e.eligible and e.declared and e.state == "EMERGENCY_WAITING"
    assert e.normal_status == "INFEASIBLE" and e.reference_missing
    assert e.rescue_reason == "NORMAL_RESERVE_IMPOSSIBLE"
    assert e.declaration_time == 0. and e.start_time is None and not e.charged
    assert searched == ["need", "need"]  # declared UNKNOWN is searched again
    trigger = next(r for r in w.events if r["type"] == "emergency_triggered")
    assert trigger["details"]["deadline_basis"] == "necessary_normal_energy_bound"
    assert before == (w.slots.reservations, w.pads.reservations)
    assert [r["type"] for r in w.events].count("emergency_triggered") == 1
    assert [r["type"] for r in w.events].count("emergency_declared") == 1
    assert w.costs["emergency"] == 0.
    assert w.submit((final(w, "normal", start=5.),)).accepted


def test_unknown_normal_search_and_soc_warning_do_not_declare(monkeypatch):
    w = world()
    w.seed_hold("warning", "inner", 0, soc=.34)
    monkeypatch.setattr(w, "_search_rescue_opportunity", lambda a: None)
    w._update_emergency()
    w.handle_emergencies()
    e = w.aircraft["warning"].emergency
    assert e.warning and e.normal_status == "UNKNOWN"
    assert not e.declared and not e.eligible and not e.pending
    assert w.submit((final(w, "warning"),)).accepted
    assert w.active["warning"].kind == "FINAL_NORMAL"


def test_dispatch_proof_event_does_not_claim_a_certified_rescue(monkeypatch):
    w = world()
    w.seed_hold("need", "outer", 18, soc=.24)
    monkeypatch.setattr(w, "_search_rescue_opportunity", lambda a: None)
    w.handle_emergencies()
    trigger = next(r for r in w.events if r["type"] == "emergency_triggered")
    declaration = next(r for r in w.events if r["type"] == "emergency_declared")
    assert trigger["details"]["deadline_basis"] == "necessary_normal_energy_bound"
    assert trigger["details"]["q_e"] is None
    assert declaration["details"]["opportunity_status"] == "UNKNOWN"
    assert declaration["details"]["trajectory_hash"] is None


def test_soc_rank_does_not_replace_proof_or_certified_last_start():
    # 29% can require rescue with a certified opportunity whose last start is
    # close, while its normal status remains UNKNOWN. 24% independently proves
    # reserve impossible even with no certified rescue opportunity.
    known = assess_rescue(.29, .001, rescue_soc=.28, rescue_feasible=True,
                          profile=RESCUE_PROFILE_V7)
    unknown = assess_rescue(.24, .001, profile=RESCUE_PROFILE_V7)
    assert known.rescue_reason == "LAST_RESCUE_START" and known.normal_status == "UNKNOWN"
    assert unknown.rescue_reason == "NORMAL_RESERVE_IMPOSSIBLE"


def test_active_transfer_declares_pending_without_cancelling_or_charging(monkeypatch):
    w = world()
    w.seed_hold("moving", "outer", 0, soc=.85)
    move = generate_transfer("moving", "outer", 0, 0, 0., config=w.config).plan
    assert move is not None and w.submit((move,)).accepted
    w.aircraft["moving"] = replace(w.aircraft["moving"], soc=.24)
    commitments = (w.slots.reservations, w.pads.reservations, dict(w.dependencies))
    monkeypatch.setattr(w, "_search_rescue_opportunity", lambda a: pytest.fail("active transfer must finish"))
    w._update_emergency()
    w.handle_emergencies()
    w._update_emergency()
    e = w.aircraft["moving"].emergency
    assert e.declared and e.pending and not e.eligible
    assert e.declaration_time == 0. and not e.charged and e.start_time is None
    assert w.active["moving"] == move and w.aircraft["moving"].status == "TRANSFER_O_M"
    trigger = next(r for r in w.events if r["type"] == "emergency_triggered")
    assert trigger["details"]["deadline_basis"] == "necessary_normal_energy_bound"
    assert trigger["details"]["assessment_ring"] == move.target_ring
    assert trigger["details"]["assessment_time"] == move.end_time
    assert commitments == (w.slots.reservations, w.pads.reservations, dict(w.dependencies))
    assert w.costs["emergency"] == 0.
    assert [r["type"] for r in w.events].count("emergency_declared") == 1


def test_prediction_uses_same_v7_bound_without_certifying_rescue_opportunity():
    w = world()
    a = w.seed_hold("need", "outer", 0, soc=.24)
    node = PlanningNode(a.ring, a.slot, 0., a.soc, 0., a.version)
    p = CandidateProvider()
    snapshot = w.build_planning_snapshot()
    assert p._emergency_prediction(snapshot, a, node) == "RESCUE_REQUIRED"
    assert p._stop_prediction(snapshot, "RESCUE_REQUIRED")
    w.rescue_rule_profile = RESCUE_PROFILE_V6
    assert p._emergency_prediction(w.build_planning_snapshot(), a, node) == "WARNING"


def test_required_rescue_preserves_ground_service_and_started_dependency():
    w = world()
    w.seed_service("ground", 0, remaining=200.)
    w.seed_hold("landing", "inner", 0, soc=.85)
    w.seed_hold("incoming", "middle", 1, soc=.85)
    departure = final(w, "landing", start=.001)
    incoming = generate_transfer("incoming", "middle", 1, 0, 0.,
                                 durations=[40.], config=w.config).plan
    assert incoming is not None and w.submit((departure,)).accepted
    assert w.submit((incoming,)).accepted
    assert departure.trajectory_hash in w.dependencies[incoming.trajectory_hash]
    w.seed_hold("need", "outer", 18, soc=.24)
    w._update_emergency()
    w.handle_emergencies()
    assert w.soft["landing"] == departure and w.active["incoming"] == incoming
    assert w.aircraft["ground"].status == "LANDED_SERVICE"
    assert any(r.aircraft_id == "ground" and r.end == 200. for r in w.pads.reservations)


def test_future_opportunity_is_reserved_and_charge_waits_for_start(monkeypatch):
    w = world()
    a = w.seed_hold("need", "outer", 18, soc=.24)
    p = final(w, a.id, emergency=True, pad=0, start=5.)
    q = w._energy(p, p.start_time, p.end_time).soc_used
    monkeypatch.setattr(w, "_search_rescue_opportunity", lambda a: (p, q, 5.))
    w.handle_emergencies()
    e = w.aircraft[a.id].emergency
    assert e.declared and e.state == "EMERGENCY_RESERVED" and not e.charged
    assert w.soft[a.id] == p and a.id not in w.active
    declaration_time = e.declaration_time
    assert w.costs["emergency"] == 0.
    w.handle_emergencies()
    w.time = 5.
    w._start_due()
    e = w.aircraft[a.id].emergency
    assert e.declaration_time == declaration_time
    assert e.state == "FINAL_EMERGENCY" and e.charged and e.start_time == 5.
    assert w.costs["emergency"] == 3600.
    w._start_due()
    assert w.costs["emergency"] == 3600.


def test_recorded_display_distinguishes_wait_reserved_active_and_legacy(tmp_path):
    w = world()
    a = w.seed_hold("need", "outer", 18, soc=.24)
    w._update_emergency()
    # Use actual record serialization/loading; no emergency-state schema fields
    # are added, and a historical undeclared low-SOC row stays undeclared.
    w._record(force=True)
    source = w.to_dict()
    b = ReplayBundle.write(tmp_path / "bundle", record=source,
                           code_commit="fixture", seed=0, command="pytest")
    render = ReplayController(b).snapshot().uavs[0]
    assert render.rescue_status == "WAIT" and render.emergency_landing
    aircraft = dict(source["states"][-1]["aircraft"][0])
    p = final(w, a.id, emergency=True, pad=0, start=5.)
    frame = {"pads": [{"aircraft_id": a.id, "plan_hash": p.trajectory_hash}]}
    assert _recorded_rescue_status(aircraft, frame, {p.trajectory_hash: p}) == "RESERVED"
    aircraft.update(status="FINAL_EMERGENCY")
    assert _recorded_rescue_status(aircraft, {}, {}) == "ACTIVE"
    aircraft.update(status="HOLD_OUTER", emergency={"declared": False})
    assert _recorded_rescue_status(aircraft, frame, {p.trajectory_hash: p}) is None
    aircraft.pop("emergency")
    assert _recorded_rescue_status(aircraft, {}, {}) is None
