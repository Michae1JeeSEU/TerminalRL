"""Small v6 rescue fixtures: warning, certified opportunities, commitments."""
from dataclasses import replace
import json

import pytest

from terminal_marl.core.rotating_slots import world as world_module
from terminal_marl.core.rotating_slots.energy import evaluate_power, integrate_energy
from terminal_marl.core.rotating_slots.models import ValidationResult
from terminal_marl.core.rotating_slots.rescue_rules import (
    RESCUE_PROFILE_V6, RESCUE_BUFFER_SECONDS_V6, assess_rescue,
    normal_energy_lower_bound,
)
from terminal_marl.core.rotating_slots.trajectory import generate_final, generate_transfer
from terminal_marl.core.rotating_slots.world import (
    EmergencyStatus, RotatingSlotWorld, STAGE2_POLICY_ID, evaluate_planning_snapshot,
)


def world():
    w = RotatingSlotWorld(experiment_policy_id=STAGE2_POLICY_ID)
    w.rescue_rule_profile = RESCUE_PROFILE_V6
    w.auto_emergency = False
    return w


def final(w, key, pad=5, start=0., durations=(35.,), emergency=False):
    a = w.aircraft[key]
    p = generate_final(key, a.ring, a.slot, pad, start, durations=durations,
        emergency=emergency, source_version=a.version, config=w.config).plan
    assert p is not None
    return p


def rescue(w, key="rescue", soc=.2):
    return w.seed_hold(key, "outer", 18, soc=soc)


def test_profile_field_is_stage2_only_for_historical_world_checkpoint_inventory():
    assert "rescue_rule_profile" not in vars(RotatingSlotWorld())
    assert RotatingSlotWorld(experiment_policy_id=STAGE2_POLICY_ID).rescue_rule_profile == "legacy_v5"


def test_unknown_high_soc_is_a_risk_label_without_an_alert_or_forcing():
    assessed = assess_rescue(.8, .001)
    assert assessed.normal_status == "UNKNOWN"
    assert not assessed.warning and not assessed.rescue_required
    assert assessed.normal_margin is None and assessed.rescue_slack_seconds is None


def test_single_normal_path_deficit_is_warning_not_global_infeasibility():
    assessed = assess_rescue(.5, .001, normal_path_soc=.3, normal_path_wait_seconds=10.)
    assert assessed.warning and assessed.normal_margin == pytest.approx(-.06)
    assert assessed.normal_status == "UNKNOWN" and not assessed.rescue_required
    assert not assess_rescue(.2, .001, rescue_feasible=False).rescue_required


def test_certified_rescue_lead_accounts_for_pad_wait_and_fifteen_second_buffer():
    assert RESCUE_BUFFER_SECONDS_V6 == 15.
    safe = assess_rescue(.32, .001, rescue_soc=.2, rescue_wait_seconds=100., rescue_feasible=True)
    urgent = assess_rescue(.32, .001, rescue_soc=.2, rescue_wait_seconds=110., rescue_feasible=True)
    unknown = assess_rescue(.32, .001, rescue_soc=.2, rescue_wait_seconds=110.)
    assert not safe.rescue_required
    assert urgent.rescue_required and urgent.rescue_reason == "LAST_RESCUE_START"
    assert urgent.rescue_slack_seconds == pytest.approx(10.)
    assert not unknown.rescue_required and unknown.rescue_slack_seconds is None


def test_soc_warning_remains_normal_schedulable_and_not_forced():
    w = world()
    w.seed_hold("normal", "inner", 0, soc=.34)
    w._update_emergency()
    w.handle_emergencies()
    a = w.aircraft["normal"]
    assert a.emergency.warning and not a.emergency.eligible and not a.emergency.declared
    p = final(w, "normal")
    assert w.submit((p,)).accepted
    assert w.active["normal"].kind == "FINAL_NORMAL"
    assert w.metrics()["cost_emergency"] == 0.


def test_disjoint_normal_final_and_certified_rescue_are_both_accepted():
    w = world()
    w.seed_hold("normal", "inner", 0, soc=.85)
    rescue(w)
    w.handle_emergencies()
    p = final(w, "normal")
    assert w.active["rescue"].pad_id != p.pad_id
    snap = w.build_planning_snapshot()
    evaluated = evaluate_planning_snapshot(snap, (p,))
    assert evaluated.accepted and w.submit((p,)).accepted
    assert set(w.active) == {"normal", "rescue"}
    assert snap.emergency_monitor["rescue_rule_profile"] == RESCUE_PROFILE_V6
    assert snap.emergency_monitor["protected_opportunities"] == (w.active["rescue"].trajectory_hash,)


def test_urgent_presence_without_a_certified_opportunity_never_blocks_normal():
    w = world()
    w.seed_hold("normal", "inner", 0, soc=.85)
    a = rescue(w)
    w.aircraft[a.id] = replace(a, emergency=EmergencyStatus(eligible=True))
    p = final(w, "normal")
    assert w._emergency_priority_wait(p) is None
    assert w.evaluate_submission((p,)).accepted


def test_same_pad_window_gives_the_actual_rescue_priority_in_snapshot_and_world():
    w = world()
    w.seed_hold("normal", "inner", 0, soc=.85)
    rescue(w)
    w.handle_emergencies()
    p = final(w, "normal", pad=0, durations=None)
    result = w.evaluate_submission((p,))
    assert result.reason == "EMERGENCY_PRIORITY_WAIT"
    assert result.details["blocked_by"] == "rescue"
    assert result.details["protected_opportunity"] == w.active["rescue"].trajectory_hash
    assert w.submit((p,)).reason == result.reason


def test_conflicting_unstarted_normal_final_is_replaced_in_the_rescue_transaction():
    w = world()
    w.seed_hold("normal","inner",0,soc=.85)
    ordinary = final(w,"normal",pad=0,start=1.,durations=None)
    assert w.submit((ordinary,)).accepted and "normal" in w.soft
    rescue(w)
    w.handle_emergencies()
    assert "rescue" in w.active and w.active["rescue"].pad_id == ordinary.pad_id
    assert "normal" not in w.soft and "normal" not in w.active
    assert w.aircraft["normal"].status == "HOLD_INNER"
    assert not any(r.aircraft_id == "normal" for r in w.pads.reservations)


def test_unknown_replacement_batch_keeps_the_existing_normal_plan_and_calendars(monkeypatch):
    w = world()
    w.seed_hold("normal","inner",0,soc=.85)
    ordinary = final(w,"normal",pad=0,start=1.,durations=None)
    assert w.submit((ordinary,)).accepted
    rescue(w)
    slots,pads = w.slots.reservations,w.pads.reservations
    monkeypatch.setattr(world_module,"validate_separation",
                        lambda *args,**kwargs: ValidationResult(False,"UNKNOWN","UNKNOWN",{}))
    w.handle_emergencies()
    assert w.soft["normal"] is ordinary and "rescue" not in w.active
    assert (w.slots.reservations,w.pads.reservations) == (slots,pads)
    assert not any(e["type"] == "soft_plan_cancelled" for e in w.events)


def test_warning_trigger_declaration_start_and_charge_are_separate_events():
    w = world()
    rescue(w, soc=.34)
    w.handle_emergencies()
    assert not w.aircraft["rescue"].emergency.eligible
    assert [e["type"] for e in w.events].count("rescue_warning") == 1
    w.aircraft["rescue"] = replace(w.aircraft["rescue"], soc=.2)
    w.handle_emergencies()
    kinds = [e["type"] for e in w.events]
    assert kinds.index("rescue_warning") < kinds.index("emergency_triggered") < kinds.index("emergency_declared") < kinds.index("maneuver_started")
    a = w.aircraft["rescue"]
    assert a.emergency.rescue_reason == "NORMAL_RESERVE_IMPOSSIBLE"
    assert a.emergency.declared and a.emergency.charged and w.costs["emergency"] == 3600.
    assert not any(r["reason"] in ("LANDING_RESERVE", "EMERGENCY_DISPOSITION_REQUIRED") for r in w.rejections)
    event = next(e for e in w.events if e["type"] == "rescue_opportunity_protected")
    assert event["details"]["pad_window_end"] > event["details"]["end_time"]
    json.dumps(w.events,allow_nan=False)


def test_bounded_unknown_search_does_not_force_or_discard_commitments(monkeypatch):
    w = world()
    rescue(w)
    monkeypatch.setattr(world_module, "evaluate_planning_snapshot",
        lambda *args, **kwargs: ValidationResult(False,"UNKNOWN","UNKNOWN",{}))
    w.handle_emergencies()
    assert not w.aircraft["rescue"].emergency.eligible
    assert not w.aircraft["rescue"].emergency.declared
    assert w.aircraft["rescue"].emergency.normal_status == "INFEASIBLE"
    event = next(e for e in w.events if e["type"] == "rescue_opportunity_search_exhausted")
    assert event["details"]["status"] == "UNKNOWN"
    assert "no_infeasibility_proof" in event["details"]["reason"]


def test_search_visits_all_pads_before_trying_next_valid_duration(monkeypatch):
    w = world()
    a = rescue(w)
    actual = world_module.evaluate_planning_snapshot
    seen = []
    def check(snapshot, plans, **kwargs):
        p = plans[0]
        seen.append((p.pad_id,p.segments[0].duration))
        if len(seen) <= 6:
            return ValidationResult(False,"REJECTED","RESOURCE_CONFLICT",{})
        return actual(snapshot, plans, **kwargs)
    monkeypatch.setattr(world_module,"evaluate_planning_snapshot",check)
    opportunity = w._search_rescue_opportunity(a)
    assert opportunity is not None
    # Pad2 has no valid geometry on this fixture; all other pads get their
    # first valid duration before any pad receives its second alternative.
    assert [pad for pad, _ in seen[:5]] == [0,1,3,4,5]
    assert seen[5][0] == 0 and seen[5][1] > seen[0][1]


def test_active_ground_and_real_started_dependency_are_preserved():
    w = world()
    w.seed_service("ground", 0, remaining=200.)
    w.seed_hold("landing", "inner", 0, soc=.85)
    w.seed_hold("incoming", "middle", 1, soc=.85)
    departure = final(w,"landing",start=.001)
    incoming = generate_transfer("incoming","middle",1,0,0.,durations=[40.],config=w.config).plan
    assert incoming is not None and w.submit((departure,)).accepted and w.submit((incoming,)).accepted
    assert departure.trajectory_hash in w.dependencies[incoming.trajectory_hash]
    rescue(w)
    w.handle_emergencies()
    assert w.soft["landing"] == departure
    assert w.active["incoming"] == incoming
    assert w.aircraft["ground"].status == "LANDED_SERVICE"
    assert any(r.aircraft_id == "ground" and r.end == 200. for r in w.pads.reservations)
    w._yield_normal_finals()
    assert w.soft["landing"] == departure and w.active["incoming"] == incoming


def test_search_is_never_called_from_physics_warning_monitor(monkeypatch):
    w = world()
    rescue(w)
    monkeypatch.setattr(w,"_search_rescue_opportunity",lambda a: pytest.fail("search belongs to dispatch"))
    w._update_emergency()
    assert w.aircraft["rescue"].emergency.warning


def test_future_pad_release_opportunity_is_protected_before_execution_and_charge():
    w = world()
    for pad in range(6):
        w.seed_service("ground"+str(pad),pad,remaining=110.)
    a = rescue(w,soc=.34)
    w.aircraft[a.id] = replace(a,emergency=EmergencyStatus(eligible=True))
    opportunity = w._search_rescue_opportunity(w.aircraft[a.id])
    assert opportunity is not None and opportunity[0].start_time == 110.
    w.handle_emergencies()
    p = w.soft[a.id]
    assert p.start_time == 110. and a.id not in w.active
    assert w.aircraft[a.id].emergency.declared and not w.aircraft[a.id].emergency.charged
    assert w.costs["emergency"] == 0.
    assert p.trajectory_hash in w.build_planning_snapshot().emergency_monitor["protected_opportunities"]


def test_dispatch_priority_uses_fresh_opportunity_cost_and_wait(monkeypatch):
    w = world()
    a = w.seed_hold("a", "outer",0,soc=.2)
    z = rescue(w,"z")
    w.aircraft[a.id] = replace(a,emergency=EmergencyStatus(eligible=True,q_e=.19))
    w.aircraft[z.id] = replace(z,emergency=EmergencyStatus(eligible=True,q_e=.01))
    plans = {key:next(r.plan for pad in range(6) if (r:=generate_final(key,"outer",aircraft.slot,pad,0.,
        emergency=True,config=w.config)).plan is not None) for key,aircraft in w.aircraft.items()}
    monkeypatch.setattr(w,"_search_rescue_opportunity",lambda a:(plans[a.id],.01 if a.id=="a" else .19,0.))
    attempts = []
    monkeypatch.setattr(w,"submit",lambda plans: attempts.append(plans[0].aircraft_id) or w._result(True))
    w.handle_emergencies()
    assert attempts == ["z","a"]
    events = [e for e in w.events if e["type"] == "emergency_priority_selected"]
    assert events[0]["details"]["rescue_slack_seconds"] < events[1]["details"]["rescue_slack_seconds"]


def test_required_takeover_is_declared_even_when_second_reservation_loses_its_opportunity(monkeypatch):
    w = world()
    w.seed_hold("first","outer",0,soc=.2)
    rescue(w,"second")
    plans = {key:next(r.plan for pad in range(6) if (r:=generate_final(key,"outer",a.slot,pad,0.,
        emergency=True,config=w.config)).plan is not None) for key,a in w.aircraft.items()}
    searches,attempts = [],[]
    def search(a):
        searches.append(a.id)
        if a.id == "second" and attempts:
            return None
        return plans[a.id],.19 if a.id == "first" else .1,0.
    def submit(batch):
        p = batch[0]
        attempts.append(p.aircraft_id)
        if p.aircraft_id == "first":
            # Mock the first successful reservation changing the resource state.
            w.pads.reserve(p.pad_id,p.aircraft_id,p.end_time-13.5,p.end_time+120.,
                           kind="emergency",plan_hash=p.trajectory_hash)
            return w._result(True)
        return w._result(False,"RESOURCE_CONFLICT")
    monkeypatch.setattr(w,"_search_rescue_opportunity",search)
    monkeypatch.setattr(w,"submit",submit)
    w.handle_emergencies()
    assert attempts == ["first","second"]
    assert searches == ["first","second","second"]
    a = w.aircraft["second"]
    assert a.emergency.eligible and a.emergency.declared
    assert a.emergency.state == "EMERGENCY_WAITING"
    assert a.emergency.declaration_time == w.time and a.emergency.declaration_ring == "outer"
    assert a.emergency.rescue_reason == "NORMAL_RESERVE_IMPOSSIBLE"
    assert a.id not in w.active and a.id not in w.soft
    assert not a.emergency.charged and a.emergency.start_time is None
    assert w.metrics()["emergency_declared"] == 2
    assert w.metrics()["emergency_trigger_count"] == 2 and w.costs["emergency"] == 0.
    events = [e for e in w.events if e["aircraft_id"] == "second"]
    kinds = [e["type"] for e in events]
    assert kinds.index("emergency_triggered") < kinds.index("emergency_declared") < kinds.index("rescue_commitment_protected")
    assert sum(e["type"] == "emergency_declared" for e in events) == 1
    assert not any(e["type"] == "maneuver_started" for e in events)


def test_source_matched_normal_evidence_updates_warning_only_and_expires():
    w = world()
    w.seed_hold("normal","inner",0,soc=.5)
    row = dict(aircraft_id="normal",source_ring="inner",source_slot=0,source_version=0,
        source_time=0.,normal_path_soc=.3,normal_path_wait_seconds=0.)
    snap = w.build_planning_snapshot()
    result = w._scheduler_port.report_normal_evidence((row,),snap.version)
    assert result.accepted and result.details["accepted_ids"] == ("normal",)
    a = w.aircraft["normal"]
    assert a.emergency.warning and a.emergency.normal_status == "UNKNOWN"
    assert a.emergency.normal_margin == pytest.approx(-.05)
    assert not a.emergency.eligible and not a.emergency.declared
    assert not w.report_normal_evidence((row,),snap.version).accepted
    w.time = 5.001
    assert w._normal_estimate(a) == {}
    w._update_emergency()
    assert w.aircraft["normal"].emergency.normal_margin is None
    assert not w.aircraft["normal"].emergency.warning


def test_normal_evidence_cannot_apply_to_a_changed_source():
    w = world()
    w.seed_hold("normal","inner",0,soc=.5)
    row = dict(aircraft_id="normal",source_ring="inner",source_slot=1,source_version=0,
               source_time=0.,normal_path_soc=.3)
    result = w.report_normal_evidence((row,),w.build_planning_snapshot().version)
    assert result.accepted and result.details["ignored_ids"] == ("normal",)
    assert w.aircraft["normal"].emergency.normal_path_soc is None


@pytest.mark.parametrize("ring", ["outer","middle","inner"])
def test_positive_normal_energy_lower_bound_is_below_validated_normal_chain(ring):
    w = world()
    bound = normal_energy_lower_bound(w.config, ring)
    # Every complete normal route must descend at least ring height. Even a
    # direct emergency final's measured energy exceeds this necessary bound.
    p = next((r.plan for pad in range(6) if (r := generate_final("sample",ring,0,pad,0.,
        emergency=True,config=w.config)).plan is not None),None)
    assert p is not None
    actual = integrate_energy(p,p.start_time,p.end_time,config=w.config)
    assert actual.valid and 0. < bound < actual.soc_used
    a = assess_rescue(.25+bound/2.,.001,normal_energy_lower_bound=bound,
                      rescue_soc=.1,rescue_feasible=True)
    assert a.normal_status == "INFEASIBLE" and a.rescue_required
    assert a.rescue_reason == "NORMAL_RESERVE_IMPOSSIBLE"
    height = next(r.height_agl_m for r in w.config.airspace.rings if r.id == ring)
    floor_current = bound*w.config.energy.reference_capacity_Ah*3600./(height/w.config.vehicle.max_descent_speed_mps)
    for t in (p.start_time,(p.start_time+p.end_time)/2.,p.end_time):
        state = p.state(t)
        power = evaluate_power(state.v,state.a,w.config)
        assert power.valid and power.current_a >= floor_current
    # Pure normal completion chain, with the unchanged trajectory validators
    # used by generate_transfer/generate_final, independently exceeds the bound.
    chain, current_ring, t = [], ring, 0.
    while current_ring != "inner":
        move = generate_transfer("normal_chain",current_ring,0,0,t,config=w.config).plan
        assert move is not None
        chain.append(move)
        current_ring, t = move.target_ring,move.end_time
    landing = next(r.plan for pad in range(6) if (r:=generate_final("normal_chain","inner",0,pad,t,config=w.config)).plan is not None)
    chain.append(landing)
    assert sum(integrate_energy(p,p.start_time,p.end_time,config=w.config).soc_used for p in chain) > bound
