"""Bounded Stage2 urgency/normal-final priority and safety commitment checks."""
from dataclasses import replace
from copy import deepcopy

import pytest

from terminal_marl.core.rotating_slots.trajectory import generate_final, generate_transfer
from terminal_marl.core.rotating_slots.world import (
    EmergencyStatus, RotatingSlotWorld, STAGE2_POLICY_ID, emergency_priority,
    evaluate_planning_snapshot,
)


def world():
    w = RotatingSlotWorld(experiment_policy_id=STAGE2_POLICY_ID)
    w.auto_emergency = False
    w._update_emergency = lambda: False
    return w


def urgent(w, key, ring="outer", slot=18, soc=.3, q=.1):
    a = w.seed_hold(key, ring, slot, soc=soc)
    w.aircraft[key] = replace(a, emergency=EmergencyStatus(eligible=True, q_e=q))
    return w.aircraft[key]


def final(w, key="normal", start=0.):
    a = w.aircraft[key]
    p = generate_final(key, a.ring, a.slot, 0, start, source_version=a.version, config=w.config).plan
    assert p is not None
    return p


def test_urgency_uses_known_rescue_cost_holding_rate_and_explicit_fallback():
    w = world()
    a = urgent(w, "low_soc", soc=.2, q=.01)
    b = replace(a, id="higher_soc", soc=.3, emergency=EmergencyStatus(eligible=True, q_e=.25))
    assert emergency_priority(b, w.config)["key"] < emergency_priority(a, w.config)["key"]
    c = replace(a, ring="inner")
    assert emergency_priority(c, w.config)["holding_soc_per_second"] != emergency_priority(a, w.config)["holding_soc_per_second"]
    unknown = replace(a, emergency=EmergencyStatus(eligible=True))
    details = emergency_priority(unknown, w.config)
    assert details["q_e_unknown"] and details["fallback"] == "soc_over_holding_rate"
    assert details["holding_seconds"] == pytest.approx(unknown.soc / details["holding_soc_per_second"])


def test_dispatch_handles_later_id_with_less_holding_time_first(monkeypatch):
    w = world()
    urgent(w, "a_less_urgent", slot=0, soc=.3, q=.1)
    urgent(w, "z_more_urgent", slot=18, soc=.2, q=.1)
    attempts = []
    monkeypatch.setattr(w, "_try_final", lambda a, emergency: attempts.append(a.id) or False)
    w.handle_emergencies()
    assert attempts == ["z_more_urgent", "a_less_urgent"]
    event = next(e for e in w.events if e["type"] == "emergency_priority_queue")
    assert event["details"]["queue"][0]["aircraft_id"] == "z_more_urgent"


def test_waiting_guard_matches_pure_snapshot_evaluation_and_recovers():
    w = world()
    w.seed_hold("normal", "inner", 0, soc=.85)
    urgent(w, "rescue")
    p = final(w)
    snapshot = w.build_planning_snapshot()
    before = (dict(w.aircraft), dict(w.active), dict(w.soft), w.slots.reservations,
              w.pads.reservations, deepcopy(w.events), deepcopy(w._energy_state()))
    evaluated = evaluate_planning_snapshot(snapshot, (p,))
    assert not evaluated.accepted and evaluated.reason == "EMERGENCY_PRIORITY_WAIT"
    assert evaluated.details["blocked_by"] == "rescue"
    assert before == (dict(w.aircraft), dict(w.active), dict(w.soft), w.slots.reservations,
                      w.pads.reservations, w.events, w._energy_state())
    assert w.submit((p,)).reason == evaluated.reason
    w.aircraft["rescue"] = replace(w.aircraft["rescue"], soc=.85, emergency=EmergencyStatus())
    assert w.submit((p,)).accepted


def test_priority_guard_does_not_block_ordinary_transfer():
    w = world()
    w.seed_hold("normal", "outer", 0, soc=.85)
    urgent(w, "rescue")
    p = generate_transfer("normal", "outer", 0, 0, 0., config=w.config).plan
    assert p is not None and w._emergency_priority_wait(p) is None
    assert w.submit((p,)).accepted


def test_unstarted_ordinary_final_yields_transactionally():
    w = world()
    w.seed_hold("normal", "inner", 0, soc=.85)
    p = final(w, start=1.)
    assert w.submit((p,)).accepted and "normal" in w.soft
    urgent(w, "rescue")
    w._yield_normal_finals()
    assert "normal" not in w.soft and "normal" not in w.active
    assert w.aircraft["normal"].status == "HOLD_INNER"
    assert not any(r.aircraft_id == "normal" for r in w.pads.reservations)
    assert any(e["type"] == "emergency_priority_yielded" for e in w.events)


def test_start_due_rechecks_waiting_and_does_not_start_cancelled_copy():
    w = world()
    w.seed_hold("normal", "inner", 0, soc=.85)
    p = final(w, start=1.)
    assert w.submit((p,)).accepted
    urgent(w, "rescue")
    # Boundary-only fixture: no physics or state advancement is asserted.
    w.time = 1.
    w._start_due()
    assert "normal" not in w.soft and "normal" not in w.active


def test_protected_release_and_active_final_are_preserved():
    w = world()
    w.seed_hold("normal", "inner", 0, soc=.85)
    p = final(w, start=1.)
    assert w.submit((p,)).accepted
    # Resource unit fixture for an already-started dependent plan's contract.
    w.dependencies["started_dependency"] = (p.trajectory_hash,)
    urgent(w, "rescue")
    pads, slots = w.pads.reservations, w.slots.reservations
    w._yield_normal_finals()
    assert w.soft["normal"] is p
    assert (w.pads.reservations, w.slots.reservations) == (pads, slots)
    event = next(e for e in w.events if e["type"] == "emergency_priority_yield_blocked")
    assert event["details"]["reason"] == "RESOURCE_DEPENDENCY"
    assert event["details"]["dependent_started_plans"] == ["started_dependency"]
    w.time = 1.
    w._start_due()
    assert w.active["normal"] is p and "normal" not in w.soft
    w._yield_normal_finals()
    assert w.active["normal"] is p
    assert any(e["details"].get("reason") == "MANEUVER_ALREADY_STARTED" for e in w.events)


def test_highest_urgency_inner_normal_exception_is_reserve_certified():
    w = world()
    urgent(w, "inner", "inner", 0, soc=.85, q=.8)
    urgent(w, "outer", "outer", 18, soc=.8, q=.01)
    p = generate_final("inner", "inner", 0, 5, 0., durations=[35.], config=w.config).plan
    assert p is not None
    assert w._emergency_priority_wait(p) is None
    assert w.submit((p,)).accepted
    assert w.active["inner"].kind == "FINAL_NORMAL"
    assert not w.aircraft["inner"].emergency.declared


def test_real_started_transfer_keeps_its_protected_normal_final_release():
    w = world()
    w.seed_hold("landing", "inner", 0, soc=.85)
    w.seed_hold("incoming", "middle", 1, soc=.85)
    departure = generate_final("landing", "inner", 0, 5, .001,
                               durations=[35.], config=w.config).plan
    incoming = generate_transfer("incoming", "middle", 1, 0, 0.,
                                 durations=[40.], config=w.config).plan
    assert departure is not None and incoming is not None
    assert w.submit((departure,)).accepted
    assert w.submit((incoming,)).accepted
    assert departure.trajectory_hash in w.dependencies[incoming.trajectory_hash]
    urgent(w, "rescue")
    w._yield_normal_finals()
    assert w.soft["landing"] is departure and w.active["incoming"] is incoming
    w.advance(.002)
    assert w.active["landing"] is departure and w.active["incoming"] is incoming
    assert not w.failed
    assert any(e["type"] == "emergency_priority_yield_blocked"
               and e["details"].get("dependent_started_plans") == [incoming.trajectory_hash]
               for e in w.events)


@pytest.mark.parametrize("reason", ["UNKNOWN", "HARD_SEPARATION"])
def test_uncertified_cancellation_keeps_original_certified_start(monkeypatch, reason):
    w = world()
    w.seed_hold("normal", "inner", 0, soc=.85)
    p = final(w, start=1.)
    assert w.submit((p,)).accepted
    urgent(w, "rescue")
    monkeypatch.setattr(w, "cancel", lambda key: w._result(False, reason))
    monkeypatch.setattr(w, "_emergency_conflict", lambda *args: pytest.fail("priority must preserve the original commitment"))
    w.time = 1.
    w._start_due()
    assert w.active["normal"] is p and "normal" not in w.soft
    assert any(e["type"] == "emergency_priority_yield_blocked"
               and e["details"]["reason"] == reason for e in w.events)


def test_legacy_policy_keeps_ordinary_final_behavior():
    w = RotatingSlotWorld()
    w.seed_hold("normal", "inner", 0, soc=.85)
    urgent(w, "rescue")
    assert w._emergency_priority_wait(final(w)) is None
    assert w.submit((final(w),)).accepted


def test_raw_soc_threshold_blocks_direct_submit_before_monitor_refresh():
    w = world()
    w.seed_hold("normal", "inner", 0, soc=.85)
    w.seed_hold("raw", "outer", 18, soc=.3)
    assert not w.aircraft["raw"].emergency.eligible
    p = final(w)
    result = w.evaluate_submission((p,))
    assert result.reason == "EMERGENCY_PRIORITY_WAIT" and result.details["blocked_by"] == "raw"
    assert w.submit((p,)).reason == result.reason
    assert not w.aircraft["raw"].emergency.eligible  # Evaluation/guard stays read-only.


def test_raw_highest_urgency_inner_can_take_reserve_safe_normal_exception():
    w = world()
    a = w.seed_hold("inner", "inner", 0, soc=.35)
    w.aircraft["inner"] = replace(a, emergency=EmergencyStatus(q_e=.12))
    w.seed_hold("outer", "outer", 18, soc=.34)
    p = generate_final("inner", "inner", 0, 5, 0., durations=[35.], config=w.config).plan
    assert p is not None
    assert w._emergency_waiting_queue()[0].id == "inner"
    assert not w.aircraft["inner"].emergency.eligible
    assert w._emergency_priority_wait(p) is None
    assert w.submit((p,)).accepted and w.active["inner"].kind == "FINAL_NORMAL"


def test_raw_threshold_dispatch_sets_eligibility_and_triggers_once(monkeypatch):
    w = world()
    w.seed_hold("raw", "outer", 18, soc=.3)
    seen = []
    monkeypatch.setattr(w, "_try_final", lambda a, emergency: seen.append(a.emergency.eligible) or False)
    w.handle_emergencies()
    w.handle_emergencies()
    assert seen == [True, True]
    triggered = [e for e in w.events if e["type"] == "emergency_triggered"]
    assert len(triggered) == 1 and triggered[0]["aircraft_id"] == "raw"
