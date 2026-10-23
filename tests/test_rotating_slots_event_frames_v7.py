"""Future recordings retain monitor events without changing dispatch or flight."""
from copy import deepcopy
import json

import pytest

from terminal_marl.core.rotating_slots.audit import audit_replay
from terminal_marl.core.rotating_slots.config import thaw
from terminal_marl.core.rotating_slots.replay import BufferedStateJSONLSink, ReplayBundle
from terminal_marl.core.rotating_slots.rescue_rules import RESCUE_PROFILE_V7
from terminal_marl.core.rotating_slots.trajectory import generate_final, generate_transfer
from terminal_marl.core.rotating_slots.world import RotatingSlotWorld, STAGE2_POLICY_ID


def warning_world(*, sink=None, normal=False, legacy_recording=False):
    w = RotatingSlotWorld(experiment_policy_id=STAGE2_POLICY_ID, state_sink=sink)
    w.rescue_rule_profile = RESCUE_PROFILE_V7
    w.auto_emergency = False
    move = generate_transfer("moving", "outer", 0, 0, .13,
                             config=w.config, durations=[40.]).plan
    assert move is not None
    q = w._energy(move, move.start_time, move.end_time).soc_used
    # Before departure SOC is above the warning threshold. Once moving, the
    # real remaining-energy assessment falls below it without proving rescue.
    w.seed_hold("moving", "outer", 0, soc=.35 + q / 2)
    if normal:
        w.seed_hold("normal", "inner", 0, soc=.85)
        final = generate_final("normal", "inner", 0, 5, 5., config=w.config).plan
        assert final is not None and w.submit((final,)).accepted
    assert w.submit((move,)).accepted
    if legacy_recording:
        record = w._record

        def omit_monitor_frame(force=False):
            # Suppress only the new monitor-only force record. All preexisting
            # tick, dispatch, submission and final recording calls still run.
            if (force and w.events and w.events[-1]["type"] == "rescue_warning"
                    and w.events[-1]["time"] == w.time
                    and w.time not in w.decision_times):
                return
            record(force=force)

        w._record = omit_monitor_frame
    return w, move


@pytest.mark.parametrize("streamed", [False, True])
def test_off_tick_monitor_warning_has_exact_physical_frame_and_audits(tmp_path, monkeypatch, streamed):
    sink = BufferedStateJSONLSink(tmp_path / "states.jsonl") if streamed else None
    w, move = warning_world(sink=sink)
    monitor_frames = []
    monitor = w._update_emergency

    def capture_monitor():
        count = len(w.events)
        triggered = monitor()
        if len(w.events) > count:
            monitor_frames.append(w._snapshot(w.metrics()).to_dict())
        return triggered

    monkeypatch.setattr(w, "_update_emergency", capture_monitor)
    try:
        w.advance(1.4, False)
        warning, = [e for e in w.events if e["type"] == "rescue_warning"]
        assert warning["time"] == pytest.approx(.33)
        assert warning["time"] not in w.decision_times
        assert w.decision_times == [0., .13]
        assert not any(e["type"] == "emergency_triggered" for e in w.events)
        bundle = ReplayBundle.write(tmp_path / "bundle", world=w,
                                    code_commit="local-future-logging-regression", eager=True)
        frames = thaw(bundle.data["states"])
        frame, = [f for f in frames if f["time"] == warning["time"]]
        assert frame == json.loads(json.dumps(monitor_frames[0]))
        aircraft, = frame["aircraft"]
        assert aircraft["emergency"]["warning"]
        assert aircraft["p"] == pytest.approx(move.state(warning["time"]).p)
        assert [f["time"] for f in frames] == [0., .13, .33, 1., 1.4]
        report = audit_replay(bundle.path)
        assert report["status"] == "PASS", report["issues"]
        if sink:
            assert len(w.records) == 1
            count = sink.frame_count
            w.flush_records()
            w._record(force=True)
            w.flush_records()
            assert sink.frame_count == count == len(frames)
            replaced = ReplayBundle.write(tmp_path / "after_flush", world=w,
                code_commit="local-future-logging-regression", eager=True)
            assert thaw(replaced.data["states"]) == frames
    finally:
        if sink:
            sink.close()


def test_legacy_frame_omission_reproduces_original_audit_failure(tmp_path):
    w, _ = warning_world(legacy_recording=True)
    w.advance(1.4, False)
    assert [f["time"] for f in w.records] == [0., .13, 1., 1.4]
    bundle = ReplayBundle.write(tmp_path / "legacy_recording", world=w,
                                code_commit="local-legacy-recording-counterexample")
    report = audit_replay(bundle.path)
    assert report["status"] == "FAIL"
    assert [(i["code"], i["event_type"], i["time"]) for i in report["issues"]] == [
        ("EVENT_FRAME_MISSING", "rescue_warning", .33)]


def physical_trace(w):
    trace = []
    advance_energy = w._advance_energy

    def capture(end):
        advance_energy(end)
        trace.append(deepcopy((w._snapshot(w.metrics()).to_dict(), w.active,
            w.soft, w.dependencies, w.costs, w._targeted_energy, w.events)))

    w._advance_energy = capture
    return trace


def test_recording_fix_preserves_all_physics_dispatch_and_normal_reservations():
    fixed, move = warning_world(normal=True)
    legacy, _ = warning_world(normal=True, legacy_recording=True)
    fixed_trace, legacy_trace = physical_trace(fixed), physical_trace(legacy)
    def normal_commitments(w):
        return (tuple(r for r in w.slots.reservations if r.aircraft_id == "normal"),
                tuple(r for r in w.pads.reservations if r.aircraft_id == "normal"),
                w.soft["normal"])

    commitments = normal_commitments(fixed)
    actions = [[], []]
    for w, captured in ((fixed, actions[0]), (legacy, actions[1])):
        w.advance(1.4, lambda snapshot, port: captured.append(snapshot.to_dict()))
    assert fixed_trace == legacy_trace
    assert actions[0] == actions[1]
    assert fixed.aircraft == legacy.aircraft
    assert fixed.active == legacy.active == {"moving": move}
    assert fixed.slots.reservations == legacy.slots.reservations
    assert fixed.pads.reservations == legacy.pads.reservations
    assert fixed.costs == legacy.costs
    assert fixed.events == legacy.events
    assert fixed.metrics() == legacy.metrics()
    assert fixed.decision_times == legacy.decision_times == [0., .13]
    assert [d["time"] for d in fixed.dispatch_timings] == [d["time"] for d in legacy.dispatch_timings]
    assert commitments == normal_commitments(fixed)
    assert len(fixed.records) == len(legacy.records) + 1
    assert [f for f in fixed.records if f["time"] != .33] == legacy.records


def test_persistent_warning_does_not_force_a_frame_at_every_physics_step():
    w, move = warning_world()
    w.advance(11.4, False)
    assert w.active == {"moving": move}
    assert w.aircraft["moving"].emergency.warning
    assert not w.aircraft["moving"].emergency.declared
    warning, = [e for e in w.events if e["type"] == "rescue_warning"]
    assert warning["time"] == .33
    assert w.decision_times == pytest.approx([0., .13, 5., 10.])
    off_ticks = [f["time"] for f in w.records
                 if abs(f["time"] - round(f["time"])) > 1e-8]
    assert off_ticks == [.13, .33, 11.4]
    assert len(w.records) == 15
