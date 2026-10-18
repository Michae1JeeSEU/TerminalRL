"""Resource/query regressions; the recorded f67 plan is never regenerated.

The compact UID26 fixture below is a read-only extraction of the 550s frame and
authoritative trajectory from traffic_rate180_seed11. Extra lifecycle frames are
test-only projections for querying intervals; they are not experiment evidence.
"""
from copy import deepcopy

import pytest

from terminal_marl.core.rotating_slots.config import load_config
from terminal_marl.core.rotating_slots.replay import ReplayBundle
from terminal_marl.core.rotating_slots.trajectory import TrajectoryPlan
from terminal_marl.visualization.rotating_slots_replay import (
    ReplayController, RotatingSlotsReplayViewer, _diagnostic_text, _hud_layout,
)


F67_COMMIT = "f67b5c8462433ba701191c493a37e4a8e472334e"
UID26_HASH = "b2be8deca9e8923a25732008fa50b16f3d68c206f0351cfd5ab7693b6921e9ab"
UID26_PLAN = {
    "aircraft_id": "req_000026", "config_hash": "0c7cce0c0c019d50f7bba5aecd5e8340925d0d16867f8f06eb51e89102343d53",
    "end_time": 593.5, "kind": "FINAL_NORMAL", "pad_id": 5,
    "profile_hash": "34b25de04b038a99a53cc4cbdd33e11aa64e72d44edf4fd5caffbdfa2c2f51e4",
    "segments": [
        {"duration": 40., "kind": "POLAR", "r": [300., 0., 0., -2000., 3000., -1200.], "start_time": 540.,
         "theta": [-18.550934149601137, -1.4000000000000001, 0., -5.058193231342186, 8.987289847013278, -3.874915938805311],
         "z": [30., 0., 0., -70., 85., -30.]},
        {"duration": 4., "kind": "DESCENT", "r": [100.], "start_time": 580., "theta": [-1.0471975511965976],
         "z": [15., -2., 0., 0., -10., 12., -4.]},
        {"duration": 5.166666666666667, "kind": "DESCENT", "r": [100.], "start_time": 584.,
         "theta": [-1.0471975511965976], "z": [11., -7.75, 0., 0., 0., -0., 0.]},
        {"duration": 4.333333333333333, "kind": "DESCENT", "r": [100.], "start_time": 589.1666666666666,
         "theta": [-1.0471975511965976], "z": [3.25, -6.5, 0., 0., 16.25, -19.5, 6.5]}],
    "source_ring": "inner", "source_slot": 1, "source_version": 3, "start_time": 540.,
    "target_ring": None, "target_slot": None, "trajectory_hash": UID26_HASH,
}
UID26_RESERVATION = dict(aircraft_id="req_000026", committed=True, dependencies=[], end=653.5,
    kind="final", plan_hash=UID26_HASH, resource=5, start=580.)
RECORDED_550_POSITION = (277.92681274989127, -27.63025767980672, 29.208984375)


def reduced_record():
    plan = TrajectoryPlan.from_dict(UID26_PLAN)
    assert plan.trajectory_hash == UID26_HASH
    terminal_ids = [dict(id=f"req_{i:06d}", status="CLEARED", soc=.8, p=(0., 0., 0.), v=(0., 0., 0.))
                    for i in range(26)]
    frames = []
    for t in (550., 580., 593.5, 600., 653.5, 660.):
        state = plan.state(t)
        status = "FINAL_NORMAL" if t < 580 else "VERTICAL_DESCENT" if t < 593.5 else "LANDED_SERVICE" if t < 653.5 else "CLEARED"
        aircraft = dict(id="req_000026", status=status, soc=.6434146008397782, p=state.p, v=state.v,
            a=state.a, jerk=state.jerk, ring="inner" if t < 580 else None, slot=1 if t < 580 else None,
            service_class="short", trajectory_hash=UID26_HASH if t < 593.5 else None,
            pad_id=5 if t >= 593.5 else None, service_end=653.5 if t >= 593.5 else None)
        frames.append(dict(time=t, aircraft=[*deepcopy(terminal_ids), aircraft], slots=[],
            pads=[deepcopy(UID26_RESERVATION)], metrics={}))
    return dict(config_hash=load_config().config_hash, sample_dt_s=.2, manifest=dict(seed=11, requests=[]),
        states=frames, trajectories={UID26_HASH: deepcopy(UID26_PLAN)}, events=[], metrics={}, rejections=[])


def controller(tmp_path, record=None):
    archive = ReplayBundle.write(tmp_path / "bundle", record=record or reduced_record(),
        code_commit=F67_COMMIT, seed=11, command="pytest reduced recorded 550s regression; not new simulation evidence")
    return ReplayController(archive)


def test_recorded_550_pad5_future_committed_owner_and_separate_clocks(tmp_path):
    replay = controller(tmp_path)
    before = (replay.bundle.path / "trajectories.json").read_bytes()
    snapshot = replay.seek(550.)
    assert snapshot.landing_zones[5].status == "reserved"
    assert snapshot.landing_zones[5].reserved_by == 26
    assert snapshot.landing_zones[5].occupied_by is None
    assert snapshot.landing_zones[5].service_remaining == 0.  # No current service countdown.
    assert snapshot.uavs[0].position == RECORDED_550_POSITION
    assert snapshot.uavs[0].committed
    pad = replay.pads_at()[5]
    row, = pad["reservations"]
    assert row["owner_id"] == "req_000026" and row["owner_uid"] == 26
    assert row["commitment"] == "COMMITTED" and row["future_interval"] and row["maneuver_started"]
    assert (row["maneuver_start"], row["descent_start"], row["touchdown_time"], row["interval_start"], row["interval_end"]) == (
        540., 580., 593.5, 580., 653.5)
    lines = "\n".join(replay.resource_overlay_lines())
    assert "P5 RESERVED UID=26" in lines
    assert "maneuver start=540.000s" in lines and "pad interval=[580.000s, 653.500s)" in lines
    assert "touchdown=593.500s" in lines
    assert replay.overlays()["admission_model"].startswith("OLD MODEL:")
    assert not replay.overlays()["upstream_requests"]
    assert (replay.bundle.path / "trajectories.json").read_bytes() == before


def test_future_reservation_service_and_clearance_sequence(tmp_path):
    replay = controller(tmp_path)
    assert replay.seek(580.).landing_zones[5].status == "reserved"
    assert replay.pads_at()[5]["reservations"][0]["future_interval"] is False
    service = replay.seek(600.).landing_zones[5]
    assert (service.status, service.occupied_by, service.service_remaining) == ("occupied", 26, 53.5)
    assert replay.seek(653.5).landing_zones[5].status == "free"
    assert not replay.pads_at()[5]["reservations"]


def test_current_service_clock_and_next_owner_are_never_combined(tmp_path):
    record = reduced_record()
    frame = record["states"][0]
    frame["aircraft"][25].update(status="LANDED_SERVICE", pad_id=5, service_end=570., service_class="short")
    frame["pads"].insert(0, dict(resource=5, aircraft_id="req_000025", start=510., end=570., kind="service", committed=True))
    replay = controller(tmp_path, record)
    pad = replay.seek(550.).landing_zones[5]
    assert pad.status == "occupied" and pad.occupied_by == 25 and pad.reserved_by == 26
    assert pad.service_remaining == 20.
    lines = "\n".join(replay.resource_overlay_lines())
    assert "SERVICE UID=25" in lines and "service end=570.000s; remaining=20.000s" in lines
    assert "COMMITTED UID=26" in lines and "pad interval=[580.000s, 653.500s)" in lines


def test_soft_future_pad_is_unstarted_removable_and_not_active_commit(tmp_path):
    record = reduced_record()
    frame = record["states"][0]
    frame["pads"] = [dict(resource=0, aircraft_id="req_000026", start=580., end=653.5,
        kind="final", committed=False, plan_hash=UID26_HASH)]
    # A future soft plan does not become a physical commitment via its calendar.
    frame["aircraft"][-1].update(status="HOLD_INNER")
    replay = controller(tmp_path, record)
    pad = replay.seek(550.).landing_zones[0]
    assert (pad.status, pad.reserved_by, pad.occupied_by, pad.service_remaining) == ("soft_pending", None, None, 0.)
    row, = replay.pads_at()[0]["reservations"]
    assert row["commitment"] == "SOFT_UNSTARTED_REMOVABLE" and not row["maneuver_started"]
    assert "SOFT_UNSTARTED_REMOVABLE" in "\n".join(replay.resource_overlay_lines())
    assert not replay.overlays()["current_commits"]


def test_slot_future_commit_soft_and_current_occupant_priorities(tmp_path):
    record = reduced_record()
    frame = record["states"][0]
    frame["aircraft"][-1].update(status="HOLD_INNER")
    def reservation(resource, committed, kind="target", start=580.):
        return dict(resource=resource, aircraft_id="req_000026", start=start, end=653.5,
                    kind=kind, committed=committed, plan_hash=UID26_HASH)
    frame["slots"] = [reservation(["outer", 0], True), reservation(["outer", 1], False),
        reservation(["inner", 1], True), reservation(["middle", 0], True, "departure_protection", 540.)]
    replay = controller(tmp_path, record)
    replay.seek(550.)
    rows = {r["id"]: r for r in replay.slots_at()}
    assert rows["outer:0"]["state"] == "RESERVED"
    assert rows["outer:1"]["state"] == "SOFT"
    assert rows["inner:1"]["state"] == "OCCUPIED"
    assert rows["middle:0"]["state"] == "DEPARTING"
    assert rows["outer:0"]["reservations"][0]["interval_start"] == 580.
    assert len(replay.overlays()["future_slot_reservations"]) == 3

    # A soft departure's source interval can begin before its maneuver starts.
    # It must not make the currently holding aircraft appear to be departing.
    frame["slots"] = [reservation(["inner", 1], False, "source", 540.)]
    soft_source = controller(tmp_path / "soft_source", record)
    assert next(r for r in soft_source.slots_at() if r["id"] == "inner:1")["state"] == "OCCUPIED"


def test_legacy_active_interval_without_committed_flag_keeps_recorded_semantics(tmp_path):
    record = reduced_record()
    frame = record["states"][0]
    frame["pads"] = [dict(resource=0, aircraft_id="req_000026", start=540., end=570., kind="final")]
    replay = controller(tmp_path, record)
    assert replay.snapshot().landing_zones[0].status == "reserved"
    assert replay.pads_at()[0]["reservations"][0]["commitment"] == "LEGACY_RECORDED_INTERVAL"


@pytest.mark.parametrize("status", ["WAITING", "CENSORED_UPSTREAM"])
def test_queue_contract_is_external_demand_without_uid_or_physical_aircraft(tmp_path, status):
    record = reduced_record()
    frame = record["states"][0]
    policy = "upstream_capacity_only_pad_arrival_v1_20261018"
    frame["policy_id"] = policy
    frame["upstream"] = [dict(id="external_wait", release_time=500., gate_id=2, initial_soc=.8,
        service_class="short", vehicle_profile_id="default", status=status, wait_seconds=50.,
        reason="HARD_SEPARATION", reason_categories=["safety"], last_attempt_time=550.)]
    frame["metrics"].update(policy_id=policy, upstream_waiting=1, censored_upstream=int(status == "CENSORED_UPSTREAM"),
        upstream_wait_time=50., cost_upstream_wait=50.)
    replay = controller(tmp_path, record)
    snapshot = replay.snapshot()
    assert len(snapshot.uavs) == 1 and "external_wait" not in replay.uid_by_id
    assert snapshot.telemetry["upstream_waiting"] == 1
    overlay = replay.overlays()
    assert overlay["policy_id"] == policy and overlay["admission_model"].startswith("AMENDED:")
    request, = overlay["upstream_requests"]
    assert (request["wait_seconds"], request["reason"], request["initial_soc"]) == (50., "HARD_SEPARATION", .8)
    assert request["status"] == status
    assert overlay["upstream_waiting_count"] == 1
    assert len(overlay["current_commits"]) == 1


@pytest.mark.parametrize("record", [
    dict(reason="HARD_SEPARATION", details=dict(pair=["a", "b"], lower_bound_m=79.)),
    dict(type="rerouted", details=dict(reason="HARD_SEPARATION", pair=["a", "b"], lower_bound_m=79.)),
])
def test_rejection_and_event_text_retain_actual_reason_and_details(record):
    text = _diagnostic_text(record)
    assert "HARD_SEPARATION" in text and '"pair": ["a", "b"]' in text and '"lower_bound_m": 79.0' in text


def test_default_hud_keeps_time_policy_minimum_and_all_pads_with_bounded_detail(tmp_path):
    record = reduced_record()
    record["rejections"] = [dict(time=550., aircraft_id="req_000009", reason="HARD_SEPARATION",
        details=dict(long_json_field="x" * 2000, pair=["a", "b"]))]
    replay = controller(tmp_path, record)
    lines = replay.hud_lines()
    text = "\n".join(lines)
    assert len(lines) <= 22 and lines[0].startswith("t=550.000s")
    assert "OLD MODEL" in lines[1] and "Current 3D minimum" in lines[2]
    assert all(f"P{i} " in text for i in range(6))
    assert "P5 RESERVED UID=26" in text
    assert "P5 COMMITTED UID=26: flight start=540.000; descent=580.000; touch=593.500; service end=653.500s" in text
    assert "long_json_field" not in text and "HARD_SEPARATION" in text
    assert len(replay.hud_lines(details=True)) <= 22
    assert "Reject detail:" in "\n".join(replay.hud_lines(details=True))
    assert replay.overlays()["diagnostic_rejections_not_executed"][0]["details"]["long_json_field"] == "x" * 2000


@pytest.mark.parametrize("size", [(1280, 900), (1280, 720), (1600, 1000)])
def test_hud_pixel_layout_is_below_controls_and_above_timeline(size):
    position, font_size = _hud_layout(size, 22)
    assert position == (10, size[1] - 115)
    assert position[1] < size[1] * .9
    assert position[1] - 22 * font_size * 1.6 > 75


def test_hud_actor_uses_top_alignment_and_detail_key_only_changes_view(tmp_path):
    class TextProperty:
        top = False
        def SetVerticalJustificationToTop(self): self.top = True
    class Actor:
        prop = TextProperty()
        def GetTextProperty(self): return self.prop
    class Plotter:
        window_size = (1280, 900)
        calls = []
        def add_text(self, text, **kwargs):
            self.calls.append((text, kwargs))
            return Actor()
    viewer = object.__new__(RotatingSlotsReplayViewer)
    viewer.controller = controller(tmp_path)
    viewer.show_details = False
    viewer._plotter = Plotter()
    actor = viewer._draw_hud()
    assert actor.prop.top
    text, kwargs = viewer._plotter.calls[-1]
    assert kwargs["position"] == (10, 785) and len(text.splitlines()) <= 22
    viewer.draw = lambda: None
    viewer.toggle_details()
    assert viewer.show_details is True and viewer.controller.time == 550.
