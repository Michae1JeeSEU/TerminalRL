"""FT-RS replay adapter extending the existing Terminal3DViewer platform."""
from __future__ import annotations

from itertools import combinations
import json
import math
import time as clock
from types import MappingProxyType

import numpy as np

from ..core.rotating_slots.replay import ReplayBundle, _plain
from ..core.rotating_slots.geometry import slot_state, slot_phase, pad_position
from .config import VisualizationConfig
from .schema import (WorldRenderSnapshot, RingRenderState, GateRenderState,
    LandingZoneRenderState, UAVRenderState, TerminalEventRenderState)
from .pyvista_viewer import Terminal3DViewer

_GROUNDED = {"LANDED_SERVICE", "CLEARED", "SERVICE", "COMPLETED"}
_FROZEN = {"ENERGY_FAILURE", "ENERGY_MISSION_FAILED", "CENSORED_DUE_TO_FAILURE", "EXECUTED_HARD_SEPARATION_VIOLATION"}
_INVISIBLE = {"CLEARED", "COMPLETED", "REROUTED", "UNRELEASED_AT_FAILURE"}
_HOLD = {"HOLD_OUTER", "HOLD_MIDDLE", "HOLD_INNER", "HOLD", "HOLDING", "STABLE"}
_ACTIVE_MANEUVER = {"INBOUND", "TRANSFER_O_M", "TRANSFER_M_I", "FINAL_NORMAL",
                    "FINAL_EMERGENCY", "VERTICAL_DESCENT"}
_UPSTREAM_POLICY = "upstream_capacity_only_pad_arrival_v1_20261018"
_UPSTREAM_POLICIES = {_UPSTREAM_POLICY,"rs_s2_loss_continue_throughput_v1_0"}


def _separation_evidence(metrics):
    refined = metrics.get("certified_continuous_minimum", metrics.get("continuous_minimum"))
    if refined is not None:
        label = ("Certified continuous minimum (whole episode)" if refined.get("status") == "CERTIFIED"
                 else "Continuous minimum refinement (not certified)")
        value = refined
    else:
        label = "Logged separation interval evidence"
        value = metrics.get("minimum_separation") or {}
    return {"label": label, "lower_bound_m": value.get("lower_bound_m"),
        "upper_bound_m": value.get("upper_bound_m"),
        "witness_time": value.get("witness_time", value.get("time")),
        "witness_distance_m": value.get("witness_distance_m"),
        "pair": value.get("pair"), "status": value.get("status")}


def _diagnostic_text(record):
    details = _plain(record.get("details", {}))
    reason = record.get("reason", details.get("reason", "not recorded"))
    # Geometry is already drawn separately; retain the recorded diagnostic fields.
    detail_text = {k: v for k, v in details.items() if k != "trajectory_points"}
    return f"reason={reason}; details={json.dumps(detail_text, ensure_ascii=False, sort_keys=True)}"


def _time_text(value):
    return "unknown" if value is None else f"{value:.3f}s"


def _hud_layout(window_size, line_count):
    """Pixel anchor for a top-aligned HUD below controls, above the timeline."""
    _, height = window_size
    top = max(80, int(height) - 115)
    available = max(40, top - 75)
    font_size = max(5, min(8, int(available / max(1, line_count) / 1.6)))
    return (10, top), font_size


class ReplayController:
    """Seek is a pure query: it never steps a world or edits a committed plan."""
    speeds = (.25, 1., 4., 10.)

    def __init__(self, bundle):
        self.bundle = bundle if isinstance(bundle, ReplayBundle) else ReplayBundle.load(bundle)
        self.time = self.bundle.time_bounds[0]
        self.paused = True
        self.speed = 1.
        self.selected_id = None
        self.follow = False
        self.trajectories = "all"
        self.slot_labels = "selected_ring"
        ids = sorted({str(a["id"]) for s in self.bundle.data["states"] for a in s.get("aircraft", ())})
        self.uid_by_id = {identity: i for i, identity in enumerate(ids)}
        self.id_by_uid = {v: k for k, v in self.uid_by_id.items()}
        self._plans = {h: self.bundle.plan(h) for h in self.bundle.data["trajectories"]}

    def seek(self, time):
        lower, upper = self.bundle.time_bounds
        self.time = min(upper, max(lower, float(time)))
        return self.snapshot()

    def step(self):
        dt=float(self.bundle.index["sample_dt_s"])
        return self.seek((math.floor((self.time+1e-10)/dt)+1)*dt)

    def jump_event(self, direction=1):
        times = sorted({float(e["time"]) for e in self.bundle.data["events"]})
        options = [t for t in times if (t-self.time)*direction > 1e-10]
        return self.seek((min(options) if direction > 0 else max(options)) if options else self.time)

    def set_speed(self, value):
        if float(value) not in self.speeds:
            raise ValueError("playback speed must be .25, 1, 4, or 10")
        self.speed = float(value)

    def advance(self, wall_seconds):
        if not self.paused:
            self.seek(self.time + max(0., wall_seconds) * self.speed)
            if self.time >= self.bundle.time_bounds[1]:
                self.paused = True
        return self.snapshot()

    def select(self, aircraft_id, *, follow=False):
        if aircraft_id is not None and str(aircraft_id) not in self.uid_by_id:
            raise ValueError("unknown aircraft")
        self.selected_id = None if aircraft_id is None else str(aircraft_id)
        self.follow = bool(follow)

    def set_trajectories(self, mode):
        if mode not in ("all", "selected", "hidden"):
            raise ValueError("trajectory mode must be all, selected, or hidden")
        self.trajectories = mode

    def aircraft_at(self):
        result = []
        for observed in self.bundle.sampled_state(self.time).get("aircraft", ()):
            aircraft = _plain(observed)
            h = aircraft.get("trajectory_hash")
            if h and aircraft.get("status") not in _FROZEN:
                if h not in self._plans:
                    raise ValueError("missing authoritative trajectory: " + str(h))
                state = self._plans[h].state(self.time)
                aircraft.update(p=state.p, v=state.v, a=state.a, jerk=state.jerk)
            elif self.time != float(self.bundle.sampled_state(self.time)["time"]) and aircraft.get("status") not in (
                _GROUNDED | _FROZEN | _INVISIBLE):
                raise ValueError("airborne arbitrary-time replay requires an authoritative plan")
            result.append(aircraft)
        return tuple(result)

    def slots_at(self):
        config = self.bundle.config
        frame = self.bundle.sampled_state(self.time)
        aircraft = self.aircraft_at()
        result = []
        for ring in config.airspace.rings:
            for i in range(ring.slot_count):
                occupant = next((a for a in aircraft if a.get("ring") == ring.id and a.get("slot") == i
                    and a.get("status") in _HOLD), None)
                reservations = self._reservations(frame.get("slots", ()), (ring.id, i))
                active = [r for r in reservations if not r["future_interval"]]
                departing = any(r.get("kind") in ("departure", "source", "departure_protection")
                    and r["commitment"] != "SOFT_UNSTARTED_REMOVABLE" for r in active)
                firm = [r for r in reservations if r["commitment"] != "SOFT_UNSTARTED_REMOVABLE"]
                state = ("DEPARTING" if departing else "OCCUPIED" if occupant else "RESERVED" if firm
                         else "SOFT" if reservations else "FREE")
                result.append({"id": f"{ring.id}:{i}", "phase": slot_phase(ring.id, i, self.time, config),
                    "position": slot_state(ring.id, i, self.time, config).p, "state": state,
                    "occupant_id": str(occupant["id"]) if occupant else None, "reservations": reservations})
        return tuple(result)

    def _reservations(self, rows, resource):
        """Unexpired calendar entries, with maneuver and resource clocks separated."""
        result = []
        for row in rows:
            key = tuple(row["resource"]) if isinstance(resource, tuple) else row["resource"]
            end = float(row["end"]) if row.get("end") is not None else math.inf
            if key != resource or self.time >= end:
                continue
            view = _plain(row)
            start = float(row["start"])
            future = start > self.time
            committed = row.get("committed") is True
            legacy = "committed" not in row and not future
            commitment = ("COMMITTED" if committed else "RECORDED_OCCUPANCY" if row.get("kind") == "occupied" and not future
                          else "LEGACY_RECORDED_INTERVAL" if legacy
                          else "SOFT_UNSTARTED_REMOVABLE")
            plan = self._plans.get(row.get("plan_hash"))
            maneuver = plan if plan and plan.kind not in ("STABLE", "GROUND") else None
            view.update(owner_id=str(row["aircraft_id"]), owner_uid=self.uid_by_id.get(str(row["aircraft_id"])),
                interval_start=start, interval_end=None if math.isinf(end) else end,
                future_interval=future, commitment=commitment,
                maneuver_start=maneuver.start_time if maneuver else None,
                maneuver_kind=maneuver.kind if maneuver else None,
                maneuver_started=bool(committed and maneuver and maneuver.start_time <= self.time),
                descent_start=min((s.start_time for s in maneuver.segments if s.kind == "DESCENT"), default=None) if maneuver else None,
                touchdown_time=maneuver.end_time if maneuver and maneuver.kind in ("FINAL_NORMAL", "FINAL_EMERGENCY") else None)
            result.append(view)
        return sorted(result, key=lambda r: (r["interval_start"], r["owner_id"], r.get("kind", "")))

    def pads_at(self):
        frame = self.bundle.sampled_state(self.time)
        aircraft = self.aircraft_at()
        result = []
        for i in range(self.bundle.config.pads.count):
            rows = self._reservations(frame.get("pads", ()), i)
            occupant = next((a for a in aircraft if a.get("pad_id") == i
                             and a.get("status") in ("SERVICE", "LANDED_SERVICE")), None)
            owner = str(occupant["id"]) if occupant else None
            firm = [r for r in rows if r["commitment"] != "SOFT_UNSTARTED_REMOVABLE"]
            # The current service clock belongs only to its actual occupant.
            service_end = occupant.get("service_end") if occupant else None
            if occupant and service_end is None:
                service_end = max((r["interval_end"] for r in rows if r["owner_id"] == owner
                    and not r["future_interval"] and r["interval_end"] is not None), default=None)
            next_reservation = next((r for r in firm if r["owner_id"] != owner), None)
            result.append({"id": i, "state": "occupied" if occupant else "reserved" if firm else "soft_pending" if rows else "free",
                "occupant_id": owner, "occupied_by": self.uid_by_id.get(owner),
                "reserved_by": next_reservation["owner_uid"] if next_reservation else None,
                "reservation_owner_id": next_reservation["owner_id"] if next_reservation else None,
                "service_end": service_end,
                "service_remaining": max(0., float(service_end) - self.time) if service_end is not None else 0.,
                "reservations": rows})
        return tuple(result)

    def overlays(self):
        frame = self.bundle.sampled_state(self.time)
        aircraft = self.aircraft_at()
        physical = [a for a in aircraft if a.get("status") not in _INVISIBLE | {"ENERGY_MISSION_FAILED"}]
        pairs = [(math.dist(a["p"], b["p"]), a["id"], b["id"]) for a, b in combinations(physical, 2)]
        current_min = min(pairs, default=None)
        episode_metrics = _plain(self.bundle.data["metrics"])
        policy = frame.get("policy_id", self.bundle.data.get("policy_id", frame.get("metrics", {}).get("policy_id",
            episode_metrics.get("policy_id"))))
        upstream = _plain(frame.get("upstream", ()))
        dispatch = [r for r in self.bundle.data.get("dispatch_timings", ()) if r["time"] <= self.time]
        predictions = [r for r in self.bundle.data.get("tentative_predictions", ()) if r.get("as_of",0.)<=self.time]
        return {"minimum_current_3d_pair": current_min, "minimum_current_time": self.time,
            "latest_dispatch_timing": _plain(dispatch[-1]) if dispatch else None,
            "tentative_predictions": _plain(predictions),
            "failed_isolated": [a for a in aircraft if a.get("status") == "ENERGY_MISSION_FAILED"],
            "failure_archive": _plain(self.bundle.data.get("failure_archive", {})),
            "logged_metrics": _plain(frame.get("metrics", {})),
            "episode_metrics": episode_metrics,
            "separation_evidence": _separation_evidence(episode_metrics),
            "fixture_banner": "ANALYTIC FIXTURE - no physical episode outcome" if episode_metrics.get("fixture_only") else None,
            "policy_id": policy,
            "admission_model": ("AMENDED: capacity-only reroute; unsafe requests wait upstream (no physical aircraft / external energy model)"
                if policy in _UPSTREAM_POLICIES else "OLD MODEL: immediate admission or reroute; no upstream waiting queue recorded"
                if policy is None else "Recorded admission policy: " + str(policy)),
            "upstream_requests": upstream,
            "upstream_waiting_count": frame.get("metrics", {}).get("upstream_waiting", len(upstream)),
            "pad_resources": self.pads_at(),
            "future_slot_reservations": [r for s in self.slots_at() for r in s["reservations"] if r["future_interval"]],
            "current_commits": [a for a in aircraft if a.get("trajectory_hash") and a.get("status") in _ACTIVE_MANEUVER],
            "future_soft_dependencies": [r for r in (*frame.get("slots", ()), *frame.get("pads", ()))
                if not r.get("committed", False) and (float(r["start"]) > self.time or
                    r.get("plan_hash") in self._plans and self._plans[r["plan_hash"]].start_time > self.time)],
            "diagnostic_rejections_not_executed": [r for r in self.bundle.data.get("rejections", ()) if r["time"] <= self.time],
            "events": [e for e in self.bundle.data["events"] if e["time"] <= self.time]}

    def resource_overlay_lines(self):
        lines = []
        for pad in self.pads_at():
            owner = f'UID={pad["occupied_by"]} ({pad["occupant_id"]})'
            if pad["occupant_id"] is not None:
                lines.append(f'P{pad["id"]} SERVICE {owner}; service end={_time_text(pad["service_end"])}; '
                             f'remaining={pad["service_remaining"]:.3f}s')
            else:
                lines.append(f'P{pad["id"]} {pad["state"].upper()}' +
                    (f' UID={pad["reserved_by"]} ({pad["reservation_owner_id"]})' if pad["reservation_owner_id"] else ""))
            for row in pad["reservations"]:
                if row["owner_id"] == pad["occupant_id"] and not row["future_interval"]:
                    continue
                lines.append(f'  {row["commitment"]} UID={row["owner_uid"]} ({row["owner_id"]}); '
                    f'maneuver start={_time_text(row["maneuver_start"])}; '
                    f'pad interval=[{_time_text(row["interval_start"])}, {_time_text(row["interval_end"])}); '
                    f'descent={_time_text(row["descent_start"])}; touchdown={_time_text(row["touchdown_time"])}')
        futures = [r for slot in self.slots_at() for r in slot["reservations"] if r["future_interval"]]
        futures.sort(key=lambda r: (r["owner_id"] != self.selected_id, r["interval_start"]))
        for row in futures[:4]:
            lines.append(f'Slot {row["resource"]} {row["commitment"]} UID={row["owner_uid"]} ({row["owner_id"]}); '
                f'interval=[{_time_text(row["interval_start"])}, {_time_text(row["interval_end"])}); '
                f'maneuver start={_time_text(row["maneuver_start"])}')
        if len(futures) > 4:
            lines.append(f'{len(futures)-4} further future slot intervals recorded (selected aircraft shown first).')
        return lines

    def hud_lines(self, *, details=False):
        """Bounded display text; full recorded diagnostics remain in overlays()."""
        overlay = self.overlays()
        minimum = overlay["minimum_current_3d_pair"]
        pair = (f'{minimum[0]:.3f}m ({minimum[1]}, {minimum[2]})' if minimum else "no pair")
        policy = ("AMENDED: capacity-only reroute; unsafe entry waits upstream" if overlay["policy_id"] in _UPSTREAM_POLICIES
                  else "OLD MODEL: immediate admission / reroute" if overlay["policy_id"] is None
                  else f'Policy: {overlay["policy_id"]}')
        evidence = overlay["separation_evidence"]
        bound = lambda x: "unknown" if x is None else f"{x:.3f}"
        lines = [f't={self.time:.3f}s | physical display aircraft={len([a for a in self.aircraft_at() if a.get("status") not in _INVISIBLE])}',
            policy, f'Current 3D minimum: {pair}',
            f'{evidence["label"]}: [{bound(evidence["lower_bound_m"])}, {bound(evidence["upper_bound_m"])}]m',
            f'Slots F free / R committed incl.future / O occupied / D departing / S soft; paths t; details d',
            f'Actual maneuvers={len(overlay["current_commits"])}; future SOFT unstarted/removable={len(overlay["future_soft_dependencies"])}']
        if self.bundle.index.get("experiment_policy_id"):
            timing=overlay["latest_dispatch_timing"]
            lines.append("Stage2 HUMAN_PENDING | tentative suffix has no execution authority")
            lines.append(f'Isolated loss markers={len(overlay["failed_isolated"])}; abstract isolation, no accident cascade')
            if timing:
                lines.append(f'Normal wall={timing["normal_wall_s"]:.4f}s; full dispatch wall={timing["dispatch_wall_s"]:.4f}s')
        for pad in overlay["pad_resources"]:
            prefix = f'P{pad["id"]} '
            if pad["occupant_id"] is not None:
                lines.append(prefix + f'SERVICE UID={pad["occupied_by"]}; end={bound(pad["service_end"])}s; remaining={pad["service_remaining"]:.1f}s')
            else:
                lines.append(prefix + pad["state"].upper() + (f' UID={pad["reserved_by"]}' if pad["reserved_by"] is not None else ""))
            future = next((r for r in pad["reservations"] if r["owner_id"] != pad["occupant_id"]), None)
            if future:
                label = "COMMITTED" if future["commitment"] == "COMMITTED" else "SOFT unstarted/removable" if future["commitment"] == "SOFT_UNSTARTED_REMOVABLE" else "recorded"
                lines.append(f'  P{pad["id"]} {label} UID={future["owner_uid"]}: flight start={bound(future["maneuver_start"])}; '
                    f'descent={bound(future["descent_start"])}; touch={bound(future["touchdown_time"])}; service end={bound(future["interval_end"])}s')
        if overlay["policy_id"] in _UPSTREAM_POLICIES or overlay["upstream_requests"]:
            lines.append(f'Upstream unresolved={overlay["upstream_waiting_count"]}; external requests only (no physical aircraft/energy simulation)')
            if overlay["upstream_requests"]:
                request = overlay["upstream_requests"][0]
                lines.append(f'Upstream {request["id"]}: {request.get("status")}; wait={request.get("wait_seconds", 0):.1f}s; reason={request.get("reason")}')
        if self.selected_id is not None:
            selected = next((a for a in self.aircraft_at() if str(a["id"]) == self.selected_id), None)
            if selected:
                lines.append(f'Selected UID={self.uid_by_id[self.selected_id]} {self.selected_id}: {selected.get("status")}; SOC={selected["soc"]:.4f}')
        rejections = overlay["diagnostic_rejections_not_executed"]
        if rejections:
            latest = rejections[-1]
            lines.append(f'Diagnostic rejects={len(rejections)}; latest {latest.get("aircraft_id")}: {latest.get("reason", "not recorded")}')
        if details:
            selected_rows = [r for r in overlay["future_slot_reservations"] if self.selected_id is None or r["owner_id"] == self.selected_id]
            for row in selected_rows[:2]:
                lines.append(f'Slot {row["resource"]} {row["commitment"]} UID={row["owner_uid"]}; interval '
                    f'[{bound(row["interval_start"])}, {bound(row["interval_end"])})')
            if rejections:
                # A bounded excerpt is sufficient for the HUD; API retains all fields.
                lines.append("Reject detail: " + _diagnostic_text(rejections[-1])[:180])
        attention = [e for e in overlay["events"] if str(e["type"]).lower() in
                     ("energy_failure", "rerouted", "executed_hard_separation_violation")]
        if attention and len(lines) < 21:
            event = attention[-1]
            lines.append(f'Latest {event["type"]}: {event.get("aircraft_id")} t={event["time"]:.3f}s; reason={event.get("details", {}).get("reason", "not recorded")}')
        lines = lines[:21]
        lines.append("Details d: " + ("ON (bounded excerpt; full data in overlay API)" if details else "OFF; toggle for slot intervals / diagnostic excerpt"))
        return lines

    def snapshot(self):
        c = self.bundle.config
        frame = self.bundle.sampled_state(self.time)
        aircraft = self.aircraft_at()
        rings = tuple(RingRenderState(r.id, r.radius_m, r.height_agl_m, r.radius_m,
            r.height_agl_m, r.slot_count, sum(a.get("ring") == r.id for a in aircraft)) for r in c.airspace.rings)
        gates = tuple(GateRenderState(i, (c.entry.radius_m*math.cos(a), c.entry.radius_m*math.sin(a),c.entry.height_agl_m),
            (c.entry.radius_m*math.cos(a), c.entry.radius_m*math.sin(a),c.entry.height_agl_m)) for i,a in enumerate(c.entry.gate_angles_rad))
        pads = []
        for pad in self.pads_at():
            pads.append(LandingZoneRenderState(pad["id"], pad_position(pad["id"], c), pad["state"],
                pad["reserved_by"], pad["occupied_by"], pad["service_remaining"]))
        uavs = []
        for a in aircraft:
            if a.get("status") in _INVISIBLE:
                continue
            uid = self.uid_by_id[str(a["id"])]
            h = a.get("trajectory_hash")
            plan = self._plans.get(h)
            visible = self.trajectories == "all" or self.trajectories == "selected" and self.selected_id == str(a["id"])
            points = ()
            if plan and visible and plan.end_time > self.time and a.get("status") not in _FROZEN:
                points = tuple(plan.state(float(t)).p for t in np.linspace(self.time, plan.end_time, 121))
            emergency = a.get("emergency", {})
            declared = bool(emergency.get("declared", False)) if isinstance(emergency, dict) else bool(emergency)
            uavs.append(UAVRenderState(uid,uid,str(a.get("status","")),a.get("ring"),tuple(a["p"]),tuple(a.get("v",(0,0,0))),
                math.sqrt(sum(v*v for v in a.get("v",(0,0,0)))),float(a["soc"]),float(a["soc"])*c.energy.derived_equivalent_capacity_kWh*3.6e6,
                c.energy.derived_equivalent_capacity_kWh*3.6e6,str(a.get("service_class","")),1,0,a.get("pad_id"),h,None,
                bool(h) and a.get("status") in _ACTIVE_MANEUVER,points,1.,None,None,declared,"emergency" if declared else "low_soc" if a["soc"]<.35 else "normal"))
        terminal = []
        for e in self.bundle.data["events"]:
            if e["time"] > self.time or str(e["type"]).lower() not in ("energy_failure","rerouted","touchdown","executed_hard_separation_violation"):
                continue
            identity = str(e.get("aircraft_id")); uid = self.uid_by_id.get(identity)
            details = e.get("details", {})
            position = details.get("position",details.get("p"))
            outcome={"energy_failure":"failed_energy","rerouted":"rerouted","touchdown":"completed",
                "executed_hard_separation_violation":"separation_violation"}[str(e["type"]).lower()]
            terminal.append(TerminalEventRenderState(uid,uid if uid is not None else -1,outcome,str(details.get("reason","")),
                float(e["time"]),tuple(position) if position is not None else None,float(details.get("soc",0.)),0.,""))
        metrics = frame.get("metrics", {})
        telemetry = MappingProxyType({k:v for k,v in metrics.items() if isinstance(v,(int,float))})
        return WorldRenderSnapshot(self.time,int(round(self.time/.2)),rings,gates,tuple(pads),tuple(uavs),0,
            sum(a.get("status") in ("SERVICE","LANDED_SERVICE") for a in aircraft),
            sum(a.get("status") in ("COMPLETED","CLEARED") for a in aircraft),False,0.,telemetry,(),tuple(terminal))


class RotatingSlotsReplayViewer(Terminal3DViewer):
    """The production PyVista viewer plus replay controls and diagnostic actors."""
    def __init__(self, controller, *, off_screen=False):
        self.controller = controller
        self.vertical_exaggeration = 1.
        self._extra_names = set()
        self._curve_charts = []
        self._chart_selected = None
        self.show_details = False
        # Widget constructors can invoke callbacks before returning a widget.
        self._time_widget = None
        self._pause_widget = None
        self._speed_widget = None
        super().__init__(controller.snapshot(), VisualizationConfig(enabled=True,show_trajectories=True,
            show_labels=True,show_target_geometry=False,playback_speed=1.),off_screen=off_screen)
        self._time_widget=self._plotter.add_slider_widget(lambda t:self.draw(controller.seek(t)),controller.bundle.time_bounds,
            value=controller.time,title="Replay time (s)",pointa=(.18,.055),pointb=(.71,.055),color="gray",
            title_height=.016,slider_width=.015,tube_width=.003)
        self._time_widget.GetRepresentation().SetLabelHeight(.015)
        self._plotter.add_key_event("space", self.toggle_pause)
        self._plotter.add_key_event("Right",lambda:self.draw(controller.step()))
        self._plotter.add_key_event("n",lambda:self.draw(controller.jump_event(1)))
        self._plotter.add_key_event("p",lambda:self.draw(controller.jump_event(-1)))
        self._plotter.add_key_event("f",lambda:setattr(controller,"follow",not controller.follow))
        self._plotter.add_key_event("t",self.cycle_trajectories)
        self._plotter.add_key_event("l",self.cycle_slot_labels)
        self._plotter.add_key_event("d",self.toggle_details)
        for key,view in (("1","top_down"),("2","side"),("3","3d")):
            self._plotter.add_key_event(key,lambda v=view:self.set_view(v))
        self.set_view("3d")
        self.draw()

    def _add_interaction_controls(self):
        # Reuse picking, replace legacy playback choices with the contract set.
        self._plotter.enable_point_picking(self._select_uav,left_clicking=False,picker="point",show_message=False,show_point=False)
        self._pause_widget=self._plotter.add_checkbox_button_widget(lambda value:setattr(self.controller,"paused",value),
            value=self.controller.paused,
            position=(10,820),size=18)
        self._plotter.add_text("Pause: Space; step: Right; events: p/n; views: 1/2/3; follow: f; paths: t; slots: l",
            position=(35,825),font_size=9,color="black",name="replay-controls")
        self._speed_widget=self._plotter.add_text_slider_widget(lambda label:self.controller.set_speed(float(label[:-1])),
            ["0.25x","1x","4x","10x"],value=float(self.controller.speeds.index(self.controller.speed)),
            pointa=(.76,.945),pointb=(.97,.945),color="gray")
        representation=self._speed_widget.GetRepresentation()
        representation.SetTitleHeight(.018)
        representation.SetSliderWidth(.012)
        representation.SetTubeWidth(.002)

    def _select_uav(self, picked_point, picker):
        super()._select_uav(picked_point,picker)
        self.controller.select(self.controller.id_by_uid.get(self._selected_uid),follow=self.controller.follow)
        self.draw()

    def _update_landing_zones(self, snapshot):
        super()._update_landing_zones(snapshot)
        for pad in snapshot.landing_zones:
            if pad.status == "soft_pending":
                self._lz_actors[pad.lz_id].GetProperty().SetColor(self._pv.Color("purple").float_rgb)

    def _landing_zone_status_line(self, pad):
        if pad.status == "soft_pending":
            return f"P{pad.lz_id}: SOFT unstarted/removable"
        return super()._landing_zone_status_line(pad)

    def toggle_pause(self):
        self.controller.paused = not self.controller.paused
        self._sync_controls()

    def _sync_controls(self):
        """Reflect controller state without emitting widget interaction events."""
        time_widget=getattr(self,"_time_widget",None)
        if time_widget is not None:
            representation=time_widget.GetRepresentation()
            if representation.GetValue()!=self.controller.time:
                representation.SetValue(self.controller.time)
        pause_widget=getattr(self,"_pause_widget",None)
        if pause_widget is not None:
            representation=pause_widget.GetRepresentation()
            state=int(self.controller.paused)
            if representation.GetState()!=state:
                representation.SetState(state)
        speed_widget=getattr(self,"_speed_widget",None)
        if speed_widget is not None:
            representation=speed_widget.GetRepresentation()
            value=float(self.controller.speeds.index(self.controller.speed))
            if representation.GetValue()!=value:
                representation.SetValue(value)
            label=f"{self.controller.speed:g}x"
            if representation.GetTitleText()!=label:
                representation.SetTitleText(label)

    def cycle_trajectories(self):
        options=("all","selected","hidden")
        self.controller.set_trajectories(options[(options.index(self.controller.trajectories)+1)%3])
        self.draw()

    def cycle_slot_labels(self):
        modes=("selected_ring","all","hidden")
        self.controller.slot_labels=modes[(modes.index(self.controller.slot_labels)+1)%len(modes)]
        self.draw()

    def toggle_details(self):
        self.show_details = not self.show_details
        self.draw()

    def _draw_hud(self):
        lines = self.controller.hud_lines(details=self.show_details)
        position, font_size = _hud_layout(self._plotter.window_size, len(lines))
        actor = self._plotter.add_text("\n".join(lines), position=position, font_size=font_size,
            color="black", name="rs-overlay")
        actor.GetTextProperty().SetVerticalJustificationToTop()
        return actor

    def set_view(self, view, *, vertical_exaggeration=1.):
        if view not in ("top_down","side","3d") or vertical_exaggeration <= 0:
            raise ValueError("invalid view or exaggeration")
        self.vertical_exaggeration=float(vertical_exaggeration)
        self._plotter.set_scale(zscale=self.vertical_exaggeration)
        {"top_down":self._plotter.view_xy,"side":self._plotter.view_xz,"3d":self._plotter.view_isometric}[view]()
        self._plotter.add_text(f"ENU metres; z display x{self.vertical_exaggeration:g}; separation uses true 3D coordinates",
            position=(10,8),font_size=8,color="black",name="coordinates")
        self._plotter.reset_camera()
        self._plotter.camera.SetWindowCenter(.18,0.)
        self._plotter.camera.Zoom(1.2)

    def draw(self, snapshot=None):
        snapshot = snapshot or self.controller.snapshot()
        self._sync_controls()
        # Clear cumulative viewer history before a backward seek: the archive
        # is cumulative up to the requested time, never the last rendered time.
        self._terminal_event_keys.clear(); self._terminal_events.clear()
        for key in list(self._event_markers):
            self._plotter.remove_actor(self._event_markers.pop(key),render=False)
        super().update(snapshot,render=False)
        for name in ("legend","pick-help","hud","selected-uav","terminal-events","pad-status"):
            self._plotter.remove_actor(name,render=False)
        for name in self._extra_names:
            self._plotter.remove_actor(name,render=False)
        self._extra_names.clear()
        slots=self.controller.slots_at()
        colors={"FREE":"gray","RESERVED":"gold","OCCUPIED":"seagreen","DEPARTING":"darkorange","SOFT":"purple"}
        selected_aircraft=next((a for a in self.controller.aircraft_at() if str(a["id"])==self.controller.selected_id),None)
        selected_ring=selected_aircraft.get("ring") if selected_aircraft else "outer"
        for state,color in colors.items():
            rows=[s for s in slots if s["state"]==state]
            if rows:
                name="rs-slots-"+state
                self._plotter.add_mesh(self._pv.PolyData(np.array([s["position"] for s in rows])),
                    color=color,point_size=6,render_points_as_spheres=True,pickable=False,name=name,render=False)
                self._extra_names.add(name)
                labelled=[s for s in rows if self.controller.slot_labels=="all" or
                    self.controller.slot_labels=="selected_ring" and s["id"].split(":")[0]==selected_ring]
                if labelled:
                    name+="-labels"
                    self._plotter.add_point_labels(np.array([s["position"] for s in labelled]),
                        [f'{s["id"]} {state[0]} {s["phase"]:.2f}' for s in labelled],font_size=7,text_color=color,
                        point_size=0,shape=None,always_visible=True,show_points=False,name=name,render=False)
                    self._extra_names.add(name)
        for uav in snapshot.uavs:
            name=f"rs-ball-{uav.uid}"
            self._plotter.add_mesh(self._pv.Sphere(radius=40.,center=uav.position),color="red" if uav.emergency_landing else "steelblue",
                opacity=.055,name=name,render=False,pickable=False)
            self._extra_names.add(name)
        overlay=self.controller.overlays()
        if overlay["fixture_banner"]:
            self._plotter.add_text(overlay["fixture_banner"],position=(10,803),font_size=10,
                color="darkred",name="rs-fixture-banner")
        else:
            self._plotter.remove_actor("rs-fixture-banner",render=False)
        for uav in snapshot.uavs:
            actor=self._trajectory_actors.get(f"trajectory-{uav.uid}")
            if actor is not None:
                actor.GetProperty().SetColor((.85,.05,.05) if uav.emergency_landing else (.1,.3,.85))
        future_hashes={r.get("plan_hash") for r in overlay["future_soft_dependencies"]}
        for h in future_hashes:
            plan=self.controller._plans.get(h)
            if plan is not None and self.controller.trajectories!="hidden":
                if self.controller.trajectories=="selected" and str(plan.aircraft_id)!=self.controller.selected_id:
                    continue
                name="rs-soft-"+h
                points=np.array([plan.state(float(t)).p for t in np.linspace(plan.start_time,plan.end_time,121)])
                self._plotter.add_mesh(self._pv.lines_from_points(points),color="purple",opacity=.45,line_width=2,
                    name=name,pickable=False,render=False)
                self._extra_names.add(name)
        for index, prediction in enumerate(overlay["tentative_predictions"]):
            if self.controller.trajectories=="hidden": break
            if self.controller.trajectories=="selected" and prediction.get("aircraft_id")!=self.controller.selected_id: continue
            for number, parameters in enumerate(prediction.get("plans", ())):
                from ..core.rotating_slots.trajectory import TrajectoryPlan
                plan=TrajectoryPlan.from_dict(parameters,config=self.controller.bundle.config)
                points=np.array([plan.state(float(t)).p for t in np.linspace(plan.start_time,plan.end_time,61)])
                name=f"rs-tentative-{index}-{number}"
                self._plotter.add_mesh(self._pv.lines_from_points(points),color="teal",opacity=.18,line_width=1,
                    name=name,pickable=False,render=False)
                self._extra_names.add(name)
        for failed in overlay["failed_isolated"]:
            name="rs-isolated-"+str(failed["id"])
            self._plotter.add_mesh(self._pv.Sphere(radius=9.,center=failed["p"]),color="black",
                name=name,pickable=False,render=False)
            self._extra_names.add(name)
        for index,rejection in enumerate(overlay["diagnostic_rejections_not_executed"]):
            points=rejection.get("details",{}).get("trajectory_points",())
            if len(points)>1:
                name=f"rs-rejected-{index}"
                self._plotter.add_mesh(self._pv.lines_from_points(np.array(points)),color="magenta",opacity=.3,
                    line_width=1,name=name,pickable=False,render=False)
                self._extra_names.add(name)
        requests={r["id"]:r for r in self.controller.bundle.data["manifest"].get("requests",())}
        for index,event in enumerate(overlay["events"]):
            if str(event["type"]).lower()=="rerouted" and event.get("aircraft_id") in requests:
                request=requests[event["aircraft_id"]]
                gate=snapshot.gates[int(request["gate_id"])]
                start=np.array(gate.gate_position); direction=start.copy(); direction[2]=0.
                name=f"rs-reroute-{index}"
                self._plotter.add_mesh(self._pv.Arrow(start=start,direction=direction,scale=100),color="orange",
                    name=name,pickable=False,render=False)
                self._extra_names.add(name)
        self._draw_hud()
        selected=self.controller.selected_id
        if selected != self._chart_selected:
            for chart in self._curve_charts: self._plotter.remove_chart(chart)
            self._curve_charts=[]; self._chart_selected=selected
        if selected and not self._curve_charts:
            rows=self.controller.bundle.curves(selected)
            fields=(("speed","blue","speed m/s"),("acceleration","green","acceleration m/s2"),
                ("jerk","orange","jerk m/s3"),("battery_power","red","battery kW"),("SOC","purple","SOC fraction"))
            for i,(field,color,label) in enumerate(fields):
                curves=self._pv.Chart2D(size=(.25,.16),loc=(.75,.73-i*.175))
                valid=[r for r in rows if r[field] is not None]
                if valid:
                    values=[r[field]/1000 if field=="battery_power" else r[field] for r in valid]
                    curves.line([r["time"] for r in valid],values,color=color)
                    padding=max(.02*(max(values)-min(values)),.001*max(1.,max(abs(v) for v in values)))
                    curves.y_range=(min(values)-padding,max(values)+padding)
                curves.title="Recorded "+label
                curves.x_label="time s"; curves.y_label=label
                self._plotter.add_chart(curves); self._curve_charts.append(curves)
        if self.controller.follow and selected:
            uav=next((a for a in snapshot.uavs if self.controller.id_by_uid[a.uid]==selected),None)
            if uav:
                self._plotter.set_focus(uav.position)
        self._plotter.render()
        return snapshot

    def screenshots(self, directory):
        from pathlib import Path
        target=Path(directory); target.mkdir(parents=True,exist_ok=True)
        result=[]
        for view in ("top_down","side","3d"):
            self.set_view(view); self.draw()
            path=target/(view+".png")
            self._plotter.screenshot(str(path)); result.append(str(path.resolve()))
        return tuple(result)

    def run(self):
        self._plotter.show(auto_close=False,interactive_update=True)
        previous=clock.monotonic()
        while not self.closed:
            now=clock.monotonic()
            self.controller.advance(now-previous); previous=now
            self.draw(); self._plotter.update()
            clock.sleep(1/30)
