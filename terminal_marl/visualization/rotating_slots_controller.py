"""Renderer-neutral FT-RS replay controller; no VTK or simulation imports."""
from __future__ import annotations

from bisect import bisect_right
from itertools import combinations
import json
import math
from types import MappingProxyType

import numpy as np

from ..core.rotating_slots.replay import ReplayBundle, _plain
from ..core.rotating_slots.trajectory import TrajectoryPlan
from ..core.rotating_slots.geometry import slot_state, slot_phase, pad_position
from .schema import (WorldRenderSnapshot, RingRenderState, GateRenderState,
    LandingZoneRenderState, UAVRenderState, TerminalEventRenderState)

_GROUNDED = {"LANDED_SERVICE", "CLEARED", "SERVICE", "COMPLETED"}
_FROZEN = {"ENERGY_FAILURE", "ENERGY_MISSION_FAILED", "CENSORED_DUE_TO_FAILURE", "EXECUTED_HARD_SEPARATION_VIOLATION"}
_INVISIBLE = {"CLEARED", "COMPLETED", "REROUTED", "UNRELEASED_AT_FAILURE"}
_HOLD = {"HOLD_OUTER", "HOLD_MIDDLE", "HOLD_INNER", "HOLD", "HOLDING", "STABLE"}
_ACTIVE_MANEUVER = {"INBOUND", "TRANSFER_O_M", "TRANSFER_M_I", "FINAL_NORMAL",
                    "FINAL_EMERGENCY", "VERTICAL_DESCENT"}
_UPSTREAM_POLICY = "upstream_capacity_only_pad_arrival_v1_20261018"
_UPSTREAM_POLICIES = {_UPSTREAM_POLICY,"rs_s2_loss_continue_throughput_v1_0"}
_STAGE3_OVERLAY_TYPES = frozenset(("WorkDescriptor", "FactualResult", "Rank"))


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


def _recorded_rescue_status(aircraft, frame, plans):
    """Show recorded need and actual rescue commitment without SOC inference."""
    e = aircraft.get("emergency", {})
    if not isinstance(e, dict) or not e.get("declared", False):
        return None
    status = aircraft.get("status")
    if status == "FINAL_EMERGENCY" or status == "VERTICAL_DESCENT" and e.get("charged", False):
        return "ACTIVE"
    if status in _GROUNDED | _FROZEN | _INVISIBLE:
        return None
    for row in (*frame.get("slots", ()), *frame.get("pads", ())):
        plan = plans.get(row.get("plan_hash"))
        if (str(row.get("aircraft_id")) == str(aircraft["id"])
                and plan is not None and plan.kind == "FINAL_EMERGENCY"):
            return "RESERVED"
    return "WAIT"


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
        self.bundle = bundle if isinstance(bundle, ReplayBundle) else ReplayBundle.load(
            bundle, stage3_trace_types=_STAGE3_OVERLAY_TYPES)
        self.config = self.bundle.config
        self._sample_times = tuple(float(row["time"]) for row in self.bundle.data["states"])
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
        self._plans = {h: TrajectoryPlan.from_dict(_plain(parameters), config=self.config)
                       for h, parameters in self.bundle.data["trajectories"].items()}

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

    def sampled_state(self):
        return self.bundle.data["states"][max(0, bisect_right(self._sample_times, self.time) - 1)]

    def aircraft_at(self):
        result = []
        for observed in self.sampled_state().get("aircraft", ()):
            aircraft = _plain(observed)
            h = aircraft.get("trajectory_hash")
            if h and aircraft.get("status") not in _FROZEN:
                if h not in self._plans:
                    raise ValueError("missing authoritative trajectory: " + str(h))
                state = self._plans[h].state(self.time)
                aircraft.update(p=state.p, v=state.v, a=state.a, jerk=state.jerk)
            elif self.time != float(self.sampled_state()["time"]) and aircraft.get("status") not in (
                _GROUNDED | _FROZEN | _INVISIBLE):
                raise ValueError("airborne arbitrary-time replay requires an authoritative plan")
            result.append(aircraft)
        return tuple(result)

    def slots_at(self):
        config = self.config
        frame = self.sampled_state()
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
        frame = self.sampled_state()
        aircraft = self.aircraft_at()
        result = []
        for i in range(self.config.pads.count):
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
        frame = self.sampled_state()
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
        stage3 = [_plain(r) for r in self.bundle.data.get("stage3_trace", ())
                  if r.get("record_type") in _STAGE3_OVERLAY_TYPES and r.get("as_of",0.)<=self.time]
        return {"minimum_current_3d_pair": current_min, "minimum_current_time": self.time,
            "latest_dispatch_timing": _plain(dispatch[-1]) if dispatch else None,
            "tentative_predictions": _plain(predictions),
            "stage3_proposals": [r for r in stage3 if r.get("record_type")=="WorkDescriptor"],
            "stage3_attempts": [r for r in stage3 if r.get("record_type")=="FactualResult"],
            "stage3_ranking": [r for r in stage3 if r.get("record_type")=="Rank"],
            "stage3_authority": "COMPUTATION_ONLY_SEE_CURRENT_COMMITS_FOR_EXECUTION",
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
        if overlay["stage3_proposals"] or overlay["stage3_attempts"]:
            lines.append(f'Stage3 computation: proposals={len(overlay["stage3_proposals"])}; attempts={len(overlay["stage3_attempts"])}; commits={len(overlay["current_commits"])}')
            if overlay["stage3_ranking"]:
                lines.append('Last proposal order: '+', '.join(overlay["stage3_ranking"][-1].get("ordered_ids",())[:3]))
            if overlay["stage3_attempts"]:
                attempt=overlay["stage3_attempts"][-1]
                lines.append(f'Last attempt {attempt.get("descriptor_id")}: {attempt.get("outcome")} / {attempt.get("reason", "")}; computation only')
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
                rescue_status = _recorded_rescue_status(selected, self.sampled_state(), self._plans)
                if rescue_status:
                    lines.append(f'Emergency {rescue_status}: recorded declaration; ' +
                                 {"WAIT": "no protected opportunity", "RESERVED": "protected rescue opportunity",
                                  "ACTIVE": "rescue maneuver executing"}[rescue_status])
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

    def snapshot(self, *, terminal_events=True):
        c = self.config
        frame = self.sampled_state()
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
                bool(h) and a.get("status") in _ACTIVE_MANEUVER,points,1.,None,None,declared,"emergency" if declared else "low_soc" if a["soc"]<=.35 else "normal",
                _recorded_rescue_status(a, frame, self._plans)))
        terminal = []
        for e in self.bundle.data["events"] if terminal_events else ():
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
