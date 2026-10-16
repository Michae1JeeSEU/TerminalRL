"""FT-RS replay adapter extending the existing Terminal3DViewer platform."""
from __future__ import annotations

from itertools import combinations
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
_FROZEN = {"ENERGY_FAILURE", "CENSORED_DUE_TO_FAILURE", "EXECUTED_HARD_SEPARATION_VIOLATION"}
_INVISIBLE = {"CLEARED", "COMPLETED", "REROUTED", "UNRELEASED_AT_FAILURE"}
_HOLD = {"HOLD_OUTER", "HOLD_MIDDLE", "HOLD_INNER", "HOLD", "HOLDING", "STABLE"}
_ACTIVE_MANEUVER = {"INBOUND", "TRANSFER_O_M", "TRANSFER_M_I", "FINAL_NORMAL",
                    "FINAL_EMERGENCY", "VERTICAL_DESCENT"}


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
                reservations = [r for r in frame.get("slots", ()) if tuple(r["resource"]) == (ring.id, i)
                    and float(r["start"]) <= self.time < float(r["end"] if r.get("end") is not None else math.inf)]
                departing = any(r.get("kind") in ("departure", "source", "departure_protection") for r in reservations)
                state = "DEPARTING" if departing else "OCCUPIED" if occupant else "RESERVED" if reservations else "FREE"
                result.append({"id": f"{ring.id}:{i}", "phase": slot_phase(ring.id, i, self.time, config),
                    "position": slot_state(ring.id, i, self.time, config).p, "state": state})
        return tuple(result)

    def overlays(self):
        frame = self.bundle.sampled_state(self.time)
        aircraft = self.aircraft_at()
        physical = [a for a in aircraft if a.get("status") not in _INVISIBLE]
        pairs = [(math.dist(a["p"], b["p"]), a["id"], b["id"]) for a, b in combinations(physical, 2)]
        current_min = min(pairs, default=None)
        episode_metrics = _plain(self.bundle.data["metrics"])
        return {"minimum_current_3d_pair": current_min, "minimum_current_time": self.time,
            "logged_metrics": _plain(frame.get("metrics", {})),
            "episode_metrics": episode_metrics,
            "separation_evidence": _separation_evidence(episode_metrics),
            "fixture_banner": "ANALYTIC FIXTURE - no physical episode outcome" if episode_metrics.get("fixture_only") else None,
            "current_commits": [a for a in aircraft if a.get("trajectory_hash") and a.get("status") in _ACTIVE_MANEUVER],
            "future_soft_dependencies": [r for r in (*frame.get("slots", ()), *frame.get("pads", ()))
                if float(r["start"]) > self.time and not r.get("committed", False)],
            "diagnostic_rejections_not_executed": [r for r in self.bundle.data.get("rejections", ()) if r["time"] <= self.time],
            "events": [e for e in self.bundle.data["events"] if e["time"] <= self.time]}

    def snapshot(self):
        c = self.bundle.config
        frame = self.bundle.sampled_state(self.time)
        aircraft = self.aircraft_at()
        rings = tuple(RingRenderState(r.id, r.radius_m, r.height_agl_m, r.radius_m,
            r.height_agl_m, r.slot_count, sum(a.get("ring") == r.id for a in aircraft)) for r in c.airspace.rings)
        gates = tuple(GateRenderState(i, (c.entry.radius_m*math.cos(a), c.entry.radius_m*math.sin(a),c.entry.height_agl_m),
            (c.entry.radius_m*math.cos(a), c.entry.radius_m*math.sin(a),c.entry.height_agl_m)) for i,a in enumerate(c.entry.gate_angles_rad))
        pads = []
        for i in range(c.pads.count):
            rows = [r for r in frame.get("pads", ()) if r["resource"] == i and r["start"] <= self.time < (r["end"] if r.get("end") is not None else math.inf)]
            occupant = next((a for a in aircraft if a.get("pad_id") == i and a.get("status") in ("SERVICE", "LANDED_SERVICE")), None)
            owner = str(occupant["id"]) if occupant else str(rows[0]["aircraft_id"]) if rows else None
            uid = self.uid_by_id.get(owner)
            end = max((float(r["end"]) for r in rows if r.get("end") is not None), default=self.time)
            pads.append(LandingZoneRenderState(i, pad_position(i,c), "occupied" if occupant else "reserved" if rows else "free",
                uid if rows and not occupant else None, uid if occupant else None, max(0.,end-self.time)))
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
        super().__init__(controller.snapshot(), VisualizationConfig(enabled=True,show_trajectories=True,
            show_labels=True,show_target_geometry=False,playback_speed=1.),off_screen=off_screen)
        widget=self._plotter.add_slider_widget(lambda t:self.draw(controller.seek(t)),controller.bundle.time_bounds,
            value=controller.time,title="Replay time (s)",pointa=(.18,.055),pointb=(.71,.055),color="gray",
            title_height=.016,slider_width=.015,tube_width=.003)
        widget.GetRepresentation().SetLabelHeight(.015)
        self._plotter.add_key_event("space", self.toggle_pause)
        self._plotter.add_key_event("Right",lambda:self.draw(controller.step()))
        self._plotter.add_key_event("n",lambda:self.draw(controller.jump_event(1)))
        self._plotter.add_key_event("p",lambda:self.draw(controller.jump_event(-1)))
        self._plotter.add_key_event("f",lambda:setattr(controller,"follow",not controller.follow))
        self._plotter.add_key_event("t",self.cycle_trajectories)
        self._plotter.add_key_event("l",self.cycle_slot_labels)
        for key,view in (("1","top_down"),("2","side"),("3","3d")):
            self._plotter.add_key_event(key,lambda v=view:self.set_view(v))
        self.set_view("3d")
        self.draw()

    def _add_interaction_controls(self):
        # Reuse picking, replace legacy playback choices with the contract set.
        self._plotter.enable_point_picking(self._select_uav,left_clicking=False,picker="point",show_message=False,show_point=False)
        self._plotter.add_checkbox_button_widget(lambda value:setattr(self.controller,"paused",value),value=True,
            position=(10,820),size=18)
        self._plotter.add_text("Pause: Space; step: Right; events: p/n; views: 1/2/3; follow: f; paths: t; slots: l",
            position=(35,825),font_size=9,color="black",name="replay-controls")
        widget=self._plotter.add_text_slider_widget(lambda label:self.controller.set_speed(float(label[:-1])),
            ["0.25x","1x","4x","10x"],value=1.,pointa=(.76,.945),pointb=(.97,.945),color="gray")
        representation=widget.GetRepresentation()
        representation.SetTitleHeight(.018)
        representation.SetSliderWidth(.012)
        representation.SetTubeWidth(.002)

    def _select_uav(self, picked_point, picker):
        super()._select_uav(picked_point,picker)
        self.controller.select(self.controller.id_by_uid.get(self._selected_uid),follow=self.controller.follow)
        self.draw()

    def toggle_pause(self):
        self.controller.paused = not self.controller.paused

    def cycle_trajectories(self):
        options=("all","selected","hidden")
        self.controller.set_trajectories(options[(options.index(self.controller.trajectories)+1)%3])
        self.draw()

    def cycle_slot_labels(self):
        modes=("selected_ring","all","hidden")
        self.controller.slot_labels=modes[(modes.index(self.controller.slot_labels)+1)%len(modes)]
        self.draw()

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
        colors={"FREE":"gray","RESERVED":"gold","OCCUPIED":"seagreen","DEPARTING":"darkorange"}
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
            self._plotter.add_text(overlay["fixture_banner"],position=(10,785),font_size=10,
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
        minimum=overlay["minimum_current_3d_pair"]
        text=[f"t={snapshot.time:.3f}s | aircraft={len(snapshot.uavs)} | service={snapshot.service_count} | cleared={snapshot.completed_count}",
            "Slot: F free / R reserved / O occupied / D departing; label mode="+self.controller.slot_labels,
            "Paths: normal blue / emergency red / future SOFT purple / rejected diagnostic magenta",
            "Pair balls radius 40 m; hard centre distance 80 m",
            f"Current true 3D minimum: {minimum} at {self.controller.time:.3f}s"]
        evidence=overlay["separation_evidence"]
        text.append(f'{evidence["label"]}: [{evidence["lower_bound_m"]}, {evidence["upper_bound_m"]}] m; '
            f'witness t={evidence["witness_time"]}s d={evidence["witness_distance_m"]}m pair={evidence["pair"]}')
        text.append(f'Actual commits={len(overlay["current_commits"])} | future SOFT reservations={len(overlay["future_soft_dependencies"])}')
        text.append("Pads 0-2: "+" / ".join(f'{p.lz_id}:{p.status} {p.service_remaining:.1f}s' for p in snapshot.landing_zones[:3]))
        text.append("Pads 3-5: "+" / ".join(f'{p.lz_id}:{p.status} {p.service_remaining:.1f}s' for p in snapshot.landing_zones[3:]))
        rejected=overlay["diagnostic_rejections_not_executed"]
        text.append(f'DIAGNOSTIC only, not executed: {len(rejected)} rejections'+
            (f'; latest {rejected[-1].get("aircraft_id")}: {rejected[-1].get("reason")}' if rejected else ""))
        if selected_aircraft:
            a=selected_aircraft; emergency=a.get("emergency",{})
            text.extend([f'Selected {a["id"]}: {a.get("status")} | {a.get("service_class")} | SOC={a["soc"]:.4f}',
                f'Ring={a.get("ring")} slot={a.get("slot")} | current commit={str(a.get("trajectory_hash"))[:12] if a.get("status") in _ACTIVE_MANEUVER else "none (stable or terminal)"}',
                f'Emergency={emergency.get("state",emergency.get("declared"))} | charged={emergency.get("charged",False)} | emergency cost={overlay["logged_metrics"].get("cost_emergency",0)}s'])
            for r in overlay["future_soft_dependencies"]:
                if str(r["aircraft_id"])==str(a["id"]):
                    text.append(f'SOFT {r["resource"]} starts {r["start"]:.3f}s; depends '+str([str(x)[:12] for x in r.get("dependencies",())]))
        for event in overlay["events"]:
            if str(event["type"]).lower() in ("energy_failure","rerouted","executed_hard_separation_violation"):
                text.append(f'{event["type"]}: {event.get("aircraft_id")} t={event["time"]:.3f}s p={event.get("details",{}).get("p")}')
        import textwrap
        lines=[line for paragraph in text for line in textwrap.wrap(paragraph.replace("|", ";"),width=105)]
        self._plotter.add_text("\n".join(lines),position=(10,605),font_size=7,color="black",name="rs-overlay")
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
