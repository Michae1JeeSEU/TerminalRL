"""FT-RS replay viewer extending the existing Terminal3DViewer platform.

ReplayController and its helpers remain importable here for compatibility.
"""
from __future__ import annotations

import math
import time as clock
import numpy as np

from .config import VisualizationConfig
from .pyvista_viewer import Terminal3DViewer
from .rotating_slots_controller import (
    ReplayController, ReplayBundle, _plain, _GROUNDED, _FROZEN, _INVISIBLE,
    _HOLD, _ACTIVE_MANEUVER, _UPSTREAM_POLICY, _UPSTREAM_POLICIES,
    _separation_evidence, _diagnostic_text, _time_text, _recorded_rescue_status,
    _hud_layout,
)


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
        selective=overlay.get("selective_replanning")
        if selective is not None:
            physical={str(a["id"]):a for a in self.controller.aircraft_at() if a.get("status") not in _INVISIBLE}
            labelled=[(aid,a) for aid,a in physical.items() if aid in selective["labels"]]
            if labelled:
                name="rs-selection-groups"
                self._plotter.add_point_labels(np.array([a["p"] for _,a in labelled]),
                    [selective["labels"][aid]+" "+aid for aid,_ in labelled],font_size=9,
                    text_color="darkgreen",point_size=0,shape=None,always_visible=True,
                    show_points=False,name=name,render=False)
                self._extra_names.add(name)
            for index,edge in enumerate(selective["blockers"]):
                source,target=str(edge.get("source")),str(edge.get("target"))
                if source in physical and target in physical:
                    name=f"rs-selection-blocker-{index}"
                    self._plotter.add_mesh(self._pv.Line(physical[source]["p"],physical[target]["p"]),
                        color="orange",line_width=3,opacity=.7,name=name,pickable=False,render=False)
                    self._extra_names.add(name)
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
