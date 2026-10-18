"""Concrete FT-RS-S1-1.0 fixtures, independent of the legacy terminal.

Fixture parameters and submitted plans are recorded by the evidence context.
An assertion only passes after the corresponding production operation ran.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import importlib.util
import math
import os
from pathlib import Path
import time

import numpy as np

from . import (load_config, slot_state, pad_position, TrajectoryPlan,
               generate_inbound, generate_transfer, generate_final,
               make_stable_plan, make_ground_plan, evaluate_power,
               integrate_energy, validate_plan, validate_separation,
               certified_minimum_distance)
from .demand import ArrivalManifest, ArrivalRequest
from .driver import DeterministicDriver
from .resources import SlotCalendar, PadCalendar, ResourceConflict, ResourceDependencyError
from .trajectory import Segment, descent_segments
from .world import RotatingSlotWorld

ROOT = Path(__file__).resolve().parents[3]
RINGS = (("outer", 600., 110., 37), ("middle", 450., 70., 28), ("inner", 300., 30., 18))
SCENARIOS = ("full_lifecycle", "all_ring_emergencies", "six_pads", "busy_release",
             "busy_failure", "threshold_crossing", "three_laps_controls")


class FixtureBlocked(RuntimeError):
    """The prescribed fixture was not executed; it cannot earn a pass."""


def world(e, **kwargs):
    e.action("RotatingSlotWorld", kwargs)
    w = RotatingSlotWorld(**kwargs)
    e.worlds.append(w)
    return w


def candidate(e, generator, *args, **kwargs):
    e.action(generator.__name__, {"args": args, **kwargs})
    result = generator(*args, **kwargs)
    e.observe("candidate", {"status": result.status, "reason": result.reason,
                             "details": result.details})
    if result.plan is not None:
        e.attempted_plans.append(result.plan.to_dict())
    return result


def feasible(e, generator, *args, **kwargs):
    result = candidate(e, generator, *args, **kwargs)
    e.check(result.accepted and result.plan is not None,
            "fixture trajectory must be FEASIBLE", "FEASIBLE", result.status)
    e.plans.append(result.plan)
    return result.plan


def seed(e, w, name="a", ring="outer", slot=0, soc=.8, service_class="standard", emergency=False):
    params = dict(aircraft_id=name, ring=ring, slot=slot, soc=soc,
                  service_class=service_class, emergency=emergency)
    e.action("seed_hold", params)
    return w.seed_hold(**params)


def service(e, w, name, pad=0, remaining=60., service_class="short", soc=.5):
    params = dict(aircraft_id=name, pad_id=pad, remaining=remaining,
                  service_class=service_class, soc=soc)
    e.action("seed_service", params)
    return w.seed_service(**params)


def advance(e, w, until, callback=None):
    e.action("advance", {"until": until, "driver": getattr(callback, "__class__", type(None)).__name__})
    from .qualification import write_json
    started = time.perf_counter(); last = [-math.inf]
    def monitored(snapshot, port):
        if snapshot.time-last[0] >= 30:
            write_json(e.output/"progress.json", {"status": "RUNNING", "case": e.case_id,
                "sim_time": snapshot.time, "phase_target_sim_time": until, "pid": os.getpid(),
                "phase_wall_seconds": time.perf_counter()-started, "metrics": port.metrics()})
            last[0] = snapshot.time
        if callback and callback is not False: callback(snapshot, port)
    w.advance(until, monitored)
    e.observe("world", {"time": w.time, "metrics": w.metrics()})
    write_json(e.output/"progress.json", {"status": "PHASE_COMPLETED", "case": e.case_id,
        "sim_time": w.time, "phase_target_sim_time": until, "pid": os.getpid(),
        "phase_wall_seconds": time.perf_counter()-started, "metrics": w.metrics()})


def submit(e, w, plans, **kwargs):
    e.action("submit", {"trajectory_hashes": [p.trajectory_hash for p in plans], **kwargs})
    r = w.submit(plans, **kwargs)
    e.observe("submit", {"accepted": r.accepted, "status": r.status, "reason": r.reason, "details": r.details})
    return r


def analytic(e, name="pair_a", z=(0.,), theta=(0., -.5), r=(100.,), duration=1.):
    c = load_config()
    p = TrajectoryPlan(name, "FIXTURE", 0, duration, profile_hash=c.profile_hash,
        config_hash=c.config_hash, segments=(Segment(0, duration, r, theta, z, "FIXTURE"),))
    e.action("analytic_fixture", p.to_dict())
    e.plans.append(p)
    return p


def separation(e, a, b, start=0., end=1., **kwargs):
    e.action("validate_separation", {"a": a.trajectory_hash, "b": getattr(b, "trajectory_hash", "broad_enclosure"),
                                     "start": start, "end": end, **kwargs})
    r = validate_separation(a, b, start, end, **kwargs)
    e.observe("separation", asdict(r))
    return r


def full_slots(e, w, rings=RINGS):
    for ring, _, _, n in rings:
        for j in range(n):
            seed(e, w, f"{ring}_{j}", ring, j, .9)


def single_final(e, *, ring="inner", emergency=False, soc=.8, service_class="short", pad=0):
    w = world(e)
    seed(e, w, ring=ring, soc=soc, service_class=service_class, emergency=emergency)
    p = feasible(e, generate_final, "a", ring, 0, pad, 0, emergency=emergency)
    return w, p


def run_geometry(case, e):
    c = load_config()
    if case == "A01":
        w = world(e)
        actual = [(r.id, r.radius_m, r.height_agl_m, r.slot_count) for r in w.config.airspace.rings]
        e.check(actual == list(RINGS), "three exact rings", list(RINGS), actual)
        e.check(c.airspace.rotation_direction == "clockwise" and c.airspace.common_angular_speed_radps == .035,
                "direction and omega", ["clockwise", .035], [c.airspace.rotation_direction, c.airspace.common_angular_speed_radps])
        e.check(len(c.entry.gate_angles_rad) == 4 and c.pads.count == 6, "four gates six pads", [4, 6], [len(c.entry.gate_angles_rad), c.pads.count])
    elif case in ("A02", "A03"):
        times = [37.137, 133.291] if case == "A02" else np.linspace(0, 3 * 2 * math.pi / .035, 2695)
        errors = []
        for ring, radius, height, n in RINGS:
            p = make_stable_plan(ring, ring, 0, end_time=float(max(times)) + 1)
            q = make_stable_plan(ring + "_other", ring, n // 2, end_time=p.end_time)
            e.plans.extend((p, q))
            initial = math.dist(p.state(0).p, q.state(0).p)
            for t in times:
                angle = -.035 * t
                expected = ((radius * math.cos(angle), radius * math.sin(angle), height),
                            (.035 * radius * math.sin(angle), -.035 * radius * math.cos(angle), 0),
                            (-.035**2 * radius * math.cos(angle), -.035**2 * radius * math.sin(angle), 0))
                st = p.state(float(t))
                errors.extend(math.dist(getattr(st, k), truth) for k, truth in zip(("p", "v", "a"), expected))
                errors.append(abs(math.dist(st.p, q.state(float(t)).p) - initial))
                errors.append(abs(math.hypot(*st.v[:2]) - radius * .035))
        e.action("evaluate_absolute_clock", {"times": list(map(float, times)), "slots": [0, "floor(n/2)"]})
        e.check(max(errors) < 1e-8, "analytic p/v/a and invariant pair geometry", "max error <1e-8", max(errors))
    elif case == "A04":
        w = world(e); full_slots(e, w)
        pairs = []
        values = list(w.aircraft.values())
        for i, a in enumerate(values):
            for b in values[i + 1:]:
                pa = w.trajectories[a.trajectory_hash]; pb = w.trajectories[b.trajectory_hash]
                r = validate_separation(TrajectoryPlan.from_dict(pa), TrajectoryPlan.from_dict(pb), 0, 10000)
                pairs.append(r.accepted)
        e.check(len(values) == 83 and all(pairs), "all83 slots, every pair continuously safe", [83, 3403, True], [len(values), len(pairs), all(pairs)])
        chords = [2 * r * math.sin(math.pi/n) for _, r, _, n in RINGS]
        e.check(min(chords) >= 100 and math.hypot(150, 40) > 155, "structural bounds", ">=100 and crossring155.2417", [chords, math.hypot(150, 40)])
    elif case == "A05":
        #83 occupied slots plus a genuinely admitted inbound requires a committed
        #release. A plain84th seed would only exercise an illegal double booking.
        raise FixtureBlocked("84-aircraft safe inbound fixture not yet constructed;83-only evidence does not satisfyA05")
    elif case in ("A06", "A08"):
        p = make_stable_plan("reverse", "outer", 0, end_time=10.) if case == "A06" else feasible(e, generate_transfer, "a", "outer", 0, 0, 0)
        if case == "A06": e.plans.append(p)
        s = p.segments[0]
        variants = [replace(s, theta=tuple(-x for x in s.theta))] if case == "A06" else [replace(s, duration=.1), replace(s, z=(121.,))]
        results = []
        for segment in variants:
            bad = replace(p, segments=(segment,), end_time=segment.end_time, trajectory_hash="")
            e.attempted_plans.append(bad.to_dict()); results.append(asdict(validate_plan(bad)))
            if case == "A06":
                state = bad.state(3.137)
                angular = (state.p[0]*state.v[1]-state.p[1]*state.v[0])/(state.p[0]**2+state.p[1]**2)
                e.check(angular > 0 and results[-1]["reason"] == "NONCANONICAL_STABLE",
                        "actualcounterclockwise translation rejected independentlyofyaw", "+.035rad/s/noncanonicalstable", [angular, results[-1]])
        e.check(all(not x["accepted"] for x in results), "invalid candidate rejected without clipping", False, results)
    elif case == "A07":
        w = world(e)
        for i, cls in enumerate(("short", "standard", "long")):
            a = seed(e, w, str(i), "outer", i * 10, .8, cls)
            e.check(a.battery_power_w == w.aircraft["0"].battery_power_w, "same motion state has same power regardless service", w.aircraft["0"].battery_power_w, a.battery_power_w)
        manifest = ArrivalManifest.generate(11, 180)
        profiles = {r.vehicle_profile_id for r in manifest.requests}
        e.check(profiles == {c.vehicle.profile_id}, "single profile in mixedservice demand", c.vehicle.profile_id, sorted(profiles))
    elif case == "A09":
        w = world(e)
        for j in range(6): service(e, w, str(j), j)
        distances = [math.dist(pad_position(j), pad_position((j + 1) % 6)) for j in range(6)]
        e.check(len(w.pads.reservations) == 6 and max(abs(d - 100) for d in distances) < 1e-10,
                "all six pads independently occupied and100m adjacent", [6, 100], [len(w.pads.reservations), distances])


def run_motion(case, e):
    if case in ("B01", "B03", "B07"):
        degrees = (37, 133) if case == "B01" else (0,) if case == "B07" else (37,)
        plans = [feasible(e, generate_transfer, f"a{deg}", "outer", 0, 0,
            0. if case == "B07" else (2 * math.pi - math.radians(deg))/.035) for deg in degrees]
        if case == "B07":
            plans += [feasible(e, generate_inbound, "in", 0, 2, 0), feasible(e, generate_final, "fin", "inner", 0, 0, 0)]
        errors = []
        for p in plans:
            if p.source_ring:
                a, b = p.state(p.start_time), slot_state(p.source_ring, p.source_slot, p.start_time)
                errors.extend(math.dist(getattr(a, k), getattr(b, k)) for k in ("p", "v", "a"))
            if p.target_ring:
                a, b = p.state(p.end_time), slot_state(p.target_ring, p.target_slot, p.end_time)
                errors.extend(math.dist(getattr(a, k), getattr(b, k)) for k in ("p", "v", "a"))
                e.check(math.dist(a.p, slot_state(p.target_ring, p.target_slot, p.start_time).p) > 1,
                        "target follows future absoluteclock", ">1m from currenttarget", math.dist(a.p, slot_state(p.target_ring, p.target_slot, p.start_time).p))
            for a, b in zip(p.segments, p.segments[1:]):
                errors.extend(math.dist(getattr(a.state(a.end_time), k), getattr(b.state(b.start_time), k)) for k in ("p", "v", "a"))
            e.check(validate_plan(p).accepted, "continuous validator accepts complete plan", True, asdict(validate_plan(p)))
        e.check(max(errors) < 1e-7, "C2 junctions", "<1e-7", max(errors))
        if case in ("B01", "B07"):
            for p in plans:
                if p.kind == "INBOUND":
                    request = ArrivalRequest(p.aircraft_id, 0., 0, .9, "short")
                    w = world(e, manifest=ArrivalManifest((request,), seed=e.seed))
                    admitted = []
                    advance(e, w, 0., lambda s, port: admitted.append(port.admit(p.aircraft_id, p)) if p.aircraft_id not in port.aircraft else None)
                    e.check(admitted and admitted[0].accepted, "knowninbound actuallyadmitted", True, [asdict(x) for x in admitted])
                else:
                    w = world(e); seed(e, w, p.aircraft_id, p.source_ring, p.source_slot, .9)
                    e.check(submit(e, w, (p,)).accepted, "knownpositive actuallycommitted", True, p.trajectory_hash)
                e.check(w.active.get(p.aircraft_id) is p or w.soft.get(p.aircraft_id) is p,
                        "checkedobject is committedobject", p.trajectory_hash, p.trajectory_hash)
                advance(e, w, p.end_time)
                actual = w.aircraft[p.aircraft_id]; endpoint = p.state(p.end_time)
                errors = [math.dist(getattr(actual, k), getattr(endpoint, k)) for k in ("p", "v", "a")]
                e.check(not w.failed and max(errors) <= 1e-6 and any(x["type"] in ("maneuver_started", "admitted") and x["details"].get("trajectory_hash") == p.trajectory_hash for x in w.events),
                        "positiveplan actuallyexecuted toC2endpoint", "nohardfailure/endpoint<=1e-6", [actual.status, errors])
    elif case == "B02":
        w = world(e); service(e, w, "release", remaining=1.); seed(e, w, ring="inner")
        outcomes = []
        def action(snapshot, wrld):
            if abs(snapshot.time - 1) < 1e-9:
                p = feasible(e, generate_final, "a", "inner", 0, 0, 1.)
                outcomes.append(submit(e, wrld, (p,)).accepted)
        advance(e, w, 1, action)
        e.check(outcomes == [True] and 1 in w.decision_times, "resource event permits action after1s hold", [True], outcomes)
    elif case == "B04":
        r = candidate(e, generate_transfer, "a", "outer", 0, 0, 0, durations=[.1])
        e.check(not r.accepted and r.status in ("NOT_FOUND", "PROVEN_INFEASIBLE"), "tooearly candidate not forced", "NOT_FOUND or proven", r.status)
    elif case in ("B05", "B06", "B08", "B09"):
        w = world(e); seed(e, w)
        p = feasible(e, generate_transfer, "a", "outer", 0, 0, 0)
        if case == "B05":
            bad = replace(p, target_ring="inner", trajectory_hash="")
            r = submit(e, w, (bad,)); e.check(not r.accepted, "normal skipring rejected", False, r.reason)
            f = feasible(e, generate_final, "a", "outer", 0, 0, 0, emergency=True)
            r = submit(e, w, (f,)); e.check(r.reason == "NO_EMERGENCY_ELIGIBILITY", "healthy directpad rejected", "NO_EMERGENCY_ELIGIBILITY", r.reason)
        elif case == "B06":
            bad = replace(p, target_ring="outer", target_slot=1, trajectory_hash="")
            r = submit(e, w, (bad,)); e.check(not r.accepted, "same ring slotchange rejected", False, r.reason)
        elif case == "B08":
            results = []
            for s in (replace(p.segments[0], r=(600., 10.)), replace(p.segments[0], theta=(0., 1.))):
                bad = replace(p, segments=(s,), trajectory_hash=""); e.attempted_plans.append(bad.to_dict())
                results.append(asdict(validate_plan(bad)))
            stable = make_stable_plan("halfturn", "outer", 0, end_time=math.pi/.035); e.plans.append(stable)
            e.check(all(not r["accepted"] for r in results) and validate_plan(stable).accepted,
                    "outward/reverse rejected;halfturn hold accepted", [False, False, True], [results, asdict(validate_plan(stable))])
        else:
            bad = replace(p, segments=(Segment(0, p.end_time, (600.,), (0.,), (110.,)),), trajectory_hash="")
            e.attempted_plans.append(bad.to_dict()); r = validate_plan(bad)
            good = feasible(e, generate_final, "descent", "inner", 0, 0, 0)
            e.check(r.reason == "HOVER_WAIT" and validate_plan(good).accepted, "hover rejected continuous descent accepted", "HOVER_WAIT", r.reason)
    elif case in ("B10", "B11"):
        p = feasible(e, generate_final, "a", "inner", 0, 0, 0)
        capture = p.segments[0].end_time
        errors = [math.dist(p.state(capture).p, (100., 0., 15.)), math.dist(p.state(capture).v, (0., 0., -.5)), math.dist(p.state(p.end_time).p, (100., 0., 0.)), math.dist(p.state(p.end_time).v, (0., 0., 0.))]
        states = [s.state(float(t)) for s in p.segments[1:] for t in np.linspace(s.start_time, s.end_time, 401)]
        peaks = [max(abs(getattr(s, k)[2]) for s in states) for k in ("v", "a", "jerk")]
        q = integrate_energy(p, capture, p.end_time)
        e.check(abs(p.end_time-capture-13.5) < 1e-8 and max(errors) < 1e-7 and q.joules > 0,
                "full13.5s analytic descent endpoint and energy", [13.5, "C2", ">0J"], [p.end_time-capture, errors, asdict(q)])
        e.check(peaks[0] <= 1.5+1e-10 and peaks[1] <= 1 and peaks[2] <= .5 and validate_plan(p).accepted,
                "continuous plan certificate and sampled extrema", [1.5, 1, .5], peaks)
    elif case == "B12":
        w = world(e); seed(e, w)
        p = feasible(e, generate_transfer, "a", "outer", 0, 0, 0)
        e.check(submit(e, w, (p,)).accepted, "start transfer", True, "accepted")
        bad = replace(p, target_slot=1, trajectory_hash=""); e.attempted_plans.append(bad.to_dict())
        e.check(submit(e, w, (bad,)).reason == "COMMITTED_MODIFICATION", "cannot redirect active plan", "COMMITTED_MODIFICATION", w.rejections[-1]["reason"])
        nextp = feasible(e, generate_transfer, "a", "middle", 0, 0, p.end_time+1, source_version=1)
        e.check(submit(e, w, (nextp,)).accepted and w.active["a"] is p, "future soft plan allowed", True, "active identity preserved")
    elif case == "B13":
        good = feasible(e, generate_final, "a", "outer", 0, 0, 0, emergency=True)
        miss = candidate(e, generate_final, "a", "outer", 0, 0, 0, emergency=True, durations=list(range(20, 181, 5)))
        e.check(miss.status == "NOT_FOUND" and validate_plan(good).accepted,
                "finite unsuccessful search is NOT_FOUND, positive fixture remains valid", ["NOT_FOUND", True], [miss.status, validate_plan(good).accepted])


def run_safety(case, e):
    if case == "D01":
        a = analytic(e)
        for d in (80., 80.5, 79.999, 78.):
            b = analytic(e, "d"+str(d), z=(d,))
            r = separation(e, a, b)
            e.check(r.accepted is (d >= 80), "unique exact80m boundary", d >= 80, asdict(r))
    elif case == "D02":
        a = analytic(e); b = analytic(e, "crossing", z=(80.999, -40., 400.))
        endpoints = [math.dist(a.state(t).p, b.state(t).p) for t in (0, .2)]
        r = separation(e, a, b, 0, .2)
        minimum = certified_minimum_distance(a, b, 0, .2)
        e.observe("certified_minimum", minimum)
        e.check(min(endpoints) > 80 and not r.accepted and minimum["upper_bound_m"] < 80,
                "safe output endpoints do not mask interior crossing", "reject;certifiedmin79.999", [endpoints, asdict(r), minimum])
    elif case == "D03":
        run_emergency("F02", e)
    elif case == "D04":
        a = analytic(e)
        class Broad:
            def state(self, t):
                from .models import KinematicState
                return KinematicState((0, 0, 100))
            def bounds(self, a, b, derivative=0): return ((-1000, 1000),)*3
        r = separation(e, a, Broad(), max_depth=0)
        e.check(r.status == "UNKNOWN" and not r.accepted, "unresolved enclosure rejects", "UNKNOWN", asdict(r))
    elif case == "D05":
        a = analytic(e)
        results = [separation(e, a, analytic(e, str(z), z=(z,))).accepted for z in (81., 79.)]
        e.check(results == [True, False], "sameXY uses real3D separation", [True, False], results)
    elif case == "D06":
        a = feasible(e, generate_transfer, "a", "outer", 0, 0, 0)
        b = make_stable_plan("later", "middle", 0, start_time=a.end_time+1, end_time=a.end_time+20); e.plans.append(b)
        r = separation(e, a, b, a.end_time+1, a.end_time+20)
        e.check(not r.accepted, "stable continuation checked after maneuver end", False, asdict(r))
    elif case == "D07":
        p = feasible(e, generate_transfer, "a", "outer", 0, 0, 0)
        w = world(e); seed(e, w)
        before = p.end_time
        object.__setattr__(p, "end_time", before+.1)
        try:
            r = validate_plan(p)
            refused = submit(e, w, (p,))
        finally: object.__setattr__(p, "end_time", before)
        e.action("tamper_after_certificate", {"field": "end_time", "old": before, "new": before+.1})
        e.check(r.reason == "TRAJECTORY_HASH_MISMATCH", "tampered plan certificate refused", "TRAJECTORY_HASH_MISMATCH", r.reason)
        e.check(refused.reason == "TRAJECTORY_HASH_MISMATCH" and not w.active,
                "tamperedplan cannot enterexecution", "TRAJECTORY_HASH_MISMATCH", asdict(refused))
    elif case == "D08":
        results = []
        for enabled in (False, True):
            w, p = single_final(e)
            w.cache_enabled = enabled
            e.check(submit(e, w, (p,)).accepted, "same final accepted", True, "accepted")
            advance(e, w, p.end_time)
            results.append({"events": w.events, "metrics": w.metrics(), "soc": w.aircraft["a"].soc})
        e.check(results[0] == results[1], "cache preserves events metrics andSOC", results[0], results[1])


def run_resources(case, e):
    if case in ("C01", "C02"):
        manifests = [ArrivalManifest.generate(s, 180) for s in (11, 23, 47, 71, 101)]
        rows = [r for m in manifests for r in m.requests]
        e.action("ArrivalManifest.generate", {"seeds": [11, 23, 47, 71, 101], "rate_per_hour": 180, "arrival_window": 3600})
        e.observe("manifests", [m.to_dict() for m in manifests])
        if case == "C01":
            counts = [sum(r.gate_id == j for r in rows) for j in range(4)]
            e.check(manifests[0] == ArrivalManifest.generate(11, 180) and all(n > 0 for n in counts),
                    "fixed repeated seed and all gates represented", "identical manifest/all4", counts)
            e.check(tuple(load_config().entry.gate_probabilities) == (.25,)*4, "frozen equal gate law", [.25]*4, list(load_config().entry.gate_probabilities))
        else:
            values = [r.initial_soc for r in rows]
            e.check(len(values) > 500 and min(values) > .60 and max(values) < .90 and len(set(values)) == len(values),
                    "large truncatedcontinuous SOC sample without boundary atoms", "N>500,0.6<SOC<0.9,unique", [len(values), min(values), max(values), len(set(values))])
            for value in (.60, .90):
                request = ArrivalRequest(str(value), 0, 0, value, "short")
                e.check(request.initial_soc == value, "boundary request value preserved", value, request.initial_soc)
    elif case in ("C03", "C05"):
        req = ArrivalRequest("r", 0, 0, .85, "short")
        w = world(e, manifest=ArrivalManifest((req,)))
        full_slots(e, w, RINGS[:1])
        p = feasible(e, generate_inbound, "r", 0, 1, 0)
        outputs = []
        def action(s, wrld):
            if "r" not in wrld.aircraft:
                outputs.append(wrld.admit("r", p))
        advance(e, w, 0, action)
        a = w.aircraft["r"]
        e.check(outputs and not outputs[0].accepted and a.status == "REROUTED" and not a.admitted,
                "fullouter request never becomes managedaircraft", "REROUTED", a.to_dict())
        e.check(w.metrics()["cost_reroute"] == 600 and len(w.active) == 0 and len([a for a in w.aircraft.values() if a.status.startswith("HOLD")]) == 37,
                "reroute once; emptyinner not an admission shortcut", 600, w.metrics())
    elif case == "C04":
        req = ArrivalRequest("r", 0, 0, .85, "short")
        w = world(e, manifest=ArrivalManifest((req,))); full_slots(e, w, RINGS[:1])
        p = feasible(e, generate_transfer, "outer_1", "outer", 1, 1, .001)
        result = submit(e, w, (p,))
        if not result.accepted:
            raise FixtureBlocked("fullouter futuredeparture fixture candidate rejected: "+result.reason)
        inbound = feasible(e, generate_inbound, "r", 0, 1, 0, durations=[55.])
        results = []
        advance(e, w, 0, lambda s, wrld: results.append(wrld.admit("r", inbound)) if "r" not in wrld.aircraft else None)
        if not results[0].accepted:
            raise FixtureBlocked("fullouter future-release inbound rejected: "+results[0].reason)
        e.check(w.aircraft["r"].status == "INBOUND" and inbound.trajectory_hash in w.dependencies,
                "fullcurrentouter admitscertified futurevacancy", "INBOUND/explicitdependency", dict(w.dependencies))
    elif case == "C06":
        c = SlotCalendar(); c.reserve(("outer", 1), "inflight1", 20, math.inf, committed=True)
        e.action("reserve_slot", {"resource": ["outer", 1], "aircraft": "inflight1", "interval": [20, None], "committed": True})
        refused = False
        try: c.reserve(("outer", 1), "inflight2", 30, math.inf, committed=True)
        except ResourceConflict: refused = True
        e.check(refused and len(c.reservations) == 1, "reservedfuture slot is notfree", "ResourceConflict", c.to_dict())
    elif case == "C07":
        w = world(e); seed(e, w)
        p = feasible(e, generate_transfer, "a", "outer", 0, 0, 0)
        e.check(submit(e, w, (p,)).accepted, "transfer accepted", True, "accepted")
        advance(e, w, .2)
        distance = math.dist(w.aircraft["a"].p, slot_state("outer", 0, .2).p)
        protected = [r for r in w.slots.reservations if r.resource == ("outer", 0) and r.start <= .2 < r.end]
        e.check(distance < 80 and protected, "sourceprotected while departure inside80m", "distance<80 andcalendarpresent", [distance, [r.to_dict() for r in protected]])
    elif case == "C08":
        req = ArrivalRequest("r", 0, 0, .85, "short")
        w = world(e, manifest=ArrivalManifest((req,))); seed(e, w, "departing", "outer", 1)
        p = feasible(e, generate_transfer, "departing", "outer", 1, 1, .001)
        e.check(submit(e, w, (p,)).accepted, "future departure committedsoft", True, "accepted")
        inbound = feasible(e, generate_inbound, "r", 0, 1, 0, durations=[55.])
        outputs = []
        advance(e, w, 0, lambda s, wrld: outputs.append(wrld.admit("r", inbound)) if "r" not in wrld.aircraft else None)
        e.check(outputs[0].accepted, "inbound relies futuredeparture", True, asdict(outputs[0]))
        r = w.cancel("departing"); e.action("cancel", {"aircraft": "departing"})
        e.check(r.reason == "RESOURCE_DEPENDENCY", "barecancel prohibited", "RESOURCE_DEPENDENCY", r.reason)
        new = feasible(e, generate_transfer, "departing", "outer", 1, 1, 0)
        r = submit(e, w, (new,))
        e.check(r.accepted and new.trajectory_hash in w.dependencies[inbound.trajectory_hash], "equivalentearlier release atomicallyreplaces dependency", True, dict(w.dependencies))
    elif case == "C09":
        plans = []
        for name, slot in (("a", 0), ("b", 1)):
            w = world(e); seed(e, w, name, slot=slot)
            p = feasible(e, generate_transfer, name, "outer", slot, 0, 0)
            e.check(submit(e, w, (p,)).accepted, "each isolatedcandidate accepted", True, name)
            plans.append(p)
        w = world(e); seed(e, w, "a"); seed(e, w, "b", slot=1)
        result = submit(e, w, plans)
        e.check(not result.accepted and not w.active, "conflicting jointbatch rejectedatomically", False, asdict(result))
    elif case == "C10":
        for cls, seconds in (("short", 60.), ("standard", 120.), ("long", 240.)):
            w, p = single_final(e, service_class=cls)
            e.check(submit(e, w, (p,)).accepted, "start final", True, cls)
            advance(e, w, p.end_time)
            a = w.aircraft["a"]
            e.check(a.touchdown_time == p.end_time and a.service_end == p.end_time+seconds,
                    "serviceclock startsat touchdown", [p.end_time, p.end_time+seconds], [a.touchdown_time, a.service_end])
            advance(e, w, a.service_end)
            e.check(w.aircraft["a"].status == "CLEARED", "serviceentity clearsat exactend", "CLEARED", w.aircraft["a"].status)
    elif case == "C11":
        w = world(e); plans = []
        for j in range(6):
            seed(e, w, str(j), "inner", j*3, service_class=("short", "standard", "long")[j % 3])
            plans.append(feasible(e, generate_final, str(j), "inner", j*3, j, 0))
        r = submit(e, w, plans)
        e.check(r.accepted and len(w.active) == 6 and len(w.pads.reservations) == 6,
                "sixsafe finals approved asjointbatch", [True, 6], [asdict(r), len(w.active)])
    elif case in ("C12", "C13"):
        w = world(e); service(e, w, "busy", remaining=30 if case == "C12" else 200)
        seed(e, w, ring="inner", soc=.8 if case == "C12" else .30, emergency=case == "C13")
        p = feasible(e, generate_final, "a", "inner", 0, 0, 0, emergency=case == "C13")
        before = w.aircraft["busy"]
        r = submit(e, w, (p,))
        e.check(r.accepted is (case == "C12") and w.aircraft["busy"] == before,
                "normalfuture availability vs nonpreemptiblecurrent service", case == "C12", asdict(r))
        if case == "C12": e.check(p.end_time-13.5 > 30, "padclear before15m", ">30", p.end_time-13.5)
        else: e.check(r.reason == "RESOURCE_CONFLICT", "emergencycannot overlapservice at arrival", "RESOURCE_CONFLICT", r.reason)
    elif case == "C14":
        reqs = (ArrivalRequest("x", .35, 0, .75, "short"), ArrivalRequest("y", .35, 1, .75, "long"))
        w = world(e, manifest=ArrivalManifest(reqs)); service(e, w, "service", remaining=.35)
        seed(e, w, "future_start", "inner", 0, .9)
        plan = feasible(e, generate_final, "future_start", "inner", 0, 0, .35)
        e.check(submit(e, w, (plan,)).accepted, "futurestart scheduledat releaseinstant", True, .35)
        advance(e, w, .6)
        kinds = [x["type"] for x in w.events if abs(x["time"]-.35) < 1e-9]
        e.check(kinds[:3] == ["pad_service_completed", "request_released", "request_released"] and w.decision_times.count(.35) == 1,
                "all sameinstant events settled before onedecision", ["pad_service_completed", "request_released", "request_released"], [kinds, w.decision_times])
        e.check("maneuver_started" in kinds and w.active.get("future_start") is plan,
                "sametime futurestart uses releasedresource atomically", "maneuver_started", kinds)
    elif case == "C15":
        p = feasible(e, generate_final, "a", "inner", 0, 0, 0)
        ground = make_ground_plan("service", 0, 0, p.end_time+60); e.plans.append(ground)
        result = separation(e, p, ground, p.end_time-13.5, p.end_time)
        e.check(not result.accepted, "unclearedground aircraft remains safetyentity", False, asdict(result))


def run_energy(case, e):
    if case == "E01":
        spec = importlib.util.spec_from_file_location("stage1_independent_energy", ROOT/"new10.16/energy_reference_v1_0.py")
        ref = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)
        states = {"hover": ((0, 0, 0), (0, 0, 0)), "descent_1p5": ((0, 0, -1.5), (0, 0, 0))}
        states.update({ring: (slot_state(ring, 0, 0).v, slot_state(ring, 0, 0).a) for ring, _, _, _ in RINGS})
        expected = ref.checkpoints(); actual = {}
        for name, (v, a) in states.items():
            power = evaluate_power(v, a); actual[name] = asdict(power)
            for key in ("battery_w", "shaft_w", "current_a"):
                e.check(abs(getattr(power, key)/expected[name][key]-1) <= 1e-6, "independentformula "+name+"/"+key, expected[name][key], getattr(power, key))
        e.action("independent_reference_checkpoints", {"states": states, "program": str(ROOT/"new10.16/energy_reference_v1_0.py")})
        e.observe("power", actual)
    elif case in ("E02", "E03"):
        st = slot_state("outer", 0, 0)
        v, a = (st.v, st.a) if case == "E02" else ((21., 0., 0.), (1., 0., 0.))
        base, changed = evaluate_power(v, (0, 0, 0)), evaluate_power(v, a)
        e.action("evaluate_power", {"v": v, "baseline_a": [0, 0, 0], "changed_a": a})
        e.check(changed.battery_w > base.battery_w, "acceleration changes requiredthrust andpower", ">baseline", [asdict(base), asdict(changed)])
    elif case == "E04":
        for emergency in (False, True):
            p = feasible(e, generate_final, "a"+str(emergency), "inner", 0, 0, 0, emergency=emergency)
            samples = [evaluate_power(s.state(float(t)).v, s.state(float(t)).a) for s in p.segments[1:] for t in np.linspace(s.start_time, s.end_time, 51)]
            q = integrate_energy(p, p.end_time-13.5, p.end_time)
            e.check(all(x.valid and x.battery_w > 0 for x in samples) and q.soc_used > 0, "normal/emergency descent consumespositiveenergy", ">0", [min(x.battery_w for x in samples), q.soc_used])
    elif case in ("E05", "E06"):
        v, a = ((0, 0, -10.), (0, 0, 0)) if case == "E05" else ((0, 0, 0), (0, 0, 100.))
        r = evaluate_power(v, a); e.action("evaluate_power", {"v": v, "a": a})
        expected = "ENERGY_MODEL_DOMAIN" if case == "E05" else "POWER_LIMIT"
        e.check(not r.valid and r.status == expected, "domain/powerlimit refuses instead ofclipping", expected, asdict(r))
        if case == "E06":
            for emergency in (False, True):
                w, p = single_final(e, emergency=emergency)
                s = p.segments[0]
                coeff = list(s.z)+[0.]*(7-len(s.z))
                #C2endpointpreserving bump: midpoint upwardacceleration is
                #large enough toexceed theactualpower model limits. This
                #negativefixture also violatesheight/dynamics; recordallfacts.
                for j, v in ((3, -1e6), (4, 3e6), (5, -3e6), (6, 1e6)): coeff[j] += v
                bad = replace(p, segments=(replace(s, z=tuple(coeff)), *p.segments[1:]), trajectory_hash="")
                state = bad.state(s.duration/2)
                power = evaluate_power(state.v, state.a)
                e.attempted_plans.append(bad.to_dict())
                e.observe("overpower_candidate", {"emergency": emergency, "state": asdict(state), "power": asdict(power),
                    "also_violates": ["altitude", "vertical_acceleration"]})
                rejected = submit(e, w, (bad,))
                e.check(power.status == "POWER_LIMIT" and not rejected.accepted and not w.active,
                        "normal/emergency overpowercandidate neverexecutes", "POWER_LIMIT andsubmissionrejected", [asdict(power), asdict(rejected)])
    elif case == "E07":
        raise FixtureBlocked("cumulativepeak fixture spanningmultiple executedmaneuvers with individually<120s peak andtotal>120s not yet constructed")
    elif case in ("E08", "E09", "E10"):
        p = feasible(e, generate_final, "a", "inner", 0, 0, 0)
        whole = integrate_energy(p, 0, p.end_time)
        before = integrate_energy(p, 0, p.end_time-13.5)
        soc = .25+whole.soc_used+(-.001 if case == "E08" else 0.)
        if case == "E09": soc = .25+before.soc_used
        w = world(e); seed(e, w, ring="inner", soc=soc)
        r = submit(e, w, (p,))
        e.check(r.accepted is (case == "E10"), "normalreserve tested atcompletedtouchdown", case == "E10", [soc, soc-whole.soc_used, asdict(r)])
        if case != "E10": e.check(r.reason == "LANDING_RESERVE", "belowreserve explicitreason", "LANDING_RESERVE", r.reason)
        if case == "E10":
            advance(e, w, p.end_time)
            a = w.aircraft["a"]
            touchdown = [x for x in w.events if x["type"] == "touchdown" and x["aircraft_id"] == "a"]
            e.check(a.status == "LANDED_SERVICE" and a.soc == .25
                    and a.normal_final_touchdown_soc == .25 and touchdown
                    and touchdown[-1]["details"]["below_reserve"] is False
                    and w.metrics()["landing_below_reserve"] == 0,
                    "exactreserve normalfinal executes complete descent to canonicaltouchdownSOC", .25,
                    {"aircraft": a.to_dict(), "touchdown": touchdown, "metrics": w.metrics()})
    elif case == "E11":
        w, p = single_final(e, emergency=True, soc=.30)
        e.check(submit(e, w, (p,)).accepted, "lowreserve rescueaccepted", True, "accepted")
        advance(e, w, p.end_time)
        m = w.metrics()
        e.check(0 < w.aircraft["a"].soc < .25 and m["emergency_landed"] == 1 and m["landing_below_reserve"] == 1 and m["energy_failures"] == 0,
                "belowreserve rescueclassifiedcorrectly", [1, 1, 0], m)
    elif case == "E12":
        w = world(e, manifest=ArrivalManifest((ArrivalRequest("unreleased", 10, 0, .75, "short"),)))
        for j in range(6): service(e, w, "pad"+str(j), j, 200)
        a = seed(e, w, "failing", soc=.00001); seed(e, w, "other", "middle")
        predicted = .00001*85*252*3600/evaluate_power(a.v, a.a).battery_w
        advance(e, w, 1.)
        e.check(w.failed and abs(w.time-predicted) < 1e-6 and 0 < w.time < .2, "failurelocated between samples", predicted, w.time)
        e.check(w.aircraft["other"].status == "CENSORED_DUE_TO_FAILURE" and w.metrics()["unreleased"] == 1,
                "otheraircraft censored andfuture demandunrevealed", "CENSORED_DUE_TO_FAILURE", w.metrics())
        old = w.time; advance(e, w, 10)
        e.check(w.time == old and w.metrics()["cost_failure"] == 1000000, "physicalepisode remainsstopped", old, w.time)
    elif case == "E13":
        req = ArrivalRequest("r", 0, 0, .85, "short")
        w = world(e, manifest=ArrivalManifest((req,)))
        p = feasible(e, generate_inbound, "r", 0, 1, 0)
        results = []
        advance(e, w, 0, lambda s, wrld: results.append(wrld.admit("r", p)) if "r" not in wrld.aircraft else None)
        e.check(results[0].accepted and w.metrics()["admitted"] == 1, "localnormalreference sufficeswithout globalcertificate", True, asdict(results[0]))
    elif case == "E14":
        raise FixtureBlocked("onecombined manifest includingnormal/emergency/reroute/failure/unreleased populations not yet constructed; peroutcome unitconservation is insufficient")


def run_emergency(case, e):
    if case == "F12":
        selected = None
        for free_pad in (5, 0, 1, 2, 3, 4):
            w = world(e)
            for j in range(6):
                if j != free_pad: service(e, w, "busy"+str(j), j, 300.)
            seed(e, w, "rescue", "outer", 0, .9, emergency=True)
            advance(e, w, 0)
            if "rescue" in w.active:
                selected = w; break
            e.observe("unique_freepad_unavailable_candidate", {"free_pad": free_pad, "rejections": w.rejections})
        if selected is None: raise FixtureBlocked("solefreepad outerrescue fixture NOT_FOUND against5occupiedgroundpads; actualattempts retained")
        p = selected.active["rescue"]
        e.check(p.kind == "FINAL_EMERGENCY" and selected.metrics()["cost_emergency"] == 3600,
                "productioninterface attemptshighpenaltyouter rescue atsolefreepad", ["FINAL_EMERGENCY", 3600], [p.kind, selected.metrics()["cost_emergency"]])
        return
    if case in ("F01", "F11"):
        for ring, _, _, _ in RINGS:
            w, p = single_final(e, ring=ring, emergency=True, soc=.9)
            q = integrate_energy(p, p.start_time, p.end_time)
            e.check(submit(e, w, (p,)).accepted, "qualifieddirectrescue starts", True, ring)
            expected = {"inner": 1200., "middle": 2400., "outer": 3600.}[ring]
            e.check(w.metrics()["cost_emergency"] == expected, "costat actualstart ring", expected, w.metrics()["cost_emergency"])
            advance(e, w, p.end_time)
            a = w.aircraft["a"]
            e.check(a.touchdown_time == p.end_time and a.status == "LANDED_SERVICE" and a.emergency.start_ring == ring,
                    "arbitraryring directrescue completes atomicdescent", [ring, p.end_time], [a.emergency.start_ring, a.touchdown_time])
            e.check(abs(.9-a.soc-q.soc_used) < 1e-8 and w.metrics()["cost_emergency"] == expected,
                    "penaltyonce and physicalenergy unmultiplied", [q.soc_used, expected], [.9-a.soc, w.metrics()["cost_emergency"]])
        if case == "F11":
            for ring, _, _, _ in RINGS:
                w = world(e); w.auto_emergency = False
                e.action("named_failed_episode_fixture", {"auto_emergency": False, "monitoring": True,
                    "failure_aircraft": "failure", "rescue_outcome": "censored_due_to_other_aircraft_failure"})
                seed(e, w, "rescue", ring, 0, .30, emergency=True)
                seed(e, w, "failure", "outer", 18, .00001)
                p = feasible(e, generate_final, "rescue", ring, 0, 0, 0, emergency=True)
                e.check(submit(e, w, (p,)).accepted, "rescuevalidlystarts beforeindependentfailure", True, ring)
                expected = {"inner": 1200., "middle": 2400., "outer": 3600.}[ring]
                advance(e, w, .2)
                rescue = w.aircraft["rescue"]; energy = integrate_energy(p, 0, w.time)
                e.check(w.failed and w.aircraft["failure"].status == "ENERGY_FAILURE" and rescue.status == "CENSORED_DUE_TO_FAILURE",
                        "otherbody failure truncates rescuewithoutclaimingrescuefailure", "ENERGY_FAILURE + CENSORED_DUE_TO_FAILURE", [w.aircraft["failure"].status, rescue.status])
                e.check(w.metrics()["cost_emergency"] == expected and abs(rescue.energy_j-energy.joules) < .01,
                        "failed episode retainsonefee andunweightedphysicalenergy", [expected, energy.joules], [w.metrics()["cost_emergency"], rescue.energy_j])
    elif case == "F02":
        p = feasible(e, generate_final, "outer_rescue", "outer", 0, 0, 0, emergency=True)
        #Candidate passes through the two occupiedring radii. Select the
        #actual closest existing slot rather than declaring a magic corridor.
        results = []
        for ring, radius, _, n in RINGS[1:]:
            samples = np.linspace(0, p.segments[0].end_time, 401)
            t = min(samples, key=lambda x: abs(math.hypot(*p.state(float(x)).p[:2])-radius))
            slot = min(range(n), key=lambda j: math.dist(p.state(float(t)).p, slot_state(ring, j, float(t)).p))
            other = make_stable_plan(ring, ring, slot, end_time=p.end_time+20); e.plans.append(other)
            r = separation(e, p, other, 0, p.end_time)
            results.append(asdict(r))
        e.check(any(not r["accepted"] for r in results), "outerrescue checked againstall crossedrings", "atleastone conflictrefused", results)
    elif case == "F03":
        w, p = single_final(e, ring="outer", emergency=True)
        #Eligibility is intentionally removed from theinitialfixture, before
        #any scheduling or execution, to model ahealthyordinaryaircraft.
        w.aircraft["a"] = replace(w.aircraft["a"], emergency=replace(w.aircraft["a"].emergency, eligible=False))
        e.action("fixture_initial_emergency_eligibility", {"aircraft": "a", "eligible": False})
        r = submit(e, w, (p,))
        e.check(r.reason == "NO_EMERGENCY_ELIGIBILITY" and w.metrics()["cost_emergency"] == 0,
                "healthyouter cannotbuyshortcut", "NO_EMERGENCY_ELIGIBILITY", r.reason)
    elif case == "F04":
        w = world(e)
        q, _ = w._reference("outer", 0, 0)
        hold = make_stable_plan("reference", "outer", 0, end_time=10)
        qhold = integrate_energy(hold, 0, 10).soc_used
        soc = .25+q+qhold-1e-6
        e.action("predictivetrigger_fixture", {"ring": "outer", "qE": q, "qHold10": qhold, "initial_soc": soc})
        e.check(soc > .35, "predictivetrigger precedesabsoluteSOC", ">.35", soc)
        seed(e, w, soc=soc); advance(e, w, 0)
        e.check(w.aircraft["a"].emergency.eligible and w.aircraft["a"].emergency.declared,
                "secondtriggeractuallydeclares beforeabsolute", True, w.aircraft["a"].to_dict())
    elif case == "F05":
        w = world(e)
        rate = evaluate_power(slot_state("inner", 0, 0).v, slot_state("inner", 0, 0).a).battery_w/(85*252*3600)
        thresholds = []
        for i in range(12):
            t = i*.2; q, _ = w._reference("inner", 0, t)
            hp = make_stable_plan("refhold", "inner", 0, start_time=t, end_time=t+10)
            qh = integrate_energy(hp, t, t+10).soc_used
            thresholds.append((t, max(.35, .25+q+qh)+rate*t))
        soc = thresholds[-1][1]-1e-9
        e.action("offheartbeat_threshold_fixture", {"ring": "inner", "initial_soc": soc, "thresholds": thresholds})
        if any(v >= soc for _, v in thresholds[:-1]):
            raise FixtureBlocked("fixedinner slot0 has earlierpredictivetrigger than2.2s for thiscomputedSOC; explicit2.2 fixture unresolved")
        seed(e, w, ring="inner", soc=soc)
        advance(e, w, 2.2)
        triggers = [x["time"] for x in w.events if x["type"] == "emergency_triggered"]
        e.check(triggers and abs(triggers[0]-2.2) <= .001 and triggers[0] < 5,
                "monitortriggers at2.2 executionstep before5heartbeat", 2.2, triggers)
    elif case == "F06":
        w = world(e); seed(e, w, soc=.36)
        p = feasible(e, generate_transfer, "a", "outer", 0, 0, 0)
        e.check(submit(e, w, (p,)).accepted, "committedtransfer starts", True, "accepted")
        advance(e, w, .2)
        a = w.aircraft["a"]
        e.check(a.emergency.pending and w.active["a"] is p and a.emergency.start_time is None,
                "duringtransfer trigger ispending currentplan unchanged", True, a.to_dict())
        advance(e, w, p.end_time)
        a = w.aircraft["a"]
        e.check(a.emergency.start_time == p.end_time and a.emergency.start_ring == "middle",
                "rescuestarts onlyat committedtargetring", [p.end_time, "middle"], [a.emergency.start_time, a.emergency.start_ring])
    elif case == "F07":
        w = world(e); seed(e, w, "normal", "inner", 9, .8); seed(e, w, "rescue", "inner", 0, .30, emergency=True)
        normal = feasible(e, generate_final, "normal", "inner", 9, 0, 1.)
        e.check(submit(e, w, (normal,)).accepted and "normal" in w.soft, "unstartedordinarypad reservation", True, "soft")
        rescue = feasible(e, generate_final, "rescue", "inner", 0, 0, 0, emergency=True)
        r = submit(e, w, (rescue,))
        e.check(r.accepted and "normal" not in w.soft and "rescue" in w.active,
                "emergencyatomically preemptsordinary softpad", True, asdict(r))
        #Astarted final remainscommitted, withnosilentdeletion.
        r = w.cancel("rescue"); e.action("cancel", {"aircraft": "rescue"})
        e.check(r.reason == "COMMITTED_MODIFICATION", "startedfinal neverpreempted", "COMMITTED_MODIFICATION", r.reason)
    elif case in ("F08", "F09"):
        w = world(e)
        for j in range(6): service(e, w, "pad"+str(j), j, .35 if case == "F08" else 2.)
        seed(e, w, "rescue", "inner", 0, .30 if case == "F08" else .023)
        advance(e, w, 250 if case == "F08" else 0)
        a = w.aircraft["rescue"]
        if case == "F08":
            e.check(abs(a.emergency.start_time) < .001 and a.touchdown_time is not None and not w.failed,
                    "depart while service busy; pad clears before protected arrival", 0., a.to_dict())
        else:
            proofs = [x for x in w.events if x["type"] == "PAD_DEADLINE_MISSED"]
            e.check(not a.emergency.deadline_missed and not proofs and a.id in w.aircraft and a.status == "HOLD_INNER" and not w.failed,
                    "no departure-time deadline proof under arrival-based pad reservations", "EMERGENCY_WAITING without global infeasibility claim", [a.to_dict(), proofs])
            advance(e, w, 30)
            e.check(w.failed and w.aircraft["rescue"].status == "ENERGY_FAILURE", "deadline followeduntilrealphysicalfailure", "ENERGY_FAILURE", w.metrics())
    elif case in ("F10", "F13"):
        w = world(e)
        for j in range(5): service(e, w, "pad"+str(j), j, 100.)
        seed(e, w, "a", "inner", 0, .30); seed(e, w, "b", "inner", 9, .30)
        advance(e, w, 0)
        active = [p for p in w.active.values() if p.kind == "FINAL_EMERGENCY"]
        waiting = [a for a in w.aircraft.values() if a.status == "HOLD_INNER"]
        e.check(len(active) == 1 and len(waiting) == 1 and waiting[0].emergency.declared,
                "onefreepad admits onlysafesubset retainingwaitingaircraft", [1, 1], [len(active), len(waiting)])
        if case == "F13":
            a = waiting[0]
            p = feasible(e, generate_final, a.id, "inner", a.slot, 5, 0)
            r = submit(e, w, (p,))
            e.check(r.reason == "EMERGENCY_DECLARED", "latcheddeclaration deniesordinaryaction", "EMERGENCY_DECLARED", r.reason)
    elif case == "F14":
        w = world(e)
        for j in range(6): service(e, w, "pad"+str(j), j, .35)
        seed(e, w, "normal", "inner", 0, .35)
        advance(e, w, 0)
        e.check(w.active["normal"].kind == "FINAL_NORMAL" and not w.aircraft["normal"].emergency.declared and w.metrics()["cost_emergency"] == 0,
                "legalnormalfinal haspriority beforedeclaration", "FINAL_NORMAL/noemergencycost", w.aircraft["normal"].to_dict())


def run_time(case, e):
    if case == "G01":
        #AhighSOC stablefixture runs100s; actualemergency monitor remainson.
        w = world(e); seed(e, w, soc=.9); advance(e, w, 100)
        expected = list(range(0, 101, 5))
        e.check(len(w.decision_times) == len(expected) and all(abs(a-b) <= .001 for a, b in zip(w.decision_times, expected)),
                "stableworld decisions only5sheartbeat", expected, w.decision_times)
        e.check(len(w.records) >= 501, "physicsoutputs do notcausefullscheduler", ">=501 samples", len(w.records))
    elif case in ("G02", "G03"):
        reqs = (ArrivalRequest("x", .125, 0, .75, "short"), ArrivalRequest("y", .125, 1, .75, "long"))
        w = world(e, manifest=ArrivalManifest(reqs)); service(e, w, "pad", remaining=.125)
        advance(e, w, .4)
        e.check(.125 in w.decision_times and w.decision_times.count(.125) == 1 and .125 in [s["time"] for s in w.records],
                "offgrid events exact andmerged", .125, w.decision_times)
        if case == "G03":
            run_energy("E12", e)
    elif case == "G04":
        results = []
        plan = feasible(e, generate_final, "a", "inner", 0, 0, 0)
        for dt in (.2, .1, .05):
            w = world(e, physics_dt=dt); seed(e, w, ring="inner", soc=.8)
            ground = service(e, w, "other_ground", 3, remaining=300)
            other = TrajectoryPlan.from_dict(w.trajectories[ground.trajectory_hash])
            e.check(submit(e, w, (plan,)).accepted, "identicalcommittedscript accepted", True, dt)
            advance(e, w, plan.end_time+60)
            results.append({"dt": dt, "energy_j": w.aircraft["a"].energy_j, "events": w.events,
                            "soc": w.aircraft["a"].soc, "p": w.aircraft["a"].p, "outcomes": w.metrics(),
                            "minimum": certified_minimum_distance(plan, other, 0, plan.end_time)})
        e.observe("dt_sweep", results)
        energies = [r["energy_j"] for r in results]
        e.check(max(energies)/min(energies)-1 <= .005 and all(event_equivalent(r["events"], results[0]["events"]) for r in results),
                "samecommittedscript energy/event convergence", "relativeenergy<=.005 identicalevents", results)
        minima = [r["minimum"] for r in results]
        e.check(all(m["status"] == "CERTIFIED" for m in minima) and max(m["upper_bound_m"] for m in minima)-min(m["lower_bound_m"] for m in minima) <= .05,
                "continuousminimum independentoutputdt", "<=.05m", minima)
    elif case == "G05":
        results = []
        manifest = ArrivalManifest((ArrivalRequest("r", 0, 0, .85, "short"),))
        for period in (1., 5., 10.):
            w = world(e, manifest=manifest, scheduler_period=period)
            advance(e, w, 300, DeterministicDriver())
            results.append({"period_s": period, "decisions": len(w.decision_times), "cpu_seconds": w.cpu,
                            "metrics": w.metrics(), "touchdown": w.aircraft["r"].touchdown_time})
        e.observe("scheduler_sensitivity", results)
        e.check(all(r["metrics"]["conservation"] for r in results) and len({r["decisions"] for r in results}) == 3,
                "actualperiod sensitivityreports countsCPUoutcomes", "conservation/3distinctdecisioncounts", results)
    elif case == "G06":
        for latency in (.2, 1.2, 5.2):
            w, p = single_final(e)
            e.check(submit(e, w, (p,)).accepted, "initialplanaccepted", True, latency)
            initial = w.aircraft["a"].soc
            r = submit(e, w, (p,), simulated_latency=latency)
            e.check(r.reason == "EXPIRED_TIME" and abs(w.time-latency) < 1e-9 and w.aircraft["a"].soc < initial and w.active["a"] is p,
                    "latencyadvancesenergy refusingstalestart", "EXPIRED_TIME/ongoingphysicalmotion", [asdict(r), w.time, w.aircraft["a"].soc])
    elif case == "G07":
        w = world(e); service(e, w, "long_service", remaining=300)
        advance(e, w, 240)
        e.check(w.aircraft["long_service"].status == "LANDED_SERVICE" and w.pads.active(0, 240),
                "resourcepreserved pastpreview240", "occupieduntil300", w.aircraft["long_service"].to_dict())
        advance(e, w, 300)
        e.check(w.aircraft["long_service"].status == "CLEARED", "actualserviceend releases", "CLEARED", w.aircraft["long_service"].status)
    elif case == "G08":
        m = ArrivalManifest.generate(11, 60)
        w = world(e, manifest=m); snapshot = w.snapshot()
        e.check(not snapshot.requests and not hasattr(snapshot, "manifest") and len(w.manifest.requests) > 0,
                "snapshot excludesunbornmanifest", "emptyrequests/manifestnotexposed", snapshot.to_dict())
        try: snapshot.metrics["offered"] = 999
        except TypeError: immutable = True
        else: immutable = False
        e.check(immutable, "scheduler snapshot immutable", True, immutable)
        callbacks = []
        def inspect_callback(s, port):
            inaccessible = []
            for name in ("manifest", "records", "events"):
                try: getattr(port, name)
                except AttributeError: inaccessible.append(name)
            metrics = port.metrics()
            readonly = False
            try: port.aircraft["fake"] = None
            except TypeError: readonly = True
            callbacks.append({"hidden_attributes": inaccessible, "metric_fields": sorted(metrics),
                              "snapshot_metric_fields": sorted(s.metrics), "readonly_current_aircraft": readonly})
        advance(e, w, 0, inspect_callback)
        observed = callbacks[0]
        e.check(observed["hidden_attributes"] == ["manifest", "records", "events"] and observed["readonly_current_aircraft"]
                and all(k not in observed["metric_fields"] and k not in observed["snapshot_metric_fields"] for k in ("planned_manifest", "unreleased")),
                "actualscheduler port hidesfuture demand andaudit records", "readonlycurrentonly/nofuturecounts", observed)
    elif case == "G09":
        import json
        from hashlib import sha256
        from .diagnostics import run_benchmarks
        supplied = os.environ.get("TERMINAL_DIAGNOSTIC_REPORT")
        if supplied:
            source = Path(supplied).resolve()
            document = json.loads(source.read_text(encoding="utf8"))
            e.action("consume_completed_diagnostic_report", {"path": str(source), "sha256": sha256(source.read_bytes()).hexdigest()})
            e.check(document.get("schema") == "FT-RS-S1-1.0-diagnostics" and document.get("protocol") == "FROZEN_FORMAL_MATRIX",
                    "formaldiagnostic source required", "FROZEN_FORMAL_MATRIX", document.get("protocol"))
            jobs = [r for r in document["jobs"] if r.get("job", {}).get("kind") == "benchmark"]
            from .qualification import task_coverage
            expected = [f"benchmark_n{n}_{mode}_r{repeat}" for n in (10, 30, 60, 83) for mode in ("stable", "inflight_attempt") for repeat in range(5)]
            result = {"jobs": jobs, "coverage": task_coverage(expected, jobs), "source_path": str(source),
                      "source_sha256": sha256(source.read_bytes()).hexdigest()}
            for r in jobs:
                for field in ("config_hash", "code_sha256", "code_commit"):
                    e.check(r.get("provenance", {}).get(field) == e.provenance[field],
                            "benchmarksource matching"+field, e.provenance[field], r.get("provenance", {}).get(field))
                from .replay import ReplayBundle
                archive = ReplayBundle.validate(r["replay_path"])
                archived_metrics = json.loads((archive.path/"metrics.json").read_text(encoding="utf8"))
                e.check(archive.index["code_commit"] == e.provenance["code_commit"] and archive.index["config_hash"] == e.provenance["config_hash"]
                        and archived_metrics == r["metrics"] and r["wall_seconds"] > 0 and r["cpu_seconds"] > 0,
                        "actualverified benchmarkarchive andmeasuredtiming", "matchingarchive+positiveCPU/wall", {
                            "id": r["id"], "archive": str(archive.path), "wall_seconds": r["wall_seconds"], "cpu_seconds": r["cpu_seconds"]})
        else:
            result = run_benchmarks(e.output/"benchmark", counts=(10, 30, 60, 83), repeats=5, duration=.2)
        e.observe("benchmark", result)
        blocked = [r for r in result["jobs"] if r["status"] == "BLOCKED"]
        if blocked:
            raise FixtureBlocked("additionalinflightbenchmark fixture unavailablefor: "+",".join(r["id"] for r in blocked))
        e.check(len(result["jobs"]) == 40 and not result["coverage"]["missing"] and not result["coverage"]["duplicates"] and not result["coverage"]["failures"],
                "actual40benchmark measurementsincludingaccepted inflight;noabsolute timePASS", 40, result["coverage"])
        for r in result["jobs"]:
            n = r["job"]["count"]; inflight = r["job"]["mode"] == "inflight_attempt"
            e.check(r["status"] == "COMPLETED" and r["actual_airborne_count"] == n+int(inflight)
                    and (not inflight or r["inflight_admitted"]),
                    "actualbenchmark stablecount pluscertifiedinflight", n+int(inflight),
                    {"id": r["id"], "actual_airborne_count": r["actual_airborne_count"], "inflight_admitted": r.get("inflight_admitted")})


def execute_scenario(name, e):
    if name == "full_lifecycle":
        requests = tuple(ArrivalRequest(f"gate{j}", j*300., j, .9, ("short", "standard", "long", "short")[j]) for j in range(4))
        w = world(e, manifest=ArrivalManifest(requests, seed=e.seed))
        advance(e, w, 1500, DeterministicDriver())
        for request in requests:
            sequence = [x["type"] for x in w.events if x.get("aircraft_id") == request.id and x["type"] in ("admitted", "inbound_completed", "transfer_completed", "touchdown", "pad_service_completed")]
            e.check(sequence == ["admitted", "inbound_completed", "transfer_completed", "transfer_completed", "touchdown", "pad_service_completed"] and w.aircraft[request.id].status == "CLEARED",
                    "actualfullnormal lifecycle gate"+str(request.gate_id), "admit/inbound/2transfers/touchdown/clear", sequence)
        e.check(w.metrics()["normal_landed"] == 4 and w.metrics()["conservation"], "allfourgate demand retained inaccounting", 4, w.metrics())
    elif name == "all_ring_emergencies": run_emergency("F01", e)
    elif name == "six_pads":
        run_resources("C11", e)
        w = e.worlds[-1]
        end = max(p.end_time for p in w.active.values())
        advance(e, w, end)
        e.check(w.metrics()["physically_landed"] == 6 and not w.failed, "all6 actualsafe touchdowns", 6, w.metrics())
        advance(e, w, end+240)
        e.check(all(a.status == "CLEARED" for a in w.aircraft.values()), "all3serviceclasses finishandclear6pads", "CLEARED", [a.status for a in w.aircraft.values()])
    elif name == "busy_release": run_emergency("F08", e)
    elif name == "busy_failure": run_emergency("F09", e)
    elif name == "threshold_crossing":
        from .qualification import analytic_record
        from .replay import ReplayBundle
        run_safety("D01", e); run_safety("D02", e)
        a = analytic(e, "cross_a"); b = analytic(e, "cross_b", z=(80.999, -40., 400.))
        minimum = certified_minimum_distance(a, b, 0, .2)
        record = analytic_record([a, b], duration=.2)
        record["metrics"] = {"fixture_only": True, "continuous_minimum": minimum}
        record["events"] = [{"time": 0., "type": "diagnostic_crossing_rejected", "aircraft_id": "cross_b", "details": minimum}]
        record["rejections"] = [{"time": 0., "aircraft_id": "cross_b", "reason": "HARD_SEPARATION",
            "details": {"diagnostic_only_not_executed": True, "trajectory_points": [b.state(float(t)).p for t in np.linspace(0, .2, 101)], "continuous_minimum": minimum}}]
        bundle = ReplayBundle.write(e.output/"threshold_replay", record=record, code_commit=e.provenance["code_commit"],
            command=e.provenance["command"], seed=e.seed, assertions=e.assertions)
        e.replay_paths.append(str(bundle.path))
    elif name == "three_laps_controls":
        from .qualification import analytic_record
        from .replay import ReplayBundle
        duration = 3*2*math.pi/.035
        plans = [make_stable_plan(ring+str(j), ring, j, end_time=duration+1) for ring, _, _, n in RINGS for j in (0, n//2)]
        e.plans.extend(plans)
        record = analytic_record(plans, duration=duration)
        record["events"] = [{"time": 2*math.pi/.035, "type": "analytic_lap", "details": {"lap": 1}},
                            {"time": 4*math.pi/.035, "type": "analytic_lap", "details": {"lap": 2}}]
        bundle = ReplayBundle.write(e.output/"three_laps_replay", record=record, code_commit=e.provenance["code_commit"],
            command=e.provenance["command"], seed=e.seed, assertions=e.assertions)
        e.replay_paths.append(str(bundle.path))
        e.action("analytic_stable_run", {"duration_s": duration, "sample_dt_s": .2, "count": 6, "telemetry": "actualconstantpowerformula"})
        e.check(abs(record["states"][-1]["time"]-duration) < 1e-9 and len(record["states"]) >= 2693,
                "3full analytic laps at0.2s", duration, [record["states"][-1]["time"], len(record["states"])])
    else: raise ValueError("unknownscenario: "+name)


def run_visual(case, e):
    from .replay import ReplayBundle
    from ...visualization.rotating_slots_replay import ReplayController, RotatingSlotsReplayViewer
    from ...visualization.pyvista_viewer import Terminal3DViewer
    scenario = {"V01": "threshold_crossing", "V02": "three_laps_controls", "V03": "three_laps_controls",
        "V04": "full_lifecycle", "V05": "threshold_crossing", "V06": "six_pads", "V07": "all_ring_emergencies",
        "V08": "busy_failure", "V09": "threshold_crossing", "V10": "full_lifecycle", "V11": "three_laps_controls"}[case]
    execute_scenario(scenario, e)
    if case == "V08": execute_scenario("busy_release", e)
    paths = e.save_replays()
    instructions = {"required_views": ["top_down", "side", "3d"], "researcher_visual_acceptance": "PENDING",
        "commands": [f'python -m terminal_marl.scripts.open_stage1_replay "{p}" --screenshots "{p}/screenshots"' for p in paths],
        "observation_required": "Inspect the namedfixture inall3views, controls andoverlays. Recordobserver/time/screenshots andexplicitacceptance separately."}
    from .qualification import write_json
    write_json(e.output/"viewer_instructions.json", instructions)
    e.observe("viewer_instructions", instructions)
    e.check(issubclass(RotatingSlotsReplayViewer, Terminal3DViewer) and (ROOT/"terminal_marl/scripts/open_stage1_replay.py").exists(),
            "existinglocalviewer entry andadapter", "Terminal3DViewer subclass/existingentry", RotatingSlotsReplayViewer.__module__)
    for path in paths:
        b = ReplayBundle.load(path); ctl = ReplayController(b)
        frame = ctl.seek(b.time_bounds[0]+.137 if b.time_bounds[1] > .137 else b.time_bounds[0])
        if case == "V02":
            e.check([(r.radius, r.height) for r in frame.rings] == [(600., 110.), (450., 70.), (300., 30.)] and len(frame.landing_zones) == 6,
                    "adapterpreserves realgeometry and6pads", [[600,110],[450,70],[300,30]], [(r.radius,r.height) for r in frame.rings])
        elif case == "V03":
            before = ctl.slots_at(); ctl.seek(b.time_bounds[1]); ctl.seek(.137)
            after = ctl.slots_at()
            e.check([r["id"] for r in before] == [r["id"] for r in after] and len(after) == 83, "stable slotIDs after3laps andseek", 83, len(after))
            for a in ctl.aircraft_at():
                expected = ctl._plans[a["trajectory_hash"]].state(.137)
                e.check(math.dist(a["p"], expected.p) < 1e-9, "seek preservesabsolutephase", expected.p, a["p"])
        elif case == "V05":
            rejected = ctl.overlays()["diagnostic_rejections_not_executed"]
            e.check(rejected and rejected[0]["details"]["diagnostic_only_not_executed"], "rejection overlayseparate fromexecution", "diagnosticonly", rejected)
        elif case == "V06":
            e.check(len(frame.landing_zones) == 6, "all6pad renderresources", 6, len(frame.landing_zones))
            touchdown = min(x["time"] for x in b.data["events"] if x["type"] == "touchdown")
            ctl.seek(touchdown+.001); e.check(ctl.snapshot().service_count == 6, "touchdown serviceentities rendered", 6, ctl.snapshot().service_count)
            ctl.seek(b.time_bounds[1]); e.check(ctl.snapshot().completed_count == 6, "allpad serviceclearance rendered", 6, ctl.snapshot().completed_count)
        elif case == "V08":
            ctl.seek(b.time_bounds[1]); events = ctl.overlays()["events"]
            e.check(any(x["type"] in ("ENERGY_FAILURE", "energy_failure", "PAD_DEADLINE_MISSED", "touchdown") for x in events),
                    "resourcewait/failure/outcome events retained", "deadline/failure orrescue", [x["type"] for x in events])
        elif case == "V09":
            ctl.seek(.05); pair = ctl.overlays()["minimum_current_3d_pair"]
            e.check(pair is not None and pair[0] < 80, "diagnosticpair displayedusing real3D atinteriorsample", "79.999m", pair)
            e.check(b.data["metrics"]["continuous_minimum"]["status"] == "CERTIFIED", "continuousminimum evidence included", "CERTIFIED", b.data["metrics"]["continuous_minimum"])
        elif case == "V10":
            identity = next(iter(ctl.uid_by_id))
            curves = b.curves(identity)
            e.check(curves and all(set(r) == {"time", "speed", "acceleration", "jerk", "battery_power", "SOC"} for r in curves), "5curves readloggedfields", "completefieldset", len(curves))
            e.check(any(r["battery_power"] and r["battery_power"] > 0 for r in curves), "powercurve hasactualtelemetry", ">0W", [r["battery_power"] for r in curves[:3]])
        elif case == "V11":
            hashes = dict(b.index["files"]); ctl.select(next(iter(ctl.uid_by_id)), follow=True)
            ctl.seek(.137); expected = ctl.snapshot()
            for speed in ctl.speeds:
                ctl.set_speed(speed); ctl.step(); ctl.jump_event(); ctl.seek(.137)
                e.check(ctl.snapshot() == expected, "controls preserve queriedstate "+str(speed), expected.time, ctl.snapshot().time)
            e.check(dict(ReplayBundle.load(path).index["files"]) == hashes and ctl.follow,
                    "controls nevermodifyarchive", True, True)
        e.check(b.data["visual_acceptance"]["researcher_visual_acceptance"] == "PENDING", "automatedadaptercannotapprovehumaninspection", "PENDING", b.index["human_acceptance"])


def execute_case(case, e):
    if case in ("A05", "E07", "E14"):
        try:
            from .extra_fixtures import execute_case as extra_case
        except ImportError:
            pass
        else:
            if extra_case(case, e): return
    handlers = {"A": run_geometry, "B": run_motion, "C": run_resources, "D": run_safety,
                "E": run_energy, "F": run_emergency, "G": run_time, "V": run_visual}
    handlers[case[0]](case, e)


def event_equivalent(left, right):
    """Eventsemantics withthe contract1ms eventtime tolerance."""
    if len(left) != len(right): return False
    for a, b in zip(left, right):
        if a["type"] != b["type"] or a.get("aircraft_id") != b.get("aircraft_id"): return False
        if abs(a["time"]-b["time"]) > .001: return False
        da, db = a.get("details", {}), b.get("details", {})
        if da.keys() != db.keys(): return False
        for key in da:
            x, y = da[key], db[key]
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                if not math.isclose(x, y, rel_tol=1e-8, abs_tol=.001): return False
            elif x != y: return False
    return True
