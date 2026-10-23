"""Read-only 0--600s medium comparison facts; never starts or changes a run.

Stable holding is reconstructed from executed maneuver boundaries, not planned
reservations. Battery watts times actual seconds gives joules. The executed
quadrature ledger excludes STABLE portions by design; its nonstable charges are
reported separately, and state cumulative holding counters check the integral.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from terminal_marl.core.rotating_slots.config import load_config
from terminal_marl.core.rotating_slots.energy import stable_power
from terminal_marl.core.rotating_slots.rescue_rules import normal_energy_lower_bound
from terminal_marl.core.rotating_slots.stage2_campaign import Stage2Codec

RINGS = ("outer", "middle", "inner")


def read_json(path):
    return json.loads(Path(path).read_bytes())


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def holding_segments(fixture, events, trajectories, end):
    """Return [aircraft,ring,start,end] only for executed stable waiting."""
    current = {a["id"]: (a.get("ring"), 0.) for a in fixture
               if a["status"] in ("HOLD_OUTER", "HOLD_MIDDLE", "HOLD_INNER")}
    active, segments, inner_entries, finals, touchdowns = {}, [], {}, {}, {}
    for aid, (ring, _) in current.items():
        if ring == "inner": inner_entries[aid] = 0.
    transfers = Counter()
    for event in sorted(events, key=lambda row: row["time"]):
        t, aid, kind = event["time"], event.get("aircraft_id"), event["type"]
        if t > end: continue
        if kind == "admitted":
            active[aid] = trajectories[event["details"]["trajectory_hash"]]
        elif kind == "maneuver_started":
            if aid in current:
                ring, start = current.pop(aid)
                segments.append((aid, ring, start, t))
            plan = trajectories[event["details"]["trajectory_hash"]]
            active[aid] = plan
            if plan["kind"].startswith("FINAL"):
                finals.setdefault(aid, dict(time=t, kind=plan["kind"]))
        elif kind in ("inbound_completed", "transfer_completed"):
            plan = active.pop(aid, None)
            if plan is None: raise ValueError("completion lacks actual maneuver_started: "+str(event))
            ring = plan["target_ring"]
            current[aid] = (ring, t)
            if kind == "transfer_completed":
                transfers[plan["source_ring"]+"->"+ring] += 1
            if ring == "inner": inner_entries.setdefault(aid, t)
        elif kind == "touchdown":
            touchdowns[aid] = dict(time=t, emergency=event["details"].get("emergency", False))
            active.pop(aid, None)
        elif kind in ("aircraft_energy_failure", "failure_isolated", "energy_failure", "aircraft_failed", "rerouted") and aid in current:
            ring, start = current.pop(aid)
            segments.append((aid, ring, start, t))
    for aid, (ring, start) in current.items(): segments.append((aid, ring, start, end))
    latencies = []
    for aid, entry in sorted(inner_entries.items()):
        final, touchdown = finals.get(aid), touchdowns.get(aid)
        latencies.append(dict(aircraft_id=aid, inner_entry_s=entry,
            final_kind=None if final is None else final["kind"],
            inner_to_final_s=None if final is None else final["time"]-entry,
            inner_to_final_censored=final is None,
            inner_to_touchdown_s=None if touchdown is None else touchdown["time"]-entry,
            inner_to_touchdown_censored=touchdown is None,
            observed_since_entry_s=end-entry))
    return segments, dict(transfers), latencies


def state_holding_integral(states, end):
    totals = dict.fromkeys(RINGS, 0.)
    for left, right in zip(states, states[1:]):
        dt = max(0., min(end, right["time"])-left["time"])
        if dt <= 0: continue
        for aircraft in left["aircraft"]:
            if aircraft["status"] == "HOLD_"+str(aircraft.get("ring", "")).upper():
                totals[aircraft["ring"]] += dt
    return totals


def queues(states, events, segments, end, lower):
    """Stable ring backlog; event declaration and sampled warning/SOC/status.

    Warning onsets/declaration timestamps come from events even when v7 omitted
    an exact monitor state frame. Warning clearing has no event and remains
    sampled. Only stable waiting portions enter the queue;
    a reserved future start remains holding until maneuver_started.
    """
    times = [row["time"] for row in states]
    snapshots = [{a["id"]: a for a in row["aircraft"]} for row in states]
    warning, declared = defaultdict(list), {}
    for event in events:
        if event["type"] == "rescue_warning": warning[event["aircraft_id"]].append(event["time"])
        elif event["type"] == "emergency_declared": declared.setdefault(event["aircraft_id"], event["time"])
    by_id = defaultdict(list)
    for aid, ring, start, stop in segments: by_id[aid].append((ring, start, stop))
    def counts(t):
        index=max(0, bisect_right(times, t+1e-9)-1)
        observed = snapshots[index]
        result = {r: dict(warning=0, declared_waiting=0, lower_bound_shortage=0,
                          normal_status_unknown=0, stable_backlog=0) for r in RINGS}
        for aid, portions in by_id.items():
            ring = next((ring for ring, start, stop in portions if start <= t < stop), None)
            a = observed.get(aid)
            if t == end and a and a["status"] in ("HOLD_OUTER", "HOLD_MIDDLE", "HOLD_INNER"):
                ring = a["ring"]
            if ring is None: continue
            row = result[ring]; row["stable_backlog"] += 1
            observed_warning=bool(a and a.get("emergency",{}).get("warning",False))
            # An omitted event-time frame must not delay a recorded onset;
            # later actual states remain authoritative for warning clearing.
            onset_since_frame=any(times[index]-1e-9 <= onset <= t for onset in warning.get(aid,()))
            row["warning"] += int(observed_warning or onset_since_frame)
            row["declared_waiting"] += int(declared.get(aid, math.inf) <= t)
            if a is not None:
                row["lower_bound_shortage"] += int(a["soc"]-.25-lower[ring] <= 0.)
                row["normal_status_unknown"] += int(a.get("emergency", {}).get("normal_status", "UNKNOWN") == "UNKNOWN")
        return result
    boundaries = sorted({0., end, *[t for t in times if 0 <= t <= end],
        *[t for t in [*(t for rows in warning.values() for t in rows),*declared.values()] if 0 <= t <= end],
        *[t for _, _, start, stop in segments for t in (start, stop)]})
    integral = {r: {key: 0. for key in counts(0.)[r]} for r in RINGS}
    for start, stop in zip(boundaries, boundaries[1:]):
        for ring, row in counts(start).items():
            for key, value in row.items(): integral[ring][key] += value*(stop-start)
    return dict(definition="stable HOLD ring backlog only; maneuvers excluded; lower_bound_shortage = soc - .25 - normal_energy_lower_bound(ring) <= 0",
        every_60s=[dict(time_s=t, rings=counts(t)) for t in range(0, int(end)+1, 60)],
        terminal=counts(end), aircraft_seconds=integral, warning_integral_status="APPROXIMATE_NO_CLEAR_EVENTS",
        integral_precision=dict(warning="approximate clearing from left state flags, exact event onsets; no warning_clear event exists",
            declared_waiting="exact event flags and stable boundaries",
            lower_bound_shortage="approximate left state SOC, not a threshold crossing interpolation",
            normal_status_unknown="approximate left state status", max_state_gap_s=max((b-a for a,b in zip(times,times[1:])),default=0.)))


def analyze_method(root, method, end):
    directory = root/method; bundle = directory/f"replay_{int(end):04d}"
    summary = read_json(directory/f"summary_{int(end):04d}.json")
    states = read_jsonl(bundle/"states.jsonl")
    events = read_jsonl(bundle/"events.jsonl")
    trajectories = read_json(bundle/"trajectories.json")
    config = load_config(root/"config.yaml")
    if (summary.get("time")!=end or not states or abs(states[0]["time"])>1e-8
        or abs(states[-1]["time"]-end)>1e-8):
        raise ValueError("summary/states do not cover the requested complete 0--end window")
    segments, transfers, latencies = holding_segments(read_json(root/"fixture.json")["aircraft"], events, trajectories, end)
    seconds = {r: math.fsum(stop-start for _, ring,start,stop in segments if ring == r) for r in RINGS}
    watts = {r: stable_power(r, config).battery_w for r in RINGS}
    joules = {r: seconds[r]*watts[r] for r in RINGS}
    state_integral = state_holding_integral(states, end)
    state_counter = math.fsum(a["holding_time"] for a in states[-1]["aircraft"])
    metrics = summary["metrics"]
    decisions=read_jsonl(directory/"decisions.jsonl")
    timing_keys=sorted({key for row in decisions for key in row.get("timings",{})})
    timing_breakdown={key:dict(wall_s=math.fsum(row.get("timings",{}).get(key,(0.,0.))[0] for row in decisions),
        cpu_s=math.fsum(row.get("timings",{}).get(key,(0.,0.))[1] for row in decisions)) for key in timing_keys}
    # Decode only the executed numerical ledger, never restore/open a run sink.
    checkpoint = read_json(directory/f"checkpoint_{int(end):04d}.json")
    world_items = dict(checkpoint["payload"]["world"]["items"])
    ledger = Stage2Codec().decode(world_items["_targeted_energy"])
    charged = math.fsum(row["charged_j"] for rows in ledger["executed"].values() for row in rows)
    tolerance = 1e-5
    checks = dict(event_vs_summary_holding_s=math.fsum(seconds.values())-metrics["holding_time"],
        state_counter_vs_summary_holding_s=state_counter-metrics["holding_time"],
        left_state_vs_event_by_ring_s={r: state_integral[r]-seconds[r] for r in RINGS},
        holding_energy_le_total=math.fsum(joules.values()) <= metrics["energy_j"]+tolerance)
    checks["holding_plus_nonstable_ledger_minus_total_j"]=math.fsum((math.fsum(joules.values()),charged,-metrics["energy_j"]))
    checks["holding_plus_nonstable_ledger_matches_total"]=abs(checks["holding_plus_nonstable_ledger_minus_total_j"])<1e-3
    checks["holding_integrals_match"] = (abs(checks["event_vs_summary_holding_s"]) < tolerance
        and abs(checks["state_counter_vs_summary_holding_s"]) < tolerance
        and all(abs(v)<tolerance for v in checks["left_state_vs_event_by_ring_s"].values()))
    return dict(time_s=end, metrics={k: metrics[k] for k in ("L_normal", "L_emergency", "L_success", "F_energy_total",
        "N_airborne_pending", "N_upstream_pending", "N_offered", "energy_j", "energy_kwh", "holding_time")},
        holding=dict(seconds_by_ring=seconds,battery_w_by_ring=watts,joules_by_ring=joules,
            total_j=math.fsum(joules.values()),kwh_by_ring={r:joules[r]/3600000. for r in RINGS},
            integration="exact actual maneuver events; future reservations retain holding until actual start",
            executed_nonstable_ledger_charge_j=charged,
            ledger_note="world._record_energy_charge excludes STABLE/GROUND; nonstable numerical ledger is not total terminal energy"),
        checks=checks, queues=queues(states,events,segments,end,{r:normal_energy_lower_bound(config,r) for r in RINGS}),
        transfer_completed_by_route=transfers, inner_latencies=latencies,
        timing={k:summary.get(k) for k in ("normal_timing","inner_algorithm_timing","dispatch_timing",
            "dispatch_wall_sum_s","executor_non_dispatch_wall_s","episode_compute_wall_s")},
        planner_timing_by_stage=timing_breakdown,
        planner_downstream_decisions=[{key:row[key] for key in ("time","downstream","downstream_integer_key",
            "virtual_pad_assignment","virtual_queue_order") if key in row} for row in decisions if "downstream" in row],
        planner_observation_fields=sorted({key for row in decisions for key in row
            if any(word in key for word in ("downstream","supply","drain","spill","validation"))}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--end", type=float, default=600.)
    args=parser.parse_args()
    root=args.root.resolve(); output=args.output.resolve()
    # Output must not overwrite any frozen run artifact.
    if output.is_relative_to(root): raise ValueError("analysis output must be outside the read-only input root")
    campaign=read_json(root/"campaign.json")
    result=dict(schema="stage2_v8_medium_analysis_v1",root=str(root),time_s=args.end,
        planning_profile=campaign.get("planning_profile","progress_tail_v7"),
        count_scope="cumulative realized 0--end metrics; never sum cumulative summaries",
        methods={m:analyze_method(root,m,args.end) for m in ("energy_slack","rh_cp_sat")})
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(dict(output=str(output),checks={m:r["checks"] for m,r in result["methods"].items()}),indent=2))


if __name__ == "__main__": main()
