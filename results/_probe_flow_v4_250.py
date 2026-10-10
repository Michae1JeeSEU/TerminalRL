"""Bounded t=250 flow-state and final-candidate evidence probe."""
import json
import math
from pathlib import Path

import numpy as np

from terminal_marl.core.flow_trajectories import FlowTrajectoryGenerator
from terminal_marl.core.safety_filter import SafetyFilter
from terminal_marl.core.runtime_traffic_safety import SafetyInvariantError
from terminal_marl.core.config import RingRole
from terminal_marl.envs.parallel_env import TerminalParallelEnv, WorldSnapshot
from terminal_marl.training.diagnose_landing_flow import _build_diagnostic_config

OUT = Path("results/flow_v4_state250.json")
NOMINAL_REASONS = {"predicted_holder_reachable_tube", "predicted_holder_separation",
                   "predicted_post_plan_holder_reachable_tube", "nominal_holder_separation",
                   "nominal_holder_swept_separation", "nominal_post_plan_holder_separation",
                   "nominal_post_plan_holder_swept_separation"}


def state(u):
    p = u.committed_plan
    return {"uid": u.uid, "phase": u.phase.value,
            "ring_role": None if u.ring_role is None else u.ring_role.value,
            "ring_progress": u.ring_progress, "soc": u.soc,
            "energy_remaining": u.energy_remaining, "assigned_lz": u.assigned_lz,
            "committed": p is not None,
            "committed_end": None if p is None else p.end_time,
            "target_phase": None if p is None else p.metadata.get("target_phase")}


def pads(env):
    return [{"id": z.lz_id, "status": z.status.value, "reserved_by": z.reserved_by,
             "occupied_by": z.occupied_by, "reservation_start": z.reservation_start,
             "expected_release_at": z.expected_release_at,
             "service_remaining": z.service_remaining} for z in env.lz_manager.zones]


def release_rows(env):
    out = {}
    for uid, row in env.last_release_telemetry.items():
        reasons = row.get("flow_candidate_rejection_reasons") or row.get("secondary_constraints") or []
        candidates = row.get("flow_candidate_durations") or []
        decisions = row.get("decision_history") or []
        out[str(uid)] = {"uid": uid, "decision": decisions[-1] if decisions else None,
                         "attempt_count": row.get("attempt_count"),
                         "primary_blocker": row.get("primary_blocker"),
                         "last_reasons": reasons[-6:], "last_flow_candidates": candidates[-6:]}
    return out


def inner_gaps(env):
    entries = []
    ring = env.airspace.actual_ring(RingRole.INNER)
    for u in env.active.values():
        if u.ring_role is not RingRole.INNER or u.committed_plan is not None:
            continue
        angle = math.atan2(float(u.position[1]), float(u.position[0])) % (2 * math.pi)
        entries.append((angle, u.uid))
    entries.sort()
    out = []
    for i, (angle, uid) in enumerate(entries):
        next_angle, next_uid = entries[(i + 1) % len(entries)]
        delta = (next_angle - angle) % (2 * math.pi)
        out.append({"uid": uid, "angle_rad": angle, "next_uid": next_uid,
                    "gap_angle_rad": delta, "gap_arc_m": ring.radius * delta,
                    "gap_chord_m": 2 * ring.radius * math.sin(delta / 2)})
    return out


def holder_worst(plan, env, snapshot, checker):
    samples = np.arange(plan.start_time, plan.end_time + 1e-9, 1.0)
    if samples[-1] < plan.end_time:
        samples = np.append(samples, plan.end_time)
    candidate = np.asarray(plan.sample_positions(samples), dtype=float)
    witnesses = []
    for holder in env.active.values():
        if holder.uid == plan.uav_id or holder.ring_role is None or holder.committed_plan is not None:
            continue
        holder_pos = np.asarray(checker._holder_positions(holder, snapshot, samples, 0.0), dtype=float)
        point, swept = checker._track_distances(candidate, holder_pos)
        pi = int(np.argmin(point)); si = int(np.argmin(swept)) if len(swept) else None
        witnesses.append({"uid": holder.uid, "point_m": float(point[pi]),
                          "point_time": float(samples[pi]),
                          "swept_m": None if si is None else float(swept[si]),
                          "swept_time": None if si is None else float(samples[si])})
    if not witnesses:
        return {"holder_count": 0}
    point = min(witnesses, key=lambda x: x["point_m"])
    swept = min((x for x in witnesses if x["swept_m"] is not None),
                key=lambda x: x["swept_m"], default=None)
    return {"holder_count": len(witnesses),
            "worst_point": {"uid": point["uid"], "min_m": point["point_m"], "time": point["point_time"]},
            "worst_swept": None if swept is None else {"uid": swept["uid"], "min_m": swept["swept_m"], "time": swept["swept_time"]}}


def probe_candidates(env):
    config = env.config
    flow = FlowTrajectoryGenerator(config, env.generator)
    snapshot = WorldSnapshot(now=env.now, active=env.active, airspace=env.airspace)
    committed = {uid: u.committed_plan for uid, u in env.active.items() if u.committed_plan is not None}
    checker = SafetyFilter(config)
    rows, skipped = [], {}
    for uav in list(env.active.values()):
        if uav.ring_role is not RingRole.INNER or uav.committed_plan is not None:
            continue
        available = tuple(env.lz_manager.rank_available_zones(uav, env.now))
        for lz_id in available:
            for duration in (20.0, 28.0, 36.0, 48.0, 60.0):
                try:
                    plan = flow.final_approach(uav, env.lz_manager.zones[lz_id], env.airspace,
                                               env.now, duration, emergency=False)
                except ValueError as exc:
                    skipped[str(exc)] = skipped.get(str(exc), 0) + 1
                    continue
                except RuntimeError as exc:
                    rows.append({"uid": uav.uid, "lz_id": lz_id, "duration": duration,
                                 "dynamics": "failed", "error": str(exc)})
                    continue
                result = checker.filter_plan(plan, committed, snapshot)
                reason = result.reason or "approved"
                row = {"uid": uav.uid, "lz_id": lz_id, "duration": duration,
                       "dynamics": "passed", "safety_reason": reason,
                       "approved": bool(result.approved)}
                if reason in NOMINAL_REASONS:
                    row["holder_worst"] = holder_worst(plan, env, snapshot, checker)
                rows.append(row)
    return rows, skipped


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    config = _build_diagnostic_config(True)
    env = TerminalParallelEnv(config)
    env.reset(seed=841301)
    actions = {}
    steps, stop_reason = 0, None
    while steps < 250:
        actions = {str(uid): np.asarray([1.0, 0.0, 0.0], dtype=np.float32) for uid in env.active}
        try:
            env.step(actions)
        except SafetyInvariantError as exc:
            stop_reason = f"safety_error: {exc}"
            break
        steps += 1
    if stop_reason is None:
        stop_reason = "max_steps" if steps >= 250 else "stopped"
    states = [state(u) for u in sorted(env.active.values(), key=lambda x: x.uid)]
    candidates, skipped = probe_candidates(env)
    payload = {"seed": 841301, "requested_steps": 250, "steps": steps,
               "sim_time": env.now, "stop_reason": stop_reason,
               "active_count": len(states), "active_final_uids": [u["uid"] for u in states if u["phase"] == "final"],
               "active": states, "pads": pads(env), "latest_release_rows": release_rows(env),
               "inner_ring_gaps": inner_gaps(env), "candidate_probes": candidates,
               "geometry_invalid_skipped": skipped,
               "config": {"rule_flow_coordination_enabled": config.rule_flow_coordination_enabled,
                          "max_speed_residual_mps": config.max_speed_residual_mps,
                          "safety_sampling_dt": config.safety_sampling_dt,
                          "physics_dt": config.physics_dt}}
    raw = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    if len(raw.encode("utf-8")) > 150000:
        raise RuntimeError(f"probe output exceeds 150KB: {len(raw.encode('utf-8'))}")
    OUT.write_text(raw + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "bytes": OUT.stat().st_size,
                      "steps": steps, "sim_time": env.now,
                      "candidate_count": len(candidates)}, sort_keys=True))


if __name__ == "__main__":
    main()
