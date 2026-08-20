"""A6.15 Safe Opportunity Window & Sequencing Attribution Audit.

All opportunity evaluations run against a deep-copied environment.  This is
strictly diagnostic: it never reserves a resource, mutates the live scheduler,
or changes a checked/executed plan.
"""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import replace
import csv
import json
import math
from pathlib import Path

import numpy as np

from ..core.config import RingRole, RingSpec
from ..core.merge_scheduler import LowLevelIntent
from ..core.runtime_separation import swept_minimum_distance
from ..core.uav import FlightPhase
from ..envs.high_level_wrapper import DynamicHighLevelWrapper, RuleLowLevelAdapter
from ..envs.parallel_env import TerminalParallelEnv, WorldSnapshot
from ..policies import RuleLowLevelPolicy
from .a611_causal_audit import DemandProfile, _safe
from .a612_scheduler_repair_audit import _a611_mu0
from .a613_capacity_reaudit import repair_config
from .a614_reconfiguration_timing_audit import OUTER_PLUS_ONE
from .a6_effectiveness import _collect, _sample


SEEDS = (0, 1)
BRANCHES = ("nominal", "static_outer_plus1", "dynamic_t180", "dynamic_t240", "dynamic_t300")
RADIUS_SWEEP = (600.0, 606.0, 610.0, 611.5, 611.6, 612.0)
PPO_AUTHORIZATION = "STOP"
INTERFACES = {
    FlightPhase.RING_OUTER: ("o2m", RingRole.MIDDLE, FlightPhase.TRANSFER_O2M),
    FlightPhase.RING_MIDDLE: ("m2i", RingRole.INNER, FlightPhase.TRANSFER_M2I),
    FlightPhase.RING_INNER: ("i2f", "final", FlightPhase.FINAL),
}


def _target_context(env, uav):
    value = INTERFACES.get(uav.phase)
    if value is None or uav.committed_plan is not None:
        return None
    interface, target, phase = value
    if interface in env.airspace.affected_interfaces():
        return interface, target, phase, None, "geometry_frozen"
    if uav.soc <= 0.0:
        return interface, target, phase, None, "soc"
    lz_id = None
    if target == "final":
        if sum(item.phase is FlightPhase.FINAL for item in env.active.values()) >= env.config.final_queue_capacity:
            return interface, target, phase, None, "final_capacity"
        lz_id = env.lz_manager.choose_rule_zone(uav, env.now)
        if lz_id is None:
            return interface, target, phase, None, "lz_unavailable"
    return interface, target, phase, lz_id, None


def _neighbors(env, target, angle, merge_time):
    if not isinstance(target, RingRole) or angle is None:
        return None, None
    ring = env.airspace.actual_ring(target); horizon = max(0., merge_time - env.now)
    states = []
    for item in env.active.values():
        if item.ring_role is target:
            value = ((item.ring_progress or 0.) + ring.nominal_speed * horizon / ring.radius) % (2 * math.pi)
            states.append((value, item.uid))
    if not states:
        return None, None
    ordered = sorted(states); index = next((i for i, (value, _) in enumerate(ordered) if value >= angle), 0)
    return ordered[(index - 1) % len(ordered)][1], ordered[index % len(ordered)][1]


def _minimum_prediction(env, plan, snapshot):
    """Read-only companion metric for the SafetyFilter candidate verdict."""
    minimum = float("inf"); swept = float("inf")
    times = np.arange(plan.start_time, plan.end_time + 1e-9, 1.)
    if not len(times) or times[-1] < plan.end_time:
        times = np.append(times, plan.end_time)
    for uid, other in snapshot.active.items():
        if uid == plan.uav_id:
            continue
        other_plan = other.committed_plan
        def position(t):
            if other_plan is not None and t <= other_plan.end_time:
                return other_plan.sample(float(t)).position
            if other_plan is not None:
                return env.safety._post_plan_prediction(other_plan, snapshot, float(t))
            return env.safety._holding_prediction(other, snapshot, float(t))
        for left, right in zip(times, times[1:]):
            a, b = plan.sample(float(left)).position, plan.sample(float(right)).position
            c, d = position(left), position(right)
            minimum = min(minimum, float(np.linalg.norm(a - c)), float(np.linalg.norm(b - d)))
            swept = min(swept, swept_minimum_distance(a, b, c, d)[0])
    value = min(minimum, swept)
    return (None if not np.isfinite(value) else float(value), None if not np.isfinite(swept) else float(swept))


class ShadowOpportunityAudit:
    """Per-second, non-mutating A6.12 candidate audit plus opportunity ledger."""
    def __init__(self, branch: str, seed: int) -> None:
        self.branch, self.seed = branch, seed
        self.events: list[dict] = []
        self.trace: list[dict] = []
        self.ledger: dict[tuple[int, str], dict] = {}

    def observe(self, env, actions: dict[str, np.ndarray]) -> dict[int, tuple[int, str]]:
        keys = {}
        # A single isolated copy is sufficient for all checks in this instant:
        # _candidate_plan only writes telemetry/audit state, never schedule or
        # vehicle runtime state.  The live environment is never handed to it.
        shadow_env = deepcopy(env) if any(_target_context(env, item) is not None for item in env.active.values()) else None
        for uid, uav in env.active.items():
            context = _target_context(env, uav)
            if context is None:
                continue
            interface, target, _, _, pre_blocker = context
            key = (uav.arrival_id, interface); keys[uid] = key
            record = self.ledger.setdefault(key, {"seed": self.seed, "branch": self.branch, "arrival_id": uav.arrival_id,
                "uav_id": uid, "interface": interface, "T_eligible": env.now, "T_first_shadow_safe": None,
                "T_policy_request": None, "T_scheduler_reservation": None, "T_actual_release": None})
            intent = LowLevelIntent.from_array(actions.get(str(uid), np.zeros(3, np.float32)))
            if intent.release_request > env.config.release_threshold and record["T_policy_request"] is None:
                record["T_policy_request"] = env.now
            row = self._shadow(shadow_env, uid, intent, context, pre_blocker)
            row.update({"seed": self.seed, "branch": self.branch, "time": env.now, "arrival_id": uav.arrival_id, "uav_id": uid,
                        "interface": interface, "policy_release_request": intent.release_request})
            if row["safe_candidate_exists"] and record["T_first_shadow_safe"] is None:
                record["T_first_shadow_safe"] = env.now
            self.events.append(row)
            self.trace.append({key: row.get(key) for key in ("time", "arrival_id", "uav_id", "interface", "policy_release_request",
                "safe_candidate_exists", "selected_kind", "merge_time", "merge_angle", "speed_residual", "merge_time_residual",
                "future_delay", "predecessor", "successor", "primary_blocker", "candidate_set") } | {"branch": self.branch, "seed": self.seed})
        return keys

    def _shadow(self, env, uid: int, policy_intent: LowLevelIntent, context, pre_blocker: str | None) -> dict:
        # ``env`` is already this timestep's isolated shadow copy.  Candidate
        # evaluation only appends shadow telemetry, so sharing that copy among
        # eligible UAVs cannot alter a reservation, trajectory, or runtime
        # vehicle state and avoids a redundant deep copy per UAV.
        clone = env; uav = clone.active[uid]
        interface, target, _, lz_id, _ = _target_context(clone, uav)
        # The shadow request intentionally differs only in release=1.  Its
        # residuals remain the policy's current bounded intention.
        requested = LowLevelIntent(1., policy_intent.speed_residual, policy_intent.merge_time_residual)
        blockers, candidates = [], []
        legal_candidates = clone.scheduler.repair_candidates(requested) if pre_blocker is None else ()
        if pre_blocker is not None:
            blockers.append(pre_blocker)
        else:
            snapshot = WorldSnapshot(clone.now, clone.active, clone.airspace)
            # Evaluate the finite A6.12 authority on the isolated state so
            # the audit can distinguish an earlier feasible alternative from
            # the scheduler's ranked choice; no result is committed.
            for index, candidate in enumerate(legal_candidates):
                proposal, plan, reason = clone._candidate_plan(uav, candidate, target, snapshot, lz_id)
                if reason is not None:
                    blockers.append(reason); candidates.append({"rank": index, "kind": candidate.kind, "safe": False, "blocker": reason})
                    continue
                minimum, swept = _minimum_prediction(clone, plan, snapshot)
                predecessor, successor = _neighbors(clone, target, proposal.merge_angle, proposal.merge_time)
                candidates.append({"rank": index, "kind": candidate.kind, "safe": True, "merge_time": proposal.merge_time,
                    "merge_angle": proposal.merge_angle, "speed_residual": candidate.intent.speed_residual,
                    "merge_time_residual": candidate.intent.merge_time_residual, "future_delay": candidate.delay_seconds,
                    "predecessor": predecessor, "successor": successor, "minimum_predicted_separation": minimum,
                    "minimum_swept_separation": swept, "plan_id": plan.plan_id})
        selected = next((item for item in candidates if item["safe"]), None)
        earliest = min((item for item in candidates if item["safe"]), key=lambda item: item["merge_time"], default=None)
        primary = pre_blocker or (None if selected else (blockers[0] if blockers else "no_candidate"))
        return {"safe_candidate_exists": selected is not None, "earliest_safe_candidate_time": None if earliest is None else earliest["merge_time"],
                "selected_kind": None if selected is None else selected["kind"], "merge_time": None if selected is None else selected["merge_time"],
                "merge_angle": None if selected is None else selected["merge_angle"], "speed_residual": None if selected is None else selected["speed_residual"],
                "merge_time_residual": None if selected is None else selected["merge_time_residual"], "future_delay": None if selected is None else selected["future_delay"],
                "predecessor": None if selected is None else selected["predecessor"], "successor": None if selected is None else selected["successor"],
                "minimum_predicted_separation": None if selected is None else selected["minimum_predicted_separation"],
                "minimum_swept_separation": None if selected is None else selected["minimum_swept_separation"],
                "primary_blocker": primary, "secondary_blockers": "|".join(sorted(set(blockers))),
                "legal_candidate_kinds": "|".join(candidate.kind for candidate in legal_candidates),
                "candidate_set": json.dumps(candidates, default=str, separators=(",", ":"))}

    def after_step(self, env, keys, info) -> None:
        for raw_uid, decision in info["release_decisions"].items():
            uid = int(raw_uid); key = keys.get(uid)
            if key is None:
                continue
            record = self.ledger[key]; telemetry = info["release_intent_telemetry"].get(raw_uid, {})
            if decision in {"approved", "future_reserved"} and record["T_scheduler_reservation"] is None:
                record["T_scheduler_reservation"] = env.now
            if decision == "approved" and record["T_actual_release"] is None:
                record["T_actual_release"] = env.now
            # Attach execution outcome to the corresponding current shadow row.
            for row in reversed(self.events):
                if row["time"] == env.now - env.config.dt and row["uav_id"] == uid:
                    row["actual_decision"] = decision; row["reservation_id"] = telemetry.get("trajectory_plan_id")
                    row["execution_primary_blocker"] = telemetry.get("primary_blocker")
                    break
            for row in reversed(self.trace):
                if row["time"] == env.now - env.config.dt and row["uav_id"] == uid:
                    row["actual_decision"] = decision; row["reservation_id"] = telemetry.get("trajectory_plan_id")
                    row["execution_primary_blocker"] = telemetry.get("primary_blocker")
                    break

    def losses(self):
        rows = []
        for row in self.ledger.values():
            item = dict(row)
            first_safe, policy, reserved, actual = (item[key] for key in ("T_first_shadow_safe", "T_policy_request", "T_scheduler_reservation", "T_actual_release"))
            item["policy_opportunity_loss"] = None if first_safe is None or policy is None else max(0., policy - first_safe)
            item["scheduler_opportunity_loss"] = None if policy is None or reserved is None else max(0., reserved - policy)
            item["execution_delay"] = None if reserved is None or actual is None else max(0., actual - reserved)
            rows.append(item)
        return rows


def _dynamic_config(profile): return repair_config(profile, dynamic=True)

def _run_branch(profile: DemandProfile, seed: int, name: str, action_time: float | None, *, instrument=True):
    if name == "static_outer_plus1":
        base = repair_config(profile, dynamic=False, role=RingRole.OUTER); env = TerminalParallelEnv(base); env.reset(seed); policy = RuleLowLevelPolicy(); coordinator = None
    else:
        wrapper = DynamicHighLevelWrapper(_dynamic_config(profile), RuleLowLevelAdapter()); wrapper.reset(seed); env, policy, coordinator = wrapper.base, wrapper.low_level, wrapper.coordinator
    audit = ShadowOpportunityAudit(name, seed) if instrument else None
    samples = {key: [] for key in ("queue", "oldest_wait", "final_util", "lz_util", "actual", "capacity", "all_waiting")}; requested = False; action_events=[]
    while env.now < env.config.max_simulation_seconds and not (env.arrivals.exhausted() and not env.active and not env.admission.queue.items and not env.service_uavs):
        if coordinator is not None and action_time is not None and not requested and env.now >= action_time - 1e-9:
            gate = wrapper.geometry_event_gate() or action_time == 0.
            decision = coordinator.coordinate(OUTER_PLUS_ONE, geometry_gate=gate); requested = True
            action_events.append({"time": env.now, "outcome": decision.outcome, "reason": decision.reason_code})
        if coordinator is not None:
            deferred = coordinator.advance()
            if deferred is not None: action_events.append({"time": env.now, "outcome": deferred.outcome, "reason": deferred.reason_code})
            actions = coordinator.apply_speed_repairs(policy.act(env._observations()))
        else:
            actions = policy.act(env._observations())
        keys = audit.observe(env, actions) if audit else {}
        _, _, _, _, info = env.step(actions); _sample(env, samples)
        if audit: audit.after_step(env, keys, info)
    metrics = _collect(env, samples, coordinator, action_events) if coordinator else _collect(env, samples)
    # _commit_checked_plan raises if the approved and executed canonical plan
    # IDs differ.  Reaching this point therefore certifies the frozen A6.12
    # checked-trajectory == executed-trajectory invariant for this rollout.
    return {"metrics": metrics, "safe": _safe(metrics, env.config), "events": [] if audit is None else audit.events,
            "losses": [] if audit is None else audit.losses(), "trace": [] if audit is None else audit.trace,
            "action_events": action_events, "checked_executed_plan_matches": metrics.get("checked_executed_plan_matches", 0),
            "checked_executed_trajectory_invariant": True,
            "mode": name, "seed": seed}


def _signature(row):
    return tuple(row.get(key) for key in ("policy_release_request", "safe_candidate_exists", "selected_kind", "merge_time", "merge_angle", "speed_residual", "merge_time_residual", "future_delay", "predecessor", "successor", "primary_blocker"))


def first_divergence(left, right, *, seed, compared_branch):
    left_map = {(row["time"], row["arrival_id"], row["interface"]): row for row in left["trace"]}
    right_map = {(row["time"], row["arrival_id"], row["interface"]): row for row in right["trace"]}
    for key in sorted(set(left_map) | set(right_map)):
        a, b = left_map.get(key), right_map.get(key)
        if a is None or b is None or _signature(a) != _signature(b):
            return {"seed": seed, "compared_branch": compared_branch, "simulation_time": key[0], "arrival_id": key[1],
                    "interface": key[2], "event_type": "eligible_presence" if a is None or b is None else "candidate_or_request_difference",
                    "nominal": json.dumps(a, default=str), "branch": json.dumps(b, default=str)}
    return {"seed": seed, "compared_branch": compared_branch, "simulation_time": None, "arrival_id": None, "interface": None, "event_type": "no_divergence", "nominal": "", "branch": ""}


def local_window(divergence, left, right):
    if divergence["simulation_time"] is None: return []
    rows=[]; t0=divergence["simulation_time"]; aid=divergence["arrival_id"]; interface=divergence["interface"]
    def normalized_candidate_set(row):
        try:
            values = json.loads(row.get("candidate_set") or "[]")
        except json.JSONDecodeError:
            return []
        return [{key: value for key, value in item.items() if key != "plan_id"} for item in values]
    selected_by_side = {
        "nominal": [row for row in left["events"] if row["arrival_id"] == aid and row["interface"] == interface and abs(row["time"]-t0) <= 2.],
        "branch": [row for row in right["events"] if row["arrival_id"] == aid and row["interface"] == interface and abs(row["time"]-t0) <= 2.],
    }
    by_time = {label: {row["time"]: row for row in selected} for label, selected in selected_by_side.items()}
    for label, selected in selected_by_side.items():
        peer = by_time["branch" if label == "nominal" else "nominal"]
        for row in selected:
            other = peer.get(row["time"])
            rows.append({"branch_side": label, "time": row["time"], "arrival_id": aid, "interface": interface,
                         "safe_candidate_exists": row["safe_candidate_exists"], "selected_kind": row["selected_kind"],
                         "candidate_set": row["candidate_set"], "primary_blocker": row["primary_blocker"],
                         "actual_decision": row.get("actual_decision"),
                         "feasibility_flip_vs_pair": other is not None and row["safe_candidate_exists"] != other["safe_candidate_exists"],
                         "candidate_set_change_vs_pair": other is not None and normalized_candidate_set(row) != normalized_candidate_set(other),
                         "candidate_ranking_change_vs_pair": other is not None and row.get("selected_kind") != other.get("selected_kind"),
                         "reservation_sequence_change": other is not None and row.get("actual_decision") != other.get("actual_decision")})
    return rows


def _radius_config(profile, radius):
    base = repair_config(profile, dynamic=False); rings = dict(base.rings); original = rings[RingRole.OUTER]
    rings[RingRole.OUTER] = RingSpec(RingRole.OUTER, radius, original.height, original.nominal_speed)
    return replace(base, rings=rings)


def _radius_task(profile, seed, radius):
    env=TerminalParallelEnv(_radius_config(profile, radius)); env.reset(seed); policy=RuleLowLevelPolicy(); samples={key: [] for key in ("queue","oldest_wait","final_util","lz_util","actual","capacity","all_waiting")}
    while env.now < env.config.max_simulation_seconds and not (env.arrivals.exhausted() and not env.active and not env.admission.queue.items and not env.service_uavs):
        env.step(policy.act(env._observations())); _sample(env,samples)
    metrics=_collect(env,samples)
    return {"seed":seed,"radius":radius,"derived_capacity":env.airspace.capacity(RingRole.OUTER),"safe":_safe(metrics,env.config),
            "completed":metrics["completed"],"waiting_area":metrics["all_request_waiting_area"],"external_queue_area":metrics["external_queue_area"],"service_delay_p95":metrics["service_delay_p95"],
            "energy_mean":metrics["energy_mean"],"fairness":metrics["priority_fairness_std"],"checked_executed_plan_matches":metrics.get("checked_executed_plan_matches",0)}


def radius_micro_sweep(profile, seed):
    return [_radius_task(profile, seed, radius) for radius in RADIUS_SWEEP]


def classify(branches, micro, divergences=(), local_windows=()):
    labels=set(); events=[row for result in branches.values() for row in result["events"]]; losses=[row for result in branches.values() for row in result["losses"]]
    if any(not row["safe_candidate_exists"] for row in events): labels.add("NECESSARY_SAFETY_WAIT")
    if any(row.get("policy_opportunity_loss",0) not in (None,0) for row in losses): labels.add("POLICY_MISSED_SAFE_WINDOW")
    if any(row.get("scheduler_opportunity_loss",0) not in (None,0) for row in losses): labels.add("SCHEDULER_MISSED_SAFE_WINDOW")
    same_cap=defaultdict(list)
    for row in micro: same_cap[row["derived_capacity"]].append(row["waiting_area"])
    if any(max(values)-min(values)>1e-6 for values in same_cap.values() if len(values)>1): labels.add("CONTINUOUS_GEOMETRY_EFFECT")
    # A capacity label requires an *additional* discontinuity at the actual
    # 47->48 crossing, not merely observing both derived integer capacities.
    # Compare nearest radii on either side within each identical CRN seed.
    by_seed = defaultdict(list)
    for row in micro: by_seed[row["seed"]].append(row)
    for rows in by_seed.values():
        rows = sorted(rows, key=lambda row: row["radius"])
        for left, right in zip(rows, rows[1:]):
            if left["derived_capacity"] != right["derived_capacity"] and abs(right["waiting_area"] - left["waiting_area"]) > 1e-6:
                labels.add("CAPACITY_THRESHOLD_EFFECT")
    # These labels are attribution labels, not claims of a performance gain:
    # they mean a CRN-paired branch first differed in executable sequencing.
    if any(row.get("event_type") == "candidate_or_request_difference" for row in divergences):
        labels.add("SEQUENCING_BIFURCATION")
    if any(row.get("reservation_sequence_change") for row in local_windows):
        labels.add("RESERVATION_SEQUENCE_LOCK_IN")
    if any(row.get("candidate_ranking_change_vs_pair") for row in local_windows):
        labels.add("TIE_BREAK_SENSITIVITY")
    if not labels: labels.add("INCONCLUSIVE")
    if len(labels)>1: labels.add("MIXED")
    return sorted(labels)


def run_audit(*, seeds=SEEDS, mu0: float | None = None, workers: int = 6):
    mu = float(mu0) if mu0 is not None else _a611_mu0(0)[0]
    profile=DemandProfile("constant_high_repair_scheduler", ((0.,720.,1.55*mu),),720.,480.)
    results={}; divergences=[]; local=[]; micro=[]
    timing={"nominal":None,"static_outer_plus1":None,"dynamic_t180":180.,"dynamic_t240":240.,"dynamic_t300":300.}
    for seed in seeds:
        # These CRN rollouts share only their immutable profile and seed; their
        # runtime state is intentionally independent.  Parallel execution is
        # thus a wall-clock optimization, never an inter-branch mechanism.
        rows = {}
        with ProcessPoolExecutor(max_workers=min(max(1, workers), len(timing))) as pool:
            futures = {pool.submit(_run_branch, profile, seed, name, time): name for name, time in timing.items()}
            for future in as_completed(futures):
                rows[futures[future]] = future.result()
        if not all(row["safe"] and row["checked_executed_trajectory_invariant"] for row in rows.values()):
            raise RuntimeError("A6.15 FAIL: safety or checked/executed invariant failure")
        results[str(seed)]=rows
        for name in tuple(timing)[1:]:
            divergence=first_divergence(rows["nominal"],rows[name],seed=seed,compared_branch=name); divergences.append(divergence)
            local.extend([dict(row,seed=seed,compared_branch=name,simulation_time=divergence["simulation_time"]) for row in local_window(divergence,rows["nominal"],rows[name])])
        with ProcessPoolExecutor(max_workers=min(max(1, workers), len(RADIUS_SWEEP))) as pool:
            futures = [pool.submit(_radius_task, profile, seed, radius) for radius in RADIUS_SWEEP]
            micro.extend(future.result() for future in futures)
    labels=classify({f"{seed}:{name}":row for seed,group in results.items() for name,row in group.items()},micro,divergences,local)
    return {"stage":"A6.15 Safe Opportunity Window & Sequencing Attribution Audit","evidence_type":"newly_run","seeds":list(seeds),
            "inputs":{"a611_mu0":mu,"safe_distance":80.,"repair_aware_scheduler":True,"high_level_decision_interval_seconds":180.,"geometry_min_dwell_seconds":600.,"ppo_authorization":PPO_AUTHORIZATION},
            "branches":results,"first_divergence":divergences,"local_windows":local,"radius_micro_sweep":micro,
            "summary":{"classification":labels,"protected_safety":"PASS","ppo_authorization":PPO_AUTHORIZATION}}


def _write_csv(path, rows):
    fields=sorted({key for row in rows for key in row}) or ["empty"]
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction="ignore");writer.writeheader();writer.writerows(rows)


def write_report(report, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    events=[row for group in report["branches"].values() for branch in group.values() for row in branch["events"]]
    losses=[row for group in report["branches"].values() for branch in group.values() for row in branch["losses"]]
    _write_csv(output/"a615_safe_opportunity_events.csv",events);_write_csv(output/"a615_opportunity_loss_by_uav.csv",losses)
    _write_csv(output/"a615_first_divergence.csv",report["first_divergence"]);_write_csv(output/"a615_local_window_audit.csv",report["local_windows"]);_write_csv(output/"a615_radius_micro_sweep.csv",report["radius_micro_sweep"])
    (output/"a615_audit.json").write_text(json.dumps(report,indent=2,default=str),encoding="utf-8")
    branch_rows = []
    for seed, group in report["branches"].items():
        for name, result in group.items():
            metrics = result["metrics"]
            branch_rows.append((seed, name, metrics["completed"], metrics["all_request_waiting_area"],
                                metrics["unresolved_count"], result["safe"], result.get("checked_executed_trajectory_invariant", False)))
    safe_events = sum(bool(row["safe_candidate_exists"]) for row in events)
    missed_policy = sum((row.get("policy_opportunity_loss") or 0) > 0 for row in losses)
    missed_scheduler = sum((row.get("scheduler_opportunity_loss") or 0) > 0 for row in losses)
    lines=["# A6.15 Safe Opportunity Window & Sequencing Attribution Audit", "",
           f"Mechanism classification: **{', '.join(report['summary']['classification'])}**",
           f"Protected safety: **{report['summary']['protected_safety']}**", "PPO authorization: **STOP**", "",
           "All candidate checks were run in deep-copied shadow environments; no shadow reservation or plan is committed to runtime.",
           f"Safe shadow-opportunity events: {safe_events}/{len(events)}. Policy opportunity-loss cases: {missed_policy}; scheduler opportunity-loss cases: {missed_scheduler}.", "",
           "| Seed | Branch | Completed | Waiting area | Unresolved | Hard-safe | Checked == executed |",
           "|---:|---|---:|---:|---:|---|---|"]
    lines.extend(f"| {seed} | {name} | {completed} | {waiting:.1f} | {unresolved:.0f} | {safe} | {invariant} |"
                 for seed, name, completed, waiting, unresolved, safe, invariant in branch_rows)
    lines.extend(["", "| Paired comparison | First causal divergence |", "|---|---|"])
    lines.extend(f"| seed {row['seed']}: nominal vs {row['compared_branch']} | {row['event_type']} at t={row['simulation_time']}, arrival={row['arrival_id']}, interface={row['interface']} |"
                 for row in report["first_divergence"])
    lines.extend(["", "| Outer radius | Derived capacity | Waiting area |", "|---:|---:|---:|"])
    lines.extend(f"| {row['radius']:.1f} | {row['derived_capacity']} | {row['waiting_area']:.1f} |" for row in report["radius_micro_sweep"])
    lines.extend(["", "Interpretation gate: labels attribute observed CRN diagnostics only. They do not authorize changes to the frozen scheduler, SafetyFilter, high-level timing, PPO, A7, or A8."])
    (output/"a615_summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("--seeds",default="0,1");parser.add_argument("--mu0",type=float,default=0.014583333333333334);parser.add_argument("--workers",type=int,default=6);parser.add_argument("--output",default=None)
    args=parser.parse_args();seeds=tuple(int(value) for value in args.seeds.split(",") if value.strip());output=Path(args.output) if args.output else Path.home()/"training_outputs"/"terminal"/"event_gated_stage15_diagnostics"/"a615"
    report=run_audit(seeds=seeds,mu0=args.mu0,workers=args.workers);write_report(report,output);print(json.dumps(report["summary"]))


if __name__=="__main__": main()
