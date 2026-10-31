"""Action-local invalidation and world-issued, standby-first latency recovery.

The filter proves separation from outward bounds of actual continuations. An
inconclusive bound is never an exemption. Certification uses the unchanged
world transaction, energy and continuous separation validators.
"""
from bisect import bisect_right
from dataclasses import replace
import math
import time

from .config import freeze
from .models import ValidationResult
from .trajectory import Interval
from .world import AIRBORNE, _planning_context


def initialize_packet(packet):
    from .planning_latency import _direct_dependencies, _reservations
    views = {}
    source, timing = packet["source"], packet["timing"]
    for plan in packet["plans"]:
        aid = plan.aircraft_id
        resources = {r.resource for r in (*source.slots, *packet["slots"])
                     if r.aircraft_id == aid}
        pads = {r.resource for r in (*source.pads, *packet["pads"])
                if r.aircraft_id == aid}
        ids, hashes = _direct_dependencies(source, (plan,), {aid: packet["starts"][aid]},
            resources, pads, packet["dependency_table"], timing.effective_time)
        ids = set(ids)
        ids.update(p.aircraft_id for p in packet["plans"] if p.trajectory_hash in hashes)
        views[plan.trajectory_hash] = dict(direct_ids=tuple(sorted(ids)), direct_hashes=hashes,
            plans=(plan,), resources=tuple(resources), pad_resources=tuple(pads),
            resource_key=_reservations(source.slots, resources, timing.effective_time),
            pad_key=_reservations(source.pads, pads, timing.effective_time),
            trajectory_certificates={key: row for key, row in packet["pair_certificates"].items() if aid in key})
    packet["action_guards"] = views
    packet["local_traffic_source"] = source
    packet["local_traffic_tables"] = dict(packet["commitment_tables"])
    packet.setdefault("original_plans", tuple(packet["plans"]))
    packet.setdefault("recovery_log", [])


def _global_guard(world, packet):
    from .planning_latency import _direct_guard
    empty = dict(packet, direct_ids=(), direct_hashes=(), plans=(), resources=(),
                 pad_resources=(), resource_key=(), pad_key=())
    return _direct_guard(world, empty)


def _delta(world, packet):
    from .planning_latency import _physical_key, _emergency_key, _motion
    source, tables = packet["local_traffic_source"], packet["local_traffic_tables"]
    changed = set()
    for aid, table in tables.items():
        actual = world.aircraft.get(aid)
        row = table[max(0, bisect_right(tuple(r[0] for r in table), world.time)-1)]
        _, key, active, soft = row
        if (actual is None or _physical_key(actual) != key
                or world.active.get(aid) != active or world.soft.get(aid) != soft
                or _emergency_key(actual) != _emergency_key(source.aircraft[aid])):
            changed.add(aid)
        if actual is not None and abs(world.time-packet["timing"].effective_time) <= 1.e-8:
            expected = _motion(source, source.aircraft[aid], world.time)[0]
            if any(math.dist(getattr(actual, k), getattr(expected, k)) > tolerance
                   for k, tolerance in (("p", 1.e-6), ("v", 1.e-7), ("a", 1.e-7))):
                changed.add(aid)
    changed.update(aid for aid, a in world.aircraft.items() if aid not in tables
                   and (a.status in AIRBORNE or a.status == "LANDED_SERVICE"))
    return changed


def _polar_box(traffic, start, end):
    """Radial/height hull over every actual segment, future switch and hold."""
    cuts = sorted({start, end, *(p.start_time for p in traffic.plans
                               if start < p.start_time < end)})
    rows = []
    for lo, hi in zip(cuts, cuts[1:]):
        plan = traffic._at((lo+hi)/2)
        for segment in plan.intervals(lo, hi):
            a, b = max(lo, segment.start_time), min(hi, segment.end_time)
            if b <= a: continue
            polar = segment.polar_bounds(a, b)
            rows.append((polar[0][0].pair(), polar[2][0].pair()))
    if not rows: return None
    return tuple((min(row[k][0] for row in rows), max(row[k][1] for row in rows))
                 for k in range(2))


def envelope_safe(first, second, start, end, config):
    """Sufficient interval proof only; no ring labels or sampled radii."""
    from .safety import _norm_lower
    if end <= start: return True
    pending = [(start, end, 0)]
    while pending:
        lo, hi, depth = pending.pop()
        one, two = _polar_box(first, lo, hi), _polar_box(second, lo, hi)
        if one is None or two is None: return False
        # Horizontal distance is at least the absolute radial difference.
        delta = tuple((Interval(*x)-Interval(*y)).pair() for x, y in zip(one, two))
        if _norm_lower(delta) >= config.safety.hard_separation_m: continue
        delta = tuple((Interval(*x)-Interval(*y)).pair()
                      for x, y in zip(first.bounds(lo, hi), second.bounds(lo, hi)))
        if _norm_lower(delta) >= config.safety.hard_separation_m: continue
        # Every accepted slab has a continuous outward enclosure. This bounded
        # refinement never accepts based on values at sample points.
        if depth >= 6: return False
        mid = (lo+hi)/2
        pending.extend(((lo, mid, depth+1), (mid, hi, depth+1)))
    return True


def _analyze(world, packet, *, certify=False, deadline=math.inf):
    from .planning_latency import _direct_guard, _traffic_signature, validate_separation, _motion
    global_result = _global_guard(world, packet)
    if not global_result.accepted: return global_result, None
    changed = _delta(world, packet)
    affected, reasons, conflicts, invalidation_details = set(), {}, [], {}
    checks, proofs, reused, extensions = 0, 0, 0, 0
    for plan in packet["plans"]:
        guard = _direct_guard(world, dict(packet, **packet["action_guards"][plan.trajectory_hash]))
        if not guard.accepted:
            affected.add(plan.trajectory_hash)
            reasons[plan.trajectory_hash] = guard.reason
            invalidation_details[plan.trajectory_hash] = dict(guard.details)
    fallback_ids = packet.get("fallback_aircraft_ids", ())
    fallback_changed = False
    fallback_change_details = {}
    if fallback_ids:
        fallback_guard = _direct_guard(world, dict(packet, direct_ids=fallback_ids,
            direct_hashes=(), plans=(), resources=(), pad_resources=(), resource_key=(), pad_key=()))
        if not fallback_guard.accepted:
            fallback_changed = True
            fallback_change_details = dict(reason=fallback_guard.reason, **dict(fallback_guard.details))
        for aid in fallback_ids:
            original, actual = packet["source"].aircraft.get(aid), world.aircraft.get(aid)
            if original is None or actual is None: continue
            before = packet["source"].energy_ledger.get("executed", {}).get(aid, ())
            after = world._energy_state()["executed"].get(aid, ())
            original_basis = original.peak_seconds-math.fsum(r["charged_peak"] for r in before)
            actual_basis = actual.peak_seconds-math.fsum(r["charged_peak"] for r in after)
            if abs(original_basis-actual_basis) > 1.e-9:
                fallback_changed = True
                fallback_change_details = dict(reason="PEAK_BASIS_CHANGED", aircraft_id=aid)
    edges = []
    def withdraw_dependents():
        while True:
            previous = set(affected)
            for candidate in packet["plans"]:
                for supporting_hash in packet["starts"][candidate.aircraft_id][1]:
                    if supporting_hash not in affected: continue
                    edge = (supporting_hash, candidate.trajectory_hash)
                    if edge not in edges: edges.append(edge)
                    if candidate.trajectory_hash not in affected:
                        affected.add(candidate.trajectory_hash)
                        reasons[candidate.trajectory_hash] = "RELEASE_SUPPORT_WITHDRAWN"
            if affected == previous: break
    withdraw_dependents()
    # Geometry-only projection preserves the same target-time wrapper identity
    # as the issued certificate, without energy integration or speculative work.
    source = world.build_planning_snapshot()
    context = _planning_context(source)
    context.time = packet["timing"].effective_time
    aircraft, active, future = {}, {}, {}
    for aid, original in source.aircraft.items():
        projected, current, later, _ = _motion(source, original, context.time)
        aircraft[aid] = projected
        if current is not None: active[aid] = current
        if later is not None: future[aid] = later
    context.aircraft, context.active, context.soft = aircraft, active, future
    soft = dict(context.soft)
    soft.update({p.aircraft_id: p for p in packet["plans"] if p.trajectory_hash not in affected})
    horizon = max([packet["pair_horizon"], world.time,
                   *(p.end_time+2*math.pi/.035 for p in (*context.active.values(), *soft.values()))])
    start = max(world.time, packet["timing"].effective_time)
    certificates = dict(packet["pair_certificates"])
    for plan in packet["plans"]:
        if plan.trajectory_hash in affected: continue
        first = context._traffic(context.aircraft[plan.aircraft_id], soft)
        for aid in sorted(set(context.aircraft)-{plan.aircraft_id}):
            other = context.aircraft.get(aid)
            if other is None or (other.status not in AIRBORNE and other.status != "LANDED_SERVICE"):
                continue
            second = context._traffic(other, soft)
            end = min(horizon, first.end_time, second.end_time)
            x, y = sorted((first, second), key=lambda p: p.aircraft_id)
            sx, sy = _traffic_signature(x), _traffic_signature(y)
            sx = (*sx[:2], start, sx[3]); sy = (*sy[:2], start, sy[3])
            key = (x.aircraft_id, y.aircraft_id)
            old = packet["pair_certificates"].get(key)
            if (aid not in changed and plan.aircraft_id not in changed and old is not None
                    and old[:2] == (sx, sy) and old[2] <= start and end <= old[3]):
                reused += 1
                continue
            if old is not None and end > old[3]: extensions += 1
            if time.perf_counter() >= deadline:
                return ValidationResult(False, "UNKNOWN", "REVALIDATION_WORK_DEADLINE"), None
            if envelope_safe(first, second, start, end, world.config):
                proofs += 1
            elif not certify:
                affected.add(plan.trajectory_hash)
                reasons[plan.trajectory_hash] = "TRAJECTORY_ENVELOPE_AMBIGUOUS"
                continue
            else:
                result = validate_separation(first, second, start, end, config=world.config)
                checks += 1
                if time.perf_counter() >= deadline:
                    return ValidationResult(False, "UNKNOWN", "REVALIDATION_WORK_DEADLINE"), None
                if not result.accepted or result.status != "SAFE":
                    affected.add(plan.trajectory_hash)
                    reasons[plan.trajectory_hash] = result.reason or result.status
                    conflicts.append(dict(action_hash=plan.trajectory_hash,
                        action_aircraft_id=plan.aircraft_id, other_aircraft_id=aid,
                        first_trajectory=_traffic_signature(first), second_trajectory=_traffic_signature(second),
                        interval=(start, end), status=result.status, reason=result.reason))
                    continue
            certificates[key] = (sx, sy, start, end)
    withdraw_dependents()
    details = dict(trigger="TRAFFIC_DELTA", affected_aircraft_ids=sorted(changed),
        affected_action_hashes=sorted(affected), retained_action_hashes=[p.trajectory_hash for p in packet["plans"]
            if p.trajectory_hash not in affected], withdrawn_action_reasons=reasons,
        action_invalidation_details=invalidation_details,
        fallback_revalidation_required=fallback_changed, fallback_change_details=fallback_change_details,
        recursive_release_edges=edges, conflicts=conflicts, pairs_checked=checks,
        reused_pairs=reused, envelope_proofs=proofs, horizon_extension_pairs=extensions, full_fallback=False)
    accepted = not affected and not fallback_changed
    return ValidationResult(accepted, "GUARDED" if accepted else "REJECTED",
                            "" if accepted else "TRAFFIC_REVALIDATION_REQUIRED", details), certificates


def guard_submission(world, packet):
    result, certificates = _analyze(world, packet)
    if result.accepted and (result.details.get("affected_aircraft_ids") or result.details.get("horizon_extension_pairs")):
        from .planning_latency import _commitment_table
        source = world.build_planning_snapshot()
        source = replace(source, aircraft=freeze(dict(source.aircraft)), active=freeze(dict(source.active)),
                         soft=freeze(dict(source.soft)), requests=freeze(dict(source.requests)))
        packet["local_traffic_source"] = source
        packet["local_traffic_tables"] = {aid: _commitment_table(source, a, packet["timing"].effective_time)
            for aid, a in source.aircraft.items() if a.status in AIRBORNE or a.status == "LANDED_SERVICE"}
        packet["pair_certificates"] = certificates
        packet["recovery_log"].append(dict(result.details, delivery="original", observed_time=world.time))
    return result


def _fallback_energy(snapshot, withdrawn, *, deadline):
    """Finite feasibility through the next normal adjustment, never forever."""
    from .planning_latency import _motion, _capacity
    context = _planning_context(snapshot)
    timing = snapshot.planning_timing
    until = timing.effective_time+(timing.effective_time-timing.snapshot_time)
    rows = []
    for aid in sorted(withdrawn):
        if time.perf_counter() >= deadline:
            return ValidationResult(False, "UNKNOWN", "REVALIDATION_WORK_DEADLINE"), rows
        original = snapshot.aircraft.get(aid)
        if original is None:
            return ValidationResult(False, "REJECTED", "FALLBACK_AIRCRAFT_MISSING"), rows
        row = dict(aircraft_id=aid, interval=(snapshot.time, until),
                   scope="next_normal_adjustment_energy_feasibility")
        if original.status not in AIRBORNE:
            rows.append(dict(row, accepted=True, work="ZERO_GROUND_OR_CLEARED", status=original.status))
            continue
        approved = tuple(p for p in (snapshot.active.get(aid), snapshot.soft.get(aid)) if p is not None)
        if ((original.emergency.eligible or original.emergency.declared or original.emergency.pending) and not approved):
            rows.append(dict(row, accepted=False, reason="UNAPPROVED_EMERGENCY_WAIT"))
            return ValidationResult(False, "REJECTED", "FALLBACK_EMERGENCY_DISPOSITION_REQUIRED"), rows
        _, _, _, portions = _motion(snapshot, original, until)
        estimates = [context._energy(plan, start, end, fine=True) for plan, start, end in portions]
        if any(not q.valid or not all(math.isfinite(value) for value in
                (q.joules, q.soc_used, q.peak_seconds, q.error_j, q.peak_error_s)) for q in estimates):
            rows.append(dict(row, accepted=False, reason="ENERGY_UNKNOWN"))
            return ValidationResult(False, "UNKNOWN", "FALLBACK_ENERGY_UNKNOWN"), rows
        total = context._sum_energy(estimates)
        # snapshot SOC/peak already contain inherited executed uncertainty;
        # subtract/add only these new continuation error allowances here.
        soc_lower = original.soc-total.soc_used-total.error_j/_capacity(snapshot.config)
        peak_upper = original.peak_seconds+total.peak_seconds+total.peak_error_s
        accepted = soc_lower > 0 and peak_upper <= 120.
        rows.append(dict(row, accepted=accepted, continuation_hashes=[p.trajectory_hash for p, _, _ in portions],
            start_soc_lower=original.soc, end_soc_lower=soc_lower, joules=total.joules,
            error_j=total.error_j, peak_seconds_upper=peak_upper, peak_error_s=total.peak_error_s,
            precision="fine", inherited_uncertainty="included_in_projected_soc_and_peak"))
        if not accepted:
            return ValidationResult(False, "REJECTED", "FALLBACK_ENERGY_INSUFFICIENT"), rows
    return ValidationResult(True, "VALID"), rows


def revalidate_submission(world, token, *, max_work_s=None, provider=None):
    from .planning_latency import prepare_submission, LOCAL_TRAFFIC_POLICY
    from .future_commitment import discard_pending
    started = time.perf_counter()
    deadline = math.inf if max_work_s is None else started+max_work_s
    registry = world._planning_latency_packets
    packet = registry[token]
    previous = packet.get("revalidation") or {}
    for superseded in (previous.get("replacement_token"), packet.pop("repair_candidate_token", None)):
        if superseded is not None: discard_pending(world,superseded,'REVALIDATION_SUPERSEDED',superseded=True)
    packet["revalidation"] = None
    if world.time > packet["timing"].ready_deadline:
        return ValidationResult(False, "REJECTED", "READY_DEADLINE_MISSED")
    # Withdrawal precedes ambiguous original-pair validation. Time needed to
    # reconsider withdrawn actions cannot consume the standby's certification.
    result, certificates = _analyze(world, packet, deadline=deadline)
    if certificates is None: return result
    affected = set(result.details["affected_action_hashes"])
    retained = tuple(p for p in packet["plans"] if p.trajectory_hash not in affected)
    details = dict(result.details)
    details["invalidated_certificates"] = [key for key in packet["pair_certificates"]
        if any(aid in result.details["affected_aircraft_ids"] for aid in key)]
    if not retained:
        return ValidationResult(False, "REJECTED", "NO_CERTIFIED_STANDBY", details)
    if time.perf_counter() >= deadline:
        return ValidationResult(False, "UNKNOWN", "REVALIDATION_WORK_DEADLINE", details)
    withdrawn_ids = ({p.aircraft_id for p in packet["plans"] if p.trajectory_hash in affected}
                     | set(packet.get("fallback_aircraft_ids", ())))
    invalid_ids = withdrawn_ids | set(result.details["affected_aircraft_ids"])
    reusable = {key: row for key, row in packet["pair_certificates"].items()
                if not invalid_ids.intersection(key)}
    prepared, standby_token = prepare_submission(world, retained, packet["timing"],
        traffic_policy=LOCAL_TRAFFIC_POLICY, pair_certificates=reusable)
    elapsed = time.perf_counter()-started
    details.update(minimum_work_s=elapsed, standby_token=standby_token,
                   delivery="standby" if affected else "original")
    if not prepared.accepted:
        details["standby_certification_failure"] = dict(prepared.details)
        return ValidationResult(False, prepared.status, prepared.reason, details)
    if time.perf_counter() >= deadline:
        discard_pending(world,standby_token,'REVALIDATION_WORK_DEADLINE')
        return ValidationResult(False, "UNKNOWN", "REVALIDATION_WORK_DEADLINE", details)
    standby = registry[standby_token]
    binding=packet.get('selection_binding')
    if affected and binding is not None and len(binding.get('objective') or ())==6:
        scores,score_evidence=_current_scores(world,packet,provider,[retained],deadline)
        if scores is None:
            discard_pending(world,standby_token,score_evidence.get('reason','CURRENT_OBJECTIVE_RESCORE_REQUIRED'))
            return ValidationResult(False,'UNKNOWN',score_evidence.get('reason','CURRENT_OBJECTIVE_RESCORE_REQUIRED'),details)
        standby['selection_binding']=dict(objective=list(scores[0]),prefix_hashes=[p.trajectory_hash for p in retained],
            first_action_hashes=[p.trajectory_hash for p in retained],certified=True,
            recovery_rescore=score_evidence,source_selection_binding=binding)
        details['scoring']=score_evidence
    elif not affected:
        standby['selection_binding']=binding
    fallback, fallback_rows = _fallback_energy(standby["projected"], withdrawn_ids, deadline=deadline)
    details["fallback_energy"] = fallback_rows
    details["minimum_work_s"] = time.perf_counter()-started
    if not fallback.accepted:
        discard_pending(world,standby_token,fallback.reason)
        return ValidationResult(False, fallback.status, fallback.reason, details)
    if time.perf_counter() >= deadline:
        discard_pending(world,standby_token,'REVALIDATION_WORK_DEADLINE')
        return ValidationResult(False, "UNKNOWN", "REVALIDATION_WORK_DEADLINE", details)
    standby["fallback_energy"] = fallback_rows
    standby["fallback_aircraft_ids"] = tuple(sorted(withdrawn_ids))
    standby["not_ready_before"] = world.time+time.perf_counter()-started
    standby["original_plans"] = packet["original_plans"]
    standby["recovery_log"] = [*packet["recovery_log"], dict(details, observed_time=world.time)]
    standby["repair_requested"] = bool(affected)
    details["pairs_checked"] += standby["certification_checks"]["pairs_checked"]
    details["reused_pairs"] += standby["certification_checks"]["reused_pairs"]
    packet["revalidation_sequence"] += 1
    attempt = packet["revalidation_sequence"]
    packet["revalidation"] = dict(attempt=attempt, replacement_token=standby_token,
        not_ready_before=standby["not_ready_before"], details=details)
    return ValidationResult(True, "REVALIDATING", details=dict(details, attempt=attempt))


def complete_revalidation(world, token, attempt, *, promote=False):
    registry = world._planning_latency_packets
    packet = registry[token]
    candidate = packet.get("revalidation")
    if candidate is None or candidate["attempt"] != attempt:
        return ValidationResult(False, "REJECTED", "UNISSUED_REVALIDATION")
    if world.time < candidate["not_ready_before"]:
        return ValidationResult(False, "WAIT", "REVALIDATION_NOT_COMPLETE", dict(due=candidate["not_ready_before"]))
    if world.time > packet["timing"].ready_deadline:
        return ValidationResult(False, "REJECTED", "READY_DEADLINE_MISSED")
    replacement = registry.get(candidate["replacement_token"])
    if replacement is None: return ValidationResult(False, "REJECTED", "UNISSUED_LATENCY_PACKET")
    result = guard_submission(world, replacement)
    if not result.accepted: return result
    if promote:
        registry[token] = replacement
        registry.pop(candidate["replacement_token"], None)
        replacement["not_ready_before"] = world.time
    return ValidationResult(True, "REVALIDATED", details=candidate["details"])


def _current_scores(world, packet, provider, alternatives, deadline):
    """All options share one fresh frame and identical continuation estimates."""
    from .planning_latency import project_visible_snapshot
    from .stage2_planning import EvaluationContext, PlanningBudget, _decision_aircraft
    from .selective_replanning.frame import build_global_frame, FrameIncomplete
    from .stage2_downstream import describe_column, SERVICE_EVALUATION_PROFILE
    if provider is None or getattr(provider, "evaluation_profile", None) != SERVICE_EVALUATION_PROFILE:
        return None, dict(reason="CURRENT_OBJECTIVE_PROVIDER_UNAVAILABLE")
    if time.perf_counter() >= deadline: return None, dict(reason="RECOVERY_SCORE_DEADLINE")
    snapshot = project_visible_snapshot(world.build_planning_snapshot(), packet["timing"].effective_time,
                                        timing=packet["timing"], fine=True)
    budget = PlanningBudget(planning_wall_deadline=deadline)
    context = EvaluationContext(snapshot, budget=budget)
    try: frame = build_global_frame(snapshot, provider, context)
    except FrameIncomplete as exc: return None, dict(reason=str(exc))
    eligible = {a.id for a in _decision_aircraft(snapshot)}
    opened = tuple(sorted({p.aircraft_id for p in packet["original_plans"]} & eligible))
    columns, demands, selections = {}, {}, []
    for plans in alternatives:
        by_id = {p.aircraft_id: p for p in plans}
        selected = []
        for aid in opened:
            if time.perf_counter() >= deadline: return None, dict(reason="RECOVERY_SCORE_DEADLINE")
            plan = by_id.get(aid)
            key = (aid, None if plan is None else plan.trajectory_hash)
            if key not in columns:
                column = provider._column(snapshot, snapshot.aircraft[aid], () if plan is None else (plan,),
                                          "KEEP" if plan is None else "MOVE", context)
                if column is None: return None, dict(reason="RECOVERY_COLUMN_UNKNOWN")
                columns[key] = column
                demands[column.signature] = describe_column(snapshot, column, provider.references, context,
                    evaluation_profile=SERVICE_EVALUATION_PROFILE)
            selected.append(columns[key])
        selections.append(tuple(selected))
    downstream = frame.downstream(demands, opened)
    scores = [downstream.key(selection) for selection in selections]
    if time.perf_counter() >= deadline: return None, dict(reason="RECOVERY_SCORE_DEADLINE")
    return scores, dict(frame_id=frame.identity, snapshot_id=snapshot.version,
        objective_schema=("F", "-L", "R", "G_service", "E", "tau"),
        scope="current_executable_prefixes", replan_ids=opened, scores=scores)


def repair_submission(world, token, provider, *, max_work_s=None):
    """Bounded original-action reintroduction after standby readiness exists."""
    from .planning_latency import prepare_submission, LOCAL_TRAFFIC_POLICY, ProjectionRejected
    from .future_commitment import discard_pending
    started = time.perf_counter()
    deadline = math.inf if max_work_s is None else started+max_work_s
    registry, packet = world._planning_latency_packets, world._planning_latency_packets[token]
    if packet.get("ready_at") is None:
        return ValidationResult(False, "REJECTED", "STANDBY_NOT_READY")
    guarded = guard_submission(world, packet)
    if not guarded.accepted: return guarded
    original, standby = packet["original_plans"], packet["plans"]
    retained = {p.trajectory_hash for p in standby}
    missing = tuple(p for p in original if p.trajectory_hash not in retained)
    attempts = [tuple((*standby, p)) for p in missing]
    if len(missing) > 1: attempts.append(tuple(original))
    valid = [(token, tuple(standby))]
    evidence = []
    for proposed in attempts[:9]:
        if time.perf_counter() >= deadline: break
        result, candidate = prepare_submission(world, proposed, packet["timing"],
            traffic_policy=LOCAL_TRAFFIC_POLICY, pair_certificates=packet["pair_certificates"])
        if result.accepted:
            prepared = registry[candidate]
            still_withdrawn = {p.aircraft_id for p in original}-{p.aircraft_id for p in proposed}
            fallback, rows = _fallback_energy(prepared["projected"], still_withdrawn, deadline=deadline)
            prepared["fallback_aircraft_ids"] = tuple(sorted(still_withdrawn))
            prepared["fallback_energy"] = rows
            prepared["original_plans"] = original
            if not fallback.accepted:
                discard_pending(world,candidate,fallback.reason)
                result = ValidationResult(False, fallback.status, fallback.reason,
                    dict(fallback_energy=rows))
        evidence.append(dict(hashes=[p.trajectory_hash for p in proposed], accepted=result.accepted,
                             status=result.status, reason=result.reason, validation_details=dict(result.details)))
        if result.accepted: valid.append((candidate, proposed))
    score_evidence = dict(reason="NO_VALID_REPAIR")
    scores = None
    if time.perf_counter() < deadline:
        try: scores, score_evidence = _current_scores(world, packet, provider, [plans for _, plans in valid], deadline)
        except ProjectionRejected as exc: score_evidence = dict(reason=str(exc))
    elapsed = time.perf_counter()-started
    chosen = 0 if scores is None or elapsed > (math.inf if max_work_s is None else max_work_s) else min(range(len(scores)), key=lambda i: (scores[i], i))
    selected_token = valid[chosen][0]
    if scores is not None:
        registry[selected_token]['selection_binding']=dict(objective=list(scores[chosen]),
            prefix_hashes=[p.trajectory_hash for p in valid[chosen][1]],
            first_action_hashes=[p.trajectory_hash for p in valid[chosen][1]],certified=True,
            recovery_rescore=score_evidence)
    for candidate, _ in valid[1:]:
        if candidate != selected_token: discard_pending(world,candidate,'REPAIR_CANDIDATE_NOT_SELECTED',superseded=True)
    details = dict(delivery="standby" if chosen == 0 else "repaired", repair_attempts=evidence,
        scoring=score_evidence, elapsed_s=elapsed, minimum_work_s=elapsed,
        timed_out=time.perf_counter() >= deadline, standby_ready_at=packet["ready_at"],
        retained_action_hashes=[p.trajectory_hash for p in valid[chosen][1]])
    packet["repair_requested"] = False
    packet["recovery_log"].append(details)
    if chosen == 0: return ValidationResult(True, "STANDBY_RETAINED", details=details)
    candidate = registry[selected_token]
    candidate["original_plans"] = original
    candidate["recovery_log"] = list(packet["recovery_log"])
    packet["repair_candidate_token"] = selected_token
    packet["repair_candidate_not_ready_before"] = world.time+elapsed
    return ValidationResult(True, "REPAIRING", details=dict(details, candidate_token=selected_token))


def complete_repair(world, token, *, handoff_delay=0.):
    """Promote only a freshly guarded repair with its own timely receipt."""
    from .planning_latency import mark_submission_ready
    from .future_commitment import discard_pending
    registry = world._planning_latency_packets
    packet = registry[token]
    candidate_token = packet.pop("repair_candidate_token", None)
    if candidate_token is None:
        guarded = guard_submission(world, packet)
        if not guarded.accepted: return guarded
        return ValidationResult(True, "STANDBY_RETAINED", details=dict(ready_at=packet["ready_at"]))
    candidate = registry.get(candidate_token)
    if candidate is None:
        return ValidationResult(False, "REJECTED", "UNISSUED_LATENCY_PACKET")
    result = mark_submission_ready(world, candidate_token, handoff_delay=handoff_delay)
    if not result.accepted:
        discard_pending(world,candidate_token,result.reason)
        # The repair never invalidates the already registered standby receipt.
        standby_guard = guard_submission(world, packet)
        if not standby_guard.accepted: return standby_guard
        return ValidationResult(True, "STANDBY_RETAINED", details=dict(
            ready_at=packet["ready_at"], repair_rejected_reason=result.reason))
    registry[token] = candidate
    registry.pop(candidate_token, None)
    return ValidationResult(True, "REPAIRED", details=dict(result.details))
