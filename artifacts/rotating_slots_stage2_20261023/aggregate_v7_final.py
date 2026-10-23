"""Read existing local Stage2 artifacts; aggregate without simulation or audit reruns."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import time

METHODS = ("energy_slack", "rh_cp_sat")
SCENARIOS = ("stress", "diagnostic", "medium")
BASE = Path("D:/7_codespace/training_outputs/terminal")
OUT = Path(__file__).with_name("v7_final_0600_facts.json")
AIRBORNE = {"INBOUND", "HOLD_OUTER", "HOLD_MIDDLE", "HOLD_INNER", "TRANSFER_O_M",
            "TRANSFER_M_I", "FINAL_NORMAL", "FINAL_EMERGENCY", "VERTICAL_DESCENT"}
QUALITY = ("L_normal", "L_emergency", "L_success", "F_energy_total", "F_battery", "F_peak",
    "N_offered", "N_admitted_total", "N_airborne_pending", "N_upstream_pending", "N_rerouted",
    "initial_airborne_count", "initial_ground_count", "manifest_offered", "planned_manifest", "unreleased",
    "emergency_declared", "emergency_trigger_count", "emergency_unresolved", "protected_rescue_opportunities",
    "rescue_warning", "rescue_search_exhausted", "landing_below_reserve", "cost_emergency", "cost_failure",
    "cost_reroute", "energy_j", "energy_kwh", "E_terminal_J", "airborne_time", "upstream_wait_time",
    "holding_time", "pad_utilization_seconds", "decision_count", "conservation", "terminated_invalid")
CANDIDATE_STAGES = ("candidate_generation_and_single_vehicle", "safe_progress_generation",
                    "reference_seed_generation", "geometry_template_screen")
VALIDATION_STAGES = ("trajectory_validation", "prefix_validation")


def load(path):
    return json.loads(Path(path).read_bytes())


def rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def digest_file(path):
    value = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def counts(values):
    def key(value):
        return json.dumps(value, sort_keys=True, ensure_ascii=False) if not isinstance(value, str) else value
    return dict(sorted(Counter(key(value) for value in values).items()))


def stats(values):
    values = sorted(float(value) for value in values if value is not None)
    if not values:
        return dict(count=0, sum_s=0., mean=None, p50=None, p95=None, p99=None, max=None, over_5s=0)
    def quantile(q):
        position = (len(values) - 1) * q
        low, high = math.floor(position), math.ceil(position)
        return values[low] + (values[high] - values[low]) * (position - low)
    total = math.fsum(values)
    return dict(count=len(values), sum_s=total, mean=total/len(values), p50=quantile(.5),
                p95=quantile(.95), p99=quantile(.99), max=values[-1], over_5s=sum(v > 5 for v in values))


def selected(value, keys=QUALITY):
    return {key: value[key] for key in keys if key in value}


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-6)


def stage_total(row, names):
    timings = row.get("timings", {})
    return sum(timings[name][0] for name in names if name in timings) if any(name in timings for name in names) else None


def timing_and_planning(decisions, dispatch):
    stages = sorted({name for row in decisions for name in row.get("timings", {})})
    solver = [item for row in decisions for item in row.get("solver", ())]
    choices = [choice for row in decisions for choice in row.get("aircraft_choices", ())]
    return dict(
        decision_rows=len(decisions), dispatch_rows=len(dispatch),
        normal_callback_wall=stats(row.get("normal_wall_s") for row in dispatch),
        inner_algorithm_wall=stats(row.get("wall_s") for row in decisions),
        dispatch_components={name: stats(row.get(name) for row in dispatch) for name in
            sorted({key for row in dispatch for key in row if key.endswith(("_wall_s", "_cpu_s"))})},
        decision_stage_timings={name: dict(wall=stats(row.get("timings", {}).get(name, (None, None))[0] for row in decisions),
            cpu=stats(row.get("timings", {}).get(name, (None, None))[1] for row in decisions)) for name in stages},
        candidate_exclusive_wall=stats(stage_total(row, CANDIDATE_STAGES) for row in decisions),
        safety_validation_wall=stats(stage_total(row, VALIDATION_STAGES) for row in decisions),
        selected_validation_wall=stats(stage_total(row, ("selected_validation",)) for row in decisions),
        solver_wall=stats(row.get("solver_wall_s") for row in decisions),
        hint_wall=stats(row.get("hint_wall_s") for row in decisions),
        solver_entered=counts(row.get("solver_entered") for row in decisions),
        strict_improvement_over_hint=counts(row.get("strict_improvement_over_hint") for row in decisions),
        hint_certified=counts(row.get("hint_certified") for row in decisions),
        final_equals_hint_calls=sum(row.get("hint_objective") is not None and row.get("final_objective") == row["hint_objective"] for row in decisions),
        solver_phase_calls=len(solver), solver_phase_status=counts(item.get("status") for item in solver),
        solver_phase_counts=counts(item.get("phase") for item in solver),
        solver_validation_status=counts(item.get("validation_status") for item in solver),
        solver_prefix_status=counts(item.get("prefix_status") for item in solver),
        solver_skip_reasons=counts(row.get("skip_reason") for row in decisions),
        solver_stop_reasons=counts(row.get("solver_stop_reason") for row in decisions),
        decision_status=counts(row.get("status") for row in decisions),
        commit_status=counts(row.get("commit_status") for row in decisions),
        fallback_calls=sum(bool(row.get("fallback")) for row in decisions),
        fallback_values=counts(row.get("fallback") for row in decisions),
        submitted_actions=sum(row.get("submitted", 0) for row in decisions),
        noop_calls=sum(row.get("submitted", 0) == 0 for row in decisions),
        stale_calls=sum(bool(row.get("stale")) for row in decisions),
        watchdog_calls=sum(bool(row.get("watchdog")) for row in decisions),
        catalog_incomplete_calls=sum(bool(row.get("catalog_incomplete")) for row in decisions),
        truncation_labels=counts(label for row in decisions for label in set(row.get("truncations", ()))),
        planning_budget_settings=[json.loads(value) for value in sorted({json.dumps(row.get("planning_budget"), sort_keys=True) for row in decisions})],
        candidate_coverage=dict(aircraft_call_observations=len(choices),
            move_columns=sum(choice.get("move_candidates", 0) for choice in choices),
            complete_chain_columns=sum(choice.get("full_normal_chain_count", 0) for choice in choices),
            partial_progress_columns=sum(choice.get("partial_progress_count", 0) for choice in choices),
            only_keep_wait_observations=sum(bool(choice.get("only_keep_wait")) for choice in choices),
            rejection_reasons=dict(sum((Counter(choice.get("rejection_reasons", {})) for choice in choices), Counter()))),
        timing_scope="Stage hierarchy is not additive. Candidate-exclusive and validation groups use the same named stages as summary rescue_observations; sum_s is seconds, including cpu-stage values when labelled cpu.")


def event_summary(path, rejection_path, plans, initial, request_ids, final_aircraft, observations):
    event_types = Counter()
    timeline = defaultdict(lambda: defaultdict(list))
    attempts = defaultdict(Counter)
    first_last_attempt = {}
    rejection_counts = defaultdict(Counter)
    rejection_samples = {}
    starts, normal_finals, touchdowns = [], [], []
    first_declaration = None
    interesting = {"rescue_warning", "emergency_triggered", "emergency_declared", "rescue_opportunity_protected",
                   "rescue_commitment_protected", "soft_plan_cancelled", "emergency_priority_yielded"}
    cohort_landings = defaultdict(Counter)
    for event in rows(path):
        if event["time"] > 600 + 1e-7:
            continue
        event_types[event["type"]] += 1
        aid, kind, details = event.get("aircraft_id"), event["type"], event.get("details", {})
        if kind in interesting:
            timeline[aid][kind].append(event)
        if kind == "emergency_declared":
            first_declaration = event["time"] if first_declaration is None else min(first_declaration, event["time"])
        if kind == "rescue_opportunity_attempt":
            attempts[aid][str(details.get("status"))+":"+str(details.get("reason"))] += 1
            sample = first_last_attempt.setdefault(aid, {"first": event, "last": event})
            sample["last"] = event
        if kind == "maneuver_started":
            trajectory = plans.get(details.get("trajectory_hash"), {})
            actual = dict(time=event["time"], aircraft_id=aid, kind=trajectory.get("kind", details.get("maneuver_kind")),
                trajectory_hash=details.get("trajectory_hash"), planned_start=trajectory.get("start_time"),
                planned_end=trajectory.get("end_time"), pad_id=trajectory.get("pad_id"))
            starts.append(actual)
            if actual["kind"] == "FINAL_NORMAL":
                normal_finals.append(actual)
            if actual["kind"] == "FINAL_EMERGENCY":
                timeline[aid]["actual_rescue_start"].append(actual)
        if kind == "touchdown":
            touchdowns.append(event)
            timeline[aid]["touchdown"].append(event)
            cohort = "initial_airborne" if aid in initial and initial[aid].get("ring") else "new_requests" if aid in request_ids else "OTHER"
            cohort_landings[cohort]["emergency" if details.get("emergency") else "normal"] += 1
    for rejection in rows(rejection_path):
        if rejection["time"] <= 600 + 1e-7:
            aid = rejection.get("aircraft_id")
            rejection_counts[aid][rejection.get("reason", "UNKNOWN")] += 1
            sample = rejection_samples.setdefault(aid, {"first": rejection, "last": rejection})
            sample["last"] = rejection
    obs = {item["aircraft_id"]: item for item in observations.get("aircraft", ())}
    focus = {"initial_inner_08", "req_000000"}
    rescue_ids = {aid for aid, events in timeline.items() if any(key in events for key in interesting)} | set(attempts) | focus
    aircraft = []
    for aid in sorted(rescue_ids):
        a = final_aircraft.get(aid)
        own = timeline.get(aid, {})
        emergency = a.get("emergency", {}) if a else {}
        declaration = emergency.get("declaration_time")
        actual_start = emergency.get("start_time")
        trigger = own.get("emergency_triggered", [{}])[0].get("time")
        wait = None if trigger is None else max(0., (600. if actual_start is None else actual_start)-trigger)
        aircraft.append(dict(aircraft_id=aid, final_state=None if a is None else selected(a,
            ("status", "ring", "slot", "soc", "peak_seconds", "energy_j", "touchdown_time", "trajectory_hash", "airborne_time")),
            final_emergency=emergency, recorded_summary_observation=obs.get(aid), events=dict(own),
            rescue_opportunity_attempt_counts=dict(attempts.get(aid, {})),
            rescue_opportunity_attempt_bookends=first_last_attempt.get(aid),
            rejection_counts=dict(rejection_counts.get(aid, {})), rejection_bookends=rejection_samples.get(aid),
            trigger_to_actual_start_wait_s=wait,
            trigger_wait_status=None if wait is None else "CENSORED_AT_600" if actual_start is None else "STARTED",
            declaration_to_actual_start_or_600_s=None if declaration is None else max(0., (600. if actual_start is None else actual_start)-declaration),
            declared_waiting_at_600=bool(emergency.get("declared") and actual_start is None and a and a["status"] in AIRBORNE)))
    normal_after = [item for item in normal_finals if first_declaration is not None and item["time"] > first_declaration + 1e-7]
    declared_by_id = {aid: values["emergency_declared"][0]["time"] for aid, values in timeline.items() if values.get("emergency_declared")}
    for item in normal_after:
        item["own_emergency_declared_before_start"] = item["aircraft_id"] in declared_by_id and declared_by_id[item["aircraft_id"]] <= item["time"] + 1e-7
    return dict(event_type_counts=dict(event_types), first_declaration_time=first_declaration,
        actual_maneuver_start_kind_counts=dict(Counter(item["kind"] for item in starts)),
        actual_normal_final_start_events=normal_finals,
        actual_normal_final_start_events_strictly_after_first_emergency_declaration=normal_after,
        actual_emergency_start_events=[item for item in starts if item["kind"] == "FINAL_EMERGENCY"],
        normal_touchdowns=sum(not event["details"].get("emergency") for event in touchdowns),
        emergency_touchdowns=sum(bool(event["details"].get("emergency")) for event in touchdowns),
        low_reserve_touchdowns=sum(bool(event["details"].get("below_reserve")) for event in touchdowns),
        touchdown_counts_by_origin={key: dict(value) for key, value in cohort_landings.items()},
        warning_unique_aircraft=sum(bool(value.get("rescue_warning")) for value in timeline.values()),
        trigger_unique_aircraft=sum(bool(value.get("emergency_triggered")) for value in timeline.values()),
        declaration_unique_aircraft=len(declared_by_id),
        rescue_aircraft=aircraft,
        declared_waiting_at_600=[dict(aircraft_id=item["aircraft_id"], **(item["final_state"] or {}),
            wait_s=item["trigger_to_actual_start_wait_s"], declaration_time=item["final_emergency"].get("declaration_time"))
            for item in aircraft if item["declared_waiting_at_600"]],
        all_rejection_reason_counts=dict(sum(rejection_counts.values(), Counter())),
        rejection_event_count=sum(sum(value.values()) for value in rejection_counts.values()),
        focus_aircraft={aid: next((item for item in aircraft if item["aircraft_id"] == aid), None) for aid in sorted(focus)},
        interpretation="Protected opportunity start/end timestamps are predicted commitments; actual rescue starts are maneuver_started events joined to trajectory kind. Waiting without an actual start is right-censored at 600. Actual normal finals after first declaration are event records, not reservation/submission records.")


def stream_states(path):
    first, last, previous, count = None, None, None, 0
    boundaries = {}
    for state in rows(path):
        assert previous is None or state["time"] > previous, (str(path), previous, state["time"])
        first = state if first is None else first
        last, previous = state, state["time"]
        count += 1
        boundary = round(state["time"]/60)*60
        if 0 <= boundary <= 600 and abs(state["time"]-boundary) < 1e-7:
            boundaries[boundary] = state["metrics"]
    return first, last, count, boundaries


def one_method(root, method, fixture, request_ids, branch):
    directory, replay = root/method, root/method/"replay_0600"
    summary, audit = load(directory/"summary_0600.json"), load(directory/"audit_0600.json")
    index, visual = load(replay/"replay.json"), load(replay/"visual_acceptance.json")
    metrics = summary["metrics"]
    decisions, dispatch = list(rows(directory/"decisions.jsonl")), list(rows(directory/"dispatch_timings.jsonl"))
    assert all(row["time"] <= 600 + 1e-7 for row in (*decisions, *dispatch))
    assert len(decisions) == len(dispatch) == metrics["decision_count"]
    first, last, frame_count, state_boundaries = stream_states(replay/"states.jsonl")
    assert first["time"] == 0 and last["time"] == 600
    assert frame_count == index["states_count"] == audit["state_frames"]
    plans = load(replay/"trajectories.json")
    initial = {aircraft["id"]: aircraft for aircraft in fixture["aircraft"]}
    final_aircraft = {aircraft["id"]: aircraft for aircraft in last["aircraft"]}
    events = event_summary(replay/"events.jsonl", replay/"rejections.jsonl", plans,
        initial, request_ids, final_aircraft, summary["rescue_observations"])
    assert events["normal_touchdowns"] == metrics["L_normal"]
    assert events["emergency_touchdowns"] == metrics["L_emergency"]
    assert events["low_reserve_touchdowns"] == metrics["landing_below_reserve"]
    assert metrics["N_offered"] == metrics["initial_airborne_count"] + metrics["manifest_offered"]
    assert metrics["L_success"] == metrics["L_normal"] + metrics["L_emergency"]
    assert close(metrics["energy_j"]/3600000., metrics["energy_kwh"])
    assert close(metrics["energy_j"], metrics["E_terminal_J"])
    numeric_differences = {key: metrics[key]-value for key, value in audit["derived_costs"].items() if key in metrics}
    count_matches = {key: metrics[key] == value for key, value in audit["counts"].items() if key in metrics}
    assert all(close(value, 0) for value in numeric_differences.values())
    assert all(count_matches.values())
    assert metrics["K_episode"] == audit["primary_key"]
    boundary = branch["boundary_s"] if branch else 0.
    inherited = branch["methods"][method]["inherited_energy_j"] if branch else 0.
    if branch:
        assert close(inherited, load(directory/"summary_0240.json")["metrics"]["energy_j"])
        assert close(summary["inherited_energy_j"], inherited)
        assert close(summary["suffix_energy_j"], metrics["energy_j"]-inherited)
        assert close(summary["suffix_energy_kwh"], (metrics["energy_j"]-inherited)/3600000.)
    curves = []
    for t in range(0, 601, 60):
        source = fixture["metrics"] if t == 0 else load(directory/f"summary_{t:04d}.json")["metrics"]
        assert t in state_boundaries, (str(directory), t)
        agrees = {key: close(source[key], state_boundaries[t][key]) if isinstance(source[key], (int, float))
                  else source[key] == state_boundaries[t][key]
                  for key in QUALITY if key in source and key in state_boundaries[t]}
        curves.append(dict(time=t, segment="V6_PREFIX" if branch and t <= boundary else "V7",
            metrics=selected(source), state_metrics_match=agrees,
            state_minus_summary={key: state_boundaries[t][key]-source[key] for key, match in agrees.items()
                if not match and isinstance(source[key], (int, float))}))
    all_planning = timing_and_planning(decisions, dispatch)
    timing_matches = {}
    for name, calculated in (("normal_timing", all_planning["normal_callback_wall"]),
                             ("dispatch_timing", all_planning["dispatch_components"]["dispatch_wall_s"])):
        timing_matches[name] = all(close(calculated[key], summary[name][key]) for key in ("count", "mean", "p50", "p95", "p99", "max", "over_5s"))
    assert all(timing_matches.values())
    assert all_planning["solver_phase_status"] == summary["cp_status_counts"]
    segments = dict(all_0_to_600=all_planning,
        inherited_prefix=timing_and_planning([row for row in decisions if row["time"] < boundary-1e-7],
            [row for row in dispatch if row["time"] < boundary-1e-7]),
        v7_suffix=timing_and_planning([row for row in decisions if row["time"] >= boundary-1e-7],
            [row for row in dispatch if row["time"] >= boundary-1e-7]))
    assert segments["inherited_prefix"]["decision_rows"] + segments["v7_suffix"]["decision_rows"] == len(decisions)
    if branch:
        assert segments["inherited_prefix"]["decision_rows"] == branch["methods"][method]["prefix_decisions_count"]
    preserved_audit = {key: value for key, value in audit.items() if key != "initial_fixture_baselines"}
    return dict(sources=dict(summary=str(directory/"summary_0600.json"), audit=str(directory/"audit_0600.json"),
            replay=str(replay), decisions=str(directory/"decisions.jsonl"), dispatch=str(directory/"dispatch_timings.jsonl")),
        metrics=metrics, cohorts=metrics["cohorts"],
        energy=dict(total_0_to_600_j=metrics["energy_j"], total_0_to_600_kwh=metrics["energy_kwh"],
            inherited_0_to_240_j=inherited, inherited_0_to_240_kwh=inherited/3600000.,
            new_suffix_j=metrics["energy_j"]-inherited, new_suffix_kwh=(metrics["energy_j"]-inherited)/3600000.,
            branch_boundary_s=boundary, summary_minus_audit_derived_costs=numeric_differences,
            definition="Cumulative modelled aircraft propulsion electricity. Inherited prefix is already included; never add it to the total again. Unspent SOC energy and ground-service charging are not included."),
        curves_every_60s=curves,
        planning_by_segment=segments,
        segment_definition="Decision/dispatch time <240 is inherited v6 prefix and time >=240 is v7 suffix for branches (1e-7 timestamp tolerance); medium has empty prefix and all calls in its v7 suffix. Boundary cumulative curves through 240 retain v6-prefix labels.",
        rescue_observations=summary["rescue_observations"], events=events,
        final_airborne=[selected(aircraft, ("id", "status", "ring", "slot", "soc", "peak_seconds", "energy_j", "touchdown_time"))
                        for aircraft in final_aircraft.values() if aircraft["status"] in AIRBORNE],
        failure_archive=load(replay/"failure_archive.json"),
        audit=preserved_audit,
        replay=dict(code_commit=index["code_commit"], states_count=index["states_count"], first_time=first["time"],
            last_time=last["time"], human_acceptance=index["human_acceptance"], visual_acceptance=visual),
        checks=dict(audit_counts_match_summary=count_matches, summary_minus_audit_derived_costs=numeric_differences,
            energy_j_kwh_units_match=True, event_touchdowns_match_summary=True, low_soc_touchdowns_match_summary=True,
            state_frame_count_matches_index_and_audit=True, strict_state_time_order=True,
            all_60s_boundaries_present=True,
            all_60s_state_metrics_match_summaries=all(all(row["state_metrics_match"].values()) for row in curves),
            boundary_state_vs_summary_mismatches=[dict(time=row["time"], fields=[key for key, match in row["state_metrics_match"].items() if not match],
                state_minus_summary=row["state_minus_summary"]) for row in curves if not all(row["state_metrics_match"].values())],
            timing_quantiles_match_summary=timing_matches, primary_key_matches_audit=True,
            raw_audit_status_and_issues_preserved=True))


def aggregate(base):
    result = dict(schema="stage2_v7_final_0600_facts_v1", evidence_type="existing_artifacts_read_only_aggregation",
        recorded_utc=datetime.now(timezone.utc).isoformat(), requested_boundary_s=600,
        interpretation="Stress/diagnostic are inherited-v6-0:240 plus v7-240:600; medium is v7 from zero. No simulation or audit rerun.",
        stress_viewer_uid_focus={"11": "initial_inner_08", "39": "req_000000"}, scenarios={})
    all_audits = []
    for scenario in SCENARIOS:
        root = base/f"stage2_rescue_need_v7_{scenario}_600s_20261023"
        campaign, budget = load(root/"campaign.json"), load(root/"budget.json")
        fixture, manifest = load(root/"fixture.json"), load(root/"manifest.json")
        branch = load(root/"branch.json") if (root/"branch.json").exists() else None
        exit_code = int((root/"exit_code.txt").read_text().strip())
        assert exit_code == 0 and campaign["common_completed_time_s"] == 600
        entry = dict(root=str(root), campaign=campaign, budget=budget, exit_code=exit_code,
            requested_cli_target_s=600, campaign_target_s_raw=campaign["target_s"],
            arrival_manifest_sha256=digest_file(root/"manifest.json"), frozen_3600s_request_count=len(manifest["requests"]),
            environment_lock_sha256=digest_file(root/"environment.lock.json"),
            policy_branch_metadata=branch,
            methods={method: one_method(root, method, fixture, {r["id"] for r in manifest["requests"]}, branch) for method in METHODS})
        if scenario != "medium":
            old = base/f"stage2_rescue_opportunity_v6_{scenario}_480s_20261022"
            comparison = {}
            for method in METHODS:
                new_metrics, old_metrics = load(root/method/"summary_0480.json")["metrics"], load(old/method/"summary_0480.json")["metrics"]
                comparison[method] = dict(new_v7_continuation=selected(new_metrics), old_v6=selected(old_metrics),
                    new_minus_old={key: new_metrics[key]-old_metrics[key] for key in QUALITY if key in new_metrics and key in old_metrics
                        and isinstance(new_metrics[key], (int, float)) and not isinstance(new_metrics[key], bool)},
                    sources=[str(root/method/"summary_0480.json"), str(old/method/"summary_0480.json")])
            entry["same_480s_v7_continuation_vs_v6"] = comparison
        entry["same_window_cp_minus_energy_slack"] = {}
        for boundary in (480, 600):
            left = load(root/"energy_slack"/f"summary_{boundary:04d}.json")["metrics"]
            right = load(root/"rh_cp_sat"/f"summary_{boundary:04d}.json")["metrics"]
            entry["same_window_cp_minus_energy_slack"][str(boundary)] = {key: right[key]-left[key] for key in QUALITY
                if key in right and key in left and isinstance(right[key], (int, float)) and not isinstance(right[key], bool)}
        for method, data in entry["methods"].items():
            audit = data["audit"]
            all_audits.append(dict(scenario=scenario, method=method, status=audit["status"],
                issue_count=audit["issue_count"], issues=audit["issues"], physical_outcome=audit["physical_outcome"],
                continuous_separation=audit["continuous_separation"]))
        result["scenarios"][scenario] = entry
    result["audit_rollup"] = dict(methods=all_audits, status_counts=dict(Counter(a["status"] for a in all_audits)),
        issue_count=sum(a["issue_count"] for a in all_audits),
        issue_codes=dict(Counter(issue["code"] for a in all_audits for issue in a["issues"])),
        event_types=dict(Counter(issue.get("event_type") for a in all_audits for issue in a["issues"])),
        interpretation="Audit status/issues are copied from existing artifacts without modification, diagnosis or conversion of FAIL to PASS.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    start, cpu_start = time.perf_counter(), time.process_time()
    result = aggregate(args.base)
    result["aggregation_wall_s"] = time.perf_counter()-start
    result["aggregation_cpu_s"] = time.process_time()-cpu_start
    args.output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n", encoding="utf-8")
    load(args.output)
    print(json.dumps(dict(output=str(args.output), bytes=args.output.stat().st_size,
        aggregation_wall_s=result["aggregation_wall_s"], audit_rollup=result["audit_rollup"],
        final_metrics={scenario: {method: selected(data["metrics"]) for method, data in entry["methods"].items()}
                       for scenario, entry in result["scenarios"].items()}), ensure_ascii=False))


if __name__ == "__main__":
    main()
