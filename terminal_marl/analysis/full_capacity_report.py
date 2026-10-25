"""Descriptive full-world campaign analysis with optimistic necessary bounds.

Seed ranges are observations, not confidence intervals. Finite-cohort outcomes
and steady offered-rate bounds have distinct denominators and are never used
as an exceedance test against each other.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import asdict
from itertools import product
import json
from pathlib import Path
from statistics import mean

from terminal_marl.analysis.capacity_bounds import (
    finite_cohort_pad_window_upper, kinematic_touchdown_lower_bound,
    loss_adjusted_pad_upper,
)
from terminal_marl.analysis.full_service_capacity import (
    atomic_json, cohort_metrics, design_metadata, digest, make_config, profiles,
    validate_spec,
)
from terminal_marl.core.rotating_slots.config import digest as config_digest
from terminal_marl.core.rotating_slots.energy import stable_power

OUTCOMES = ("normal_touchdown", "emergency", "failure", "rerouted",
            "upstream_pending", "airborne_pending")
COUNTS = ("offered", *OUTCOMES, "normal_within_sla", "normal_reserve_ge_025",
          "normal_within_sla_and_reserve_ge_025")
METRICS = ("normal_fraction_all_offered", "ontime_reserve_fraction_all_offered",
           "normal_delay_p95_s", "observation_normal_touchdowns_per_h",
           "cohort_normal_touchdowns_per_h", "pad_busy_utilization",
           "upstream_drift_per_h", "airborne_drift_per_h", "wall_s", "cpu_s",
           "finite_integer_count_upper", "energy_identity_max_abs_residual_j",
           "holding_time_max_abs_residual_s", "audit_stable_holding_energy_j",
           "audit_derived_move_energy_j", "audit_outer_holding_s",
           "audit_middle_holding_s", "audit_inner_holding_s", *COUNTS,
           *(f"{name}_fraction_all_offered" for name in OUTCOMES))
KEY_FIELDS = ("profile", "rate_per_hour", "seed", "method")


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _events(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _key(params):
    return tuple(params[name] for name in KEY_FIELDS)


def _max_abs(values):
    return max((abs(value) for value in values), default=None)


def holding_energy_audit(aircraft, events, trajectories, requests, config, *,
                         final_time_s, failures=None):
    """Reconstruct executed HOLD residence; planned-only paths contribute zero.

    Completion events establish entry into HOLD. A later actually started
    maneuver establishes exit. INBOUND starts are evidenced by the engine's
    admitted event and matching admission/plan time; later starts use
    maneuver_started. Explicit failure timestamps truncate residence.
    Missing event evidence produces an omitted row and reason rather than a
    guessed flight/hold time. E_move is a residual, not an independent integral.
    """
    failures = failures or {}
    plans = trajectories if isinstance(trajectories, dict) else {
        row["trajectory_hash"]: row for row in trajectories}
    powers = {ring.id: stable_power(ring.id, config).battery_w for ring in config.airspace.rings}
    capacity = config.energy.reference_capacity_Ah * config.energy.equivalent_voltage_V * 3600.
    initial = {row["id"]: row["initial_soc"] for row in requests}
    starts, completions, failure_events = defaultdict(list), defaultdict(list), defaultdict(list)
    for event in events:
        key = event.get("aircraft_id")
        kind = event["type"]
        if kind in ("maneuver_started", "admitted"):
            starts[key].append(event)
        elif kind in ("inbound_completed", "transfer_completed"):
            completions[key].append(event)
        elif kind in ("aircraft_energy_failure", "failure_isolated"):
            failure_events[key].append(float(event["time"]))
    rows = []
    total_ring_s = {ring: 0. for ring in powers}
    for key, state in sorted(aircraft.items()):
        if not state.get("admitted", True):
            continue
        reasons = []
        failure_time = failures.get(key, {}).get("time")
        if failure_time is None and failure_events[key]:
            failure_time = min(failure_events[key])
        if (state.get("status") in ("ENERGY_MISSION_FAILED", "ENERGY_FAILURE")
                and failure_time is None):
            reasons.append("failure timestamp absent from archive and events")
        terminal = min(float(final_time_s), float(failure_time)) if failure_time is not None else float(final_time_s)
        executed = sorted(starts[key], key=lambda event: float(event["time"]))
        ring_s = {ring: 0. for ring in powers}
        for index, event in enumerate(executed):
            trajectory_hash = event.get("details", {}).get("trajectory_hash")
            plan = plans.get(trajectory_hash)
            if plan is None:
                reasons.append("started trajectory missing: " + str(trajectory_hash))
                continue
            if (plan.get("aircraft_id") != key
                    or abs(float(plan["start_time"]) - float(event["time"])) > 1.e-7):
                reasons.append("started trajectory aircraft/time mismatch: " + str(trajectory_hash))
                continue
            if event["type"] == "admitted" and (
                    plan.get("kind") != "INBOUND" or state.get("admission_time") is None
                    or abs(float(state["admission_time"]) - float(event["time"])) > 1.e-7):
                reasons.append("admitted event lacks matching INBOUND/admission-time evidence")
                continue
            ring = plan.get("target_ring")
            if ring is None or plan.get("kind") not in ("INBOUND", "TRANSFER"):
                continue
            end = float(plan["end_time"])
            if end > terminal + 1.e-7:
                continue
            completion_kind = "inbound_completed" if plan["kind"] == "INBOUND" else "transfer_completed"
            if not any(done["type"] == completion_kind and abs(float(done["time"]) - end) <= 1.e-7
                       for done in completions[key]):
                reasons.append("completion event missing for started trajectory: " + str(trajectory_hash))
                continue
            if ring not in powers:
                reasons.append("unknown target ring: " + str(ring))
                continue
            next_start = float(executed[index + 1]["time"]) if index + 1 < len(executed) else terminal
            if index + 1 == len(executed) and failure_time is None and not str(state.get("status", "")).startswith("HOLD"):
                reasons.append("last completed ring arrival has no later start or terminal HOLD/failure evidence")
                continue
            stop = min(next_start, terminal)
            if stop < end - 1.e-7:
                reasons.append("next maneuver precedes arrival completion")
                continue
            ring_s[ring] += max(0., stop - end)
        if key not in initial:
            reasons.append("initial request SOC missing")
        if reasons:
            rows.append(dict(id=key, status="omitted", reasons=sorted(set(reasons))))
            continue
        hold = sum(ring_s.values())
        hold_energy = sum(powers[ring] * seconds for ring, seconds in ring_s.items())
        energy = float(state.get("energy_j", 0.))
        drop = capacity * (initial[key] - float(state["soc"]))
        row = dict(id=key, status="audited", ring_holding_s=ring_s,
                   reconstructed_holding_s=hold, reported_holding_s=float(state.get("holding_time", 0.)),
                   holding_residual_s=hold - float(state.get("holding_time", 0.)),
                   stable_holding_energy_j=hold_energy, energy_j=energy,
                   derived_move_energy_j=energy - hold_energy,
                   battery_soc_drop_j=drop, battery_identity_residual_j=energy - drop,
                   terminal_time_s=terminal, failure_time_s=failure_time)
        rows.append(row)
        for ring, seconds in ring_s.items():
            total_ring_s[ring] += seconds
    audited = [row for row in rows if row["status"] == "audited"]
    return dict(per_aircraft=rows, audited_aircraft=len(audited), omitted_aircraft=len(rows) - len(audited),
                ring_holding_s=total_ring_s, stable_holding_power_w=powers,
                holding_time_max_abs_residual_s=_max_abs(row["holding_residual_s"] for row in audited),
                energy_identity_max_abs_residual_j=_max_abs(row["battery_identity_residual_j"] for row in audited),
                stable_holding_energy_j=sum(row["stable_holding_energy_j"] for row in audited),
                derived_move_energy_j=sum(row["derived_move_energy_j"] for row in audited),
                population="All admitted aircraft through final simulation time, including warmup arrivals.",
                interpretation="Executed-event residence reconstruction; move energy is total energy minus stable-hold estimate, not independent integration.")


def analyze_job(directory):
    """Read one completed immutable job and independently recompute its cohort."""
    directory = Path(directory)
    identity, result = _load(directory / "identity.json"), _load(directory / "results.json")
    if result.get("status") != "COMPLETED":
        raise ValueError("job result is not COMPLETED")
    params = identity["job"]
    if result.get("identity_hash") != digest(identity) or result.get("params") != params:
        raise ValueError("result identity/params mismatch")
    profile = profiles()[params["profile"]]
    config = make_config(profile)
    stored_config, manifest = _load(directory / "config.json"), _load(directory / "manifest.json")
    if (config.config_hash != identity["config_hash"] or config_digest(stored_config) != config.config_hash
            or stored_config != config.to_dict()):
        raise ValueError("stored config does not match reconstructed profile/hash")
    if digest(manifest) != identity["manifest_hash"] or manifest.get("profile") != profile.name:
        raise ValueError("manifest hash/profile mismatch")
    latent_hash = digest(manifest["latent_marks"])
    if latent_hash != manifest["latent_marks_hash"]:
        raise ValueError("latent marks hash mismatch")
    requests = manifest["manifest"]["requests"]
    events = _events(directory / "events.jsonl")
    aircraft, trajectories = _load(directory / "aircraft.json"), _load(directory / "trajectories.json")
    start, duration = params["warmup_s"], params["measurement_s"]
    end = start + duration
    failures = result["safety"].get("failures", {})
    cohort = cohort_metrics(requests, aircraft, events, start_s=start, end_s=end,
                            sla_delay_s=params["sla_delay_s"], failure_ids=failures)
    measured_requests = [row for row in requests if start <= row["release_time"] < end]
    lower = kinematic_touchdown_lower_bound(config)
    service = {name: float(value.duration_s) for name, value in config.pads.service_classes.items()}
    finite = finite_cohort_pad_window_upper(
        [row["release_time"] for row in measured_requests],
        [service[row["service_class"]] for row in measured_requests],
        sla_s=params["sla_delay_s"], pads=profile.pad_count,
        touchdown_lower_s=lower["touchdown_lower_s"],
        protected_s=lower["protected_descent_s"], grid_step_s=60.)
    rate_bounds = {f"{epsilon:.2f}": loss_adjusted_pad_upper(profile.service_seconds,
                   profile.service_probabilities, pads=profile.pad_count, epsilon=epsilon,
                   protected_s=lower["protected_descent_s"]) for epsilon in (0., .01, .05, .10)}
    boundaries = result["boundaries"]
    boundary_start = next(row for row in boundaries if row["label"] == "measurement_start")
    boundary_end = next(row for row in boundaries if row["label"] == "measurement_end")
    final_time = next(row["sim_time_s"] for row in boundaries if row["label"] == "final")
    audit = holding_energy_audit(aircraft, events, trajectories, requests, config,
                                 final_time_s=final_time, failures=failures)
    observation = result["observation_window"]
    offered = cohort["offered"]
    joint = cohort["normal_within_sla_and_reserve_ge_025"]
    stored_cohort_matches = all(cohort[field] == result["measurement_cohort"][field] for field in COUNTS)
    metadata = design_metadata(profile, config)
    inner = next(ring for ring in metadata["rings"] if ring["id"] == "inner")
    identity_residual = _max_abs(float(row["residual_j"]) for row in result.get("energy_identity", []))
    row = dict(**{name: params[name] for name in KEY_FIELDS}, job_directory=str(directory.resolve()),
        config_hash=config.config_hash, manifest_hash=identity["manifest_hash"],
        latent_marks_hash=latent_hash, source_hash=identity.get("source", {}).get("sha256"),
        input_identity_hash=digest(identity), config_hash_verified=True, manifest_hash_verified=True,
        inner_radius_m=profile.inner_radius_m, pad_count=profile.pad_count,
        service_mean_s=metadata["service_mean_s"], service_variance_s2=metadata["service_variance_s2"],
        soc_mean=profile.soc_mean, soc_std=profile.soc_std, soc_low=profile.soc_low, soc_high=profile.soc_high,
        inner_design_K=inner["design_K"], inner_stable_holding_power_w=inner["stable_holding_power_w"],
        demand_kind=params["demand_kind"], warmup_s=start, measurement_s=duration,
        drain_s=params["drain_s"], sla_delay_s=params["sla_delay_s"],
        **{name: cohort[name] for name in COUNTS},
        normal_fraction_all_offered=cohort["normal_touchdown"] / offered if offered else None,
        ontime_fraction_all_offered=cohort["normal_ontime_fraction_all_offered"],
        ontime_reserve_fraction_all_offered=cohort["normal_ontime_reserve_fraction_all_offered"],
        observed_joint_fraction_meets_095=(joint / offered >= .95) if offered else None,
        normal_delay_p95_s=cohort["normal_delay_quantiles_s"]["p95"],
        normal_touchdown_soc_unknown=cohort["normal_touchdown_soc_unknown"],
        observation_touchdowns_per_h=observation["touchdown_rate_per_h"],
        observation_normal_touchdowns_per_h=observation["normal_touchdown_rate_per_h"],
        cohort_normal_touchdowns_per_h=3600. * cohort["normal_touchdown"] / duration,
        pad_busy_utilization=observation["pad_busy_utilization"],
        upstream_start=boundary_start["upstream"], upstream_end=boundary_end["upstream"],
        airborne_start=boundary_start["airborne"], airborne_end=boundary_end["airborne"],
        upstream_drift_per_h=3600. * (boundary_end["upstream"] - boundary_start["upstream"]) / duration,
        airborne_drift_per_h=3600. * (boundary_end["airborne"] - boundary_start["airborne"]) / duration,
        population_conservation=cohort["population_conservation"],
        world_population_conservation=result["safety"]["conservation"], stored_cohort_matches=stored_cohort_matches,
        energy_identity_max_abs_residual_j=identity_residual,
        holding_time_max_abs_residual_s=audit["holding_time_max_abs_residual_s"],
        audit_omitted_aircraft=audit["omitted_aircraft"],
        audit_stable_holding_energy_j=audit["stable_holding_energy_j"],
        audit_derived_move_energy_j=audit["derived_move_energy_j"],
        finite_integer_count_upper=finite["integer_count_upper"],
        finite_relaxed_count_upper=finite["relaxed_count_upper"],
        finite_bound_observed_joint_count=joint,
        finite_bound_observed_count_le_upper=joint <= finite["integer_count_upper"],
        finite_bound_violation=joint > finite["integer_count_upper"],
        touchdown_lower_s=lower["touchdown_lower_s"], wall_s=result["wall_s"], cpu_s=result["cpu_s"])
    for name in OUTCOMES:
        row[f"{name}_fraction_all_offered"] = cohort[name] / offered if offered else None
    for epsilon, bound in rate_bounds.items():
        row[f"loss_{epsilon}_offered_rate_upper_per_h"] = bound["offered_rate_upper_per_h"]
    for ring, seconds in audit["ring_holding_s"].items():
        row[f"audit_{ring}_holding_s"] = seconds
    return dict(row=row, cohort=cohort, profile_parameters=asdict(profile),
                finite_cohort_bound={key: value for key, value in finite.items()
                                     if key not in ("dual_alpha", "dual_z", "eligible_indices")},
                loss_adjusted_bounds=rate_bounds, kinematic_lower_bound=lower, mechanism_audit=audit)


def aggregate_rows(rows):
    """Equal-weight seed mean and observed min/max; no confidence claim."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["profile"], row["rate_per_hour"], row["method"])].append(row)
    result = []
    for (profile, rate, method), values in sorted(grouped.items()):
        output = dict(profile=profile, rate_per_hour=rate, method=method,
                      number_seeds=len({row["seed"] for row in values}),
                      observed_seeds_meeting_joint_095=sum(row["observed_joint_fraction_meets_095"] is True for row in values),
                      seeds_with_defined_joint_fraction=sum(row["observed_joint_fraction_meets_095"] is not None for row in values))
        for field in METRICS:
            numbers = [float(row[field]) for row in values if row.get(field) is not None]
            output.update({f"{field}_mean": mean(numbers) if numbers else None,
                           f"{field}_min": min(numbers) if numbers else None,
                           f"{field}_max": max(numbers) if numbers else None,
                           f"{field}_n": len(numbers)})
        result.append(output)
    return result


def paired_differences(rows):
    """Only construct available same-input pairs; expose rejected/missing pairs."""
    lookup = {_key(row): row for row in rows}
    method_pairs, factor_pairs, unavailable = [], [], []
    for profile, rate, seed in sorted({(r["profile"], r["rate_per_hour"], r["seed"]) for r in rows}):
        left, right = (lookup.get((profile, rate, seed, method)) for method in ("energy_slack", "rh_cp_sat"))
        if left is None or right is None:
            unavailable.append(dict(kind="method", profile=profile, rate_per_hour=rate, seed=seed,
                                    reason="one or both method results absent"))
            continue
        matches = {name: bool(left.get(name)) and left.get(name) == right.get(name)
                   for name in ("manifest_hash", "config_hash", "source_hash")}
        matches.update({name: left.get(name) == right.get(name)
                        for name in ("warmup_s", "measurement_s", "drain_s", "sla_delay_s")})
        if not all(matches.values()):
            unavailable.append(dict(kind="method", profile=profile, rate_per_hour=rate, seed=seed,
                                    reason="input identity mismatch", matches=matches))
            continue
        pair = dict(profile=profile, rate_per_hour=rate, seed=seed,
                    difference="rh_cp_sat_minus_energy_slack", **{name + "_matches": value for name, value in matches.items()})
        for field in METRICS:
            a, b = left.get(field), right.get(field)
            pair[field + "_delta"] = b - a if a is not None and b is not None else None
        method_pairs.append(pair)
    for row in sorted(rows, key=_key):
        if row["profile"] == "half6":
            continue
        base = lookup.get(("half6", row["rate_per_hour"], row["seed"], row["method"]))
        matches = base is not None and bool(row.get("latent_marks_hash")) and base.get("latent_marks_hash") == row.get("latent_marks_hash")
        matches = matches and all(base.get(name) == row.get(name) for name in (
            "source_hash", "warmup_s", "measurement_s", "drain_s", "sla_delay_s"))
        if not matches:
            unavailable.append(dict(kind="factor", **{name: row[name] for name in KEY_FIELDS},
                                    reason="half6 baseline absent or latent marks differ"))
            continue
        pair = dict(**{name: row[name] for name in KEY_FIELDS}, difference="profile_minus_half6",
                    latent_marks_hash_matches=True,
                    interpretation="Raw matched endpoint difference; not a causal proof.")
        for field in METRICS:
            a, b = base.get(field), row.get(field)
            pair[field + "_delta"] = b - a if a is not None and b is not None else None
        factor_pairs.append(pair)
    def summarize(pairs, keys):
        groups = defaultdict(list)
        for pair in pairs:
            groups[tuple(pair[key] for key in keys)].append(pair)
        summaries = []
        for key, values in sorted(groups.items()):
            row = dict(zip(keys, key))
            row["number_paired_seeds"] = len({value["seed"] for value in values})
            for field in METRICS:
                numbers = [value[field + "_delta"] for value in values if value[field + "_delta"] is not None]
                for suffix, value in (("mean", mean(numbers) if numbers else None),
                                      ("min", min(numbers) if numbers else None),
                                      ("max", max(numbers) if numbers else None)):
                    row[f"{field}_delta_{suffix}"] = value
            summaries.append(row)
        return summaries
    return dict(method_cp_minus_es=method_pairs, factors_minus_half6=factor_pairs,
                method_difference_summary=summarize(method_pairs, ("profile", "rate_per_hour")),
                factor_difference_summary=summarize(factor_pairs, ("profile", "rate_per_hour", "method")),
                unavailable=unavailable)


def analyze_campaign(campaign_dir):
    campaign_dir = Path(campaign_dir)
    campaign = _load(campaign_dir / "campaign.json")
    spec = validate_spec(campaign["spec"])
    summary = _load(campaign_dir / "summary.json") if (campaign_dir / "summary.json").exists() else None
    expected = set(product(spec["profiles"], spec["rates_per_hour"], spec["seeds"], spec["methods"]))
    discovered, read_errors = defaultdict(list), []
    for directory in sorted((campaign_dir / "jobs").glob("*")):
        if not directory.is_dir():
            continue
        try:
            discovered[_key(_load(directory / "identity.json")["job"])].append(directory)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            read_errors.append(dict(directory=str(directory), error=str(exc)))
    duplicates = [{"key": list(key), "directories": [str(path) for path in paths]}
                  for key, paths in sorted(discovered.items()) if len(paths) > 1]
    analyses, failed, incomplete, missing, unexpected = [], [], [], [], []
    for key in sorted(expected):
        paths = discovered.get(key, [])
        if not paths:
            missing.append(list(key))
            continue
        if len(paths) != 1:
            continue
        directory = paths[0]
        if not (directory / "results.json").exists():
            incomplete.append(dict(key=list(key), directory=str(directory), status="NO_RESULT"))
            continue
        try:
            result = _load(directory / "results.json")
            job_params = _load(directory / "identity.json")["job"]
            for field in ("warmup_s", "measurement_s", "drain_s", "sla_delay_s", "progress_interval_s", "demand_kind"):
                if job_params.get(field) != spec[field]:
                    raise ValueError("job protocol differs from campaign spec: " + field)
            if result.get("status") != "COMPLETED":
                target = failed if result.get("status") in ("FAILED", "INVALID") else incomplete
                target.append(dict(key=list(key), directory=str(directory), status=result.get("status"),
                                   exception=result.get("exception")))
            else:
                analyses.append(analyze_job(directory))
        except (OSError, ValueError, KeyError, TypeError, StopIteration) as exc:
            failed.append(dict(key=list(key), directory=str(directory), status="ANALYSIS_INPUT_ERROR", error=str(exc)))
    for key in sorted(set(discovered) - expected):
        unexpected.append(dict(key=list(key), directories=[str(path) for path in discovered[key]]))
    summary_rows = summary.get("jobs", []) if summary else []
    summary_duplicates = sum(count - 1 for count in Counter(row.get("directory") for row in summary_rows).values() if count > 1)
    rows = [analysis["row"] for analysis in analyses]
    coverage = dict(expected_jobs=len(expected), complete_jobs=len(rows), failed_jobs=len(failed),
                    missing_jobs=len(missing), incomplete_jobs=len(incomplete),
                    duplicate_jobs=sum(len(row["directories"]) - 1 for row in duplicates),
                    duplicate_keys=len(duplicates), unexpected_jobs=sum(len(row["directories"]) for row in unexpected),
                    unreadable_jobs=len(read_errors), summary_present=summary is not None,
                    summary_duplicate_jobs=summary_duplicates,
                    summary_expected_jobs=summary.get("expected_jobs") if summary else None,
                    summary_finished_jobs=summary.get("finished_jobs") if summary else None,
                    summary_completed_rows=sum(row.get("status") == "COMPLETED" for row in summary_rows),
                    summary_all_completed=summary.get("all_completed", False) if summary else False)
    coverage["all_completed"] = (len(rows) == len(expected) and not any((failed, missing, incomplete, duplicates,
        unexpected, read_errors, summary_duplicates)) and summary is not None and summary.get("all_completed") is True
        and summary.get("expected_jobs") == len(expected) and summary.get("finished_jobs") == len(expected)
        and len(summary_rows) == len(expected) and all(row.get("status") == "COMPLETED" for row in summary_rows))
    return dict(schema="full_capacity_report_v1", campaign_directory=str(campaign_dir.resolve()),
                spec=spec, coverage=coverage, jobs=analyses, per_job=rows,
                per_profile_rate_method=aggregate_rows(rows), paired=paired_differences(rows),
                issues=dict(failed=failed, missing=missing, incomplete=incomplete, duplicates=duplicates,
                            unexpected=unexpected, unreadable=read_errors),
                interpretation=["Seed mean/min/max are descriptive observations, not confidence intervals.",
                    "The 0.95 threshold counts observed per-seed joint on-time and reserve fractions; it is not a capacity estimate.",
                    "Finite-cohort count bounds are compared only to that cohort's observed joint count.",
                    "Steady offered-rate bounds are not exceedance tests for drained cohort throughput.",
                    "A solver's catalog objective or bound is not a global terminal-capacity certificate."])


def _csv(path, rows):
    fields = sorted({field for row in rows for field in row})
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(report, output, *, figures=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    _csv(output / "per_job.csv", report["per_job"])
    atomic_json(output / "per_job.json", report["per_job"])
    _csv(output / "per_profile_rate_method.csv", report["per_profile_rate_method"])
    _csv(output / "paired_methods.csv", report["paired"]["method_cp_minus_es"])
    _csv(output / "paired_factors.csv", report["paired"]["factors_minus_half6"])
    _csv(output / "paired_methods_summary.csv", report["paired"]["method_difference_summary"])
    _csv(output / "paired_factors_summary.csv", report["paired"]["factor_difference_summary"])
    if figures:
        report["figures"] = plot_report(report, output / "figures")
    atomic_json(output / "report.json", report)
    return report


def plot_report(report, output):
    """Plot only completed observations; ranges are observed seed extrema."""
    if report["spec"]["demand_kind"] == "single":
        return dict(status="skipped", reason="Single-request qualification uses a placeholder rate, not a traffic rate.")
    if not report["per_job"]:
        return dict(status="skipped", reason="No verified completed jobs to plot.")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "svg.fonttype": "none", "pdf.fonttype": 42})
    colors = ("#52799B", "#B37A5B", "#71917D", "#967BA4", "#B09C62", "#65767B")
    markers = ("o", "s", "^", "D", "v", "P", "X", "<")
    grouped = report["per_profile_rate_method"]
    plot_data = []
    for row in grouped:
        for metric in ("ontime_reserve_fraction_all_offered", "pad_busy_utilization", *
                       (f"{name}_fraction_all_offered" for name in OUTCOMES)):
            plot_data.append(dict(profile=row["profile"], method=row["method"], rate_per_hour=row["rate_per_hour"],
                metric=metric, mean=row[metric + "_mean"], minimum=row[metric + "_min"],
                maximum=row[metric + "_max"], number_seeds=row["number_seeds"]))
    geometry = {row["inner_radius_m"]: dict(inner_radius_m=row["inner_radius_m"],
                design_K=row["inner_design_K"], stable_power_kw=row["inner_stable_holding_power_w"] / 1000.)
                for row in report["per_job"]}
    _csv(output / "plot_data.csv", plot_data)
    _csv(output / "geometry_plot_data.csv", [geometry[key] for key in sorted(geometry)])
    files = []
    for method in sorted({row["method"] for row in grouped}):
        method_rows = [row for row in grouped if row["method"] == method]
        names = sorted({row["profile"] for row in method_rows})
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        for index, name in enumerate(names):
            values = sorted((row for row in method_rows if row["profile"] == name), key=lambda row: row["rate_per_hour"])
            for axis, metric in ((axes[0, 0], "ontime_reserve_fraction_all_offered"), (axes[0, 1], "pad_busy_utilization")):
                defined = [row for row in values if row[metric + "_mean"] is not None]
                x = [row["rate_per_hour"] for row in defined]
                axis.plot(x, [row[metric + "_mean"] for row in defined], label=name,
                          color=colors[index % len(colors)], marker=markers[index % len(markers)], linewidth=1.2, markersize=4)
                axis.fill_between(x, [row[metric + "_min"] for row in defined],
                                  [row[metric + "_max"] for row in defined], color=colors[index % len(colors)], alpha=.12)
        axes[0, 0].axhline(.95, color="0.4", linestyle="--", linewidth=.8)
        axes[0, 0].set(title="(a) On-time normal touchdown with reserve", ylabel="Fraction of all offered requests", ylim=(0., 1.02))
        axes[0, 1].set(title="(b) Physical pad busy utilization", ylabel="Busy pad time / available pad time", ylim=(0., max(1.02, max(row["pad_busy_utilization_max"] for row in method_rows) * 1.02)))
        selected = "half6" if "half6" in names else names[0]
        reference = sorted((row for row in method_rows if row["profile"] == selected), key=lambda row: row["rate_per_hour"])
        for index, outcome in enumerate(OUTCOMES):
            metric = outcome + "_fraction_all_offered"
            defined = [row for row in reference if row[metric + "_mean"] is not None]
            axes[1, 0].plot([row["rate_per_hour"] for row in defined], [row[metric + "_mean"] for row in defined],
                            color=colors[index], marker=markers[index], label=outcome.replace("_", " "), markersize=4)
        axes[1, 0].set(title=f"(c) Cohort outcomes: {selected}", ylabel="Fraction of all offered requests", ylim=(0., 1.02))
        geom = [geometry[key] for key in sorted(geometry)]
        axes[1, 1].plot([row["inner_radius_m"] for row in geom], [row["design_K"] for row in geom],
                        color=colors[0], marker="o", label="Design K")
        twin = axes[1, 1].twinx()
        twin.plot([row["inner_radius_m"] for row in geom], [row["stable_power_kw"] for row in geom],
                  color=colors[1], marker="s", linestyle="--", label="Stable power")
        axes[1, 1].set(title="(d) Configured inner-ring geometry and power", xlabel="Inner radius (m)",
                       ylabel="Configured slots", ylim=(0., max(row["design_K"] for row in geom) * 1.15))
        twin.set(ylabel="Stable holding power (kW)", ylim=(0., max(row["stable_power_kw"] for row in geom) * 1.15))
        for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
            axis.set_xlabel("Offered rate (aircraft/h)")
            axis.set_xlim(left=0.)
            axis.legend(fontsize=7, ncol=2, frameon=False)
            axis.grid(alpha=.2)
        fig.suptitle(f"{method}: means and observed seed min-max; no confidence intervals\nSeed counts and all numerical points: plot_data.csv", fontsize=11)
        for suffix in ("svg", "png"):
            path = output / f"full_capacity_{method}.{suffix}"
            fig.savefig(path, dpi=300)
            files.append(str(path.resolve()))
        plt.close(fig)
    contract = {"Core Conclusion": "Descriptive factor and method observations; no certified capacity claim.",
                "Evidence Hierarchy": "Verified completed job outputs; equal-weight seed means and observed extrema.",
                "Figure Archetype": "quantitative grid", "Panel Mapping": ["joint success", "pad utilization", "cohort outcomes", "configured geometry/power"],
                "Reviewer Risk": "Finite horizons, seed counts, incomplete campaign coverage, optimistic bounds.",
                "Export Bundle": ["SVG with editable text", "300 dpi PNG", "plot_data.csv", "geometry_plot_data.csv"],
                "caption": "Bands denote observed seed minima and maxima, not confidence intervals. Seed counts are in plot_data.csv; no significance test is shown."}
    qa = dict(real_completed_inputs_only=True, source_data_exported=True, svg_text_editable=True,
              seed_ranges_labeled=True, axes_include_zero=True, raster_dpi=300,
              manual_visual_review="required on generated actual-result figures before publication")
    atomic_json(output / "figure_contract.json", contract)
    atomic_json(output / "figure_qa.json", qa)
    return dict(status="generated", files=files, contract=contract, qa=qa)
