"""Explicit full-request denominators and censored outcomes."""
from dataclasses import asdict
from statistics import mean


def report(env):
    outcomes = []
    for a in env.aircraft.values():
        final = (a.state if a.state in ("normal_landed", "emergency_landed", "failed_energy") else
                 "not_arrived" if a.arrival_time > env.time else "censored")
        landed = final.endswith("landed")
        duration = a.completion_time-a.arrival_time if landed else None
        outcomes.append({"aircraft_id": a.aircraft_id, "arrival_time": a.arrival_time,
                         "admitted_time": a.admitted_time, "outcome": final, "state_at_cutoff": a.state,
                         "reason": a.reason if a.reason else ("observation_cutoff" if final == "censored" else None),
                         "initial_soc": a.initial_soc, "final_soc": a.soc, "minimum_soc": a.minimum_soc,
                         "energy_consumed_soc": a.consumed, "completion_time": a.completion_time,
                         "landing_duration_seconds": duration, "unimpeded_duration_seconds": a.unimpeded_duration,
                         "delay_seconds": duration-a.unimpeded_duration if landed else None,
                         "observed_request_seconds": max(0., min(env.time, a.completion_time if a.completion_time is not None else env.time)-a.arrival_time),
                         "planned_landing_time": a.plan.end if a.plan else None,
                         "emergency_planned_energy_soc": a.emergency_energy,
                         "emergency_reserved_resource_seconds": a.emergency_resource_seconds})
    counts = {state: sum(row["outcome"] == state for row in outcomes)
              for state in ("normal_landed", "emergency_landed", "failed_energy", "censored", "not_arrived")}
    n = len(outcomes)
    landed = [row for row in outcomes if row["outcome"].endswith("landed")]
    arrived = [row for row in outcomes if row["outcome"] != "not_arrived"]
    normal = [row for row in outcomes if row["outcome"] == "normal_landed"]
    emergency = [row for row in outcomes if row["outcome"] == "emergency_landed"]
    average = lambda key, rows: mean(row[key] for row in rows) if rows else None
    summary = {"request_count": n, "outcomes": counts, "conservation_ok": sum(counts.values()) == n,
               "arrived_request_count": len(arrived),
               "normal_proportion_all_requests": counts["normal_landed"]/n if n else 0.,
               "emergency_proportion_all_requests": counts["emergency_landed"]/n if n else 0.,
               "unserved_count": counts["failed_energy"]+counts["censored"],
               "observation_seconds": env.time,
               "throughput_landings_per_second": len(landed)/env.time if env.time else 0.,
               "normal_throughput_per_second": len(normal)/env.time if env.time else 0.,
               "emergency_throughput_per_second": len(emergency)/env.time if env.time else 0.,
               "normal_landing_metric_denominator": len(normal),
               "emergency_landing_metric_denominator": len(emergency),
               "normal_landing_duration_mean_seconds": average("landing_duration_seconds", normal),
               "normal_delay_mean_seconds": average("delay_seconds", normal),
               "normal_delay_max_seconds": max((row["delay_seconds"] for row in normal), default=None),
               "emergency_landing_duration_mean_seconds": average("landing_duration_seconds", emergency),
               "emergency_delay_mean_seconds_vs_normal_route": average("delay_seconds", emergency),
               "landing_metric_denominator": len(landed),
               "landing_duration_mean_seconds_conditional_landed": average("landing_duration_seconds", landed),
               "delay_mean_seconds_conditional_landed": average("delay_seconds", landed),
               "energy_mean_soc_all_requests": average("energy_consumed_soc", outcomes),
               "energy_mean_soc_arrived_requests": average("energy_consumed_soc", arrived),
               "observed_request_exposure_seconds": sum(row["observed_request_seconds"] for row in outcomes),
               "state_at_cutoff_counts": {state: sum(a.state == state for a in env.aircraft.values())
                                          for state in ("pending", "external", "holding", "scheduled", "normal_landed", "emergency_landed", "failed_energy")},
               "censored_state_counts": {state: sum(row["outcome"] == "censored" and row["state_at_cutoff"] == state for row in outcomes)
                                         for state in ("pending", "external", "holding", "scheduled")},
               "minimum_soc_all_requests": min((row["minimum_soc"] for row in outcomes), default=None),
               "separation_violation_pair_ticks": env.separation_violations,
               "spacing_uncertified_pair_ticks": env.spacing_uncertified,
               "infeasible_intent_count": env.infeasible_intents, "unknown_intent_count": env.unknown_intents,
               "online_decision_seconds": env.decision_seconds, "online_decision_calls": env.decision_calls,
               "search_candidates": env.search_candidates, "search_status_counts": env.search_status_counts,
               "admission_rejection_counts": env.admission_rejection_counts,
               "emergency_activation_count": sum(a.plan is not None and a.plan.emergency for a in env.aircraft.values()),
               "emergency_planned_energy_soc": sum(a.emergency_energy for a in env.aircraft.values()),
               "emergency_reserved_resource_seconds": sum(a.emergency_resource_seconds for a in env.aircraft.values())}
    return {"schema": "fixed-terminal-stage1-v1", "scheduler": "feasibility_exerciser_not_strong_baseline",
            "assumptions": {"calibration": "synthetic_research_defaults_not_hardware_calibrated",
                            "kinematics": "piecewise_arc_line; instantaneous_heading_changes; no_acceleration_model",
                            "external_queue": "abstract_boundary_wait; hold_SOC_rate; no_external_spatial_model",
                            "normal_domain": "SOC_bounded_all_ring_wait_vectors; current_commitments_fixed; others_hold",
                            "search_limit": "unknown_does_not_authorize_emergency",
                            "cutoff": "observation_only; plans_may_extend_beyond_horizon"},
            "config": asdict(env.config), "scenario": asdict(env.scenario),
            "capacity": [{"ring": i, "geometric": env.config.geometric_capacity(i),
                          "operational": env.config.operational_capacity(i)} for i in range(len(env.config.rings))],
            "summary": summary, "requests": outcomes}
