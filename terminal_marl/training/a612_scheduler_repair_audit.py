"""A6.12 prospective safe merge scheduling audit (diagnostic only, no PPO)."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import csv
import json
from pathlib import Path
import pickle

from ..core.config import SimConfig
from ..core.merge_scheduler import LowLevelIntent
from ..envs.parallel_env import TerminalParallelEnv
from ..policies import RuleLowLevelPolicy
from .a611_causal_audit import REQUIRED_SAFETY, SAFETY_REJECT_LIMIT, _base_config, demand_profiles, sustainable_rate, DemandProfile
from .a6_effectiveness import _collect, _sample


PROFILE_NAMES = ("constant_high", "low_high_low_ramp", "medium_burst", "long_burst", "double_burst")
INTERFACES = ("o2m", "m2i", "i2f")


def _hard_safe(metrics: dict, config: SimConfig) -> bool:
    return (all(metrics.get(name) == 0 for name in REQUIRED_SAFETY)
            and metrics.get("final_overcapacity_steps") == 0
            and metrics.get("max_concurrent_final", 0) <= config.final_queue_capacity)


def _enable_repair(env: TerminalParallelEnv) -> None:
    """Install one immutable config consistently after unpickling a frozen state."""
    config = replace(env.config, repair_aware_scheduler=True)
    env.config = config
    for item in (env.energy, env.priority, env.generator, env.admission, env.airspace, env.scheduler,
                 env.safety, env.separation_monitor, env.runtime_traffic):
        if hasattr(item, "config"):
            # EnergyModel is a frozen dataclass; this is a cloned, frozen
            # diagnostic state rather than a live A6.11 artifact.
            object.__setattr__(item, "config", config)


def _a611_mu0(seed: int) -> tuple[float, str]:
    """Prefer the frozen A6.11 calibration artifact over re-running it."""
    artifact = Path.home() / "training_outputs" / "terminal" / "event_gated_stage15_diagnostics" / "a611" / "a611_capacity_binding_audit.json"
    if artifact.exists():
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        if payload.get("seed") == seed:
            return float(payload["mu0"]["mu0_per_second"]), "preexisting_artifact"
    return float(sustainable_rate(seed)["mu0_per_second"]), "newly_run"


def _capture_a611_rejections(config: SimConfig, seed: int) -> list[dict]:
    """Stop immediately after the eight frozen pre-rejection payloads exist."""
    env = TerminalParallelEnv(config); env.reset(seed); policy = RuleLowLevelPolicy(); records: list[dict] = []
    while env.now < config.max_simulation_seconds and len(records) < SAFETY_REJECT_LIMIT:
        actions = policy.act(env._observations())
        interfaces = {uid: ("o2m" if uav.phase.value == "ring_outer" else "m2i" if uav.phase.value == "ring_middle" else "i2f")
                      for uid, uav in env.active.items() if uav.committed_plan is None and uav.phase.value in {"ring_outer", "ring_middle", "ring_inner"}}
        state = pickle.dumps(env, protocol=pickle.HIGHEST_PROTOCOL)
        _, _, _, _, info = env.step(actions)
        for raw_uid, decision in info["release_decisions"].items():
            uid = int(raw_uid)
            if decision == "safety_reject" and uid in interfaces and len(records) < SAFETY_REJECT_LIMIT:
                records.append({"payload": state, "uid": uid, "interface": interfaces[uid], "time": env.now - config.dt})
    return records


def frozen_rejection_regression(seed: int = 0) -> list[dict]:
    """Replay the A6.11 pre-rejection states against the online scheduler."""
    mu, _ = _a611_mu0(seed)
    # This is the same deterministic pilot profile and capture routine used in
    # A6.11; the payload is a state immediately before the rejected action.
    profile = DemandProfile("constant_high_pilot", ((0., 720., 1.55 * mu),), 720., 480.)
    records = _capture_a611_rejections(_base_config(profile, dynamic=False), seed)
    if len(records) != SAFETY_REJECT_LIMIT:
        raise RuntimeError(f"A6.12 requires {SAFETY_REJECT_LIMIT} frozen A6.11 rejections, found {len(records)}")
    rows: list[dict] = []
    for index, record in enumerate(records, 1):
        env: TerminalParallelEnv = pickle.loads(record["payload"]); _enable_repair(env)
        uav = env.active[record["uid"]]
        # The replay supplies the original rule action once.  A delayed safe
        # solution must become a single future reservation/trajectory, rather
        # than a sequence of trial actions at later timesteps.
        decision = env._release(uav, LowLevelIntent(1.0, 0.0, 0.0))
        if decision == "future_reserved":
            # Advance the deterministic scheduled-hold interval.  The
            # environment, not this audit, performs the sole future commit.
            for _ in range(int(max(env.config.prospective_repair_delays_seconds)) + 2):
                env.step()
                if uav.committed_plan is not None:
                    decision = "approved"; break
        telemetry = env.last_release_telemetry.get(uav.uid, {})
        reservation = env.scheduler.reservations.get(uav.merge_reservation_id) if uav.merge_reservation_id else None
        safe = env.safety.filter_plan(uav.committed_plan,
                                      {uid: item.committed_plan for uid, item in env.active.items() if item.uid != uav.uid and item.committed_plan is not None},
                                      type("Snapshot", (), {"now": env.now, "active": env.active, "airspace": env.airspace})()).approved if uav.committed_plan else False
        rows.append({"frozen_state": index, "time": record["time"], "interface": record["interface"],
                     "decision": decision, "repair_found": decision == "approved", "scheduled_delay": telemetry.get("scheduled_delay"),
                     "repair_used": telemetry.get("repair_used"), "candidate_speed": telemetry.get("applied_speed_residual"),
                     "candidate_merge": telemetry.get("applied_merge_time_residual"), "reservation_id": None if reservation is None else reservation.reservation_id,
                     "reservation_status": None if reservation is None else reservation.status, "plan_id": telemetry.get("trajectory_plan_id"),
                     "checked_equals_executed": bool(safe and telemetry.get("trajectory_plan_id") == getattr(uav.committed_plan, "plan_id", None)),
                     "hard_safe": safe})
    if not all(row["repair_found"] and row["hard_safe"] and row["checked_equals_executed"] for row in rows):
        failed = [{key: row.get(key) for key in ("frozen_state", "interface", "decision", "scheduled_delay", "repair_used", "reservation_status")} for row in rows if not row["repair_found"]]
        raise RuntimeError(f"A6.12 STOP: online scheduler failed an A6.11 frozen repair witness: {failed}")
    return rows


def _interface_summary(records: dict[int, dict], now: float) -> dict:
    result = {}
    for interface in INTERFACES:
        rows = [row for row in records.values() if row["interface"] == interface]
        approved = [row for row in rows if row["approval_time"] is not None]
        waits = sum(row["decision_history"].count("safety_wait") for row in rows)
        wait_to_approval = [row["approval_time"] - row["first_attempt_time"] for row in approved]
        result[interface] = {"unique_release_intents": len(rows), "approved": len(approved),
                             "approval_rate": len(approved) / max(1, len(rows)),
                             "safety_wait_rate": waits / max(1, sum(len(row["decision_history"]) for row in rows)),
                             "safety_reject_rate": sum(row["decision_history"].count("safety_reject") for row in rows) / max(1, len(rows)),
                             "retries_per_successful_transfer": sum(row["retry_count"] for row in approved) / max(1, len(approved)),
                             "wait_to_approved_release_mean": sum(wait_to_approval) / max(1, len(wait_to_approval)),
                             "wait_to_approved_release_p95": 0.0 if not wait_to_approval else float(__import__("numpy").percentile(wait_to_approval, 95))}
    return result


def _run_profile(config: SimConfig, seed: int) -> dict:
    env = TerminalParallelEnv(config); env.reset(seed); policy = RuleLowLevelPolicy()
    samples = {key: [] for key in ("queue", "oldest_wait", "final_util", "lz_util", "actual", "capacity", "all_waiting")}
    while env.now < config.max_simulation_seconds and not (env.arrivals.exhausted() and not env.active and not env.admission.queue.items and not env.service_uavs):
        actions = policy.act(env._observations()); env.step(actions); _sample(env, samples)
    metrics = _collect(env, samples)
    if not _hard_safe(metrics, config):
        protected = {key: metrics.get(key) for key in (*REQUIRED_SAFETY, "final_overcapacity_steps")}
        raise RuntimeError(f"A6.12 FAIL safety: seed={seed} metrics={protected}")
    return {"metrics": metrics, "interfaces": _interface_summary(env.release_intents, env.now),
            "release_intents": list(env.release_intents.values()), "hard_safe": True,
            "nominal_capacity": {role.value: env.airspace.capacity(role) for role in env.airspace._actual}}


def run_audit(*, seed: int = 0, stage: str = "screen", profiles_to_run: tuple[str, ...] | None = None) -> dict:
    frozen = frozen_rejection_regression(seed)
    if stage == "frozen":
        return {"stage": "A6.12 frozen-state regression", "evidence_type": "newly_run", "frozen_regression": frozen,
                "summary": {"protected_safety": "PASS", "ppo_authorization": "STOP"}}
    mu, mu_evidence = _a611_mu0(seed); profiles = demand_profiles(mu)
    rows: dict[str, dict] = {}
    # Strict fail-fast funnel: current baseline first, then repair-aware for
    # the same CRN profile.  No PPO/high-level policy is constructed here.
    selected_profiles = PROFILE_NAMES if profiles_to_run is None else profiles_to_run
    unknown = set(selected_profiles) - set(PROFILE_NAMES)
    if unknown: raise ValueError(f"unsupported A6.12 profiles: {sorted(unknown)}")
    for name in selected_profiles:
        profile = profiles[name]
        current_config = _base_config(profile, dynamic=False)
        repaired_config = replace(current_config, repair_aware_scheduler=True)
        current = _run_profile(current_config, seed)
        repaired = _run_profile(repaired_config, seed)
        rows[name] = {"current_scheduler": current, "repair_aware_scheduler": repaired}
    deltas = []
    for name, row in rows.items():
        for interface in INTERFACES:
            left, right = row["current_scheduler"]["interfaces"][interface], row["repair_aware_scheduler"]["interfaces"][interface]
            deltas.append({"profile": name, "interface": interface, "approval_rate_delta": right["approval_rate"] - left["approval_rate"],
                           "safety_wait_rate_delta": right["safety_wait_rate"] - left["safety_wait_rate"],
                           "retries_delta": right["retries_per_successful_transfer"] - left["retries_per_successful_transfer"]})
    improved_flow = any(row["repair_aware_scheduler"]["metrics"]["completed"] > row["current_scheduler"]["metrics"]["completed"] or
                        row["repair_aware_scheduler"]["metrics"]["all_request_waiting_area"] < row["current_scheduler"]["metrics"]["all_request_waiting_area"]
                        for row in rows.values())
    improved_approvals = any(row["approval_rate_delta"] > 0.01 for row in deltas)
    classification = "SCHEDULER_CONSERVATISM_CONFIRMED" if improved_approvals and improved_flow else ("SCHEDULER_REPAIR_NO_TERMINAL_GAIN" if improved_approvals else "MOST_REJECTIONS_TRULY_INFEASIBLE")
    return {"stage": "A6.12 prospective safe merge scheduling", "evidence_type": "newly_run", "seed": seed,
            "inputs": {"a611_mu0": mu, "a611_mu0_evidence_type": mu_evidence},
            "nominal_geometry_capacity": {"outer": 47, "middle": 31, "inner": 15}, "frozen_regression": frozen,
            "comparison": rows, "interface_deltas": deltas,
            "summary": {"protected_safety": "PASS", "classification": classification, "ppo_authorization": "STOP", "profiles_run": list(selected_profiles),
                        "recommend_a613_capacity_value_reaudit": classification == "SCHEDULER_CONSERVATISM_CONFIRMED"}}


def write_report(report: dict, output: str | Path) -> None:
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    (output / "a612_scheduler_repair_audit.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    with (output / "a612_frozen_rejection_regression.csv").open("w", newline="", encoding="utf-8") as handle:
        rows = report["frozen_regression"]; writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["frozen_state"]); writer.writeheader(); writer.writerows(rows)
    with (output / "a612_interface_flow_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["profile", "scheduler", "interface", "unique_release_intents", "approval_rate", "safety_wait_rate", "safety_reject_rate", "retries_per_successful_transfer", "wait_to_approved_release_mean", "wait_to_approved_release_p95"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for profile, conditions in report.get("comparison", {}).items():
            for scheduler, condition in conditions.items():
                for interface, row in condition["interfaces"].items(): writer.writerow({"profile": profile, "scheduler": scheduler, "interface": interface, **row})
    with (output / "a612_terminal_performance_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["profile", "scheduler", "completed", "external_queue_area", "all_request_waiting_area", "unresolved_count", "energy_mean", "priority_fairness_std", "holding_outer", "holding_middle", "holding_inner", "final_utilization", "lz_utilization", "hard_separation_violations", "nmac", "collisions", "final_overcapacity_steps"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for profile, conditions in report.get("comparison", {}).items():
            for scheduler, condition in conditions.items(): writer.writerow({"profile": profile, "scheduler": scheduler, **{key: condition["metrics"].get(key) for key in fields[2:]}})
    summary = report["summary"]
    lines = ["# A6.12 Prospective Safe Merge Scheduling & Repair Integration", "", f"Classification: **{summary.get('classification', 'FROZEN_REGRESSION')}**", f"Protected safety: **{summary['protected_safety']}**", "PPO authorization: **STOP**", "", "## Evidence", "", "- Evidence type: newly_run; deterministic common-random-number rule-policy simulations.", "- Fixed nominal geometry only; no high-level action, PPO, checkpoint, or safety-threshold change.", "- The frozen-state table reports whether all eight A6.11 witnesses are found online with a canonical checked plan.", "", "## Protocol risks", "", "- Initial screen is one seed per profile; multi-seed confirmation remains required before generalizing mechanism size.", "- The repair grid is finite within the existing ±2 m/s / ±6 s authority and bounded delay set; infeasible means infeasible within that legal grid."]
    (output / "a612_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, default=0); parser.add_argument("--stage", choices=("frozen", "screen"), default="screen"); parser.add_argument("--profiles", nargs="*", choices=PROFILE_NAMES, default=None); parser.add_argument("--output", default=None)
    args = parser.parse_args(); output = Path(args.output) if args.output else Path.home() / "training_outputs" / "terminal" / "event_gated_stage15_diagnostics" / "a612"
    report = run_audit(seed=args.seed, stage=args.stage, profiles_to_run=None if args.profiles is None else tuple(args.profiles)); write_report(report, output); print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
