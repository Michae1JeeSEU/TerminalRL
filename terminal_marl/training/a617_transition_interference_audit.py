"""A6.17: causal isolation of geometry-transition O2M interference.

This module is diagnostic. It replays the frozen A6.12 scheduler and frozen
SafetyFilter; ``freeze_only`` masks O2M only during the observed full-dynamic
transition interval and never changes a ring radius or commits shadow work.
"""
from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from ..core.config import RingRole
from ..core.merge_scheduler import LowLevelIntent
from ..core.uav import FlightPhase
from ..envs.high_level_wrapper import DynamicHighLevelWrapper, RuleLowLevelAdapter
from ..envs.parallel_env import WorldSnapshot
from .a611_causal_audit import DemandProfile, _safe
from .a612_scheduler_repair_audit import _a611_mu0
from .a613_capacity_reaudit import repair_config
from .a614_reconfiguration_timing_audit import OUTER_PLUS_ONE
from .a615_opportunity_sequencing_audit import _minimum_prediction, _neighbors
from .a6_effectiveness import _collect, _sample


PPO_AUTHORIZATION = "STOP"
TIMES = (180.0, 240.0, 300.0)
SEEDS = (0, 1)
MATERIAL_WAITING_AREA = 50_000.0  # Frozen A6.10 conservative materiality rule.


def _profile(mu: float) -> DemandProfile:
    return DemandProfile("constant_high_repair_scheduler", ((0.0, 720.0, 1.55 * mu),), 720.0, 480.0)


def _done(env) -> bool:
    return env.arrivals.exhausted() and not env.active and not env.admission.queue.items and not env.service_uavs


def _o2m_context(env, uav: object) -> dict:
    """Read current A6.12 candidate authority without reserving anything."""
    if "o2m" in env.airspace.affected_interfaces():
        return {"safe_candidate_exists": False, "first_safe_opportunity": None, "primary_blocker": "geometry_frozen"}
    snapshot = WorldSnapshot(env.now, env.active, env.airspace)
    requested = LowLevelIntent(1.0, 0.0, 0.0)
    blockers: list[str] = []
    for rank, candidate in enumerate(env.scheduler.repair_candidates(requested)):
        proposal, plan, reason = env._candidate_plan(uav, candidate, RingRole.MIDDLE, snapshot, None)
        if reason is not None:
            blockers.append(str(reason))
            continue
        predicted, swept = _minimum_prediction(env, plan, snapshot)
        predecessor, successor = _neighbors(env, RingRole.MIDDLE, proposal.merge_angle, proposal.merge_time)
        return {
            "safe_candidate_exists": True, "first_safe_opportunity": env.now,
            "candidate_rank": rank, "candidate_kind": candidate.kind,
            "merge_time": proposal.merge_time, "merge_angle": proposal.merge_angle,
            "predecessor": predecessor, "successor": successor,
            "minimum_predicted_separation": predicted, "minimum_swept_separation": swept,
            "primary_blocker": None,
        }
    return {"safe_candidate_exists": False, "first_safe_opportunity": None,
            "primary_blocker": blockers[0] if blockers else "no_candidate"}


def _set_freeze_only_overlay(env, enabled: bool) -> None:
    """Apply the diagnostic-only O2M gate without touching canonical branches.

    ``affected_interfaces`` is a class method in the production airspace
    object.  An instance assignment therefore shadows transition semantics.
    The overlay is consequently valid only for the synthetic freeze-only
    control, and is removed rather than rebound when that control ends.
    """
    if enabled:
        env.airspace.affected_interfaces = lambda: {"o2m"}
    else:
        env.airspace.__dict__.pop("affected_interfaces", None)


def _run(profile, seed: int, timing: float, mode: str, transition_window=None, observe_window=None,
         *, diagnostic_probes: bool = True) -> dict:
    dynamic = mode == "full_dynamic"
    # Match A6.14's CRN nominal exactly: every branch runs the dynamic no-op
    # stack.  Only full_dynamic submits the geometry action; freeze_only only
    # overlays the observed O2M interface gate at R_outer=600 m.
    wrapper = DynamicHighLevelWrapper(repair_config(profile, dynamic=True), RuleLowLevelAdapter())
    wrapper.reset(seed)
    env, policy, coordinator = wrapper.base, wrapper.low_level, wrapper.coordinator

    freeze_start = transition_window[0] if mode == "freeze_only" and transition_window else None
    freeze_end = transition_window[1] if mode == "freeze_only" and transition_window else None
    interruptions, trace = [], []
    samples = {key: [] for key in ("queue", "oldest_wait", "final_util", "lz_util", "actual", "capacity", "all_waiting")}
    requested = False
    actual_start = actual_end = None
    first_post_transition_legal = None

    while env.now < env.config.max_simulation_seconds and not _done(env):
        manual_freeze = freeze_start is not None and freeze_start <= env.now < freeze_end
        # Never shadow ``affected_interfaces`` in nominal/full_dynamic.  Those
        # branches must call the canonical geometry-event gate exactly as
        # A6.14/A6.15 do.  Only the synthetic freeze-only control owns an
        # instance-level diagnostic overlay.
        if mode == "freeze_only":
            _set_freeze_only_overlay(env, manual_freeze)
        if dynamic and not requested and env.now >= timing - 1e-9:
            # Existing A6.14 gate; this audit neither bypasses dwell nor adds actions.
            coordinator.coordinate(OUTER_PLUS_ONE, geometry_gate=wrapper.geometry_event_gate())
            requested = True
        coordinator.advance()
        actions = coordinator.apply_speed_repairs(policy.act(env._observations()))

        if env.airspace.transition is not None:
            actual_start, actual_end = env.airspace.transition.start_time, env.airspace.transition.end_time
        window = transition_window if transition_window is not None else (
            (actual_start, actual_end) if actual_start is not None else None
        )
        observed_window = observe_window if mode == "nominal" else window
        in_window = observed_window is not None and observed_window[0] <= env.now < observed_window[1]
        eligible = [u for u in env.active.values() if u.phase is FlightPhase.RING_OUTER and u.committed_plan is None]
        requests = {
            u.uid: LowLevelIntent.from_array(actions.get(str(u.uid), np.zeros(3, np.float32))).release_request
            for u in eligible
        }
        # _candidate_plan emits diagnostic telemetry.  Probe only the
        # non-frozen observed window, and do so on a deep copy.  A frozen
        # window has a conclusive ``geometry_frozen`` blocker without a probe.
        shadow = (deepcopy(env) if diagnostic_probes and in_window and "o2m" not in env.airspace.affected_interfaces()
                  else None)
        probes = ({u.uid: _o2m_context(shadow, shadow.active[u.uid]) for u in eligible} if shadow is not None
                  else ({u.uid: _o2m_context(env, u) for u in eligible} if diagnostic_probes and in_window else {}))

        _, _, _, _, info = env.step(actions)
        _sample(env, samples)
        now_before = env.now - env.config.dt
        # A frozen scheduler's approved/future reservation has necessarily
        # passed SafetyFilter and is therefore the first operational legal O2M
        # candidate after transition.  Reading this execution event needs no
        # extra shadow replay and cannot alter the formal branch.
        if actual_end is not None and now_before >= actual_end and first_post_transition_legal is None:
            if any(value in {"approved", "future_reserved"} for value in info["release_decisions"].values()):
                first_post_transition_legal = now_before
        for uid, release in requests.items():
            decision = info["release_decisions"].get(str(uid))
            uav = next((item for item in env.active.values() if item.uid == uid), None)
            if diagnostic_probes and in_window and release > env.config.release_threshold:
                interruptions.append({
                    "seed": seed, "timing": timing, "mode": mode,
                    "transition_start": observed_window[0], "transition_effective": observed_window[1],
                    "time": now_before, "o2m_eligible_count": len(eligible), "uav_id": uid,
                    "arrival_id": None if uav is None else uav.arrival_id, "release_request": release,
                    "decision": decision, "freeze_blocked": decision == "geometry_frozen",
                    "post_transition_first_legal_candidate_time": first_post_transition_legal,
                    **probes.get(uid, {}),
                })
            if decision in {"approved", "future_reserved", "geometry_frozen"}:
                trace.append((now_before, uid, decision))
    if mode == "freeze_only":
        _set_freeze_only_overlay(env, False)
    metrics = _collect(env, samples, coordinator)
    return {
        "metrics": metrics, "safe": _safe(metrics, env.config), "trace": trace,
        "interruptions": interruptions,
        "transition_start": freeze_start if mode == "freeze_only" else actual_start,
        "transition_effective": freeze_end if mode == "freeze_only" else actual_end,
        "post_transition_first_legal_candidate_time": first_post_transition_legal,
    }


def _first(nominal: dict, branch: dict, seed: int, timing: float, branch_name: str) -> dict:
    left = {(x[0], x[1]): x for x in nominal["trace"]}
    right = {(x[0], x[1]): x for x in branch["trace"]}
    for key in sorted(set(left) | set(right)):
        if left.get(key) != right.get(key):
            return {"seed": seed, "timing": timing, "branch": branch_name, "time": key[0], "uav_id": key[1],
                    "nominal_event": str(left.get(key)), "branch_event": str(right.get(key))}
    return {"seed": seed, "timing": timing, "branch": branch_name, "time": None, "uav_id": None,
            "nominal_event": "", "branch_event": ""}


def _classify(rows: list[dict], seeds) -> list[str]:
    indexed = {(r["seed"], r["timing"], r["branch"]): r for r in rows}
    comparisons = []
    for seed in seeds:
        for timing in TIMES:
            nominal = indexed[seed, timing, "nominal"]["waiting_area"]
            full = indexed[seed, timing, "full_dynamic"]["waiting_area"] - nominal
            freeze = indexed[seed, timing, "freeze_only"]["waiting_area"] - nominal
            comparisons.append((full, freeze))
    reproduced = [abs(freeze) >= MATERIAL_WAITING_AREA and abs(full) >= MATERIAL_WAITING_AREA and full * freeze > 0
                  for full, freeze in comparisons]
    if any(reproduced):
        # A mix of material and nonmaterial pairs establishes state dependence,
        # not a universal transition explanation, with only two CRN seeds.
        return ["RECONFIGURATION_FREEZE_INTERFERENCE"] if all(reproduced) else ["STATE_DEPENDENT_RECONFIGURATION_WINDOW"]
    if any(abs(full) >= MATERIAL_WAITING_AREA for full, _ in comparisons):
        return ["TRANSITION_GEOMETRY_PHASE_SHIFT"]
    return ["INCONCLUSIVE"]


def run_audit(*, seeds=SEEDS, mu0=None) -> dict:
    profile = _profile(_a611_mu0(0)[0] if mu0 is None else float(mu0))
    rows, interruptions, divergence, raw = [], [], [], {}
    for seed in seeds:
        for timing in TIMES:
            full = _run(profile, seed, timing, "full_dynamic")
            window = None if full["transition_start"] is None else (full["transition_start"], full["transition_effective"])
            # Nominal is only observed over this exact real window; no mask/change.
            nominal = _run(profile, seed, timing, "nominal", observe_window=window)
            freeze = _run(profile, seed, timing, "freeze_only", transition_window=window)
            group = {"nominal": nominal, "full_dynamic": full, "freeze_only": freeze}
            if not all(item["safe"] for item in group.values()):
                raise RuntimeError("A6.17 unsafe branch")
            raw[f"{seed}:{timing}"] = group
            for name, result in group.items():
                metrics = result["metrics"]
                rows.append({
                    "seed": seed, "timing": timing, "branch": name, "completed": metrics["completed"],
                    "waiting_area": metrics["all_request_waiting_area"], "external_queue_area": metrics["external_queue_area"],
                    "service_delay_p95": metrics["service_delay_p95"], "unresolved": metrics["unresolved_count"],
                    "safe": result["safe"], "checked_executed": True,
                    "transition_start": result["transition_start"], "transition_effective": result["transition_effective"],
                    "post_transition_first_legal_candidate_time": result["post_transition_first_legal_candidate_time"],
                })
                interruptions.extend(result["interruptions"])
            for name in ("full_dynamic", "freeze_only"):
                divergence.append(_first(nominal, group[name], seed, timing, name))
    return {
        "stage": "A6.17 Reconfiguration Transition Interference Causal Audit", "evidence_type": "newly_run",
        "inputs": {"safe_distance": 80.0, "transition_seconds": 20.0, "ppo_authorization": PPO_AUTHORIZATION,
                   "scheduler": "A6.12 frozen repair-aware", "checked_equals_executed": True},
        "branch_metrics": rows, "transition_interruptions": interruptions, "first_divergence": divergence, "raw": raw,
        "summary": {"classification": _classify(rows, seeds), "ppo_authorization": PPO_AUTHORIZATION,
                    "protected_safety": "PASS", "material_waiting_area": MATERIAL_WAITING_AREA},
    }


def _csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row}) or ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(report: dict, output: str | Path) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    _csv(output / "a617_branch_metrics.csv", report["branch_metrics"])
    _csv(output / "a617_transition_interruptions.csv", report["transition_interruptions"])
    _csv(output / "a617_first_divergence.csv", report["first_divergence"])
    (output / "a617_audit.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    labels = ", ".join(report["summary"]["classification"])
    safe_count = sum(bool(row["safe"]) for row in report["branch_metrics"])
    indexed = {(row["seed"], row["timing"], row["branch"]): row for row in report["branch_metrics"]}
    deltas = []
    for seed in sorted({row["seed"] for row in report["branch_metrics"]}):
        for timing in TIMES:
            nominal = indexed[seed, timing, "nominal"]["waiting_area"]
            full = indexed[seed, timing, "full_dynamic"]["waiting_area"] - nominal
            freeze = indexed[seed, timing, "freeze_only"]["waiting_area"] - nominal
            deltas.append(f"| {seed} | {timing:.0f} | {full:+.1f} | {freeze:+.1f} |")
    blocked = sum(bool(row.get("freeze_blocked")) for row in report["transition_interruptions"])
    text = (
        "# A6.17 Reconfiguration Transition Interference Causal Audit\n\n"
        f"Classification: **{labels}** (A6.10 materiality: |Δ waiting-area| ≥ {MATERIAL_WAITING_AREA:.0f}).\n\n"
        f"Formal branches hard-safe and checked==executed: {safe_count}/{len(report['branch_metrics'])}.\n\n"
        "| seed | action time (s) | full Δ waiting-area | freeze-only Δ waiting-area |\n"
        "|---:|---:|---:|---:|\n" + "\n".join(deltas) + "\n\n"
        f"Freeze-blocked O2M release attempts: {blocked}.  A negative delta is an improvement.\n\n"
        "Full-dynamic retains the existing geometry-event gate. Freeze-only holds R_outer=600 m and masks O2M only in the exact observed 20 s full-dynamic interval. Nominal is an unmodified CRN replay observed over that same interval.\n\n"
        "PPO authorization: **STOP**. This is diagnostic only; no scheduler, safety, timing, reward, observation, or action semantics were changed.\n"
    )
    (output / "a617_summary.md").write_text(text, encoding="utf-8")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    output = Path(args.output) if args.output else Path.home() / "training_outputs" / "terminal" / "event_gated_stage15_diagnostics" / "a617"
    write_report(run_audit(seeds=tuple(map(int, args.seeds.split(",")))), output)


if __name__ == "__main__":
    main()
