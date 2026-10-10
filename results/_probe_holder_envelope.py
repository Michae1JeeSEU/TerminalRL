"""T0 holder candidate envelope probe; never advances an episode."""
from dataclasses import replace
import json
from pathlib import Path

from terminal_marl.core.config import RingRole
from terminal_marl.core.safety_filter import SafetyFilter
from terminal_marl.envs.parallel_env import TerminalParallelEnv, WorldSnapshot
from terminal_marl.training.diagnose_landing_flow import _build_diagnostic_config

OUT = Path("results/holder_envelope_probe_841301.json")


def check(filter_, plan, snapshot):
    result = filter_.filter_plan(plan, {}, snapshot)
    return {"approved": bool(result.approved), "reason": result.reason,
            "checked_plan_id": result.checked_plan_id,
            "checked_trajectory_hash": result.checked_trajectory_hash,
            "min_track_audit": dict(filter_.last_predicted_track_audit)}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    config = _build_diagnostic_config(True)
    env = TerminalParallelEnv(config)
    env.reset(seed=841301)
    flow = env.flow_coordinator
    snapshot = WorldSnapshot(now=env.now, active=env.active, airspace=env.airspace)
    baseline_config = replace(config, rule_flow_coordination_enabled=False,
                              max_speed_residual_mps=2.0)
    nominal_config = replace(baseline_config, max_speed_residual_mps=0.0)
    baseline_filter = SafetyFilter(baseline_config)
    nominal_filter = SafetyFilter(nominal_config)
    durations = {RingRole.OUTER: (48.0, 60.0, 72.0, 84.0),
                 RingRole.MIDDLE: (60.0, 72.0, 84.0, 96.0)}
    rows = []
    for uid in (8, 3, 4, 15):
        uav = env.active.get(uid)
        base = {"uid": uid, "time": env.now, "present": uav is not None}
        if uav is None:
            rows.append(base); continue
        role = uav.ring_role
        base.update({"source_ring": None if role is None else role.value,
                     "phase": uav.phase.value, "soc": float(uav.soc),
                     "energy_remaining": float(uav.energy_remaining)})
        if role not in durations:
            base["skipped"] = "no_adjacent_ring_target"
            rows.append(base); continue
        target = RingRole.MIDDLE if role is RingRole.OUTER else RingRole.INNER
        base["target_ring"] = target.value
        for duration in durations[role]:
            row = dict(base, duration=duration)
            try:
                plan = flow.trajectory.ring_transfer(uav, target, env.airspace,
                                                     env.now, duration)
                row.update({"plan_id": plan.plan_id, "plan_start": plan.start_time,
                            "plan_end": plan.end_time, "mode": plan.mode,
                            "energy_guard_reason": env._candidate_energy_guard_reason(
                                uav, plan, target),
                            "safety_baseline": check(baseline_filter, plan, snapshot),
                            "safety_nominal": check(nominal_filter, plan, snapshot)})
            except Exception as exc:  # record exact bounded probe failure
                row["generation_error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
    result = {"seed": 841301, "time": env.now, "flow_v1_probe": True,
              "filter_configs": {
                  "baseline": {"rule_flow_coordination_enabled": False,
                               "max_speed_residual_mps": 2.0},
                  "nominal": {"rule_flow_coordination_enabled": False,
                               "max_speed_residual_mps": 0.0}},
              "candidates": rows}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "bytes": OUT.stat().st_size,
                      "candidates": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
