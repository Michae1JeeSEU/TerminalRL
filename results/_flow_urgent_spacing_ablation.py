"""Process-local urgent M->I spacing fallback ablation."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import pickle

from results._probe_uid16_blockers import _normal_candidate
from terminal_marl.core.config import RingRole
from terminal_marl.core.landing_flow import LandingFlowCoordinator
from terminal_marl.training import diagnose_landing_flow as diagnostic


SEED = 841501
OUTPUT = Path("results/flow_dev_841501_urgent_spacing_ablation.json")
PREFLIGHT_SNAPSHOT = Path("results/flow_uid16_blockers_v2.t250.pkl")
COMMITS: list[dict] = []
COUNTS: Counter[str] = Counter()
STATE_KEY = "_urgent_spacing_ablation_state"


def _preflight() -> dict:
    with PREFLIGHT_SNAPSHOT.open("rb") as handle:
        env = pickle.load(handle)
    uav = env.active.get(16)
    row = _normal_candidate(env, uav, 72.0, 0.0)
    diag = (row.get("energy") or {}).get("diagnostic") or {}
    ring = env.airspace.actual_ring(RingRole.MIDDLE)
    circuit = 2.0 * math.pi * float(ring.radius) / float(ring.nominal_speed)
    power = float(env.energy.phase_power(uav.phase))
    budget = circuit * power
    slack = diag.get("slack_energy")
    valid = (
        row.get("dynamics_pass")
        and bool((row.get("scheduler") or row.get("scheduler_guard") or {}).get("proposal", (row.get("scheduler_guard") or {}).get("can_reserve_proposal", False)))
        and not bool((row.get("inner_departure_headway") or {}).get("pass"))
        and bool((row.get("energy") or {}).get("pass"))
        and bool((row.get("safety") or {}).get("approved"))
        and isinstance(slack, (int, float))
        and math.isfinite(float(slack))
        and 0.0 <= float(slack) <= budget
    )
    result = {
        "snapshot": str(PREFLIGHT_SNAPSHOT),
        "time": float(env.now),
        "uid": int(uav.uid),
        "duration": 72.0,
        "bias_mps": 0.0,
        "queue_reason": (row.get("inner_departure_headway") or {}).get("reason"),
        "slack_energy": slack,
        "circuit_budget": budget,
        "safety": row.get("safety"),
        "passed": bool(valid),
    }
    if not valid:
        raise RuntimeError(f"frozen t250 D72/bias0 preflight failed: {result}")
    return result


def _budget(env, uav):
    ring = env.airspace.actual_ring(RingRole.MIDDLE)
    circuit = 2.0 * math.pi * float(ring.radius) / float(ring.nominal_speed)
    return circuit * float(env.energy.phase_power(uav.phase))


def _install_patch() -> None:
    original_release = LandingFlowCoordinator.release
    original_candidate = LandingFlowCoordinator._candidate_plan
    original_headway = LandingFlowCoordinator._inner_departure_headway_reason

    def headway(self, uav, plan):
        state = self.__dict__.get(STATE_KEY)
        reason = original_headway(self, uav, plan)
        if state is not None and state.get("active") and uav.uid == state.get("uid"):
            state["real_headway_reason"] = reason
            return None
        return reason

    def candidate(self, uav, target, lz_id, duration, tangential_speed_bias_mps=0.0):
        state = self.__dict__.get(STATE_KEY)
        if state is None or not state.get("active") or uav.uid != state.get("uid"):
            return original_candidate(self, uav, target, lz_id, duration, tangential_speed_bias_mps)
        state["real_headway_reason"] = None
        plan, reason = original_candidate(self, uav, target, lz_id, duration, tangential_speed_bias_mps)
        real_reason = state.get("real_headway_reason")
        if real_reason is None or plan is None or reason is not None:
            return plan, reason
        diagnostic_row = self.env.energy_reachability_diagnostics.get(uav.uid, {})
        slack = diagnostic_row.get("slack_energy")
        budget = _budget(self.env, uav)
        state["last_slack"] = slack
        state["last_budget"] = budget
        if not isinstance(slack, (int, float)) or not math.isfinite(float(slack)) or float(slack) < 0.0 or float(slack) > budget:
            COUNTS["spacing_slack_guard"] += 1
            return plan, "urgent_spacing_slack_guard"
        state["accepted_slack"] = float(slack)
        state["accepted_budget"] = float(budget)
        return plan, None

    def release(self, uav, intent):
        result = original_release(self, uav, intent)
        if result == "approved" or uav.committed_plan is not None:
            return result
        if uav.ring_role is not RingRole.MIDDLE or uav.committed_plan is not None:
            return result
        row = self.env.last_release_telemetry.get(uav.uid, {})
        reasons = row.get("flow_candidate_rejection_reasons", ())
        if "inner_departure_headway" not in reasons:
            return result
        prior_deadline = float(uav.safety_retry_after)
        state = {"active": True, "uid": int(uav.uid), "accepted_slack": None, "accepted_budget": None}
        self.__dict__[STATE_KEY] = state
        uav.safety_retry_after = float(self.env.now)
        try:
            second = original_release(self, uav, intent)
        finally:
            self.__dict__.pop(STATE_KEY, None)
            if uav.committed_plan is None:
                uav.safety_retry_after = prior_deadline
        if second == "approved" and state.get("accepted_slack") is not None:
            COMMITS.append({
                "time": float(self.env.now),
                "uid": int(uav.uid),
                "soc": float(uav.soc),
                "slack_energy": float(state["accepted_slack"]),
                "circuit_budget": float(state["accepted_budget"]),
            })
            COUNTS["commits"] += 1
        else:
            COUNTS[f"second_pass_{second}"] += 1
        return second

    LandingFlowCoordinator._inner_departure_headway_reason = headway
    LandingFlowCoordinator._candidate_plan = candidate
    LandingFlowCoordinator.release = release


def main() -> int:
    if OUTPUT.exists() or Path(f"{OUTPUT}.progress.json").exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    preflight = _preflight()
    _install_patch()
    result = diagnostic.run_diagnostic(
        seed=SEED, coordinated=True, max_steps=1800, output=OUTPUT,
    )
    result["urgent_spacing_ablation"] = {
        "preflight": preflight,
        "policy": "second pass only after original inner_departure_headway; scheduler, energy, SafetyFilter retained",
        "attempt_counts": dict(sorted(COUNTS.items())),
        "exception_commits": COMMITS,
    }
    temporary = OUTPUT.with_name(f".{OUTPUT.name}.tmp")
    temporary.write_text(json.dumps(diagnostic._jsonable(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(OUTPUT)
    print(json.dumps({"output": str(OUTPUT), "steps": result["steps"], "stop_reason": result["stop_reason"], "exception_commits": len(COMMITS)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
