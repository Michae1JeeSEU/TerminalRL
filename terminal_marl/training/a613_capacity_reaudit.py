"""A6.13 capacity-value re-audit after the A6.12 scheduler safety closure.

This is a deterministic rule-policy diagnostic.  It does not train PPO or
alter checkpoints, safety thresholds, or high-level action semantics.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import csv
import json
from pathlib import Path

from ..core.config import RingRole
from .a611_causal_audit import (
    EXPANSIONS, DemandProfile, _base_config, _decomposition, _run_dynamic, _run_fixed,
    _safe, static_expanded_rings,
)
from .a612_scheduler_repair_audit import _a611_mu0


def repair_config(profile: DemandProfile, *, dynamic: bool, role: RingRole | None = None):
    base = _base_config(profile, dynamic=dynamic)
    if role is not None:
        base = replace(base, rings=static_expanded_rings(base, role))
    return replace(base, repair_aware_scheduler=True)


def _task(profile: DemandProfile, seed: int, kind: str, role: RingRole) -> tuple[str, dict]:
    if kind == "static":
        result, _ = _run_fixed(repair_config(profile, dynamic=False, role=role), seed)
    elif kind == "dynamic":
        result, _ = _run_dynamic(repair_config(profile, dynamic=True), seed, EXPANSIONS[role])
    else:
        raise ValueError(kind)
    return f"{kind}_{role.value}_plus1", result


def run_audit(*, seed: int = 0, workers: int = 6) -> dict:
    mu, mu_evidence = _a611_mu0(seed)
    profile = DemandProfile("constant_high_repair_scheduler", ((0., 720., 1.55 * mu),), 720., 480.)
    nominal, _ = _run_fixed(repair_config(profile, dynamic=False), seed)
    if not _safe(nominal["metrics"], repair_config(profile, dynamic=False)):
        raise RuntimeError("A6.13 FAIL: repair-aware nominal branch is not hard-safe")
    rows = {"nominal": nominal}
    with ProcessPoolExecutor(max_workers=min(workers, 6)) as pool:
        futures = [pool.submit(_task, profile, seed, kind, role) for role in RingRole for kind in ("static", "dynamic")]
        for future in as_completed(futures):
            name, result = future.result()
            if not _safe(result["metrics"], repair_config(profile, dynamic=name.startswith("dynamic"))):
                raise RuntimeError(f"A6.13 FAIL: unsafe branch {name}")
            rows[name] = result
    conclusions = {}
    for role in RingRole:
        static, dynamic = rows[f"static_{role.value}_plus1"], rows[f"dynamic_{role.value}_plus1"]
        conclusions[role.value] = _decomposition(nominal, static, dynamic)
    labels = set(conclusions.values())
    overall = "CAPACITY_CONTROL_VALUE_PROVEN" if "CAPACITY_VALUE_PROVEN" in labels else (
        "RECONFIGURATION_TIMESCALE_MISMATCH" if "RECONFIGURATION_COST_OFFSETS_GAIN" in labels else "CAPACITY_NOT_BINDING")
    return {"stage": "A6.13 repair-aware capacity-value re-audit", "evidence_type": "newly_run", "seed": seed,
            "inputs": {"a611_mu0": mu, "a611_mu0_evidence_type": mu_evidence, "fixed_geometry_capacity": {"outer": 47, "middle": 31, "inner": 15}},
            "profile": {"name": profile.name, "segments": profile.segments, "duration_seconds": profile.duration_seconds, "drain_seconds": profile.drain_seconds},
            "comparison": rows, "ring_conclusions": conclusions,
            "summary": {"protected_safety": "PASS", "a613_result": overall, "ppo_authorization": "STOP", "workers": min(workers, 6)}}


def write_report(report: dict, output: str | Path) -> None:
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    (output / "a613_capacity_reaudit.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    with (output / "a613_static_dynamic_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["condition", "completed", "waiting_area", "unresolved", "energy", "fairness", "safe"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for condition, row in report["comparison"].items():
            metrics = row["metrics"]
            writer.writerow({"condition": condition, "completed": metrics["completed"], "waiting_area": metrics["all_request_waiting_area"],
                             "unresolved": metrics["unresolved_count"], "energy": metrics["energy_mean"],
                             "fairness": metrics["priority_fairness_std"], "safe": row["safe"]})
    lines = ["# A6.13 Repair-aware Capacity-Value Re-audit", "", f"Protected safety: **{report['summary']['protected_safety']}**", f"Result: **{report['summary']['a613_result']}**", "PPO authorization: **STOP**", "", "| Ring | Conclusion |", "|---|---|"]
    lines.extend(f"| {ring} | {conclusion} |" for ring, conclusion in report["ring_conclusions"].items())
    lines.extend(["", "Evidence type: newly_run. This is one-seed, deterministic CRN mechanism evidence; it is not a final multi-seed PPO authorization."])
    (output / "a613_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, default=0); parser.add_argument("--workers", type=int, default=6); parser.add_argument("--output", default=None)
    args = parser.parse_args(); output = Path(args.output) if args.output else Path.home() / "training_outputs" / "terminal" / "event_gated_stage15_diagnostics" / "a613"
    report = run_audit(seed=args.seed, workers=args.workers); write_report(report, output); print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
