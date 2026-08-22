"""A6.19: repair A6.17's diagnostic purity, then canonical re-run (PPO STOP)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .a611_causal_audit import _safe
from .a612_scheduler_repair_audit import _a611_mu0
from .a617_transition_interference_audit import (
    MATERIAL_WAITING_AREA, PPO_AUTHORIZATION, TIMES, _profile, _run, run_audit as a617_run_audit,
)
from .a618_canonical_replay_audit import (
    ACTION_TIME, HISTORICAL_NOMINAL, HISTORICAL_WAITING, SEED, _execute,
)


GATE_TIME = 302.0
GATE_UAV = "16"
STAGE = "A6.19 A6.17 Diagnostic Purity Repair & Canonical Re-run"


def _default_a618_artifact() -> Path:
    return Path.home() / "training_outputs" / "terminal" / "event_gated_stage15_diagnostics" / "a618" / "a618_audit.json"


def _trace_at(result: dict, time: float) -> dict | None:
    return next((row for row in result["trace"] if float(row["time"]) == time), None)


def _reference_a618(path: str | Path | None = None) -> dict:
    """Read the already-established A6.14/A6.15 canonical t=302 evidence."""
    artifact_path = Path(path) if path is not None else _default_a618_artifact()
    report = json.loads(artifact_path.read_text(encoding="utf-8"))
    a614 = next(row for row in report["replays"] if row["entry"] == "a614")
    a615 = next(row for row in report["replays"] if row["entry"] == "a615_instrumented")
    divergence = next(row for row in report["first_divergence"] if row["comparison"] == "a614_vs_a617_direct")
    return {
        "artifact": str(artifact_path), "a614": a614, "a615": a615,
        "t302": json.loads(divergence["left"]),
        "a615_identical_to_a614": next(row for row in report["first_divergence"]
                                         if row["comparison"] == "a614_vs_a615_instrumented")["time"] is None,
    }


def _metrics_equal(left: dict, right: dict) -> bool:
    keys = ("all_request_waiting_area", "completed", "unresolved_count", "external_queue_area", "service_delay_p95")
    return all(left["metrics"][key] == right["metrics"][key] for key in keys) and left["safe"] == right["safe"]


def run_replay_gate(*, reference_path: str | Path | None = None) -> dict:
    """Require repaired A6.17 to reproduce A6.14/A6.15 before the sweep."""
    reference = _reference_a618(reference_path)
    profile = _profile(_a611_mu0(0)[0])
    # A6.17's deep-copy/probe observers are independently toggled.  The
    # release trace remains collected in both runs, so the comparison checks
    # observer purity rather than merely matching one terminal scalar.
    observers_off = _run(profile, SEED, ACTION_TIME, "full_dynamic", diagnostic_probes=False)
    observers_on = _run(profile, SEED, ACTION_TIME, "full_dynamic", diagnostic_probes=True)
    canonical = _execute("a617_direct", trace=True)
    observed = _trace_at(canonical, GATE_TIME)
    expected = reference["t302"]
    fields = ("pre_state_hash", "action_hash", "rng_hash", "reservation_hash", "release_decisions",
              "transition_before", "transition_after", "post_state_hash")
    trace_match = observed is not None and all(observed[field] == expected[field] for field in fields)
    release_frozen = observed is not None and f'[["{GATE_UAV}","geometry_frozen"]' in observed["release_decisions"]
    canonical_delta = canonical["waiting_area"] - HISTORICAL_NOMINAL
    checks = {
        "a614_waiting_reference": reference["a614"]["waiting_area"] == HISTORICAL_WAITING,
        "a615_waiting_reference": reference["a615"]["waiting_area"] == HISTORICAL_WAITING,
        "a615_trace_matches_a614": reference["a615_identical_to_a614"],
        "observers_on_off_metrics_equal": _metrics_equal(observers_off, observers_on),
        "observers_on_off_release_trace_equal": observers_off["trace"] == observers_on["trace"],
        "t302_uav16_geometry_frozen": release_frozen,
        "t302_prestate_action_rng_reservation_event_match": trace_match,
        "canonical_waiting_matches_a614": abs(canonical["waiting_area"] - HISTORICAL_WAITING) < 1e-6,
        "canonical_delta_matches_a614": abs(canonical_delta - (HISTORICAL_WAITING - HISTORICAL_NOMINAL)) < 1e-6,
        "canonical_hard_safe": canonical["safe"],
    }
    return {
        "evidence_type": "newly_run_with_preexisting_a618_reference", "seed": SEED, "timing": ACTION_TIME,
        "uav_id": GATE_UAV, "time": GATE_TIME, "reference": reference,
        "observers_off": {"metrics": observers_off["metrics"], "safe": observers_off["safe"], "trace": observers_off["trace"]},
        "observers_on": {"metrics": observers_on["metrics"], "safe": observers_on["safe"], "trace": observers_on["trace"]},
        "canonical_a617": canonical, "canonical_delta_vs_nominal": canonical_delta,
        "checks": checks, "passed": all(checks.values()),
    }


def _classify(rows: list[dict]) -> list[str]:
    indexed = {(row["seed"], row["timing"], row["branch"]): row for row in rows}
    pairs = []
    for seed in sorted({row["seed"] for row in rows}):
        for timing in TIMES:
            nominal = indexed[seed, timing, "nominal"]["waiting_area"]
            pairs.append((seed, timing, indexed[seed, timing, "full_dynamic"]["waiting_area"] - nominal,
                          indexed[seed, timing, "freeze_only"]["waiting_area"] - nominal))
    freeze_causal = [item for item in pairs if abs(item[2]) >= MATERIAL_WAITING_AREA
                     and abs(item[3]) >= MATERIAL_WAITING_AREA and item[2] * item[3] > 0]
    geometry_only = [item for item in pairs if abs(item[2]) >= MATERIAL_WAITING_AREA
                     and abs(item[3]) < MATERIAL_WAITING_AREA]
    if freeze_causal and geometry_only:
        return ["MIXED"]
    if freeze_causal:
        return (["RECONFIGURATION_FREEZE_INTERFERENCE"] if len(freeze_causal) == len(pairs)
                else ["STATE_DEPENDENT_RECONFIGURATION_WINDOW"])
    if geometry_only:
        return ["TRANSITION_GEOMETRY_PHASE_SHIFT"]
    return ["INCONCLUSIVE"]


def run_audit(*, reference_path: str | Path | None = None, gate_only: bool = False) -> dict:
    gate = run_replay_gate(reference_path=reference_path)
    if not gate["passed"]:
        return {"stage": STAGE, "evidence_type": gate["evidence_type"], "replay_gate": gate,
                "branch_metrics": [], "transition_interruptions": [], "first_divergence": [],
                "summary": {"classification": ["REPLAY_GATE_FAIL"], "ppo_authorization": PPO_AUTHORIZATION,
                            "protected_safety": "NOT_RUN_AFTER_GATE_FAILURE"}}
    if gate_only:
        return {"stage": STAGE, "evidence_type": gate["evidence_type"], "replay_gate": gate,
                "branch_metrics": [], "transition_interruptions": [], "first_divergence": [],
                "summary": {"classification": ["INCONCLUSIVE"], "ppo_authorization": PPO_AUTHORIZATION,
                            "protected_safety": "GATE_ONLY"}}
    rerun = a617_run_audit(seeds=(0, 1))
    if not all(row["safe"] and row["checked_executed"] for row in rerun["branch_metrics"]):
        raise RuntimeError("A6.19 formal branch safety failure")
    return {"stage": STAGE, "evidence_type": "newly_run", "replay_gate": gate,
            "inputs": {**rerun["inputs"], "ppo_authorization": PPO_AUTHORIZATION,
                       "formal_runtime_changes": "none; A6.17 diagnostic harness only"},
            "branch_metrics": rerun["branch_metrics"], "transition_interruptions": rerun["transition_interruptions"],
            "first_divergence": rerun["first_divergence"], "raw": rerun["raw"],
            "summary": {"classification": _classify(rerun["branch_metrics"]), "ppo_authorization": PPO_AUTHORIZATION,
                        "protected_safety": "PASS", "material_waiting_area": MATERIAL_WAITING_AREA,
                        "old_a617_classification": "INVALIDATED_BY_A6.18_DIAGNOSTIC_SIDE_EFFECT"}}


def _csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row}) or ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def write_report(report: dict, output: str | Path) -> None:
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    gate = report["replay_gate"]
    gate_row = {"seed": gate["seed"], "timing": gate["timing"], "time": gate["time"], "uav_id": gate["uav_id"],
                "passed": gate["passed"], "canonical_waiting_area": gate["canonical_a617"]["waiting_area"],
                "delta_vs_nominal": gate["canonical_delta_vs_nominal"], **gate["checks"]}
    _csv(output / "a619_replay_gate.csv", [gate_row])
    _csv(output / "a619_branch_metrics.csv", report["branch_metrics"])
    _csv(output / "a619_transition_interruptions.csv", report["transition_interruptions"])
    _csv(output / "a619_first_divergence.csv", report["first_divergence"])
    (output / "a619_audit.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if not report["branch_metrics"]:
        body = "The replay gate did not authorize the full A6.17 re-run."
    else:
        indexed = {(row["seed"], row["timing"], row["branch"]): row for row in report["branch_metrics"]}
        deltas = []
        for seed in (0, 1):
            for timing in TIMES:
                nominal = indexed[seed, timing, "nominal"]["waiting_area"]
                full = indexed[seed, timing, "full_dynamic"]["waiting_area"] - nominal
                freeze = indexed[seed, timing, "freeze_only"]["waiting_area"] - nominal
                deltas.append(f"| {seed} | {timing:.0f} | {full:+.1f} | {freeze:+.1f} |")
        safe_count = sum(bool(row["safe"] and row["checked_executed"]) for row in report["branch_metrics"])
        body = (f"Formal branches hard-safe and checked==executed: {safe_count}/{len(report['branch_metrics'])}.\n\n"
                "| seed | action time (s) | canonical full Δ waiting-area | freeze-only Δ waiting-area |\n"
                "|---:|---:|---:|---:|\n" + "\n".join(deltas) + "\n\n"
                "The old A6.17 classification is invalidated. Results above use canonical full-dynamic semantics; no scheduler, safety, transition, timing, reward, observation, action-space, or PPO change was made.")
    (output / "a619_summary.md").write_text(
        f"# {STAGE}\n\nClassification: **{', '.join(report['summary']['classification'])}**.\n\n"
        f"Replay gate: **{'PASS' if gate['passed'] else 'FAIL'}**. At t=302, seed1/t300/UAV16 is `geometry_frozen`; repaired A6.17 waiting-area is {gate['canonical_a617']['waiting_area']:.1f} "
        f"(Δ vs nominal {gate['canonical_delta_vs_nominal']:+.1f}).\n\n{body}\n\n"
        "PPO authorization: **STOP**. Evidence type is recorded in `a619_audit.json`; the A6.14/A6.15 t=302 reference is the prior A6.18 canonical artifact.",
        encoding="utf-8")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None); parser.add_argument("--reference", default=None)
    parser.add_argument("--gate-only", action="store_true")
    args = parser.parse_args()
    output = Path(args.output) if args.output else Path.home() / "training_outputs" / "terminal" / "event_gated_stage15_diagnostics" / "a619"
    write_report(run_audit(reference_path=args.reference, gate_only=args.gate_only), output)


if __name__ == "__main__":
    main()
