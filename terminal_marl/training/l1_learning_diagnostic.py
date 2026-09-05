"""Small, frozen L1 Full-Teacher PPO/BC learning diagnostic.

This runner is intentionally not a curriculum transition or a long L1 run.
It compares independent five-update branches that all load the same safe L1
checkpoint, retain the fixed seed partition, and stop before a further PPO
update if a safety hard gate fails.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from typing import Any

from .train_low_v2_ppo import L1FullTeacherProtocol, L1SafetyGateError, l1_full_teacher_config, train_l1_full_teacher


FROZEN_SAFETY_ENVIRONMENT_COMMIT = "449514d"
BC_WEIGHTS = (0.05, 0.10, 0.20)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and float("-inf") < float(value) < float("inf")


def _safety_zero(metrics: dict[str, Any]) -> bool:
    return all(int(metrics.get(name, 0)) == 0 for name in (
        "collisions", "nmac", "hard_separation_violations", "safety_invariant_errors",
    ))


def _comparison(baseline: dict[str, Any], final: dict[str, Any]) -> dict[str, Any]:
    baseline_warm = baseline["by_scenario"]["warm_start_finite_cohort"]
    final_warm = final["by_scenario"]["warm_start_finite_cohort"]
    base_overall = baseline["overall"]
    final_overall = final["overall"]
    def rate(values: dict[str, Any], key: str) -> float | None:
        value = values.get(key)
        return None if value is None else float(value)
    warm_base, warm_final = rate(baseline_warm, "teacher_interventions_per_100_release_intents"), rate(final_warm, "teacher_interventions_per_100_release_intents")
    overall_base, overall_final = rate(base_overall, "teacher_interventions_per_100_release_intents"), rate(final_overall, "teacher_interventions_per_100_release_intents")
    p95_base, p95_final = rate(base_overall, "correction_magnitude_p95"), rate(final_overall, "correction_magnitude_p95")
    shield_base, shield_final = rate(base_overall, "shield_interventions_per_100_release_intents"), rate(final_overall, "shield_interventions_per_100_release_intents")
    completion_base, completion_final = rate(base_overall, "completion_ratio"), rate(final_overall, "completion_ratio")
    throughput_base, throughput_final = rate(base_overall, "throughput_per_second"), rate(final_overall, "throughput_per_second")
    # A 5% reduction on both rates is deliberately modest: this is only a
    # five-update diagnostic, not evidence for a final curriculum promotion.
    clear_drop = (warm_base is not None and warm_final is not None and warm_final <= 0.95 * warm_base and
                  overall_base is not None and overall_final is not None and overall_final <= 0.95 * overall_base)
    p95_ok = p95_base is not None and p95_final is not None and p95_final <= p95_base + 1e-9
    shield_ok = shield_base is not None and shield_final is not None and shield_final <= shield_base + 1e-9
    completion_ok = completion_base is not None and completion_final is not None and completion_final >= 0.95 * completion_base
    throughput_ok = throughput_base is not None and throughput_final is not None and throughput_final >= 0.95 * throughput_base
    return {
        "warm_teacher_rate_baseline": warm_base, "warm_teacher_rate_update5": warm_final,
        "overall_teacher_rate_baseline": overall_base, "overall_teacher_rate_update5": overall_final,
        "correction_p95_baseline": p95_base, "correction_p95_update5": p95_final,
        "shield_rate_baseline": shield_base, "shield_rate_update5": shield_final,
        "completion_baseline": completion_base, "completion_update5": completion_final,
        "throughput_baseline": throughput_base, "throughput_update5": throughput_final,
        "clear_teacher_dependency_improvement": clear_drop,
        "p95_not_increased": p95_ok, "shield_not_worse": shield_ok,
        "completion_healthy": completion_ok, "throughput_healthy": throughput_ok,
        "effective": bool(clear_drop and p95_ok and shield_ok and completion_ok and throughput_ok),
    }


def _ppo_health(rows: list[dict[str, Any]]) -> dict[str, Any]:
    updates = [{name: row.get(name) for name in (
        "update", "actor_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "explained_variance", "grad_norm",
    )} for row in rows if row.get("record_type") == "ppo_update"]
    finite = all(_finite(value) for row in updates for key, value in row.items() if key != "update")
    # These diagnostic bounds flag an optimizer pathology; they are not a
    # substitute for the teacher-dependence acceptance criterion.
    bounded = bool(updates) and all(abs(float(row["approx_kl"])) <= 0.05 and float(row["clip_fraction"]) <= 0.30 and
                                   float(row["entropy"]) > 0.0 and float(row["grad_norm"]) <= 100.0 for row in updates)
    return {"updates": updates, "healthy": bool(finite and bounded), "finite": finite, "within_diagnostic_bounds": bounded}


def _result_for_branch(path: Path, weight: float) -> dict[str, Any]:
    manifest = json.loads((path / "l1_full_teacher.manifest.json").read_text(encoding="utf-8"))
    validation = json.loads((path / "l1_validation_summary.json").read_text(encoding="utf-8"))["records"]
    labels = {row["label"]: row for row in validation}
    baseline, update3, update5 = labels["early_baseline"], labels["after_update_3"], labels["after_update_5"]
    training_rows = _read_jsonl(path / "l1_training_metrics.jsonl")
    comparison = _comparison(baseline, update5)
    safety = bool(manifest["safety_passed"] and _safety_zero(baseline["overall"]) and _safety_zero(update3["overall"]) and _safety_zero(update5["overall"]))
    reasons = update5["overall"]["teacher_correction_reasons"]
    head = max((name for name in ("release", "speed", "merge-time", "combined")), key=lambda name: reasons.get(name, 0))
    return {
        "status": "completed", "bc_weight": weight, "output_dir": str(path), "checkpoint": manifest["checkpoint"],
        "safety_passed": safety, "baseline": baseline, "update3": update3, "update5": update5,
        "comparison": comparison, "most_common_teacher_correction": head,
        "action_saturation_update5": update5["overall"]["requested_action_saturation"],
        "ppo_diagnostics": _ppo_health(training_rows),
        "effective": bool(safety and comparison["effective"]),
    }


def _markdown(results: list[dict[str, Any]], recommendation: float | None, allowed: bool) -> str:
    lines = ["# L1 Learning Diagnostic Summary", ""]
    lines.append("- PPO/BC provenance PASS" if all(row.get("provenance_pass", True) for row in results) else "- PPO/BC provenance FAIL")
    for result in results:
        weight = result["bc_weight"]
        if result["status"] != "completed":
            lines.append(f"- BC={weight:.2f} 结果：STOPPED — {result['error']}")
            continue
        overall = result["update5"]["overall"]
        warm = result["update5"]["by_scenario"]["warm_start_finite_cohort"]
        comparison = result["comparison"]
        lines.append(
            f"- BC={weight:.2f} 结果：safety={'PASS' if result['safety_passed'] else 'FAIL'}；"
            f"overall teacher={overall['teacher_interventions_per_100_release_intents']:.3f}/100 intents；"
            f"warm teacher={warm['teacher_interventions_per_100_release_intents']:.3f}/100 intents；"
            f"correction P95={overall['correction_magnitude_p95']:.5f}；"
            f"shield={overall['shield_interventions_per_100_release_intents']:.3f}/100 intents；"
            f"completion={overall['completion_ratio']:.3f}；throughput={overall['throughput_per_second']:.5f}；"
            f"effective={'yes' if comparison['effective'] else 'no'}."
        )
        lines.append(f"- BC={weight:.2f} action saturation：{result['action_saturation_update5']}；"
                     f"PPO diagnostics={'healthy' if result['ppo_diagnostics']['healthy'] else 'needs diagnostic'}。")
        lines.append(f"- BC={weight:.2f} teacher correction 贡献最多的 action head：{result['most_common_teacher_correction']}。")
    lines.append(f"- 推荐 BC weight：{'BC=' + format(recommendation, '.2f') if recommendation is not None else '无；三组均未满足明确改善门槛'}。")
    lines.append(f"- 是否允许开始正式 L1 Full-Teacher PPO：{'允许' if allowed else '不允许'}。")
    if not allowed:
        lines.append("- 若不允许，下一步具体诊断项：检查 learning rate、entropy coefficient、approx KL/clip fraction、advantage 标准化与尺度、value loss 主导性，以及 release/speed/merge-time 三个 head 的分布与饱和。")
    return "\n".join(lines) + "\n"


def _run_branch(args: tuple[str, str, str, str, float, int]) -> dict[str, Any]:
    """Run one isolated BC branch; suitable for a separate OS process."""
    c0_report, teacher_dataset_path, initial_checkpoint, branch_text, weight, rollout_workers = args
    branch = Path(branch_text)
    protocol = L1FullTeacherProtocol(updates=5, episodes_per_update=4, rollout_workers=rollout_workers,
                                     max_steps=1200, validation_every=3, rolling_window_updates=5,
                                     teacher_bc_weight=weight, ppo_epochs=2, ppo_minibatch_size=256,
                                     optimizer_seed=1701, stop_on_safety_failure=True)
    try:
        train_l1_full_teacher(c0_report=c0_report, teacher_dataset_path=teacher_dataset_path,
                              output_dir=branch, protocol=protocol, resume_checkpoint=initial_checkpoint)
        result = _result_for_branch(branch, weight)
        result["provenance_pass"] = True
        return result
    except (L1SafetyGateError, RuntimeError) as error:
        return {"status": "stopped", "bc_weight": weight, "error": str(error), "safety_passed": False,
                "provenance_pass": True, "output_dir": str(branch)}


def run_l1_learning_diagnostic(*, c0_report: str | Path, teacher_dataset_path: str | Path,
                               initial_checkpoint: str | Path, output_dir: str | Path,
                               rollout_workers: int = 4) -> dict[str, Any]:
    """Run three independent, five-update BC branches and write the decision."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    frozen = {
        "source_commit": FROZEN_SAFETY_ENVIRONMENT_COMMIT,
        "safety_sampling_dt": 1.0, "physics_dt": 0.2, "mu_hat": 0.003, "lambda_uav_per_second": 0.001575,
        "teacher": "FULL", "priority": 1, "mission": "standard", "soc": 0.8,
        "geometry": "nominal", "gates": "uniform", "lz": "all_free", "high_level": "OFF",
    }
    capacity = json.loads(Path(c0_report).read_text(encoding="utf-8"))["frozen_capacity"]
    rate = sum(capacity["traffic_levels"]["medium"]) / 2.0
    config_guard = l1_full_teacher_config(medium_rate=rate, reset_mode="warm_start_rings", demand_model="finite_cohort",
                                          warm_start_uavs=8, cohort_uavs=8, horizon_seconds=1200.0)
    if not (abs(float(capacity["mu_hat"]) - frozen["mu_hat"]) < 1e-12 and
            abs(rate - frozen["lambda_uav_per_second"]) < 1e-12 and
            config_guard.safety_sampling_dt == frozen["safety_sampling_dt"] and
            config_guard.physics_dt == frozen["physics_dt"] and
            config_guard.initial_priority == frozen["priority"] and
            config_guard.teacher_mode == "teacher_full"):
        raise RuntimeError("L1 learning diagnostic refused: frozen safety/environment configuration mismatch")
    provenance = {
        "ppo_behavior_action": "a_requested", "ppo_old_log_prob": "log_pi(a_requested|s)",
        "ppo_ratio_uses_a_applied": False, "teacher_bc_target": "a_teacher",
        "teacher_bc_mask": "teacher_used && a_teacher is not None",
    }
    manifest = {
        "stage": "L1_learning_diagnostic", "evidence_type": "newly_run", "frozen_safety_environment": frozen,
        "initial_checkpoint": str(initial_checkpoint), "bc_weights": list(BC_WEIGHTS), "ppo_bc_provenance": provenance,
        "protocol": {"updates": 5, "validation": ["baseline", "update_3", "update_5"], "independent_branches": True,
                     "same_train_validation_seeds": True, "no_inter_branch_resume": True, "stop_on_safety_failure": True},
    }
    (destination / "l1_learning_diagnostic.manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # The branches have no shared model, optimizer, output directory, or
    # resume lineage.  Parallel OS processes reduce wall-clock time without
    # changing their seed partition or PPO configuration.
    branch_args = [(str(c0_report), str(teacher_dataset_path), str(initial_checkpoint),
                    str(destination / f"bc_{weight:.2f}"), weight, rollout_workers) for weight in BC_WEIGHTS]
    with ProcessPoolExecutor(max_workers=len(BC_WEIGHTS)) as executor:
        results = list(executor.map(_run_branch, branch_args))
    eligible = [row for row in results if row.get("effective")]
    recommendation = min((row["bc_weight"] for row in eligible), default=None)
    allowed = recommendation is not None and all(row.get("safety_passed", False) for row in results)
    summary = {"manifest": manifest, "results": results, "recommended_bc_weight": recommendation,
               "allow_formal_l1_full_teacher": allowed}
    (destination / "l1_learning_diagnostic_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    (destination / "l1_learning_diagnostic_summary.md").write_text(_markdown(results, recommendation, allowed), encoding="utf-8")
    return summary


def _main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Run the small L1 PPO/BC learning diagnostic only.")
    parser.add_argument("--c0-report", required=True)
    parser.add_argument("--teacher-dataset", required=True)
    parser.add_argument("--initial-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run_l1_learning_diagnostic(c0_report=args.c0_report, teacher_dataset_path=args.teacher_dataset,
                                                 initial_checkpoint=args.initial_checkpoint, output_dir=args.output,
                                                 rollout_workers=args.workers), indent=2))


if __name__ == "__main__":
    _main()
