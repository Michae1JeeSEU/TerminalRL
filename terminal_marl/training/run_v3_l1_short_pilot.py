"""Fresh, bounded V3 L1 Full-Teacher PPO short pilot.

This runner deliberately has no resume/checkpoint import option.  It exists
only to collect a reproducible six-update V3 pilot and diagnose whether the
release head collapses to conservative WAIT behaviour before any longer L1
continuation is considered.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields
import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from ..algo.shared_ppo import RolloutBuffer, SharedPPO, Transition
from ..core.checkpoints import save_checkpoint
from ..core.low_level_action import HEAD_NAMES, HEAD_CARDINALITIES, indices_to_normalized
from ..core.observations import LOW_OBS_V2
from .audit_low_teacher_dataset import audit_teacher_dataset
from .low_v2_seed_partitions import LowV2SeedPartition
from .train_low_v2_ppo import (
    L1EpisodeSummary, _episode_design, _model_state, _read_frozen_capacity,
    _read_seed_partition, _rollout_task, _scenario_name, _summary_metrics,
    l1_full_teacher_config,
)


MU_HAT = .003
LAMBDA = .001575
VALIDATION_DESIGN_INDICES = (0, 1, 3, 9, 4, 8)  # two warm, two empty-finite, two Poisson


@dataclass(frozen=True)
class V3L1ShortPilotProtocol:
    updates: int = 6
    episodes_per_update: int = 4
    rollout_workers: int = 4
    max_steps: int = 1200
    ppo_epochs: int = 2
    minibatch_size: int = 256
    teacher_bc_weight: float = .05

    def __post_init__(self) -> None:
        if (self.updates, self.episodes_per_update, self.rollout_workers, self.max_steps,
                self.ppo_epochs, self.minibatch_size) != (6, 4, 4, 1200, 2, 256):
            raise ValueError("V3 short pilot protocol is frozen at 6 updates × 4 episodes, 4 workers, 1200 steps")
        if self.teacher_bc_weight != .05:
            raise ValueError("V3 short pilot fixes teacher_bc_weight at 0.05")


def _serializable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_serializable(item) for item in value]
    return value


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_serializable(payload), indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_serializable(row), allow_nan=False) + "\n")


def _collect(tasks: list[tuple], workers: int) -> list[tuple[list[Transition], L1EpisodeSummary]]:
    if workers == 1:
        return [_rollout_task(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_rollout_task, tasks))


def _target(row: Transition) -> np.ndarray:
    return row.action if row.teacher_action is None else row.teacher_action


def _distribution_metrics(rows: list[Transition], model_state: dict[str, torch.Tensor]) -> dict:
    """Head-level request/teacher distributions under the frozen rollout actor."""
    if not rows:
        raise RuntimeError("pilot metric collection received no transitions")
    actions = np.stack([row.action for row in rows]).astype(np.int64, copy=False)
    targets = np.stack([_target(row) for row in rows]).astype(np.int64, copy=False)
    changed = actions != targets
    policy = SharedPPO(LOW_OBS_V2.dimension, teacher_bc_weight=.05)
    policy.model.load_state_dict(model_state); policy.model.eval()
    observations = np.stack([row.observation for row in rows]).astype(np.float32, copy=False)
    probability_chunks = []
    entropy_chunks = []
    with torch.no_grad():
        for start in range(0, len(observations), 1024):
            distribution = policy.model.actor_distribution(torch.as_tensor(observations[start:start + 1024]))
            probability_chunks.append([head.probs.cpu().numpy() for head in distribution.heads])
            entropy_chunks.append(distribution.entropy_per_head().cpu().numpy())
    probabilities = [np.concatenate([chunk[head] for chunk in probability_chunks]) for head in range(3)]
    entropies = np.concatenate(entropy_chunks)

    def counts(values: np.ndarray, cardinality: int) -> dict[str, int]:
        return {str(index): int((values == index).sum()) for index in range(cardinality)}

    combined = {name: 0 for name in ("none", "release-only", "speed-only", "merge-only", "release+speed", "release+merge", "speed+merge", "all-three")}
    hold_reject = 0
    for row, delta in zip(rows, changed):
        names = tuple(name for name, flag in zip(HEAD_NAMES, delta) if flag)
        if not names:
            if row.teacher_used and row.teacher_action is None:
                hold_reject += 1
            combined["none"] += 1
        elif names == ("release",): combined["release-only"] += 1
        elif names == ("speed",): combined["speed-only"] += 1
        elif names == ("merge_time",): combined["merge-only"] += 1
        elif names == ("release", "speed"): combined["release+speed"] += 1
        elif names == ("release", "merge_time"): combined["release+merge"] += 1
        elif names == ("speed", "merge_time"): combined["speed+merge"] += 1
        elif names == HEAD_NAMES: combined["all-three"] += 1
        else: raise RuntimeError(f"unexpected V3 correction combination {names}")
    release_requested = actions[:, 0]
    release_target = targets[:, 0]
    release_probability = probabilities[0].mean(axis=0)
    head = {
        "release": {
            "policy_probability": {"WAIT": float(release_probability[0]), "RELEASE": float(release_probability[1])},
            "requested": {"WAIT": int((release_requested == 0).sum()), "RELEASE": int((release_requested == 1).sum())},
            "teacher_target": {"WAIT": int((release_target == 0).sum()), "RELEASE": int((release_target == 1).sum())},
            "RELEASE_to_WAIT": int(((release_requested == 1) & (release_target == 0)).sum()),
            "WAIT_to_RELEASE": int(((release_requested == 0) & (release_target == 1)).sum()),
            "false_release_rate": None if not np.any(release_requested == 1) else float(np.mean(release_target[release_requested == 1] == 0)),
            "false_wait_rate": None if not np.any(release_requested == 0) else float(np.mean(release_target[release_requested == 0] == 1)),
            "entropy": float(entropies[:, 0].mean()),
            "teacher_correction_rate": float(changed[:, 0].mean()),
        },
    }
    for index, name in ((1, "speed"), (2, "merge_time")):
        head[name] = {
            "requested": counts(actions[:, index], HEAD_CARDINALITIES[index]),
            "teacher_target": counts(targets[:, index], HEAD_CARDINALITIES[index]),
            "teacher_correction_rate": float(changed[:, index].mean()),
            "entropy": float(entropies[:, index].mean()),
        }
    return {"transitions": len(rows), "heads": head, "combined_corrections": combined,
            "teacher_hold_reject": hold_reject, "per_head_correction_rate": {
                name: float(changed[:, index].mean()) for index, name in enumerate(HEAD_NAMES)
            }}


def _records_metrics(records: list[tuple[list[Transition], L1EpisodeSummary]], model_state: dict[str, torch.Tensor]) -> dict:
    summaries = [summary for _, summary in records]
    transitions = [row for rows, _ in records for row in rows]
    return {"environment": _summary_metrics(summaries), "distribution": _distribution_metrics(transitions, model_state)}


def _validation_records(policy: SharedPPO, partition: LowV2SeedPartition, *, update: int,
                        protocol: V3L1ShortPilotProtocol) -> list[tuple[list[Transition], L1EpisodeSummary]]:
    state = _model_state(policy)
    tasks = []
    for episode, (design_index, seed) in enumerate(zip(VALIDATION_DESIGN_INDICES, partition.validation[:len(VALIDATION_DESIGN_INDICES)])):
        config, _ = _episode_design(design_index, medium_rate=LAMBDA, horizon_seconds=float(protocol.max_steps))
        tasks.append((state, config, seed, update, episode, protocol.max_steps, True, None))
    return _collect(tasks, protocol.rollout_workers)


def _validation_metrics(records: list[tuple[list[Transition], L1EpisodeSummary]], state: dict[str, torch.Tensor]) -> dict:
    groups = {
        "overall": records,
        "warm_start": [row for row in records if row[1].reset_mode == "warm_start_rings"],
        "from_empty": [row for row in records if row[1].reset_mode == "flow_from_empty"],
        "from_empty_finite": [row for row in records if row[1].reset_mode == "flow_from_empty" and row[1].demand_model == "finite_cohort"],
        "poisson": [row for row in records if row[1].demand_model == "poisson"],
    }
    return {name: _records_metrics(rows, state) for name, rows in groups.items() if rows}


def _training_groups(records: list[tuple[list[Transition], L1EpisodeSummary]], state: dict[str, torch.Tensor]) -> dict:
    groups = {
        "warm_start": [row for row in records if row[1].reset_mode == "warm_start_rings"],
        "from_empty": [row for row in records if row[1].reset_mode == "flow_from_empty"],
    }
    return {name: _records_metrics(rows, state) for name, rows in groups.items() if rows}


def _collapse_diagnostic(updates: list[dict], validations: list[dict]) -> dict:
    """Record evidence; never alter class weights/reward to hide a collapse."""
    release_probabilities = [row["distribution"]["heads"]["release"]["policy_probability"]["RELEASE"] for row in updates]
    requested_rates = [row["distribution"]["heads"]["release"]["requested"]["RELEASE"] /
                       max(1, row["distribution"]["transitions"]) for row in updates]
    near_zero = [value < .05 for value in release_probabilities]
    sustained_near_zero = any(left and right for left, right in zip(near_zero, near_zero[1:]))
    base = validations[0]["metrics"]["overall"]["environment"]
    final = validations[-1]["metrics"]["overall"]["environment"]
    final_release = validations[-1]["metrics"]["overall"]["distribution"]["heads"]["release"]
    completion_decline = (base["completed"] > 0 and final["completed"] < .5 * base["completed"])
    queue_growth = final["external_queue_mean"] > base["external_queue_mean"] + 1.0
    final_dual_collapse = final_release["policy_probability"]["RELEASE"] < .05 and \
        final_release["requested"]["RELEASE"] / max(1, validations[-1]["metrics"]["overall"]["distribution"]["transitions"]) < .05
    collapse = bool(sustained_near_zero or (final_dual_collapse and (completion_decline or queue_growth)))
    return {"conservative_release_collapse": collapse, "criteria": {
        "release_probability_near_zero_threshold": .05, "sustained_near_zero": sustained_near_zero,
        "final_dual_collapse": final_dual_collapse, "completion_decline": completion_decline, "queue_growth": queue_growth,
    }, "training_release_probability": release_probabilities, "training_requested_release_rate": requested_rates}


def _learning_trend(updates: list[dict]) -> str:
    burdens = [row["environment"]["teacher_interventions_per_100_release_intents"] for row in updates]
    valid = [value for value in burdens if value is not None]
    if len(valid) < 2: return "INCONCLUSIVE"
    if valid[-1] <= .95 * valid[0]: return "IMPROVING"
    if len(valid) >= 3 and all(value > 1.10 * valid[0] for value in valid[-3:]): return "WORSENING"
    return "INCONCLUSIVE"


def _markdown(report: dict) -> str:
    lines = ["# V3 L1 Full-Teacher PPO Short Pilot", "", "Evidence type: newly_run. Fresh V3 categorical actor/critic; no continuous/D3/D4 checkpoint was loaded.", "",
             f"- Training samples: {report['training_samples']}", f"- Updates / episodes: 6 / 24", f"- Safety: {'PASS' if report['safety_passed'] else 'FAIL'}",
             f"- Conservative release collapse: {str(report['collapse']['conservative_release_collapse']).lower()}",
             f"- Learning trend: {report['learning_trend']}", ""]
    for validation in report["validation"]:
        overall = validation["metrics"]["overall"]
        release = overall["distribution"]["heads"]["release"]
        lines += [f"## {validation['label']}", "", f"- Teacher burden overall / warm: {overall['environment']['teacher_interventions_per_100_release_intents']} / {validation['metrics']['warm_start']['environment']['teacher_interventions_per_100_release_intents']}",
                  f"- Per-head correction rate: {overall['distribution']['per_head_correction_rate']}",
                  f"- Release requested WAIT/RELEASE: {release['requested']}; teacher target: {release['teacher_target']}",
                  f"- RELEASE→WAIT / WAIT→RELEASE: {release['RELEASE_to_WAIT']} / {release['WAIT_to_RELEASE']}",
                  f"- False-release / false-wait: {release['false_release_rate']} / {release['false_wait_rate']}",
                  f"- Completion / throughput / queue mean: {overall['environment']['completed']} / {overall['environment']['throughput_per_second']} / {overall['environment']['external_queue_mean']}",
                  f"- Shield burden: {overall['environment']['shield_interventions_per_100_release_intents']}", ""]
    lines += ["## PPO health", ""]
    for row in report["updates"]:
        health = row["ppo"]
        warm = row.get("by_scenario", {}).get("warm_start", {}).get("environment", {}).get("teacher_interventions_per_100_release_intents")
        lines.append(f"- Update {row['update'] + 1}: teacher overall/warm={row['environment']['teacher_interventions_per_100_release_intents']} / {warm}; completion={row['environment']['completed']}; queue={row['environment']['external_queue_mean']}; actor={health['actor_loss']:.6f}, value={health['value_loss']:.6f}, KL={health['approx_kl']:.6f}, clip={health['clip_fraction']:.6f}, entropy={health['entropy']:.6f}, grad_norm={health['grad_norm']:.6f}")
    lines += ["", f"V3 L1 Short Pilot = {'PASS' if report['pilot_passed'] else 'FAIL'}",
              f"conservative_release_collapse = {str(report['collapse']['conservative_release_collapse']).lower()}",
              f"longer_L1_diagnostic_allowed = {str(report['longer_L1_diagnostic_allowed']).lower()}",
              "formal_L1_training_allowed = false", ""]
    return "\n".join(lines)


def run_v3_l1_short_pilot(*, c0_report: str | Path, teacher_dataset: str | Path, output_dir: str | Path) -> dict:
    protocol = V3L1ShortPilotProtocol()
    audit = audit_teacher_dataset(teacher_dataset)
    if not audit.passed: raise RuntimeError("V3 pilot dataset audit failed: " + "; ".join(audit.errors[:10]))
    mu_hat, medium = _read_frozen_capacity(c0_report)
    if mu_hat != MU_HAT or abs(sum(medium) / 2.0 - LAMBDA) > 1e-12:
        raise RuntimeError("V3 pilot traffic calibration drift")
    partition = _read_seed_partition(teacher_dataset)
    destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=True)
    root = l1_full_teacher_config(medium_rate=LAMBDA, reset_mode="warm_start_rings", demand_model="finite_cohort",
                                  warm_start_uavs=8, cohort_uavs=8, horizon_seconds=float(protocol.max_steps))
    if root.physics_dt != .2 or root.safety_sampling_dt != 1.0:
        raise RuntimeError("V3 pilot physics/safety sampling configuration drift")
    torch.manual_seed(2401); np.random.seed(2401)
    policy = SharedPPO(LOW_OBS_V2.dimension, teacher_bc_weight=protocol.teacher_bc_weight)
    validation_records: list[dict] = []
    validation_raw: list[dict] = []
    update_path = destination / "v3_l1_short_pilot_updates.jsonl"
    validation_path = destination / "v3_l1_short_pilot_validations.jsonl"
    progress_path = destination / "v3_l1_short_pilot.progress.json"

    def progress(status: str, completed_updates: int) -> None:
        _write_json(progress_path, {"stage": "V3_L1_short_pilot", "status": status,
                                    "completed_updates": completed_updates, "target_updates": protocol.updates,
                                    "validation_labels": [record["label"] for record in validation_records],
                                    "fresh_initialization": True, "resume_checkpoint": None,
                                    "safety_passed_so_far": all(
                                        group["environment"][metric] == 0 for record in validation_records
                                        for group in record["metrics"].values()
                                        for metric in ("collisions", "nmac", "hard_separation_violations")
                                    )})

    progress("initializing", 0)
    print("V3 L1 short pilot initialized; fresh actor/critic, no resume checkpoint", flush=True)

    def validate(label: str, update: int) -> None:
        records = _validation_records(policy, partition, update=update, protocol=protocol)
        state = _model_state(policy)
        metrics = _validation_metrics(records, state)
        validation_records.append({"label": label, "update": update, "metrics": metrics})
        _write_jsonl(validation_path, validation_records)
        for _, summary in records:
            validation_raw.append({**asdict(summary), "label": label, "scenario": _scenario_name(summary)})
        if any(value != 0 for group in metrics.values() for value in (
            group["environment"]["collisions"], group["environment"]["nmac"], group["environment"]["hard_separation_violations"],
        )):
            raise RuntimeError("V3 pilot validation safety hard gate failed; ActionTapes were retained if an invariant fired")
        progress(f"validation_{label}_complete", max(0, update))
        print(f"validation {label} complete", flush=True)

    validate("baseline", 0)
    train_rows: list[dict] = []
    per_update: list[dict] = []
    seed_cursor = 0
    for update in range(protocol.updates):
        state = _model_state(policy)
        tasks = []
        designs = []
        for episode in range(protocol.episodes_per_update):
            design_index = update * protocol.episodes_per_update + episode
            config, design = _episode_design(design_index, medium_rate=LAMBDA, horizon_seconds=float(protocol.max_steps))
            seed = partition.train[seed_cursor % len(partition.train)]; seed_cursor += 1
            tasks.append((state, config, seed, update, episode, protocol.max_steps, False, str(destination / "safety_action_tapes")))
            designs.append(design)
        collected = _collect(tasks, protocol.rollout_workers)
        buffer = RolloutBuffer()
        for (transitions, summary), design in zip(collected, designs):
            if summary.collisions or summary.nmac or summary.hard_separation_violations:
                raise RuntimeError("V3 pilot training safety hard gate failed; ActionTape retained")
            for transition in transitions: buffer.add(transition)
            train_rows.append({**asdict(summary), **design, "scenario": _scenario_name(summary), "update": update})
        before_update = _records_metrics(collected, state)
        ppo = policy.update(buffer, epochs=protocol.ppo_epochs, minibatch_size=protocol.minibatch_size)
        if not all(np.isfinite(value) for value in ppo.values()):
            raise FloatingPointError("V3 pilot PPO numerical health failure")
        per_update.append({"update": update, **before_update, "by_scenario": _training_groups(collected, state), "ppo": ppo})
        _write_jsonl(update_path, per_update)
        progress(f"update_{update + 1}_complete", update + 1)
        print(f"update {update + 1}/6 complete", flush=True)
        if update + 1 in (3, 6): validate(f"update{update + 1}", update + 1)
    collapse = _collapse_diagnostic(per_update, validation_records)
    learning = _learning_trend(per_update)
    safety = all(row["environment"][metric] == 0 for row in per_update for metric in ("collisions", "nmac", "hard_separation_violations")) and \
             all(group["environment"][metric] == 0 for record in validation_records for group in record["metrics"].values()
                 for metric in ("collisions", "nmac", "hard_separation_violations"))
    ppo_health = all(np.isfinite(row["ppo"][key]) for row in per_update for key in
                     ("actor_loss", "value_loss", "entropy", "entropy_release", "entropy_speed", "entropy_merge_time", "approx_kl", "clip_fraction", "explained_variance", "grad_norm"))
    last = validation_records[-1]["metrics"]["overall"]["environment"]
    baseline = validation_records[0]["metrics"]["overall"]["environment"]
    completion_healthy = last["completed"] >= .5 * baseline["completed"] if baseline["completed"] else last["external_queue_max"] <= baseline["external_queue_max"] + 1
    pilot_passed = bool(safety and ppo_health and not collapse["conservative_release_collapse"] and completion_healthy and learning != "WORSENING")
    checkpoint = destination / "v3_l1_short_pilot.pt"
    save_checkpoint(checkpoint, policy, root, stage="V3_L1_short_pilot", fresh_initialization=True,
                    prohibited_resume_sources=["continuous", "D3", "D4"])
    report = {"stage": "V3_L1_short_pilot", "evidence_type": "newly_run", "protocol": asdict(protocol),
              "fresh_initialization": True, "resume_checkpoint": None, "prohibited_resume_sources": ["continuous", "D3", "D4"],
              "frozen": {"action_schema": "V3_factorized_categorical", "head_cardinalities": [2, 5, 5], "teacher": "FULL",
                         "mu_hat": MU_HAT, "lambda": LAMBDA, "priority": 1, "mission": "standard", "initial_soc": .8,
                         "geometry": "nominal", "physics_dt": .2, "safety_sampling_dt": 1.0, "teacher_bc_weight": .05},
              "c0_report": str(c0_report), "teacher_dataset": str(teacher_dataset), "teacher_dataset_audit": asdict(audit),
              "training_samples": sum(row["distribution"]["transitions"] for row in per_update), "updates": per_update,
              "validation": validation_records, "collapse": collapse, "learning_trend": learning, "safety_passed": safety,
              "ppo_health_passed": ppo_health, "completion_throughput_healthy": completion_healthy,
              "pilot_passed": pilot_passed, "longer_L1_diagnostic_allowed": pilot_passed, "formal_L1_training_allowed": False,
              "checkpoint": str(checkpoint)}
    _write_jsonl(destination / "v3_l1_short_pilot_train_episodes.jsonl", train_rows)
    _write_jsonl(destination / "v3_l1_short_pilot_validation_episodes.jsonl", validation_raw)
    _write_json(destination / "v3_l1_short_pilot.json", report)
    (destination / "v3_l1_short_pilot_summary.md").write_text(_markdown(report), encoding="utf-8")
    return report


def backfill_training_group_metrics(output_dir: str | Path) -> dict:
    """Add missing per-update warm/from-empty metrics from immutable episode rows.

    This is an analysis-only repair for an already completed pilot: it does
    not touch the policy, checkpoint, seed ordering, transitions, or labels.
    """
    destination = Path(output_dir)
    report_path = destination / "v3_l1_short_pilot.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (destination / "v3_l1_short_pilot_train_episodes.jsonl").read_text(encoding="utf-8").splitlines() if line]
    summary_fields = {field.name for field in fields(L1EpisodeSummary)}
    for update in report["updates"]:
        summaries = [L1EpisodeSummary(**{key: row[key] for key in summary_fields})
                     for row in rows if row["update"] == update["update"]]
        update["by_scenario"] = {
            "warm_start": {"environment": _summary_metrics([row for row in summaries if row.reset_mode == "warm_start_rings"])},
            "from_empty": {"environment": _summary_metrics([row for row in summaries if row.reset_mode == "flow_from_empty"])},
        }
    _write_json(report_path, report)
    (destination / "v3_l1_short_pilot_summary.md").write_text(_markdown(report), encoding="utf-8")
    _write_jsonl(destination / "v3_l1_short_pilot_updates.jsonl", report["updates"])
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run the fresh frozen V3 L1 PPO short pilot; no resume support.")
    parser.add_argument("--c0-report", required=True); parser.add_argument("--teacher-dataset", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        report = run_v3_l1_short_pilot(c0_report=args.c0_report, teacher_dataset=args.teacher_dataset, output_dir=args.output)
    except Exception as error:
        _write_json(Path(args.output) / "v3_l1_short_pilot.progress.json", {
            "stage": "V3_L1_short_pilot", "status": "FAILED", "error_type": type(error).__name__, "error": str(error),
        })
        raise
    print(json.dumps({key: report[key] for key in ("training_samples", "safety_passed", "learning_trend", "collapse", "pilot_passed", "longer_L1_diagnostic_allowed", "formal_L1_training_allowed")}, indent=2))


if __name__ == "__main__":
    _main()
