"""Fresh natural-flow support collection for the V4.2 P2 gate audit.

The collector is deliberately a data-collection boundary rather than a
training entry point.  The formal candidate policy remains authoritative for
all execution requests; the V4.1 Teacher is invoked only by the label-only
sidecar in :func:`student_rollout_v4_2.rollout`.  Historical D1 and diagnostic
artifacts are bound read-only and are never read as training rows here.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from terminal_marl.core.safety_trace import SafetyTraceRecorder
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training.evaluate_v4_2_p2_v4 import (
    CANDIDATE_SHA,
    CANDIDATE_STAGE,
    TRAINING_PROTOCOL_SHA,
    load_evaluation_policy,
)
from terminal_marl.training.run_v4_2_i2_dagger_preflight import _policy_tape_logprob_exact
from terminal_marl.training.run_v4_2_ppo_p2_resume_v4 import p2_config, p2_source_hashes
from terminal_marl.training.run_v4_2_v2_1_formal_i0_r3 import _write
from terminal_marl.training.run_v4_2_v2_1_formal_i0_r4 import historical_registry
from terminal_marl.training.run_v4_2_v2_1_i1_v2 import _sha, reserve_once
from terminal_marl.training.student_rollout_v4_2 import rollout


PROTOCOL = Path("terminal_marl/configs/low_v4_2/ppo_p2_gate_support_collection_v1.yaml")
PROTOCOL_SHA = "8e614001e2475c2e1821036b88aeea93aa14afec468320eb7fd5792afca82aac"

FORMAL_TRAIN_SEEDS = tuple(range(772001, 772010))
FORMAL_DEV_SEEDS = tuple(range(773001, 773007))
FUTURE_QUALIFICATION_SEEDS = tuple(range(774001, 774007))
SMOKE_SEEDS = (772000, 773000)
SMOKE_STEPS = 160
SCENARIOS = ("flow6", "flow8", "flow12")
INTERFACES = ("o2m", "m2i", "i2f")
PHASE_INTERFACE = {"ring_outer": "o2m", "ring_middle": "m2i", "ring_inner": "i2f"}

D1_DATASET_SHA = "270e54ef6ed0c36f535e2a0f25cef7cf9fdd6c91016ee295e7e336fb800cacb0"
D1_MANIFEST_SHA = "a6319c291e5c91e155fd1c7fb20ae95dcdd4973733abfe83fc10c5e6006f472c"
DIAGNOSTIC_MANIFEST_SHA = "b290e2d5f69749e1109bff2c5f348458732a22fbe8c0ce34c15ce98880eefc93"

TRAIN_SUPPORT_SEEDS = FORMAL_TRAIN_SEEDS
DEV_SUPPORT_SEEDS = FORMAL_DEV_SEEDS


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def protocol() -> dict[str, Any]:
    """Load the frozen collection protocol after checking its byte hash."""
    _require(_sha(PROTOCOL) == PROTOCOL_SHA, "P2 support protocol changed after preregistration")
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict), "support protocol must be a mapping")
    _require(value.get("formal", {}).get("train_support_seeds") == list(FORMAL_TRAIN_SEEDS),
             "formal train-support seeds changed")
    _require(value.get("formal", {}).get("dev_support_seeds") == list(FORMAL_DEV_SEEDS),
             "formal dev-support seeds changed")
    _require(value.get("formal", {}).get("future_qualification_seeds") == list(FUTURE_QUALIFICATION_SEEDS),
             "future qualification seeds changed")
    _require(value.get("formal", {}).get("scenarios") == list(SCENARIOS), "formal scenario cycle changed")
    _require(value.get("smoke", {}).get("seeds") == list(SMOKE_SEEDS), "smoke seeds changed")
    _require(value.get("smoke", {}).get("steps") == SMOKE_STEPS, "smoke horizon changed")
    _require(value.get("bindings", {}).get("d1_dataset_sha256") == D1_DATASET_SHA, "D1 dataset binding changed")
    _require(value.get("bindings", {}).get("d1_manifest_sha256") == D1_MANIFEST_SHA, "D1 manifest binding changed")
    _require(value.get("bindings", {}).get("diagnostic_manifest_sha256") == DIAGNOSTIC_MANIFEST_SHA,
             "diagnostic manifest binding changed")
    return value


def support_config(n: int):
    """Return the P2 natural-flow config with label-only Teacher sidecar."""
    _require(n in {6, 8, 12}, "support scenario must be flow6, flow8, or flow12")
    # This is the only semantic config transformation permitted by the
    # protocol.  In particular, the Teacher never becomes an execution
    # scheduler and the P2 reward remains unchanged.
    from dataclasses import replace

    return replace(
        p2_config(n),
        experiment_mode="fixed_teacher_heavy",
        scheduler_mode="execution",
        teacher_enabled=True,
        teacher_mode="teacher_label_only",
        teacher_release_label_enabled=False,
    )


# Descriptive aliases keep the collection API explicit for callers and tests.
collection_config = support_config
diagnostic_config = support_config


def _scenario_for(index: int) -> str:
    return SCENARIOS[index % len(SCENARIOS)]


def _planned_episodes(*, smoke: bool) -> list[dict[str, Any]]:
    if smoke:
        return [
            {"split": "train", "seed": SMOKE_SEEDS[0], "scenario": "flow6", "max_steps": SMOKE_STEPS},
            {"split": "dev", "seed": SMOKE_SEEDS[1], "scenario": "flow6", "max_steps": SMOKE_STEPS},
        ]
    plan = []
    for split, seeds in (("train", FORMAL_TRAIN_SEEDS), ("dev", FORMAL_DEV_SEEDS)):
        for index, seed in enumerate(seeds):
            scenario = _scenario_for(index)
            config = support_config(int(scenario.removeprefix("flow")))
            plan.append({
                "split": split,
                "seed": int(seed),
                "scenario": scenario,
                "max_steps": int(config.max_simulation_seconds / config.dt),
            })
    return plan


def _actor_rows(rows: Sequence[Mapping[str, Any]], *, split: str, scenario: str) -> list[dict[str, Any]]:
    """Keep only actor decisions on the three gate interfaces."""
    result: list[dict[str, Any]] = []
    for source in rows:
        if not bool(source.get("actor_mask")):
            continue
        interface = PHASE_INTERFACE.get(str(source.get("phase")))
        if interface is None:
            continue
        row = dict(source)
        row["support_split"] = split
        row["scenario"] = scenario
        row["interface"] = interface
        result.append(row)
    return result


filter_actor_rows = _actor_rows


def _finite_88d(row: Mapping[str, Any]) -> bool:
    values = row.get("observation_v2_1")
    if not isinstance(values, list) or len(values) != 88:
        return False
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.shape == (88,) and np.isfinite(array).all())


def _thresholds(split: str) -> dict[str, float | int]:
    if split == "train":
        return {"min_rows": 3000, "max_rows": 30000, "class_min": 600, "class_fraction": 0.20,
                "interface_min": 500, "interface_class_min": 120}
    if split == "dev":
        return {"min_rows": 1800, "max_rows": 20000, "class_min": 350, "class_fraction": 0.20,
                "interface_min": 250, "interface_class_min": 60}
    raise ValueError(f"unknown support split: {split}")


def split_metrics(
    rows: Sequence[Mapping[str, Any]],
    split: str | None = None,
    records: Sequence[Mapping[str, Any]] | None = None,
    planned_seeds: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Aggregate frozen support gates and descriptive completion metrics."""
    selected = [dict(row) for row in rows if split is None or row.get("support_split") == split]
    teacher = Counter("RELEASE" if int(row.get("teacher_gate", -1)) == 1 else "WAIT"
                      for row in selected if int(row.get("teacher_gate", -1)) in {0, 1})
    interfaces: dict[str, dict[str, int]] = {}
    for name in INTERFACES:
        group = [row for row in selected if row.get("interface") == name]
        interfaces[name] = {
            "total": len(group),
            "WAIT": sum(int(row.get("teacher_gate", -1)) == 0 for row in group),
            "RELEASE": sum(int(row.get("teacher_gate", -1)) == 1 for row in group),
        }
    record_rows = [dict(item) for item in (records or ()) if split is None or item.get("split") == split]
    requested = sum(int(item.get("requested", item.get("config_requested", 0))) for item in record_rows)
    completed = sum(int(item.get("completed", 0)) for item in record_rows)
    execution_complete = all(bool(item.get("episode_execution_complete", False)) for item in record_rows) if record_rows else False
    episode_gates = bool(record_rows) and all(
        bool(item.get("gates")) and all(bool(value) for value in item["gates"].values())
        for item in record_rows
    )
    expected_seeds = set(int(value) for value in (planned_seeds or ()))
    observed_seeds = {int(row["seed"]) for row in selected if "seed" in row}
    events = [str(row.get("decision_event_id")) for row in selected]
    return {
        "split": split,
        "rows": len(selected),
        "teacher": {"WAIT": int(teacher["WAIT"]), "RELEASE": int(teacher["RELEASE"])},
        "teacher_class_proportions": {
            name: (float(teacher[name] / len(selected)) if selected else 0.0)
            for name in ("WAIT", "RELEASE")
        },
        "interface": interfaces,
        "schema_88d_finite": bool(selected) and all(_finite_88d(row) for row in selected),
        "decision_event_id_unique": len(events) == len(set(events)) and all(event not in {"None", ""} for event in events),
        "observed_seeds": sorted(observed_seeds),
        "planned_seeds": sorted(expected_seeds),
        "seed_coverage": observed_seeds == expected_seeds if expected_seeds else bool(observed_seeds),
        "episodes_planned": len(record_rows),
        "episodes_execution_complete": sum(bool(item.get("episode_execution_complete", False)) for item in record_rows),
        "all_planned_episodes_execution_complete": execution_complete,
        "all_episode_gates": episode_gates,
        "completion_requested": {
            "completed": completed,
            "requested": requested,
            "fraction": (float(completed / requested) if requested else None),
        },
    }


def split_gates(metrics: Mapping[str, Any], split: str | None = None, *, formal: bool = True) -> dict[str, bool]:
    """Evaluate the fixed cardinality/class/interface/schema gates."""
    name = split or str(metrics.get("split"))
    limits = _thresholds(name)
    classes = metrics.get("teacher", {})
    proportions = metrics.get("teacher_class_proportions", {})
    interface = metrics.get("interface", {})
    cardinality = int(limits["min_rows"]) <= int(metrics.get("rows", 0)) <= int(limits["max_rows"])
    class_counts = all(int(classes.get(label, 0)) >= int(limits["class_min"]) for label in ("WAIT", "RELEASE"))
    class_fraction = all(float(proportions.get(label, 0.0)) >= float(limits["class_fraction"]) for label in ("WAIT", "RELEASE"))
    interface_counts = all(int(interface.get(item, {}).get("total", 0)) >= int(limits["interface_min"]) for item in INTERFACES)
    interface_class_counts = all(
        int(interface.get(item, {}).get(label, 0)) >= int(limits["interface_class_min"])
        for item in INTERFACES for label in ("WAIT", "RELEASE")
    )
    gates = {
        "cardinality": cardinality if formal else True,
        "teacher_class_minimums": class_counts if formal else True,
        "teacher_class_proportions": class_fraction if formal else True,
        "interface_minimums": interface_counts if formal else True,
        "interface_class_minimums": interface_class_counts if formal else True,
        "schema_88d_finite": bool(metrics.get("schema_88d_finite", False)),
        "decision_event_id_unique": bool(metrics.get("decision_event_id_unique", False)),
        "seed_coverage": bool(metrics.get("seed_coverage", False)),
        "all_planned_episodes_execution_complete": bool(metrics.get("all_planned_episodes_execution_complete", False)),
        "all_episode_gates": bool(metrics.get("all_episode_gates", False)),
    }
    gates["all"] = all(gates.values())
    return gates


def support_gates(
    rows: Sequence[Mapping[str, Any]],
    split: str,
    records: Sequence[Mapping[str, Any]] | None = None,
    planned_seeds: Sequence[int] | None = None,
    *,
    formal: bool = True,
) -> dict[str, Any]:
    metrics = split_metrics(rows, split, records, planned_seeds)
    return {"metrics": metrics, "gates": split_gates(metrics, split, formal=formal)}


def formal_authorization(
    train_gates: Mapping[str, Any],
    dev_gates: Mapping[str, Any],
    *,
    smoke: bool = False,
) -> dict[str, Any]:
    """Return the protocol's deliberately narrow post-collection authority."""
    all_pass = (not smoke) and bool(train_gates.get("all")) and bool(dev_gates.get("all"))
    return {
        "SUPPORT_COLLECTION": "PASS" if all_pass else "FAIL",
        "SUPPORT_FROZEN": bool(all_pass),
        "LEARNABILITY_AUDIT_ALLOWED": bool(all_pass),
        "GATE_REPAIR_ALLOWED": False,
        "PPO_ALLOWED": False,
        "curriculum_promotion_allowed": False,
    }


authorization = formal_authorization


def _write_jsonl_immutable(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    """Write one support artifact once and return its immutable byte SHA."""
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n")
    return _sha(path)


def _tape_flags(tape_path: Path) -> dict[str, bool]:
    """Read-only checks over the action tape's label-only provenance."""
    payload = _read_json(tape_path)
    rows = payload.get("action_tape", [])
    label_rows = [row for row in rows if row.get("teacher_label_only")]
    # ``rollout`` performs the per-action authoritative checks before it
    # returns.  The tape provides an independent label-only marker for every
    # step on which labels were recorded.
    return {
        "teacher_label_only": bool(label_rows) and all(row.get("teacher_label_only") is True for row in label_rows),
        "student_authoritative": bool(rows) and all(not row.get("teacher_used", False) for row in rows),
    }


def _episode_record(
    policy: Any,
    *,
    split: str,
    seed: int,
    scenario: str,
    output: Path,
    max_steps: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    n = int(scenario.removeprefix("flow"))
    config = support_config(n)
    steps, summary = rollout(
        policy,
        config,
        seed=seed,
        max_steps=max_steps,
        output=output,
        deterministic=False,
        collect_labels=True,
    )
    labels_path = output / "labels.jsonl"
    raw_labels = [json.loads(line) for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = _actor_rows(raw_labels, split=split, scenario=scenario)
    replay = SafetyTraceRecorder.replay_action_tape(TerminalParallelEnv(config), summary["action_tape"])
    summary["replay_exact"] = bool(replay["exact"])
    summary["requested_logprob_exact"] = bool(_policy_tape_logprob_exact(policy, config, summary["action_tape"]))
    tape_flags = _tape_flags(Path(summary["action_tape"]))
    safety_ok = not any(int(value) for value in summary.get("safety", {}).values())
    frozen = bool(summary.get("terminal")) or int(summary.get("steps", 0)) == int(max_steps)
    summary.update({
        "split": split,
        "scenario": scenario,
        "seed": int(seed),
        "requested": int(config.total_requested_uavs),
        "completed": int(summary.get("completed", 0)),
        "episode_execution_complete": bool(frozen and safety_ok),
        "terminal_or_frozen_horizon": bool(frozen),
        "safety_all_zero": bool(safety_ok),
        "shield_zero": int(summary.get("counts", {}).get("shield", 0)) == 0,
        "label_rows": len(raw_labels),
        "actor_rows": len(rows),
        # Label sidecars intentionally include non-actor world rows.  The
        # retained dataset must contain actor/interface rows only, but it
        # must not require every raw sidecar row to be an actor row.
        "label_actor_rows_only": bool(raw_labels) and bool(rows) and all(
            bool(row.get("actor_mask")) and row.get("interface") in INTERFACES for row in rows
        ),
        "teacher_label_only": bool(tape_flags["teacher_label_only"]),
        "student_authoritative": bool(tape_flags["student_authoritative"]),
        "training_called": False,
        "max_steps": int(max_steps),
    })
    summary["gates"] = {
        "terminal_or_frozen_horizon": summary["terminal_or_frozen_horizon"],
        "safety_all_zero": summary["safety_all_zero"],
        "shield_zero": summary["shield_zero"],
        "ActionTape_exact_replay": summary["replay_exact"],
        "requested_logprob_exact": summary["requested_logprob_exact"],
        "label_actor_rows": summary["label_actor_rows_only"] and summary["actor_rows"] > 0,
        "teacher_label_only": summary["teacher_label_only"],
        "student_authoritative": summary["student_authoritative"],
        "no_policy_update": summary["training_called"] is False,
    }
    return summary, rows


def _validate_bindings(
    checkpoint: Path,
    d1_dataset: Path,
    d1_manifest: Path,
    diagnostic_manifest: Path,
) -> dict[str, str]:
    _require(_sha(checkpoint) == CANDIDATE_SHA, "candidate checkpoint SHA mismatch")
    _require(_sha(d1_dataset) == D1_DATASET_SHA, "D1 dataset SHA mismatch")
    _require(_sha(d1_manifest) == D1_MANIFEST_SHA, "D1 manifest SHA mismatch")
    _require(_sha(diagnostic_manifest) == DIAGNOSTIC_MANIFEST_SHA, "diagnostic manifest SHA mismatch")
    return {
        "candidate_checkpoint_sha256": CANDIDATE_SHA,
        "d1_dataset_sha256": D1_DATASET_SHA,
        "d1_manifest_sha256": D1_MANIFEST_SHA,
        "diagnostic_manifest_sha256": DIAGNOSTIC_MANIFEST_SHA,
        "d1_rows_mixed": False,
        "diagnostic_rows_mixed": False,
    }


def _cross_split_identity(rows: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    train = [row for row in rows if row.get("support_split") == "train"]
    dev = [row for row in rows if row.get("support_split") == "dev"]
    train_seeds, dev_seeds = {int(row["seed"]) for row in train}, {int(row["seed"]) for row in dev}
    train_events, dev_events = {str(row["decision_event_id"]) for row in train}, {str(row["decision_event_id"]) for row in dev}
    return {
        "train_dev_seed_disjoint": not (train_seeds & dev_seeds),
        "train_dev_event_disjoint": not (train_events & dev_events),
    }


def run(
    output: Path,
    checkpoint: Path,
    d1_dataset: Path,
    d1_manifest: Path,
    diagnostic_manifest: Path,
    history_root: Path,
    *,
    smoke: bool = False,
) -> dict[str, Any]:
    """Collect one fresh smoke or formal train/dev support cohort."""
    p = protocol()
    _require(p["candidate_checkpoint_sha256"] == CANDIDATE_SHA, "candidate binding changed")
    _require(p["formal"]["train_support_seeds"] == list(FORMAL_TRAIN_SEEDS), "train seeds changed")
    _require(p["formal"]["dev_support_seeds"] == list(FORMAL_DEV_SEEDS), "dev seeds changed")
    _require(p["formal"]["future_qualification_seeds"] == list(FUTURE_QUALIFICATION_SEEDS), "future seeds changed")
    bindings = _validate_bindings(checkpoint, d1_dataset, d1_manifest, diagnostic_manifest)
    candidate, _ = load_evaluation_policy(
        checkpoint,
        expected_sha=CANDIDATE_SHA,
        expected_stage=CANDIDATE_STAGE,
        expected_formal=True,
        expected_protocol_sha=TRAINING_PROTOCOL_SHA,
        expected_source_hashes=p2_source_hashes(),
    )
    output.mkdir(parents=True, exist_ok=False)
    plan = _planned_episodes(smoke=smoke)
    train_seeds = set(FORMAL_TRAIN_SEEDS)
    dev_seeds = set(FORMAL_DEV_SEEDS)
    future_seeds = set(FUTURE_QUALIFICATION_SEEDS)
    _require(not (train_seeds & dev_seeds or train_seeds & future_seeds or dev_seeds & future_seeds),
             "support seed groups are not mutually exclusive")
    if not smoke:
        history = historical_registry(history_root, output)
        _require(not ((train_seeds | dev_seeds | future_seeds) & history["seeds"]),
                 "support or future qualification seed already consumed")
        reserve_once(
            checkpoint.parent / "p2_gate_support_collection_consumed.json",
            {"protocol_sha256": PROTOCOL_SHA, "train_seeds": sorted(train_seeds),
             "dev_seeds": sorted(dev_seeds), "future_qualification_seeds": sorted(future_seeds),
             "output": str(output)},
        )
        registry = {"historical_seed_count": len(history["seeds"]), "checked_future_seeds": True}
    else:
        registry = {"historical_seed_count": None, "checked_future_seeds": False}
    _write(output / "pre_reset_registry.json", {
        "protocol_sha256": PROTOCOL_SHA,
        "smoke": smoke,
        "planned_episodes": plan,
        "formal_seed_reservation": not smoke,
        "consumed_marker_created": not smoke,
        "historical_registry": registry,
        "d1_dataset_rows_opened": False,
        "diagnostic_rows_opened": False,
    })

    records: list[dict[str, Any]] = []
    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "dev": []}
    for item in plan:
        split, seed, scenario = str(item["split"]), int(item["seed"]), str(item["scenario"])
        episode_output = output / f"{split}_{seed}_{scenario}"
        try:
            summary, rows = _episode_record(
                candidate, split=split, seed=seed, scenario=scenario,
                output=episode_output, max_steps=int(item["max_steps"]),
            )
        except Exception as exc:
            # Preserve the raw terminal condition as a failed planned episode
            # and stop immediately on a safety/collection exception.
            records.append({"split": split, "seed": seed, "scenario": scenario,
                            "episode_execution_complete": False, "exception": repr(exc),
                            "requested": int(scenario.removeprefix("flow")), "completed": 0,
                            "gates": {"collection_exception": False}})
            _write(output / "progress.json", {"completed": len(records), "total": len(plan), "records": records})
            break
        summary["rows"] = len(rows)
        records.append(summary)
        rows_by_split[split].extend(rows)
        _write(output / "progress.json", {"completed": len(records), "total": len(plan), "records": records})
        if any(int(value) for value in summary.get("safety", {}).values()):
            break

    for split, split_rows in rows_by_split.items():
        _write_jsonl_immutable(output / f"{split}_support.jsonl", split_rows)
    cross = _cross_split_identity(rows_by_split["train"] + rows_by_split["dev"])
    planned_train = FORMAL_TRAIN_SEEDS if not smoke else (SMOKE_SEEDS[0],)
    planned_dev = FORMAL_DEV_SEEDS if not smoke else (SMOKE_SEEDS[1],)
    # Cross-split independence is a global gate.  Add it to both metric views
    # so the split-level gate report is self-contained.
    metrics = {
        split: split_metrics(rows_by_split[split], split, records, planned)
        for split, planned in (("train", planned_train), ("dev", planned_dev))
    }
    metrics["train"]["cross_split_seed_disjoint"] = cross["train_dev_seed_disjoint"]
    metrics["train"]["cross_split_event_disjoint"] = cross["train_dev_event_disjoint"]
    metrics["dev"]["cross_split_seed_disjoint"] = cross["train_dev_seed_disjoint"]
    metrics["dev"]["cross_split_event_disjoint"] = cross["train_dev_event_disjoint"]
    gates = {
        split: split_gates(metrics[split], split, formal=not smoke)
        for split in ("train", "dev")
    }
    for split in gates:
        gates[split]["cross_split_seed_disjoint"] = cross["train_dev_seed_disjoint"]
        gates[split]["cross_split_event_disjoint"] = cross["train_dev_event_disjoint"]
        gates[split]["all"] = all(gates[split].values())
    auth = formal_authorization(gates["train"], gates["dev"], smoke=smoke)
    output_hashes = {split: _sha(output / f"{split}_support.jsonl") for split in ("train", "dev")}
    complete = (not smoke) and len(records) == len(plan)
    result = {
        "stage": "V4_2_P2_GATE_SUPPORT_COLLECTION",
        "smoke": smoke,
        "protocol_sha256": PROTOCOL_SHA,
        "candidate_checkpoint_sha256": CANDIDATE_SHA,
        "bindings": bindings,
        "records": records,
        "metrics": metrics,
        "gates": gates,
        "authorization": auth,
        "SUPPORT_COLLECTION": auth["SUPPORT_COLLECTION"],
        "SUPPORT_FROZEN": auth["SUPPORT_FROZEN"],
        "LEARNABILITY_AUDIT_ALLOWED": auth["LEARNABILITY_AUDIT_ALLOWED"],
        "GATE_REPAIR_ALLOWED": False,
        "PPO_ALLOWED": False,
        "curriculum_promotion_allowed": False,
        "formal_episode_plan_complete": complete,
        "training_called": False,
        "student_authoritative": True,
        "teacher_label_only": True,
        "support_files": {
            "train": str(output / "train_support.jsonl"),
            "dev": str(output / "dev_support.jsonl"),
            "sha256": output_hashes,
        },
        "source_hashes": {
            **p2_source_hashes(),
            "support_runner": _sha(Path(__file__)),
            "support_protocol": _sha(PROTOCOL),
        },
        "d1_rows_mixed": False,
        "diagnostic_rows_mixed": False,
        "future_qualification_rollouts": 0,
        "future_qualification_consumed_marker": False,
    }
    _write(output / "manifest.json", result)
    print(json.dumps({"SUPPORT_COLLECTION": result["SUPPORT_COLLECTION"], "smoke": smoke,
                      "records": len(records), "train_rows": metrics["train"]["rows"],
                      "dev_rows": metrics["dev"]["rows"]}, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "--output-dir", dest="output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--d1-dataset", type=Path, required=True)
    parser.add_argument("--d1-manifest", type=Path, required=True)
    parser.add_argument("--diagnostic-manifest", type=Path, required=True)
    parser.add_argument("--history-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    torch.set_num_threads(1)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2), flush=True)
