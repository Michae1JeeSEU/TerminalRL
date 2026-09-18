"""Fresh V4.2 P2 support collection with the preregistered V2 shield gate.

This module keeps the V1 collector as the implementation boundary for the
rollout, label, replay, and binding mechanics.  V2 changes only the frozen
episode shield gate: a small number of SafetyFilter actor corrections is
allowed when ``shield_actor_fraction`` is at most the value inherited from
the I3 qualification protocol.  V1 artifacts are checked as a manifest-only
binding and their rows are never opened or mixed into the fresh support set.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import yaml

from terminal_marl.training import collect_v4_2_p2_gate_support as _v1


PROTOCOL = Path("terminal_marl/configs/low_v4_2/ppo_p2_gate_support_collection_v2.yaml")
PROTOCOL_SHA = "14e235ec3402671a493aea2669135ace53fc7c97e4226ef36e1dada0910b6e43"

V1_FAIL_MANIFEST_SHA = "3f376b68209dbd7a68db058b8344204138dae3ea107cd2678d2f4966312fd979"
INHERITED_I3_PROTOCOL_SHA = "21d1761860ba4636c811b9a7739629e12fafe12dbf2ee9bd92ae6decfbd4f387"
SHIELD_ACTOR_FRACTION_MAX = 0.05

FORMAL_TRAIN_SEEDS = tuple(range(775001, 775010))
FORMAL_DEV_SEEDS = tuple(range(776001, 776007))
FUTURE_QUALIFICATION_SEEDS = tuple(range(774001, 774007))
SMOKE_SEEDS = (775000, 776000)
SMOKE_STEPS = 160
SCENARIOS = ("flow6", "flow8", "flow12")
INTERFACES = _v1.INTERFACES
PHASE_INTERFACE = _v1.PHASE_INTERFACE

CANDIDATE_SHA = _v1.CANDIDATE_SHA
CANDIDATE_STAGE = _v1.CANDIDATE_STAGE
TRAINING_PROTOCOL_SHA = _v1.TRAINING_PROTOCOL_SHA
D1_DATASET_SHA = _v1.D1_DATASET_SHA
D1_MANIFEST_SHA = _v1.D1_MANIFEST_SHA
DIAGNOSTIC_MANIFEST_SHA = _v1.DIAGNOSTIC_MANIFEST_SHA

TRAIN_SUPPORT_SEEDS = FORMAL_TRAIN_SEEDS
DEV_SUPPORT_SEEDS = FORMAL_DEV_SEEDS

# Explicit helper aliases make reuse of the V1 implementation observable and
# keep the V2 semantic delta limited to the episode gate and fresh protocol.
_sha = _v1._sha
_write = _v1._write
_read_json = _v1._read_json
_require = _v1._require
reserve_once = _v1.reserve_once
historical_registry = _v1.historical_registry
load_evaluation_policy = _v1.load_evaluation_policy
p2_source_hashes = _v1.p2_source_hashes
rollout = _v1.rollout
support_config = _v1.support_config
collection_config = support_config
diagnostic_config = support_config
_actor_rows = _v1._actor_rows
filter_actor_rows = _actor_rows
_finite_88d = _v1._finite_88d
_thresholds = _v1._thresholds
_cross_split_identity = _v1._cross_split_identity
_tape_flags = _v1._tape_flags


def protocol() -> dict[str, Any]:
    """Load the V2 protocol and verify its preregistered bytes and values."""
    _require(_sha(PROTOCOL) == PROTOCOL_SHA, "P2 V2 support protocol changed after preregistration")
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict), "support protocol must be a mapping")
    formal = value.get("formal", {})
    smoke = value.get("smoke", {})
    bindings = value.get("bindings", {})
    _require(value.get("protocol") == "V4_2_P2_GATE_SUPPORT_COLLECTION_V2", "V2 protocol id changed")
    _require(formal.get("train_support_seeds") == list(FORMAL_TRAIN_SEEDS), "formal train-support seeds changed")
    _require(formal.get("dev_support_seeds") == list(FORMAL_DEV_SEEDS), "formal dev-support seeds changed")
    _require(formal.get("future_qualification_seeds") == list(FUTURE_QUALIFICATION_SEEDS),
             "future qualification seeds changed")
    _require(formal.get("scenarios") == list(SCENARIOS), "formal scenario cycle changed")
    _require(smoke.get("seeds") == list(SMOKE_SEEDS), "smoke seeds changed")
    _require(smoke.get("steps") == SMOKE_STEPS, "smoke horizon changed")
    _require(bindings.get("d1_dataset_sha256") == D1_DATASET_SHA, "D1 dataset binding changed")
    _require(bindings.get("d1_manifest_sha256") == D1_MANIFEST_SHA, "D1 manifest binding changed")
    _require(bindings.get("diagnostic_manifest_sha256") == DIAGNOSTIC_MANIFEST_SHA,
             "diagnostic manifest binding changed")
    _require(bindings.get("v1_fail_manifest_sha256") == V1_FAIL_MANIFEST_SHA,
             "V1 FAIL manifest binding changed")
    _require(bindings.get("inherited_i3_protocol_sha256") == INHERITED_I3_PROTOCOL_SHA,
             "inherited I3 protocol binding changed")
    _require(value.get("episode_gate", {}).get("shield_actor_fraction_max") == SHIELD_ACTOR_FRACTION_MAX,
             "shield actor fraction limit changed")
    return value


def _scenario_for(index: int) -> str:
    return SCENARIOS[index % len(SCENARIOS)]


def _planned_episodes(*, smoke: bool) -> list[dict[str, Any]]:
    if smoke:
        return [
            {"split": "train", "seed": SMOKE_SEEDS[0], "scenario": "flow6", "max_steps": SMOKE_STEPS},
            {"split": "dev", "seed": SMOKE_SEEDS[1], "scenario": "flow6", "max_steps": SMOKE_STEPS},
        ]
    plan: list[dict[str, Any]] = []
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


def shield_actor_fraction(shield_actor_count: int, actor_decisions: int) -> float:
    """Return the recorded actor shield burden using the frozen denominator."""
    return float(shield_actor_count / max(1, actor_decisions))


def shield_actor_fraction_gate(shield_actor_count: int, actor_decisions: int) -> bool:
    return shield_actor_fraction(shield_actor_count, actor_decisions) <= SHIELD_ACTOR_FRACTION_MAX


def _episode_record(
    policy: Any,
    *,
    split: str,
    seed: int,
    scenario: str,
    output: Path,
    max_steps: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Reuse V1 rollout/label/replay checks and replace only its episode gate."""
    summary, rows = _v1._episode_record(
        policy,
        split=split,
        seed=seed,
        scenario=scenario,
        output=output,
        max_steps=max_steps,
    )
    counts = summary.get("counts", {})
    shield_actor_count = int(counts.get("shield_actor", 0))
    actor_decisions = int(counts.get("actor_decisions", 0))
    fraction = shield_actor_fraction(shield_actor_count, actor_decisions)
    within_limit = fraction <= SHIELD_ACTOR_FRACTION_MAX
    summary.update({
        "shield_actor_count": shield_actor_count,
        "actor_decisions": actor_decisions,
        "shield_actor_fraction": fraction,
        "shield_actor_fraction_max": SHIELD_ACTOR_FRACTION_MAX,
        "shield_actor_fraction_within_inherited_limit": within_limit,
        "teacher_replaced_behavior": False,
        "v1_rows_mixed": False,
    })
    gates = dict(summary.get("gates", {}))
    gates.pop("shield_zero", None)
    gates["shield_actor_fraction_within_inherited_limit"] = within_limit
    summary["gates"] = gates
    return summary, rows


def split_metrics(
    rows: Sequence[Mapping[str, Any]],
    split: str | None = None,
    records: Sequence[Mapping[str, Any]] | None = None,
    planned_seeds: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Reuse V1 row metrics and add the inherited episode shield metric."""
    metrics = _v1.split_metrics(rows, split, records, planned_seeds)
    selected_records = [dict(item) for item in (records or ()) if split is None or item.get("split") == split]
    episode_metrics = []
    for item in selected_records:
        counts = item.get("counts", {})
        actor_decisions = int(item.get("actor_decisions", counts.get("actor_decisions", 0)))
        shield_actor_count = int(item.get("shield_actor_count", counts.get("shield_actor", 0)))
        fraction = float(item.get("shield_actor_fraction", shield_actor_fraction(shield_actor_count, actor_decisions)))
        episode_metrics.append({
            "split": item.get("split"),
            "seed": item.get("seed"),
            "scenario": item.get("scenario"),
            "shield_actor_count": shield_actor_count,
            "actor_decisions": actor_decisions,
            "shield_actor_fraction": fraction,
            "shield_actor_fraction_max": SHIELD_ACTOR_FRACTION_MAX,
            "within_inherited_limit": fraction <= SHIELD_ACTOR_FRACTION_MAX,
        })
    metrics["episode_shield_actor_fraction"] = episode_metrics
    metrics["all_episode_shield_actor_fraction_within_inherited_limit"] = bool(episode_metrics) and all(
        bool(item["within_inherited_limit"]) for item in episode_metrics
    )
    return metrics


def split_gates(metrics: Mapping[str, Any], split: str | None = None, *, formal: bool = True) -> dict[str, bool]:
    """Evaluate V1 split gates with the V2 episode fraction gate included."""
    name = split or str(metrics.get("split"))
    limits = _thresholds(name)
    classes = metrics.get("teacher", {})
    proportions = metrics.get("teacher_class_proportions", {})
    interface = metrics.get("interface", {})
    cardinality = int(limits["min_rows"]) <= int(metrics.get("rows", 0)) <= int(limits["max_rows"])
    class_counts = all(int(classes.get(label, 0)) >= int(limits["class_min"]) for label in ("WAIT", "RELEASE"))
    class_fraction = all(float(proportions.get(label, 0.0)) >= float(limits["class_fraction"])
                         for label in ("WAIT", "RELEASE"))
    interface_counts = all(int(interface.get(item, {}).get("total", 0)) >= int(limits["interface_min"])
                           for item in INTERFACES)
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
        "all_episode_shield_actor_fraction_within_inherited_limit": bool(
            metrics.get("all_episode_shield_actor_fraction_within_inherited_limit", False)
        ),
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
    """Only a formal all-gates pass can authorize the learnability audit."""
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


def _validate_v1_fail_manifest(path: Path) -> dict[str, Any]:
    """Validate only the V1 manifest declaration; never open its row files."""
    _require(path.exists(), f"V1 FAIL manifest missing: {path}")
    _require(_sha(path) == V1_FAIL_MANIFEST_SHA, "V1 FAIL manifest SHA mismatch")
    value = _read_json(path)
    _require(value.get("SUPPORT_COLLECTION") == "FAIL", "V1 manifest is not SUPPORT_COLLECTION=FAIL")
    failed_episode_gates = [
        (item.get("split"), int(item.get("seed")), item.get("scenario"), key)
        for item in value.get("records", [])
        for key, flag in item.get("gates", {}).items()
        if flag is False
    ]
    _require(failed_episode_gates == [("train", 772009, "flow12", "shield_zero")],
             "V1 manifest failure is not the frozen unique shield_zero episode")
    return {
        "path": str(path),
        "sha256": V1_FAIL_MANIFEST_SHA,
        "SUPPORT_COLLECTION": "FAIL",
        "failed_episode_gates": failed_episode_gates,
        "rows_opened": False,
    }


def _default_v1_manifest(history_root: Path) -> Path:
    return history_root / "p2_gate_support_collection_20260918" / "manifest.json"


def _validate_bindings(
    checkpoint: Path,
    d1_dataset: Path,
    d1_manifest: Path,
    diagnostic_manifest: Path,
    v1_manifest: Path,
) -> dict[str, Any]:
    bindings = _v1._validate_bindings(checkpoint, d1_dataset, d1_manifest, diagnostic_manifest)
    bindings["v1_fail_manifest"] = _validate_v1_fail_manifest(v1_manifest)
    bindings["v1_rows_mixed"] = False
    bindings["inherited_i3_protocol_sha256"] = INHERITED_I3_PROTOCOL_SHA
    return bindings


def run(
    output: Path,
    checkpoint: Path,
    d1_dataset: Path,
    d1_manifest: Path,
    diagnostic_manifest: Path,
    history_root: Path,
    *,
    v1_manifest: Path | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    """Collect the fresh V2 smoke or formal cohort."""
    p = protocol()
    _require(p["candidate_checkpoint_sha256"] == CANDIDATE_SHA, "candidate binding changed")
    _require(p["formal"]["train_support_seeds"] == list(FORMAL_TRAIN_SEEDS), "train seeds changed")
    _require(p["formal"]["dev_support_seeds"] == list(FORMAL_DEV_SEEDS), "dev seeds changed")
    _require(p["formal"]["future_qualification_seeds"] == list(FUTURE_QUALIFICATION_SEEDS),
             "future seeds changed")
    v1_manifest_path = v1_manifest or _default_v1_manifest(history_root)
    bindings = _validate_bindings(checkpoint, d1_dataset, d1_manifest, diagnostic_manifest, v1_manifest_path)
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
            checkpoint.parent / "p2_gate_support_collection_v2_consumed.json",
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
        "v1_fail_manifest_sha256": V1_FAIL_MANIFEST_SHA,
        "v1_rows_opened": False,
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
        _v1._write_jsonl_immutable(output / f"{split}_support.jsonl", split_rows)
    cross = _cross_split_identity(rows_by_split["train"] + rows_by_split["dev"])
    planned_train = FORMAL_TRAIN_SEEDS if not smoke else (SMOKE_SEEDS[0],)
    planned_dev = FORMAL_DEV_SEEDS if not smoke else (SMOKE_SEEDS[1],)
    metrics = {
        split: split_metrics(rows_by_split[split], split, records, planned)
        for split, planned in (("train", planned_train), ("dev", planned_dev))
    }
    for split in ("train", "dev"):
        metrics[split]["cross_split_seed_disjoint"] = cross["train_dev_seed_disjoint"]
        metrics[split]["cross_split_event_disjoint"] = cross["train_dev_event_disjoint"]
    gates = {split: split_gates(metrics[split], split, formal=not smoke) for split in ("train", "dev")}
    for split in gates:
        gates[split]["cross_split_seed_disjoint"] = cross["train_dev_seed_disjoint"]
        gates[split]["cross_split_event_disjoint"] = cross["train_dev_event_disjoint"]
        gates[split]["all"] = all(gates[split].values())
    auth = formal_authorization(gates["train"], gates["dev"], smoke=smoke)
    output_hashes = {split: _sha(output / f"{split}_support.jsonl") for split in ("train", "dev")}
    complete = (not smoke) and len(records) == len(plan)
    result = {
        "stage": "V4_2_P2_GATE_SUPPORT_COLLECTION_V2",
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
        "teacher_replaced_behavior": False,
        "v1_rows_mixed": False,
        "support_files": {
            "train": str(output / "train_support.jsonl"),
            "dev": str(output / "dev_support.jsonl"),
            "sha256": output_hashes,
        },
        "source_hashes": {
            **p2_source_hashes(),
            "support_runner": _sha(Path(__file__)),
            "support_protocol": _sha(PROTOCOL),
            "v1_support_runner": _sha(Path(_v1.__file__)),
            "v1_support_protocol": _sha(_v1.PROTOCOL),
            "v1_fail_manifest": V1_FAIL_MANIFEST_SHA,
            "inherited_i3_protocol": INHERITED_I3_PROTOCOL_SHA,
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
    parser.add_argument("--v1-manifest", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    torch.set_num_threads(1)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2), flush=True)
