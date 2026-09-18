"""Fresh calibrated-candidate P2 gate support collection.

V3 repeats the natural-flow support boundary under the fixed-temperature
calibrated candidate distribution.  It reuses V2's episode, row-filtering,
replay, and support-gate mechanics, but has its own frozen protocol, seed
groups, predecessor bindings, and one-shot reservation marker.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import yaml

from terminal_marl.training import collect_v4_2_p2_gate_support_v2 as v2
from terminal_marl.training import qualify_v4_2_p2_gate_student_v1 as qualification


PROTOCOL = Path("terminal_marl/configs/low_v4_2/ppo_p2_gate_support_collection_v3.yaml")
PROTOCOL_SHA = "88ec0a7b7ef6bcc771f9388490ba43a378e0f99ebc03e1753c6cf58293ed78a1"
STAGE = "V4_2_P2_GATE_SUPPORT_COLLECTION_V3"
CANDIDATE_SHA = "61e03e02e3fb204d2db4e814097543905f51e1a3772626fdac1d7bc02ab8ca6b"
CANDIDATE_ARCHITECTURE_ID = "v4_2_hierarchical_gate_temperature_calibrated_residual_adapter_v3"
CANDIDATE_TEMPERATURE = 2.5
SUPPORT_V2_MANIFEST_SHA = "09402508bc02155c1369de477dc320d5ca596768d4de6461f4a0526f2c7d9304"
FAILED_QUALIFICATION_MANIFEST_SHA = "c781ffc1230ee6c86db0a127ae8d17b014571c3b391feb27bf06643b9e5ed3c8"
QUALIFICATION_PROTOCOL_SHA = "8064bebd115cad253e22ea019f678bb0a5c9085bb8a47989319fd9956a54aa65"
QUALIFICATION_STAGE = "V4_2_P2_GATE_STUDENT_QUALIFICATION_V1"
RESERVATION_MARKER = "p2_gate_support_collection_v3_consumed.json"
SHIELD_ACTOR_FRACTION_MAX = v2.SHIELD_ACTOR_FRACTION_MAX
FORMAL_TRAIN_SEEDS = tuple(range(780001, 780010))
FORMAL_DEV_SEEDS = tuple(range(781001, 781007))
FUTURE_QUALIFICATION_SEEDS = tuple(range(782001, 782007))
SMOKE_SEEDS = (780000, 781000)
SMOKE_SCENARIOS = ("flow6", "flow6")
SMOKE_STEPS = 160
SCENARIOS = v2.SCENARIOS
INTERFACES = v2.INTERFACES

# Reused behavior is explicit and limited to the V2 collector boundary.
_sha = v2._sha
_write = v2._write
_read_json = v2._read_json
_require = v2._require
reserve_once = v2.reserve_once
historical_registry = v2.historical_registry
support_config = v2.support_config
collection_config = support_config
diagnostic_config = support_config
_episode_record = v2._episode_record
filter_actor_rows = v2.filter_actor_rows
_actor_rows = v2._actor_rows
_finite_88d = v2._finite_88d
_thresholds = v2._thresholds
split_metrics = v2.split_metrics
split_gates = v2.split_gates
support_gates = v2.support_gates
formal_authorization = v2.formal_authorization


def protocol() -> dict[str, Any]:
    """Load and validate the byte-frozen V3 support protocol."""
    _require(PROTOCOL.is_file(), f"V3 support protocol missing: {PROTOCOL}")
    if PROTOCOL_SHA:
        _require(_sha(PROTOCOL) == PROTOCOL_SHA, "P2 V3 support protocol changed after preregistration")
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict), "V3 support protocol must be a mapping")
    _require(value.get("protocol") == STAGE, "V3 support protocol id changed")
    _require(value.get("candidate_checkpoint_sha256") == CANDIDATE_SHA,
             "V3 candidate checkpoint binding changed")
    _require(value.get("candidate_architecture_id") == CANDIDATE_ARCHITECTURE_ID,
             "V3 candidate architecture binding changed")
    _require(float(value.get("candidate_temperature")) == CANDIDATE_TEMPERATURE,
             "V3 candidate temperature binding changed")
    predecessors = value.get("predecessors", {})
    expected_predecessors = {
        "support_v2_manifest_sha256": SUPPORT_V2_MANIFEST_SHA,
        "failed_qualification_manifest_sha256": FAILED_QUALIFICATION_MANIFEST_SHA,
        "qualification_protocol_sha256": QUALIFICATION_PROTOCOL_SHA,
    }
    for key, expected in expected_predecessors.items():
        _require(predecessors.get(key) == expected, f"V3 predecessor binding changed: {key}")
    formal = value.get("formal", {})
    _require(tuple(formal.get("train_support_seeds", ())) == FORMAL_TRAIN_SEEDS,
             "V3 train-support seeds changed")
    _require(tuple(formal.get("dev_support_seeds", ())) == FORMAL_DEV_SEEDS,
             "V3 dev-support seeds changed")
    _require(tuple(formal.get("future_qualification_seeds", ())) == FUTURE_QUALIFICATION_SEEDS,
             "V3 future qualification seeds changed")
    _require(tuple(formal.get("scenarios", ())) == SCENARIOS, "V3 scenario cycle changed")
    _require(formal.get("scenario_assignment") == "cycle_in_list_order_per_split",
             "V3 scenario assignment changed")
    _require(formal.get("deterministic") is False, "V3 collection is not stochastic")
    _require(formal.get("teacher_mode") == "teacher_label_only",
             "V3 Teacher mode changed")
    _require(formal.get("teacher_release_label_enabled") is False,
             "V3 Teacher release-label binding changed")
    _require(formal.get("scheduler_mode") == "execution", "V3 scheduler mode changed")
    _require(formal.get("consumed_marker") == RESERVATION_MARKER,
             "V3 reservation marker changed")
    _require(formal.get("qualification_seed_range_audited") == "780000-782999",
             "V3 qualification seed audit range changed")
    smoke = value.get("smoke", {})
    _require(tuple(smoke.get("seeds", ())) == SMOKE_SEEDS, "V3 smoke seeds changed")
    _require(tuple(smoke.get("scenarios", ())) == SMOKE_SCENARIOS,
             "V3 smoke scenarios changed")
    _require(smoke.get("steps") == SMOKE_STEPS, "V3 smoke horizon changed")
    _require(smoke.get("reserve_formal_seeds") is False
             and smoke.get("create_consumed_marker") is False
             and smoke.get("authorization") is False,
             "V3 smoke authorization changed")
    bindings = value.get("bindings", {})
    for key, expected in {
        "support_v2_manifest_sha256": SUPPORT_V2_MANIFEST_SHA,
        "failed_qualification_manifest_sha256": FAILED_QUALIFICATION_MANIFEST_SHA,
        "qualification_protocol_sha256": QUALIFICATION_PROTOCOL_SHA,
        "observation_schema": "phaseA_low_obs_v2_1_retry_state",
        "observation_dimension": 88,
        "action_schema": "low_action_schema_v4_2_hierarchical",
        "source_usage": "manifest_only_predecessor_bindings_no_rows_mixed",
    }.items():
        _require(bindings.get(key) == expected, f"V3 binding changed: {key}")
    episode_gate = value.get("episode_gate", {})
    _require(episode_gate.get("shield_actor_fraction_max") == SHIELD_ACTOR_FRACTION_MAX,
             "V3 shield actor fraction limit changed")
    _require(episode_gate.get("inherited_from") == "i3_v2_1_qualification.yaml",
             "V3 inherited episode gate changed")
    collection = value.get("collection", {})
    for key, expected in {
        "observation_schema": "phaseA_low_obs_v2_1_retry_state",
        "observation_dimension": 88,
        "action_schema": "low_action_schema_v4_2_hierarchical",
        "output_files": ["train_support.jsonl", "dev_support.jsonl"],
        "replay": "exact_action_tape",
        "requested_logprob": "exact",
        "natural_flow": True,
        "no_policy_update": True,
        "student_authoritative": True,
        "teacher_label_only": True,
        "teacher_replaced_behavior": False,
        "calibrated_candidate_distribution": True,
        "failed_qualification_rows_mixed": False,
    }.items():
        _require(collection.get(key) == expected, f"V3 collection binding changed: {key}")
    _require(value.get("authorization", {}).get("always_false") == [
        "GATE_REPAIR_ALLOWED", "PPO_ALLOWED", "curriculum_promotion_allowed",
    ], "V3 authorization boundary changed")
    return value


def _validate_support_v2_manifest(path: Path) -> dict[str, Any]:
    """Validate only V2 status metadata; never open V2 support row files."""
    _require(path.is_file(), f"support V2 manifest missing: {path}")
    _require(_sha(path) == SUPPORT_V2_MANIFEST_SHA, "support V2 manifest SHA mismatch")
    value = _read_json(path)
    expected = {
        "stage": "V4_2_P2_GATE_SUPPORT_COLLECTION_V2",
        "SUPPORT_COLLECTION": "PASS",
        "SUPPORT_FROZEN": True,
        "LEARNABILITY_AUDIT_ALLOWED": True,
        "GATE_REPAIR_ALLOWED": False,
        "PPO_ALLOWED": False,
        "curriculum_promotion_allowed": False,
        "future_qualification_rollouts": 0,
        "future_qualification_consumed_marker": False,
    }
    for key, wanted in expected.items():
        _require(value.get(key) == wanted, f"support V2 predecessor changed: {key}")
    for key in (
        "support_v2_rows_mixed",
        "failed_qualification_rows_mixed",
        "failed_qualification_labels_mixed",
        "v1_rows_mixed",
    ):
        if key in value:
            _require(value.get(key) is False, f"support V2 predecessor row-mixing changed: {key}")
    return {
        "path": str(path),
        "sha256": SUPPORT_V2_MANIFEST_SHA,
        "stage": value["stage"],
        "SUPPORT_COLLECTION": value["SUPPORT_COLLECTION"],
        "SUPPORT_FROZEN": value["SUPPORT_FROZEN"],
        "LEARNABILITY_AUDIT_ALLOWED": value["LEARNABILITY_AUDIT_ALLOWED"],
        "rows_opened": False,
    }


def _validate_failed_qualification_manifest(path: Path) -> dict[str, Any]:
    """Validate terminal qualification metadata without opening its rows."""
    _require(path.is_file(), f"failed qualification manifest missing: {path}")
    _require(_sha(path) == FAILED_QUALIFICATION_MANIFEST_SHA,
             "failed qualification manifest SHA mismatch")
    value = _read_json(path)
    expected = {
        "stage": QUALIFICATION_STAGE,
        "status": "FAIL",
        "P2_GATE_STUDENT_QUALIFICATION": "FAIL",
        "P2_EFFECTIVENESS_EVALUATION_ALLOWED": False,
        "PPO_ALLOWED": False,
        "P3_ALLOWED": False,
        "curriculum_promotion_allowed": False,
        "checkpoint_promotable": False,
        "qualification_only": True,
        "recovered_from_completed_artifacts": True,
        "formal_rollouts_repeated": False,
        "formal_seed_resets": 0,
        "environment_rollout": True,
        "teacher_called": True,
    }
    for key, wanted in expected.items():
        _require(value.get(key) == wanted, f"failed qualification binding changed: {key}")
    for key in (
        "support_v2_rows_mixed",
        "failed_qualification_rows_mixed",
        "failed_qualification_labels_mixed",
        "v1_rows_mixed",
    ):
        if key in value:
            _require(value.get(key) is False, f"failed qualification row-mixing changed: {key}")
    completeness = value.get("completeness", {})
    _require(completeness.get("episodes") == 6 and completeness.get("expected_episodes") == 6,
             "failed qualification episode completion changed")
    _require(completeness.get("requested") == 52 and completeness.get("completed") == 52,
             "failed qualification requested completion changed")
    _require(len(value.get("records", [])) == 6, "failed qualification record count changed")
    sampled_checks = value.get("sampled_action_checks", {})
    sampled_failed = [key for key in ("false_WAIT_rate", "false_RELEASE_rate")
                      if sampled_checks.get(key) is False]
    _require(sampled_failed == ["false_WAIT_rate"],
             "failed qualification sampled-action failure shape changed")
    _require(sampled_checks.get("false_RELEASE_rate") is True,
             "failed qualification false-RELEASE gate changed")
    probability_checks = value.get("probability_checks", {})
    probability_failed = [key for key in (
        "macro_F1", "WAIT_recall", "RELEASE_recall", "brier", "binary_nll",
    ) if probability_checks.get(key) is False]
    _require(probability_failed == ["RELEASE_recall"],
             "failed qualification probability failure shape changed")
    metrics = value.get("sampled_action_metrics", {})
    _require(abs(float(metrics.get("false_wait_rate")) - 106 / 318) < 1e-9,
             "failed qualification false-WAIT metric changed")
    _require(abs(float(metrics.get("false_release_rate")) - 29 / 665) < 1e-9,
             "failed qualification false-RELEASE metric changed")
    probabilities = value.get("probability_metrics", {})
    _require(abs(float(probabilities.get("RELEASE_recall")) - 0.625786) < 1e-6,
             "failed qualification RELEASE recall metric changed")
    _require(value.get("row_checks", {}).get("all") is True,
             "failed qualification row integrity changed")
    return {
        "path": str(path),
        "sha256": FAILED_QUALIFICATION_MANIFEST_SHA,
        "stage": value["stage"],
        "status": value["status"],
        "rows_opened": False,
        "sampled_failures": sampled_failed,
        "probability_failures": probability_failed,
        "recovered_from_completed_artifacts": True,
    }


def _load_calibrated_candidate(checkpoint: Path):
    """Load the fixed calibrated policy through the qualification loader."""
    policy, payload = qualification._load_calibrated_checkpoint(
        checkpoint, expected_sha256=CANDIDATE_SHA,
    )
    _require(payload.get("architecture_id") == CANDIDATE_ARCHITECTURE_ID,
             "calibrated candidate architecture changed")
    _require(float(payload.get("temperature")) == CANDIDATE_TEMPERATURE,
             "calibrated candidate temperature changed")
    _require(payload.get("STUDENT_QUALIFICATION_ALLOWED") is True,
             "calibrated candidate qualification authorization changed")
    return policy, payload


def _scenario_for(index: int) -> str:
    return SCENARIOS[index % len(SCENARIOS)]


def _planned_episodes(*, smoke: bool) -> list[dict[str, Any]]:
    if smoke:
        return [
            {"split": "train", "seed": SMOKE_SEEDS[0], "scenario": SMOKE_SCENARIOS[0],
             "max_steps": SMOKE_STEPS},
            {"split": "dev", "seed": SMOKE_SEEDS[1], "scenario": SMOKE_SCENARIOS[1],
             "max_steps": SMOKE_STEPS},
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


def _validate_seed_groups(*, smoke: bool) -> None:
    train = set(FORMAL_TRAIN_SEEDS)
    dev = set(FORMAL_DEV_SEEDS)
    future = set(FUTURE_QUALIFICATION_SEEDS)
    _require(not (train & dev or train & future or dev & future),
             "V3 formal seed groups are not mutually exclusive")
    if smoke:
        _require(not (set(SMOKE_SEEDS) & (train | dev | future)),
                 "V3 smoke seed overlaps formal seed groups")


def _validate_candidate_inputs(
    checkpoint: Path,
    support_v2_manifest: Path,
    failed_qualification_manifest: Path,
) -> tuple[dict[str, Any], dict[str, Any], Any, dict[str, Any]]:
    support_binding = _validate_support_v2_manifest(support_v2_manifest)
    failed_binding = _validate_failed_qualification_manifest(failed_qualification_manifest)
    candidate, checkpoint_payload = _load_calibrated_candidate(checkpoint)
    return support_binding, failed_binding, candidate, checkpoint_payload


def _reserve_formal(
    checkpoint: Path,
    history_root: Path,
    output: Path,
) -> tuple[Path, dict[str, Any]]:
    marker = checkpoint.parent / RESERVATION_MARKER
    _require(not marker.exists(), f"V3 consumed marker already exists: {marker}")
    history = historical_registry(history_root, output)
    consumed = set(int(value) for value in history.get("seeds", set()))
    formal = set(FORMAL_TRAIN_SEEDS) | set(FORMAL_DEV_SEEDS) | set(FUTURE_QUALIFICATION_SEEDS)
    _require(not (formal & consumed), "V3 support or future qualification seed already consumed")
    reserve_once(marker, {
        "protocol_sha256": PROTOCOL_SHA or _sha(PROTOCOL),
        "train_support_seeds": sorted(FORMAL_TRAIN_SEEDS),
        "dev_support_seeds": sorted(FORMAL_DEV_SEEDS),
        "future_qualification_seeds": sorted(FUTURE_QUALIFICATION_SEEDS),
        "output": str(output),
        "candidate_checkpoint_sha256": CANDIDATE_SHA,
    })
    return marker, {"historical_seed_count": len(consumed), "checked_future_seeds": True}


def run(
    output: Path,
    checkpoint: Path,
    support_v2_manifest: Path,
    failed_qualification_manifest: Path,
    history_root: Path,
    *,
    smoke: bool = False,
) -> dict[str, Any]:
    """Collect the calibrated-candidate V3 smoke or formal cohort."""
    protocol_value = protocol()
    output = Path(output)
    checkpoint = Path(checkpoint)
    support_v2_manifest = Path(support_v2_manifest)
    failed_qualification_manifest = Path(failed_qualification_manifest)
    history_root = Path(history_root)
    _require(not output.exists(), f"V3 support output already exists: {output}")
    _validate_seed_groups(smoke=smoke)
    support_binding, failed_binding, candidate, checkpoint_payload = _validate_candidate_inputs(
        checkpoint, support_v2_manifest, failed_qualification_manifest,
    )
    if not smoke:
        _require(history_root.is_dir(), f"V3 history root missing: {history_root}")
        _require(not (checkpoint.parent / RESERVATION_MARKER).exists(),
                 f"V3 consumed marker already exists: {checkpoint.parent / RESERVATION_MARKER}")
    output.mkdir(parents=True, exist_ok=False)
    plan = _planned_episodes(smoke=smoke)
    marker: Path | None = None
    registry: dict[str, Any]
    if not smoke:
        marker, registry = _reserve_formal(checkpoint, history_root, output)
    else:
        registry = {"historical_seed_count": None, "checked_future_seeds": False}
    _write(output / "pre_reset_registry.json", {
        "protocol_sha256": PROTOCOL_SHA or _sha(PROTOCOL),
        "smoke": smoke,
        "planned_episodes": plan,
        "formal_seed_reservation": not smoke,
        "consumed_marker_created": not smoke,
        "historical_registry": registry,
        "support_v2_manifest_sha256": SUPPORT_V2_MANIFEST_SHA,
        "failed_qualification_manifest_sha256": FAILED_QUALIFICATION_MANIFEST_SHA,
        "support_v2_rows_opened": False,
        "failed_qualification_rows_opened": False,
        "failed_qualification_labels_opened": False,
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
            _write(output / "progress.json", {"completed": len(records), "total": len(plan),
                                                "records": records})
            break
        summary["rows"] = len(rows)
        records.append(summary)
        rows_by_split[split].extend(rows)
        _write(output / "progress.json", {"completed": len(records), "total": len(plan),
                                            "records": records})
        if any(int(value) for value in summary.get("safety", {}).values()):
            break

    for split, split_rows in rows_by_split.items():
        _v1 = v2._v1
        _v1._write_jsonl_immutable(output / f"{split}_support.jsonl", split_rows)
    cross = v2._cross_split_identity(rows_by_split["train"] + rows_by_split["dev"])
    planned_train = FORMAL_TRAIN_SEEDS if not smoke else (SMOKE_SEEDS[0],)
    planned_dev = FORMAL_DEV_SEEDS if not smoke else (SMOKE_SEEDS[1],)
    metrics = {
        split: split_metrics(rows_by_split[split], split, records, planned)
        for split, planned in (("train", planned_train), ("dev", planned_dev))
    }
    for split in ("train", "dev"):
        metrics[split]["cross_split_seed_disjoint"] = cross["train_dev_seed_disjoint"]
        metrics[split]["cross_split_event_disjoint"] = cross["train_dev_event_disjoint"]
    gates = {split: split_gates(metrics[split], split, formal=not smoke)
             for split in ("train", "dev")}
    for split in gates:
        gates[split]["cross_split_seed_disjoint"] = cross["train_dev_seed_disjoint"]
        gates[split]["cross_split_event_disjoint"] = cross["train_dev_event_disjoint"]
        gates[split]["all"] = all(gates[split].values())
    auth = formal_authorization(gates["train"], gates["dev"], smoke=smoke)
    output_hashes = {split: _sha(output / f"{split}_support.jsonl") for split in ("train", "dev")}
    complete = (not smoke) and len(records) == len(plan)
    result = {
        "stage": STAGE,
        "smoke": smoke,
        "protocol_sha256": PROTOCOL_SHA or _sha(PROTOCOL),
        "candidate_checkpoint_sha256": CANDIDATE_SHA,
        "candidate_architecture_id": CANDIDATE_ARCHITECTURE_ID,
        "candidate_temperature": CANDIDATE_TEMPERATURE,
        "checkpoint_payload": {
            "stage": checkpoint_payload.get("stage"),
            "architecture_id": checkpoint_payload.get("architecture_id"),
            "temperature": checkpoint_payload.get("temperature"),
            "qualification_only": checkpoint_payload.get("qualification_only"),
            "checkpoint_promotable": checkpoint_payload.get("checkpoint_promotable"),
        },
        "predecessor_bindings": {
            "support_v2_manifest_sha256": SUPPORT_V2_MANIFEST_SHA,
            "failed_qualification_manifest_sha256": FAILED_QUALIFICATION_MANIFEST_SHA,
            "qualification_protocol_sha256": QUALIFICATION_PROTOCOL_SHA,
        },
        "bindings": {
            "support_v2_manifest": support_binding,
            "failed_qualification_manifest": failed_binding,
            "support_v2_rows_mixed": False,
            "failed_qualification_rows_mixed": False,
            "failed_qualification_labels_mixed": False,
        },
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
        "optimizer_called": False,
        "student_authoritative": True,
        "teacher_label_only": True,
        "teacher_replaced_behavior": False,
        "support_v2_rows_opened": False,
        "failed_qualification_rows_opened": False,
        "failed_qualification_labels_opened": False,
        "support_v2_rows_mixed": False,
        "failed_qualification_rows_mixed": False,
        "failed_qualification_labels_mixed": False,
        "future_qualification_rollouts": 0,
        "future_qualification_consumed_marker": False,
        "formal_marker": str(marker) if marker is not None else None,
        "formal_marker_created": marker is not None,
        "support_files": {
            "train": str(output / "train_support.jsonl"),
            "dev": str(output / "dev_support.jsonl"),
            "sha256": output_hashes,
        },
        "source_hashes": {
            "support_runner": _sha(Path(__file__)),
            "support_protocol": _sha(PROTOCOL),
            "support_v2_runner": _sha(Path(v2.__file__)),
            "support_v2_protocol": _sha(v2.PROTOCOL),
            "qualification_runner": _sha(Path(qualification.__file__)),
            "qualification_protocol": _sha(qualification.PROTOCOL),
            "support_v2_manifest": SUPPORT_V2_MANIFEST_SHA,
            "failed_qualification_manifest": FAILED_QUALIFICATION_MANIFEST_SHA,
            "candidate_checkpoint": CANDIDATE_SHA,
        },
    }
    _write(output / "manifest.json", result)
    print(json.dumps({"stage": STAGE, "SUPPORT_COLLECTION": result["SUPPORT_COLLECTION"],
                      "smoke": smoke, "records": len(records),
                      "train_rows": metrics["train"]["rows"],
                      "dev_rows": metrics["dev"]["rows"]}, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "--output-dir", dest="output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--support-v2-manifest", type=Path, required=True)
    parser.add_argument("--failed-qualification-manifest", type=Path, required=True)
    parser.add_argument("--history-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    torch.set_num_threads(1)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2), flush=True)
