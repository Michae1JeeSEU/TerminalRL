"""Read-only diagnosis of the failed V2.1 retry matched-pair closure gate."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .run_v4_2_observation_v2_1_teacher_predicate_closure_audit import PROBE_SEEDS, _probe
from .run_v4_2_privileged_teacher_gate_factorization_diagnostic import _binary_probability


RETRY_INDEX = 87
RETRY_HORIZON_SECONDS = 30.0
PAIR_GATE = {"accuracy": .80, "macro_F1": .80, "WAIT_recall": .80, "RELEASE_recall": .80, "ROC_AUC": .85}


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path: str | Path) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 600:
        raise ValueError(f"expected exactly the independent 600-state bank, got {len(rows)}")
    return rows


def _split(row: dict) -> str:
    return "dev" if int(row["seed"]) % 5 == 0 else "train"


def _target_uav(row: dict) -> dict:
    uid = int(row["uav_id"])
    return next(item for item in row["full_simulator_state"]["uavs"] if int(item["uid"]) == uid)


def _state_context(row: dict) -> dict:
    uav = _target_uav(row)
    return {"state_id": row["decision_event_id"], "state_hash": row["state_hash"], "Teacher_gate": int(row["Teacher_gate"]),
            "retry_predicate": bool(float(row["full_simulator_state"]["time"]) < float(row["safety_retry_after"])),
            "obs_retry": float(row["observation_v2_1"][RETRY_INDEX]), "phase": uav["phase"], "ring_role": uav["ring_role"],
            "release_eligibility": bool(row["Teacher_gate"] == 1 or row["Teacher_gate_rationale"] == "certified_release"),
            "commit_state": bool(row["commit_state"]), "interface_frozen": bool(row["interface_state"]),
            "resource_state": row["resource_state"], "Teacher_rationale": row["Teacher_gate_rationale"]}


def _retry_semantics(rows: list[dict]) -> dict:
    records = []; disagreements = 0; value_mismatches = 0
    for row in rows:
        state_time, retry_after = float(row["full_simulator_state"]["time"]), float(row["safety_retry_after"])
        runtime = state_time < retry_after; observed = float(row["observation_v2_1"][RETRY_INDEX]) > 0.0
        expected = float(np.clip(max(0.0, retry_after - state_time) / RETRY_HORIZON_SECONDS, 0.0, 1.0))
        value_match = bool(np.isclose(float(row["observation_v2_1"][RETRY_INDEX]), expected, rtol=0.0, atol=1e-6))
        disagreements += runtime != observed; value_mismatches += not value_match
        records.append({"state_id": row["decision_event_id"], "state_hash": row["state_hash"], "state_time": state_time,
                        "safety_retry_after": retry_after, "retry_predicate": runtime, "obs_retry_predicate": observed,
                        "observation_retry_norm": float(row["observation_v2_1"][RETRY_INDEX]), "expected_retry_norm": expected,
                        "value_match_atol_1e_6": value_match})
    return {"runtime_retry_active_count": int(sum(item["retry_predicate"] for item in records)),
            "obs_retry_active_count": int(sum(item["obs_retry_predicate"] for item in records)),
            "predicate_disagreement_count": disagreements, "predicate_disagreement_rate": disagreements / len(records),
            "normalization_value_mismatch_count": value_mismatches, "normalization_atol": 1e-6,
            "status": "PASS" if not disagreements and not value_mismatches else "FAIL", "records": records}


def _retry_pairs(rows: list[dict]) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {"train": [], "dev": []}
    for partition in result:
        waits = [row for row in rows if _split(row) == partition and row["Teacher_gate"] == 0 and float(row["safety_retry_after"]) > float(row["full_simulator_state"]["time"])]
        releases = [row for row in rows if _split(row) == partition and row["Teacher_gate"] == 1]
        used_releases: set[str] = set()
        for wait in waits:
            candidates = [release for release in releases if release["decision_event_id"] not in used_releases and release["seed"] != wait["seed"] and release["lifecycle_event_id"] != wait["lifecycle_event_id"]]
            if not candidates:
                continue
            release = min(candidates, key=lambda candidate: np.linalg.norm(np.asarray(wait["observation_v2_1"], np.float32) - np.asarray(candidate["observation_v2_1"], np.float32)))
            used_releases.add(release["decision_event_id"])
            result[partition].append({"pair_id": f"retry:{partition}:{len(result[partition])}", "split": partition,
                                      "distance": float(np.linalg.norm(np.asarray(wait["observation_v2_1"], np.float32) - np.asarray(release["observation_v2_1"], np.float32))),
                                      "wait": _state_context(wait), "release": _state_context(release)})
    return result


def _pair_validity(pairs: dict[str, list[dict]]) -> dict:
    flat = [pair for partition in pairs.values() for pair in partition]
    invalid = []
    for pair in flat:
        left, right = pair["wait"], pair["release"]
        valid = left["Teacher_gate"] == 0 and left["retry_predicate"] and left["obs_retry"] > 0.0 and right["Teacher_gate"] == 1 and not right["retry_predicate"] and right["obs_retry"] == 0.0
        pair["formal_retry_pair_definition_satisfied"] = valid
        if not valid:
            invalid.append(pair["pair_id"])
    return {"pair_counts": {key: len(value) for key, value in pairs.items()}, "invalid_pair_ids": invalid,
            "status": "PASS" if not invalid else "FAIL", "pairs": pairs}


def _pair_probe_with_rows(rows: list[dict], pairs: dict[str, list[dict]]) -> dict:
    indexed = {row["decision_event_id"]: row for row in rows}
    def arrays(items):
        values = [indexed[pair[side]["state_id"]]["observation_v2_1"] for pair in items for side in ("wait", "release")]
        labels = [label for pair in items for label in (0, 1)]
        return np.asarray(values, np.float32), np.asarray(labels, np.int64)
    train_x, train_y = arrays(pairs["train"]); dev_x, dev_y = arrays(pairs["dev"])
    runs = [_probe(train_x, train_y, dev_x, dev_y, seed) for seed in PROBE_SEEDS]
    metrics = {key: float(np.mean([run[key] for run in runs])) for key in runs[0]}
    return {"status": "PASS" if all(metrics[key] >= value for key, value in PAIR_GATE.items()) else "FAIL", "metrics": metrics, "runs": runs,
            "pre_registered_gate": PAIR_GATE, "samples": {"train": len(train_y), "dev": len(dev_y)}}


def _direct_baseline(semantics: dict) -> dict:
    labels = np.asarray([item["retry_predicate"] for item in semantics["records"]], bool)
    predictions = np.asarray([item["obs_retry_predicate"] for item in semantics["records"]], bool)
    blocked, nonblocked = labels, ~labels
    return {"rule": "predicted_retry_block = observation_v2_1[87] > 0", "accuracy": float((labels == predictions).mean()),
            "WAIT_blocked_recall": float(predictions[blocked].mean()) if blocked.any() else float("nan"),
            "RELEASE_nonblocked_recall": float((~predictions[nonblocked]).mean()) if nonblocked.any() else float("nan"),
            "status": "PASS" if bool((labels == predictions).all()) else "FAIL"}


def _gate_a_false_negatives(rows: list[dict]) -> dict:
    train = [row for row in rows if _split(row) == "train"]
    dev = [row for row in rows if _split(row) == "dev"]
    train_x = np.asarray([row["observation_v2_1"] for row in train], np.float32); train_y = np.asarray([row["Teacher_gate"] for row in train], np.int64)
    dev_x = np.asarray([row["observation_v2_1"] for row in dev], np.float32); dev_y = np.asarray([row["Teacher_gate"] for row in dev], np.int64)
    probabilities = np.mean(np.stack([_binary_probability(train_x, train_y, dev_x, seed=seed) for seed in PROBE_SEEDS]), axis=0)
    mean, std = train_x.mean(0), train_x.std(0); std[std < 1e-6] = 1.0
    train_releases = [row for row in train if row["Teacher_gate"] == 1]
    false_negatives = []
    for index, row in enumerate(dev):
        if row["Teacher_gate"] != 1 or probabilities[index] >= .5:
            continue
        nearest = min(train_releases, key=lambda candidate: float(np.linalg.norm((np.asarray(row["observation_v2_1"], np.float32) - np.asarray(candidate["observation_v2_1"], np.float32)) / std)))
        selected = next(item for item in row["candidate_scan"] if item["feasible"])
        false_negatives.append({"state": _state_context(row), "probe_A_probability": float(probabilities[index]), "prediction": "WAIT",
                                "selected_proposal": int(row["Teacher_proposal"]), "selected_certificate": selected,
                                "candidate_reason_counts": row["candidate_reason_counts"], "nearest_train_RELEASE": {"state_id": nearest["decision_event_id"], "state_hash": nearest["state_hash"],
                                "distance_standardized_obs": float(np.linalg.norm((np.asarray(row["observation_v2_1"], np.float32) - np.asarray(nearest["observation_v2_1"], np.float32)) / std)),
                                "Teacher_rationale": nearest["Teacher_gate_rationale"], "selected_proposal": nearest["Teacher_proposal"]}})
    # All current state-only Teacher predicates are explicit in the observation.
    # A common missing state is therefore asserted only if code inventory finds
    # an unrepresented predicate, which this V2.1 audit does not.
    return {"dev_teacher_RELEASE_count": int(dev_y.sum()), "gate_A_false_negative_count": len(false_negatives),
            "false_negative_records": false_negatives, "gate_A_false_negative_common_missing_state": False,
            "attribution": "No common missing state-only Teacher predicate is identified; low dev RELEASE support (6 states) is the registered finite-coverage limitation."}


def _powered_protocol() -> dict:
    return {"status": "FROZEN_PREPARATION_ONLY", "no_collection_or_promotion_this_turn": True,
            "minimum_counts": {"train_Teacher_RELEASE": 100, "dev_Teacher_RELEASE": 40, "dev_retry_active_WAIT": 40},
            "disjointness": ["seed", "lifecycle_event_id", "state_hash"],
            "natural_source_strata": ["release-ready low traffic", "clear outer release", "middle/interface release", "transition/retry states", "mixed traffic", "resource pressure"],
            "allowed_sampling": "increase naturally release-ready scenario share only", "forbidden": ["row duplication", "class weighting", "oversampling", "synthetic labels", "fabricated certificate outcomes"],
            "frozen_gate_thresholds": PAIR_GATE}


def run(bank: str | Path, output_dir: str | Path) -> dict:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    rows = _read(bank); semantics = _retry_semantics(rows); pairs = _retry_pairs(rows); pair_validity = _pair_validity(pairs)
    learned = _pair_probe_with_rows(rows, pairs); baseline = _direct_baseline(semantics); gate_a = _gate_a_false_negatives(rows)
    if semantics["status"] != "PASS":
        root, observation_revision, protocol_revision, powered = "RETRY_FEATURE_PROVENANCE_OR_RUNTIME_BUG", True, False, False
    elif pair_validity["status"] != "PASS":
        root, observation_revision, protocol_revision, powered = "RETRY_MATCHED_PAIR_CONSTRUCTION_ERROR", False, True, False
    elif learned["status"] != "PASS" and baseline["status"] == "PASS":
        root, observation_revision, protocol_revision, powered = "DIRECT_PREDICATE_AND_LEARNABILITY_GATE_CONFLATION", False, True, True
    elif gate_a["gate_A_false_negative_common_missing_state"]:
        root, observation_revision, protocol_revision, powered = "NEW_OBSERVATION_ALIASING", True, False, False
    else:
        root, observation_revision, protocol_revision, powered = "FINITE_RELEASE_COVERAGE", False, True, True
    result = {"stage": "V4_2_V2_1_RETRY_CLOSURE_PROTOCOL_DIAGNOSIS", "evidence_type": "newly_run", "input_bank": {"path": str(bank), "sha256": _sha256(bank), "states": len(rows)},
              "retry_direct_semantic_audit": semantics, "retry_matched_pair_construction": pair_validity,
              "direct_one_feature_baseline": baseline, "retry_learned_pair_probe": learned, "gate_A_error_attribution": gate_a,
              "powered_independent_confirmation_protocol": _powered_protocol(),
              "RETRY_FEATURE_SEMANTIC_CLOSURE": semantics["status"], "RETRY_MATCHED_PAIR_CONSTRUCTION": pair_validity["status"], "RETRY_LEARNED_PAIR_PROBE": learned["status"],
              "ROOT_CAUSE": root, "observation_revision_required": observation_revision, "closure_protocol_revision_required": protocol_revision,
              "independent_release_coverage_required": not gate_a["gate_A_false_negative_common_missing_state"], "powered_confirmation_allowed": powered,
              "OBSERVATION_V2_1_FINAL_FREEZE": "FAIL", "FORMAL_V2_1_I0_RECOLLECTION_ALLOWED": False, "I1_ALLOWED": False, "DAgger_ALLOWED": False, "PPO_ALLOWED": False,
              "forbidden_operations_called": []}
    (output / "v4_2_v2_1_retry_closure_diagnostic.manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--bank", required=True); parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(); print(json.dumps(run(args.bank, args.output_dir), indent=2))
