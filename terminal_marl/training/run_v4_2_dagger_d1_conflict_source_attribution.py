"""Read-only V4.2 D1 conflict-source attribution diagnostic.

The formal D1 JSONL is never modified.  This writes only immutable derived
views, metrics, and diagnostic manifests, and cannot authorize a DAgger rerun,
Round 2, or PPO.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from ..algo.joint_ppo_v4_2 import hierarchical_bc_loss
from ..core.low_level_joint_action_v4_2 import RELEASE_GATE
from .run_v4_2_dagger_r1_release_retention_diagnostic import (_d1_disagreement,
    _load_policy, _prediction_details, _sha256)
from .run_v4_2_i1_hierarchical_bc import _arrays, _metrics, _rows


NEW_SOURCE = "student_closed_loop_dagger_round1"
DEV_SEEDS = (9812, 9813, 9814)
SEEDS = (5101, 5102, 5103)
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = .0003
FAMILIES = ("clear_outer_RELEASE", "middle_or_interface_RELEASE", "persistent_middle_WAIT", "persistent_other_WAIT", "transition_WAIT")


def _phase(row: dict) -> str:
    match = re.search(r":phase-([^:]+)", row.get("lifecycle_event_id", ""))
    return match.group(1) if match else "unknown"


def _family(row: dict) -> str:
    phase, release = _phase(row), row["gate_target"] == RELEASE_GATE
    if release and phase == "ring_outer":
        return "clear_outer_RELEASE"
    if release:
        return "middle_or_interface_RELEASE"
    if phase == "ring_middle" and row.get("semantic_sampling_reason") == "bounded_sparse_persistence":
        return "persistent_middle_WAIT"
    if row.get("semantic_sampling_reason") == "bounded_sparse_persistence":
        return "persistent_other_WAIT"
    return "transition_WAIT"


def _hash_rows(rows: list[dict], path: Path) -> str:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return _sha256(path)


def _tape_record_index(tape_dir: str | Path) -> dict[tuple[int, int], dict]:
    index: dict[tuple[int, int], dict] = {}
    for path in Path(tape_dir).glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8")); seed = int(payload["episode_seed"])
        for record in payload.get("records", []):
            index[(seed, int(record["step"]))] = record
    return index


def _step(row: dict) -> int:
    match = re.search(r":step-(\d+):", row["decision_event_id"])
    if not match:
        raise ValueError(f"D1 event has no step: {row['decision_event_id']}")
    return int(match.group(1))


def _state_context(row: dict, state: dict) -> dict:
    release_state = state.get("release_state", {})
    histories = [value.get("decision_history", []) for value in release_state.values()]
    interfaces = sorted({value.get("interface") for value in release_state.values() if value.get("interface")})
    reservations = state.get("reservations", [])
    rejected = sum("safety_reject" in history for history in histories)
    return {"source": row.get("source"), "phase": _phase(row), "teacher_target": "RELEASE" if row["gate_target"] else "WAIT",
            "teacher_proposal": row.get("proposal_target"), "semantic_sampling_reason": row.get("semantic_sampling_reason"),
            "lifecycle_event_id": row.get("lifecycle_event_id"), "reservation_count": len(reservations),
            "reservation_state": "reservation_present" if reservations else "no_reservation",
            "interfaces": interfaces, "interface_headway_state": "interface_or_headway_active" if interfaces else "no_interface_record",
            "safety_rejected_release_count": rejected, "conflict_or_recovery_state": "recovery_or_conflict" if rejected else "no_recorded_recovery_conflict"}


def _post_wait_release(row: dict, records: dict[tuple[int, int], dict]) -> bool:
    if row["gate_target"] != RELEASE_GATE or _step(row) == 0:
        return False
    prior = records.get((int(row["seed"]), _step(row) - 1), {})
    return prior.get("teacher_gate", {}).get(str(row["uav_id"])) == 0


def _family_profiles(rows: list[dict], state_index: dict[tuple[int, int], dict]) -> dict:
    profiles: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        profiles[_family(row)].append(_state_context(row, state_index.get((int(row["seed"]), _step(row)), {})))
    result = {}
    for family in FAMILIES:
        items = profiles[family]
        result[family] = {"count": len(items), "teacher_targets": dict(Counter(item["teacher_target"] for item in items)),
                          "phase": dict(Counter(item["phase"] for item in items)),
                          "reservation_state": dict(Counter(item["reservation_state"] for item in items)),
                          "interface_headway_state": dict(Counter(item["interface_headway_state"] for item in items)),
                          "conflict_or_recovery_state": dict(Counter(item["conflict_or_recovery_state"] for item in items)),
                          "lifecycle_persistence": dict(Counter(item["semantic_sampling_reason"] for item in items))}
    return result


def _train_once(checkpoint: str | Path, train_rows: list[dict], dev: tuple[torch.Tensor, ...], frozen: tuple[torch.Tensor, ...], *, seed: int) -> dict:
    policy = _load_policy(checkpoint, learning_rate=LEARNING_RATE)
    train = _arrays(train_rows); x, gate, proposal, _ = train
    torch.manual_seed(seed); np.random.seed(seed)
    for _ in range(EPOCHS):
        for ids in (np.random.permutation(len(x))[offset:offset + BATCH_SIZE] for offset in range(0, len(x), BATCH_SIZE)):
            gate_dist, proposal_dist = policy.model.distributions(x[ids])
            loss, _ = hierarchical_bc_loss(gate_dist.logits, proposal_dist.logits, gate[ids], proposal[ids])
            policy.optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(policy.model.parameters(), 1.0); policy.optimizer.step()
    return {"seed": seed, "development": _metrics(policy, *dev), "frozen_posthoc": _metrics(policy, *frozen),
            "D1_disagreement": _d1_disagreement(policy, *train)}


def _mean(rows: list[dict], section: str, key: str) -> float:
    if section == "D1_disagreement":
        return float(np.mean([row[section][key] for row in rows]))
    return float(np.mean([row[section]["gate"][key] for row in rows]))


def _summarize(rows: list[dict]) -> dict:
    return {"development": {"WAIT_recall": _mean(rows, "development", "WAIT_recall"), "RELEASE_recall": _mean(rows, "development", "RELEASE_recall"),
                             "macro_F1": _mean(rows, "development", "macro_F1")},
            "frozen_posthoc": {"WAIT_recall": _mean(rows, "frozen_posthoc", "WAIT_recall"), "RELEASE_recall": _mean(rows, "frozen_posthoc", "RELEASE_recall"),
                               "macro_F1": _mean(rows, "frozen_posthoc", "macro_F1")},
            "D1_disagreement": {"decoded_action": _mean(rows, "D1_disagreement", "decoded_action_disagreement"),
                                  "gate": _mean(rows, "D1_disagreement", "gate_disagreement"),
                                  "proposal_on_RELEASE": _mean(rows, "D1_disagreement", "proposal_disagreement_on_teacher_RELEASE")}}


def _condition(checkpoint: str | Path, name: str, rows: list[dict], dev: tuple[torch.Tensor, ...], frozen: tuple[torch.Tensor, ...]) -> dict:
    runs = [_train_once(checkpoint, rows, dev, frozen, seed=seed) for seed in SEEDS]
    return {"condition": name, "train_rows": len(rows), "seeds": list(SEEDS), "runs": runs, "mean": _summarize(runs)}


def _flip_neighborhoods(i1_checkpoint: str | Path, r1_checkpoint: str | Path, frozen_rows: list[dict], all_rows: list[dict], new_rows: list[dict]) -> dict:
    i1, r1 = _load_policy(i1_checkpoint, learning_rate=LEARNING_RATE), _load_policy(r1_checkpoint, learning_rate=LEARNING_RATE)
    frozen = _arrays(frozen_rows); first, second = _prediction_details(i1, frozen[0]), _prediction_details(r1, frozen[0])
    target = frozen[1].cpu().numpy() == RELEASE_GATE; selected = target & (first["gate"] == RELEASE_GATE) & (second["gate"] != RELEASE_GATE)
    nx = np.asarray([row["observation"] for row in new_rows], dtype=np.float32)
    new_i1, new_r1 = _prediction_details(i1, torch.as_tensor(nx)), _prediction_details(r1, torch.as_tensor(nx))
    per_flip = []
    family_accumulator: dict[str, list[dict]] = defaultdict(list)
    for index in np.where(selected)[0]:
        row = frozen_rows[int(index)]; distance = np.linalg.norm(nx - np.asarray(row["observation"], dtype=np.float32), axis=1) / np.sqrt(nx.shape[1])
        nearest = np.argsort(distance)[:10]; flip_phase = row.get("phase")
        entries = []
        for position in nearest:
            event = new_rows[int(position)]; family = _family(event)
            item = {"decision_event_id": event["decision_event_id"], "family": family, "distance": float(distance[position]),
                    "teacher_target": "RELEASE" if event["gate_target"] else "WAIT", "teacher_proposal": event.get("proposal_target"),
                    "I1_gate_margin": float(new_i1["gate_margin"][position]), "R1_gate_margin": float(new_r1["gate_margin"][position]),
                    "same_phase_as_flip": _phase(event) == flip_phase, "lifecycle_relation": "different_seed_lifecycle" if event["seed"] != row.get("seed") else "same_seed_different_lifecycle"}
            entries.append(item); family_accumulator[family].append(item)
        per_flip.append({"decision_event_id": row["decision_event_id"], "state_hash": row["state_hash"], "source": row["source"], "scenario": row.get("scenario_class"),
                         "teacher_proposal": row.get("proposal_target"), "I1_P_RELEASE": float(first["release_probability"][index]), "R1_P_RELEASE": float(second["release_probability"][index]),
                         "nearest_new_D1_events": entries})
    summary = {}
    for family in FAMILIES:
        items = family_accumulator[family]
        summary[family] = {"nearest_top10_count": len(items), "mean_distance": float(np.mean([item["distance"] for item in items])) if items else None,
                           "min_distance": float(np.min([item["distance"] for item in items])) if items else None,
                           "teacher_targets": dict(Counter(item["teacher_target"] for item in items)),
                           "mean_I1_gate_margin": float(np.mean([item["I1_gate_margin"] for item in items])) if items else None,
                           "mean_R1_gate_margin": float(np.mean([item["R1_gate_margin"] for item in items])) if items else None,
                           "same_phase_fraction": float(np.mean([item["same_phase_as_flip"] for item in items])) if items else None,
                           "lifecycle_relations": dict(Counter(item["lifecycle_relation"] for item in items))}
    return {"flipped_release_state_count": len(per_flip), "flips": per_flip, "family_neighborhood_summary": summary}


def _deltas(condition: dict, baseline: dict) -> dict:
    value, base = condition["mean"], baseline["mean"]
    return {"dev_RELEASE_recall": value["development"]["RELEASE_recall"] - base["development"]["RELEASE_recall"],
            "dev_WAIT_recall": value["development"]["WAIT_recall"] - base["development"]["WAIT_recall"],
            "dev_macro_F1": value["development"]["macro_F1"] - base["development"]["macro_F1"],
            "D1_decoded_disagreement": value["D1_disagreement"]["decoded_action"] - base["D1_disagreement"]["decoded_action"],
            "frozen_RELEASE_recall_posthoc": value["frozen_posthoc"]["RELEASE_recall"] - base["frozen_posthoc"]["RELEASE_recall"]}


def _classify(results: dict) -> tuple[str, list[str], bool, dict]:
    full, parent = results["full_new_train"], results["parent_only"]
    dominant = []
    evidence = {}
    for family in FAMILIES:
        add, remove = results[f"add_{family}"], results[f"remove_{family}"]
        add_delta, remove_delta = _deltas(add, parent), _deltas(remove, full)
        # A family is localized only if all seeds show the direction and its
        # removal recovers >= .03 frozen RELEASE recall on average.  This is a
        # predeclared attribution criterion, not a qualification threshold.
        include_harms_dev = all(run["development"]["gate"]["RELEASE_recall"] < parent["mean"]["development"]["RELEASE_recall"] - .02 for run in add["runs"])
        remove_recovers = all(run["frozen_posthoc"]["gate"]["RELEASE_recall"] > full["mean"]["frozen_posthoc"]["RELEASE_recall"] + .01 for run in remove["runs"])
        localized = include_harms_dev and remove_recovers and remove_delta["frozen_RELEASE_recall_posthoc"] >= .03
        if localized:
            dominant.append(family)
        evidence[family] = {"add_vs_parent": add_delta, "remove_vs_full": remove_delta, "include_harms_dev_all_seeds": include_harms_dev,
                            "remove_recovers_frozen_all_seeds": remove_recovers, "localized_criterion": localized}
    if len(dominant) == 1:
        root, localized = "SPECIFIC_D1_STATE_FAMILY_CONFLICT", True
    elif len(dominant) > 1:
        root, localized = "DAgger_OCCUPANCY_MIXTURE_CONFLICT", False
    else:
        # If the full mixture harms development relative to parent but no single
        # leave-one-out family recovers it, the conflict is occupancy-mixture.
        full_harms_dev = _deltas(full, parent)["dev_RELEASE_recall"] < -.02
        root, localized = ("DAgger_OCCUPANCY_MIXTURE_CONFLICT", False) if full_harms_dev else ("UNRESOLVED", False)
    return root, dominant, localized, evidence


def run(i1_checkpoint: str | Path, r1_checkpoint: str | Path, d1_path: str | Path, frozen_cohort: str | Path, tape_dir: str | Path, output_dir: str | Path) -> dict:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    rows, frozen_rows = _rows(d1_path), _rows(frozen_cohort)
    parent = [row for row in rows if row.get("source") != NEW_SOURCE]
    new = [row for row in rows if row.get("source") == NEW_SOURCE]
    if len(parent) != 1500 or len(new) != 502:
        raise ValueError("Attribution requires formal D1 with 1500 parent and 502 new student-visited rows")
    dev_rows = [row for row in new if int(row["seed"]) in DEV_SEEDS]
    train_new = [row for row in new if int(row["seed"]) not in DEV_SEEDS]
    if set(row["decision_event_id"] for row in dev_rows) & set(row["decision_event_id"] for row in train_new):
        raise AssertionError("development split overlaps D1 training view")
    family_rows = {family: [row for row in train_new if _family(row) == family] for family in FAMILIES}
    dev_path = output / "dagger_development_split.jsonl"; train_path = output / "D1_attribution_training_view.jsonl"
    dev_hash, train_hash = _hash_rows(dev_rows, dev_path), _hash_rows(parent + train_new, train_path)
    records = _tape_record_index(tape_dir)
    state_index = {key: value.get("pre_state", {}) for key, value in records.items()}
    profiles = _family_profiles(new, state_index)
    dev_context = [_state_context(row, state_index.get((int(row["seed"]), _step(row)), {})) for row in dev_rows]
    dev_coverage = {
        "WAIT": any(row["gate_target"] == 0 for row in dev_rows),
        "RELEASE": any(row["gate_target"] == RELEASE_GATE for row in dev_rows),
        "clear_release": any(_family(row) == "clear_outer_RELEASE" for row in dev_rows),
        "conflict_recovery": any(item["conflict_or_recovery_state"] == "recovery_or_conflict" for item in dev_context),
        "post_WAIT_release": any(_post_wait_release(row, records) for row in dev_rows),
        "interface_headway": any(item["interface_headway_state"] == "interface_or_headway_active" for item in dev_context),
    }
    dev = _arrays(dev_rows); frozen = _arrays(frozen_rows)
    conditions: dict[str, dict] = {}
    conditions["parent_only"] = _condition(i1_checkpoint, "parent_only", parent, dev, frozen)
    conditions["full_new_train"] = _condition(i1_checkpoint, "full_new_train", parent + train_new, dev, frozen)
    for family in FAMILIES:
        conditions[f"add_{family}"] = _condition(i1_checkpoint, f"add_{family}", parent + family_rows[family], dev, frozen)
        conditions[f"remove_{family}"] = _condition(i1_checkpoint, f"remove_{family}", parent + [row for row in train_new if _family(row) != family], dev, frozen)
    root, dominant, localized, attribution = _classify(conditions)
    result = {"stage": "V4_2_DAGGER_D1_CONFLICT_SOURCE_ATTRIBUTION", "evidence_type": "newly_run",
              "inputs": {"I1_checkpoint_sha256": _sha256(i1_checkpoint), "R1_checkpoint_sha256": _sha256(r1_checkpoint), "D1_sha256": _sha256(d1_path),
                         "frozen_cohort_sha256": _sha256(frozen_cohort), "D1_rows": len(rows), "new_D1_events": len(new)},
              "development_split": {"path": str(dev_path), "sha256": dev_hash, "rows": len(dev_rows), "seeds": list(DEV_SEEDS),
                                    "source": NEW_SOURCE, "disjoint_from_training_view": True, "family_counts": dict(Counter(_family(row) for row in dev_rows)),
                                    "required_families_present": all(any(_family(row) == family for row in dev_rows) for family in FAMILIES), "coverage": dev_coverage},
              "attribution_training_view": {"path": str(train_path), "sha256": train_hash, "rows": len(parent) + len(train_new), "new_rows": len(train_new), "seeds_excluded_for_dev": list(DEV_SEEDS)},
              "family_profiles": profiles, "flipped_release_neighborhoods": _flip_neighborhoods(i1_checkpoint, r1_checkpoint, frozen_rows, rows, new),
              "hyperparameters": {"epochs": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE, "seeds_per_condition": list(SEEDS), "conditions": list(conditions)},
              "conditions": conditions, "family_attribution": attribution, "ROOT_CAUSE": root, "dominant_conflicting_families": dominant,
              "conflict_is_localized": localized, "development_split_ready": bool(all([len(dev_rows) > 0, set(row["seed"] for row in dev_rows).isdisjoint(set(row["seed"] for row in train_new)), all(any(_family(row) == family for row in dev_rows) for family in FAMILIES), all(dev_coverage.values())])),
              "recommended_round1_protocol_change": "Freeze this independent DAgger development split for aggregation/epoch protocol decisions; retain the frozen cohort exclusively as a regression qualification gate. Do not promote any diagnostic condition.",
              "ROUND1_RERUN_ALLOWED": False, "ROUND2_ALLOWED": False, "PPO_ALLOWED": False, "forbidden_operations_called": [],
              "environment": {"python": platform.python_version(), "torch": torch.__version__, "cuda_available": torch.cuda.is_available()}}
    (output / "v4_2_dagger_d1_conflict_source_attribution.manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--i1-checkpoint", required=True); parser.add_argument("--r1-checkpoint", required=True); parser.add_argument("--d1", required=True)
    parser.add_argument("--frozen-cohort", required=True); parser.add_argument("--tape-dir", required=True); parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(); print(json.dumps(run(args.i1_checkpoint, args.r1_checkpoint, args.d1, args.frozen_cohort, args.tape_dir, args.output_dir), indent=2))
