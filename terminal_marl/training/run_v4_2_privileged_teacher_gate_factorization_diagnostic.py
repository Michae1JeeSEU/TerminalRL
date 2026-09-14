"""Read-only diagnosis of whether V4.2's Teacher gate is deployably factored.

The input is the frozen V2.1 closure-audit bank.  This module neither rebuilds
the bank nor calls the simulator: every probe operates only on stored 88D
observations and the recorded Teacher catalogue scan.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch


PROBE_SEEDS = (9131, 9132, 9133)
# Probe A retains the closure-audit operating-point requirements.  Probe B
# asks an information/separability question, so it is preregistered on ranking
# and balanced discrimination rather than an arbitrary operating threshold on
# its highly imbalanced 25-proposal table.
GATE_A = {"accuracy": .80, "macro_F1": .80, "negative_recall": .80, "positive_recall": .80, "ROC_AUC": .85}
GATE_B = {"macro_F1": .80, "ROC_AUC": .85}
STAGE = "V4_2_PRIVILEGED_TEACHER_GATE_FACTORIZATION_DIAGNOSTIC"


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_bank(path: str | Path) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not 300 <= len(rows) <= 600:
        raise ValueError(f"expected the 300-600 state closure bank, got {len(rows)} rows")
    if any(len(row["observation_v2_1"]) != 88 or len(row["candidate_scan"]) != 26 for row in rows):
        raise ValueError("closure bank must contain 88D observations and complete 26-action scans")
    return rows


def _partition(row: dict) -> str:
    """A state, its lifecycle, and all 25 associated proposals move together."""
    return "dev" if int(row["seed"]) % 5 == 0 else "train"


def _assert_disjoint(rows: list[dict]) -> dict:
    by_state: dict[str, set[str]] = {}
    by_lifecycle: dict[str, set[str]] = {}
    for row in rows:
        split = _partition(row)
        by_state.setdefault(row["state_hash"], set()).add(split)
        by_lifecycle.setdefault(row["lifecycle_event_id"], set()).add(split)
    leaking_states = [key for key, values in by_state.items() if len(values) != 1]
    leaking_lifecycles = [key for key, values in by_lifecycle.items() if len(values) != 1]
    if leaking_states or leaking_lifecycles:
        raise RuntimeError("seed/lifecycle split leakage in closure bank")
    return {"split_rule": "seed % 5 == 0 => dev; all proposals remain with their state",
            "seed_disjoint": True, "lifecycle_disjoint": True,
            "state_hash_disjoint": True, "leaking_state_hash_count": 0,
            "leaking_lifecycle_count": 0,
            "state_counts": dict(Counter(_partition(row) for row in rows)),
            "seed_counts": {split: len(set(row["seed"] for row in rows if _partition(row) == split)) for split in ("train", "dev")}}


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    positive, negative = labels == 1, labels == 0
    if not positive.any() or not negative.any():
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[positive].sum() - positive.sum() * (positive.sum() + 1) / 2) / (positive.sum() * negative.sum()))


def _binary_metrics(probability: np.ndarray, labels: np.ndarray, *, positive_name: str, negative_name: str) -> dict:
    prediction = probability >= .5
    negative, positive = labels == 0, labels == 1
    negative_recall = float((prediction[negative] == 0).mean()) if negative.any() else float("nan")
    positive_recall = float((prediction[positive] == 1).mean()) if positive.any() else float("nan")
    negative_precision = float(((prediction == 0) & negative).sum() / max(1, (prediction == 0).sum()))
    positive_precision = float(((prediction == 1) & positive).sum() / max(1, (prediction == 1).sum()))
    negative_f1 = 2 * negative_precision * negative_recall / max(1e-12, negative_precision + negative_recall)
    positive_f1 = 2 * positive_precision * positive_recall / max(1e-12, positive_precision + positive_recall)
    return {"accuracy": float((prediction == labels).mean()), "macro_F1": float((negative_f1 + positive_f1) / 2),
            f"{negative_name}_recall": negative_recall, f"{positive_name}_recall": positive_recall,
            "ROC_AUC": _auc(probability, labels), "predicted_positive_count": int(prediction.sum()),
            "positive_count": int(positive.sum()), "negative_count": int(negative.sum())}


def _binary_probability(train_x: np.ndarray, train_y: np.ndarray, dev_x: np.ndarray, *, seed: int) -> np.ndarray:
    torch.manual_seed(seed); np.random.seed(seed)
    mean, std = train_x.mean(axis=0), train_x.std(axis=0); std[std < 1e-6] = 1.0
    tx = torch.as_tensor((train_x - mean) / std, dtype=torch.float32)
    ty = torch.as_tensor(train_y, dtype=torch.float32)
    dx = torch.as_tensor((dev_x - mean) / std, dtype=torch.float32)
    model = torch.nn.Linear(tx.shape[1], 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=.03)
    for _ in range(300):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(tx).squeeze(-1), ty)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    return torch.sigmoid(model(dx).squeeze(-1)).detach().numpy()


def _binary_probe(train_x: np.ndarray, train_y: np.ndarray, dev_x: np.ndarray, dev_y: np.ndarray, *, seed: int, positive_name: str, negative_name: str) -> tuple[dict, np.ndarray]:
    probability = _binary_probability(train_x, train_y, dev_x, seed=seed)
    return _binary_metrics(probability, dev_y, positive_name=positive_name, negative_name=negative_name), probability


def _mean_runs(runs: list[dict]) -> dict:
    return {key: float(np.mean([row[key] for row in runs])) for key in runs[0]}


def _binary_view(rows: list[dict], *, proposal_conditioned: bool) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[dict]]:
    data: dict[str, list[np.ndarray]] = {"train": [], "dev": []}
    labels: dict[str, list[int]] = {"train": [], "dev": []}
    audit_rows: list[dict] = []
    for state in rows:
        split = _partition(state); observation = np.asarray(state["observation_v2_1"], dtype=np.float32)
        if not proposal_conditioned:
            certified = int(any(bool(item["feasible"]) for item in state["candidate_scan"] if item["decoded"]["release_request"] > 0.0))
            if certified != int(state["Teacher_gate"]):
                raise RuntimeError("Teacher gate does not equal any certified RELEASE in stored scan")
            data[split].append(observation); labels[split].append(certified)
            continue
        for candidate in state["candidate_scan"]:
            if candidate["decoded"]["release_request"] <= 0.0:
                continue
            # These two values identify one of the fixed 25 proposals.  No
            # execution verdict or Teacher target is part of the input.
            proposal_features = np.asarray([candidate["decoded"]["speed_residual"], candidate["decoded"]["merge_time_residual"]], dtype=np.float32)
            certified = int(bool(candidate["feasible"]))
            data[split].append(np.concatenate((observation, proposal_features)))
            labels[split].append(certified)
            audit_rows.append({"split": split, "seed": int(state["seed"]), "lifecycle_event_id": state["lifecycle_event_id"],
                               "state_hash": state["state_hash"], "proposal_id": int(candidate["proposal_id"] - 1), "joint_index": int(candidate["proposal_id"]),
                               "speed_residual": float(proposal_features[0]), "merge_time_residual": float(proposal_features[1]),
                               "reservation_feasible": candidate["reservation_feasible"], "trajectory_dynamics_feasible": candidate["dynamics_feasible"],
                               "safety_certified": candidate["SafetyFilter_certified"], "holder_tube_certified": candidate["predicted_holder_reachable_tube"],
                               "final_proposal_certified": bool(certified), "teacher_result": candidate["teacher_result"],
                               "teacher_gate": int(state["Teacher_gate"]), "teacher_best_proposal": state["Teacher_proposal"]})
    return {key: np.asarray(value, np.float32) for key, value in data.items()}, {key: np.asarray(value, np.int64) for key, value in labels.items()}, audit_rows


def _binary_report(rows: list[dict], *, name: str, proposal_conditioned: bool, positive_name: str, negative_name: str, gate: dict[str, float]) -> tuple[dict, list[dict]]:
    x, y, audit_rows = _binary_view(rows, proposal_conditioned=proposal_conditioned)
    if len(np.unique(y["train"])) < 2 or len(np.unique(y["dev"])) < 2:
        return {"name": name, "status": "NOT_EVALUABLE", "reason": "both classes are required in train and dev", "samples": {key: int(len(value)) for key, value in y.items()}, "class_counts": {key: dict(Counter(map(int, value))) for key, value in y.items()}}, audit_rows
    paired_runs = [_binary_probe(x["train"], y["train"], x["dev"], y["dev"], seed=seed, positive_name=positive_name, negative_name=negative_name) for seed in PROBE_SEEDS]
    runs, probabilities = [value[0] for value in paired_runs], [value[1] for value in paired_runs]
    metrics = _mean_runs(runs)
    canonical = {"accuracy": metrics["accuracy"], "macro_F1": metrics["macro_F1"], "negative_recall": metrics[f"{negative_name}_recall"], "positive_recall": metrics[f"{positive_name}_recall"], "ROC_AUC": metrics["ROC_AUC"]}
    passed = all(canonical[key] >= value for key, value in gate.items())
    result = {"name": name, "status": "PASS" if passed else "FAIL", "model": "linear_logistic_regression; no class weighting", "input": "obs_88d + [speed_residual, merge_time_residual]" if proposal_conditioned else "obs_88d", "target": positive_name,
            "samples": {key: int(len(value)) for key, value in y.items()}, "class_counts": {key: dict(Counter(map(int, value))) for key, value in y.items()},
            "metrics": metrics, "runs": runs, "pre_registered_gate": gate}
    if proposal_conditioned:
        result["mean_dev_probability"] = np.mean(np.stack(probabilities), axis=0).astype(float).tolist()
    return result, audit_rows


def _proposal_reason_breakdown(audit_rows: list[dict], probe_b: dict) -> dict:
    if probe_b.get("status") == "NOT_EVALUABLE":
        return {"status": "NOT_EVALUABLE"}
    probabilities = np.asarray(probe_b.pop("mean_dev_probability"), dtype=float)
    dev_rows = [row for row in audit_rows if row["split"] == "dev"]
    if len(probabilities) != len(dev_rows):
        raise RuntimeError("proposal diagnostic probability/record alignment failed")
    groups = {"reservation": lambda reason: reason == "headway_conflict",
              "dynamics": lambda reason: reason == "trajectory_infeasible",
              "SafetyFilter": lambda reason: "safety" in reason.lower() and "holder" not in reason.lower(),
              "holder_tube": lambda reason: "holder" in reason.lower()}
    return {name: {"train_count": int(sum(check(row["teacher_result"]) for row in audit_rows if row["split"] == "train")),
                   "dev_count": int(sum(check(row["teacher_result"]) for row in dev_rows)),
                   "dev_rejected_recall_at_fixed_0_5": None if not any(check(row["teacher_result"]) for row in dev_rows) else float((probabilities[[check(row["teacher_result"]) for row in dev_rows]] < .5).mean()),
                   "classification": "action_conditioned"}
            for name, check in groups.items()}


def _proposal_selection(rows: list[dict]) -> dict:
    selected = [row for row in rows if int(row["Teacher_gate"]) == 1 and row["Teacher_proposal"] is not None]
    split_rows = {split: [row for row in selected if _partition(row) == split] for split in ("train", "dev")}
    target_counts = {split: dict(Counter(int(row["Teacher_proposal"]) for row in items)) for split, items in split_rows.items()}
    if not split_rows["train"] or not split_rows["dev"] or len(target_counts["train"]) < 2 or len(target_counts["dev"]) < 2:
        return {"status": "NOT_EVALUABLE", "reason": "closure bank has insufficient split-disjoint Teacher proposal diversity", "release_state_counts": {key: len(value) for key, value in split_rows.items()}, "target_proposal_counts": target_counts}
    train_x = np.asarray([row["observation_v2_1"] for row in split_rows["train"]], dtype=np.float32)
    dev_x = np.asarray([row["observation_v2_1"] for row in split_rows["dev"]], dtype=np.float32)
    train_y = np.asarray([row["Teacher_proposal"] for row in split_rows["train"]], dtype=np.int64)
    dev_y = np.asarray([row["Teacher_proposal"] for row in split_rows["dev"]], dtype=np.int64)
    catalogue = {int(item["proposal_id"] - 1): item["decoded"] for item in rows[0]["candidate_scan"] if item["decoded"]["release_request"] > 0.0}
    runs = []
    for seed in PROBE_SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        mean, std = train_x.mean(0), train_x.std(0); std[std < 1e-6] = 1.0
        tx = torch.as_tensor((train_x - mean) / std, dtype=torch.float32); ty = torch.as_tensor(train_y, dtype=torch.long)
        dx = torch.as_tensor((dev_x - mean) / std, dtype=torch.float32)
        model = torch.nn.Linear(88, 25); optimizer = torch.optim.Adam(model.parameters(), lr=.03)
        for _ in range(400):
            loss = torch.nn.functional.cross_entropy(model(tx), ty); optimizer.zero_grad(); loss.backward(); optimizer.step()
        logits = model(dx).detach().numpy(); ranked = np.argsort(-logits, axis=1)
        top1, top3 = ranked[:, 0], ranked[:, :3]
        speed = np.asarray([catalogue[int(value)]["speed_residual"] for value in top1]) == np.asarray([catalogue[int(value)]["speed_residual"] for value in dev_y])
        merge = np.asarray([catalogue[int(value)]["merge_time_residual"] for value in top1]) == np.asarray([catalogue[int(value)]["merge_time_residual"] for value in dev_y])
        runs.append({"top_1": float((top1 == dev_y).mean()), "top_3": float(np.asarray([label in choices for label, choices in zip(dev_y, top3)]).mean()), "speed_agreement": float(speed.mean()), "merge_agreement": float(merge.mean())})
    metrics = _mean_runs(runs)
    return {"status": "PASS" if metrics["top_1"] >= .60 and metrics["top_3"] >= .80 else "FAIL", "model": "linear_25_way_classifier; no class weighting", "release_state_counts": {key: len(value) for key, value in split_rows.items()}, "target_proposal_counts": target_counts, "metrics": metrics, "runs": runs,
            "pre_registered_gate": {"top_1": .60, "top_3": .80}}


def _closure_reclassification() -> dict:
    return {
        "reservation": {"classification": "ACTION_CONDITIONED", "reason": "Teacher calls propose(candidate) then can_reserve(proposal)."},
        "interface": {"classification": "STATE_ONLY", "reason": "affected_interfaces is checked before any candidate evaluation."},
        "headway": {"classification": "ACTION_CONDITIONED", "reason": "headway rejection is only observed for a constructed proposal; current propose() normally enforces earliest headway."},
        "conflict_recovery": {"classification": "ACTION_CONDITIONED", "reason": "trajectory and SafetyFilter/holder-tube verdicts arise only after a proposal is constructed."},
        "revised_closure_requirements": {"state_only": ["retry", "commit", "phase/ring", "interface frozen", "final capacity", "LZ availability"], "action_conditioned": ["proposal reservation", "headway feasibility", "trajectory dynamics", "SafetyFilter certificate", "holder-tube certificate"], "rule": "Do not require action-conditioned rationales to pass the state-only matched-pair gate; assess them only in proposal-conditioned diagnostics."}}


def run(bank_path: str | Path, output_dir: str | Path) -> dict:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    rows = _read_bank(bank_path); split = _assert_disjoint(rows)
    probe_a, _ = _binary_report(rows, name="A_current_gate_any_certified_release_exists", proposal_conditioned=False, positive_name="RELEASE", negative_name="WAIT", gate=GATE_A)
    probe_b, proposal_records = _binary_report(rows, name="B_proposal_conditioned_certification", proposal_conditioned=True, positive_name="certified", negative_name="rejected", gate=GATE_B)
    proposal_path = output / "state_proposal_certification_records.jsonl"
    proposal_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in proposal_records), encoding="utf-8")
    probe_b["rejection_reason_breakdown"] = _proposal_reason_breakdown(proposal_records, probe_b)
    probe_b["record_path"] = str(proposal_path); probe_b["record_sha256"] = _sha256(proposal_path)
    probe_c = _proposal_selection(rows)
    a_pass = probe_a.get("status") == "PASS"; b_pass = probe_b.get("status") == "PASS"; c_pass = probe_c.get("status") == "PASS"
    if not a_pass and b_pass and c_pass:
        root, factorization, obs_revision, factorization_revision, supervision_revision = ("GATE_BEFORE_PROPOSAL_INFORMATION_MISMATCH", "PROPOSAL_CONDITIONED_GATE", False, True, True)
    elif a_pass:
        root, factorization, obs_revision, factorization_revision, supervision_revision = ("CURRENT_V4_2_GATE_POTENTIALLY_DEPLOYABLE", "KEEP_V4_2_HIERARCHICAL_PENDING_FORMAL_CLOSURE", False, False, True)
    elif not b_pass:
        root, factorization, obs_revision, factorization_revision, supervision_revision = ("DEPLOYABLE_INFORMATION_INSUFFICIENT_FOR_PROPOSAL_FEASIBILITY", "UNDECIDED", True, False, True)
    else:
        root, factorization, obs_revision, factorization_revision, supervision_revision = ("UNRESOLVED", "UNDECIDED", False, False, True)
    result = {"stage": STAGE, "evidence_type": "newly_run", "input_closure_bank": {"path": str(bank_path), "sha256": _sha256(bank_path), "unique_semantic_states": len(rows), "observation": "phaseA_low_obs_v2_1_retry_state", "observation_dim": 88, "catalogue_outcomes_per_state": 25},
              "split": split, "Probe_A_current_gate": probe_a, "Probe_B_proposal_conditioned_certification": probe_b, "Probe_C_proposal_selection": probe_c,
              "closure_rationale_reclassification": _closure_reclassification(), "execution_boundary_preserved": True,
              "actor_never_receives_catalogue_masks_or_certificates": True,
              "current_gate_any_certified_separable": a_pass, "proposal_conditioned_certification_separable": b_pass, "proposal_selection_learnable": c_pass,
              "ROOT_CAUSE": root, "recommended_action_factorization": factorization, "observation_revision_required": obs_revision,
              "action_factorization_revision_required": factorization_revision, "teacher_supervision_revision_required": supervision_revision,
              "OBSERVATION_V2_1_FINAL_FREEZE": "FAIL", "FORMAL_V2_1_I0_RECOLLECTION_ALLOWED": False, "I1_ALLOWED": False, "DAgger_ALLOWED": False, "PPO_ALLOWED": False,
              "forbidden_operations_called": []}
    (output / "v4_2_privileged_teacher_gate_factorization_diagnostic.manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", required=True, help="Existing closure-audit audit_state_bank.jsonl")
    parser.add_argument("--output-dir", required=True)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.bank, arguments.output_dir), indent=2))
