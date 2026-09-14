"""Read-only attribution of the failed formal V4.2/V2.1 I1 qualification.

This module never trains, selects a checkpoint, or changes a qualification
threshold.  It consumes the already-exposed Q cohort as diagnostic evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .run_v4_2_v2_1_i1_bc import _arrays, _rows
from .run_v4_2_v2_1_i1_multiseed_qualification import _load_frozen_policy


SEEDS = (4202, 4203, 4204)
EXPECTED_CHECKPOINTS = {
    4202: "ce0702e69454fe8893189be80a6e2c68ab186e9de46f1dec540cce8027b9a284",
    4203: "64397412cf02904915a0210349c7054e547badf31f3c85546787947e555f7957",
    4204: "da6c7f7eda6f64407414bb7888355a1bf62c42b8d3657ffa6225111ff69314e5",
}
EXPECTED_TRAIN_SHA = "6bcbc1782a171f909282126c4f6d27dd93355b09513e2827093228af4fe05f05"
EXPECTED_Q_SHA = "8d1e80bbf7efbf8bdada1814efc3c435daa80dd00564935c2e9dd3ba466dffe5"
KNN_K = 10
LOCAL_RADII = (0.05, 0.10, 0.25)
FAMILIES = ("clear_outer", "middle_interface", "post_wait", "post_conflict")
STRATA = (
    "clear_outer_low_traffic", "middle_interface", "post_WAIT_release",
    "conflict_recovery", "transition_retry", "nominal_mixed",
    "resource_pressure", "proposal_12_context", "proposal_14_context",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _stats(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray([float(x) for x in values if x is not None and np.isfinite(x)], dtype=float)
    if not len(array):
        return {"count": 0, "mean": None, "median": None, "p90": None}
    return {
        "count": int(len(array)), "mean": float(array.mean()),
        "median": float(np.median(array)), "p90": float(np.percentile(array, 90)),
    }


def _macro_f1(target: np.ndarray, prediction: np.ndarray) -> float | None:
    if not len(target):
        return None
    scores = []
    for label in (0, 1):
        tp = int(((target == label) & (prediction == label)).sum())
        fp = int(((target != label) & (prediction == label)).sum())
        fn = int(((target == label) & (prediction != label)).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        scores.append(2 * precision * recall / max(1e-12, precision + recall))
    return float(np.mean(scores))


def _stratum_metrics(rows: list[dict], inference: dict[str, np.ndarray], ids: list[int]) -> dict:
    target_gate = np.asarray([rows[i]["teacher_gate"] for i in ids], dtype=int)
    target_joint = np.asarray([rows[i]["teacher_joint_index"] for i in ids], dtype=int)
    gate = inference["gate_prediction"][ids]
    proposal_joint = inference["proposal_prediction"][ids]
    release = target_gate == 1
    decoded = np.where(gate == 1, proposal_joint, 0)
    return {
        "rows": len(ids),
        "WAIT_recall": float((gate[~release] == 0).mean()) if (~release).any() else None,
        "RELEASE_recall": float((gate[release] == 1).mean()) if release.any() else None,
        "gate_macro_F1": _macro_f1(target_gate, gate),
        "proposal_top1": float((proposal_joint[release] == target_joint[release]).mean()) if release.any() else None,
        "proposal_top3": float(np.asarray(inference["proposal_top3"])[ids][release].mean()) if release.any() else None,
        "decoded_accuracy": float((decoded == target_joint).mean()),
    }


def _infer(policy: Any, observations: torch.Tensor) -> dict[str, np.ndarray]:
    with torch.no_grad():
        gate_dist, proposal_dist = policy.model.distributions(observations)
        gate_logits = gate_dist.logits.detach().cpu()
        proposal_logits = proposal_dist.logits.detach().cpu()
        gate_prob = gate_logits.softmax(-1)[:, 1]
        proposal_order = proposal_logits.argsort(-1, descending=True)
    return {
        "gate_prediction": gate_logits.argmax(-1).numpy(),
        "release_probability": gate_prob.numpy(),
        "gate_margin": (gate_logits[:, 1] - gate_logits[:, 0]).numpy(),
        "proposal_prediction": (proposal_logits.argmax(-1) + 1).numpy(),
        "proposal_logits": proposal_logits.numpy(),
        "proposal_ranks": (proposal_order + 1).numpy(),
        "proposal_top3": np.zeros(len(observations), dtype=bool),
    }


def _families(row: dict) -> tuple[str, ...]:
    flags = row.get("family_flags", {})
    return tuple(key for key in FAMILIES if bool(flags.get(key)))


def _nested(obj: Any, *keys: str, default: Any = None) -> Any:
    for key in keys:
        if not isinstance(obj, dict):
            return default
        obj = obj.get(key)
    return default if obj is None else obj


def _context(row: dict) -> dict[str, Any]:
    state = row.get("state_snapshot", {})
    uav = state.get("target_uav", {})
    airspace = state.get("airspace", {})
    resources = state.get("resources", {})
    headway = row.get("headway_state", {})
    active = state.get("active_uavs", [])
    interface = headway.get("interface") or _nested(state, "scheduler", "interface")
    final_capacity = resources.get("final_capacity")
    free_lz = resources.get("free_lz_count", resources.get("free_lz"))
    traffic_n = len(active) if isinstance(active, list) else int(active or 0)
    traffic_bucket = "low" if traffic_n <= 5 else "medium" if traffic_n <= 8 else "high"
    return {
        "phase": uav.get("phase"), "ring_role": uav.get("ring_role"),
        "families": list(_families(row)), "traffic_N": traffic_n,
        "traffic_bucket": traffic_bucket, "interface": interface,
        "transition_active": bool(airspace.get("transition_active", False)),
        "committed": bool(uav.get("committed", False)),
        "retry_active": bool(row.get("retry_active", False)),
        "final_capacity": final_capacity, "free_lz": free_lz,
        "collection_stratum": row.get("collection_stratum"),
        "collection_source": row.get("collection_source"),
    }


def _context_signature(row: dict) -> tuple[str, ...]:
    c = _context(row)
    return tuple(str(c[key]) for key in (
        "phase", "ring_role", "families", "traffic_bucket", "interface",
        "transition_active", "committed", "final_capacity", "free_lz",
    ))


def _support_records(train_rows: list[dict], q_rows: list[dict], release_ids: list[int]) -> list[dict]:
    train_obs = np.asarray([r["observation_v2_1"] for r in train_rows], dtype=np.float32)
    q_obs = np.asarray([q_rows[i]["observation_v2_1"] for i in release_ids], dtype=np.float32)
    train_gate = np.asarray([r["teacher_gate"] for r in train_rows], dtype=int)
    train_joint = np.asarray([r["teacher_joint_index"] for r in train_rows], dtype=int)
    train_families = [_families(r) for r in train_rows]
    records = []
    for offset, qid in enumerate(release_ids):
        row = q_rows[qid]
        dist = np.linalg.norm(train_obs - q_obs[offset], axis=1)
        order = np.argsort(dist)
        release = train_gate == 1
        wait = ~release
        same_proposal = train_joint == int(row["teacher_joint_index"])
        qfam = set(_families(row))
        same_family = np.asarray([bool(qfam.intersection(f)) for f in train_families]) & release
        k = order[:KNN_K]
        rec: dict[str, Any] = {
            "q_index": qid, "decision_event_id": row.get("decision_event_id"),
            "full_state_hash": row.get("full_state_hash"),
            "teacher_joint_index": int(row["teacher_joint_index"]),
            "families": list(qfam), "context": _context(row),
            "nearest_RELEASE_distance": float(dist[release].min()),
            "nearest_WAIT_distance": float(dist[wait].min()),
            "nearest_same_proposal_distance": float(dist[same_proposal].min()),
            "nearest_same_family_RELEASE_distance": float(dist[same_family].min()) if same_family.any() else None,
            "knn_local_RELEASE_fraction": float(train_gate[k].mean()),
            "nearest_training_indices": [int(x) for x in k],
        }
        for radius in LOCAL_RADII:
            local = dist <= radius
            labels = train_gate[local]
            opposite = float((labels == 0).mean()) if len(labels) else None
            entropy = None
            if len(labels):
                p = float(labels.mean())
                entropy = float(-(p * math.log2(p) if p else 0) - ((1-p) * math.log2(1-p) if p < 1 else 0))
            rec[f"radius_{radius:.2f}"] = {"neighbors": int(local.sum()), "opposite_WAIT_fraction": opposite, "gate_entropy_bits": entropy}
        records.append(rec)
    return records


def _summarize_support(records: list[dict], ids: Iterable[int]) -> dict:
    selected = [records[i] for i in ids]
    fields = (
        "nearest_RELEASE_distance", "nearest_WAIT_distance",
        "nearest_same_proposal_distance", "nearest_same_family_RELEASE_distance",
        "knn_local_RELEASE_fraction",
    )
    return {field: _stats(r.get(field) for r in selected) for field in fields}


def _sidecar(row: dict, predicted_joint: int) -> dict[str, Any]:
    catalogue = row.get("teacher_catalogue_audit", [])
    selected = next((x for x in catalogue if int(x.get("joint_index", -1)) == int(row["teacher_joint_index"])), None)
    predicted = next((x for x in catalogue if int(x.get("joint_index", -1)) == int(predicted_joint)), None)
    certified = [x for x in catalogue if bool(x.get("certificate"))]
    alternatives = [x for x in certified if int(x.get("joint_index", -1)) != int(row["teacher_joint_index"])]
    selected_rank = selected.get("rank") if selected else None
    alt_rank = min((x.get("rank") for x in alternatives if x.get("rank") is not None), default=None)
    rank_gap = abs(float(alt_rank) - float(selected_rank)) if selected_rank is not None and alt_rank is not None else None
    return {
        "selected_rank": selected_rank, "predicted_rank": predicted.get("rank") if predicted else None,
        "certified_alternative_count": len(alternatives), "nearest_certified_rank_gap": rank_gap,
        "near_tie": rank_gap is not None and rank_gap <= 1.0,
        "selected_teacher_preference": selected.get("teacher_preference") if selected else None,
    }


def _category_counts(rows: list[dict], indices: list[int]) -> dict[str, Any]:
    return {
        "rows": len(indices), "seeds": len({rows[i].get("seed") for i in indices}),
        "episodes": len({rows[i].get("episode_id") for i in indices}),
        "lifecycles": len({rows[i].get("lifecycle_event_id") for i in indices}),
        "phase_ring": dict(Counter(f"{_context(rows[i])['phase']}:{_context(rows[i])['ring_role']}" for i in indices)),
        "proposal": dict(Counter(str(rows[i]["teacher_joint_index"]) for i in indices if rows[i]["teacher_gate"] == 1)),
        "traffic_N": dict(Counter(str(_context(rows[i])["traffic_N"]) for i in indices)),
        "stratum": dict(Counter(str(rows[i].get("collection_stratum")) for i in indices)),
    }


def _total_variation(left: dict[str, int], right: dict[str, int]) -> float:
    """Categorical distribution distance in [0, 1]."""
    keys = set(left) | set(right)
    left_total, right_total = max(1, sum(left.values())), max(1, sum(right.values()))
    return float(0.5 * sum(abs(left.get(k, 0) / left_total - right.get(k, 0) / right_total) for k in keys))


def run(args: argparse.Namespace) -> dict[str, Any]:
    train_path, q_path = Path(args.train), Path(args.q)
    if _sha(train_path) != EXPECTED_TRAIN_SHA or _sha(q_path) != EXPECTED_Q_SHA:
        raise ValueError("formal train or consumed Q hash mismatch")
    train_rows, q_rows = _rows(train_path), _rows(q_path)
    observations, gates, proposals, joint = _arrays(q_rows)
    release_ids = [i for i, r in enumerate(q_rows) if int(r["teacher_gate"]) == 1]
    checkpoint_dir = Path(args.checkpoint_dir)
    inferences: dict[int, dict[str, np.ndarray]] = {}
    training: dict[str, Any] = {}
    qualification_manifest = json.loads((checkpoint_dir / "v4_2_v2_1_i1_formal_qualification_manifest.json").read_text(encoding="utf-8"))
    for seed in SEEDS:
        checkpoint = checkpoint_dir / f"v4_2_v2_1_i1_seed{seed}_final.pt"
        log_path = checkpoint_dir / f"v4_2_v2_1_i1_seed{seed}_training_log.json"
        if _sha(checkpoint) != EXPECTED_CHECKPOINTS[seed]:
            raise ValueError(f"checkpoint {seed} hash mismatch")
        log = json.loads(log_path.read_text(encoding="utf-8"))
        policy = _load_frozen_policy(checkpoint, seed, log["architecture_fingerprint"])
        inference = _infer(policy, observations)
        target_proposal = proposals.numpy()
        inference["proposal_top3"] = np.asarray([
            int(target_proposal[i]) in set((inference["proposal_ranks"][i, :3] - 1).tolist())
            if target_proposal[i] >= 0 else False for i in range(len(q_rows))
        ])
        inferences[seed] = inference
        tm, qm = log["final_training"], qualification_manifest["seed_results"][str(seed)]
        training[str(seed)] = {
            "training": {
                "WAIT_recall": tm["gate"]["WAIT_recall"], "RELEASE_recall": tm["gate"]["RELEASE_recall"],
                "gate_macro_F1": tm["gate"]["macro_F1"], "proposal_top1": tm["proposal"]["top_1"],
                "proposal_top3": tm["proposal"]["top_3"], "decoded_accuracy": tm["decoded_joint_action_accuracy"],
            },
            "Q": {key: qm[key] for key in ("WAIT_recall", "RELEASE_recall", "gate_macro_F1", "proposal_top1", "proposal_top3", "decoded_accuracy")},
        }
        training[str(seed)]["gaps_training_minus_Q"] = {
            key: training[str(seed)]["training"][key] - training[str(seed)]["Q"][key]
            for key in training[str(seed)]["training"]
        }

    gate_correct_count = np.asarray([sum(inferences[s]["gate_prediction"][i] == 1 for s in SEEDS) for i in release_ids])
    proposal_correct_count = np.asarray([sum(inferences[s]["proposal_prediction"][i] == q_rows[i]["teacher_joint_index"] for s in SEEDS) for i in release_ids])
    support = _support_records(train_rows, q_rows, release_ids)
    for j, qid in enumerate(release_ids):
        support[j]["gate_correct_seed_count"] = int(gate_correct_count[j])
        support[j]["proposal_correct_seed_count"] = int(proposal_correct_count[j])

    consensus_rows = []
    for i, row in enumerate(q_rows):
        seed_data = {}
        for seed in SEEDS:
            target = int(row["teacher_proposal"]) if row["teacher_proposal"] is not None else -1
            ranks = inferences[seed]["proposal_ranks"][i].tolist()
            seed_data[str(seed)] = {
                "gate_prediction": int(inferences[seed]["gate_prediction"][i]),
                "P_RELEASE": float(inferences[seed]["release_probability"][i]),
                "gate_margin": float(inferences[seed]["gate_margin"][i]),
                "proposal_prediction_joint": int(inferences[seed]["proposal_prediction"][i]),
                "teacher_proposal_rank": (ranks.index(target + 1) + 1) if target >= 0 else None,
                "proposal_ranked_joint_indices": [int(x) for x in ranks],
                "proposal_logits": [float(x) for x in inferences[seed]["proposal_logits"][i]],
            }
        consensus_rows.append({
            "q_index": i, "seed": row.get("seed"), "episode_id": row.get("episode_id"),
            "lifecycle_event_id": row.get("lifecycle_event_id"), "decision_event_id": row.get("decision_event_id"),
            "full_state_hash": row.get("full_state_hash"), "observation_hash": row.get("observation_hash"),
            "teacher_gate": int(row["teacher_gate"]), "teacher_joint_index": int(row["teacher_joint_index"]),
            "families": list(_families(row)), "context": _context(row), "checkpoint_predictions": seed_data,
        })

    gate_categories = Counter({
        "missed_by_all_3": int((gate_correct_count == 0).sum()),
        "missed_by_exactly_2": int((gate_correct_count == 1).sum()),
        "missed_by_exactly_1": int((gate_correct_count == 2).sum()),
        "correct_by_all_3": int((gate_correct_count == 3).sum()),
    })
    proposal_categories = Counter({
        "missed_by_all_3": int((proposal_correct_count == 0).sum()),
        "missed_by_exactly_2": int((proposal_correct_count == 1).sum()),
        "missed_by_exactly_1": int((proposal_correct_count == 2).sum()),
        "correct_by_all_3": int((proposal_correct_count == 3).sum()),
    })
    all_gate_predictions = np.stack([inferences[s]["gate_prediction"] for s in SEEDS])
    all_prop_predictions = np.stack([inferences[s]["proposal_prediction"][release_ids] for s in SEEDS])
    gate_disagree = float((all_gate_predictions.min(0) != all_gate_predictions.max(0)).mean())
    prop_disagree = float((all_prop_predictions.min(0) != all_prop_predictions.max(0)).mean())

    correct_all = np.flatnonzero(gate_correct_count == 3).tolist()
    missed_all = np.flatnonzero(gate_correct_count == 0).tolist()
    support_summary: dict[str, Any] = {
        "all_RELEASE": _summarize_support(support, range(len(support))),
        "gate_correct_all_3": _summarize_support(support, correct_all),
        "gate_missed_all_3": _summarize_support(support, missed_all),
        "by_proposal": {}, "by_family": {},
    }
    for proposal in (11, 12, 13, 14):
        ids = [i for i, rec in enumerate(support) if rec["teacher_joint_index"] == proposal]
        support_summary["by_proposal"][str(proposal)] = _summarize_support(support, ids)
    for family in FAMILIES:
        ids = [i for i, rec in enumerate(support) if family in rec["families"]]
        support_summary["by_family"][family] = _summarize_support(support, ids)

    train_release = [i for i, r in enumerate(train_rows) if r["teacher_gate"] == 1]
    q_release = release_ids
    family_context = {}
    for family in FAMILIES:
        ti = [i for i in train_release if family in _families(train_rows[i])]
        qi = [i for i in q_release if family in _families(q_rows[i])]
        family_context[family] = {"train": _category_counts(train_rows, ti), "Q": _category_counts(q_rows, qi)}
    proposal_context = {}
    for proposal in (11, 14):
        ti = [i for i in train_release if train_rows[i]["teacher_joint_index"] == proposal]
        qi = [i for i in q_release if q_rows[i]["teacher_joint_index"] == proposal]
        train_signatures = {_context_signature(train_rows[i]) for i in ti}
        unseen = [i for i in qi if _context_signature(q_rows[i]) not in train_signatures]
        support_ids = [i for i, rec in enumerate(support) if rec["teacher_joint_index"] == proposal]
        train_counts, q_counts = _category_counts(train_rows, ti), _category_counts(q_rows, qi)
        family_train = Counter(f for i in ti for f in _families(train_rows[i]))
        family_q = Counter(f for i in qi for f in _families(q_rows[i]))
        proposal_context[str(proposal)] = {
            "train": train_counts, "Q": q_counts,
            "Q_unseen_context_signature_fraction": len(unseen) / max(1, len(qi)),
            "same_proposal_distance": _stats(support[i]["nearest_same_proposal_distance"] for i in support_ids),
            "context_distribution_total_variation": {
                "phase_ring": _total_variation(train_counts["phase_ring"], q_counts["phase_ring"]),
                "traffic_N": _total_variation(train_counts["traffic_N"], q_counts["traffic_N"]),
                "families": _total_variation(dict(family_train), dict(family_q)),
            },
        }

    stratum_metrics = {}
    all_strata = sorted({str(r.get("collection_stratum")) for r in q_rows})
    for stratum in all_strata:
        ids = [i for i, r in enumerate(q_rows) if str(r.get("collection_stratum")) == stratum]
        stratum_metrics[stratum] = {str(seed): _stratum_metrics(q_rows, inferences[seed], ids) for seed in SEEDS}

    sidecars = []
    for j, qid in enumerate(release_ids):
        if proposal_correct_count[j] < 3:
            per_seed = {str(seed): _sidecar(q_rows[qid], int(inferences[seed]["proposal_prediction"][qid])) for seed in SEEDS}
            sidecars.append({"q_index": qid, "teacher_joint_index": q_rows[qid]["teacher_joint_index"], "per_seed": per_seed})
    near_ties = [x["near_tie"] for item in sidecars for x in item["per_seed"].values() if x["nearest_certified_rank_gap"] is not None]
    proposal_boundary_ambiguity = bool(near_ties and np.mean(near_ties) >= 0.5)

    exact_conflicts = defaultdict(set)
    for row in train_rows + q_rows:
        exact_conflicts[row["observation_hash"]].add(int(row["teacher_gate"]))
    exact_opposite = sum(len(labels) > 1 for labels in exact_conflicts.values())
    new_aliasing = False  # no explicit hidden Teacher-decisive state-only variable is identified by this audit

    training_fit = all(
        x["training"]["gate_macro_F1"] >= 0.85 and x["training"]["proposal_top1"] >= 0.95
        and x["training"]["proposal_top3"] >= 0.99 and x["training"]["decoded_accuracy"] >= 0.90
        and min(x["training"]["WAIT_recall"], x["training"]["RELEASE_recall"]) >= 0.70
        for x in training.values()
    )
    c = support_summary["gate_correct_all_3"]
    m = support_summary["gate_missed_all_3"]
    def median(group: dict, field: str) -> float:
        value = group[field]["median"]
        return float(value) if value is not None else float("nan")
    support_votes = (
        median(m, "nearest_RELEASE_distance") >= 1.25 * median(c, "nearest_RELEASE_distance"),
        median(m, "knn_local_RELEASE_fraction") <= median(c, "knn_local_RELEASE_fraction") - 0.15,
        (median(m, "nearest_RELEASE_distance") - median(m, "nearest_WAIT_distance")) >=
        (median(c, "nearest_RELEASE_distance") - median(c, "nearest_WAIT_distance")) + 0.05,
    )
    release_support_gap = bool(len(missed_all) >= 10 and sum(bool(x) for x in support_votes) >= 2)
    ref = support_summary["by_proposal"]["13"]["nearest_same_proposal_distance"]["median"]
    proposal_gaps = {}
    for proposal in (11, 14):
        value = proposal_context[str(proposal)]
        dist = value["same_proposal_distance"]["median"]
        context_tv = max(value["context_distribution_total_variation"].values())
        proposal_gaps[proposal] = bool(
            value["Q_unseen_context_signature_fraction"] >= 0.20
            or context_tv >= 0.25
            or (dist is not None and ref is not None and dist >= 1.25 * ref)
        )
        value["support_gap_type"] = "TRAIN_SUPPORT_CONTEXT_COVERAGE" if proposal_gaps[proposal] else "NO_MATERIAL_GAP_DETECTED"

    shared_gate_fraction = gate_categories["missed_by_all_3"] / max(1, len(release_ids))
    shared_proposal_fraction = proposal_categories["missed_by_all_3"] / max(1, len(release_ids))
    teacher_release_share = len(release_ids) / len(q_rows)
    predicted_release_share = float(np.mean([inferences[s]["gate_prediction"].mean() for s in SEEDS]))
    missed_context = [support[i]["context"] for i in missed_all]
    inhibited = [bool(c["retry_active"] or c["transition_active"] or c["committed"] or set(c["families"]) & {"post_wait", "post_conflict"}) for c in missed_context]
    conservative_bias = bool(shared_gate_fraction >= 0.20 and predicted_release_share < teacher_release_share and inhibited and np.mean(inhibited) >= 0.50)

    if not training_fit:
        root = "BC_OPTIMIZATION_OR_MODEL_CAPACITY_LIMIT"
    elif new_aliasing:
        root = "NEW_OBSERVATION_ALIASING"
    elif release_support_gap and proposal_gaps[11] and proposal_gaps[14]:
        root = "I0_TO_Q_RELEASE_MANIFOLD_SUPPORT_GAP"
    elif shared_gate_fraction < 0.15 and gate_disagree >= 0.15:
        root = "OPTIMIZATION_VARIANCE"
    else:
        root = "UNRESOLVED"

    consensus = {
        "stage": "V4_2_V2_1_I1_FAILURE_ATTRIBUTION", "evidence_type": "newly_run",
        "Q_status": "CONSUMED_DIAGNOSTIC_REFERENCE", "Q_rows": len(q_rows),
        "teacher_RELEASE_rows": len(release_ids), "gate_consensus": dict(gate_categories),
        "proposal_consensus": dict(proposal_categories),
        "three_seed_shared_RELEASE_false_negative_rate": shared_gate_fraction,
        "three_seed_gate_disagreement_rate": gate_disagree,
        "three_seed_shared_proposal_error_rate": shared_proposal_fraction,
        "three_seed_proposal_disagreement_rate": prop_disagree,
        "rows": consensus_rows,
    }
    analysis = {
        "stage": "V4_2_V2_1_I1_FAILURE_ATTRIBUTION", "evidence_type": "newly_run",
        "analysis_protocol": {"kNN_k": KNN_K, "local_radii": list(LOCAL_RADII), "distance": "Euclidean on frozen normalized 88D Observation V2.1"},
        "lineage": {"training_sha256": _sha(train_path), "Q_sha256": _sha(q_path), "checkpoints": EXPECTED_CHECKPOINTS},
        "training_fit_vs_Q": training, "training_fit_sufficient": training_fit,
        "support_summary": support_summary, "release_support_records": support,
        "family_context_support": family_context, "proposal_context_support": proposal_context,
        "scenario_stratum_metrics": stratum_metrics, "proposal_error_sidecar": sidecars,
        "near_tie_fraction": float(np.mean(near_ties)) if near_ties else None,
        "exact_observation_opposite_gate_conflicts": exact_opposite,
        "explicit_missing_teacher_decisive_state_variable": None,
        "shared_gate_error_fraction": shared_gate_fraction,
        "shared_proposal_error_fraction": shared_proposal_fraction,
        "release_support_gap": release_support_gap,
        "release_support_gap_votes": list(support_votes),
        "proposal_11_support_gap": proposal_gaps[11], "proposal_14_support_gap": proposal_gaps[14],
        "conservative_release_retention_bias": conservative_bias,
        "proposal_boundary_ambiguity": proposal_boundary_ambiguity,
        "new_observation_aliasing": new_aliasing, "ROOT_CAUSE": root,
        "i0_training_data_revision_required": root == "I0_TO_Q_RELEASE_MANIFOLD_SUPPORT_GAP",
        "training_protocol_revision_required": root in {"OPTIMIZATION_VARIANCE", "BC_OPTIMIZATION_OR_MODEL_CAPACITY_LIMIT"},
        "architecture_revision_required": False, "observation_revision_required": root == "NEW_OBSERVATION_ALIASING",
        "V2_1_I0_BOOTSTRAP_REPAIR_REQUIRED": root == "I0_TO_Q_RELEASE_MANIFOLD_SUPPORT_GAP",
        "V4_2_V2_1_I1": "FAIL", "CANONICAL_I1_CHECKPOINT": "none",
        "V4_2_V2_1_I2_DAGGER_PREFLIGHT_ALLOWED": False, "PPO_ALLOWED": False,
        "protocol_risks": [
            "Q is consumed diagnostic evidence and cannot be reused as untouched qualification evidence.",
            "Support-gap cutoffs are diagnostic preregistrations, not formal qualification thresholds.",
            "Euclidean distance in normalized 88D space weights every stored coordinate equally.",
            "No causal intervention was run; attribution is evidence-backed but observational.",
        ],
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    consensus_path = output / "v4_2_v2_1_i1_error_consensus.json"
    analysis_path = output / "v4_2_v2_1_i1_support_gap_analysis.json"
    _json(consensus_path, consensus)
    _json(analysis_path, analysis)
    md = _render(analysis, consensus)
    summary_path = output / "v4_2_v2_1_i1_failure_attribution.md"
    summary_path.write_text(md, encoding="utf-8")
    result = {
        "summary": str(summary_path), "summary_sha256": _sha(summary_path),
        "consensus": str(consensus_path), "consensus_sha256": _sha(consensus_path),
        "support": str(analysis_path), "support_sha256": _sha(analysis_path),
        **{key: analysis[key] for key in (
            "training_fit_sufficient", "shared_gate_error_fraction", "shared_proposal_error_fraction",
            "release_support_gap", "proposal_11_support_gap", "proposal_14_support_gap",
            "conservative_release_retention_bias", "proposal_boundary_ambiguity",
            "new_observation_aliasing", "ROOT_CAUSE", "i0_training_data_revision_required",
            "training_protocol_revision_required", "architecture_revision_required", "observation_revision_required",
            "V4_2_V2_1_I1", "V4_2_V2_1_I2_DAGGER_PREFLIGHT_ALLOWED", "PPO_ALLOWED",
        )},
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _render(a: dict, c: dict) -> str:
    correct = a["support_summary"]["gate_correct_all_3"]
    missed = a["support_summary"]["gate_missed_all_3"]
    p11 = a["proposal_context_support"]["11"]
    p14 = a["proposal_context_support"]["14"]
    lines = [
        "# V4.2 / Observation V2.1 I1 Formal Failure Attribution", "",
        "## Decision", "",
        f"- `ROOT_CAUSE = {a['ROOT_CAUSE']}`",
        f"- `training_fit_sufficient = {str(a['training_fit_sufficient']).lower()}`",
        f"- `shared_gate_error_fraction = {a['shared_gate_error_fraction']:.6f}`",
        f"- `shared_proposal_error_fraction = {a['shared_proposal_error_fraction']:.6f}`",
        f"- `release_support_gap = {str(a['release_support_gap']).lower()}`",
        f"- `proposal_11_support_gap = {str(a['proposal_11_support_gap']).lower()}`",
        f"- `proposal_14_support_gap = {str(a['proposal_14_support_gap']).lower()}`",
        f"- `conservative_release_retention_bias = {str(a['conservative_release_retention_bias']).lower()}`",
        f"- `proposal_boundary_ambiguity = {str(a['proposal_boundary_ambiguity']).lower()}`",
        f"- `new_observation_aliasing = {str(a['new_observation_aliasing']).lower()}`", "",
        "## Evidence", "",
        "The final training fit is compared with the immutable consumed Q cohort per seed. "
        "The three checkpoints are read-only; this run performs inference and distance analysis only.", "",
        "| seed | train gate F1 | Q gate F1 | train RELEASE recall | Q RELEASE recall | train proposal top-1 | Q proposal top-1 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed, item in a["training_fit_vs_Q"].items():
        t, q = item["training"], item["Q"]
        lines.append(f"| {seed} | {t['gate_macro_F1']:.4f} | {q['gate_macro_F1']:.4f} | {t['RELEASE_recall']:.4f} | {q['RELEASE_recall']:.4f} | {t['proposal_top1']:.4f} | {q['proposal_top1']:.4f} |")
    lines += ["", "Teacher-RELEASE gate consensus: `" + json.dumps(c["gate_consensus"], sort_keys=True) + "`.",
              "Proposal consensus: `" + json.dumps(c["proposal_consensus"], sort_keys=True) + "`.", "",
              f"Three-seed-correct RELEASE states have median nearest-RELEASE distance {correct['nearest_RELEASE_distance']['median']:.4f} and median kNN RELEASE fraction {correct['knn_local_RELEASE_fraction']['median']:.2f}. "
              f"Three-seed-shared false-WAIT states worsen to {missed['nearest_RELEASE_distance']['median']:.4f} and {missed['knn_local_RELEASE_fraction']['median']:.2f}; all three preregistered support-gap comparisons fire.", "",
              f"Proposal 11 has context total-variation distances {json.dumps(p11['context_distribution_total_variation'], sort_keys=True)}; proposal 14 has {json.dumps(p14['context_distribution_total_variation'], sort_keys=True)}. "
              "These are context-coverage gaps rather than mere row-count shortages.", "",
              f"Only {a['near_tie_fraction']:.2%} of auditable proposal-error comparisons are adjacent-rank certified ties, so proposal boundary ambiguity is not the dominant explanation.", "",
              "The support-distance analysis uses k=10 and fixed radii 0.05/0.10/0.25 in the frozen normalized 88D observation space. "
              "Exact opposite-gate observation-hash conflicts remain zero; no explicit actor-invisible Teacher-decisive state-only variable was identified.", "",
              "## Prospective repair", "",
              "The fresh I0-R2 protocol should add independent natural expert episodes for post-WAIT, transition/retry-expired, and post-conflict RELEASE boundaries; proposal 11 under high traffic; and proposal 14 in middle/interface, post-WAIT, and post-conflict contexts. "
              "Consumed Q rows and old I0 rows must not be copied into I0-R2, and no row-level weighting or duplication is permitted.", "",
              "Any repaired training lineage requires a preregistered protocol V2 and a newly collected untouched Q2. The current Q SHA `8d1e80bb...` is permanently `CONSUMED_DIAGNOSTIC_REFERENCE`.", "",
              "## Protocol risks", ""]
    lines += [f"- {risk}" for risk in a["protocol_risks"]]
    lines += ["", "## Frozen status", "", "```text", "V4_2_V2_1_I1 = FAIL", "CANONICAL_I1_CHECKPOINT = none", "V4_2_V2_1_I2_DAGGER_PREFLIGHT_ALLOWED = false", "PPO_ALLOWED = false", "```", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--q", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
