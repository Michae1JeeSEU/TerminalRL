"""Read-only attribution of the failed V4.2/V2.1 I0-R2 support gate.

The consumed Q cohort is diagnostic-only.  This module neither changes a
qualification threshold nor creates training rows; it decomposes geometric
coverage from local gate-label purity and compares R2 with the old I0-R.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .run_v4_2_v2_1_i1_bc import _rows


EXPECTED_R2_SHA = "c2f0f947a2b02d8977757924cc2bfe3ddeb68e0552e9158aedd8cc432394fd3a"
EXPECTED_OLD_SHA = "6bcbc1782a171f909282126c4f6d27dd93355b09513e2827093228af4fe05f05"
EXPECTED_Q_SHA = "8d1e80bbf7efbf8bdada1814efc3c435daa80dd00564935c2e9dd3ba466dffe5"
EXPECTED_CONSENSUS_SHA = "934d12a1b58589661846b5924d5d4e95f342e80b2eec4c67b0b9f6e6fcbc0a01"
K = 10
RADII = (0.05, 0.10, 0.25, 0.50, 1.00)
ENTROPY_RADII = (0.05, 0.10, 0.25)
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 4215
PRIMARY_SUPPORT_RADIUS = 0.25  # diagnostic reporting radius; not a formal gate


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _stats(values: Iterable[float | None]) -> dict[str, float | int | None]:
    a = np.asarray([float(v) for v in values if v is not None and np.isfinite(v)], dtype=float)
    if not len(a):
        return {"count": 0, "mean": None, "median": None, "p10": None, "p90": None}
    return {"count": int(len(a)), "mean": float(a.mean()), "median": float(np.median(a)),
            "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90))}


def _entropy(p: float) -> float:
    return float(-(p * math.log2(p) if p else 0.0) - ((1.0 - p) * math.log2(1.0 - p) if p < 1.0 else 0.0))


def _target_uav(row: dict) -> dict:
    return row.get("state_snapshot", {}).get("target_uav", {})


def _families(row: dict) -> tuple[str, ...]:
    flags = row.get("family_flags", {})
    return tuple(k for k in ("clear_outer", "middle_interface", "post_wait", "post_conflict") if flags.get(k))


def _is_transition_retry(row: dict) -> bool:
    kind = row.get("transition_ledger", {}).get("wait_release_transition_kind")
    return row.get("collection_stratum") in {"transition_retry", "retry_expired_transition_release"} or kind in {
        "retry_WAIT_to_RELEASE", "transition_WAIT_to_RELEASE"
    }


def _groups(row: dict) -> set[str]:
    groups = {"all_RELEASE"}
    families = set(_families(row))
    if "post_wait" in families:
        groups.add("post_WAIT")
    if "post_conflict" in families:
        groups.add("post_conflict")
    if _is_transition_retry(row):
        groups.add("transition_retry")
    joint = int(row.get("teacher_joint_index", -1))
    if joint == 11:
        groups.add("proposal11")
    if joint == 14:
        groups.add("proposal14")
    return groups


def _context(row: dict) -> dict[str, Any]:
    uav = _target_uav(row)
    state = row.get("state_snapshot", {})
    active = state.get("active_uavs", [])
    ledger = row.get("transition_ledger", {})
    headway = row.get("headway_state", {})
    family = "+".join(_families(row)) or "none"
    previous = ledger.get("previous_wait_rationale") or ledger.get("previous_WAIT_rationale") or "none"
    conflict = bool(row.get("family_flags", {}).get("post_conflict"))
    return {
        "traffic_N": str(len(active)), "phase": str(uav.get("phase")),
        "ring_role": str(uav.get("ring_role")), "family": family,
        "previous_WAIT_rationale": str(previous),
        "interface_context": str(headway.get("interface", "none")),
        "conflict_context": "post_conflict" if conflict else "non_conflict",
    }


def _neighbor_payload(rows: list[dict], order: np.ndarray, distances: np.ndarray) -> list[dict]:
    result = []
    for i in order[:K]:
        row = rows[int(i)]
        result.append({
            "row_index": int(i), "distance": float(distances[int(i)]),
            "teacher_gate": int(row["teacher_gate"]),
            "proposal_id": int(row["teacher_joint_index"]) if int(row["teacher_gate"]) == 1 else None,
            "family_flags": row.get("family_flags", {}),
            "lifecycle_event_id": row.get("lifecycle_event_id"),
        })
    return result


def _support_record(reference_row: dict, candidate_rows: list[dict], candidate_obs: np.ndarray,
                    candidate_gate: np.ndarray, candidate_joint: np.ndarray) -> dict[str, Any]:
    q = np.asarray(reference_row["observation_v2_1"], dtype=np.float32)
    distance = np.linalg.norm(candidate_obs - q, axis=1)
    order = np.argsort(distance, kind="stable")
    release = candidate_gate == 1
    wait = ~release
    proposal = int(reference_row["teacher_joint_index"])
    same_proposal = release & (candidate_joint == proposal)
    qfamilies = set(_families(reference_row))
    same_family = release & np.asarray([bool(qfamilies.intersection(_families(r))) for r in candidate_rows])
    kidx = order[:K]
    p = float(candidate_gate[kidx].mean())
    nearest_release = float(distance[release].min()) if release.any() else None
    nearest_wait = float(distance[wait].min()) if wait.any() else None
    nearest_same_proposal = float(distance[same_proposal].min()) if same_proposal.any() else None
    nearest_same_family = float(distance[same_family].min()) if same_family.any() else None
    radii = {}
    for radius in ENTROPY_RADII:
        local = candidate_gate[distance <= radius]
        lp = float(local.mean()) if len(local) else None
        radii[f"{radius:.2f}"] = {
            "neighbor_count": int(len(local)), "RELEASE_fraction": lp,
            "opposite_WAIT_fraction": None if lp is None else 1.0 - lp,
            "gate_entropy_bits": None if lp is None else _entropy(lp),
        }
    return {
        "nearest_RELEASE_distance": nearest_release,
        "nearest_WAIT_distance": nearest_wait,
        "nearest_same_proposal_RELEASE_distance": nearest_same_proposal,
        "nearest_same_family_RELEASE_distance": nearest_same_family,
        "knn_RELEASE_count": int(candidate_gate[kidx].sum()),
        "knn_WAIT_count": int(K - candidate_gate[kidx].sum()),
        "knn_RELEASE_fraction": p, "knn_gate_entropy_bits": _entropy(p),
        "covered_by_release_neighbor": nearest_release is not None and nearest_release <= PRIMARY_SUPPORT_RADIUS,
        "local_neighborhood_mixed": bool(0 < candidate_gate[kidx].sum() < K),
        "radius_entropy": radii,
        "nearest_neighbors": _neighbor_payload(candidate_rows, order, distance),
    }


def _records(reference_rows: list[dict], candidate_rows: list[dict]) -> list[dict]:
    obs = np.asarray([r["observation_v2_1"] for r in candidate_rows], dtype=np.float32)
    gate = np.asarray([int(r["teacher_gate"]) for r in candidate_rows], dtype=np.int8)
    joint = np.asarray([int(r["teacher_joint_index"]) for r in candidate_rows], dtype=np.int16)
    output = []
    for q_index, row in enumerate(reference_rows):
        if int(row["teacher_gate"]) != 1:
            continue
        support = _support_record(row, candidate_rows, obs, gate, joint)
        output.append({
            "q_index": q_index, "decision_event_id": row.get("decision_event_id"),
            "full_state_hash": row.get("full_state_hash"), "groups": sorted(_groups(row)),
            "teacher_proposal": int(row["teacher_joint_index"]), "context": _context(row), **support,
        })
    return output


def _coverage_summary(records: list[dict]) -> dict[str, Any]:
    output = {}
    for group in ("all_RELEASE", "post_WAIT", "transition_retry", "post_conflict", "proposal11", "proposal14"):
        selected = [r for r in records if group in r["groups"]]
        distance = [r["nearest_RELEASE_distance"] for r in selected]
        output[group] = {
            "states": len(selected),
            "nearest_RELEASE_distance": _stats(distance),
            "nearest_WAIT_distance": _stats(r["nearest_WAIT_distance"] for r in selected),
            "nearest_same_proposal_RELEASE_distance": _stats(r["nearest_same_proposal_RELEASE_distance"] for r in selected),
            "nearest_same_family_RELEASE_distance": _stats(r["nearest_same_family_RELEASE_distance"] for r in selected),
            "knn_RELEASE_fraction": _stats(r["knn_RELEASE_fraction"] for r in selected),
            "release_radius_coverage": {
                "distance_0": float(np.mean([d == 0.0 for d in distance])) if distance else None,
                **{f"distance_le_{radius:.2f}": float(np.mean([d <= radius for d in distance])) if distance else None for radius in RADII},
            },
            "covered_and_mixed_fraction": float(np.mean([
                r["covered_by_release_neighbor"] and r["local_neighborhood_mixed"] for r in selected
            ])) if selected else None,
            "knn_gate_entropy_bits": _stats(r["knn_gate_entropy_bits"] for r in selected),
            "radius_entropy": {f"{radius:.2f}": {
                "gate_entropy_bits": _stats(r["radius_entropy"][f"{radius:.2f}"]["gate_entropy_bits"] for r in selected),
                "opposite_WAIT_fraction": _stats(r["radius_entropy"][f"{radius:.2f}"]["opposite_WAIT_fraction"] for r in selected),
            } for radius in ENTROPY_RADII},
        }
    return output


def _q_intrinsic(q_rows: list[dict]) -> tuple[list[dict], dict[str, Any]]:
    obs = np.asarray([r["observation_v2_1"] for r in q_rows], dtype=np.float32)
    gate = np.asarray([int(r["teacher_gate"]) for r in q_rows], dtype=np.int8)
    records = []
    for q_index, row in enumerate(q_rows):
        if int(row["teacher_gate"]) != 1:
            continue
        distance = np.linalg.norm(obs - obs[q_index], axis=1)
        distance[q_index] = np.inf
        idx = np.argsort(distance, kind="stable")[:K]
        p = float(gate[idx].mean())
        records.append({
            "q_index": q_index, "groups": sorted(_groups(row)),
            "Q_intrinsic_RELEASE_fraction": p,
            "Q_intrinsic_RELEASE_count": int(gate[idx].sum()),
            "Q_intrinsic_gate_entropy_bits": _entropy(p),
            "neighbor_q_indices": [int(i) for i in idx],
            "neighbor_distances": [float(distance[i]) for i in idx],
            "neighbor_gates": [int(gate[i]) for i in idx],
        })
    summary = {}
    for group in ("all_RELEASE", "post_WAIT", "transition_retry", "post_conflict", "proposal11", "proposal14"):
        selected = [r for r in records if group in r["groups"]]
        summary[group] = {
            "states": len(selected),
            "Q_intrinsic_knn_RELEASE_fraction": _stats(r["Q_intrinsic_RELEASE_fraction"] for r in selected),
            "Q_intrinsic_gate_entropy_bits": _stats(r["Q_intrinsic_gate_entropy_bits"] for r in selected),
        }
    return records, summary


def _prior_audit(rows: list[dict], q_rows: list[dict], records: list[dict]) -> dict[str, Any]:
    def prior(selected: list[dict]) -> dict[str, float | int]:
        release = sum(int(r["teacher_gate"]) == 1 for r in selected)
        wait = len(selected) - release
        return {"rows": len(selected), "WAIT": wait, "RELEASE": release,
                "RELEASE_fraction": release / max(1, len(selected)), "WAIT_to_RELEASE": wait / max(1, release)}
    groups: dict[str, Callable[[dict], bool]] = {
        "global_R2": lambda _: True,
        "post_WAIT": lambda r: r.get("collection_stratum") in {"post_wait_release", "post_wait_proposal14"},
        "transition_retry": lambda r: r.get("collection_stratum") == "retry_expired_transition_release",
        "post_conflict": lambda r: r.get("collection_stratum") in {"conflict_recovery_release", "post_conflict_proposal14"},
    }
    result = {name: prior([r for r in rows if predicate(r)]) for name, predicate in groups.items()}
    result["global_Q"] = prior(q_rows)
    result["family_local_Q"] = {
        "post_WAIT": prior([r for r in q_rows if r.get("collection_stratum") == "post_WAIT_release"]),
        "transition_retry": prior([r for r in q_rows if r.get("collection_stratum") == "transition_retry"]),
        "post_conflict": prior([r for r in q_rows if r.get("collection_stratum") == "conflict_recovery"]),
    }
    train_prior = float(result["global_R2"]["RELEASE_fraction"])
    adjusted = []
    for rec in records:
        p = float(rec["knn_RELEASE_fraction"])
        numerator = p / max(train_prior, 1e-12)
        denominator = numerator + (1.0 - p) / max(1.0 - train_prior, 1e-12)
        adjusted.append({"q_index": rec["q_index"], "groups": rec["groups"],
                         "raw": p, "class_prior_adjusted": numerator / max(denominator, 1e-12)})
    result["diagnostic_adjusted_scores"] = {
        group: {"raw": _stats(x["raw"] for x in adjusted if group in x["groups"]),
                "class_prior_adjusted": _stats(x["class_prior_adjusted"] for x in adjusted if group in x["groups"])}
        for group in ("all_RELEASE", "post_WAIT", "transition_retry", "post_conflict")
    }
    result["adjusted_metric_is_qualification_gate"] = False
    return result


def _improvement(old_records: list[dict], r2_records: list[dict]) -> dict[str, Any]:
    old = {r["q_index"]: r for r in old_records}
    result = {}
    for group in ("post_WAIT", "transition_retry", "post_conflict", "proposal11", "proposal14"):
        pairs = [(old[r["q_index"]], r) for r in r2_records if group in r["groups"]]
        old_dist = np.asarray([a["nearest_RELEASE_distance"] for a, _ in pairs], dtype=float)
        new_dist = np.asarray([b["nearest_RELEASE_distance"] for _, b in pairs], dtype=float)
        old_knn = np.asarray([a["knn_RELEASE_fraction"] for a, _ in pairs], dtype=float)
        new_knn = np.asarray([b["knn_RELEASE_fraction"] for _, b in pairs], dtype=float)
        om, nm = float(np.median(old_dist)), float(np.median(new_dist))
        result[group] = {
            "states": len(pairs), "old_nearest_RELEASE_median": om,
            "R2_nearest_RELEASE_median": nm,
            "distance_improvement_ratio_old_over_R2": None if nm == 0 else om / nm,
            "distance_relative_reduction": None if om == 0 else (om - nm) / om,
            "fraction_distance_improved": float(np.mean(new_dist < old_dist)),
            "old_knn_RELEASE_fraction_median": float(np.median(old_knn)),
            "R2_knn_RELEASE_fraction_median": float(np.median(new_knn)),
            "knn_fraction_improvement": float(np.median(new_knn) - np.median(old_knn)),
        }
    return result


def _failed_state_coverage(consensus: dict, old_records: list[dict], r2_records: list[dict]) -> dict[str, Any]:
    old = {r["q_index"]: r for r in old_records}
    r2 = {r["q_index"]: r for r in r2_records}
    shared_fn, shared_prop = [], []
    for row in consensus["rows"]:
        if int(row["teacher_gate"]) != 1:
            continue
        predictions = list(row["checkpoint_predictions"].values())
        qid = int(row["q_index"])
        if all(int(x["gate_prediction"]) == 0 for x in predictions):
            shared_fn.append(qid)
        if all(int(x["proposal_prediction_joint"]) != int(row["teacher_joint_index"]) for x in predictions):
            shared_prop.append(qid)

    def detail(ids: list[int], proposal: bool) -> dict[str, Any]:
        rows = []
        for qid in ids:
            before, after = old[qid], r2[qid]
            primary = after["nearest_same_proposal_RELEASE_distance"] if proposal else after["nearest_RELEASE_distance"]
            rows.append({
                "q_index": qid, "groups": after["groups"], "context": after["context"],
                "old_nearest_RELEASE_distance": before["nearest_RELEASE_distance"],
                "R2_nearest_RELEASE_distance": after["nearest_RELEASE_distance"],
                "old_same_family_distance": before["nearest_same_family_RELEASE_distance"],
                "R2_same_family_distance": after["nearest_same_family_RELEASE_distance"],
                "old_same_proposal_distance": before["nearest_same_proposal_RELEASE_distance"],
                "R2_same_proposal_distance": after["nearest_same_proposal_RELEASE_distance"],
                "old_knn_RELEASE_fraction": before["knn_RELEASE_fraction"],
                "R2_knn_RELEASE_fraction": after["knn_RELEASE_fraction"],
                "support_closed_at_0_25": primary is not None and primary <= PRIMARY_SUPPORT_RADIUS,
                "nearest_R2_neighbors": after["nearest_neighbors"],
            })
        def primary_distance(row: dict) -> float:
            value = row["R2_same_proposal_distance"] if proposal else row["R2_nearest_RELEASE_distance"]
            return float("inf") if value is None else float(value)
        return {
            "count": len(rows), "primary_diagnostic_radius": PRIMARY_SUPPORT_RADIUS,
            "support_closed_fraction": float(np.mean([r["support_closed_at_0_25"] for r in rows])) if rows else None,
            "coverage_by_radius": {f"{radius:.2f}": float(np.mean([
                primary_distance(r) <= radius for r in rows
            ])) if rows else None for radius in RADII},
            "rows": rows,
        }
    return {"shared_gate_false_negative_RELEASE": detail(shared_fn, False),
            "shared_proposal_errors": detail(shared_prop, True)}


def _distribution(rows: list[dict], field: str) -> dict[str, int]:
    return dict(Counter(_context(r)[field] for r in rows))


def _tv(left: dict[str, int], right: dict[str, int]) -> float:
    keys = set(left) | set(right)
    nl, nr = max(1, sum(left.values())), max(1, sum(right.values()))
    return float(0.5 * sum(abs(left.get(k, 0) / nl - right.get(k, 0) / nr) for k in keys))


def _tv_decomposition(r2_rows: list[dict], q_rows: list[dict]) -> dict[str, Any]:
    r2 = [r for r in r2_rows if int(r["teacher_gate"]) == 1 and int(r["teacher_joint_index"]) == 14]
    q = [r for r in q_rows if int(r["teacher_gate"]) == 1 and int(r["teacher_joint_index"]) == 14]
    fields = ("traffic_N", "phase", "ring_role", "family", "previous_WAIT_rationale", "interface_context", "conflict_context")
    dimensions = {}
    for field in fields:
        rd, qd = _distribution(r2, field), _distribution(q, field)
        keys = sorted(set(rd) | set(qd))
        nr, nq = max(1, len(r2)), max(1, len(q))
        dimensions[field] = {
            "TV": _tv(rd, qd), "R2_counts": rd, "Q_counts": qd,
            "R2_distribution": {k: rd.get(k, 0) / nr for k in keys},
            "Q_distribution": {k: qd.get(k, 0) / nq for k in keys},
            "absolute_probability_difference": {k: abs(rd.get(k, 0) / nr - qd.get(k, 0) / nq) for k in keys},
        }
    dominant = max(dimensions, key=lambda x: dimensions[x]["TV"])
    formal_dimensions = ("traffic_N", "phase", "ring_role", "family")
    formal_dominant = max(formal_dimensions, key=lambda x: dimensions[x]["TV"])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        boot = [q[i] for i in rng.integers(0, len(q), size=len(q))]
        samples.append(_tv(_distribution(r2, formal_dominant), _distribution(boot, formal_dominant)))
    a = np.asarray(samples)
    return {
        "R2_proposal14_rows": len(r2), "Q_proposal14_rows": len(q),
        "dimensions": dimensions, "all_dimensions_dominant": dominant,
        "formal_gate_dominant_dimension": formal_dominant,
        "formal_observed_max_TV": dimensions[formal_dominant]["TV"],
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED,
                      "dimension": formal_dominant, "median": float(np.median(a)),
                      "CI95": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))],
                      "fraction_le_formal_threshold_0_20": float(np.mean(a <= 0.20))},
    }


def _render(result: dict[str, Any]) -> str:
    s = result["summary"]
    q = result["q_intrinsic"]["summary"]
    tv = result["proposal14_tv"]
    return "\n".join([
        "# V4.2 / Observation V2.1 I0-R2 Support-Closure Failure Attribution", "",
        "## Decision", "",
        f"- `ROOT_CAUSE = {s['ROOT_CAUSE']}`",
        f"- `LOCAL_GATE_BOUNDARY_MIXING = {str(s['LOCAL_GATE_BOUNDARY_MIXING']).lower()}`",
        f"- `KNN_PURITY_GATE_NOT_CALIBRATED_TO_REFERENCE_MANIFOLD = {str(s['KNN_PURITY_GATE_NOT_CALIBRATED_TO_REFERENCE_MANIFOLD']).lower()}`",
        f"- `new_collection_required = {str(s['new_collection_required']).lower()}`",
        f"- `support_metric_revision_required = {str(s['support_metric_revision_required']).lower()}`", "",
        "## Key evidence", "",
        f"R2 radius coverage for all Q RELEASE states: `{json.dumps(s['release_radius_coverage'], sort_keys=True)}`.",
        f"Q-intrinsic kNN RELEASE medians: post-WAIT `{q['post_WAIT']['Q_intrinsic_knn_RELEASE_fraction']['median']}`, transition/retry `{q['transition_retry']['Q_intrinsic_knn_RELEASE_fraction']['median']}`, post-conflict `{q['post_conflict']['Q_intrinsic_knn_RELEASE_fraction']['median']}`.",
        f"R2-to-Q medians remain `{s['post_WAIT_R2_knn_RELEASE_fraction']}`, `{s['transition_retry_R2_knn_RELEASE_fraction']}`, and `{s['post_conflict_R2_knn_RELEASE_fraction']}` respectively.",
        f"Shared three-seed false-negative support closure is `{s['shared_FN_support_closed_fraction']}`; shared proposal-error closure is `{s['shared_proposal_error_support_closed_fraction']}` at the diagnostic radius 0.25.",
        f"Proposal-14 formal dominant TV dimension is `{tv['formal_gate_dominant_dimension']}` with observed TV `{tv['formal_observed_max_TV']}` and bootstrap 95% interval `{tv['bootstrap']['CI95']}`.", "",
        "## Interpretation", "",
        s["interpretation"], "",
        "The failed R2 decision is not changed retrospectively. Any revised support metric is prospective only; the consumed Q remains a diagnostic reference and no Q row enters training.", "",
        "## Evidence inventory", "",
        "- Status: `newly_run` read-only distance/entropy/bootstrap attribution.",
        "- Inputs: R2 failed qualification artifact, old I0-R, consumed Q, and frozen three-seed error consensus.",
        "- Aggregation: per Q Teacher-RELEASE state, then family/proposal medians; Q intrinsic analysis is leave-one-out.",
        "- Protocol risks: normalized 88D Euclidean distance is descriptive rather than causal; bootstrap quantifies finite-Q sampling only; Q is consumed and cannot become a new qualification cohort.", "",
        "## Locks", "",
        "```text",
        "V4_2_V2_1_I0_R2 = FAIL",
        "V4_2_V2_1_I1_V2_PROTOCOL_PREREGISTRATION_ALLOWED = false",
        "V4_2_V2_1_I2_DAGGER_PREFLIGHT_ALLOWED = false",
        "PPO_ALLOWED = false",
        "```", "",
    ])


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = {name: Path(getattr(args, name)) for name in ("r2", "old", "q", "consensus")}
    expected = {"r2": EXPECTED_R2_SHA, "old": EXPECTED_OLD_SHA, "q": EXPECTED_Q_SHA,
                "consensus": EXPECTED_CONSENSUS_SHA}
    hashes = {name: _sha(path) for name, path in paths.items()}
    if hashes != expected:
        raise ValueError(f"frozen input hash mismatch: {hashes}")
    r2_rows, old_rows, q_rows = _rows(paths["r2"]), _rows(paths["old"]), _rows(paths["q"])
    consensus = json.loads(paths["consensus"].read_text(encoding="utf-8"))
    r2_records = _records(q_rows, r2_rows)
    old_records = _records(q_rows, old_rows)
    r2_summary = _coverage_summary(r2_records)
    old_summary = _coverage_summary(old_records)
    q_records, q_summary = _q_intrinsic(q_rows)
    priors = _prior_audit(r2_rows, q_rows, r2_records)
    improvement = _improvement(old_records, r2_records)
    failed = _failed_state_coverage(consensus, old_records, r2_records)
    tv = _tv_decomposition(r2_rows, q_rows)

    key = ("post_WAIT", "transition_retry", "post_conflict")
    local_mixing = bool(np.mean([r2_summary[g]["covered_and_mixed_fraction"] for g in key]) >= 0.50)
    intrinsic_low = [q_summary[g]["Q_intrinsic_knn_RELEASE_fraction"]["median"] < 0.60 for g in key]
    purity_misaligned = bool(sum(intrinsic_low) >= 2 and local_mixing)
    fn_closed = failed["shared_gate_false_negative_RELEASE"]["support_closed_fraction"]
    prop_closed = failed["shared_proposal_errors"]["support_closed_fraction"]
    true_residual = bool((fn_closed is not None and fn_closed < 0.80) or
                         (prop_closed is not None and prop_closed < 0.80))
    proposal14_robust = bool(tv["formal_observed_max_TV"] > 0.20 and tv["bootstrap"]["CI95"][0] > 0.20)
    if purity_misaligned and not true_residual and not proposal14_robust:
        root = "SUPPORT_CLOSURE_METRIC_MISALIGNED"
    elif purity_misaligned and (true_residual or proposal14_robust):
        root = "MIXED_SUPPORT_AND_METRIC_FAILURE"
    elif true_residual:
        root = "TRUE_RESIDUAL_RELEASE_SUPPORT_GAP"
    else:
        root = "UNRESOLVED"
    summary = {
        "release_radius_coverage": r2_summary["all_RELEASE"]["release_radius_coverage"],
        "post_WAIT_R2_knn_RELEASE_fraction": r2_summary["post_WAIT"]["knn_RELEASE_fraction"]["median"],
        "transition_retry_R2_knn_RELEASE_fraction": r2_summary["transition_retry"]["knn_RELEASE_fraction"]["median"],
        "post_conflict_R2_knn_RELEASE_fraction": r2_summary["post_conflict"]["knn_RELEASE_fraction"]["median"],
        "shared_FN_support_closed_fraction": fn_closed,
        "shared_proposal_error_support_closed_fraction": prop_closed,
        "LOCAL_GATE_BOUNDARY_MIXING": local_mixing,
        "KNN_PURITY_GATE_NOT_CALIBRATED_TO_REFERENCE_MANIFOLD": purity_misaligned,
        "SUPPORT_DISTANCE_CLOSED_BUT_LOCAL_PURITY_GATE_FAILED": bool(not true_residual and purity_misaligned),
        "ROOT_CAUSE": root,
        "new_collection_required": root in {"TRUE_RESIDUAL_RELEASE_SUPPORT_GAP", "MIXED_SUPPORT_AND_METRIC_FAILURE"},
        "support_metric_revision_required": root in {"SUPPORT_CLOSURE_METRIC_MISALIGNED", "MIXED_SUPPORT_AND_METRIC_FAILURE"},
        "observation_revision_required": False,
        "explicit_teacher_decisive_state_only_actor_invisible_variable": None,
        "interpretation": (
            "R2 provides near RELEASE witnesses for the prior I1 error states, while both the R2 neighborhoods and the consumed-Q leave-one-out neighborhoods remain gate-mixed. "
            "The failed purity conjunct therefore measures boundary composition, not just RELEASE support coverage."
            if purity_misaligned and not true_residual else
            "The evidence contains both natural boundary mixing and remaining context/support mismatch; a prospective protocol must address both without changing the current R2 verdict."
            if root == "MIXED_SUPPORT_AND_METRIC_FAILURE" else
            "Prior failed states still lack sufficient near same-family/same-proposal RELEASE support under the fixed diagnostic radii."
            if root == "TRUE_RESIDUAL_RELEASE_SUPPORT_GAP" else
            "The frozen diagnostics do not support a unique attribution."
        ),
        "V4_2_V2_1_I0_R2": "FAIL",
        "V4_2_V2_1_I1_V2_PROTOCOL_PREREGISTRATION_ALLOWED": False,
        "V4_2_V2_1_I2_DAGGER_PREFLIGHT_ALLOWED": False,
        "PPO_ALLOWED": False,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    intrinsic_path = output / "v4_2_v2_1_i0_r2_q_intrinsic_knn.json"
    failed_path = output / "v4_2_v2_1_i0_r2_failed_state_coverage.json"
    tv_path = output / "v4_2_v2_1_i0_r2_proposal14_tv_decomposition.json"
    summary_path = output / "v4_2_v2_1_i0_r2_support_closure_attribution.md"
    intrinsic_artifact = {"stage": "I0_R2_SUPPORT_CLOSURE_ATTRIBUTION", "evidence_type": "newly_run",
                          "input_hashes": hashes, "protocol": {"k": K, "radii": RADII, "distance": "normalized_88D_Euclidean"},
                          "R2_to_Q": {"records": r2_records, "summary": r2_summary},
                          "old_I0_R_to_Q_summary": old_summary, "Q_leave_one_out": {"records": q_records, "summary": q_summary},
                          "class_prior_sensitivity": priors, "old_to_R2_improvement": improvement}
    failed_artifact = {"stage": "I0_R2_FAILED_STATE_COVERAGE", "evidence_type": "newly_run",
                       "input_hashes": hashes, **failed}
    tv_artifact = {"stage": "I0_R2_PROPOSAL14_TV_DECOMPOSITION", "evidence_type": "newly_run",
                   "input_hashes": hashes, **tv}
    _write(intrinsic_path, intrinsic_artifact)
    _write(failed_path, failed_artifact)
    _write(tv_path, tv_artifact)
    result = {"stage": "I0_R2_SUPPORT_CLOSURE_ATTRIBUTION", "evidence_type": "newly_run",
              "input_hashes": hashes, "summary": summary, "q_intrinsic": intrinsic_artifact["Q_leave_one_out"],
              "R2_to_Q_summary": r2_summary, "old_to_R2_improvement": improvement,
              "failed_state_coverage": {k: {x: v for x, v in value.items() if x != "rows"} for k, value in failed.items()},
              "proposal14_tv": tv, "class_prior_sensitivity": priors,
              "artifacts": {"q_intrinsic": str(intrinsic_path), "failed_state_coverage": str(failed_path),
                            "proposal14_tv": str(tv_path)}}
    summary_path.write_text(_render(result), encoding="utf-8")
    result["artifacts"]["summary"] = str(summary_path)
    result["artifact_sha256"] = {name: _sha(Path(path)) for name, path in result["artifacts"].items()}
    print(json.dumps({"summary": summary, "artifact_sha256": result["artifact_sha256"]}, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r2", required=True)
    parser.add_argument("--old", required=True)
    parser.add_argument("--q", required=True)
    parser.add_argument("--consensus", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
