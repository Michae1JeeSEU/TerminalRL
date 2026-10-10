from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


NEW_DIR = Path(
    r"D:\7_codespace\training_outputs\terminal\p3_neighbor_blocker_ranking_20261010\nine_cycle"
)
OLD_DIR = Path(
    r"D:\7_codespace\training_outputs\terminal\p3_neighbor_prefix_ring_reuse_20261010\nine_cycle"
)
OUTPUT = Path(r"D:\7_codespace\terminal\.tmp\p3_blocker_ranking_comparison_20261010.json")
EXPECTED_SEEDS = [893511, 893512, 893513, 893601, 893602, 893603, 893604, 893605, 893606]
DENSE_SEEDS = [893512, 893513, 893602, 893603, 893605, 893606]
FIXED_UPDATE = 5000

GATE_RATE_KEYS = (
    "WAIT_recall",
    "WAIT_false_release",
    "RELEASE_recall",
    "gate_accuracy",
    "balanced_accuracy",
    "proposal_top1",
    "proposal_top3",
    "joint_accuracy",
)
PAIR_RATE_KEYS = (
    "blocked_recall",
    "safe_recall",
    "blocker_top1",
    "pair_accuracy",
)
ALL_RATE_KEYS = GATE_RATE_KEYS + PAIR_RATE_KEYS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_meta(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256(path), "exists": path.exists()}


def rate_value(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def recover_count(
    rate: Any,
    denominator: int,
    label: str,
    concerns: list[dict[str, Any]],
    tolerance: float = 1e-4,
) -> tuple[int | None, float | None]:
    """Recover integer counts from stored float means and record rounding facts."""
    if denominator == 0:
        if rate is not None:
            concerns.append(
                {"label": label, "kind": "non_null_zero_denominator", "reported_rate": rate}
            )
        return None, None
    if rate is None:
        concerns.append({"label": label, "kind": "null_nonzero_denominator", "denominator": denominator})
        return None, None
    product = float(rate) * denominator
    rounded = int(round(product))
    residual = product - rounded
    if abs(residual) > tolerance:
        concerns.append(
            {
                "label": label,
                "kind": "non_integer_reconstruction",
                "reported_rate": float(rate),
                "denominator": denominator,
                "product": product,
                "rounded_numerator": rounded,
                "residual": residual,
            }
        )
    return rounded, residual


def parse_gate(group: dict[str, Any], label: str, concerns: list[dict[str, Any]]) -> dict[str, Any]:
    overall = group["overall"]
    gate = overall["gate"]
    confusion = [[int(value) for value in row] for row in gate["confusion"]]
    if len(confusion) != 2 or any(len(row) != 2 for row in confusion):
        raise ValueError(f"{label}: gate confusion is not 2x2")
    rows = int(overall["rows"])
    wait_support = sum(confusion[0])
    release_support = sum(confusion[1])
    if wait_support + release_support != rows:
        raise ValueError(f"{label}: confusion denominator {wait_support + release_support} != rows {rows}")
    wait_hits = confusion[0][0]
    wait_false = confusion[0][1]
    release_hits = confusion[1][1]
    gate_hits = confusion[0][0] + confusion[1][1]
    proposal = overall["proposal"]
    proposal_rows = int(proposal["rows"]) if proposal is not None else 0
    proposal_top1_hits, _ = recover_count(
        None if proposal is None else proposal["top1"],
        proposal_rows,
        f"{label}.proposal_top1",
        concerns,
    )
    proposal_top3_hits, _ = recover_count(
        None if proposal is None else proposal["top3"],
        proposal_rows,
        f"{label}.proposal_top3",
        concerns,
    )
    joint_hits, _ = recover_count(
        overall.get("joint_accuracy"),
        rows,
        f"{label}.joint_accuracy",
        concerns,
    )
    return {
        "rows": rows,
        "wait_support": wait_support,
        "release_support": release_support,
        "proposal_rows": proposal_rows,
        "confusion": confusion,
        "wait_hits": wait_hits,
        "wait_false_hits": wait_false,
        "release_hits": release_hits,
        "gate_hits": gate_hits,
        "proposal_top1_hits": proposal_top1_hits,
        "proposal_top3_hits": proposal_top3_hits,
        "joint_hits": joint_hits,
        "WAIT_recall": rate_value(wait_hits, wait_support),
        "WAIT_false_release": rate_value(wait_false, wait_support),
        "RELEASE_recall": rate_value(release_hits, release_support),
        "gate_accuracy": rate_value(gate_hits, rows),
        "balanced_accuracy": (
            (rate_value(wait_hits, wait_support) + rate_value(release_hits, release_support)) / 2
            if wait_support and release_support
            else None
        ),
        "proposal_top1": rate_value(proposal_top1_hits, proposal_rows),
        "proposal_top3": rate_value(proposal_top3_hits, proposal_rows),
        "joint_accuracy": rate_value(joint_hits, rows),
        "stored": {
            "WAIT_recall": gate.get("WAIT_recall"),
            "RELEASE_recall": gate.get("RELEASE_recall"),
            "gate_accuracy": gate.get("accuracy"),
            "balanced_accuracy": gate.get("balanced_accuracy"),
            "proposal_top1": None if proposal is None else proposal.get("top1"),
            "proposal_top3": None if proposal is None else proposal.get("top3"),
            "joint_accuracy": overall.get("joint_accuracy"),
        },
    }


def parse_pairs(group: dict[str, Any], label: str, concerns: list[dict[str, Any]]) -> dict[str, Any]:
    queries = int(group["queries"])
    known_pairs = int(group["known_pairs"])
    blocked_pairs = int(group["blocked_pairs"])
    safe_pairs = known_pairs - blocked_pairs
    blocked_queries = int(group["blocked_queries"])
    if safe_pairs < 0:
        raise ValueError(f"{label}: blocked_pairs exceeds known_pairs")
    if blocked_queries < 0 or blocked_queries > queries:
        raise ValueError(f"{label}: blocked_queries outside query support")
    blocked_hits, _ = recover_count(
        group.get("blocked_recall"),
        blocked_pairs,
        f"{label}.blocked_recall",
        concerns,
    )
    safe_hits, _ = recover_count(
        group.get("safe_recall"),
        safe_pairs,
        f"{label}.safe_recall",
        concerns,
    )
    blocker_hits, _ = recover_count(
        group.get("blocker_top1"),
        blocked_queries,
        f"{label}.blocker_top1",
        concerns,
    )
    pair_hits = None if blocked_hits is None or safe_hits is None else blocked_hits + safe_hits
    auxiliary_loss = float(group["auxiliary_loss"])
    if not math.isfinite(auxiliary_loss):
        concerns.append({"label": f"{label}.auxiliary_loss", "kind": "nonfinite", "value": auxiliary_loss})
    ranking_loss = group.get("blocker_ranking_loss")
    if ranking_loss is not None:
        ranking_loss = float(ranking_loss)
        if not math.isfinite(ranking_loss):
            concerns.append({"label": f"{label}.blocker_ranking_loss", "kind": "nonfinite", "value": ranking_loss})
    return {
        "queries": queries,
        "known_pairs": known_pairs,
        "blocked_pairs": blocked_pairs,
        "safe_pairs": safe_pairs,
        "blocked_queries": blocked_queries,
        "blocked_hits": blocked_hits,
        "safe_hits": safe_hits,
        "blocker_hits": blocker_hits,
        "pair_hits": pair_hits,
        "blocked_recall": rate_value(blocked_hits, blocked_pairs),
        "safe_recall": rate_value(safe_hits, safe_pairs),
        "blocker_top1": rate_value(blocker_hits, blocked_queries),
        "pair_accuracy": rate_value(pair_hits, known_pairs),
        "auxiliary_loss": auxiliary_loss,
        "blocker_ranking_loss": ranking_loss,
        "blocker_ranking_queries": group.get("blocker_ranking_queries"),
        "stored": {
            "blocked_recall": group.get("blocked_recall"),
            "safe_recall": group.get("safe_recall"),
            "blocker_top1": group.get("blocker_top1"),
            "auxiliary_loss": group.get("auxiliary_loss"),
            "blocker_ranking_loss": ranking_loss,
        },
    }


def fixed_curve(fold: dict[str, Any], label: str) -> dict[str, Any]:
    matches = [row for row in fold.get("curve", []) if int(row["update"]) == FIXED_UPDATE]
    if len(matches) != 1:
        raise ValueError(f"{label}: expected one update={FIXED_UPDATE}, found {len(matches)}")
    return matches[0]


def parse_fold(
    fold: dict[str, Any],
    source: str,
    fold_index: int,
    concerns: list[dict[str, Any]],
    require_reference: bool,
) -> dict[str, Any]:
    seed = int(fold["held_seed"])
    curve = fixed_curve(fold, f"{source}.seed{seed}")
    train_gate = parse_gate(curve["train"], f"{source}.seed{seed}.train", concerns)
    held_gate = parse_gate(curve["validation"], f"{source}.seed{seed}.validation", concerns)
    train_pairs = parse_pairs(curve["train_pairs"], f"{source}.seed{seed}.train_pairs", concerns)
    held_pairs = parse_pairs(curve["validation_pairs"], f"{source}.seed{seed}.validation_pairs", concerns)
    reference_pairs = None
    if require_reference:
        if "validation_reference_pairs" not in curve:
            raise ValueError(f"{source}.seed{seed}: missing validation_reference_pairs")
        reference_pairs = parse_pairs(
            curve["validation_reference_pairs"],
            f"{source}.seed{seed}.validation_reference_pairs",
            concerns,
        )
    return {
        "held_seed": seed,
        "arm": fold["arm"],
        "fold_index": fold_index,
        "train_rows": int(fold["train_rows"]),
        "held_rows": int(fold["held_rows"]),
        "train_auxiliary_queries": int(fold["train_auxiliary_queries"]),
        "held_auxiliary_queries": int(fold["held_auxiliary_queries"]),
        "elapsed_seconds": float(fold["elapsed_seconds"]),
        "train_gate": train_gate,
        "held_gate": held_gate,
        "train_pairs": train_pairs,
        "held_pairs": held_pairs,
        "reference_pairs": reference_pairs,
    }


def compact_gate(gate: dict[str, Any]) -> dict[str, Any]:
    return {
        "supports": {
            "rows": gate["rows"],
            "WAIT": gate["wait_support"],
            "RELEASE": gate["release_support"],
            "proposal": gate["proposal_rows"],
        },
        "numerators": {
            "WAIT_recall": gate["wait_hits"],
            "WAIT_false_release": gate["wait_false_hits"],
            "RELEASE_recall": gate["release_hits"],
            "gate_accuracy": gate["gate_hits"],
            "proposal_top1": gate["proposal_top1_hits"],
            "proposal_top3": gate["proposal_top3_hits"],
            "joint_accuracy": gate["joint_hits"],
        },
        "rates": {key: gate[key] for key in GATE_RATE_KEYS},
        "confusion": gate["confusion"],
    }


def compact_pairs(pairs: dict[str, Any]) -> dict[str, Any]:
    return {
        "supports": {
            "queries": pairs["queries"],
            "known_pairs": pairs["known_pairs"],
            "blocked_pairs": pairs["blocked_pairs"],
            "safe_pairs": pairs["safe_pairs"],
            "blocked_queries_scene_support": pairs["blocked_queries"],
        },
        "numerators": {
            "blocked_recall": pairs["blocked_hits"],
            "safe_recall": pairs["safe_hits"],
            "blocker_top1": pairs["blocker_hits"],
            "pair_accuracy": pairs["pair_hits"],
        },
        "rates": {key: pairs[key] for key in PAIR_RATE_KEYS},
        "auxiliary_loss": pairs["auxiliary_loss"],
        "blocker_ranking_loss": pairs["blocker_ranking_loss"],
        "blocker_ranking_queries": pairs["blocker_ranking_queries"],
    }


def split_metric_records(records: list[dict[str, Any]], gate_key: str, pair_key: str) -> list[dict[str, Any]]:
    return [
        {"held_seed": record["held_seed"], "gate": record[gate_key], "pair": record[pair_key]}
        for record in records
    ]


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    gate_keys = ["rows", "wait_support", "release_support", "proposal_rows"]
    gate_sum = {key: 0 for key in gate_keys}
    gate_num_keys = ["wait_hits", "wait_false_hits", "release_hits", "gate_hits", "proposal_top1_hits", "proposal_top3_hits", "joint_hits"]
    for key in gate_num_keys:
        gate_sum[key] = 0
    pair_keys = ["queries", "known_pairs", "blocked_pairs", "safe_pairs", "blocked_queries"]
    pair_sum = {key: 0 for key in pair_keys}
    pair_num_keys = ["blocked_hits", "safe_hits", "blocker_hits", "pair_hits"]
    for key in pair_num_keys:
        pair_sum[key] = 0
    losses = []
    for record in records:
        gate = record["gate"]
        pairs = record["pair"]
        for key in gate_keys + gate_num_keys:
            gate_sum[key] += gate[key]
        for key in pair_keys:
            pair_sum[key] += pairs[key]
        for key in pair_num_keys:
            if pairs[key] is None:
                pair_sum[key] = None
            elif pair_sum[key] is not None:
                pair_sum[key] += pairs[key]
        losses.append(float(pairs["auxiliary_loss"]))

    def micro_rate(numerator_key: str, denominator_key: str) -> dict[str, Any]:
        numerator = gate_sum.get(numerator_key, pair_sum.get(numerator_key))
        denominator = gate_sum.get(denominator_key, pair_sum.get(denominator_key))
        if numerator is None or denominator is None:
            return {"value": None, "numerator": numerator, "denominator": denominator}
        return {
            "value": rate_value(numerator, denominator),
            "numerator": numerator,
            "denominator": denominator,
        }

    micro = {
        "WAIT_recall": micro_rate("wait_hits", "wait_support"),
        "WAIT_false_release": micro_rate("wait_false_hits", "wait_support"),
        "RELEASE_recall": micro_rate("release_hits", "release_support"),
        "gate_accuracy": micro_rate("gate_hits", "rows"),
        "proposal_top1": micro_rate("proposal_top1_hits", "proposal_rows"),
        "proposal_top3": micro_rate("proposal_top3_hits", "proposal_rows"),
        "joint_accuracy": micro_rate("joint_hits", "rows"),
        "blocked_recall": {
            "value": rate_value(pair_sum["blocked_hits"], pair_sum["blocked_pairs"]),
            "numerator": pair_sum["blocked_hits"],
            "denominator": pair_sum["blocked_pairs"],
        },
        "safe_recall": {
            "value": rate_value(pair_sum["safe_hits"], pair_sum["safe_pairs"]),
            "numerator": pair_sum["safe_hits"],
            "denominator": pair_sum["safe_pairs"],
        },
        "blocker_top1": {
            "value": rate_value(pair_sum["blocker_hits"], pair_sum["blocked_queries"]),
            "numerator": pair_sum["blocker_hits"],
            "denominator": pair_sum["blocked_queries"],
        },
        "pair_accuracy": {
            "value": rate_value(pair_sum["pair_hits"], pair_sum["known_pairs"]),
            "numerator": pair_sum["pair_hits"],
            "denominator": pair_sum["known_pairs"],
        },
    }
    if micro["WAIT_recall"]["value"] is not None and micro["RELEASE_recall"]["value"] is not None:
        micro["balanced_accuracy"] = {
            "value": (micro["WAIT_recall"]["value"] + micro["RELEASE_recall"]["value"]) / 2,
            "numerator": {
                "WAIT_hits": gate_sum["wait_hits"],
                "RELEASE_hits": gate_sum["release_hits"],
            },
            "denominator": {
                "WAIT": gate_sum["wait_support"],
                "RELEASE": gate_sum["release_support"],
            },
        }
    else:
        micro["balanced_accuracy"] = {"value": None, "numerator": None, "denominator": None}
    micro["auxiliary_loss"] = {
        "value": sum(losses) / len(losses) if losses else None,
        "count_folds": len(losses),
        "denominator": "fold_count (unweighted mean; pair loss is not a rate)",
    }

    macro: dict[str, Any] = {}
    for key in GATE_RATE_KEYS + PAIR_RATE_KEYS:
        values = []
        null_seeds = []
        for record in records:
            source = record["gate"] if key in GATE_RATE_KEYS else record["pair"]
            value = source[key]
            if value is None:
                null_seeds.append(record["held_seed"])
            else:
                values.append(value)
        macro[key] = {
            "value": sum(values) / len(values) if values else None,
            "count_folds": len(values),
            "null_denominator_seeds": null_seeds,
        }
    # Balanced accuracy uses per-fold balanced values only when both class supports exist.
    balanced_values = [record["gate"]["balanced_accuracy"] for record in records if record["gate"]["balanced_accuracy"] is not None]
    macro["balanced_accuracy"] = {
        "value": sum(balanced_values) / len(balanced_values) if balanced_values else None,
        "count_folds": len(balanced_values),
        "null_denominator_seeds": [
            record["held_seed"] for record in records if record["gate"]["balanced_accuracy"] is None
        ],
    }
    loss_values = [record["pair"]["auxiliary_loss"] for record in records]
    macro["auxiliary_loss"] = {
        "value": sum(loss_values) / len(loss_values) if loss_values else None,
        "count_folds": len(loss_values),
        "denominator": "fold_count (unweighted mean; pair loss is not a rate)",
    }
    return {
        "fold_count": len(records),
        "supports": {
            **gate_sum,
            **pair_sum,
        },
        "micro": micro,
        "macro": macro,
    }


def metric_value(summary: dict[str, Any], key: str) -> float | None:
    if key == "auxiliary_loss":
        return summary["macro"][key]["value"]
    return summary["micro"][key]["value"]


def delta_summary(new: dict[str, Any], old: dict[str, Any], keys: tuple[str, ...] = ALL_RATE_KEYS) -> dict[str, Any]:
    return {
        key: {
            "delta_pp": None
            if metric_value(new, key) is None or metric_value(old, key) is None
            else 100 * (metric_value(new, key) - metric_value(old, key)),
            "new": metric_value(new, key),
            "old": metric_value(old, key),
        }
        for key in keys
    }


def macro_delta(new: dict[str, Any], old: dict[str, Any], keys: tuple[str, ...] = ALL_RATE_KEYS) -> dict[str, Any]:
    out = {}
    for key in keys:
        n = new["macro"][key]
        o = old["macro"][key]
        out[key] = {
            "delta_pp": None if n["value"] is None or o["value"] is None else 100 * (n["value"] - o["value"]),
            "new": n,
            "old": o,
        }
    return out


def compact_split(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "held_seed": record["held_seed"],
        "gate": compact_gate(record["gate"]),
        "pair": compact_pairs(record["pair"]),
    }


def sign_counts(rows: list[dict[str, Any]], delta_field: str) -> dict[str, dict[str, dict[str, int]]]:
    result: dict[str, dict[str, dict[str, int]]] = {}
    for group, keys in (("gate", GATE_RATE_KEYS), ("pair", PAIR_RATE_KEYS)):
        result[group] = {}
        for key in keys:
            counts = {"+": 0, "-": 0, "0": 0, "null": 0}
            for row in rows:
                value = row[delta_field][group][key]
                if value is None:
                    counts["null"] += 1
                elif value > 0:
                    counts["+"] += 1
                elif value < 0:
                    counts["-"] += 1
                else:
                    counts["0"] += 1
            result[group][key] = counts
    return result


def split_support(record: dict[str, Any], pair_key: str = "reference_pairs") -> dict[str, int]:
    gate = record["held_gate"]
    pairs = record[pair_key]
    return {
        "rows": gate["rows"],
        "WAIT": gate["wait_support"],
        "RELEASE": gate["release_support"],
        "proposal": gate["proposal_rows"],
        "queries": pairs["queries"],
        "known_pairs": pairs["known_pairs"],
        "blocked_pairs": pairs["blocked_pairs"],
        "blocked_queries_scene_support": pairs["blocked_queries"],
    }


def validate_root(
    directory: Path,
    result: dict[str, Any],
    expected_seeds: list[int],
    require_reference: bool,
    errors: list[str],
    require_progress: bool,
) -> list[dict[str, Any]]:
    if result.get("status") != "COMPLETE":
        errors.append(f"{directory}: result status={result.get('status')!r}")
    if result.get("error") is not None:
        errors.append(f"{directory}: result error={result.get('error')!r}")
    folds = result.get("folds", [])
    if len(folds) != len(expected_seeds):
        errors.append(f"{directory}: fold count={len(folds)} expected={len(expected_seeds)}")
    actual = sorted(int(fold["held_seed"]) for fold in folds)
    if actual != sorted(expected_seeds):
        errors.append(f"{directory}: held seeds={actual} expected={sorted(expected_seeds)}")
    progress_statuses = []
    for seed in expected_seeds:
        progress = directory / f"fold_{seed}_native_pair_auxiliary.progress.json"
        if not progress.exists():
            if require_progress:
                errors.append(f"{directory}: missing progress {progress.name}")
            progress_statuses.append({"seed": seed, "status": "MISSING", "error": None})
            continue
        progress_data = read_json(progress)
        progress_statuses.append({"seed": seed, "status": progress_data.get("status"), "error": progress_data.get("error")})
        if progress_data.get("status") != "COMPLETE":
            errors.append(f"{directory}: seed {seed} progress status={progress_data.get('status')!r}")
        if progress_data.get("error") is not None:
            errors.append(f"{directory}: seed {seed} progress error={progress_data.get('error')!r}")
        marks = [int(row["update"]) for row in progress_data.get("curve", [])]
        if marks != [0, 100, 500, 1000, 2000, 5000]:
            errors.append(f"{directory}: seed {seed} marks={marks}")
    sibling = directory.parent
    exit_path = sibling / "nine_cycle.exit"
    log_path = sibling / "nine_cycle.log"
    if not exit_path.exists():
        errors.append(f"{directory}: missing sibling nine_cycle.exit")
    elif exit_path.read_text(encoding="utf-8").strip() != "0":
        errors.append(f"{directory}: sibling exit={exit_path.read_text(encoding='utf-8').strip()!r}")
    if not log_path.exists():
        errors.append(f"{directory}: missing sibling nine_cycle.log")
    else:
        log = read_json(log_path)
        if log.get("status") != "COMPLETE" or log.get("error") is not None:
            errors.append(f"{directory}: sibling log={log!r}")
    return progress_statuses


def main() -> None:
    errors: list[str] = []
    concerns: list[dict[str, Any]] = []
    new_protocol_path = NEW_DIR / "protocol.json"
    old_protocol_path = OLD_DIR / "protocol.json"
    new_result_path = NEW_DIR / "result.json"
    old_result_path = OLD_DIR / "result.json"
    new_protocol = read_json(new_protocol_path)
    old_protocol = read_json(old_protocol_path)
    new_result = read_json(new_result_path)
    old_result = read_json(old_result_path)
    new_progress_statuses = validate_root(NEW_DIR, new_result, EXPECTED_SEEDS, True, errors, True)
    old_progress_statuses = validate_root(OLD_DIR, old_result, EXPECTED_SEEDS, True, errors, True)
    if new_protocol.get("Q_accessed") or old_protocol.get("Q_accessed"):
        errors.append("protocol Q_accessed is not false")
    if new_protocol.get("environment_steps") != 0 or old_protocol.get("environment_steps") != 0:
        errors.append("protocol environment_steps is not zero")
    if new_protocol.get("checkpoint_written") or old_protocol.get("checkpoint_written"):
        errors.append("protocol checkpoint_written is not false")
    if new_protocol.get("loss_variant") != "native_pair_auxiliary_plus_blocker_ranking":
        errors.append("new protocol loss_variant is not blocker ranking")
    if float(new_protocol.get("blocker_ranking_weight", float("nan"))) != 0.25:
        errors.append("new protocol blocker_ranking_weight is not 0.25")
    if old_protocol.get("loss_variant") is not None or old_protocol.get("blocker_ranking_weight") is not None:
        errors.append("control protocol unexpectedly records blocker ranking")

    new_records = [
        parse_fold(fold, "new", index, concerns, True)
        for index, fold in enumerate(new_result["folds"])
    ]
    old_records = [
        parse_fold(fold, "control", index, concerns, True)
        for index, fold in enumerate(old_result["folds"])
    ]
    new_by_seed = {record["held_seed"]: record for record in new_records}
    old_by_seed = {record["held_seed"]: record for record in old_records}
    if set(new_by_seed) != set(EXPECTED_SEEDS) or set(old_by_seed) != set(EXPECTED_SEEDS):
        errors.append("new/control seed sets are not identical expected nine seeds")
    if any(record["arm"] != "native_pair_auxiliary" for record in new_records + old_records):
        errors.append("a result contains a non-native arm")

    support_checks = []
    for seed in EXPECTED_SEEDS:
        if seed not in new_by_seed or seed not in old_by_seed:
            continue
        new_record = new_by_seed[seed]
        old_record = old_by_seed[seed]
        new_reference = split_support(new_record, "reference_pairs")
        old_reference = split_support(old_record, "reference_pairs")
        new_expanded = split_support(new_record, "held_pairs")
        old_expanded = split_support(old_record, "held_pairs")
        reference_equal = {key: new_reference[key] == old_reference[key] for key in new_reference}
        expanded_equal = {key: new_expanded[key] == old_expanded[key] for key in new_expanded}
        if not all(reference_equal.values()):
            errors.append(
                f"seed {seed}: reference supports differ: new={new_reference} control={old_reference}"
            )
        if not all(expanded_equal.values()):
            errors.append(
                f"seed {seed}: expanded supports differ: new={new_expanded} control={old_expanded}"
            )
        support_checks.append({
            "held_seed": seed,
            "reference": {"blocker_ranking": new_reference, "ring_reuse": old_reference, "equal": reference_equal},
            "expanded": {"blocker_ranking": new_expanded, "ring_reuse": old_expanded, "equal": expanded_equal},
        })

    paired_new_reference = split_metric_records(new_records, "held_gate", "reference_pairs")
    paired_old_reference = split_metric_records(old_records, "held_gate", "reference_pairs")
    paired_new_expanded = split_metric_records(new_records, "held_gate", "held_pairs")
    paired_old_expanded = split_metric_records(old_records, "held_gate", "held_pairs")
    all_new_reference = aggregate(paired_new_reference)
    all_old_reference = aggregate(paired_old_reference)
    dense_new_reference = aggregate([record for record in paired_new_reference if record["held_seed"] in DENSE_SEEDS])
    dense_old_reference = aggregate([record for record in paired_old_reference if record["held_seed"] in DENSE_SEEDS])
    all_new_expanded = aggregate(paired_new_expanded)
    all_old_expanded = aggregate(paired_old_expanded)
    dense_new_expanded = aggregate([record for record in paired_new_expanded if record["held_seed"] in DENSE_SEEDS])
    dense_old_expanded = aggregate([record for record in paired_old_expanded if record["held_seed"] in DENSE_SEEDS])

    per_seed = []
    for seed in EXPECTED_SEEDS:
        nr = new_by_seed[seed]
        orow = old_by_seed[seed]
        new_gate = nr["held_gate"]
        old_gate = orow["held_gate"]
        new_reference_pair = nr["reference_pairs"]
        old_reference_pair = orow["reference_pairs"]
        new_expanded_pair = nr["held_pairs"]
        old_expanded_pair = orow["held_pairs"]
        gate_delta = {}
        for key in GATE_RATE_KEYS:
            nv = new_gate[key]
            ov = old_gate[key]
            gate_delta[key] = None if nv is None or ov is None else 100 * (nv - ov)
        reference_pair_delta = {}
        expanded_pair_delta = {}
        for key in PAIR_RATE_KEYS:
            nv = new_reference_pair[key]
            ov = old_reference_pair[key]
            reference_pair_delta[key] = None if nv is None or ov is None else 100 * (nv - ov)
            nv = new_expanded_pair[key]
            ov = old_expanded_pair[key]
            expanded_pair_delta[key] = None if nv is None or ov is None else 100 * (nv - ov)
        per_seed.append(
            {
                "held_seed": seed,
                "blocker_ranking_validation_reference": {"gate": compact_gate(new_gate), "pair": compact_pairs(new_reference_pair)},
                "ring_reuse_validation_reference": {"gate": compact_gate(old_gate), "pair": compact_pairs(old_reference_pair)},
                "delta_reference_new_minus_control_pp": {"gate": gate_delta, "pair": reference_pair_delta},
                "blocker_ranking_expanded_validation": {"gate": compact_gate(new_gate), "pair": compact_pairs(new_expanded_pair)},
                "ring_reuse_expanded_validation": {"gate": compact_gate(old_gate), "pair": compact_pairs(old_expanded_pair)},
                "delta_expanded_new_minus_control_pp": {"gate": gate_delta, "pair": expanded_pair_delta},
                "dense_subset": seed in DENSE_SEEDS,
            }
        )

    dense_tradeoffs = []
    for row in per_seed:
        if not row["dense_subset"]:
            continue
        dense_tradeoffs.append(
            {
                "held_seed": row["held_seed"],
                "delta_pp": row["delta_reference_new_minus_control_pp"],
                "signs": {
                    group: {
                        key: None if value is None else ("+" if value > 0 else "-" if value < 0 else "0")
                        for key, value in values.items()
                    }
                    for group, values in row["delta_reference_new_minus_control_pp"].items()
                },
            }
        )

    dense_expanded_tradeoffs = []
    for row in per_seed:
        if not row["dense_subset"]:
            continue
        dense_expanded_tradeoffs.append(
            {
                "held_seed": row["held_seed"],
                "delta_pp": row["delta_expanded_new_minus_control_pp"],
                "signs": {
                    group: {
                        key: None if value is None else ("+" if value > 0 else "-" if value < 0 else "0")
                        for key, value in values.items()
                    }
                    for group, values in row["delta_expanded_new_minus_control_pp"].items()
                },
            }
        )

    train_held = {
        "blocker_ranking": {
            "train_actor": aggregate(split_metric_records(new_records, "train_gate", "train_pairs")),
            "held_actor_reference": aggregate(split_metric_records(new_records, "held_gate", "reference_pairs")),
            "held_actor_expanded": aggregate(split_metric_records(new_records, "held_gate", "held_pairs")),
            "train_pair": aggregate(split_metric_records(new_records, "train_gate", "train_pairs")),
            "held_pair_reference": aggregate(split_metric_records(new_records, "held_gate", "reference_pairs")),
            "held_pair_expanded": all_new_expanded,
        },
        "ring_reuse": {
            "train_actor": aggregate(split_metric_records(old_records, "train_gate", "train_pairs")),
            "held_actor_reference": aggregate(split_metric_records(old_records, "held_gate", "reference_pairs")),
            "held_actor_expanded": aggregate(split_metric_records(old_records, "held_gate", "held_pairs")),
            "train_pair": aggregate(split_metric_records(old_records, "train_gate", "train_pairs")),
            "held_pair_reference": aggregate(split_metric_records(old_records, "held_gate", "reference_pairs")),
            "held_pair_expanded": all_old_expanded,
        },
    }

    new_exit = NEW_DIR.parent / "nine_cycle.exit"
    new_log = NEW_DIR.parent / "nine_cycle.log"
    old_exit = OLD_DIR.parent / "nine_cycle.exit"
    old_log = OLD_DIR.parent / "nine_cycle.log"
    macro_wait_exclusion = {
        "reference": {
            "blocker_ranking": {
                "WAIT_recall": all_new_reference["macro"]["WAIT_recall"],
                "WAIT_false_release": all_new_reference["macro"]["WAIT_false_release"],
            },
            "ring_reuse": {
                "WAIT_recall": all_old_reference["macro"]["WAIT_recall"],
                "WAIT_false_release": all_old_reference["macro"]["WAIT_false_release"],
            },
        },
        "expanded": {
            "blocker_ranking": {
                "WAIT_recall": all_new_expanded["macro"]["WAIT_recall"],
                "WAIT_false_release": all_new_expanded["macro"]["WAIT_false_release"],
            },
            "ring_reuse": {
                "WAIT_recall": all_old_expanded["macro"]["WAIT_recall"],
                "WAIT_false_release": all_old_expanded["macro"]["WAIT_false_release"],
            },
        },
        "expected_excluded_seed": 893601,
        "expected_count_folds": 8,
    }
    output = {
        "schema": "p3_blocker_ranking_comparison_20261010_v1",
        "status": "PASS" if not errors else "FAIL",
        "fixed_update": FIXED_UPDATE,
        "metric_formulas": {
            "WAIT_false_release": "confusion[WAIT,RELEASE] / WAIT support = 1 - WAIT recall",
            "RELEASE_recall": "confusion[RELEASE,RELEASE] / RELEASE support",
            "proposal_top1_top3": "conditional RELEASE proposal hits / conditional RELEASE proposal rows",
            "joint_accuracy": "gate-correct and (WAIT or proposal-correct) hits / actor rows",
            "blocked_recall": "pair score >= 0 hits / known blocked pair support",
            "safe_recall": "pair score < 0 hits / known safe pair support",
            "blocker_top1": "top queried known-blocked neighbor hits / scenes with at least one blocked pair",
            "pair_accuracy": "blocked hits + safe hits / known pair support",
            "micro": "sum recovered integer numerators / sum explicit denominators",
            "macro": "unweighted mean of per-fold rates; folds with zero denominator excluded and counted",
        },
        "inputs": {
            "blocker_ranking": {
                "directory": str(NEW_DIR),
                "protocol": file_meta(new_protocol_path),
                "result": file_meta(new_result_path),
                "exit": file_meta(new_exit),
                "log": file_meta(new_log),
                "result_status": new_result.get("status"),
                "result_error": new_result.get("error"),
                "protocol_json": new_protocol,
            },
            "ring_reuse": {
                "directory": str(OLD_DIR),
                "protocol": file_meta(old_protocol_path),
                "result": file_meta(old_result_path),
                "exit": file_meta(old_exit),
                "log": file_meta(old_log),
                "result_status": old_result.get("status"),
                "result_error": old_result.get("error"),
                "protocol_json": old_protocol,
            },
        },
        "checks": {
            "new_progress": new_progress_statuses,
            "control_progress": old_progress_statuses,
            "control_progress_files_available": all(
                item["status"] != "MISSING" for item in old_progress_statuses
            ),
            "exact_seed_set": EXPECTED_SEEDS,
            "all_native_pair_auxiliary": all(record["arm"] == "native_pair_auxiliary" for record in new_records + old_records),
            "all_fixed5000": len(new_records) == 9 and len(old_records) == 9,
            "reference_supports_equal": all(
                all(item["reference"]["equal"].values()) for item in support_checks
            ),
            "expanded_supports_equal": all(
                all(item["expanded"]["equal"].values()) for item in support_checks
            ),
            "held_support_check_count": len(support_checks),
            "Q_accessed_false": not (new_protocol.get("Q_accessed") or old_protocol.get("Q_accessed")),
            "environment_steps_zero": new_protocol.get("environment_steps") == 0 and old_protocol.get("environment_steps") == 0,
            "checkpoint_written_false": not (new_protocol.get("checkpoint_written") or old_protocol.get("checkpoint_written")),
            "all_recovered_metrics_finite": not any(item.get("kind") == "nonfinite" for item in concerns),
            "wait_macro_excludes_893601_count8": all(
                macro_wait_exclusion[group][run][metric]["count_folds"] == 8
                for group in ("reference", "expanded")
                for run in ("blocker_ranking", "ring_reuse")
                for metric in ("WAIT_recall", "WAIT_false_release")
            ),
            "rounding_concern_count": len(concerns),
            "errors": errors,
        },
        "support_checks": support_checks,
        "blocker_ranking_vs_ring_reuse_reference": {
            "per_seed": per_seed,
            "all9": {
                "blocker_ranking_validation_reference_pairs": all_new_reference,
                "ring_reuse_validation_reference_pairs": all_old_reference,
                "delta_new_minus_control": {
                    "micro": delta_summary(all_new_reference, all_old_reference),
                    "macro": macro_delta(all_new_reference, all_old_reference),
                },
            },
            "dense6": {
                "seeds": DENSE_SEEDS,
                "blocker_ranking_validation_reference_pairs": dense_new_reference,
                "ring_reuse_validation_reference_pairs": dense_old_reference,
                "delta_new_minus_control": {
                    "micro": delta_summary(dense_new_reference, dense_old_reference),
                    "macro": macro_delta(dense_new_reference, dense_old_reference),
                },
                "individual_tradeoffs": dense_tradeoffs,
                "sign_counts": sign_counts(
                    [row for row in per_seed if row["dense_subset"]],
                    "delta_reference_new_minus_control_pp",
                ),
            },
        },
        "blocker_ranking_vs_ring_reuse_expanded": {
            "note": "Both runs validation_pairs use the same expanded old+supplemental+prefix support; reported separately from original-held reference pairs.",
            "all9": {
                "blocker_ranking_expanded_validation_pairs": all_new_expanded,
                "ring_reuse_expanded_validation_pairs": all_old_expanded,
                "delta_new_minus_control": {
                    "micro": delta_summary(all_new_expanded, all_old_expanded),
                    "macro": macro_delta(all_new_expanded, all_old_expanded),
                },
            },
            "dense6": {
                "seeds": DENSE_SEEDS,
                "blocker_ranking_expanded_validation_pairs": dense_new_expanded,
                "ring_reuse_expanded_validation_pairs": dense_old_expanded,
                "delta_new_minus_control": {
                    "micro": delta_summary(dense_new_expanded, dense_old_expanded),
                    "macro": macro_delta(dense_new_expanded, dense_old_expanded),
                },
                "individual_tradeoffs": dense_expanded_tradeoffs,
                "sign_counts": sign_counts(
                    [row for row in per_seed if row["dense_subset"]],
                    "delta_expanded_new_minus_control_pp",
                ),
            },
            "per_seed": [
                {
                    "held_seed": record["held_seed"],
                    "blocker_ranking_expanded_validation": record["blocker_ranking_expanded_validation"],
                    "ring_reuse_expanded_validation": record["ring_reuse_expanded_validation"],
                    "delta_expanded_new_minus_control_pp": record["delta_expanded_new_minus_control_pp"],
                }
                for record in per_seed
            ],
        },
        "macro_wait_exclusion": macro_wait_exclusion,
        "train_vs_held_fixed5000": train_held,
        "rounding_and_noninteger_details": concerns,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "output": str(OUTPUT), "errors": errors, "rounding_concerns": len(concerns)}))


if __name__ == "__main__":
    main()
