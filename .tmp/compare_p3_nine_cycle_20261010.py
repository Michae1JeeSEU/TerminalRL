from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean


METRIC_NAMES = (
    "WAIT_recall",
    "WAIT_false_release",
    "RELEASE_recall",
    "gate_accuracy",
    "balanced_accuracy",
    "proposal_top1",
    "proposal_top3",
    "joint_accuracy",
    "blocked_recall",
    "safe_recall",
    "blocker_top1",
    "auxiliary_loss",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_result(directory: Path) -> tuple[dict, dict, dict]:
    protocol_path = directory / "protocol.json"
    result_path = directory / "result.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    return protocol, result, {
        "directory": str(directory),
        "protocol_path": str(protocol_path),
        "result_path": str(result_path),
        "protocol_sha256": sha256(protocol_path),
        "result_sha256": sha256(result_path),
    }


def fixed_curve(fold: dict) -> dict:
    matches = [row for row in fold["curve"] if int(row["update"]) == 5000]
    if len(matches) != 1:
        raise ValueError(
            f"held_seed={fold.get('held_seed')} has {len(matches)} update=5000 curve records"
        )
    return matches[0]


def gate_metrics(group: dict) -> dict:
    overall = group["overall"]
    gate = overall["gate"]
    confusion = [[int(v) for v in row] for row in gate["confusion"]]
    wait_support = confusion[0][0] + confusion[0][1]
    release_support = confusion[1][0] + confusion[1][1]
    rows = int(overall["rows"])
    if wait_support + release_support != rows:
        raise ValueError("gate confusion denominator does not match rows")
    wait_recall = None if gate["WAIT_recall"] is None else float(gate["WAIT_recall"])
    release_recall = None if gate["RELEASE_recall"] is None else float(gate["RELEASE_recall"])
    false_release = None if wait_recall is None else 1.0 - wait_recall
    if wait_support and false_release is not None and abs(false_release - confusion[0][1] / wait_support) > 1e-6:
        raise ValueError("WAIT_false_release is not 1-WAIT_recall")
    proposal = overall["proposal"]
    return {
        "rows": rows,
        "wait_support": wait_support,
        "release_support": release_support,
        "confusion": confusion,
        "WAIT_recall": wait_recall,
        "WAIT_false_release": false_release,
        "RELEASE_recall": release_recall,
        "gate_accuracy": None if gate["accuracy"] is None else float(gate["accuracy"]),
        "balanced_accuracy": None
        if gate["balanced_accuracy"] is None
        else float(gate["balanced_accuracy"]),
        "proposal_rows": int(proposal["rows"]),
        "proposal_top1": None if proposal["top1"] is None else float(proposal["top1"]),
        "proposal_top3": None if proposal["top3"] is None else float(proposal["top3"]),
        "joint_accuracy": None
        if overall["joint_accuracy"] is None
        else float(overall["joint_accuracy"]),
    }


def pair_metrics(group: dict) -> dict:
    pairs = group
    known = int(pairs["known_pairs"])
    blocked = int(pairs["blocked_pairs"])
    safe = known - blocked
    return {
        "queries": int(pairs["queries"]),
        "known_pairs": known,
        "blocked_pairs": blocked,
        "safe_pairs": safe,
        "blocked_queries": int(pairs["blocked_queries"]),
        "blocked_recall": None if blocked == 0 else float(pairs["blocked_recall"]),
        "safe_recall": None if safe == 0 else float(pairs["safe_recall"]),
        "blocker_top1": None
        if int(pairs["blocked_queries"]) == 0
        else float(pairs["blocker_top1"]),
        "auxiliary_loss": float(pairs["auxiliary_loss"]),
    }


def metrics_at_5000(fold: dict) -> dict:
    curve = fixed_curve(fold)
    return {
        "train": {
            "gate": gate_metrics(curve["train"]),
            "pair": pair_metrics(curve["train_pairs"]),
        },
        "validation": {
            "gate": gate_metrics(curve["validation"]),
            "pair": pair_metrics(curve["validation_pairs"]),
        },
    }


def flatten(metrics: dict) -> dict:
    gate = metrics["gate"]
    pair = metrics["pair"]
    return {name: (gate.get(name) if name in gate else pair.get(name)) for name in METRIC_NAMES}


def compare_metrics(new: dict, old: dict) -> dict:
    n = flatten(new["validation"])
    o = flatten(old["validation"])
    return {name: None if n[name] is None or o[name] is None else n[name] - o[name] for name in METRIC_NAMES}


def fold_record(fold: dict, source_meta: dict, index: int) -> dict:
    metrics = metrics_at_5000(fold)
    return {
        "held_seed": int(fold["held_seed"]),
        "arm": fold["arm"],
        "fold_index": index,
        "train_rows": int(fold["train_rows"]),
        "held_rows": int(fold["held_rows"]),
        "train_auxiliary_queries": int(fold["train_auxiliary_queries"]),
        "held_auxiliary_queries": int(fold["held_auxiliary_queries"]),
        "elapsed_seconds": float(fold["elapsed_seconds"]),
        "source": source_meta,
        "metrics": metrics,
    }


def average_metric(records: list[dict], split: str, metric: str) -> float | None:
    values = []
    for record in records:
        value = flatten(record["metrics"][split]).get(metric)
        if value is not None:
            values.append(value)
    return None if not values else mean(values)


def denominator(record: dict, split: str, metric: str) -> int | None:
    gate = record["metrics"][split]["gate"]
    pair = record["metrics"][split]["pair"]
    return {
        "WAIT_recall": gate["wait_support"],
        "WAIT_false_release": gate["wait_support"],
        "RELEASE_recall": gate["release_support"],
        "gate_accuracy": gate["rows"],
        "balanced_accuracy": None,
        "proposal_top1": gate["proposal_rows"],
        "proposal_top3": gate["proposal_rows"],
        "joint_accuracy": gate["rows"],
        "blocked_recall": pair["blocked_pairs"],
        "safe_recall": pair["safe_pairs"],
        "blocker_top1": pair["blocked_queries"],
        "auxiliary_loss": pair["queries"],
    }[metric]


def aggregate(records: list[dict], split: str) -> dict:
    macro = {name: average_metric(records, split, name) for name in METRIC_NAMES}
    micro = {}
    supports = {}
    for name in METRIC_NAMES:
        weighted = []
        denom_total = 0
        for record in records:
            value = flatten(record["metrics"][split]).get(name)
            denom = denominator(record, split, name)
            if value is None or denom is None or denom <= 0:
                continue
            weighted.append(value * denom)
            denom_total += denom
        micro[name] = None if not weighted else sum(weighted) / denom_total
        supports[name] = denom_total if denom_total else None
    if micro["WAIT_recall"] is not None and micro["RELEASE_recall"] is not None:
        micro["balanced_accuracy"] = (micro["WAIT_recall"] + micro["RELEASE_recall"]) / 2.0
        supports["balanced_accuracy"] = {
            "wait_support": supports["WAIT_recall"],
            "release_support": supports["RELEASE_recall"],
        }
    return {"fold_count": len(records), "macro": macro, "micro": micro, "micro_support": supports}


def delta_aggregate(new: dict, old: dict) -> dict:
    return {
        section: {
            name: None
            if new[section][name] is None or old[section][name] is None
            else new[section][name] - old[section][name]
            for name in METRIC_NAMES
        }
        for section in ("macro", "micro")
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--new",
        type=Path,
        default=Path(
            r"D:\7_codespace\training_outputs\terminal\p3_neighbor_nine_cycle_cv_20261009\nine_cycle"
        ),
    )
    parser.add_argument(
        "--old",
        type=Path,
        default=Path(
            r"D:\7_codespace\training_outputs\terminal\p3_neighbor_relation_cv_20261009\six_cycle"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(r"D:\7_codespace\terminal\.tmp\p3_nine_cycle_comparison_20261010.json"),
    )
    args = parser.parse_args()

    new_protocol, new_result, new_meta = load_result(args.new)
    old_protocol, old_result, old_meta = load_result(args.old)

    errors: list[str] = []
    if new_result.get("status") != "COMPLETE":
        errors.append(f"new status={new_result.get('status')!r}")
    if old_result.get("status") != "COMPLETE":
        errors.append(f"old status={old_result.get('status')!r}")
    new_folds = new_result.get("folds", [])
    old_folds = old_result.get("folds", [])
    if len(new_folds) != 9:
        errors.append(f"new fold count={len(new_folds)} expected 9")
    if len(old_folds) != 12:
        errors.append(f"old fold count={len(old_folds)} expected 12")

    new_records = [fold_record(fold, new_meta, i) for i, fold in enumerate(new_folds)]
    old_native = [
        (i, fold)
        for i, fold in enumerate(old_folds)
        if fold.get("arm") == "native_pair_auxiliary"
    ]
    old_records = [fold_record(fold, old_meta, i) for i, fold in old_native]

    new_by_seed = {r["held_seed"]: r for r in new_records}
    old_by_seed = {r["held_seed"]: r for r in old_records}
    if len(new_by_seed) != len(new_records):
        errors.append("duplicate new held_seed")
    if len(old_by_seed) != len(old_records):
        errors.append("duplicate old native held_seed")
    expected_old = [893511, 893512, 893513, 893601, 893602, 893603]
    expected_new = expected_old + [893604, 893605, 893606]
    if sorted(old_by_seed) != expected_old:
        errors.append(f"old native held seeds={sorted(old_by_seed)} expected={expected_old}")
    if sorted(new_by_seed) != expected_new:
        errors.append(f"new held seeds={sorted(new_by_seed)} expected={expected_new}")
    if any(r["arm"] != "native_pair_auxiliary" for r in new_records):
        errors.append("new result contains a non-native arm")

    comparable = []
    for seed in expected_old:
        if seed not in new_by_seed or seed not in old_by_seed:
            errors.append(f"missing comparable seed {seed}")
            continue
        nr = new_by_seed[seed]
        orow = old_by_seed[seed]
        comparable.append(
            {
                "held_seed": seed,
                "new_fold_index": nr["fold_index"],
                "old_native_fold_index": orow["fold_index"],
                "old": orow,
                "new": nr,
                "delta_new_minus_old": compare_metrics(nr["metrics"], orow["metrics"]),
            }
        )

    new_comparable = [new_by_seed[s] for s in expected_old if s in new_by_seed]
    new_supplemental = [new_by_seed[s] for s in [893604, 893605, 893606] if s in new_by_seed]
    aggregate_old = aggregate([old_by_seed[s] for s in expected_old if s in old_by_seed], "validation")
    aggregate_new_comparable = aggregate(new_comparable, "validation")
    aggregate_new_supplemental = aggregate(new_supplemental, "validation")
    aggregate_new_all = aggregate(new_records, "validation")

    checks = {
        "new_status_complete": new_result.get("status") == "COMPLETE",
        "old_status_complete": old_result.get("status") == "COMPLETE",
        "new_fold_count": len(new_folds),
        "old_fold_count": len(old_folds),
        "new_native_only": all(r["arm"] == "native_pair_auxiliary" for r in new_records),
        "new_fixed5000_present": len(new_records) == 9,
        "old_native_fixed5000_present": len(old_records) == 6,
        "old_comparable_seed_set": sorted(old_by_seed) == expected_old,
        "new_seed_set": sorted(new_by_seed) == expected_new,
        "no_query_or_q_accessed_in_protocol": not bool(
            new_protocol.get("Q_accessed") or old_protocol.get("Q_accessed")
        ),
        "errors": errors,
    }

    output = {
        "schema": "p3_nine_cycle_comparison_20261010_v1",
        "fixed_update": 5000,
        "metric_semantics": {
            "WAIT_false_release": "1 - validation gate WAIT_recall; equivalently confusion[WAIT,RELEASE]/WAIT support",
            "micro": "support-weighted from stored rates; balanced_accuracy uses aggregated WAIT/RELEASE recalls",
            "train": "included per fold as a development fit reference; primary comparison uses validation",
        },
        "inputs": {
            "new_nine_cycle": {**new_meta, "status": new_result.get("status"), "protocol": new_protocol},
            "old_six_cycle": {**old_meta, "status": old_result.get("status"), "protocol": old_protocol},
        },
        "checks": checks,
        "comparable_old6_native_vs_new6_native": comparable,
        "new_supplemental3_native": new_supplemental,
        "aggregates_validation": {
            "old6_native": aggregate_old,
            "new6_comparable_native": aggregate_new_comparable,
            "new3_supplemental_native": aggregate_new_supplemental,
            "new9_all_native": aggregate_new_all,
            "new6_minus_old6": delta_aggregate(aggregate_new_comparable, aggregate_old),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": "PASS" if not errors else "FAIL", "errors": errors}))


if __name__ == "__main__":
    main()
