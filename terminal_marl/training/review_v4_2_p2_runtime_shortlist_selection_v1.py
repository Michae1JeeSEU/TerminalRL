"""Read-only selection review after the K=3 formal PPO heldout failure."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


STAGE = "V4_2_P2_RUNTIME_SHORTLIST_SELECTION_REVIEW_V1"
PROTOCOL = Path("terminal_marl/configs/low_v4_2/ppo_p2_runtime_shortlist_selection_review_v1.yaml")
PROTOCOL_SHA = "537ed4fc01aecf1d6d28ab09518c10599041ba24beffe80d2dc8dc01a7e6254a"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protocol() -> dict[str, Any]:
    if _sha256(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("runtime-shortlist selection-review protocol changed")
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    if value.get("protocol") != STAGE:
        raise ValueError("runtime-shortlist selection-review protocol id changed")
    return value


def review(*, heldout_path: Path, runtime_certification_path: Path,
           original_checkpoint: Path) -> dict[str, Any]:
    p = protocol()
    bindings = p["bindings"]
    if _sha256(heldout_path) != bindings["heldout_manifest_sha256"]:
        raise ValueError("heldout manifest SHA mismatch")
    if _sha256(runtime_certification_path) != bindings["runtime_certification_manifest_sha256"]:
        raise ValueError("runtime certification manifest SHA mismatch")
    if _sha256(original_checkpoint) != bindings["original_candidate_checkpoint_sha256"]:
        raise ValueError("original candidate checkpoint SHA mismatch")
    heldout = json.loads(heldout_path.read_text(encoding="utf-8"))
    certification = json.loads(runtime_certification_path.read_text(encoding="utf-8"))
    decision = heldout.get("decision", {})
    false_decisions = {
        key for key, value in decision.items()
        if key != "all" and value is not True
    }
    aggregates = heldout.get("aggregates", {})
    initial = aggregates.get("initial_k3", {})
    rule = aggregates.get("rule", {})
    initial_records = [
        record for record in heldout.get("records", [])
        if record.get("controller") == "initial_k3"
    ]
    gates = {
        "heldout_failure_preserved": (
            heldout.get("stage") == "V4_2_P2_RUNTIME_SHORTLIST_HELDOUT_V1"
            and heldout.get("PASS") is False
            and decision.get("all") is False
            and false_decisions == {"trained_request_to_touchdown_nonworse_than_initial"}
        ),
        "heldout_episode_integrity": (
            decision.get("all_records_complete") is True
            and decision.get("all_episode_gates") is True
        ),
        "formal_checkpoint_rejected": (
            heldout.get("trained_checkpoint_sha256")
            == bindings["rejected_formal_checkpoint_sha256"]
            and heldout.get("checkpoint_promotable") is False
            and heldout.get("STATIC_GEOMETRY_QUALIFICATION_ALLOWED") is False
        ),
        "original_k3_runtime_certification_pass": (
            certification.get("stage") == "V4_2_P2_GATE_RUNTIME_SHORTLIST_EVALUATION_V2"
            and certification.get("status") == "COMPLETE"
            and certification.get("decision", {}).get("all") is True
            and certification.get("RUNTIME_SHORTLIST_FREEZE") is True
            and certification.get("candidate_checkpoint_sha256")
            == bindings["original_candidate_checkpoint_sha256"]
        ),
        "original_k3_heldout_complete_and_safe": (
            initial.get("records") == 6
            and initial.get("requests") == initial.get("completed") == 52
            and initial.get("completion_fraction") == 1.0
            and initial.get("safety_all_zero") is True
            and initial.get("all_gates") is True
        ),
        "original_k3_heldout_contract_pass": (
            len(initial_records) == 6
            and all(
                record.get("result", {}).get("runtime_shortlist_checks", {}).get("all") is True
                for record in initial_records
            )
        ),
        "original_k3_wait_better_than_rule": (
            initial.get("weighted_wait_seconds") is not None
            and rule.get("weighted_wait_seconds") is not None
            and initial["weighted_wait_seconds"] < rule["weighted_wait_seconds"]
        ),
        "original_k3_request_to_touchdown_better_than_rule": (
            initial.get("request_to_touchdown_seconds") is not None
            and rule.get("request_to_touchdown_seconds") is not None
            and initial["request_to_touchdown_seconds"]
            < rule["request_to_touchdown_seconds"]
        ),
        "frozen_proposal_subgraph": decision.get("frozen_proposal_subgraph") is True,
    }
    passed = all(gates.values())
    return {
        "stage": STAGE,
        "PASS": passed,
        "gates": gates,
        "SELECTED_LOW_LEVEL_CANDIDATE": "original_k3" if passed else None,
        "selected_checkpoint_sha256": (
            bindings["original_candidate_checkpoint_sha256"] if passed else None
        ),
        "REJECTED_FORMAL_CHECKPOINT": bool(passed),
        "P2_PPO_STOP": bool(passed),
        "P3_PROTOCOL_PREREGISTRATION_ALLOWED": bool(passed),
        "P3_ALLOWED": False,
        "checkpoint_promotable": False,
        "protocol_sha256": PROTOCOL_SHA,
        "heldout_manifest_sha256": bindings["heldout_manifest_sha256"],
        "runtime_certification_manifest_sha256":
            bindings["runtime_certification_manifest_sha256"],
        "initial_k3": initial,
        "rule": rule,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--runtime-certification-manifest", type=Path, required=True)
    parser.add_argument("--original-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = review(
        heldout_path=args.heldout_manifest,
        runtime_certification_path=args.runtime_certification_manifest,
        original_checkpoint=args.original_checkpoint,
    )
    path = args.output / "review.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "stage": result["stage"],
        "PASS": result["PASS"],
        "P3_PROTOCOL_PREREGISTRATION_ALLOWED":
            result["P3_PROTOCOL_PREREGISTRATION_ALLOWED"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
