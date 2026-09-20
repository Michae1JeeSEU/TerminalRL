"""Read-only admission review for the frozen K=3 runtime shortlist boundary."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


STAGE = "V4_2_P2_RUNTIME_SHORTLIST_ADMISSION_REVIEW_V1"
PROTOCOL = Path("terminal_marl/configs/low_v4_2/ppo_p2_runtime_shortlist_admission_review_v1.yaml")
PROTOCOL_SHA = "076c4feb5937e500050e9bb66486408363dc127a33713d276d96d658eab95152"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protocol() -> dict[str, Any]:
    if _sha256(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("runtime-shortlist admission protocol changed")
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    if value.get("protocol") != STAGE:
        raise ValueError("runtime-shortlist admission protocol id changed")
    return value


def source_hashes() -> dict[str, str]:
    paths = (
        Path("terminal_marl/envs/parallel_env.py"),
        Path("terminal_marl/core/safety_trace.py"),
        Path("terminal_marl/training/student_rollout_v4_2.py"),
        Path("terminal_marl/training/run_v4_2_i2_dagger_preflight.py"),
        Path("terminal_marl/algo/student_ppo_v4_2.py"),
        Path(__file__),
    )
    return {str(path): _sha256(path) for path in paths}


def review_gates(manifest: Mapping[str, Any], p: Mapping[str, Any]) -> dict[str, bool]:
    bindings = p["bindings"]
    records = list(manifest.get("records", ()))
    counts = Counter(str(row.get("controller")) for row in records)
    controllers = list(bindings["expected_controllers"])
    episode_gates = bool(records) and all(
        bool(row.get("result", {}).get("gates", {}).get("all")) for row in records
    )
    completed = {
        name: sum(int(row.get("result", {}).get("completed", 0)) for row in records
                  if row.get("controller") == name)
        for name in controllers
    }
    requests = {
        name: sum(int(row.get("result", {}).get("requests", 0)) for row in records
                  if row.get("controller") == name)
        for name in controllers
    }
    k3 = [row.get("result", {}) for row in records if row.get("controller") == "candidate_k3"]
    fallback_total = sum(int(row.get("runtime_shortlist_checks", {}).get("fallback_selected", 0)) for row in k3)
    max_checked = max((int(row.get("runtime_shortlist_checks", {}).get("max_candidates_checked", 0)) for row in k3), default=0)
    decision = manifest.get("decision", {})
    deltas = manifest.get("k3_minus_k1", {})
    return {
        "manifest_stage": manifest.get("stage") == "V4_2_P2_GATE_RUNTIME_SHORTLIST_EVALUATION_V2",
        "protocol_binding": manifest.get("protocol_sha256") == bindings["runtime_shortlist_protocol_sha256"],
        "formal_complete": manifest.get("status") == "COMPLETE" and not bool(manifest.get("smoke")),
        "record_shape": len(records) == int(bindings["expected_records"])
            and all(counts[name] == int(bindings["expected_records_per_controller"]) for name in controllers),
        "episode_gates": episode_gates,
        "completion": all(completed[name] == requests[name] == int(bindings["expected_requests_per_controller"])
                          for name in controllers),
        "decision": bool(decision.get("all")) and bool(manifest.get("RUNTIME_SHORTLIST_FREEZE")),
        "safety": all(bool(manifest.get("aggregates", {}).get(name, {}).get("safety_all_zero")) for name in controllers),
        "k3_contract": bool(k3) and all(bool(row.get("runtime_shortlist_checks", {}).get("all")) for row in k3),
        "k3_bounded": max_checked == int(p["required_evidence"]["max_candidates_checked"]),
        "k3_fallback_exercised": fallback_total > 0 and bool(decision.get("k3_fallback_selected_nonzero")),
        "requested_likelihood": all(bool(row.get("requested_logprob_exact")) for row in k3),
        "teacher_free": all(bool(row.get("no_teacher")) for row in k3),
        "effectiveness": bool(decision.get("k3_wait_nonworse"))
            and bool(decision.get("k3_request_to_touchdown_nonworse"))
            and bool(decision.get("at_least_one_effectiveness_metric_strictly_better"))
            and float(deltas.get("weighted_wait_seconds", 1.0)) <= 0.0
            and float(deltas.get("request_to_touchdown_seconds", 1.0)) <= 0.0,
        "upstream_scope_preserved": manifest.get("PPO_ALLOWED") is False
            and manifest.get("checkpoint_promotable") is False,
    }


def review(*, manifest_path: Path, checkpoint_path: Path, output: Path) -> dict[str, Any]:
    p = protocol()
    bindings = p["bindings"]
    if _sha256(manifest_path) != bindings["runtime_shortlist_manifest_sha256"]:
        raise ValueError("runtime-shortlist manifest SHA mismatch")
    if _sha256(checkpoint_path) != bindings["candidate_checkpoint_sha256"]:
        raise ValueError("candidate checkpoint SHA mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    gates = review_gates(manifest, p)
    passed = bool(gates) and all(gates.values())
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "stage": STAGE, "PASS": passed, "gates": gates,
        "protocol_sha256": PROTOCOL_SHA,
        "runtime_shortlist_manifest": str(manifest_path),
        "runtime_shortlist_manifest_sha256": _sha256(manifest_path),
        "candidate_checkpoint": str(checkpoint_path),
        "candidate_checkpoint_sha256": _sha256(checkpoint_path),
        "source_hashes": source_hashes(),
        "ENTRY_PROTOCOL_PREREGISTRATION_ALLOWED": passed,
        "ENTRY_VALIDATION_ALLOWED": False,
        "PPO_ALLOWED": False,
        "P3_ALLOWED": False,
        "checkpoint_promotable": False,
    }
    (output / "review.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(review(
        manifest_path=args.manifest,
        checkpoint_path=args.candidate_checkpoint,
        output=args.output,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
