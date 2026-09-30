"""Run the fixed proposal-support control for the registered P3 scenes.

The control is a frozen off-policy proposal intervention.  It keeps the frozen gate, legacy
actor mask, keyed random streams, physics, safety, energy, emergency, and
capacity paths unchanged.  On legacy RELEASE rows only, it samples from a
fixed epsilon mixture of the source proposal distribution.  No training or
qualification is performed by this module.
"""
from __future__ import annotations

import argparse
import copy
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from terminal_marl.core.low_level_joint_action_v4_2 import HierarchicalActionV42, RELEASE_GATE, WAIT_GATE
from terminal_marl.training import paired_gate_diagnostic as paired
from terminal_marl.training import run_p3_forced_wait_frozen_20260930 as frozen
from terminal_marl.training import run_p3_gate_release_control_20260930 as gate
from terminal_marl.training import run_p3_opportunity_coverage_20260930 as opportunity


STAGE = "P3_PROPOSAL_SUPPORT_CONTROL_20261001_V1"
PPO_ALLOWED = False
HELDOUT_USED = False
TEACHER_USED = False
OPTIMIZER_USED = False
OFF_POLICY = True
NO_LOGPROB = True
NO_TRAINING = True
EPSILON = 0.10
Q_FLOOR_MIN = 0.004
PROPOSAL_COUNT = 25
REFERENCE_ROOT = Path("/home/user/training_outputs/terminal/p3_followup_20260930/forced_wait_frozen")
OUTPUT_DIR = Path("/home/user/training_outputs/terminal/p3_followup_20260930/proposal_support_control_20261001")
_PAIR_ORDER = (1, 5, 9, 13, 17, 21, 14, 15, 16)
_PAIR_BY_ID = {int(item["pair_id"]): dict(item) for item in opportunity.PAIR_SPECS}
PAIR_SPECS = tuple(_PAIR_BY_ID[pair_id] for pair_id in _PAIR_ORDER)
CONTROL_NAME = "legacy_release_epsilon_uniform_proposal_support"
MAX_STEPS = int(frozen.MAX_STEPS)


class ProposalSupportControlError(ValueError):
    """Raised when the fixed proposal-support contract is not maintained."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProposalSupportControlError(message)


def _jsonable(value: Any) -> Any:
    return gate._jsonable(value)


def _read_json(path: str | Path) -> dict[str, Any]:
    return gate._read_json(path)


def _write_json(path: Path, value: Mapping[str, Any], *, refuse_existing: bool = False) -> None:
    gate._write_json(path, value, refuse_existing=refuse_existing)


def _sha256(path: str | Path) -> str:
    return gate._sha256(path)


def _proposal_mixture(q: Sequence[float], *, epsilon: float = EPSILON) -> np.ndarray:
    """Return the fixed mixture without changing ``q``.

    The epsilon-zero branch is a test-only parity helper: it returns a copy of
    the source vector exactly, including a non-unit total, because the frozen
    CDF sampler performs its own normalization.
    """

    source = np.asarray(q, dtype=np.float64)
    _require(source.ndim == 1 and source.size == PROPOSAL_COUNT, "proposal q must contain 25 entries")
    _require(np.all(np.isfinite(source)) and np.all(source >= 0.0), "proposal q must be finite and nonnegative")
    _require(math.isfinite(float(epsilon)) and 0.0 <= float(epsilon) <= 1.0, "epsilon is outside [0, 1]")
    if float(epsilon) == 0.0:
        _require(float(source.sum()) > 0.0, "proposal q has no positive mass")
        return source.copy()
    total = float(source.sum())
    _require(total > 0.0 and math.isfinite(total), "proposal q has no positive mass")
    mixed = (1.0 - float(epsilon)) * (source / total) + float(epsilon) / PROPOSAL_COUNT
    _require(np.all(np.isfinite(mixed)), "proposal mixture is non-finite")
    # epsilon=0 is a test-only equivalence path to the frozen request.  The
    # registered control itself uses EPSILON=0.10 and is subject to the floor.
    if float(epsilon) > 0.0:
        _require(float(mixed.min()) >= Q_FLOOR_MIN - 1e-12, "proposal mixture violates q floor")
    _require(abs(float(mixed.sum()) - 1.0) <= 1e-12, "proposal mixture is not normalized")
    return mixed.copy()


def _support_keyed_request(policy: Any, env: Any, keyed_seed: int, *, epsilon: float = EPSILON) -> dict[str, Any]:
    """Build one request with source gate/mask and epsilon-mixed RELEASE proposals."""

    caller_rng = paired._rng_snapshot()
    try:
        context = paired._details_context(policy, env)
        actions: dict[str, np.ndarray] = {}
        shortlists: dict[str, tuple[np.ndarray, ...]] = {}
        rows: list[dict[str, Any]] = []
        step = int(context.get("step", getattr(env, "step_count", 0)))
        release_count = 0
        wait_count = 0
        for index, uid in enumerate(context["uids"]):
            arrival_id = frozen._arrival_id(env, uid)
            actor_mask = bool(context["actor_mask"][index])
            gate_uniform = None
            proposal_uniform = None
            proposal: int | None = None
            shortlist_value: tuple[np.ndarray, ...] = ()
            source_q = np.asarray(context["proposal_probabilities"][index], dtype=np.float64)
            source_logits = np.asarray(context["proposal_logits"][index], dtype=np.float64)
            q_before = source_q.copy()
            logits_before = source_logits.copy()
            if actor_mask:
                gate_uniform = paired.keyed_uniform(int(keyed_seed), step, arrival_id, "gate")
                sampled_gate = paired._cdf_sample(context["gate_probabilities"][index], gate_uniform)
                if sampled_gate == RELEASE_GATE:
                    release_count += 1
                    proposal_uniform = paired.keyed_uniform(int(keyed_seed), step, arrival_id, "proposal")
                    mixed_q = _proposal_mixture(source_q, epsilon=epsilon)
                    proposal = paired._cdf_sample(mixed_q, proposal_uniform)
                    action = HierarchicalActionV42(RELEASE_GATE, proposal).normalized.copy()
                    shortlist_value = paired._shortlist_from_logits(proposal, source_logits)
                    shortlists[str(uid)] = shortlist_value
                    gate_value = RELEASE_GATE
                else:
                    wait_count += 1
                    mixed_q = None
                    action = HierarchicalActionV42(WAIT_GATE, None).normalized.copy()
                    gate_value = WAIT_GATE
            else:
                wait_count += 1
                mixed_q = None
                action = HierarchicalActionV42(WAIT_GATE, None).normalized.copy()
                gate_value = WAIT_GATE
            _require(np.array_equal(source_q, q_before), "source proposal q was mutated")
            _require(np.array_equal(source_logits, logits_before), "source proposal logits were mutated")
            actions[str(uid)] = action
            rows.append({
                "uid": str(uid),
                "arrival_id": int(arrival_id),
                "actor_mask": actor_mask,
                "gate": int(gate_value),
                "legacy_gate_sample": int(gate_value),
                "proposal": None if proposal is None else int(proposal),
                "gate_uniform": gate_uniform,
                "proposal_uniform": proposal_uniform,
                "source_q": source_q.tolist(),
                "mixed_q": None if mixed_q is None else mixed_q.tolist(),
                "source_q_unchanged": True,
                "source_logits_unchanged": True,
                "action": np.asarray(action, dtype=np.float32).tolist(),
                "shortlist": [np.asarray(item, dtype=np.float32).tolist() for item in shortlist_value],
            })
        return {
            "actions": actions,
            "shortlists": shortlists,
            "rows": rows,
            "step": step,
            "keyed_seed": int(keyed_seed),
            "keying": "arrival_id|absolute_tick|stream",
            "control": {
                "name": CONTROL_NAME,
                "epsilon": float(epsilon),
                "q_floor_min": Q_FLOOR_MIN,
                "proposal_count": PROPOSAL_COUNT,
                "sampled_first": True,
                "source_gate_unchanged": True,
                "source_mask_unchanged": True,
                "source_logits_unchanged": True,
                "off_policy": OFF_POLICY,
                "logprob_recorded": False,
                "release_rows": release_count,
                "wait_rows": wait_count,
            },
        }
    finally:
        paired._restore_rng(caller_rng)


def _integrity_and_metrics(arm: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    checks = gate._integrity_checks(arm, reference)
    metrics = gate._business_metrics(arm)
    return {"checks": checks, "metrics": metrics}


def _run_support_arm(source: Mapping[str, Any], config: Any, spec: Mapping[str, Any],
                     reference: Mapping[str, Any], progress: Any) -> dict[str, Any]:
    original_keyed = frozen._keyed_request
    frozen._keyed_request = _support_keyed_request
    try:
        arm = frozen._run_arm(
            copy.deepcopy(source["policy"]), config,
            env_seed=int(spec["env_seed"]), keyed_seed=int(spec["keyed_policy_seed"]),
            mode="legacy", progress=progress, pair_id=int(spec["pair_id"]),
        )
    finally:
        frozen._keyed_request = original_keyed
    result = _integrity_and_metrics(arm, reference)
    arm["proposal_support_control"] = {
        "name": CONTROL_NAME, "epsilon": EPSILON, "q_floor_min": Q_FLOOR_MIN,
        "proposal_count": PROPOSAL_COUNT, "sampled_first": True,
        "off_policy": OFF_POLICY, "logprob_recorded": False, "training": False,
    }
    arm["control_metrics"] = result["metrics"]
    arm["reference_control_checks"] = result["checks"]
    if arm.get("status") == "completed" and result["checks"].get("technical_ok") is not True:
        arm["status"] = "technical_error"
        arm["error"] = {"type": "ProposalSupportIntegrityError", "checks": result["checks"]}
        arm["control_metrics"] = gate._business_metrics(arm)
    return arm


def _run_pair(source: Mapping[str, Any], config: Any, spec: Mapping[str, Any],
              reference: Mapping[str, Any], output: Path) -> dict[str, Any]:
    pair_id = int(spec["pair_id"])

    def progress(row: Mapping[str, Any]) -> None:
        _write_json(output / "progress.json", {
            "stage": STAGE, "phase": "arm", "status": "RUNNING", **dict(row),
        })

    try:
        arm = _run_support_arm(source, config, spec, reference, progress)
    except Exception as exc:
        arm = {
            "mode": "legacy", "env_seed": int(spec["env_seed"]),
            "keyed_policy_seed": int(spec["keyed_policy_seed"]),
            "status": "technical_error", "censored": None,
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "reference_control_checks": {"known": False, "technical_ok": False},
            "control_metrics": {"status": "technical_error"},
        }
    arm_path = output / "arms" / f"pair_{pair_id:04d}_support.json"
    _write_json(arm_path, arm, refuse_existing=True)
    status = str(arm.get("status"))
    pair_status = "PAIR_TECHNICAL_ERROR" if status in {"error", "technical_error", "safety_violation"} else (
        "PAIR_CENSORED" if status == "censored" else "PAIR_OK"
    )
    reference_arm = reference["arm"]
    support_metrics = gate._business_metrics(arm)
    reference_metrics = gate._business_metrics(reference_arm)
    support_failed = gate._issue_rows(arm, "failed")
    support_unresolved = gate._issue_rows(arm, "unresolved")
    support_diverted = gate._issue_rows(arm, "diverted")
    reference_failed = gate._issue_rows(reference_arm, "failed")
    reference_unresolved = gate._issue_rows(reference_arm, "unresolved")
    reference_diverted = gate._issue_rows(reference_arm, "diverted")
    known = bool(
        support_metrics.get("complete_scene_metrics_known")
        and reference_metrics.get("complete_scene_metrics_known")
    )
    def new_ids(rows: Sequence[Mapping[str, Any]], old: Sequence[Mapping[str, Any]]) -> list[int] | None:
        if not known:
            return None
        old_ids = {int(item["arrival_id"]) for item in old}
        return [int(item["arrival_id"]) for item in rows if int(item["arrival_id"]) not in old_ids]
    summary = {
        "stage": STAGE, "pair": dict(spec), "status": pair_status,
        "arm_path": str(arm_path), "reference": dict(reference["metadata"]),
        "reference_control_checks": arm.get("reference_control_checks", {}),
        "control_metrics": support_metrics, "support_metrics": support_metrics,
        "reference_metrics": reference_metrics,
        "complete_scene_delta": gate._paired_business_delta(support_metrics, reference_metrics),
        "failure_comparison_known": bool(known and reference_metrics.get("complete_scene_metrics_known")),
        "control_failed_rows": support_failed, "control_unresolved_rows": support_unresolved,
        "control_diverted_rows": support_diverted,
        "reference_failed_rows": reference_failed, "reference_unresolved_rows": reference_unresolved,
        "reference_diverted_rows": reference_diverted,
        "new_failed_arrival_ids": new_ids(support_failed, reference_failed),
        "new_unresolved_arrival_ids": new_ids(support_unresolved, reference_unresolved),
        "new_diverted_arrival_ids": new_ids(support_diverted, reference_diverted),
        "raw_reward_excluding_rejection_penalty": arm.get("raw_reward_excluding_rejection_penalty"),
        "release_rejection_penalty": arm.get("release_rejection_penalty"),
        "off_policy": OFF_POLICY, "logprob_recorded": False, "training": False,
    }
    _write_json(output / "pairs" / f"pair_{pair_id:04d}.json", summary, refuse_existing=True)
    return {"summary": summary, "arm": arm}


def _build_preflight(reference_root: Path = REFERENCE_ROOT) -> dict[str, Any]:
    base = gate._build_preflight(reference_root)
    gate._validate_preflight(base, reference_root)
    return {
        "stage": STAGE, "phase": "preflight", "status": "PREFLIGHT_OK",
        "PPO_ALLOWED": PPO_ALLOWED, "heldout_used": HELDOUT_USED,
        "teacher_used": TEACHER_USED, "optimizer_used": OPTIMIZER_USED,
        "off_policy": OFF_POLICY, "no_logprob": NO_LOGPROB, "no_training": NO_TRAINING,
        "proposal_support": {
            "name": CONTROL_NAME, "epsilon": EPSILON, "q_floor_min": Q_FLOOR_MIN,
            "proposal_count": PROPOSAL_COUNT, "sampled_first": True,
            "pair_ids": list(_PAIR_ORDER), "keying": "arrival_id|absolute_tick|stream",
            "gate_and_mask": "source_legacy_unchanged", "physics_safety_energy_emergency_capacity": "unchanged",
        },
        "base_gate_preflight": base,
        "reference": base["reference"], "reference_arms": base["reference_arms"],
        "runtime": base["runtime"], "runner_sha256": _sha256(__file__),
        "qualification": "not_a_qualification; fixed off_policy proposal_support control",
    }


def _validate_preflight(payload: Mapping[str, Any], reference_root: Path) -> None:
    _require(payload.get("stage") == STAGE and payload.get("status") == "PREFLIGHT_OK",
             "proposal-support preflight is not successful")
    for key, expected in (("PPO_ALLOWED", False), ("heldout_used", False), ("teacher_used", False),
                          ("optimizer_used", False), ("off_policy", True), ("no_logprob", True),
                          ("no_training", True)):
        _require(payload.get(key) is expected, f"proposal-support flag {key} changed")
    _require(payload.get("runner_sha256") == _sha256(__file__), "proposal-support runner changed after preflight")
    support = payload.get("proposal_support", {})
    _require(support.get("pair_ids") == list(_PAIR_ORDER), "proposal-support pair grid changed")
    _require(float(support.get("epsilon")) == EPSILON and float(support.get("q_floor_min")) == Q_FLOOR_MIN,
             "proposal-support parameters changed")
    _require(int(support.get("proposal_count")) == PROPOSAL_COUNT, "proposal count changed")
    base = payload.get("base_gate_preflight")
    _require(isinstance(base, Mapping), "base gate preflight missing")
    _require(payload.get("runtime") == base.get("runtime"), "outer runtime binding differs from nested validation")
    _require(payload.get("reference") == base.get("reference"), "outer reference binding differs from nested validation")
    _require(payload.get("reference_arms") == base.get("reference_arms"),
             "outer reference-arm binding differs from nested validation")
    gate._validate_preflight(base, reference_root)


def _phase_inputs(output: Path, reference_root: Path):
    preflight = _read_json(output / "preflight.json")
    _validate_preflight(preflight, reference_root)
    _op_preflight, config, source, layout, _ = gate._opportunity_inputs(reference_root)
    return preflight, config, source, layout


def run_preflight(output: Path, reference_root: Path = REFERENCE_ROOT) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    payload = _build_preflight(reference_root)
    _write_json(output / "preflight.json", payload, refuse_existing=True)
    _write_json(output / "progress.json", {
        "stage": STAGE, "phase": "preflight", "status": "PREFLIGHT_OK", "environment_created": False,
    })
    return payload


def _validate_pilot_reuse(pilot: Mapping[str, Any], preflight: Mapping[str, Any], output: Path) -> None:
    _require(pilot.get("status") in {"PILOT_OK", "PILOT_CENSORED"}, "full phase requires retained pilot")
    _require(pilot.get("pair", {}).get("pair") == PAIR_SPECS[0], "pilot pair does not match pair 1")
    _require(pilot.get("preflight_runner_sha256") == preflight.get("runner_sha256"), "pilot runner binding changed")
    _require(pilot.get("preflight_sha256") == _sha256(output / "preflight.json"),
             "stored pilot preflight changed")
    expected_binding = {
        "proposal_support": preflight["proposal_support"],
        "runtime": preflight["runtime"],
        "reference": preflight["reference"],
        "reference_arms": preflight["reference_arms"],
        "base_gate_runner_sha256": preflight["base_gate_preflight"]["runner_sha256"],
        "self_runner_sha256": preflight["runner_sha256"],
    }
    _require(pilot.get("preflight_binding") == expected_binding, "pilot preflight binding changed")
    arm_path = output / "arms" / "pair_0001_support.json"
    pair_path = output / "pairs" / "pair_0001.json"
    artifacts = pilot.get("pilot_artifacts", {})
    _require(arm_path.is_file() and pair_path.is_file(), "pilot artifact is missing")
    _require(artifacts.get("arm_path") == str(arm_path), "pilot arm path changed")
    _require(artifacts.get("pair_path") == str(pair_path), "pilot pair path changed")
    _require(artifacts.get("arm_sha256") == _sha256(arm_path), "pilot arm artifact changed")
    _require(artifacts.get("pair_sha256") == _sha256(pair_path), "pilot pair artifact changed")
    stored = _read_json(output / "pairs" / "pair_0001.json")
    arm = _read_json(output / "arms" / "pair_0001_support.json")
    _require(stored.get("pair") == PAIR_SPECS[0] and stored.get("status") in {"PAIR_OK", "PAIR_CENSORED"},
             "stored pilot pair changed")
    _require(stored == pilot.get("pair"), "pilot summary differs from stored pair")
    expected_status = "PAIR_OK" if pilot.get("status") == "PILOT_OK" else "PAIR_CENSORED"
    expected_arm_status = "completed" if pilot.get("status") == "PILOT_OK" else "censored"
    _require(stored.get("status") == expected_status, "stored pilot status changed")
    _require(arm.get("status") == expected_arm_status, "stored pilot arm status changed")
    expected_control = {
        "name": CONTROL_NAME, "epsilon": EPSILON, "q_floor_min": Q_FLOOR_MIN,
        "proposal_count": PROPOSAL_COUNT, "sampled_first": True,
        "off_policy": OFF_POLICY, "logprob_recorded": False, "training": False,
    }
    _require(arm.get("proposal_support_control") == expected_control,
             "stored pilot proposal-support parameters changed")
    _require(gate._business_metrics(arm) == stored.get("control_metrics"),
             "stored pilot metrics do not match arm")
    _require(arm.get("reference_control_checks") == stored.get("reference_control_checks"),
             "stored pilot integrity checks do not match arm")
    checks = arm.get("reference_control_checks", {})
    if expected_arm_status == "completed":
        _require(checks.get("technical_ok") is True, "stored natural pilot integrity is incomplete")
    else:
        metrics = stored.get("control_metrics", {})
        _require(checks.get("technical_ok") is None, "stored censored pilot integrity is not unknown")
        _require(metrics.get("complete_scene_metrics_known") is False,
                 "censored pilot complete metrics must remain unknown")
        _require(stored.get("complete_scene_delta", {}).get("complete_scene_delta_known") is False,
                 "censored pilot delta must remain unknown")
        for key in ("new_failed_arrival_ids", "new_unresolved_arrival_ids", "new_diverted_arrival_ids"):
            _require(stored.get(key) is None, f"censored pilot {key} must remain unknown")


def run_pilot(output: Path, reference_root: Path = REFERENCE_ROOT) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    preflight, config, source, layout = _phase_inputs(output, reference_root)
    spec = PAIR_SPECS[0]
    reference = layout["_arms"][str(spec["pair_id"])]
    reference["metadata"] = layout["arms"][str(spec["pair_id"])]
    result = _run_pair(source, config, spec, reference, output)
    pair_status = result["summary"]["status"]
    status = "PILOT_OK" if pair_status == "PAIR_OK" else "PILOT_CENSORED" if pair_status == "PAIR_CENSORED" else "PILOT_TECHNICAL_ERROR"
    payload = {
        "stage": STAGE, "phase": "pilot", "status": status,
        "PPO_ALLOWED": PPO_ALLOWED, "off_policy": OFF_POLICY,
        "pair": result["summary"], "preflight_runner_sha256": preflight["runner_sha256"],
        "preflight_sha256": _sha256(output / "preflight.json"),
        "pilot_artifacts": {
            "arm_path": result["summary"]["arm_path"],
            "arm_sha256": _sha256(output / "arms" / "pair_0001_support.json"),
            "pair_path": str(output / "pairs" / "pair_0001.json"),
            "pair_sha256": _sha256(output / "pairs" / "pair_0001.json"),
        },
        "preflight_binding": {
            "proposal_support": preflight["proposal_support"],
            "runtime": preflight["runtime"],
            "reference": preflight["reference"],
            "reference_arms": preflight["reference_arms"],
            "base_gate_runner_sha256": preflight["base_gate_preflight"]["runner_sha256"],
            "self_runner_sha256": preflight["runner_sha256"],
        },
        "qualification": "not_a_qualification",
    }
    _write_json(output / "pilot.json", payload, refuse_existing=True)
    _write_json(output / "progress.json", payload)
    if status == "PILOT_TECHNICAL_ERROR":
        raise ProposalSupportControlError("pilot stopped after technical or safety failure")
    return payload


def run_full(output: Path, reference_root: Path = REFERENCE_ROOT) -> dict[str, Any]:
    pilot = _read_json(output / "pilot.json")
    preflight, config, source, layout = _phase_inputs(output, reference_root)
    _validate_pilot_reuse(pilot, preflight, output)
    summaries = [dict(pilot["pair"])]
    for spec in PAIR_SPECS[1:]:
        reference = layout["_arms"][str(spec["pair_id"])]
        reference["metadata"] = layout["arms"][str(spec["pair_id"])]
        result = _run_pair(source, config, spec, reference, output)
        summaries.append(result["summary"])
        if result["summary"]["status"] == "PAIR_TECHNICAL_ERROR":
            payload = {
                "stage": STAGE, "phase": "full", "status": "FULL_TECHNICAL_ERROR",
                "PPO_ALLOWED": PPO_ALLOWED, "off_policy": OFF_POLICY, "pair_summaries": summaries,
            }
            _write_json(output / "full.json", payload, refuse_existing=True)
            _write_json(output / "progress.json", payload)
            raise ProposalSupportControlError("full phase stopped after technical or safety failure")
    review = gate._business_review(summaries)
    payload = {
        "stage": STAGE, "phase": "full", "status": "FULL_OK",
        "PPO_ALLOWED": PPO_ALLOWED, "off_policy": OFF_POLICY,
        "pair_count": len(summaries), "pair_summaries": summaries,
        "business_review": review,
        "qualification": "not_a_qualification; fixed proposal-support control",
    }
    _write_json(output / "full.json", payload, refuse_existing=True)
    _write_json(output / "progress.json", payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "pilot", "full"), required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--reference-root", type=Path, default=REFERENCE_ROOT)
    args = parser.parse_args(argv)
    try:
        if args.phase == "preflight":
            run_preflight(args.output_dir, args.reference_root)
        elif args.phase == "pilot":
            run_pilot(args.output_dir, args.reference_root)
        else:
            run_full(args.output_dir, args.reference_root)
    except Exception as exc:
        _write_json(args.output_dir / "progress.json", {
            "stage": STAGE, "phase": args.phase, "status": "FAILED",
            "error": {"type": type(exc).__name__, "message": str(exc)},
        })
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
