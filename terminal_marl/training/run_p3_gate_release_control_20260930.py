"""Run the frozen always-RELEASE gate control for P3.

This is an off-policy, no-training control.  It reuses the frozen arm runner,
the V2 opportunity preflight, and the completed legacy reference arms.  Only
the gate factor is intervened on; proposal sampling and the native execution
path remain unchanged.
"""
from __future__ import annotations

import argparse
import copy
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from terminal_marl.core.low_level_joint_action_v4_2 import HierarchicalActionV42, RELEASE_GATE, WAIT_GATE
from terminal_marl.training import paired_gate_diagnostic as paired
from terminal_marl.training import run_p3_forced_wait_frozen_20260930 as frozen
from terminal_marl.training import run_p3_opportunity_coverage_20260930 as opportunity


STAGE = "P3_GATE_RELEASE_CONTROL_20260930_V1"
PPO_ALLOWED = False
HELDOUT_USED = False
TEACHER_USED = False
OPTIMIZER_USED = False
OFF_POLICY = True
NO_LOGPROB = True
NO_TRAINING = True
REFERENCE_ROOT = Path("/home/user/training_outputs/terminal/p3_followup_20260930/forced_wait_frozen")
OPPORTUNITY_ROOT = Path("/home/user/training_outputs/terminal/p3_followup_20260930/opportunity_coverage_v2")
OUTPUT_DIR = Path("/home/user/training_outputs/terminal/p3_followup_20260930/gate_release_control")
PAIR_SPECS = tuple(dict(item) for item in opportunity.PAIR_SPECS)
WINDOWLESS_CONTROL = "all_actors_legacy_mask_true_force_release_else_canonical_wait"
MAX_STEPS = int(frozen.MAX_STEPS)


class GateReleaseControlError(ValueError):
    """Raised when the fixed gate-control contract cannot be maintained."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GateReleaseControlError(message)


def _jsonable(value: Any) -> Any:
    return opportunity._jsonable(value)


def _read_json(path: str | Path) -> dict[str, Any]:
    return opportunity._read_json(path)


def _write_json(path: Path, value: Mapping[str, Any], *, refuse_existing: bool = False) -> None:
    opportunity._write_json(path, value, refuse_existing=refuse_existing)


def _sha256(path: str | Path) -> str:
    return opportunity._sha256(path)


def _sync_path(path: str | Path) -> Path:
    return opportunity._sync_path(path)


def _config_snapshot(config: Any) -> dict[str, Any]:
    return opportunity._config_snapshot(config)


def _binding_equal(left: Any, right: Any, key: str) -> bool:
    return opportunity._binding_equal(left, right, key)


def _reference_layout(reference_root: Path) -> dict[str, Any]:
    return opportunity._reference_layout(reference_root)


def _opportunity_inputs(reference_root: Path) -> tuple[dict[str, Any], Any, dict[str, Any], dict[str, Any], Path]:
    """Read the already completed V2 opportunity preflight and its bindings."""

    preflight_path = _sync_path(OPPORTUNITY_ROOT / "preflight.json")
    _require(preflight_path.is_file(), f"V2 opportunity preflight missing: {preflight_path}")
    preflight, config, source, layout = opportunity._phase_config(OPPORTUNITY_ROOT, reference_root)
    _require(preflight.get("status") == "PREFLIGHT_OK", "V2 opportunity preflight is not successful")
    return preflight, config, source, layout, preflight_path


def _build_preflight(reference_root: Path = REFERENCE_ROOT) -> dict[str, Any]:
    opportunity_preflight, config, source, layout, opportunity_path = _opportunity_inputs(reference_root)
    reference = {key: value for key, value in layout.items() if key != "_arms"}
    reference_arms = {
        str(pair_id): {
            "sha256": metadata["sha256"],
            "env_seed": int(metadata["env_seed"]),
            "keyed_policy_seed": int(metadata["keyed_policy_seed"]),
        }
        for pair_id, metadata in layout["arms"].items()
    }
    return {
        "stage": STAGE,
        "phase": "preflight",
        "status": "PREFLIGHT_OK",
        "PPO_ALLOWED": PPO_ALLOWED,
        "heldout_used": HELDOUT_USED,
        "teacher_used": TEACHER_USED,
        "optimizer_used": OPTIMIZER_USED,
        "off_policy": OFF_POLICY,
        "no_logprob": NO_LOGPROB,
        "no_training": NO_TRAINING,
        "control": {
            "name": WINDOWLESS_CONTROL,
            "max_steps": MAX_STEPS,
            "pair_ids": [int(item["pair_id"]) for item in PAIR_SPECS],
            "proposal_sampling": "frozen_q|same_arrival_id_absolute_tick_proposal_uniform",
            "shortlist": "sampled_proposal_rank1|source_logits_top2_fill",
            "gate_uniform_recorded": True,
            "native_safety_unchanged": True,
        },
        "runtime": {
            "config_snapshot": _config_snapshot(config),
            "opportunity_preflight_sha256": _sha256(opportunity_path),
            "opportunity_runner_sha256": opportunity_preflight.get("runner_sha256"),
            "source_binding": frozen._source_binding(source),
            "max_steps": MAX_STEPS,
        },
        "reference": reference,
        "reference_arms": reference_arms,
        "runner_sha256": _sha256(__file__),
        "qualification": "not_a_qualification; off_policy gate-factor control",
    }


def _validate_preflight(payload: Mapping[str, Any], reference_root: Path) -> None:
    _require(payload.get("stage") == STAGE, "gate-control preflight stage mismatch")
    _require(payload.get("status") == "PREFLIGHT_OK", "gate-control preflight is not successful")
    for key, expected in (("PPO_ALLOWED", False), ("heldout_used", False),
                          ("teacher_used", False), ("optimizer_used", False),
                          ("off_policy", True), ("no_logprob", True), ("no_training", True)):
        _require(payload.get(key) is expected, f"gate-control flag {key} changed")
    _require(payload.get("runner_sha256") == _sha256(__file__), "gate-control runner changed after preflight")
    live_preflight, config, source, layout, opportunity_path = _opportunity_inputs(reference_root)
    runtime = payload.get("runtime", {})
    _require(runtime.get("opportunity_preflight_sha256") == _sha256(opportunity_path),
             "V2 opportunity preflight changed after gate-control preflight")
    _require(runtime.get("opportunity_runner_sha256") == live_preflight.get("runner_sha256"),
             "V2 opportunity runner binding changed")
    _require(runtime.get("config_snapshot") == _config_snapshot(config),
             "gate-control config differs from V2 opportunity config")
    live_binding = frozen._source_binding(source)
    stored_binding = runtime.get("source_binding", {})
    for key in ("source_model_digest", "candidate_digest_before_warmup", "resolved_checkpoint", "warmup_checkpoint"):
        _require(_binding_equal(live_binding.get(key), stored_binding.get(key), key),
                 f"gate-control source binding changed for {key}")
    _require(int(runtime.get("max_steps", -1)) == MAX_STEPS, "gate-control max step cap changed")
    _require(payload.get("reference", {}).get("full_sha256") == layout.get("full_sha256"),
             "gate-control reference full binding changed")
    stored_arms = payload.get("reference_arms", {})
    for pair_id, metadata in layout.get("arms", {}).items():
        old = stored_arms.get(str(pair_id), {})
        for key in ("sha256", "env_seed", "keyed_policy_seed"):
            _require(old.get(key) == metadata.get(key), f"reference arm {pair_id} binding changed for {key}")
    _require(payload.get("control", {}).get("pair_ids") == [int(item["pair_id"]) for item in PAIR_SPECS],
             "gate-control pair grid changed")


def _phase_inputs(output: Path, reference_root: Path) -> tuple[
    dict[str, Any], dict[str, Any], Any, dict[str, Any], dict[str, Any]
]:
    preflight = _read_json(output / "preflight.json")
    _validate_preflight(preflight, reference_root)
    opportunity_preflight, config, source, layout, _ = _opportunity_inputs(reference_root)
    return preflight, opportunity_preflight, config, source, layout


def _control_keyed_request(policy: Any, env: Any, keyed_seed: int) -> dict[str, Any]:
    """Build the forced-gate request while preserving proposal CRN and logits."""

    caller_rng = paired._rng_snapshot()
    try:
        context = paired._details_context(policy, env)
        actions: dict[str, np.ndarray] = {}
        shortlists: dict[str, tuple[np.ndarray, ...]] = {}
        rows: list[dict[str, Any]] = []
        step = int(context.get("step", getattr(env, "step_count", 0)))
        forced_release = 0
        canonical_wait = 0
        for index, uid in enumerate(context["uids"]):
            arrival_id = frozen._arrival_id(env, uid)
            actor_mask = bool(context["actor_mask"][index])
            gate_uniform = None
            proposal_uniform = None
            legacy_gate = WAIT_GATE
            proposal: int | None = None
            shortlist_value: tuple[np.ndarray, ...] = ()
            if actor_mask:
                forced_release += 1
                gate_uniform = paired.keyed_uniform(int(keyed_seed), step, arrival_id, "gate")
                legacy_gate = paired._cdf_sample(context["gate_probabilities"][index], gate_uniform)
                proposal_uniform = paired.keyed_uniform(int(keyed_seed), step, arrival_id, "proposal")
                proposal = paired._cdf_sample(context["proposal_probabilities"][index], proposal_uniform)
                action = HierarchicalActionV42(RELEASE_GATE, proposal).normalized.copy()
                shortlist_value = paired._shortlist_from_logits(
                    int(proposal), context["proposal_logits"][index]
                )
                shortlists[str(uid)] = shortlist_value
                gate = RELEASE_GATE
            else:
                canonical_wait += 1
                action = HierarchicalActionV42(WAIT_GATE, None).normalized.copy()
                gate = WAIT_GATE
            actions[str(uid)] = action
            rows.append({
                "uid": str(uid),
                "arrival_id": int(arrival_id),
                "actor_mask": actor_mask,
                "gate": int(gate),
                "legacy_gate_sample": int(legacy_gate),
                "proposal": None if proposal is None else int(proposal),
                "gate_uniform": gate_uniform,
                "proposal_uniform": proposal_uniform,
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
                "forced_release_count": forced_release,
                "canonical_wait_count": canonical_wait,
                "proposal_q_frozen": True,
                "proposal_logits_frozen": True,
                "off_policy": True,
                "logprob_recorded": False,
            },
        }
    finally:
        paired._restore_rng(caller_rng)


def _integrity_checks(arm: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    reference_arm = reference["arm"]
    request_identity_match = arm.get("request_identity") == reference_arm.get("request_identity")
    request_count_match = (
        int(arm.get("generated_requests", -1)) == 24
        and int(arm.get("request_identity_count", -1)) == 24
        and int(reference_arm.get("generated_requests", -1)) == 24
    )
    ledger = arm.get("ledger", {})
    energy = arm.get("energy_ledger", {}).get("coverage", {})
    wait_integrity = arm.get("wait_integrity", {})
    ledger_ok = bool(ledger.get("conservation", {}).get("requested_equals_resolved_union_unresolved")) and bool(
        ledger.get("terminal_alignment", {}).get("valid")
    )
    energy_ok = energy.get("valid") is True
    wait_ok = bool(wait_integrity) and all(value is True for value in wait_integrity.values())
    model_digest_match = arm.get("model_digest_before") == reference_arm.get("model_digest_before")
    source_ok = arm.get("model_unchanged") is True and arm.get("caller_rng_unchanged") is True
    safety_ok = arm.get("safety", {}).get("safety_violation") is False
    natural = arm.get("status") == "completed"
    technical_ok = None if not natural else all((
        request_identity_match, request_count_match, ledger_ok, energy_ok, wait_ok,
        model_digest_match, source_ok, safety_ok,
    ))
    return {
        "request_identity_match": request_identity_match,
        "request_count_24": request_count_match,
        "ledger_energy_wait_integrity": ledger_ok and energy_ok and wait_ok,
        "ledger_conservation": ledger_ok,
        "energy_coverage": energy_ok,
        "wait_elapsed_integrity": wait_ok,
        "source_model_digest_match": model_digest_match,
        "source_model_unchanged": arm.get("model_unchanged") is True,
        "caller_rng_unchanged": arm.get("caller_rng_unchanged") is True,
        "hard_safety_clear": safety_ok,
        "technical_ok": technical_ok,
        "known": bool(natural),
    }


def _observed_int(mapping: Mapping[str, Any], key: str) -> int | None:
    value = mapping.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_id_list(value: Any) -> set[int] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    try:
        parsed = [int(item) for item in value]
    except (TypeError, ValueError):
        return None
    if len(parsed) != len(set(parsed)):
        return None
    return set(parsed)


def _request_elapsed_total(arm: Mapping[str, Any]) -> tuple[float | None, bool]:
    wait = arm.get("wait_metrics", {})
    values = wait.get("request_elapsed_seconds_by_arrival", {}) if isinstance(wait, Mapping) else {}
    if not isinstance(values, Mapping):
        return None, False
    observed: dict[int, float] = {}
    for key, value in values.items():
        try:
            number = float(value)
            arrival_id = int(key)
        except (TypeError, ValueError):
            return None, False
        if not math.isfinite(number) or arrival_id in observed:
            return None, False
        observed[arrival_id] = number
    requested = arm.get("ledger", {}).get("requested_ids", ())
    requested_ids = _parse_id_list(requested)
    complete = bool(requested_ids) and set(observed) == requested_ids
    if not complete:
        return None, False
    return float(sum(observed.values())), True


def _energy_total_consumed(arm: Mapping[str, Any]) -> tuple[float | None, bool]:
    """Sum the real rows schema; do not invent a total_consumed field."""

    energy = arm.get("energy_ledger", {})
    if not isinstance(energy, Mapping):
        return None, False
    rows = energy.get("rows")
    if isinstance(rows, Mapping):
        raw_rows = list(rows.values())
    elif isinstance(rows, Sequence) and not isinstance(rows, (str, bytes, bytearray)):
        raw_rows = list(rows)
    else:
        return None, False
    coverage = energy.get("coverage", {})
    if not isinstance(coverage, Mapping):
        return None, False
    ledger_requested_ids = _parse_id_list(arm.get("ledger", {}).get("requested_ids", ()))
    coverage_requested_ids = _parse_id_list(coverage.get("requested_ids"))
    if not ledger_requested_ids or coverage_requested_ids != ledger_requested_ids:
        return None, False
    unique_ids = _parse_id_list(energy.get("unique_arrival_ids"))
    if unique_ids is None:
        return None, False
    requested_ids = ledger_requested_ids
    seen: set[int] = set()
    total = 0.0
    for row in raw_rows:
        if not isinstance(row, Mapping):
            return None, False
        try:
            arrival_id = int(row["arrival_id"])
            value = float(row["energy_consumed"])
        except (KeyError, TypeError, ValueError):
            return None, False
        if arrival_id in seen or not math.isfinite(value):
            return None, False
        seen.add(arrival_id)
        total += value
    valid_coverage = coverage.get("valid") is True
    complete = bool(valid_coverage and seen == requested_ids and unique_ids == seen)
    if not complete:
        return None, False
    return float(total), True


def _business_metrics(arm: Mapping[str, Any]) -> dict[str, Any]:
    outcomes = arm.get("outcome_counts", {})
    wait = arm.get("wait_metrics", {})
    natural = arm.get("status") == "completed"
    request_elapsed_total, request_elapsed_complete = _request_elapsed_total(arm)
    energy_total, energy_complete = _energy_total_consumed(arm)
    wait_total = wait.get("all_request_cumulative_wait_seconds")
    wait_complete = bool(
        isinstance(wait_total, (int, float))
        and math.isfinite(float(wait_total))
        and isinstance(arm.get("wait_integrity"), Mapping)
        and all(value is True for value in arm["wait_integrity"].values())
    )
    return {
        "status": arm.get("status"),
        "normal": _observed_int(outcomes, "normal"),
        "emergency": _observed_int(outcomes, "emergency"),
        "failed": _observed_int(outcomes, "failed"),
        "diverted": _observed_int(outcomes, "diverted"),
        "unresolved": _observed_int(outcomes, "unresolved"),
        "generated": _observed_int(arm, "generated_requests"),
        "admitted": _observed_int(arm, "admitted_count"),
        "completed": _observed_int(arm, "completed_count"),
        "resolved": _observed_int(arm, "resolved_count"),
        "wait_seconds": wait_total,
        "wait_total_complete": wait_complete,
        "makespan_seconds": arm.get("simulation_seconds"),
        "request_elapsed_total_seconds": request_elapsed_total,
        "energy_total_consumed": energy_total,
        "request_elapsed_total_complete": request_elapsed_complete,
        "energy_total_consumed_field": "arm.energy_ledger.rows[*].energy_consumed",
        "energy_total_consumed_observed": energy_total is not None,
        "energy_total_consumed_complete": energy_complete,
        "reward_total": arm.get("reward_total"),
        "raw_reward_excluding_rejection_penalty": arm.get("raw_reward_excluding_rejection_penalty"),
        "release_rejection_penalty": arm.get("release_rejection_penalty"),
        "complete_scene_metrics_known": bool(natural),
    }


def _paired_business_delta(control: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    known = bool(
        control.get("complete_scene_metrics_known")
        and reference.get("complete_scene_metrics_known")
        and control.get("wait_total_complete") is True
        and reference.get("wait_total_complete") is True
        and control.get("request_elapsed_total_complete") is True
        and reference.get("request_elapsed_total_complete") is True
        and control.get("energy_total_consumed_complete") is True
        and reference.get("energy_total_consumed_complete") is True
    )
    if not known:
        return {
            "complete_scene_delta_known": False,
            "normal": None,
            "wait_seconds": None,
            "makespan_seconds": None,
            "request_elapsed_total_seconds": None,
            "energy_total_consumed": None,
            "resolved": None,
            "raw_reward_excluding_rejection_penalty": None,
            "release_rejection_penalty": None,
        }

    def delta(key: str) -> float | int | None:
        left, right = control.get(key), reference.get(key)
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            return None
        value = float(left) - float(right)
        return int(value) if key in {"normal", "resolved"} and value.is_integer() else value

    return {
        "complete_scene_delta_known": True,
        "normal": delta("normal"),
        "wait_seconds": delta("wait_seconds"),
        "makespan_seconds": delta("makespan_seconds"),
        "request_elapsed_total_seconds": delta("request_elapsed_total_seconds"),
        "energy_total_consumed": delta("energy_total_consumed"),
        "resolved": delta("resolved"),
        "raw_reward_excluding_rejection_penalty": delta("raw_reward_excluding_rejection_penalty"),
        "release_rejection_penalty": delta("release_rejection_penalty"),
    }


def _issue_rows(arm: Mapping[str, Any], category: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    outcomes = arm.get("ledger", {}).get("outcomes", {})
    for arrival_id, outcome in outcomes.items() if isinstance(outcomes, Mapping) else ():
        actual = str(getattr(outcome, "value", outcome))
        if frozen._terminal_category(actual) == category:
            rows.append({"arrival_id": int(arrival_id), "outcome": actual})
    if category == "unresolved":
        known = {int(row["arrival_id"]) for row in rows}
        for arrival_id in arm.get("ledger", {}).get("unresolved_ids", ()):
            if int(arrival_id) not in known:
                rows.append({"arrival_id": int(arrival_id), "outcome": "unresolved"})
    return sorted(rows, key=lambda item: item["arrival_id"])


def _business_review(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(len(summaries) == len(PAIR_SPECS), "business review requires all nine fixed pair summaries")
    control = [item.get("control_metrics", {}) for item in summaries]
    reference = [item.get("reference_metrics", {}) for item in summaries]
    deltas = [item.get("complete_scene_delta", {}) for item in summaries]
    known = all(
        item.get("status") == "completed"
        and item.get("complete_scene_metrics_known") is True
        and delta.get("complete_scene_delta_known") is True
        for item, delta in zip(control, deltas)
    )
    no_new_failure = known and all(
        not item.get("new_failed_arrival_ids")
        and not item.get("new_unresolved_arrival_ids")
        and not item.get("new_diverted_arrival_ids")
        for item in summaries
    )
    normal_values = [item.get("normal") for item in control + reference]
    normal_known = all(isinstance(value, (int, float)) for value in normal_values)
    normal_not_lower = known and normal_known and sum(int(item.get("normal")) for item in control) >= sum(
        int(item.get("normal")) for item in reference
    )
    completed_values = [item.get("completed") for item in control + reference]
    completed_known = all(isinstance(value, (int, float)) for value in completed_values)
    completed_not_lower = known and completed_known and sum(int(item.get("completed")) for item in control) >= sum(
        int(item.get("completed")) for item in reference
    )
    admitted_values = [item.get("admitted") for item in control]
    admitted_all_24 = known and all(value == 24 for value in admitted_values)
    wait_values = [item.get("wait_seconds") for item in control + reference]
    wait_known = all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in wait_values)
    wait_not_higher = known and wait_known and sum(float(item["wait_seconds"]) for item in control) <= sum(
        float(item["wait_seconds"]) for item in reference
    )
    strict = False
    failed_values = [item.get("failed") for item in control + reference]
    failed_known = all(isinstance(value, (int, float)) for value in failed_values)
    request_elapsed_values = [item.get("request_elapsed_total_seconds") for item in control + reference]
    request_elapsed_known = all(
        isinstance(value, (int, float)) and math.isfinite(float(value)) for value in request_elapsed_values
    )
    energy_values = [item.get("energy_total_consumed") for item in control + reference]
    energy_known = all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in energy_values)
    if known and wait_known and normal_known and failed_known:
        strict = (
            sum(int(item.get("normal")) for item in control) > sum(int(item.get("normal")) for item in reference)
            or sum(int(item.get("failed")) for item in control) < sum(int(item.get("failed")) for item in reference)
            or sum(float(item["wait_seconds"]) for item in control) < sum(float(item["wait_seconds"]) for item in reference)
            or (
                request_elapsed_known
                and sum(float(item["request_elapsed_total_seconds"]) for item in control)
                < sum(float(item["request_elapsed_total_seconds"]) for item in reference)
            )
            or (
                energy_known
                and sum(float(item["energy_total_consumed"]) for item in control)
                < sum(float(item["energy_total_consumed"]) for item in reference)
            )
        )
    def aggregate(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        complete = all(item.get("complete_scene_metrics_known") is True for item in items)

        def total(key: str) -> float | int | None:
            values = [item.get(key) for item in items]
            if not complete or not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values):
                return None
            return float(sum(float(value) for value in values))

        return {
            "complete_scene_aggregate_known": complete,
            "total_normal": total("normal"),
            "total_completed_successful": total("completed"),
            "total_failed": total("failed"),
            "total_admitted": total("admitted"),
            "total_resolved": total("resolved"),
            "total_wait_seconds": total("wait_seconds"),
            "total_request_elapsed_seconds": total("request_elapsed_total_seconds"),
            "total_energy_consumed": total("energy_total_consumed"),
            "makespan_seconds_by_pair": [item.get("makespan_seconds") for item in items],
        }

    return {
        "pair_count": len(summaries),
        "all_nine_natural": known,
        "no_new_failure_or_unresolved": no_new_failure,
        "normal_total_not_lower": normal_not_lower,
        "completed_successful_total_not_lower": completed_not_lower,
        "admitted_each_scene_24": admitted_all_24,
        "wait_total_not_higher": wait_not_higher,
        "strict_business_improvement": strict,
        "candidate_review_pass": bool(
            no_new_failure and normal_not_lower and completed_not_lower
            and admitted_all_24 and wait_not_higher and strict
        ),
        "control_aggregate_metrics": aggregate(control),
        "reference_aggregate_metrics": aggregate(reference),
        "formal_qualification_required": True,
    }


def _run_control_arm(source: Mapping[str, Any], config: Any, spec: Mapping[str, Any],
                     reference: Mapping[str, Any], progress: Any, output: Path) -> dict[str, Any]:
    policy = copy.deepcopy(source["policy"])
    original_keyed = frozen._keyed_request
    counts = {"requests": 0, "forced_release": 0, "canonical_wait": 0}

    def wrapped_keyed(current_policy: Any, env: Any, keyed_seed: int) -> dict[str, Any]:
        request = _control_keyed_request(current_policy, env, keyed_seed)
        counts["requests"] += 1
        control = request.get("control", {})
        counts["forced_release"] += int(control.get("forced_release_count", 0))
        counts["canonical_wait"] += int(control.get("canonical_wait_count", 0))
        return request

    frozen._keyed_request = wrapped_keyed
    try:
        arm = frozen._run_arm(
            policy, config, env_seed=int(spec["env_seed"]), keyed_seed=int(spec["keyed_policy_seed"]),
            mode="legacy", progress=progress, pair_id=int(spec["pair_id"]),
        )
    finally:
        frozen._keyed_request = original_keyed
    arm["control"] = {
        "name": WINDOWLESS_CONTROL,
        "off_policy": OFF_POLICY,
        "logprob_recorded": False,
        "training": False,
        **counts,
    }
    arm["control_metrics"] = _business_metrics(arm)
    checks = _integrity_checks(arm, reference)
    arm["reference_control_checks"] = checks
    if arm.get("status") == "completed" and checks.get("technical_ok") is not True:
        arm["status"] = "technical_error"
        arm["error"] = {
            "type": "ControlReferenceIntegrityError",
            "message": "control arm failed fixed reference/integrity checks",
            "checks": checks,
        }
    arm["control_metrics"] = _business_metrics(arm)
    return arm


def _run_pair(source: Mapping[str, Any], config: Any, spec: Mapping[str, Any],
              reference: Mapping[str, Any], output: Path) -> dict[str, Any]:
    pair_id = int(spec["pair_id"])

    def progress(row: Mapping[str, Any]) -> None:
        _write_json(output / "progress.json", {
            "stage": STAGE, "phase": "arm", "status": "RUNNING", **dict(row),
        })

    try:
        arm = _run_control_arm(source, config, spec, reference, progress, output)
    except Exception as exc:
        arm = {
            "mode": "legacy", "env_seed": int(spec["env_seed"]),
            "keyed_policy_seed": int(spec["keyed_policy_seed"]),
            "status": "technical_error", "censored": None,
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "reference_control_checks": {"known": False, "technical_ok": False},
            "control_metrics": {"status": "technical_error"},
        }
    _write_json(output / "arms" / f"pair_{pair_id:04d}_control.json", arm, refuse_existing=True)
    status = str(arm.get("status"))
    if status in {"error", "technical_error", "safety_violation"}:
        pair_status = "PAIR_TECHNICAL_ERROR"
    elif status == "censored":
        pair_status = "PAIR_CENSORED"
    else:
        pair_status = "PAIR_OK"
    reference_arm = reference["arm"]
    control_metrics = _business_metrics(arm)
    reference_metrics = _business_metrics(reference_arm)
    failed_rows = _issue_rows(arm, "failed")
    unresolved_rows = _issue_rows(arm, "unresolved")
    diverted_rows = _issue_rows(arm, "diverted")
    reference_failed_rows = _issue_rows(reference_arm, "failed")
    reference_unresolved_rows = _issue_rows(reference_arm, "unresolved")
    reference_diverted_rows = _issue_rows(reference_arm, "diverted")
    pair = {
        "stage": STAGE,
        "pair": dict(spec),
        "status": pair_status,
        "arm_path": str(output / "arms" / f"pair_{pair_id:04d}_control.json"),
        "reference": dict(reference["metadata"]),
        "reference_control_checks": arm.get("reference_control_checks", {}),
        "control_metrics": control_metrics,
        "reference_metrics": reference_metrics,
        "complete_scene_delta": _paired_business_delta(control_metrics, reference_metrics),
        "failure_comparison_known": bool(
            control_metrics.get("complete_scene_metrics_known")
            and reference_metrics.get("complete_scene_metrics_known")
        ),
        "control_failed_rows": failed_rows,
        "control_unresolved_rows": unresolved_rows,
        "control_diverted_rows": diverted_rows,
        "reference_failed_rows": reference_failed_rows,
        "reference_unresolved_rows": reference_unresolved_rows,
        "reference_diverted_rows": reference_diverted_rows,
        "new_failed_arrival_ids": (
            None if not control_metrics.get("complete_scene_metrics_known")
            else [int(row["arrival_id"]) for row in failed_rows if int(row["arrival_id"]) not in {
                int(item["arrival_id"]) for item in reference_failed_rows
            }]
        ),
        "new_unresolved_arrival_ids": (
            None if not control_metrics.get("complete_scene_metrics_known")
            else [int(row["arrival_id"]) for row in unresolved_rows if int(row["arrival_id"]) not in {
                int(item["arrival_id"]) for item in reference_unresolved_rows
            }]
        ),
        "new_diverted_arrival_ids": (
            None if not control_metrics.get("complete_scene_metrics_known")
            else [int(row["arrival_id"]) for row in diverted_rows if int(row["arrival_id"]) not in {
                int(item["arrival_id"]) for item in reference_diverted_rows
            }]
        ),
        "raw_reward_excluding_rejection_penalty": arm.get("raw_reward_excluding_rejection_penalty"),
        "release_rejection_penalty": arm.get("release_rejection_penalty"),
        "off_policy": OFF_POLICY,
        "logprob_recorded": False,
        "training": False,
    }
    _write_json(output / "pairs" / f"pair_{pair_id:04d}.json", pair, refuse_existing=True)
    return {"summary": pair, "arm": arm}


def run_preflight(output: Path, reference_root: Path = REFERENCE_ROOT) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    payload = _build_preflight(reference_root)
    _write_json(output / "preflight.json", payload, refuse_existing=True)
    _write_json(output / "progress.json", {
        "stage": STAGE, "phase": "preflight", "status": "PREFLIGHT_OK",
        "environment_created": False, "off_policy": OFF_POLICY,
    })
    return payload


def _validate_pilot_reuse(pilot: Mapping[str, Any], preflight: Mapping[str, Any],
                          output: Path) -> None:
    _require(pilot.get("status") in {"PILOT_OK", "PILOT_CENSORED"},
             "full phase requires a retained pilot result")
    expected = PAIR_SPECS[0]
    summary = pilot.get("pair", {})
    _require(summary.get("pair") == expected, "pilot pair does not match pair 1")
    _require(summary.get("status") in {"PAIR_OK", "PAIR_CENSORED"},
             "pilot pair is not reusable")
    _require(pilot.get("preflight_runner_sha256") == preflight.get("runner_sha256"),
             "pilot preflight runner binding changed")
    stored = _read_json(output / "pairs" / "pair_0001.json")
    arm = _read_json(output / "arms" / "pair_0001_control.json")
    _require(stored.get("status") in {"PAIR_OK", "PAIR_CENSORED"} and stored.get("pair") == expected,
             "stored pilot pair artifact changed")
    _require(arm.get("status") in {"completed", "censored"}, "stored pilot arm has no retained result")
    _require(arm.get("control", {}).get("off_policy") is True, "stored pilot arm is not off-policy control")
    if arm.get("status") == "completed":
        _require(arm.get("reference_control_checks", {}).get("technical_ok") is True,
                 "stored pilot arm integrity check is incomplete")
    else:
        _require(arm.get("reference_control_checks", {}).get("technical_ok") is None,
                 "stored censored pilot integrity state is not unknown")
    for key in ("control_metrics", "reference_metrics", "reference_control_checks"):
        _require(stored.get(key) == summary.get(key), f"pilot summary differs from stored pair for {key}")


def run_pilot(output: Path, reference_root: Path = REFERENCE_ROOT) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    preflight, _op_preflight, config, source, layout = _phase_inputs(output, reference_root)
    spec = PAIR_SPECS[0]
    reference = layout["_arms"][str(spec["pair_id"])]
    reference["metadata"] = layout["arms"][str(spec["pair_id"])]
    result = _run_pair(source, config, spec, reference, output)
    pair_status = result["summary"]["status"]
    status = (
        "PILOT_OK" if pair_status == "PAIR_OK"
        else "PILOT_CENSORED" if pair_status == "PAIR_CENSORED"
        else "PILOT_TECHNICAL_ERROR"
    )
    payload = {
        "stage": STAGE, "phase": "pilot", "status": status,
        "PPO_ALLOWED": PPO_ALLOWED, "off_policy": OFF_POLICY,
        "pair": result["summary"],
        "preflight_runner_sha256": preflight["runner_sha256"],
        "qualification": "not_a_qualification",
    }
    _write_json(output / "pilot.json", payload, refuse_existing=True)
    _write_json(output / "progress.json", payload)
    if status == "PILOT_TECHNICAL_ERROR":
        raise GateReleaseControlError("pilot stopped after technical or safety failure")
    return payload


def run_full(output: Path, reference_root: Path = REFERENCE_ROOT) -> dict[str, Any]:
    pilot = _read_json(output / "pilot.json")
    preflight, _op_preflight, config, source, layout = _phase_inputs(output, reference_root)
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
                "PPO_ALLOWED": PPO_ALLOWED, "off_policy": OFF_POLICY,
                "pair_summaries": summaries,
            }
            _write_json(output / "full.json", payload, refuse_existing=True)
            _write_json(output / "progress.json", payload)
            raise GateReleaseControlError("full phase stopped after technical or safety failure")
    payload = {
        "stage": STAGE, "phase": "full", "status": "FULL_OK",
        "PPO_ALLOWED": PPO_ALLOWED, "off_policy": OFF_POLICY,
        "pair_count": len(summaries), "pair_summaries": summaries,
        "business_review": _business_review(summaries),
        "qualification": "not_a_qualification; fixed off-policy gate control",
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
