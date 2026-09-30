"""Run the balanced 24-pair proposal-support comparison.

The nine already validated proposal-support control pairs are imported from
their immutable run.  This entry point runs only the remaining fifteen pairs
from the registered six-environment by four-keyed-seed grid.  It is frozen
off-policy, descriptive, and does not train or qualify P3.
"""
from __future__ import annotations

import argparse
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from terminal_marl.training import run_p3_forced_wait_frozen_20260930 as frozen
from terminal_marl.training import run_p3_proposal_support_control_20261001 as support


STAGE = "P3_PROPOSAL_SUPPORT_BALANCED_20261001_V1"
PPO_ALLOWED = False
HELDOUT_USED = False
TEACHER_USED = False
OPTIMIZER_USED = False
OFF_POLICY = True
REFERENCE_ROOT = Path("/home/user/training_outputs/terminal/p3_followup_20260930/forced_wait_frozen")
V1_ROOT = Path("/home/user/training_outputs/terminal/p3_followup_20260930/proposal_support_control_20261001")
OUTPUT_DIR = Path("/home/user/training_outputs/terminal/p3_followup_20260930/proposal_support_balanced_20261001")
V1_FULL_SHA256 = "0178f6cef9083babab207d1fbd0e71cc50b4ebd90ffc2e4601caaeb07a8668b5"
PAIR_SPECS = tuple(dict(item) for item in frozen.PAIR_SPECS)
IMPORTED_PAIR_IDS = frozenset(int(item["pair_id"]) for item in support.PAIR_SPECS)
EXPECTED_CONTROL = {
    "name": support.CONTROL_NAME, "epsilon": support.EPSILON,
    "q_floor_min": support.Q_FLOOR_MIN, "proposal_count": support.PROPOSAL_COUNT,
    "sampled_first": True, "off_policy": True, "logprob_recorded": False,
    "training": False,
}
ENERGY_ROUNDTRIP_ABS_TOL = 1e-8


class BalancedProposalSupportError(ValueError):
    """Raised when a fixed reference or balanced-run contract changes."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BalancedProposalSupportError(message)


def _metrics_equal_with_energy_roundtrip(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    """Compare immutable metrics, allowing only JSON energy-sum roundoff."""

    if set(actual) != set(expected):
        return False
    for key in actual:
        left, right = actual[key], expected[key]
        if key == "energy_total_consumed":
            numeric = lambda value: isinstance(value, (int, float)) and not isinstance(value, bool)
            if numeric(left) and numeric(right) and math.isfinite(float(left)) and math.isfinite(float(right)):
                if not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=ENERGY_ROUNDTRIP_ABS_TOL):
                    return False
                continue
        if left != right:
            return False
    return True


def _validate_v1_arm(summary: Mapping[str, Any], arm: Mapping[str, Any], pair_id: int) -> None:
    _require(summary.get("status") == "PAIR_OK" and arm.get("status") == "completed",
             f"proposal-support v1 pair {pair_id} is not natural and complete")
    _require(arm.get("reference_control_checks", {}).get("technical_ok") is True,
             f"proposal-support v1 pair {pair_id} technical check is not PASS")
    _require(_metrics_equal_with_energy_roundtrip(
        support.gate._business_metrics(arm), summary.get("control_metrics", {})),
             f"proposal-support v1 pair {pair_id} arm metrics differ from immutable summary")
    _require(arm.get("reference_control_checks") == summary.get("reference_control_checks"),
             f"proposal-support v1 pair {pair_id} integrity checks differ from immutable summary")
    _require(arm.get("proposal_support_control") == EXPECTED_CONTROL,
             f"proposal-support v1 pair {pair_id} proposal-support parameters changed")


def _read_json(path: str | Path) -> dict[str, Any]:
    return support._read_json(path)


def _write_json(path: Path, value: Mapping[str, Any], *, refuse_existing: bool = False) -> None:
    support._write_json(path, value, refuse_existing=refuse_existing)


def _sha256(path: str | Path) -> str:
    return support._sha256(path)


def _reference_arms(reference_root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    root = Path(reference_root)
    full_path = root / "full.json"
    full = _read_json(full_path)
    _require(full.get("status") == "FULL_OK", "24-pair legacy reference is not FULL_OK")
    _require(int(full.get("pair_count", -1)) == 24, "legacy reference pair count is not 24")
    _require([int(value) for value in full.get("pair_ids", ())] == list(range(1, 25)),
             "legacy reference pair IDs are not 1..24")
    arms: dict[str, dict[str, Any]] = {}
    for spec in PAIR_SPECS:
        pair_id = int(spec["pair_id"])
        path = root / "arms" / f"pair_{pair_id:04d}_legacy.json"
        arm = _read_json(path)
        _require(arm.get("status") == "completed" and arm.get("censored") is False,
                 f"legacy reference arm {pair_id} is not natural")
        _require(int(arm.get("env_seed", -1)) == int(spec["env_seed"]),
                 f"legacy reference environment seed mismatch for pair {pair_id}")
        _require(int(arm.get("keyed_policy_seed", -1)) == int(spec["keyed_policy_seed"]),
                 f"legacy reference keyed seed mismatch for pair {pair_id}")
        _require((full.get("pair_statuses") or {}).get(str(pair_id), {}).get("legacy") == "completed",
                 f"legacy reference status mismatch for pair {pair_id}")
        ledger = arm.get("ledger", {})
        energy = arm.get("energy_ledger", {}).get("coverage", {})
        wait = arm.get("wait_integrity", {})
        _require(int(arm.get("generated_requests", -1)) == 24 and int(arm.get("admitted_count", -1)) == 24,
                 f"legacy reference request denominator mismatch for pair {pair_id}")
        _require(ledger.get("conservation", {}).get("requested_equals_resolved_union_unresolved") is True,
                 f"legacy reference ledger mismatch for pair {pair_id}")
        _require(ledger.get("terminal_alignment", {}).get("valid") is True,
                 f"legacy reference terminal alignment mismatch for pair {pair_id}")
        _require(energy.get("valid") is True and all(value is True for value in wait.values()),
                 f"legacy reference energy/wait integrity mismatch for pair {pair_id}")
        _require(arm.get("model_unchanged") is True and arm.get("caller_rng_unchanged") is True,
                 f"legacy reference source/RNG mismatch for pair {pair_id}")
        _require(arm.get("safety", {}).get("safety_violation") is False,
                 f"legacy reference safety mismatch for pair {pair_id}")
        metadata = {
            "path": str(path), "sha256": _sha256(path),
            "env_seed": int(spec["env_seed"]),
            "keyed_policy_seed": int(spec["keyed_policy_seed"]),
        }
        arms[str(pair_id)] = {"arm": arm, "metadata": metadata}
    return {
        "full_sha256": _sha256(full_path), "full_path": str(full_path),
        "pair_count": 24, "pair_ids": list(range(1, 25)),
        "arms": {key: value["metadata"] for key, value in arms.items()},
    }, arms


def _v1_guard(reference_root: Path) -> tuple[dict[str, Any], Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    preflight, config, source, _layout = support._phase_inputs(V1_ROOT, reference_root)
    pilot = _read_json(V1_ROOT / "pilot.json")
    support._validate_pilot_reuse(pilot, preflight, V1_ROOT)
    full_path = V1_ROOT / "full.json"
    full = _read_json(full_path)
    _require(_sha256(full_path) == V1_FULL_SHA256, "proposal-support v1 full artifact SHA changed")
    _require(full.get("status") == "FULL_OK" and int(full.get("pair_count", -1)) == 9,
             "proposal-support v1 full artifact is not the registered nine-pair result")
    _require([int(item.get("pair", {}).get("pair_id", -1)) for item in full.get("pair_summaries", ())]
             == [int(item["pair_id"]) for item in support.PAIR_SPECS],
             "proposal-support v1 pair order changed")
    pair_artifacts: dict[str, Any] = {}
    for summary in full["pair_summaries"]:
        pair_id = int(summary["pair"]["pair_id"])
        pair_path = V1_ROOT / "pairs" / f"pair_{pair_id:04d}.json"
        arm_path = V1_ROOT / "arms" / f"pair_{pair_id:04d}_support.json"
        pair = _read_json(pair_path)
        arm = _read_json(arm_path)
        _require(pair == summary, f"proposal-support v1 pair {pair_id} summary changed")
        _validate_v1_arm(summary, arm, pair_id)
        pair_artifacts[str(pair_id)] = {
            "pair_sha256": _sha256(pair_path), "arm_sha256": _sha256(arm_path),
            "pair_path": str(pair_path), "arm_path": str(arm_path),
        }
    guard = {
        "root": str(V1_ROOT), "full_sha256": _sha256(full_path),
        "preflight_sha256": _sha256(V1_ROOT / "preflight.json"),
        "pilot_sha256": _sha256(V1_ROOT / "pilot.json"),
        "runner_sha256": preflight.get("runner_sha256"),
        "runtime": preflight.get("runtime"),
        "source_binding": preflight.get("runtime", {}).get("source_binding"),
        "config_snapshot": preflight.get("runtime", {}).get("config_snapshot"),
        "pair_artifacts": pair_artifacts,
    }
    return guard, config, source, preflight, full


def _build_preflight(reference_root: Path = REFERENCE_ROOT) -> dict[str, Any]:
    v1, config, _source, v1_preflight, _full = _v1_guard(reference_root)
    reference, _arms = _reference_arms(reference_root)
    return {
        "stage": STAGE, "phase": "preflight", "status": "PREFLIGHT_OK",
        "PPO_ALLOWED": PPO_ALLOWED, "heldout_used": HELDOUT_USED,
        "teacher_used": TEACHER_USED, "optimizer_used": OPTIMIZER_USED,
        "off_policy": OFF_POLICY,
        "runner_sha256": _sha256(__file__),
        "pair_grid": [dict(item) for item in PAIR_SPECS],
        "imported_pair_ids": sorted(IMPORTED_PAIR_IDS),
        "proposal_support": {
            "epsilon": support.EPSILON, "q_floor_min": support.Q_FLOOR_MIN,
            "proposal_count": support.PROPOSAL_COUNT, "sampled_first": True,
        },
        "v1_guard": v1,
        "reference": reference,
        "runtime": v1_preflight.get("runtime"),
        "source_binding": v1_preflight.get("runtime", {}).get("source_binding"),
        "config_snapshot": v1_preflight.get("runtime", {}).get("config_snapshot"),
        "qualification": "not_a_qualification; balanced descriptive control",
    }


def _validate_preflight(payload: Mapping[str, Any], reference_root: Path) -> None:
    _require(payload.get("stage") == STAGE and payload.get("status") == "PREFLIGHT_OK",
             "balanced preflight is not successful")
    for key, expected in (("PPO_ALLOWED", False), ("heldout_used", False), ("teacher_used", False),
                          ("optimizer_used", False), ("off_policy", True)):
        _require(payload.get(key) is expected, f"balanced flag {key} changed")
    _require(payload.get("runner_sha256") == _sha256(__file__), "balanced runner changed after preflight")
    _require(payload.get("pair_grid") == [dict(item) for item in PAIR_SPECS], "balanced pair grid changed")
    _require(payload.get("imported_pair_ids") == sorted(IMPORTED_PAIR_IDS), "imported pair set changed")
    _require(payload.get("proposal_support", {}).get("epsilon") == support.EPSILON,
             "proposal epsilon changed")
    current_v1, _config, _source, current_preflight, _full = _v1_guard(reference_root)
    _require(payload.get("v1_guard") == current_v1, "proposal-support v1 immutable guard changed")
    _require(payload.get("runtime") == current_preflight.get("runtime"), "balanced runtime binding changed")
    _require(payload.get("source_binding") == current_preflight.get("runtime", {}).get("source_binding"),
             "balanced source binding changed")
    _require(payload.get("config_snapshot") == current_preflight.get("runtime", {}).get("config_snapshot"),
             "balanced config binding changed")
    current_reference, _arms = _reference_arms(reference_root)
    _require(payload.get("reference") == current_reference, "legacy reference binding changed")


def _phase_inputs(output: Path, reference_root: Path):
    preflight = _read_json(output / "preflight.json")
    _validate_preflight(preflight, reference_root)
    v1, config, source, _v1_preflight, full = _v1_guard(reference_root)
    _reference, arms = _reference_arms(reference_root)
    return preflight, config, source, v1, full, arms


def _finite_sum(values: Sequence[Any]) -> float | None:
    if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values):
        return None
    return float(sum(float(value) for value in values))


def _aggregate24(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(len(summaries) == 24, "balanced aggregate requires all 24 pair summaries")
    pair_ids = [int(item.get("pair", {}).get("pair_id", -1)) for item in summaries]
    _require(len(set(pair_ids)) == 24 and sorted(pair_ids) == list(range(1, 25)),
             "balanced aggregate requires unique pair IDs 1..24")
    control = [item.get("control_metrics", {}) for item in summaries]
    reference = [item.get("reference_metrics", {}) for item in summaries]
    natural_known = all(
        item.get("status") == "PAIR_OK"
        and metrics.get("complete_scene_metrics_known") is True
        and item.get("complete_scene_delta", {}).get("complete_scene_delta_known") is True
        for item, metrics in zip(summaries, control)
    )
    def total(key: str) -> float | None:
        return _finite_sum([item.get(key) for item in control]) if natural_known else None
    def reference_total(key: str) -> float | None:
        return _finite_sum([item.get(key) for item in reference]) if natural_known else None
    technical_ok_all = all(
        item.get("reference_control_checks", {}).get("technical_ok") is True for item in summaries
    )
    normal = total("normal")
    normal_ref = reference_total("normal")
    completed = total("completed")
    completed_ref = reference_total("completed")
    wait = total("wait_seconds")
    wait_ref = reference_total("wait_seconds")
    failed = total("failed")
    failed_ref = reference_total("failed")
    elapsed = total("request_elapsed_total_seconds")
    elapsed_ref = reference_total("request_elapsed_total_seconds")
    energy = total("energy_total_consumed")
    energy_ref = reference_total("energy_total_consumed")
    extra_keys = ("emergency", "diverted", "unresolved", "generated", "admitted", "resolved",
                  "raw_reward_excluding_rejection_penalty", "release_rejection_penalty")
    extra_totals = {key: total(key) for key in extra_keys}
    extra_reference_totals = {key: reference_total(key) for key in extra_keys}
    issue_fields = ("new_failed_arrival_ids", "new_unresolved_arrival_ids", "new_diverted_arrival_ids")
    issue_lists_known = all(isinstance(item.get(key), list) for item in summaries for key in issue_fields)
    no_new = None if not natural_known else (
        all(not item.get(key) for item in summaries for key in issue_fields)
        if issue_lists_known else False
    )
    hard_safe = None if not natural_known else all(
        item.get("reference_control_checks", {}).get("hard_safety_clear") is True for item in summaries
    )
    normal_ok = None if normal is None or normal_ref is None else normal >= normal_ref
    completed_ok = None if completed is None or completed_ref is None else completed >= completed_ref
    wait_ok = None if wait is None or wait_ref is None else wait <= wait_ref
    strict = None if not natural_known else bool(
        (normal is not None and normal_ref is not None and normal > normal_ref)
        or (failed is not None and failed_ref is not None and failed < failed_ref)
        or (wait is not None and wait_ref is not None and wait < wait_ref)
        or (elapsed is not None and elapsed_ref is not None and elapsed < elapsed_ref)
        or (energy is not None and energy_ref is not None and energy < energy_ref)
    )
    admitted = None if not natural_known else all(item.get("admitted") == 24 for item in control)
    criterion = all(value is True for value in (natural_known, technical_ok_all, no_new, hard_safe,
                                                admitted, normal_ok, completed_ok, wait_ok, strict))
    return {
        "pair_count": 24, "all_24_natural": natural_known,
        "technical_checks_all_pass": technical_ok_all,
        "hard_safety_clear_all": hard_safe, "admitted_each_scene_24": admitted,
        "new_issue_lists_known": None if not natural_known else issue_lists_known,
        "no_new_failure_or_unresolved_or_diverted": no_new,
        "normal_total": normal, "reference_normal_total": normal_ref,
        "completed_total": completed, "reference_completed_total": completed_ref,
        "failed_total": failed, "reference_failed_total": failed_ref,
        "wait_total_seconds": wait, "reference_wait_total_seconds": wait_ref,
        "request_elapsed_total_seconds": elapsed, "reference_request_elapsed_total_seconds": elapsed_ref,
        "energy_total_consumed": energy, "reference_energy_total_consumed": energy_ref,
        **{f"{key}_total": value for key, value in extra_totals.items()},
        **{f"reference_{key}_total": value for key, value in extra_reference_totals.items()},
        "normal_not_lower": normal_ok, "successful_not_lower": completed_ok,
        "wait_not_higher": wait_ok, "strict_business_gain": strict,
        "criterion_pass": criterion, "formal_qualification_required": True,
    }


def _profiles(summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    expected: dict[int, dict[str, Any]] = {}
    for spec in PAIR_SPECS:
        expected.setdefault(int(spec["env_seed"]), {"pair_ids": [], "keyed_policy_seeds": []})
        expected[int(spec["env_seed"])]["pair_ids"].append(int(spec["pair_id"]))
        expected[int(spec["env_seed"])]["keyed_policy_seeds"].append(int(spec["keyed_policy_seed"]))
    result = []
    observed_envs = {int(item["pair"]["env_seed"]) for item in summaries}
    _require(observed_envs == set(expected), "balanced profiles require all six environment seeds")
    for env_seed in sorted(expected):
        rows = [item for item in summaries if int(item["pair"]["env_seed"]) == env_seed]
        pair_ids = [int(item["pair"]["pair_id"]) for item in rows]
        keyed_seeds = [int(item["pair"]["keyed_policy_seed"]) for item in rows]
        _require(len(rows) == 4 and len(set(pair_ids)) == 4 and len(set(keyed_seeds)) == 4,
                 f"balanced profile {env_seed} requires four distinct keyed pairs")
        _require(sorted(pair_ids) == sorted(expected[env_seed]["pair_ids"]),
                 f"balanced profile {env_seed} pair IDs changed")
        _require(sorted(keyed_seeds) == sorted(expected[env_seed]["keyed_policy_seeds"]),
                 f"balanced profile {env_seed} keyed seeds changed")
        complete = all(item.get("control_metrics", {}).get("complete_scene_metrics_known") is True for item in rows)
        control_normal = _finite_sum([item.get("control_metrics", {}).get("normal") for item in rows]) if complete else None
        control_wait = _finite_sum([item.get("control_metrics", {}).get("wait_seconds") for item in rows]) if complete else None
        reference_normal = _finite_sum([item.get("reference_metrics", {}).get("normal") for item in rows]) if complete else None
        reference_wait = _finite_sum([item.get("reference_metrics", {}).get("wait_seconds") for item in rows]) if complete else None
        result.append({
            "env_seed": env_seed,
            "pair_ids": pair_ids,
            "keyed_policy_seeds": keyed_seeds,
            "pair_count": len(rows),
            "descriptive_only": True,
            "metrics_known": complete,
            "control_normal_total": control_normal,
            "control_wait_total_seconds": control_wait,
            "reference_normal_total": reference_normal,
            "reference_wait_total_seconds": reference_wait,
        })
    return result


def run_preflight(output: Path, reference_root: Path = REFERENCE_ROOT) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    payload = _build_preflight(reference_root)
    _write_json(output / "preflight.json", payload, refuse_existing=True)
    _write_json(output / "progress.json", {
        "stage": STAGE, "phase": "preflight", "status": "PREFLIGHT_OK", "environment_created": False,
    })
    return payload


def run_full(output: Path, reference_root: Path = REFERENCE_ROOT) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    _preflight, config, source, _v1_guard_payload, v1_full, references = _phase_inputs(output, reference_root)
    imported = {int(item["pair"]["pair_id"]): dict(item) for item in v1_full["pair_summaries"]}
    summaries = [imported[pair_id] for pair_id in sorted(imported)]
    _require(set(imported) == IMPORTED_PAIR_IDS, "v1 imported pair IDs are incomplete")
    for spec in PAIR_SPECS:
        pair_id = int(spec["pair_id"])
        if pair_id in IMPORTED_PAIR_IDS:
            continue
        reference = references[str(pair_id)]
        _write_json(output / "progress.json", {
            "stage": STAGE, "phase": "arm", "status": "RUNNING", "pair_id": pair_id,
        })
        result = support._run_pair(source, config, spec, reference, output)
        summaries.append(result["summary"])
        if result["summary"].get("status") == "PAIR_TECHNICAL_ERROR":
            payload = {
                "stage": STAGE, "phase": "full", "status": "FULL_TECHNICAL_ERROR",
                "PPO_ALLOWED": PPO_ALLOWED, "off_policy": OFF_POLICY,
                "imported_pair_ids": sorted(IMPORTED_PAIR_IDS), "pair_summaries": summaries,
            }
            _write_json(output / "full.json", payload, refuse_existing=True)
            _write_json(output / "progress.json", payload)
            raise BalancedProposalSupportError("full phase stopped after technical or safety failure")
    summaries.sort(key=lambda item: int(item["pair"]["pair_id"]))
    payload = {
        "stage": STAGE, "phase": "full", "status": "FULL_OK",
        "PPO_ALLOWED": PPO_ALLOWED, "off_policy": OFF_POLICY,
        "pair_count": 24, "imported_pair_ids": sorted(IMPORTED_PAIR_IDS),
        "pair_summaries": summaries, "profiles": _profiles(summaries),
        "balanced_business_review": _aggregate24(summaries),
        "qualification": "not_a_qualification; balanced descriptive proposal-support control",
    }
    _write_json(output / "full.json", payload, refuse_existing=True)
    _write_json(output / "progress.json", payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "full"), required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--reference-root", type=Path, default=REFERENCE_ROOT)
    args = parser.parse_args(argv)
    try:
        if args.phase == "preflight":
            run_preflight(args.output_dir, args.reference_root)
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
