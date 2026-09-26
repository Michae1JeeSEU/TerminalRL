"""Bounded WAIT/RELEASE action-credit diagnostic for existing P3 tapes.

The diagnostic reads the already-collected proposal-head rows, replays the
corresponding action tapes, and evaluates one-step branches from a deepcopy of
each selected pre-action state.  It does not sample a policy or update model
weights.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from terminal_marl.core.safety_trace import SafetyTraceRecorder
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training import collect_p3_proposal_head_probe_v1 as collector
from terminal_marl.training import diagnose_p3_k3_coverage_v1 as diagnostic
from terminal_marl.training import p3_action_credit
from terminal_marl.training import student_rollout_v4_2
from terminal_marl.training.k3_gate_feasibility import k3_certified_mass


FORMAL_ROOT = Path("/home/user/training_outputs/terminal/p3_adaptive_kl_formal_v7_20260926")
ROWS_PATH = Path(
    "/home/user/training_outputs/terminal/"
    "p3_proposal_head_probe_v1_20260926/collection/rows.json"
)
EXPECTED_SELECTIONS = 16
PROPOSAL_COUNT = 25


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _rng_snapshot() -> tuple[Any, tuple[Any, ...], torch.Tensor]:
    return random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone()


def _restore_rng(snapshot: tuple[Any, tuple[Any, ...], torch.Tensor]) -> None:
    random.setstate(snapshot[0])
    np.random.set_state(snapshot[1])
    torch.random.set_rng_state(snapshot[2].clone())


def _rng_equal(left: tuple[Any, tuple[Any, ...], torch.Tensor], right: tuple[Any, tuple[Any, ...], torch.Tensor]) -> bool:
    return bool(diagnostic._rng_equal(left, right))


def _read_rows() -> list[dict[str, Any]]:
    _require(ROWS_PATH.is_file(), f"proposal rows missing: {ROWS_PATH}")
    payload = json.loads(ROWS_PATH.read_text(encoding="utf-8"))
    _require(isinstance(payload, list), "proposal rows must be a JSON list")
    return [dict(row) for row in payload if isinstance(row, Mapping)]


def _q_k3(row: Mapping[str, Any]) -> tuple[float, list[float]]:
    logits = torch.as_tensor(row["source_proposal_logits"], dtype=torch.float32)
    certified = {int(value) for value in row.get("certified_proposals", [])}
    mass = k3_certified_mass(logits, certified, shortlist_k=3)
    probabilities = torch.softmax(logits, dim=-1).detach().cpu().tolist()
    return float(mass["certified_mass"]), [float(value) for value in probabilities]


def _select_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), []).append(dict(row))
    selected: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for seed in sorted(by_seed):
        seed_rows = by_seed[seed]
        enriched = []
        for row in seed_rows:
            q_k3, probabilities = _q_k3(row)
            item = dict(row)
            item["q_k3"] = q_k3
            item["proposal_probabilities"] = probabilities
            enriched.append(item)
        empty = next((row for row in enriched if not row.get("certified_proposals")), None)
        high = next(
            (row for row in enriched if row.get("certified_proposals") and float(row["q_k3"]) >= 0.99),
            None,
        )
        low = [
            row for row in enriched
            if row.get("certified_proposals") and float(row["q_k3"]) < 0.01
        ]
        candidates = ([('first_C_empty', empty)] if empty is not None else [])
        candidates += ([('first_qK3_ge_0.99', high)] if high is not None else [])
        candidates += [('qK3_between_0_and_0.01', row) for row in low]
        for reason, row in candidates:
            if row is None:
                continue
            key = (int(row["seed"]), int(row["step"]), int(row["uid"]), int(row["arrival_id"]))
            if key in seen:
                continue
            seen.add(key)
            row["selection_reason"] = reason
            selected.append(row)
    _require(len(selected) == EXPECTED_SELECTIONS, f"expected {EXPECTED_SELECTIONS} selected rows, got {len(selected)}")
    return selected


def _step_inputs(
    raw_actions: Mapping[Any, Any],
    raw_shortlists: Mapping[Any, Sequence[Any]],
) -> tuple[dict[str, np.ndarray], dict[str, tuple[np.ndarray, ...]]]:
    actions = {
        str(uid): np.asarray(action, dtype=np.float32)
        .copy()
        for uid, action in (raw_actions or {}).items()
    }
    shortlists = {
        str(uid): tuple(np.asarray(action, dtype=np.float32).copy() for action in values)
        for uid, values in (raw_shortlists or {}).items()
    }
    return actions, shortlists


def _world(observations: Mapping[Any, Any]) -> np.ndarray:
    values = [observations[key] for key in sorted(observations, key=str)]
    if not values:
        return np.empty((0, 88), dtype=np.float32)
    return np.asarray(values, dtype=np.float32).reshape(-1, 88)


def _target_value(mapping: Mapping[Any, Any], uid: int, default: Any = 0.0) -> Any:
    return mapping.get(str(uid), mapping.get(uid, default))


def _safety_counters(env: TerminalParallelEnv) -> dict[str, float]:
    names = {"collisions", "nmac", "hard_separation_violations", "uncertified_execution"}
    counters: dict[str, float] = {}
    for key, value in env.telemetry.items():
        if str(key) not in names:
            continue
        try:
            counters[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return counters


def _counter_delta(before: Mapping[str, float], after: Mapping[str, float]) -> dict[str, float]:
    keys = sorted(set(before) | set(after))
    return {key: float(after.get(key, 0.0) - before.get(key, 0.0)) for key in keys}


def _real_terminal(env: TerminalParallelEnv) -> bool:
    queue = getattr(getattr(env, "admission", None), "queue", None)
    exhausted = getattr(getattr(env, "arrivals", None), "exhausted", False)
    arrivals_exhausted = bool(exhausted() if callable(exhausted) else exhausted)
    return bool(arrivals_exhausted and not env.active and not getattr(queue, "items", []) and not env.service_uavs)


def _branch_result(
    base_env: TerminalParallelEnv,
    source_policy: Mapping[str, Any],
    actions: Mapping[str, np.ndarray],
    shortlists: Mapping[str, tuple[np.ndarray, ...]],
    uid: int,
    proposal: int | None,
    logits: Sequence[float],
    rng_before: tuple[Any, tuple[Any, ...], torch.Tensor],
    *,
    shortlist_k: int = 3,
) -> dict[str, Any]:
    _restore_rng(rng_before)
    shadow = copy.deepcopy(base_env)
    diagnostic._restore_shadow_methods(shadow)
    branch_actions, branch_shortlists = p3_action_credit.branch_inputs(
        actions, shortlists, uid, proposal, logits, shortlist_k=shortlist_k
    )
    branch_shortlists = branch_shortlists or {}
    safety_before = _safety_counters(shadow)
    next_observations, rewards, terminated, truncated, info = shadow.step(
        branch_actions,
        proposal_shortlists=branch_shortlists,
    )
    safety_after = _safety_counters(shadow)
    reward_components = _target_value(info.get("reward_components", {}), uid, {})
    real_terminal = _real_terminal(shadow)
    next_value = 0.0 if real_terminal else float(
        student_rollout_v4_2.world_value(source_policy["policy"], _world(next_observations))
    )
    raw_release_telemetry = _target_value(info.get("release_intent_telemetry", {}), uid, {})
    target_release_telemetry = {}
    if isinstance(raw_release_telemetry, Mapping):
        for key in (
            "primary_blocker",
            "secondary_constraints",
            "decision_history",
            "runtime_shortlist_rank",
            "reason_counts",
        ):
            if key in raw_release_telemetry:
                target_release_telemetry[key] = _jsonable(raw_release_telemetry[key])
    result = {
        "shortlist_k": int(shortlist_k),
        "proposal": proposal,
        "reward": float(sum(float(value) for value in rewards.values())),
        "target_reward": float(_target_value(rewards, uid, 0.0)),
        "reward_components": reward_components,
        "decision": _target_value(info.get("release_decisions", {}), uid, None),
        "next_value": next_value,
        "terminated": bool(real_terminal),
        "target_terminated": bool(_target_value(terminated, uid, False)),
        "target_truncated": bool(_target_value(truncated, uid, False)),
        "real_terminal": bool(real_terminal),
        "safety_counter_delta": _counter_delta(safety_before, safety_after),
        "target_release_telemetry": target_release_telemetry,
    }
    return result


def _evaluate_branches(
    base_env: TerminalParallelEnv,
    row: Mapping[str, Any],
    source_policy: Mapping[str, Any],
    actions: Mapping[str, np.ndarray],
    shortlists: Mapping[str, tuple[np.ndarray, ...]],
    branch_rng: tuple[Any, tuple[Any, ...], torch.Tensor],
) -> dict[str, Any]:
    uid = int(row["uid"])
    logits = [float(value) for value in row["source_proposal_logits"]]
    probabilities = [float(value) for value in row["proposal_probabilities"]]
    caller_rng = _rng_snapshot()
    try:
        live_digest_before = collector._digest(SafetyTraceRecorder._state(base_env))
        model_digest_before = collector.evaluator._model_digest(source_policy["policy"])
        wait = _branch_result(base_env, source_policy, actions, shortlists, uid, None, logits, branch_rng)
        releases = [
            _branch_result(
                base_env,
                source_policy,
                actions,
                shortlists,
                uid,
                proposal,
                logits,
                branch_rng,
                shortlist_k=3,
            )
            for proposal in range(PROPOSAL_COUNT)
        ]
        runtime_singleton_branches = [
            _branch_result(
                base_env,
                source_policy,
                actions,
                shortlists,
                uid,
                proposal,
                logits,
                branch_rng,
                shortlist_k=1,
            )
            for proposal in range(PROPOSAL_COUNT)
        ]
        _restore_rng(branch_rng)
        branch_rng_restored = _rng_equal(branch_rng, _rng_snapshot())
        live_digest_after = collector._digest(SafetyTraceRecorder._state(base_env))
        model_digest_after = collector.evaluator._model_digest(source_policy["policy"])
        _require(live_digest_before == live_digest_after, "branch diagnostic changed base state")
        _require(model_digest_before == model_digest_after, "branch diagnostic changed model")
        _require(branch_rng_restored, "branch diagnostic did not restore branch RNG")
        wait_summary = {key: wait[key] for key in ("reward", "next_value", "terminated", "decision")}
        release_summary = [
            {key: branch[key] for key in ("reward", "next_value", "terminated", "decision")}
            for branch in releases
        ]
        summary = p3_action_credit.summarize_branches(wait_summary, release_summary, probabilities)
        catalogue_certified_proposals = sorted(
            {int(value) for value in row.get("certified_proposals", [])}
        )
        runtime_certified_proposals = sorted(
            {
                int(branch["proposal"])
                for branch in runtime_singleton_branches
                if branch.get("proposal") is not None and branch.get("decision") == "approved"
            }
        )
        catalogue_set = set(catalogue_certified_proposals)
        runtime_set = set(runtime_certified_proposals)
        runtime_q_k3 = float(
            k3_certified_mass(
                torch.as_tensor(logits, dtype=torch.float32),
                runtime_set,
                shortlist_k=3,
            )["certified_mass"]
        )
        return {
            "wait": wait,
            "releases": releases,
            "runtime_singleton_branches": runtime_singleton_branches,
            "summary": summary,
            "actual_k3_approved_mass": float(summary["approved_mass"]),
            "catalogue_certified_proposals": catalogue_certified_proposals,
            "runtime_certified_proposals": runtime_certified_proposals,
            "catalogue_false_positive_indices": sorted(catalogue_set - runtime_set),
            "catalogue_false_negative_indices": sorted(runtime_set - catalogue_set),
            "q_k3_runtime_labels": runtime_q_k3,
            "runtime_q_vs_approved_mass_abs_error": abs(
                runtime_q_k3 - float(summary["approved_mass"])
            ),
            "label_scope": "same_prestate_fixed_other_tape_actions_runtime_singleton",
            "live_state_digest_before": live_digest_before,
            "live_state_digest_after": live_digest_after,
            "model_digest_before": model_digest_before,
            "model_digest_after": model_digest_after,
            "branch_rng_restored": branch_rng_restored,
        }
    finally:
        _restore_rng(caller_rng)


def _prefix_tape(tape: Path, step: int, directory: Path) -> Path:
    payload = json.loads(tape.read_text(encoding="utf-8"))
    rows = list(payload.get("action_tape", []))
    _require(0 <= int(step) < len(rows), f"selected step outside tape: {step}")
    payload["action_tape"] = rows[: int(step) + 1]
    path = directory / "prefix.json"
    path.write_text(json.dumps(_jsonable(payload), sort_keys=True, allow_nan=False), encoding="utf-8")
    return path


def _run_selected(
    row: Mapping[str, Any],
    record: Mapping[str, Any],
    config: Any,
    source_policy: Mapping[str, Any],
    *,
    smoke: bool,
) -> dict[str, Any]:
    tape = Path(str(record["action_tape"]))
    temp_directory = tempfile.TemporaryDirectory(prefix="p3_action_credit_") if smoke else None
    try:
        replay_tape = _prefix_tape(tape, int(row["step"]), Path(temp_directory.name)) if smoke else tape
        tape_payload = json.loads(replay_tape.read_text(encoding="utf-8"))
        tape_rows = list(tape_payload.get("action_tape", []))
        _require(int(row["step"]) < len(tape_rows), "selected step missing from replay tape")
        env = TerminalParallelEnv(config)
        original_step = env.step
        captured: dict[str, Any] = {}

        def observed_step(*args: Any, **kwargs: Any) -> Any:
            if int(env.step_count) == int(row["step"]) and not captured:
                uid = int(row["uid"])
                actor = env.active.get(uid)
                _require(actor is not None, f"selected uid is not active at step {row['step']}: {uid}")
                _require(int(actor.arrival_id) == int(row["arrival_id"]), "selected arrival_id mismatch")
                observation = np.asarray(env._observations()[str(uid)], dtype=np.float32)
                expected = np.asarray(row["observation"], dtype=np.float32)
                _require(np.array_equal(observation, expected), "selected observation mismatch")
                raw_actions = args[0] if args else kwargs.get("actions", {})
                raw_shortlists = (
                    kwargs.get("proposal_shortlists")
                    if "proposal_shortlists" in kwargs
                    else (args[1] if len(args) > 1 else {})
                )
                actions, shortlists = _step_inputs(raw_actions or {}, raw_shortlists or {})
                captured.update({
                    "env": copy.deepcopy(env),
                    "live_state_digest": collector._digest(SafetyTraceRecorder._state(env)),
                    "actions": actions,
                    "shortlists": shortlists,
                    "rng_before": _rng_snapshot(),
                })
            return original_step(*args, **kwargs)

        env.step = observed_step  # type: ignore[method-assign]
        try:
            replay = SafetyTraceRecorder.replay_action_tape(env, replay_tape)
        finally:
            env.step = original_step  # type: ignore[method-assign]
        _require(replay.get("exact") is True, f"action-tape replay was not exact: {replay}")
        outcomes_match = None
        if not smoke:
            outcomes_match = collector._summary_outcomes(tape) == collector._replay_outcomes(env)
            _require(outcomes_match, "full replay outcomes differ from source summary")
        _require(captured, "selected step was not observed")
        branches = _evaluate_branches(
            captured["env"],
            row,
            source_policy,
            captured["actions"],
            captured["shortlists"],
            captured["rng_before"],
        )
        return {
            "seed": int(row["seed"]),
            "step": int(row["step"]),
            "uid": int(row["uid"]),
            "arrival_id": int(row["arrival_id"]),
            "scenario": str(record["scenario"]),
            "selection_reason": str(row["selection_reason"]),
            "q_k3": float(row["q_k3"]),
            "observation": _jsonable(row["observation"]),
            "source_proposal_logits": _jsonable(row["source_proposal_logits"]),
            "action_tape": str(tape),
            "replay_tape": str(replay_tape),
            "replay": {key: value for key, value in replay.items() if key != "steps"},
            "outcomes_match": outcomes_match,
            "selected_state_digest": captured["live_state_digest"],
            "label_scope": "same_prestate_fixed_other_tape_actions_runtime_singleton",
            "branches": branches,
        }
    finally:
        if temp_directory is not None:
            temp_directory.cleanup()


def run(*, output_dir: Path, smoke: bool) -> dict[str, Any]:
    _require(not output_dir.exists(), f"refusing existing output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "progress.json", {"stage": "P3_ACTION_CREDIT", "status": "RUNNING", "processed": 0})
    reports: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    selected_count = 0
    manifest_path: Path | None = None
    try:
        rows = _read_rows()
        selected = _select_rows(rows)
        selected_count = len(selected)
        protocol, configs, records, manifest_path = collector._load_source_inputs(FORMAL_ROOT)
        records_by_seed = {int(record["seed"]): record for record in records}
        policy_by_scenario: dict[str, dict[str, Any]] = {}
        target_rows = selected[:1] if smoke else selected
        for index, row in enumerate(target_rows, start=1):
            record = records_by_seed[int(row["seed"])]
            scenario = str(record["scenario"])
            if scenario not in policy_by_scenario:
                policy_by_scenario[scenario] = collector._load_source_policy(protocol, configs[scenario])
            source_policy = policy_by_scenario[scenario]
            report = _run_selected(row, record, configs[scenario], source_policy, smoke=smoke)
            reports.append(report)
            branch_report = report["branches"]
            runtime_rows.append(
                {
                    "seed": int(report["seed"]),
                    "step": int(report["step"]),
                    "uid": int(report["uid"]),
                    "arrival_id": int(report["arrival_id"]),
                    "scenario": str(report["scenario"]),
                    "observation": report["observation"],
                    "source_proposal_logits": report["source_proposal_logits"],
                    "catalogue_certified_proposals": branch_report["catalogue_certified_proposals"],
                    "runtime_certified_proposals": branch_report["runtime_certified_proposals"],
                    "q_k3_runtime_labels": branch_report["q_k3_runtime_labels"],
                    "label_scope": "same_prestate_fixed_other_tape_actions_runtime_singleton",
                }
            )
            _write_json(output_dir / f"report_{index:02d}.json", report)
            _write_json(
                output_dir / "runtime_rows.json",
                {
                    "stage": "P3_ACTION_CREDIT",
                    "status": "PARTIAL_DIAGNOSTIC_ROWS",
                    "rows": runtime_rows,
                    "processed_count": len(runtime_rows),
                    "selected_count": selected_count,
                    "partial_diagnostic_rows": True,
                    "not_a_complete_training_set": True,
                    "protocol_scope": "diagnostic_reuses_existing_source_protocol_inputs; no_new_protocol",
                },
            )
            _write_json(
                output_dir / "progress.json",
                {
                    "stage": "P3_ACTION_CREDIT",
                    "status": "RUNNING",
                    "processed": index,
                    "expected": len(target_rows),
                    "last_seed": int(row["seed"]),
                    "last_step": int(row["step"]),
                },
            )
            print(
                json.dumps(
                    {"stage": "P3_ACTION_CREDIT", "status": "RUNNING", "processed": index, "expected": len(target_rows)},
                    sort_keys=True,
                ),
                flush=True,
            )
        result = {
            "stage": "P3_ACTION_CREDIT",
            "status": "SMOKE_COMPLETED" if smoke else "COMPLETED",
            "smoke": bool(smoke),
            "rows_path": str(ROWS_PATH),
            "formal_root": str(FORMAL_ROOT),
            "source_manifest": str(manifest_path),
            "selected_count": selected_count,
            "processed_count": len(reports),
            "reports": reports,
            "runtime_rows": str(output_dir / "runtime_rows.json"),
            "partial_diagnostic_rows": True,
            "not_a_complete_training_set": True,
            "protocol_scope": "diagnostic_reuses_existing_source_protocol_inputs; no_new_protocol",
            "policy_sampling": False,
            "policy_update": False,
            "PPO_ALLOWED": False,
        }
        _write_json(
            output_dir / "runtime_rows.json",
            {
                "stage": "P3_ACTION_CREDIT",
                "status": "PARTIAL_DIAGNOSTIC_ROWS",
                "rows": runtime_rows,
                "processed_count": len(runtime_rows),
                "selected_count": selected_count,
                "partial_diagnostic_rows": True,
                "not_a_complete_training_set": True,
                "protocol_scope": "diagnostic_reuses_existing_source_protocol_inputs; no_new_protocol",
            },
        )
        _write_json(output_dir / "manifest.json", result)
        _write_json(output_dir / "progress.json", {"stage": "P3_ACTION_CREDIT", "status": result["status"], "processed": len(reports)})
        return result
    except Exception as exc:
        failure = {
            "stage": "P3_ACTION_CREDIT",
            "status": "FAILED",
            "smoke": bool(smoke),
            "rows_path": str(ROWS_PATH),
            "formal_root": str(FORMAL_ROOT),
            "source_manifest": None if manifest_path is None else str(manifest_path),
            "selected_count": selected_count,
            "processed_count": len(reports),
            "reports": reports,
            "runtime_rows": str(output_dir / "runtime_rows.json"),
            "partial_diagnostic_rows": True,
            "not_a_complete_training_set": True,
            "protocol_scope": "diagnostic_reuses_existing_source_protocol_inputs; no_new_protocol",
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "policy_sampling": False,
            "policy_update": False,
            "PPO_ALLOWED": False,
        }
        _write_json(
            output_dir / "runtime_rows.json",
            {
                "stage": "P3_ACTION_CREDIT",
                "status": "PARTIAL_DIAGNOSTIC_ROWS",
                "rows": runtime_rows,
                "processed_count": len(runtime_rows),
                "selected_count": selected_count,
                "partial_diagnostic_rows": True,
                "not_a_complete_training_set": True,
                "protocol_scope": "diagnostic_reuses_existing_source_protocol_inputs; no_new_protocol",
            },
        )
        _write_json(output_dir / "manifest.json", failure)
        _write_json(output_dir / "progress.json", failure)
        print(json.dumps(_jsonable(failure), sort_keys=True), flush=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(output_dir=args.output_dir, smoke=bool(args.smoke))
    print(json.dumps(_jsonable({key: result[key] for key in ("stage", "status", "selected_count", "processed_count")})))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
