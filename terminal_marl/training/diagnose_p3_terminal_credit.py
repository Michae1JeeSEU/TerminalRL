"""Full-cycle WAIT/RELEASE credit diagnostic for the frozen V4.2 source policy.

The runner replays each selected source tape once, captures clean pre-step
environment snapshots, and evaluates three branches from each snapshot.  The
first branch step keeps every non-target request from the source tape.  After
that step, the source policy is sampled against each new observation until the
scene reaches the environment terminal condition or the registered horizon.
This module is diagnostic only: it does not update a policy or modify reward
implementation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from terminal_marl.core.safety_trace import SafetyTraceRecorder
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training import collect_p3_proposal_head_probe_v1 as collector
from terminal_marl.training import diagnose_p3_action_credit as action_credit_diagnostic
from terminal_marl.training import k3_gate_feasibility
from terminal_marl.training import p3_action_credit
from terminal_marl.training import student_rollout_v4_2
from terminal_marl.training import terminal_lifecycle_metrics
from terminal_marl.training import train_p3_runtime_rank_repair as runtime_rank_trainer


STAGE = "P3_TERMINAL_CREDIT_V1"
PROTOCOL = Path("docs/evidence/v4_2/p3_terminal_credit_protocol_20260928.md")
SOURCE_COLLECTION_DIR = Path(
    "/home/user/training_outputs/terminal/"
    "p3_runtime_rank_repair_20260926/collection"
)
FORMAL_ROOT = collector.FORMAL_ROOT
DEFAULT_OUTPUT_DIR = Path(
    "/home/user/training_outputs/terminal/p3_terminal_credit_20260928"
)
SMOKE_OUTPUT_DIR = Path(
    "/home/user/training_outputs/terminal/p3_terminal_credit_20260928_smoke"
)
SOURCE_SEEDS = (873101, 873102, 873103)
REPEAT_COUNT = 2
BRANCH_NAMES = ("wait", "source_release", "runtime_certified_release")
GAMMA = 0.99
OBSERVATION_DIM = 88
PROPOSAL_COUNT = 25
LOW_SOC_THRESHOLD = 0.65


class TerminalCreditError(ValueError):
    """Raised when the registered terminal-credit input/output contract fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TerminalCreditError(message)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "value") and isinstance(value.value, str):
        return value.value
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _read_json(path: str | Path) -> Any:
    target = Path(path)
    _require(target.is_file(), f"JSON input missing: {target}")
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TerminalCreditError(f"invalid JSON input: {target}") from exc


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rng_snapshot() -> tuple[Any, tuple[Any, ...], torch.Tensor]:
    return random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone()


def _restore_rng(snapshot: tuple[Any, tuple[Any, ...], torch.Tensor]) -> None:
    random.setstate(snapshot[0])
    np.random.set_state(snapshot[1])
    torch.random.set_rng_state(snapshot[2].clone())


def _rng_equal(
    left: tuple[Any, tuple[Any, ...], torch.Tensor],
    right: tuple[Any, tuple[Any, ...], torch.Tensor],
) -> bool:
    return bool(action_credit_diagnostic._rng_equal(left, right))


def _seeded_rng(seed: int) -> tuple[Any, tuple[Any, ...], torch.Tensor]:
    random.seed(int(seed))
    np.random.seed(int(seed) & 0xFFFFFFFF)
    torch.manual_seed(int(seed))
    return _rng_snapshot()


def _identity(row: Mapping[str, Any]) -> tuple[int, int, int]:
    return int(row["step"]), int(row["uid"]), int(row["arrival_id"])


def _policy_object(source_policy: Any) -> Any:
    if isinstance(source_policy, Mapping):
        return source_policy.get("policy", source_policy)
    return getattr(source_policy, "policy", source_policy)


def _model_digest(source_policy: Any) -> str:
    policy = _policy_object(source_policy)
    return str(collector.evaluator._model_digest(policy))


def _state_digest(env: TerminalParallelEnv) -> str:
    return str(collector._digest(SafetyTraceRecorder._state(env)))


def _real_terminal(env: TerminalParallelEnv) -> bool:
    return bool(action_credit_diagnostic._real_terminal(env))


def _source_logits(row: Mapping[str, Any]) -> np.ndarray:
    values = np.asarray(row.get("source_proposal_logits"), dtype=np.float32)
    _require(values.shape == (PROPOSAL_COUNT,), "source proposal logits must be 25D")
    _require(bool(np.isfinite(values).all()), "source proposal logits are not finite")
    return values


def _q_k3(row: Mapping[str, Any]) -> float:
    return float(
        k3_gate_feasibility.k3_certified_mass(
            torch.as_tensor(_source_logits(row)),
            {int(value) for value in row.get("certified_proposals", ())},
            shortlist_k=3,
        )["certified_mass"]
    )


def _select_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {seed: [] for seed in SOURCE_SEEDS}
    for source in rows:
        row = dict(source)
        seed = int(row.get("seed", -1))
        if seed in grouped and str(row.get("split")) == "train":
            row["q_k3"] = _q_k3(row)
            grouped[seed].append(row)
    selected: list[dict[str, Any]] = []
    for seed in SOURCE_SEEDS:
        ordered = sorted(grouped[seed], key=lambda row: _identity(row))
        _require(ordered, f"no train runtime rows for seed {seed}")
        empty = [row for row in ordered if not row.get("certified_proposals")]
        nonempty = [row for row in ordered if row.get("certified_proposals")]
        picks: list[tuple[dict[str, Any], str]] = []
        picks.append((empty[0], "earliest_empty_c") if empty else (ordered[0], "earliest_row_fallback"))
        if nonempty:
            below = [row for row in nonempty if float(row["q_k3"]) < 0.999]
            picks.append((below[0] if below else nonempty[0], "earliest_nonempty_qk3_lt_0_999" if below else "earliest_nonempty"))
            picks.append((nonempty[len(nonempty) // 2], "median_nonempty"))
            picks.append((nonempty[-1], "latest_nonempty"))
        else:
            picks.extend((ordered[0], reason) for reason in ("median_nonempty_fallback", "latest_nonempty_fallback"))
        seen: set[tuple[int, int, int]] = set()
        for row, reason in picks:
            identity = _identity(row)
            if identity not in seen:
                value = dict(row)
                value["selection_reason"] = reason
                selected.append(value)
                seen.add(identity)
        remaining = [row for row in ordered if _identity(row) not in seen]
        needed = 4 - len(seen)
        if needed > 0:
            _require(len(remaining) >= needed, f"seed {seed} has fewer than four selectable rows")
            indices = [
                int(round(index * (len(remaining) - 1) / max(needed - 1, 1)))
                for index in range(needed)
            ]
            used: set[int] = set()
            for index in indices:
                if index in used:
                    index = next(candidate for candidate in range(len(remaining)) if candidate not in used)
                used.add(index)
                row = remaining[index]
                value = dict(row)
                value["selection_reason"] = "uniform_fill"
                selected.append(value)
                seen.add(_identity(row))
    return selected


def _validate_target(env: TerminalParallelEnv, row: Mapping[str, Any]) -> np.ndarray:
    uid = int(row["uid"])
    actor = env.active.get(uid)
    _require(actor is not None, f"target uid is not active: {uid}")
    _require(int(actor.arrival_id) == int(row["arrival_id"]), "target arrival_id mismatch")
    observations = env._observations()
    observation = observations.get(str(uid), observations.get(uid))
    _require(observation is not None, f"target observation missing: uid={uid}")
    value = np.asarray(observation, dtype=np.float32)
    expected = np.asarray(row["observation"], dtype=np.float32)
    _require(value.shape == (OBSERVATION_DIM,), "live target observation is not 88D")
    _require(np.array_equal(value, expected), "selected row observation does not match replay state")
    return value.copy()


def _request_signature(request: Any) -> dict[str, Any]:
    mission = getattr(request, "mission_type", None)
    return {
        "arrival_id": int(request.arrival_id),
        "request_time": float(request.request_time),
        "initial_soc": float(request.initial_soc),
        "base_priority": int(request.base_priority),
        "gate_id": int(request.gate_id),
        "mission": getattr(mission, "value", mission),
    }


def _request_capture(env: TerminalParallelEnv) -> list[dict[str, Any]]:
    captured: list[dict[str, Any]] = []
    original = env.arrivals.requests_until

    def requests_until(now: float):
        requests = original(now)
        captured.extend(_request_signature(request) for request in requests)
        return requests

    env.arrivals.requests_until = requests_until  # type: ignore[method-assign]
    return captured


def _replay_seed(
    *,
    record: Mapping[str, Any],
    config: Any,
    target_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    tape = Path(str(record["action_tape"]))
    payload = _read_json(tape)
    tape_rows = list(payload.get("action_tape", payload.get("records", [])))
    _require(tape_rows, f"empty source action tape: {tape}")
    _require(
        all(isinstance(item, Mapping) and "a_requested" in item and "observation_hash" in item for item in tape_rows),
        "source tape rows must contain a_requested and observation_hash",
    )
    by_step: dict[int, list[dict[str, Any]]] = {}
    for row in target_rows:
        by_step.setdefault(int(row["step"]), []).append(dict(row))
    environment = TerminalParallelEnv(config)
    original_step = environment.step
    snapshots: dict[tuple[int, int, int], TerminalParallelEnv] = {}
    captures: dict[tuple[int, int, int], dict[str, Any]] = {}
    parent_rng = _rng_snapshot()

    def observed_step(*args: Any, **kwargs: Any) -> Any:
        step = int(environment.step_count)
        rows_here = by_step.get(step, ())
        if rows_here:
            raw_actions = args[0] if args else kwargs.get("actions", {})
            raw_shortlists = (
                kwargs.get("proposal_shortlists")
                if "proposal_shortlists" in kwargs
                else (args[1] if len(args) > 1 else {})
            )
            actions, shortlists = action_credit_diagnostic._step_inputs(
                raw_actions or {}, raw_shortlists or {},
            )
            for row in rows_here:
                key = _identity(row)
                observation = _validate_target(environment, row)
                environment.step = original_step  # type: ignore[method-assign]
                try:
                    snapshot = copy.deepcopy(environment)
                finally:
                    environment.step = observed_step  # type: ignore[method-assign]
                snapshot._step_observer = None
                snapshots[key] = snapshot
                captures[key] = {
                    "observation": observation,
                    "actions": actions,
                    "shortlists": shortlists,
                    "step": step,
                }
        return original_step(*args, **kwargs)

    environment.step = observed_step  # type: ignore[method-assign]
    try:
        replay = SafetyTraceRecorder.replay_action_tape(environment, tape)
    finally:
        environment.step = original_step  # type: ignore[method-assign]
        _restore_rng(parent_rng)
    _require(replay.get("exact") is True, f"source replay is not exact: {replay}")
    for row in target_rows:
        _require(_identity(row) in snapshots, f"selected row was not observed: {_identity(row)}")
    return {
        "seed": int(record["seed"]),
        "scenario": str(record["scenario"]),
        "split": str(record["split"]),
        "action_tape": str(tape),
        "replay": {key: value for key, value in replay.items() if key != "steps"},
        "snapshots": snapshots,
        "captures": captures,
    }


def _sample_proposal(logits: np.ndarray) -> int:
    distribution = torch.distributions.Categorical(logits=torch.as_tensor(logits, dtype=torch.float32))
    return int(distribution.sample().item())


def _runtime_proposal(row: Mapping[str, Any], logits: np.ndarray) -> int | None:
    certified = {int(value) for value in row.get("certified_proposals", ())}
    if not certified:
        return None
    ranked = np.argsort(-logits, kind="stable").tolist()
    return next(int(value) for value in ranked if int(value) in certified)


def _reward_components(info: Mapping[str, Any]) -> dict[str, float]:
    totals: dict[str, float] = {}
    components = info.get("reward_components", {})
    if isinstance(components, Mapping):
        for values in components.values():
            if not isinstance(values, Mapping):
                continue
            for name, value in values.items():
                if str(name) == "total":
                    continue
                totals[str(name)] = totals.get(str(name), 0.0) + float(value)
    return totals


def _transition(
    *,
    observations: Mapping[Any, Any],
    actions: Mapping[Any, Any],
    shortlists: Mapping[Any, Any],
    following: Mapping[Any, Any],
    rewards: Mapping[Any, Any],
    terminated: Mapping[Any, Any],
    truncated: Mapping[Any, Any],
    info: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "rewards": _jsonable(rewards),
        "reward": float(sum(float(value) for value in rewards.values())),
        "reward_components": _reward_components(info),
        "terminated": _jsonable(terminated),
        "truncated": _jsonable(truncated),
        "time": _jsonable(info.get("time")),
        "outcomes": _jsonable(info.get("outcomes", {})),
    }


def _source_request(
    policy: Any,
    env: TerminalParallelEnv,
    observations: Mapping[Any, Any],
    config: Any,
) -> tuple[dict[str, Any], dict[str, Any], tuple[Any, ...]]:
    details = student_rollout_v4_2._student_request_details(
        policy,
        env,
        dict(observations),
        deterministic=False,
        shortlist_k=int(getattr(config, "runtime_proposal_shortlist_k", 1)),
    )
    requested = details[3]
    shortlists = details[9]
    return requested, shortlists, details


def _continue_source(
    *,
    policy: Any,
    env: TerminalParallelEnv,
    observations: Mapping[Any, Any],
    config: Any,
    horizon: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    current = dict(observations)
    transitions: list[dict[str, Any]] = []
    while not _real_terminal(env) and int(env.step_count) < int(horizon):
        actions, shortlists, _details = _source_request(policy, env, current, config)
        following, rewards, terminated, truncated, info = env.step(actions, shortlists)
        transitions.append(
            _transition(
                observations=current,
                actions=actions,
                shortlists=shortlists,
                following=following,
                rewards=rewards,
                terminated=terminated,
                truncated=truncated,
                info=info,
            )
        )
        current = dict(following)
    return current, transitions


def _aggregate_rewards(transitions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    components: dict[str, float] = {}
    discounted_components: dict[str, float] = {}
    undiscounted = 0.0
    discounted = 0.0
    for index, transition in enumerate(transitions):
        reward = float(transition.get("reward", 0.0))
        weight = GAMMA ** index
        undiscounted += reward
        discounted += weight * reward
        values = transition.get("reward_components", {})
        if isinstance(values, Mapping):
            for name, value in values.items():
                components[str(name)] = components.get(str(name), 0.0) + float(value)
                discounted_components[str(name)] = discounted_components.get(str(name), 0.0) + weight * float(value)
    return {
        "suffix_undiscounted_reward": float(undiscounted),
        "suffix_discounted_reward": float(discounted),
        "suffix_reward_components": components,
        "suffix_discounted_reward_components": discounted_components,
        "steps": len(transitions),
    }


REQUIRED_SAFETY_COUNTERS = (
    "collisions",
    "nmac",
    "hard_separation_violations",
    "checked_executed_plan_hash_mismatches",
)
OPTIONAL_SAFETY_COUNTERS = ("uncertified_execution",)
SAFETY_TELEMETRY_COUNTERS = REQUIRED_SAFETY_COUNTERS + (
    "checked_executed_plan_matches",
) + OPTIONAL_SAFETY_COUNTERS


def _telemetry_counter(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safety_snapshot(env: TerminalParallelEnv) -> dict[str, Any]:
    telemetry = getattr(env, "telemetry", {})
    snapshot = {
        name: _telemetry_counter(telemetry.get(name))
        for name in SAFETY_TELEMETRY_COUNTERS
    }
    snapshot["uncertified_execution_available"] = snapshot["uncertified_execution"] is not None
    return snapshot


def _safety_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, float | None]:
    delta: dict[str, float | None] = {}
    for name in SAFETY_TELEMETRY_COUNTERS:
        left = before.get(name)
        right = after.get(name)
        delta[name] = None if left is None or right is None else float(right) - float(left)
    return delta


def _ledger_summary(
    *,
    env: TerminalParallelEnv,
    captured: Sequence[Mapping[str, Any]],
    first_seen: Mapping[int, Mapping[str, Any]],
    config: Any,
) -> dict[str, Any]:
    rows = terminal_lifecycle_metrics._merge_terminal_rows(
        captured,
        terminal_lifecycle_metrics._completed_rows(env, first_seen),
    )
    requested_count = int(config.total_requested_uavs)
    expected_ids = set(range(requested_count))
    ledger = getattr(env, "ledger", None)
    requested_ids = {int(value) for value in getattr(ledger, "requested", set())}
    ledger_values = {
        int(key): str(terminal_lifecycle_metrics._enum_value(value))
        for key, value in getattr(ledger, "outcomes", {}).items()
    }
    observed_ids = {int(row["arrival_id"]) for row in rows}
    missing_ids = sorted(expected_ids - observed_ids)
    unexpected_ids = sorted(observed_ids - expected_ids)
    ledger_without_rows = sorted(set(ledger_values) - observed_ids)
    unresolved_requested = sorted(requested_ids - set(ledger_values))
    rows_without_ledger = sorted(observed_ids - set(ledger_values))
    outcome_mismatch = sorted(
        arrival_id
        for arrival_id, outcome in ledger_values.items()
        if arrival_id in {int(row["arrival_id"]) for row in rows}
        and str(next(row for row in rows if int(row["arrival_id"]) == arrival_id).get("outcome")) != outcome
    )
    summary = terminal_lifecycle_metrics.summarize_terminal_rows(
        rows,
        low_soc_threshold=LOW_SOC_THRESHOLD,
        requested_count=requested_count,
        missing_arrival_ids=missing_ids,
        enforce_unique=True,
    )
    summary.update(
        {
            "replay_exact": None,
            "terminal_continuation": _real_terminal(env),
            "requested_arrival_ids": sorted(expected_ids),
            "observed_arrival_ids": sorted(observed_ids),
            "ledger_requested_ids": sorted(requested_ids),
            "ledger_outcomes_without_rows": ledger_without_rows,
            "ledger_unresolved_requested_ids": unresolved_requested,
            "rows_without_ledger_outcome": rows_without_ledger,
            "outcome_mismatch_arrival_ids": outcome_mismatch,
            "all_requests_observed": not missing_ids,
            "ledger_alignment": not (
                missing_ids
                or unexpected_ids
                or ledger_without_rows
                or unresolved_requested
                or rows_without_ledger
                or outcome_mismatch
            ),
        }
    )
    return summary


def _run_branch(
    *,
    policy: Any,
    base_env: TerminalParallelEnv,
    row: Mapping[str, Any],
    source_actions: Mapping[Any, Any],
    source_shortlists: Mapping[Any, Any],
    logits: np.ndarray,
    branch_name: str,
    branch_rng: tuple[Any, tuple[Any, ...], torch.Tensor],
    config: Any,
    horizon: int,
) -> dict[str, Any]:
    parent_rng = _rng_snapshot()
    base_digest = _state_digest(base_env)
    model_before = _model_digest(policy)
    proposal: int | None
    no_alternative = False
    try:
        _restore_rng(branch_rng)
        if branch_name == "wait":
            proposal = None
        elif branch_name == "source_release":
            proposal = _sample_proposal(logits)
        elif branch_name == "runtime_certified_release":
            proposal = _runtime_proposal(row, logits)
            if proposal is None:
                no_alternative = True
                proposal = _sample_proposal(logits)
        else:
            raise TerminalCreditError(f"unknown branch: {branch_name}")
        actions, shortlists = p3_action_credit.branch_inputs(
            source_actions,
            source_shortlists,
            int(row["uid"]),
            proposal,
            logits,
            shortlist_k=3,
        )
        branch_env = copy.deepcopy(base_env)
        action_credit_diagnostic.diagnostic._restore_shadow_methods(branch_env)
        branch_env._step_observer = None
        arrival_signatures = _request_capture(branch_env)
        captured, first_seen = terminal_lifecycle_metrics._capture_terminal_rows(branch_env)
        before_observations = branch_env._observations()
        _validate_target(branch_env, row)
        safety_before = _safety_snapshot(branch_env)
        following, rewards, terminated, truncated, info = branch_env.step(actions, shortlists)
        transitions = [
            _transition(
                observations=before_observations,
                actions=actions,
                shortlists=shortlists,
                following=following,
                rewards=rewards,
                terminated=terminated,
                truncated=truncated,
                info=info,
            )
        ]
        first_real_terminal = _real_terminal(branch_env)
        first_next_value = 0.0 if first_real_terminal else student_rollout_v4_2.world_value(
            policy,
            action_credit_diagnostic._world(following),
        )
        first_proxy = float(transitions[0]["reward"]) + GAMMA * float(first_next_value)
        _restore_rng(branch_rng)
        _current, continuation = _continue_source(
            policy=policy,
            env=branch_env,
            observations=following,
            config=config,
            horizon=horizon,
        )
        transitions.extend(continuation)
        lifecycle = _ledger_summary(
            env=branch_env,
            captured=captured,
            first_seen=first_seen,
            config=config,
        )
        safety_after = _safety_snapshot(branch_env)
        safety_delta = _safety_delta(safety_before, safety_after)
        model_after = _model_digest(policy)
        result = {
            "branch": branch_name,
            "proposal": proposal,
            "no_alternative": no_alternative,
            "reused_source_branch": False,
            "target_arrival_id": int(row["arrival_id"]),
            "terminal": _real_terminal(branch_env),
            "horizon": int(horizon),
            "steps": len(transitions),
            "transitions": transitions,
            "rewards": _aggregate_rewards(transitions),
            "first_step": {
                "actions": _jsonable(actions),
                "proposal_shortlists": _jsonable(shortlists),
                "reward": float(transitions[0]["reward"]),
                "next_value": float(first_next_value),
                "r_plus_gamma_v": float(first_proxy),
            },
            "lifecycle": lifecycle,
            "target_terminal_row": next(
                (dict(item) for item in lifecycle.get("rows", ()) if int(item.get("arrival_id")) == int(row["arrival_id"])),
                None,
            ),
            "arrival_signatures": arrival_signatures,
            "safety": {
                "before": safety_before,
                "after": safety_after,
                "delta": safety_delta,
                "required_delta": {
                    name: safety_delta[name]
                    for name in REQUIRED_SAFETY_COUNTERS
                },
                "required_available": {
                    name: safety_before.get(name) is not None and safety_after.get(name) is not None
                    for name in REQUIRED_SAFETY_COUNTERS
                },
                "uncertified_execution": {
                    "before": safety_before.get("uncertified_execution"),
                    "after": safety_after.get("uncertified_execution"),
                    "delta": safety_delta.get("uncertified_execution"),
                    "available": bool(safety_after.get("uncertified_execution_available", False)),
                },
                "checked_executed_plan_matches": safety_after.get("checked_executed_plan_matches"),
                "telemetry": dict(branch_env.telemetry),
            },
            "weights": {
                "before": model_before,
                "after": model_after,
                "unchanged": bool(model_before == model_after),
            },
            "base_state_digest": base_digest,
            "base_state_unchanged": bool(_state_digest(base_env) == base_digest),
            "rng_restored": False,
        }
        if model_before != model_after:
            result["error"] = {"type": "WeightsChanged", "message": "source policy changed during branch"}
        return result
    except BaseException as exc:
        result = {
            "branch": branch_name,
            "proposal": locals().get("proposal"),
            "no_alternative": bool(locals().get("no_alternative", False)),
            "reused_source_branch": False,
            "target_arrival_id": int(row["arrival_id"]),
            "terminal": False,
            "steps": 0,
            "transitions": [],
            "rewards": _aggregate_rewards([]),
            "arrival_signatures": [],
            "base_state_digest": base_digest,
            "base_state_unchanged": bool(_state_digest(base_env) == base_digest),
            "rng_restored": False,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        return result
    finally:
        result = locals().get("result")
        _restore_rng(parent_rng)
        if isinstance(result, dict):
            result["rng_restored"] = _rng_equal(parent_rng, _rng_snapshot())


def _copy_branch(value: Mapping[str, Any], *, branch_name: str, no_alternative: bool) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    result["branch"] = branch_name
    result["reused_source_branch"] = True
    result["no_alternative"] = bool(no_alternative)
    return result


def _compare_pair(wait: Mapping[str, Any], release: Mapping[str, Any]) -> dict[str, Any]:
    wait_life = wait.get("lifecycle", {}) if isinstance(wait.get("lifecycle"), Mapping) else {}
    release_life = release.get("lifecycle", {}) if isinstance(release.get("lifecycle"), Mapping) else {}
    wait_outcomes = wait_life.get("outcomes", {}) if isinstance(wait_life.get("outcomes"), Mapping) else {}
    release_outcomes = release_life.get("outcomes", {}) if isinstance(release_life.get("outcomes"), Mapping) else {}
    success = {"completed", "emergency_landed"}
    wait_failed = sum(int(value) for key, value in wait_outcomes.items() if str(key) not in success)
    release_failed = sum(int(value) for key, value in release_outcomes.items() if str(key) not in success)
    wait_normal = int(wait_outcomes.get("completed", 0))
    release_normal = int(release_outcomes.get("completed", 0))
    wait_priority = wait_life.get("priority_weighted_wait_seconds")
    release_priority = release_life.get("priority_weighted_wait_seconds")
    wait_low = wait_life.get("low_soc", {}) if isinstance(wait_life.get("low_soc"), Mapping) else {}
    release_low = release_life.get("low_soc", {}) if isinstance(release_life.get("low_soc"), Mapping) else {}
    wait_low_count = int(wait_low.get("count", 0))
    release_low_count = int(release_low.get("count", 0))
    low_comparable = wait_low_count > 0 and release_low_count > 0
    low_unavailable = (wait_low_count == 0) != (release_low_count == 0)
    wait_low_wait = wait_low.get("mean_wait_seconds")
    release_low_wait = release_low.get("mean_wait_seconds")
    if release_failed < wait_failed:
        direction = "release"
        reason = "failed_count"
    elif release_failed > wait_failed:
        direction = "wait"
        reason = "failed_count"
    elif release_normal > wait_normal:
        direction = "release"
        reason = "normal_count"
    elif release_normal < wait_normal:
        direction = "wait"
        reason = "normal_count"
    else:
        priority_available = wait_priority is not None and release_priority is not None
        low_available = low_comparable and wait_low_wait is not None and release_low_wait is not None
        dimensions: list[tuple[str, float, float]] = []
        if priority_available:
            dimensions.append(("priority", float(release_priority), float(wait_priority)))
        if low_available:
            dimensions.append(("low_soc", float(release_low_wait), float(wait_low_wait)))
        incomplete = low_unavailable or (low_comparable and not low_available)
        release_better = bool(dimensions) and not incomplete and all(candidate <= baseline for _, candidate, baseline in dimensions) and any(candidate < baseline for _, candidate, baseline in dimensions)
        wait_better = bool(dimensions) and not incomplete and all(baseline <= candidate for _, candidate, baseline in dimensions) and any(baseline < candidate for _, candidate, baseline in dimensions)
        if release_better:
            direction, reason = "release", "wait_pareto"
        elif wait_better:
            direction, reason = "wait", "wait_pareto"
        elif incomplete:
            direction, reason = "uncertain", "low_soc_noncomparable"
        else:
            direction, reason = "tie", "wait_tie_or_noncomparable"
    return {
        "direction": direction,
        "reason": reason,
        "failed": {"wait": wait_failed, "release": release_failed},
        "normal": {"wait": wait_normal, "release": release_normal},
        "priority_weighted_wait_seconds": {"wait": wait_priority, "release": release_priority},
        "low_soc": {
            "wait_count": wait_low_count,
            "release_count": release_low_count,
            "comparable": low_comparable,
            "wait_mean_wait_seconds": wait_low_wait if low_comparable else None,
            "release_mean_wait_seconds": release_low_wait if low_comparable else None,
        },
    }


def _branch_acceptance(branch: Mapping[str, Any]) -> bool:
    lifecycle = branch.get("lifecycle")
    safety = branch.get("safety")
    weights = branch.get("weights")
    required_delta = safety.get("required_delta") if isinstance(safety, Mapping) else None
    required_after = safety.get("after") if isinstance(safety, Mapping) else None
    required_available = safety.get("required_available") if isinstance(safety, Mapping) else None
    return bool(
        branch.get("terminal") is True
        and isinstance(lifecycle, Mapping)
        and lifecycle.get("all_requests_observed") is True
        and lifecycle.get("ledger_alignment") is True
        and branch.get("base_state_unchanged") is True
        and branch.get("rng_restored") is True
        and branch.get("arrival_signature_match") is True
        and isinstance(weights, Mapping)
        and weights.get("unchanged") is True
        and isinstance(required_delta, Mapping)
        and isinstance(required_after, Mapping)
        and isinstance(required_available, Mapping)
        and all(required_available.get(name) is True for name in REQUIRED_SAFETY_COUNTERS)
        and all(required_delta.get(name) is not None and float(required_delta[name]) == 0.0 for name in REQUIRED_SAFETY_COUNTERS)
        and all(required_after.get(name) is not None and float(required_after[name]) == 0.0 for name in REQUIRED_SAFETY_COUNTERS)
        and not branch.get("error")
    )


def _state_acceptance(report: Mapping[str, Any]) -> bool:
    branches = report.get("branches")
    return bool(
        isinstance(branches, Mapping)
        and set(branches) == set(BRANCH_NAMES)
        and all(_branch_acceptance(branch) for branch in branches.values())
        and report.get("arrival_signature_match") is True
        and report.get("base_state_unchanged") is True
        and report.get("source_policy_weights_unchanged") is True
    )


def _branch_summary(branch: Mapping[str, Any], path: Path) -> dict[str, Any]:
    lifecycle = branch.get("lifecycle", {})
    summary = {
        "branch": branch.get("branch"),
        "proposal": branch.get("proposal"),
        "no_alternative": branch.get("no_alternative"),
        "reused_source_branch": branch.get("reused_source_branch"),
        "terminal": branch.get("terminal"),
        "steps": branch.get("steps"),
        "rewards": branch.get("rewards"),
        "first_step": branch.get("first_step"),
        "outcomes": lifecycle.get("outcomes") if isinstance(lifecycle, Mapping) else None,
        "all_requests_observed": lifecycle.get("all_requests_observed") if isinstance(lifecycle, Mapping) else False,
        "ledger_alignment": lifecycle.get("ledger_alignment") if isinstance(lifecycle, Mapping) else False,
        "safety": branch.get("safety"),
        "weights": branch.get("weights"),
        "base_state_unchanged": branch.get("base_state_unchanged"),
        "rng_restored": branch.get("rng_restored"),
        "arrival_signature_match": branch.get("arrival_signature_match"),
        "error": branch.get("error"),
        "branch_file": str(path),
    }
    return summary


def _state_report(
    *,
    state_index: int,
    row: Mapping[str, Any],
    snapshot: TerminalParallelEnv,
    capture: Mapping[str, Any],
    policy: Any,
    config: Any,
    repeat: int,
    horizon: int,
) -> dict[str, Any]:
    parent_rng = _rng_snapshot()
    branch_rng = _seeded_rng(20260928 + 1000 * int(state_index) + int(repeat))
    base_digest = _state_digest(snapshot)
    source_actions = capture["actions"]
    source_shortlists = capture["shortlists"]
    logits = _source_logits(row)
    branches: dict[str, dict[str, Any]] = {}
    result: dict[str, Any] | None = None
    try:
        branches["wait"] = _run_branch(
            policy=policy,
            base_env=snapshot,
            row=row,
            source_actions=source_actions,
            source_shortlists=source_shortlists,
            logits=logits,
            branch_name="wait",
            branch_rng=branch_rng,
            config=config,
            horizon=horizon,
        )
        branches["source_release"] = _run_branch(
            policy=policy,
            base_env=snapshot,
            row=row,
            source_actions=source_actions,
            source_shortlists=source_shortlists,
            logits=logits,
            branch_name="source_release",
            branch_rng=branch_rng,
            config=config,
            horizon=horizon,
        )
        runtime = _runtime_proposal(row, logits)
        if runtime is None:
            branches["runtime_certified_release"] = _copy_branch(
                branches["source_release"],
                branch_name="runtime_certified_release",
                no_alternative=True,
            )
        elif int(branches["source_release"].get("proposal", -1)) == int(runtime):
            branches["runtime_certified_release"] = _copy_branch(
                branches["source_release"],
                branch_name="runtime_certified_release",
                no_alternative=False,
            )
        else:
            branches["runtime_certified_release"] = _run_branch(
                policy=policy,
                base_env=snapshot,
                row=row,
                source_actions=source_actions,
                source_shortlists=source_shortlists,
                logits=logits,
                branch_name="runtime_certified_release",
                branch_rng=branch_rng,
                config=config,
                horizon=horizon,
            )
        signatures = [branches[name].get("arrival_signatures", []) for name in BRANCH_NAMES]
        signature_match = all(signatures[index] == signatures[0] for index in range(1, len(signatures)))
        if not signature_match:
            error = {"type": "ExternalArrivalMismatch", "message": "branch arrival signatures differ"}
            for branch in branches.values():
                branch["arrival_signature_match"] = False
                branch["error"] = error
        else:
            for branch in branches.values():
                branch["arrival_signature_match"] = True
        comparison = _compare_pair(branches["wait"], branches["source_release"])
        directions = [comparison["direction"]] * REPEAT_COUNT
        result = {
            "state_index": int(state_index),
            "repeat": int(repeat),
            "seed": int(row["seed"]),
            "scenario": str(row.get("scenario")),
            "split": str(row.get("split")),
            "step": int(row["step"]),
            "uid": int(row["uid"]),
            "arrival_id": int(row["arrival_id"]),
            "selection_reason": str(row.get("selection_reason")),
            "q_k3": float(row.get("q_k3", _q_k3(row))),
            "branch_rng_seed": 20260928 + 1000 * int(state_index) + int(repeat),
            "base_state_digest": base_digest,
            "base_state_unchanged": bool(_state_digest(snapshot) == base_digest),
            "branches": branches,
            "arrival_signature_match": bool(signature_match),
            "comparison": comparison,
            "label": comparison["direction"] if signature_match and comparison["direction"] in {"wait", "release"} else "uncertain",
            "source_policy_weights_unchanged": all(
                bool(branch.get("weights", {}).get("unchanged", False))
                for branch in branches.values()
            ),
            "parent_rng_unchanged": False,
        }
        return result
    finally:
        _restore_rng(parent_rng)
        if result is not None:
            result["parent_rng_unchanged"] = _rng_equal(parent_rng, _rng_snapshot())


def _summarize_states(states: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[int, int, int, int], list[Mapping[str, Any]]] = {}
    for state in states:
        key = (int(state["state_index"]), int(state["seed"]), int(state["step"]), int(state["uid"]))
        grouped.setdefault(key, []).append(state)
    labels: list[dict[str, Any]] = []
    for key, repeats in sorted(grouped.items()):
        directions = [str(item.get("label", "uncertain")) for item in repeats]
        label = directions[0] if len(directions) == REPEAT_COUNT and len(set(directions)) == 1 and directions[0] in {"wait", "release"} else "uncertain"
        labels.append({
            "state": key,
            "repeat_labels": directions,
            "label": label,
            "same_direction": len(set(directions)) == 1,
        })
    return {"states": labels, "count": len(labels)}


def run(
    *,
    output_dir: str | Path,
    collection_dir: str | Path = SOURCE_COLLECTION_DIR,
    formal_root: str | Path = FORMAL_ROOT,
    smoke: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir)
    _require(not output.exists(), f"output directory must be fresh: {output}")
    source_rows, collection_report = runtime_rank_trainer._load_collection(collection_dir)
    selected = _select_rows(source_rows)
    selected = selected[:1] if smoke else selected
    output.mkdir(parents=True, exist_ok=False)
    selection = {
        "stage": STAGE,
        "protocol": {"path": str(PROTOCOL), "sha256": _sha256(PROTOCOL)},
        "source_collection": collection_report,
        "source_checkpoint_sha256": collector.SOURCE_CHECKPOINT_SHA256,
        "smoke": bool(smoke),
        "repeat_count": 1 if smoke else REPEAT_COUNT,
        "branch_names": list(BRANCH_NAMES),
        "selected_rows": selected,
    }
    _write_json(output / "selection.json", selection)
    reports: list[dict[str, Any]] = []
    snapshots_by_seed: dict[int, dict[str, Any]] = {}
    protocol, configs, source_records, formal_manifest = collector._load_source_inputs(Path(formal_root))
    records_by_seed = {int(record["seed"]): dict(record) for record in source_records}
    for row in selected:
        seed = int(row["seed"])
        _require(seed in records_by_seed, f"selected row seed has no source record: {seed}")
        row["scenario"] = str(records_by_seed[seed]["scenario"])
    _write_json(output / "selection.json", selection)
    policies: dict[str, Any] = {}
    try:
        for state_index, row in enumerate(selected):
            seed = int(row["seed"])
            scenario = str(row["scenario"])
            if scenario not in policies:
                policies[scenario] = collector._load_source_policy(protocol, configs[scenario])
            if seed not in snapshots_by_seed:
                seed_rows = [item for item in selected if int(item["seed"]) == seed]
                snapshots_by_seed[seed] = _replay_seed(
                    record=records_by_seed[seed],
                    config=configs[scenario],
                    target_rows=seed_rows,
                )
            replay = snapshots_by_seed[seed]
            key = _identity(row)
            snapshot = replay["snapshots"][key]
            capture = replay["captures"][key]
            repeat_count = 1 if smoke else REPEAT_COUNT
            horizon = int(round(float(configs[scenario].max_simulation_seconds) / float(configs[scenario].dt)))
            for repeat in range(repeat_count):
                report = _state_report(
                    state_index=state_index,
                    row=row,
                    snapshot=snapshot,
                    capture=capture,
                    policy=_policy_object(policies[scenario]),
                    config=configs[scenario],
                    repeat=repeat,
                    horizon=horizon,
                )
                state_dir = output / "states" / f"state_{state_index:03d}_rep_{repeat}"
                branch_paths = {
                    branch_name: state_dir / f"{branch_name}.json"
                    for branch_name in BRANCH_NAMES
                }
                for branch_name, branch in report.get("branches", {}).items():
                    _write_json(branch_paths[branch_name], branch)
                accepted = _state_acceptance(report)
                report["acceptance"] = accepted
                report["branches"] = {
                    branch_name: _branch_summary(branch, branch_paths[branch_name])
                    for branch_name, branch in report.get("branches", {}).items()
                }
                _write_json(state_dir / "report.json", report)
                reports.append(report)
                _write_json(
                    output / "progress.json",
                    {"stage": STAGE, "status": "RUNNING", "states_processed": len(reports), "states_expected": len(selected) * repeat_count},
                )
                if not accepted:
                    raise TerminalCreditError(f"terminal credit state failed acceptance: state={state_index} repeat={repeat}")
        summary = _summarize_states(reports)
        result = {
            "stage": STAGE,
            "status": "SMOKE_COMPLETED" if smoke else "COMPLETED",
            "output_dir": str(output),
            "protocol": selection["protocol"],
            "source_collection": collection_report,
            "source_formal_manifest": str(formal_manifest),
            "selection": str(output / "selection.json"),
            "states": reports,
            "labels": summary,
            "states_processed": len(reports),
            "states_expected": len(selected) * (1 if smoke else REPEAT_COUNT),
            "branches_per_state": len(BRANCH_NAMES),
            "training": False,
            "reward_modified": False,
            "policy_updated": False,
            "PPO_ALLOWED": False,
        }
        _write_json(output / "summary.json", result)
        _write_json(output / "progress.json", {"stage": STAGE, "status": result["status"], "states_processed": len(reports), "states_expected": result["states_expected"]})
        return result
    except BaseException as exc:
        result = {
            "stage": STAGE,
            "status": "FAILED",
            "output_dir": str(output),
            "protocol": selection["protocol"],
            "source_collection": collection_report,
            "selection": str(output / "selection.json"),
            "states": reports,
            "states_processed": len(reports),
            "states_expected": len(selected) * (1 if smoke else REPEAT_COUNT),
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "training": False,
            "reward_modified": False,
            "policy_updated": False,
            "PPO_ALLOWED": False,
        }
        _write_json(output / "summary.json", result)
        _write_json(output / "progress.json", {"stage": STAGE, "status": "FAILED", "states_processed": len(reports), "states_expected": result["states_expected"], "error": result["error"]})
        return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--collection-dir", type=Path, default=SOURCE_COLLECTION_DIR)
    parser.add_argument("--formal-root", type=Path, default=FORMAL_ROOT)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(
        output_dir=args.output_dir,
        collection_dir=args.collection_dir,
        formal_root=args.formal_root,
        smoke=bool(args.smoke),
    )
    print(json.dumps(_jsonable(result), sort_keys=True), flush=True)
    return 0 if result["status"] in {"COMPLETED", "SMOKE_COMPLETED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
