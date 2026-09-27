"""Reconstruct raw and discounted terminal returns from existing V7 tapes.

This audit consumes already recorded action tapes.  It does not sample a
policy or perform an update.  A single exact tape replay is wrapped so the
environment reward returned by each ``step`` and the terminal lifecycle hooks
are captured together.
"""
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from terminal_marl.core.safety_trace import SafetyTraceRecorder
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training import evaluate_p3_adaptive_kl_v7 as evaluator
from terminal_marl.training import terminal_lifecycle_metrics as lifecycle


STAGE = "P3_TERMINAL_CREDIT_AUDIT_V1"
GAMMA = 0.99
DEFAULT_MANIFEST = Path.home() / "training_outputs/terminal/p3_adaptive_kl_terminal_dev_metrics_20260926/manifest.json"
DEFAULT_OUTPUT_ROOT = Path.home() / "training_outputs/terminal/p3_terminal_credit_20260928/reward_order"
COMPONENT_NAMES = (
    "time",
    "progress",
    "release",
    "touchdown",
    "failure",
    "teacher",
    "shield",
    "urgency",
    "emergency",
)
NEW_ADMISSION_PHASE = "newly_admitted_this_step"
REQUIRED_GROUPS = {
    "run1_dev_before_training": "source_03c0",
    "run0_dev_after_0080": "final_run0",
    "run1_dev_after_0240": "final_run1",
    "run2_dev_after_0240": "final_run2",
}


class AuditInputError(ValueError):
    """Raised when an existing audit input is incomplete or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditInputError(message)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and callable(value.item):
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"JSON input missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise AuditInputError(f"{name} must be finite numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise AuditInputError(f"{name} must be finite numeric") from exc
    if not math.isfinite(number):
        raise AuditInputError(f"{name} must be finite numeric")
    return number


def _enum_value(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw)


def _augment_new_admission_phases(
    env: TerminalParallelEnv,
    pre_step_phase: Mapping[str, Any],
    rewards: Mapping[str, Any],
) -> dict[str, Any]:
    """Label reward rows for UAVs admitted during the current step.

    A newly admitted UAV is absent from ``env.active`` before ``step`` but
    may receive a reward row before the step returns.  Its exact pre-step
    ring phase therefore does not exist; retain the reward and expose that
    fact explicitly instead of assigning a synthetic ring phase.
    """
    phases = {str(uid): phase for uid, phase in pre_step_phase.items()}
    post_step_uids = {str(uid) for uid in getattr(env, "active", {})}
    post_step_uids.update(str(uid) for uid in getattr(env, "service_uavs", {}))
    post_step_uids.update(
        str(getattr(uav, "uid"))
        for uav in getattr(env, "completed_uavs", ())
        if getattr(uav, "uid", None) is not None
    )
    for uid in rewards:
        uid_key = str(uid)
        if uid_key in phases:
            continue
        _require(
            uid_key in post_step_uids,
            f"pre-step phase missing for non-admitted uid {uid_key}",
        )
        phases[uid_key] = NEW_ADMISSION_PHASE
    return phases


def _resolve_artifact(value: Any, *, manifest_path: Path) -> Path:
    raw = str(value)
    direct = Path(raw)
    if direct.is_file():
        return direct
    marker = "/home/user/training_outputs/terminal/"
    normalized = raw.replace("\\", "/")
    if marker in normalized:
        relative = PurePosixPath(normalized.split(marker, 1)[1])
        local_root = manifest_path.parent.parent
        candidate = local_root.joinpath(*relative.parts)
        if candidate.is_file():
            return candidate
    raise AuditInputError(f"artifact input missing: {raw}")


def _load_records(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = _read_json(manifest_path)
    _require(manifest.get("status") == "COMPLETED", "terminal dev manifest is not COMPLETED")
    _require(int(manifest.get("expected_records", -1)) == 24, "terminal dev manifest expected_records changed")
    _require(int(manifest.get("processed_records", -1)) == 24, "terminal dev manifest processed_records changed")
    _require(int(manifest.get("error_records", 0)) == 0, "terminal dev manifest contains errors")
    groups = manifest.get("groups")
    _require(isinstance(groups, Mapping), "terminal dev manifest groups must be a mapping")
    records_root = manifest_path.parent / "records"
    records: list[dict[str, Any]] = []
    for group, arm in REQUIRED_GROUPS.items():
        _require(group in groups, f"required terminal dev group missing: {group}")
        files = sorted((records_root / group).glob("*.json"))
        _require(len(files) == 6, f"terminal dev group {group} must contain six records")
        for path in files:
            record = _read_json(path)
            _require(record.get("group") == group, f"record group mismatch: {path}")
            _require(record.get("status") == "completed", f"record is not completed: {path}")
            _require(record.get("scenario") in {"flow6", "flow12", "dense24"}, f"record scenario invalid: {path}")
            _require(record.get("seed") is not None and record.get("action_tape"), f"record identity incomplete: {path}")
            record["_record_path"] = str(path)
            record["arm"] = arm
            records.append(record)
    _require(len(records) == 24, "terminal dev record count is not 24")
    identities = {(str(row["scenario"]), int(row["seed"])) for row in records}
    for arm in REQUIRED_GROUPS.values():
        selected = [row for row in records if row["arm"] == arm]
        _require(len({(str(row["scenario"]), int(row["seed"])) for row in selected}) == 6,
                 f"duplicate identities in {arm}")
    source_ids = {(str(row["scenario"]), int(row["seed"])) for row in records if row["arm"] == "source_03c0"}
    for arm in ("final_run0", "final_run1", "final_run2"):
        candidate_ids = {(str(row["scenario"]), int(row["seed"])) for row in records if row["arm"] == arm}
        _require(candidate_ids == source_ids, f"{arm} does not pair the source six seeds")
    _require(len(identities) == 6, "terminal dev seed/scenario identities are not paired")
    return manifest, records


class ReturnAccumulator:
    """Collect one-step rewards without filling missing fields with zero."""

    def __init__(self, *, gamma: float = GAMMA) -> None:
        self.gamma = _finite(gamma, "gamma")
        if not 0.0 < self.gamma <= 1.0:
            raise AuditInputError("gamma must be in (0, 1]")
        self.steps = 0
        self.raw_undiscounted = 0.0
        self.raw_discounted = 0.0
        self.component_undiscounted = {name: 0.0 for name in COMPONENT_NAMES}
        self.component_discounted = {name: 0.0 for name in COMPONENT_NAMES}
        self.progress_by_pre_step_phase: dict[str, dict[str, float]] = {}
        self.raw_components_consistent = True
        self.step_records: list[dict[str, Any]] = []

    def add_step(
        self,
        rewards: Mapping[str, Any],
        info: Mapping[str, Any],
        *,
        pre_step_phase: Mapping[str, Any],
        step_index: int,
    ) -> None:
        _require(isinstance(rewards, Mapping), f"step {step_index} rewards must be a mapping")
        _require(isinstance(info, Mapping), f"step {step_index} info must be a mapping")
        reward_components = info.get("reward_components")
        _require(isinstance(reward_components, Mapping), f"step {step_index} reward_components missing")
        discount = self.gamma ** int(step_index)
        step_components = {name: 0.0 for name in COMPONENT_NAMES}
        step_raw = 0.0
        step_component_total = 0.0
        step_consistent = True
        for uid, raw_reward in rewards.items():
            uid_key = str(uid)
            reward = _finite(raw_reward, f"step {step_index} reward[{uid_key}]")
            components = reward_components.get(uid_key)
            _require(isinstance(components, Mapping), f"step {step_index} reward_components[{uid_key}] missing")
            values: dict[str, float] = {}
            for name in COMPONENT_NAMES:
                _require(name in components, f"step {step_index} component {name} missing for uid {uid_key}")
                values[name] = _finite(components[name], f"step {step_index} component {name}[{uid_key}]")
                step_components[name] += values[name]
            _require("total" in components, f"step {step_index} component total missing for uid {uid_key}")
            component_total = _finite(components["total"], f"step {step_index} component total[{uid_key}]")
            step_raw += reward
            step_component_total += component_total
            step_consistent = step_consistent and math.isclose(
                reward, component_total, rel_tol=1e-6, abs_tol=1e-6,
            ) and math.isclose(
                component_total, sum(values.values()), rel_tol=1e-6, abs_tol=1e-6,
            )
            _require(uid_key in pre_step_phase, f"step {step_index} pre-step phase missing for uid {uid_key}")
            phase = _enum_value(pre_step_phase[uid_key])
            progress = values["progress"]
            phase_values = self.progress_by_pre_step_phase.setdefault(
                phase, {"undiscounted": 0.0, "discounted": 0.0},
            )
            phase_values["undiscounted"] += progress
            phase_values["discounted"] += discount * progress
        step_consistent = step_consistent and math.isclose(
            step_raw, step_component_total, rel_tol=1e-6, abs_tol=1e-6,
        ) and math.isclose(
            step_raw, sum(step_components.values()), rel_tol=1e-6, abs_tol=1e-6,
        )
        self.raw_components_consistent = self.raw_components_consistent and step_consistent
        self.raw_undiscounted += step_raw
        self.raw_discounted += discount * step_raw
        for name in COMPONENT_NAMES:
            self.component_undiscounted[name] += step_components[name]
            self.component_discounted[name] += discount * step_components[name]
        self.steps = max(self.steps, int(step_index) + 1)
        self.step_records.append({
            "step": int(step_index),
            "discount": discount,
            "raw_reward": step_raw,
            "component_total": step_component_total,
            "components": step_components,
            "components_consistent": bool(step_consistent),
        })

    def summary(self) -> dict[str, Any]:
        component_raw_sum = sum(self.component_undiscounted.values())
        component_discounted_sum = sum(self.component_discounted.values())
        return {
            "gamma": self.gamma,
            "steps": self.steps,
            "raw_return_undiscounted": self.raw_undiscounted,
            "raw_return_discounted": self.raw_discounted,
            "components_undiscounted": dict(self.component_undiscounted),
            "components_discounted": dict(self.component_discounted),
            "components_undiscounted_sum": component_raw_sum,
            "components_discounted_sum": component_discounted_sum,
            "raw_components_consistent": bool(
                self.raw_components_consistent
                and math.isclose(self.raw_undiscounted, component_raw_sum, rel_tol=1e-6, abs_tol=1e-6)
                and math.isclose(self.raw_discounted, component_discounted_sum, rel_tol=1e-6, abs_tol=1e-6)
            ),
            "progress_by_pre_step_phase": dict(self.progress_by_pre_step_phase),
        }


def _terminal_summary_from_replay(
    env: TerminalParallelEnv,
    captured: Sequence[Mapping[str, Any]],
    first_seen: Mapping[int, Mapping[str, Any]],
    replay: Mapping[str, Any],
    *,
    low_soc_threshold: float,
) -> dict[str, Any]:
    rows = lifecycle._merge_terminal_rows(captured, lifecycle._completed_rows(env, first_seen))
    requested_count = int(getattr(env.config, "total_requested_uavs"))
    expected_ids = set(range(requested_count))
    observed_ids = {int(row["arrival_id"]) for row in rows}
    missing_ids = sorted(expected_ids - observed_ids)
    unexpected_ids = sorted(observed_ids - expected_ids)
    ledger = getattr(env, "ledger", None)
    ledger_requested = {int(value) for value in getattr(ledger, "requested", set())}
    ledger_values = {
        int(key): _enum_value(value) for key, value in getattr(ledger, "outcomes", {}).items()
    }
    row_values = {int(row["arrival_id"]): str(row.get("outcome")) for row in rows}
    ledger_without_rows = sorted(set(ledger_values) - observed_ids)
    ledger_unresolved_requested = sorted(ledger_requested - set(ledger_values))
    rows_without_ledger_outcome = sorted(observed_ids - set(ledger_values))
    ledger_alignment = not unexpected_ids and not ledger_without_rows and not ledger_unresolved_requested and not rows_without_ledger_outcome and not any(
        arrival_id in row_values and row_values[arrival_id] != outcome
        for arrival_id, outcome in ledger_values.items()
    )
    summary = lifecycle.summarize_terminal_rows(
        rows,
        low_soc_threshold=float(low_soc_threshold),
        requested_count=requested_count,
        missing_arrival_ids=missing_ids,
        enforce_unique=True,
    )
    summary.update({
        "replay_exact": bool(replay.get("exact")),
        "steps_expected": replay.get("steps_expected"),
        "steps_replayed": replay.get("steps_replayed"),
        "requested_arrival_ids": sorted(expected_ids),
        "observed_arrival_ids": sorted(observed_ids),
        "unexpected_arrival_ids": unexpected_ids,
        "ledger_requested_ids": sorted(ledger_requested),
        "ledger_outcomes_without_rows": ledger_without_rows,
        "ledger_unresolved_requested_ids": ledger_unresolved_requested,
        "rows_without_ledger_outcome": rows_without_ledger_outcome,
        "all_requests_observed": not missing_ids,
        "ledger_alignment": ledger_alignment,
    })
    return summary


def _original_summary(tape: Path) -> tuple[Path, dict[str, Any]]:
    summary_path = tape.parent.parent / "summary.json"
    summary = _read_json(summary_path)
    return summary_path, summary


def _replay_record(
    record: Mapping[str, Any],
    *,
    manifest_path: Path,
    configs: Mapping[str, Any],
    low_soc_threshold: float,
    failed_replay_path: Path | None = None,
) -> dict[str, Any]:
    scenario = str(record["scenario"])
    tape = _resolve_artifact(record["action_tape"], manifest_path=manifest_path)
    summary_path, original = _original_summary(tape)
    _require(scenario in configs, f"config missing for scenario {scenario}")
    env = TerminalParallelEnv(configs[scenario])
    captured, first_seen = lifecycle._capture_terminal_rows(env)
    captured_step = env.step
    accumulator = ReturnAccumulator(gamma=GAMMA)

    def observed_step(*args: Any, **kwargs: Any):
        pre_step_phase = {
            str(uid): getattr(uav, "phase", None)
            for uid, uav in getattr(env, "active", {}).items()
        }
        step_index = int(getattr(env, "step_count", 0))
        result = captured_step(*args, **kwargs)
        _require(isinstance(result, tuple) and len(result) == 5, "env.step must return the five-field environment tuple")
        reward_phases = _augment_new_admission_phases(env, pre_step_phase, result[1])
        accumulator.add_step(
            result[1], result[4], pre_step_phase=reward_phases, step_index=step_index,
        )
        return result

    env.step = observed_step  # type: ignore[method-assign]
    try:
        replay = SafetyTraceRecorder.replay_action_tape(env, tape)
    finally:
        env.step = captured_step  # type: ignore[method-assign]
    _require(isinstance(replay, Mapping), f"replay result is not a mapping: {tape}")
    if not bool(replay.get("exact")) and failed_replay_path is not None:
        _write_json(failed_replay_path, replay)
    _require(bool(replay.get("exact")), f"action tape replay is not exact: {tape}")
    _require(int(replay.get("steps_expected", -1)) == accumulator.steps == int(replay.get("steps_replayed", -2)),
             f"replay step count mismatch: {tape}")
    terminal = _terminal_summary_from_replay(
        env, captured, first_seen, replay, low_soc_threshold=low_soc_threshold,
    )
    returns = accumulator.summary()
    _require(
        returns["raw_components_consistent"] is True,
        f"reconstructed raw reward/components mismatch: {tape}",
    )
    _require(
        terminal["ledger_alignment"] is True,
        f"terminal lifecycle ledger is not aligned: {tape}",
    )
    _require(
        terminal["all_requests_observed"] is True,
        f"terminal lifecycle has missing requests: {tape}",
    )
    original_components = original.get("reward_component_totals")
    original_total = original.get("reward_total")
    original_available = isinstance(original_components, Mapping) and original_total is not None
    if original_available:
        original_total_value = _finite(original_total, "original reward_total")
        _require(
            math.isclose(
                returns["raw_return_undiscounted"],
                original_total_value,
                rel_tol=1e-6,
                abs_tol=1e-5,
            ),
            f"reconstructed raw return differs from original summary: {tape}",
        )
    else:
        original_total_value = None if original_total is None else _finite(original_total, "original reward_total")
    outcome_counts = terminal.get("outcomes") if isinstance(terminal.get("outcomes"), Mapping) else {}
    return {
        "stage": STAGE,
        "status": "COMPLETED",
        "arm": str(record["arm"]),
        "group": str(record["group"]),
        "scenario": scenario,
        "seed": int(record["seed"]),
        "action_tape": str(tape),
        "summary_path": str(summary_path),
        "replay": {
            "exact": bool(replay.get("exact")),
            "steps_expected": int(replay.get("steps_expected")),
            "steps_replayed": int(replay.get("steps_replayed")),
        },
        "returns": returns,
        "original_summary": {
            "available": bool(original_available),
            "reward_total": original_total_value,
            "reward_component_totals": None if not isinstance(original_components, Mapping) else {
                str(key): _finite(value, f"original component {key}") for key, value in original_components.items()
            },
            "reward_components_consistent": original.get("reward_components_consistent"),
            "steps": original.get("steps"),
            "outcomes": original.get("outcomes"),
        },
        "terminal_metrics": terminal,
        "outcomes": {
            "normal": int(outcome_counts.get("completed", 0)),
            "emergency": int(outcome_counts.get("emergency_landed", 0)),
            "failed_energy": int(outcome_counts.get("failed_energy", 0)),
            "unresolved": int(terminal.get("missing_request_count", 0)),
        },
        "waits": {
            "mean_wait_seconds": terminal.get("mean_wait_seconds"),
            "priority_weighted_wait_seconds": terminal.get("priority_weighted_wait_seconds"),
            "low_soc_mean_wait_seconds": (terminal.get("low_soc") or {}).get("mean_wait_seconds")
            if isinstance(terminal.get("low_soc"), Mapping) else None,
        },
    }


def _comparison_row(rows: Sequence[Mapping[str, Any]], *, scenario: str, seed: int) -> dict[str, Any]:
    result: dict[str, Any] = {"scenario": scenario, "seed": int(seed)}
    for arm in REQUIRED_GROUPS.values():
        matches = [
            item for item in rows
            if str(item.get("arm")) == arm
            and str(item.get("scenario")) == str(scenario)
            and int(item.get("seed")) == int(seed)
        ]
        _require(
            len(matches) == 1,
            f"comparison identity must have one record: arm={arm}, scenario={scenario}, seed={seed}, count={len(matches)}",
        )
        row = matches[0]
        returns = row["returns"]
        original = row["original_summary"]
        waits = row["waits"]
        outcomes = row["outcomes"]
        result[arm] = {
            "original_reward_total": original.get("reward_total"),
            "reconstructed_raw_return": returns.get("raw_return_undiscounted"),
            "reconstructed_discounted_return": returns.get("raw_return_discounted"),
            "normal": outcomes.get("normal"),
            "emergency": outcomes.get("emergency"),
            "failed_energy": outcomes.get("failed_energy"),
            "unresolved": outcomes.get("unresolved"),
            "mean_wait_seconds": waits.get("mean_wait_seconds"),
            "low_soc_mean_wait_seconds": waits.get("low_soc_mean_wait_seconds"),
            "priority_weighted_wait_seconds": waits.get("priority_weighted_wait_seconds"),
        }
    return result


def _write_progress(output_root: Path, *, status: str, processed: int, expected: int, records: Sequence[str] = (), error: str | None = None) -> None:
    payload: dict[str, Any] = {
        "stage": STAGE,
        "status": status,
        "gamma": GAMMA,
        "processed_records": int(processed),
        "expected_records": int(expected),
        "record_files": list(records),
    }
    if error is not None:
        payload["error"] = str(error)
    _write_json(output_root / "progress.json", payload)


def correct_summary_sidecar(summary_path: Path, sidecar_path: Path) -> dict[str, Any]:
    """Rewrite only the paired comparison section from existing records."""
    summary = _read_json(Path(summary_path))
    records = summary.get("records")
    _require(isinstance(records, list) and len(records) == 24, "summary records must contain 24 records")
    identities = sorted({(str(row.get("scenario")), int(row.get("seed"))) for row in records})
    _require(len(identities) == 6, "summary records must contain six paired identities")
    comparisons = [
        _comparison_row(records, scenario=scenario, seed=seed)
        for scenario, seed in identities
    ]
    identity_counts: dict[str, dict[str, int]] = {}
    for scenario, seed in identities:
        key = f"{scenario}|{seed}"
        identity_counts[key] = {
            arm: sum(
                1 for row in records
                if str(row.get("arm")) == arm
                and str(row.get("scenario")) == scenario
                and int(row.get("seed")) == seed
            )
            for arm in REQUIRED_GROUPS.values()
        }
    checks = {
        "record_count": len(records),
        "pair_count": len(comparisons),
        "pair_identity_count_matches_records": len(comparisons) == len(identities) == 6,
        "each_pair_has_one_record_per_arm": all(
            all(count == 1 for count in counts.values())
            for counts in identity_counts.values()
        ),
        "record_identity_counts": identity_counts,
        "exact_records": sum(bool(row["replay"]["exact"]) for row in records),
        "components_consistent_records": sum(
            bool(row["returns"]["raw_components_consistent"]) for row in records
        ),
        "ledger_aligned_records": sum(
            bool(row["terminal_metrics"]["ledger_alignment"]) for row in records
        ),
        "all_requests_observed_records": sum(
            bool(row["terminal_metrics"]["all_requests_observed"]) for row in records
        ),
        "unresolved_total": sum(int(row["outcomes"]["unresolved"]) for row in records),
    }
    _require(checks["pair_identity_count_matches_records"], "corrected pair identity count mismatch")
    _require(checks["each_pair_has_one_record_per_arm"], "corrected pair identity arm count mismatch")
    corrected = {
        "stage": summary.get("stage"),
        "status": summary.get("status"),
        "mode": summary.get("mode"),
        "gamma": summary.get("gamma"),
        "record_count": len(records),
        "source_summary": str(summary_path),
        "groups": summary.get("groups"),
        "paired_source_vs_candidates": comparisons,
        "checks": checks,
    }
    _write_json(Path(sidecar_path), corrected)
    return corrected


def run(*, manifest_path: Path = DEFAULT_MANIFEST, output_root: Path = DEFAULT_OUTPUT_ROOT, smoke: bool = False) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    output_root = Path(output_root)
    _require(not output_root.exists(), f"refusing existing output root: {output_root}")
    manifest, records = _load_records(manifest_path)
    if smoke:
        records = [
            row for row in records
            if row["arm"] == "source_03c0" and str(row["scenario"]) == "flow6" and int(row["seed"]) == 873101
        ]
        _require(len(records) == 1, "smoke must select source flow6 seed 873101 exactly once")
    protocol = evaluator._protocol()
    configs = evaluator._build_configs(protocol)
    low_soc_threshold = float((protocol.get("reward") or {}).get("low_level_soc_urgency_threshold", .65))
    output_root.mkdir(parents=True, exist_ok=False)
    expected = len(records)
    _write_progress(output_root, status="RUNNING", processed=0, expected=expected)
    output_records: list[dict[str, Any]] = []
    record_files: list[str] = []
    try:
        for index, record in enumerate(records, start=1):
            result = _replay_record(
                record,
                manifest_path=manifest_path,
                configs=configs,
                low_soc_threshold=low_soc_threshold,
                failed_replay_path=output_root / "failed_replay.json",
            )
            output_path = output_root / "records" / str(record["arm"]) / f"{record['scenario']}_{int(record['seed'])}.json"
            _write_json(output_path, result)
            output_records.append(result)
            record_files.append(str(output_path))
            _write_progress(output_root, status="RUNNING", processed=index, expected=expected, records=record_files)
        comparisons = [
            _comparison_row(output_records, scenario=scenario, seed=seed)
            for scenario, seed in sorted({(str(row["scenario"]), int(row["seed"])) for row in output_records})
            if len([row for row in output_records if str(row["scenario"]) == scenario and int(row["seed"]) == seed]) == len(REQUIRED_GROUPS)
        ]
        groups_summary: dict[str, Any] = {}
        for arm in REQUIRED_GROUPS.values():
            selected = [row for row in output_records if row["arm"] == arm]
            groups_summary[arm] = {
                "records": len(selected),
                "raw_return_undiscounted_sum": sum(float(row["returns"]["raw_return_undiscounted"]) for row in selected),
                "raw_return_discounted_sum": sum(float(row["returns"]["raw_return_discounted"]) for row in selected),
                "normal": sum(int(row["outcomes"]["normal"]) for row in selected),
                "emergency": sum(int(row["outcomes"]["emergency"]) for row in selected),
                "failed_energy": sum(int(row["outcomes"]["failed_energy"]) for row in selected),
                "unresolved": sum(int(row["outcomes"]["unresolved"]) for row in selected),
            }
        summary = {
            "stage": STAGE,
            "status": "COMPLETED",
            "mode": "smoke" if smoke else "full",
            "manifest": str(manifest_path),
            "manifest_status": manifest.get("status"),
            "gamma": GAMMA,
            "record_count": len(output_records),
            "groups": groups_summary,
            "paired_source_vs_candidates": comparisons,
            "records": output_records,
        }
        _write_json(output_root / "summary.json", summary)
        _write_progress(output_root, status="COMPLETED", processed=expected, expected=expected, records=record_files)
        return summary
    except Exception as exc:
        _write_progress(output_root, status="FAILED", processed=len(output_records), expected=expected, records=record_files, error=f"{type(exc).__name__}: {exc}")
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(manifest_path=args.manifest, output_root=args.output_root, smoke=args.smoke)
    except (AuditInputError, FileNotFoundError, OSError, ValueError) as exc:
        print(json.dumps({"stage": STAGE, "status": "ERROR", "error": str(exc)}, sort_keys=True), flush=True)
        return 2
    print(json.dumps({"stage": STAGE, "status": result["status"], "mode": result["mode"], "record_count": result["record_count"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
