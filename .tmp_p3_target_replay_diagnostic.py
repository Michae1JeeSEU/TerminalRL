"""Observe selected target UAVs during exact replay of a formal P3 tape.

This is a bounded forensic helper for the three recorded formal episodes.  It
does not sample a policy or alter the simulator decisions.  The environment
methods are wrapped on one instance only; every wrapper records state around
the original call and returns the original result unchanged.  The report is
written only after ``SafetyTraceRecorder.replay_action_tape`` reports an exact
replay.
"""
from __future__ import annotations

import argparse
import copy
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from terminal_marl.core.config import RingRole
from terminal_marl.core.low_level_joint_action_v4_2 import (
    hierarchical_from_normalized_v4_2,
)
from terminal_marl.core.safety_trace import SafetyTraceRecorder
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training.v4_1_joint_teacher import (
    teacher_joint_expert_v4_1,
    teacher_joint_label_v4_1,
)
from terminal_marl.visualization import p3_formal_replay


STAGE = "P3_FORMAL_TARGET_REPLAY_DIAGNOSTIC_V1"
TARGETS_BY_RUN = {0: (4,), 1: (230, 271), 2: (108,)}
FOCUS_TIMES_BY_UID = {
    4: (250.0, 300.0, 320.0, 430.0, 850.0),
    230: (300.0, 430.0, 500.0, 850.0),
    271: (280.0, 330.0, 350.0, 390.0, 400.0, 414.0, 430.0, 850.0),
    108: (200.0, 300.0, 382.0, 430.0, 436.0, 440.0, 450.0, 480.0, 489.0, 850.0),
}
FOCUS_RADIUS = 2.0
SHADOW_TIMES_BY_UID = {
    4: (120.0, 200.0, 280.0),
    230: (300.0, 500.0),
    271: (280.0, 330.0, 390.0),
    108: (200.0, 300.0, 440.0, 480.0),
}


def _jsonable(value: Any) -> Any:
    """Convert observed runtime values without invoking simulator behavior."""

    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "value") and isinstance(value.value, str):
        return value.value
    return repr(value)


def _copy_json(value: Any) -> Any:
    """Take a detached, JSON-safe snapshot of an observed object."""

    return json.loads(json.dumps(_jsonable(value), sort_keys=True, allow_nan=False))


def _telemetry_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    keys = sorted(set(before) | set(after), key=str)
    result: dict[str, Any] = {}
    for key in keys:
        left = before.get(key, 0)
        right = after.get(key, 0)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            result[str(key)] = right - left
    return result


def _find_uav(env: TerminalParallelEnv, uid: int) -> Any | None:
    if uid in env.active:
        return env.active[uid]
    if uid in env.service_uavs:
        return env.service_uavs[uid]
    for uav in env.completed_uavs:
        if int(uav.uid) == uid:
            return uav
    return None


def _uav_state(env: TerminalParallelEnv, uid: int) -> dict[str, Any]:
    uav = _find_uav(env, int(uid))
    if uav is None:
        return {"uid": int(uid), "present": False}
    return {
        "uid": int(uid),
        "present": True,
        "arrival_id": int(uav.arrival_id),
        "phase": uav.phase.value,
        "ring_role": None if uav.ring_role is None else uav.ring_role.value,
        "soc": float(uav.soc),
        "energy_remaining": float(uav.energy_remaining),
        "energy_consumed": float(uav.energy_consumed),
        "committed": bool(uav.committed_plan is not None),
        "assigned_lz": None if uav.assigned_lz is None else int(uav.assigned_lz),
        "emergency_landing": bool(uav.emergency_landing),
        "safety_retry_after": float(uav.safety_retry_after),
        "emergency_retry_after": float(uav.emergency_retry_after),
        "dynamic_priority": float(uav.dynamic_priority),
        "external_wait_time": float(uav.external_wait_time),
        "internal_wait_time": float(uav.internal_wait_time),
        "terminal_outcome": (
            None if uav.terminal_outcome is None else uav.terminal_outcome.value
        ),
        "terminal_reason": uav.terminal_reason,
    }


def _ring_occupancy(env: TerminalParallelEnv) -> dict[str, Any]:
    return {
        "actual": {
            role.value: int(env.airspace.occupancy(role)) for role in RingRole
        },
        "committed_incoming": {
            role.value: int(env.airspace._committed.get(role, 0))
            for role in RingRole
        },
        "active_ring_holders": {
            role.value: sum(
                1 for uav in env.active.values() if uav.ring_role is role
            )
            for role in RingRole
        },
    }


def _world_state(env: TerminalParallelEnv) -> dict[str, Any]:
    return {
        "time": float(env.now),
        "step": int(env.step_count),
        "active_count": int(len(env.active)),
        "queue_count": int(len(env.admission.queue.items)),
        "service_count": int(len(env.service_uavs)),
        "completed_count": int(len(env.completed_uavs)),
        "ring_occupancy": _ring_occupancy(env),
    }


def _lz_state(env: TerminalParallelEnv, uid: int | None = None) -> list[dict[str, Any]]:
    now = float(env.now)
    result: list[dict[str, Any]] = []
    for zone in env.lz_manager.zones:
        available_at = env.lz_manager.earliest_release_time(zone.lz_id, now=now)
        result.append(
            {
                "lz_id": int(zone.lz_id),
                "status": zone.status.value,
                "reserved_by": None if zone.reserved_by is None else int(zone.reserved_by),
                "reservation_start": zone.reservation_start,
                "expected_release_at": zone.expected_release_at,
                "occupied_by": None if zone.occupied_by is None else int(zone.occupied_by),
                "service_remaining": float(zone.service_remaining),
                "available_at": float(available_at) if math.isfinite(available_at) else None,
                "mission_compatible": (
                    None
                    if uid is None or _find_uav(env, uid) is None
                    else bool(_find_uav(env, uid).mission_type in zone.compatible_missions)
                ),
            }
        )
    return result


def _energy_diagnostic(env: TerminalParallelEnv, uid: int) -> dict[str, Any]:
    uav = _find_uav(env, uid)
    diagnostic = env.energy_reachability_diagnostics.get(uid)
    result = {} if diagnostic is None else dict(diagnostic)
    result["current_energy"] = None if uav is None else float(uav.energy_remaining)
    result["current_soc"] = None if uav is None else float(uav.soc)
    return _copy_json(result)


def _safety_decisions(env: TerminalParallelEnv) -> dict[str, Any]:
    return {str(uid): str(reason) for uid, reason in env.last_safety_decisions.items()}


def _requested_gate_proposal(actions: Any, uid: int) -> dict[str, Any]:
    if not isinstance(actions, Mapping):
        return {"gate": None, "proposal": None}
    raw = actions.get(str(uid), actions.get(uid))
    if raw is None:
        return {"gate": None, "proposal": None}
    try:
        decoded = hierarchical_from_normalized_v4_2(
            np.asarray(raw, dtype=np.float32)
        )
        return {
            "gate": int(decoded.gate),
            "proposal": None if decoded.proposal is None else int(decoded.proposal),
        }
    except (TypeError, ValueError):
        return {"gate": None, "proposal": None, "decode_error": True}


def _intent_payload(intent: Any) -> dict[str, Any]:
    return {
        "release_request": float(intent.release_request),
        "speed_residual": float(intent.speed_residual),
        "merge_time_residual": float(intent.merge_time_residual),
    }


def _shortlist_payload(shortlist: Any) -> list[dict[str, Any]] | None:
    if shortlist is None:
        return None
    return [_intent_payload(intent) for intent in tuple(shortlist)]


def _release_row(env: TerminalParallelEnv, uid: int) -> Any:
    return _copy_json(env.last_release_telemetry.get(uid))


def _focus_detail(env: TerminalParallelEnv, uid: int) -> dict[str, Any]:
    """Capture energy, LZ, safety, and release fields for a focus row."""

    return {
        "energy_diagnostic": _energy_diagnostic(env, uid),
        "lz": _lz_state(env, uid),
        "safety_decisions": _safety_decisions(env),
        "release_telemetry": _release_row(env, uid),
    }


def _focus_due(uid: int, time_value: float) -> bool:
    return any(
        abs(float(time_value) - focus) <= FOCUS_RADIUS
        for focus in FOCUS_TIMES_BY_UID.get(uid, ())
    )


def _shadow_due(uid: int, time_value: float) -> bool:
    return any(abs(float(time_value) - focus) <= 1e-9 for focus in SHADOW_TIMES_BY_UID.get(uid, ()))


def _restore_original_methods(shadow: TerminalParallelEnv) -> None:
    """Remove instance wrappers from a deepcopy before the label probe."""

    cls = type(shadow)
    for name in (
        "step",
        "_release",
        "_try_emergency_landing",
        "_candidate_energy_guard_reason",
    ):
        setattr(shadow, name, getattr(cls, name).__get__(shadow, cls))


def _shadow_state_only_label(env: TerminalParallelEnv, uid: int) -> dict[str, Any]:
    """Probe observation and state-only Teacher only on an isolated deepcopy."""

    current = _find_uav(env, uid)
    if current is None or uid not in env.active:
        return {
            "uid": int(uid),
            "status": "not_active",
            "observation": None,
            "observation_dimension": None,
            "selected_label": None,
            "certified_release_candidates": [],
        }

    try:
        observations = env._observations()
        observation = observations.get(str(uid))
        if observation is None:
            raise RuntimeError(f"observation missing for active uid={uid}")
        shadow = copy.deepcopy(env)
        _restore_original_methods(shadow)
        shadow_uav = shadow.active[int(uid)]
        selected_label = teacher_joint_label_v4_1(
            shadow, shadow_uav, state_only=True
        )
        _selected_again, audit = teacher_joint_expert_v4_1(
            shadow, shadow_uav, audit_catalogue=True, state_only=True
        )
        certified = [
            {
                "rank": int(row.get("rank", -1)),
                "joint_index": int(row.get("joint_index", -1)),
                "decoded": _copy_json(row.get("decoded", {})),
                "reason": row.get("reason"),
            }
            for row in audit
            if bool(row.get("feasible"))
            and float(row.get("decoded", {}).get("release_request", 0.0)) > 0.0
        ]
        return {
            "uid": int(uid),
            "status": "evaluated",
            "observation": _jsonable(np.asarray(observation, dtype=float)),
            "observation_dimension": int(np.asarray(observation).size),
            "selected_label": _copy_json(selected_label.as_dict()),
            "certified_release_candidates": certified,
            "certified_release_exists": bool(certified),
            "audit_count": len(audit),
        }
    except Exception as exc:
        return {
            "uid": int(uid),
            "status": "probe_error",
            "observation": None,
            "observation_dimension": None,
            "selected_label": None,
            "certified_release_candidates": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def _install_observers(
    env: TerminalParallelEnv,
    targets: tuple[int, ...],
    timeline: list[dict[str, Any]],
    releases: list[dict[str, Any]],
    emergencies: list[dict[str, Any]],
    energy_candidates: list[dict[str, Any]],
    shadow_probes: list[dict[str, Any]],
) -> None:
    target_set = set(targets)
    shadow_done: set[tuple[int, float]] = set()

    original_step = env.step

    def audited_step(*args: Any, **kwargs: Any) -> Any:
        actions = args[0] if args else kwargs.get("actions")
        pre = {
            "world": _world_state(env),
            "targets": {
                str(uid): _uav_state(env, uid) for uid in targets
            },
            "requested": {
                str(uid): _requested_gate_proposal(actions, uid) for uid in targets
            },
        }
        for uid in targets:
            focus_key = (uid, float(env.now))
            if _shadow_due(uid, float(env.now)) and focus_key not in shadow_done:
                shadow_done.add(focus_key)
                probe = _shadow_state_only_label(env, uid)
                probe.update({"time": float(env.now), "step": int(env.step_count)})
                shadow_probes.append(probe)
        pre_focus = {
            str(uid): _focus_detail(env, uid)
            for uid in targets
            if _focus_due(uid, float(env.now))
        }
        if pre_focus:
            pre["focus_detail"] = pre_focus
        error: str | None = None
        result: Any = None
        try:
            result = original_step(*args, **kwargs)
            return result
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            post = {
                "world": _world_state(env),
                "targets": {
                    str(uid): _uav_state(env, uid) for uid in targets
                },
            }
            post_focus = {
                str(uid): _focus_detail(env, uid)
                for uid in targets
                if _focus_due(uid, float(env.now))
            }
            if post_focus:
                post["focus_detail"] = post_focus
            timeline.append(
                {
                    "step": int(pre["world"]["step"]),
                    "time_before": float(pre["world"]["time"]),
                    "time_after": float(env.now),
                    "pre": pre,
                    "post": post,
                    **({"error": error} if error is not None else {}),
                }
            )

    original_release = env._release

    def audited_release(uav: Any, intent: Any, shortlist: Any = None) -> Any:
        uid = int(uav.uid)
        if uid not in target_set:
            return original_release(uav, intent, shortlist)
        before_telemetry = dict(env.telemetry)
        before_row = _release_row(env, uid)
        before = {
            "time": float(env.now),
            "step": int(env.step_count),
            "state": _uav_state(env, uid),
            "intent": _intent_payload(intent),
            "shortlist": _shortlist_payload(shortlist),
            "energy_diagnostic": _energy_diagnostic(env, uid),
            "lz": _lz_state(env, uid),
            "safety_decisions": _safety_decisions(env),
            "release_telemetry": before_row,
        }
        result: Any = None
        error: str | None = None
        try:
            result = original_release(uav, intent, shortlist)
            return result
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            after_telemetry = dict(env.telemetry)
            after_row = _release_row(env, uid)
            releases.append(
                {
                    "time": float(before["time"]),
                    "step": int(before["step"]),
                    "uid": uid,
                    "intent": before["intent"],
                    "shortlist": before["shortlist"],
                    "returned": result,
                    "error": error,
                    "before": before,
                    "after": {
                        "state": _uav_state(env, uid),
                        "energy_diagnostic": _energy_diagnostic(env, uid),
                        "lz": _lz_state(env, uid),
                        "safety_decisions": _safety_decisions(env),
                        "release_telemetry": after_row,
                        "telemetry_delta": _telemetry_delta(
                            before_telemetry, after_telemetry
                        ),
                    },
                }
            )

    original_emergency = env._try_emergency_landing

    def audited_emergency(uav: Any, *, force: bool = False) -> bool:
        uid = int(uav.uid)
        if uid not in target_set:
            return original_emergency(uav, force=force)
        before_telemetry = dict(env.telemetry)
        before = {
            "time": float(env.now),
            "step": int(env.step_count),
            "uid": uid,
            "force": bool(force),
            "state": _uav_state(env, uid),
            "energy_diagnostic": _energy_diagnostic(env, uid),
            "lz": _lz_state(env, uid),
            "safety_decisions": _safety_decisions(env),
        }
        result: bool | None = None
        error: str | None = None
        try:
            result = bool(original_emergency(uav, force=force))
            return result
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            after_telemetry = dict(env.telemetry)
            emergencies.append(
                {
                    "time": float(before["time"]),
                    "step": int(before["step"]),
                    "uid": uid,
                    "force": bool(force),
                    "returned": result,
                    "error": error,
                    "before": before,
                    "after": {
                        "state": _uav_state(env, uid),
                        "energy_diagnostic": _energy_diagnostic(env, uid),
                        "lz": _lz_state(env, uid),
                        "safety_decisions": _safety_decisions(env),
                        "telemetry_delta": _telemetry_delta(
                            before_telemetry, after_telemetry
                        ),
                    },
                }
            )

    original_energy_guard = env._candidate_energy_guard_reason

    def audited_energy_guard(uav: Any, plan: Any, target: Any) -> Any:
        uid = int(uav.uid)
        result = original_energy_guard(uav, plan, target)
        if uid in target_set:
            energy_candidates.append(
                {
                    "time": float(env.now),
                    "step": int(env.step_count),
                    "uid": uid,
                    "target": getattr(target, "value", target),
                    "returned": result,
                    "plan_id": getattr(plan, "plan_id", None),
                    "plan_metadata": _copy_json(getattr(plan, "metadata", {})),
                    "energy": _energy_diagnostic(env, uid),
                    "lz": _lz_state(env, uid),
                }
            )
        return result

    # Instance assignment deliberately keeps these as plain closures.  The
    # original bound methods are retained and invoked with the exact inputs.
    env.step = audited_step  # type: ignore[method-assign]
    env._release = audited_release  # type: ignore[method-assign]
    env._try_emergency_landing = audited_emergency  # type: ignore[method-assign]
    env._candidate_energy_guard_reason = audited_energy_guard  # type: ignore[method-assign]


def _phase_segments_fixed(timeline: list[dict[str, Any]], uid: int) -> list[dict[str, Any]]:
    """Build contiguous phase intervals from the recorded post-step rows."""

    segments: list[dict[str, Any]] = []
    for row in timeline:
        target = row["post"]["targets"][str(uid)]
        key = (
            target.get("phase") if target.get("present") else None,
            bool(target.get("present")),
        )
        if not segments or segments[-1]["_key"] != key:
            segments.append(
                {
                    "_key": key,
                    "step_start": int(row["step"]),
                    "step_end": int(row["step"]),
                    "time_start": float(row["time_after"]),
                    "time_end": float(row["time_after"]),
                    "soc_start": target.get("soc"),
                    "soc_end": target.get("soc"),
                }
            )
        else:
            segments[-1]["step_end"] = int(row["step"])
            segments[-1]["time_end"] = float(row["time_after"])
            segments[-1]["soc_end"] = target.get("soc")
    for segment in segments:
        phase, present = segment.pop("_key")
        segment["phase"] = phase
        segment["present"] = present
    return segments


def _focus_rows(timeline: list[dict[str, Any]], uid: int) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for focus in FOCUS_TIMES_BY_UID.get(uid, ()):
        result[str(int(focus))] = [
            row
            for row in timeline
            if abs(float(row["time_before"]) - focus) <= FOCUS_RADIUS
            or abs(float(row["time_after"]) - focus) <= FOCUS_RADIUS
        ]
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_index(manifest: Mapping[str, Any], run_path: Path) -> int:
    raw = manifest.get("run_index", run_path.name.removeprefix("run"))
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise p3_formal_replay.ReplayInputError(
            f"run index is invalid: {raw!r}"
        ) from exc


def replay_diagnostic(run_dir: str | Path, output: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir).resolve()
    output_path = Path(output).resolve()
    if not run_path.is_dir():
        raise p3_formal_replay.ReplayInputError(f"run directory missing: {run_path}")
    if output_path.exists():
        raise p3_formal_replay.ReplayInputError(
            f"refusing to overwrite existing output: {output_path}"
        )
    if output_path == run_path or run_path in output_path.parents:
        raise p3_formal_replay.ReplayInputError(
            "diagnostic output must be independent of the formal run directory"
        )

    manifest_path = run_path / "manifest.json"
    manifest = p3_formal_replay._read_json(manifest_path)
    record_index, record = p3_formal_replay._last_failed_record(manifest)
    run_index = _run_index(manifest, run_path)
    targets = TARGETS_BY_RUN.get(run_index)
    if targets is None:
        raise p3_formal_replay.ReplayInputError(
            f"unsupported formal run index for bounded targets: {run_index}"
        )
    checkpoint, checkpoint_sha, checkpoint_fields = p3_formal_replay._bind_checkpoint(
        manifest, record, run_path
    )
    raw_tape = record.get("action_tape")
    tape = p3_formal_replay._resolve_artifact(raw_tape, run_path, "episodes")
    scenario = str(record.get("scenario") or "")
    if not scenario:
        raise p3_formal_replay.ReplayInputError("failed record scenario is missing")
    try:
        seed = int(record["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise p3_formal_replay.ReplayInputError("failed record seed is invalid") from exc

    config = p3_formal_replay._scenario_config(manifest, scenario)
    env = TerminalParallelEnv(config)
    timeline: list[dict[str, Any]] = []
    releases: list[dict[str, Any]] = []
    emergencies: list[dict[str, Any]] = []
    energy_candidates: list[dict[str, Any]] = []
    shadow_probes: list[dict[str, Any]] = []
    _install_observers(
        env,
        targets,
        timeline,
        releases,
        emergencies,
        energy_candidates,
        shadow_probes,
    )
    replay = SafetyTraceRecorder.replay_action_tape(env, tape)
    if replay.get("exact") is not True:
        return {
            "stage": STAGE,
            "run_index": run_index,
            "run_dir": str(run_path),
            "record_index": record_index,
            "action_tape": str(tape),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "replay": {
                key: value for key, value in replay.items() if key != "steps"
            },
            "exact": False,
            "diagnostic_written": False,
        }

    target_reports: dict[str, Any] = {}
    for uid in targets:
        target_releases = [row for row in releases if row["uid"] == uid]
        target_emergencies = [row for row in emergencies if row["uid"] == uid]
        target_energy = [row for row in energy_candidates if row["uid"] == uid]
        target_reports[str(uid)] = {
            "phase_segments": _phase_segments_fixed(timeline, uid),
            "timeline": timeline,
            "focus_windows": _focus_rows(timeline, uid),
            "release_attempts": target_releases,
            "safety_rejects": [
                row
                for row in target_releases
                if row.get("returned") == "safety_reject"
            ],
            "emergency_attempts": target_emergencies,
            "candidate_energy_checks": target_energy,
            "state_only_shadow_probes": [
                row for row in shadow_probes if row["uid"] == uid
            ],
        }

    report: dict[str, Any] = {
        "stage": STAGE,
        "run_index": run_index,
        "run_dir": str(run_path),
        "manifest": str(manifest_path),
        "record_index": record_index,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "record_checkpoint_sha256_fields": checkpoint_fields,
        "action_tape": str(tape),
        "action_tape_sha256": _sha256(tape),
        "scenario": scenario,
        "seed": seed,
        "targets": list(targets),
        "original_outcomes": p3_formal_replay._record_outcomes(record),
        "replay": {
            key: value for key, value in replay.items() if key != "steps"
        },
        "exact": True,
        "diagnostic_written": True,
        "target_reports": target_reports,
    }
    output_path.mkdir(parents=True, exist_ok=False)
    report_path = output_path / "target_replay_diagnostic.json"
    report_path.write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Observe fixed formal P3 target UAVs during exact action-tape replay."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = replay_diagnostic(args.run_dir, args.output)
    except Exception as exc:
        print(
            json.dumps(
                {"stage": STAGE, "exact": False, "error": f"{type(exc).__name__}: {exc}"},
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(_jsonable(report), indent=2, sort_keys=True))
    return 0 if report.get("exact") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
