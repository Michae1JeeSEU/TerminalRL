"""Export each E04 coverage-repair evaluation tape to the existing viewer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from terminal_marl.core.safety_trace import SafetyTraceRecorder
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training import (
    diagnose_p3_bounded_gate_v1 as bounded,
    diagnose_v4_2_p3_mixed_priority_soc_pilot_v2 as lifecycle,
    evaluate_random_soc_gate_repair_v1 as base,
)
from terminal_marl.visualization.snapshot_playback import SnapshotRecorder, write_playback


EVALUATION_ROOT = Path(
    "/home/user/training_outputs/terminal/"
    "p3_gate_coverage_repair_v2_20260925/evaluation"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/user/training_outputs/terminal/"
    "p3_formal_policy_visualizations_20260925"
)
FRAME_STRIDE = 2
TARGET_ARRIVALS = {0: (4,), 1: (18, 20), 2: (16,)}


def _jsonable(value: Any) -> Any:
    return base._jsonable(value)


def _read_summary(run_index: int) -> dict[str, Any]:
    path = EVALUATION_ROOT / f"run{int(run_index)}" / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"summary must be an object: {path}")
    if value.get("source") != "E04 gate coverage repair":
        raise ValueError(f"unexpected evaluation source: {value.get('source')!r}")
    return value


def schema_check() -> dict[str, Any]:
    config = bounded.build_config()
    if str(config.v4_2_rejected_release_hold_mode) != "canonical_wait":
        raise ValueError("coverage replay requires canonical_wait")
    env = TerminalParallelEnv(config)
    env.reset(850218)
    if not env.active:
        raise ValueError("schema check reset produced no active UAV")
    sample = next(iter(env.active.values()))
    required = (
        "uid", "arrival_id", "base_priority", "energy_remaining", "energy_consumed",
        "energy_capacity", "soc", "external_wait_time", "internal_wait_time",
        "request_time", "touchdown_time",
    )
    missing = [name for name in required if not hasattr(sample, name)]
    if missing:
        raise ValueError(f"lifecycle row fields missing: {missing}")
    return {
        "schema_check": "passed",
        "required_fields": list(required),
        "action_schema_version": str(config.action_schema_version),
        "observation_schema_version": str(config.observation_schema_version),
        "rejected_release_hold_mode": str(config.v4_2_rejected_release_hold_mode),
    }


def _lifecycle_from_env(env: TerminalParallelEnv) -> dict[str, Any]:
    protocol = base._load_yaml_protocol()
    threshold = float(protocol["attributes"]["low_soc_analysis_threshold"])
    rows: list[dict[str, Any]] = []
    for uav in env.completed_uavs:
        initial_soc = float((uav.energy_remaining + uav.energy_consumed) / uav.energy_capacity)
        rows.append({
            "uid": int(uav.uid),
            "arrival_id": int(uav.arrival_id),
            "base_priority": int(uav.base_priority),
            "initial_soc": initial_soc,
            "landing_soc": float(uav.soc),
            "wait_seconds": float(uav.external_wait_time + uav.internal_wait_time),
            "request_to_touchdown_seconds": float(uav.touchdown_time - uav.request_time),
        })
    return {
        "replay_exact": True,
        **lifecycle.summarize_lifecycle_rows(rows, low_soc_threshold=threshold),
    }


def _frame_observations(frames: list[dict[str, Any]], arrival_id: int) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for frame in frames:
        for uav in frame.get("uavs", []):
            if int(uav.get("arrival_id", -1)) == int(arrival_id):
                observations.append({
                    "time": float(frame["time"]),
                    "step_count": int(frame["step_count"]),
                    "phase": str(uav.get("phase", "")),
                    "ring_role": uav.get("ring_role"),
                    "uid": int(uav["uid"]),
                    "soc": float(uav.get("soc", 0.0)),
                })
    return observations


def _step_observations(history: list[dict[str, Any]], arrival_id: int) -> list[dict[str, Any]]:
    return [
        row for row in history
        if int(row["arrival_id"]) == int(arrival_id)
    ]


def _ring_transitions(observations: list[dict[str, Any]]) -> tuple[dict[str, float], str | None]:
    transitions: dict[str, float] = {}
    previous_ring: str | None = None
    last_ring: str | None = None
    for row in observations:
        ring = row.get("ring_role")
        if ring is None:
            continue
        ring = str(ring)
        if previous_ring is not None and ring != previous_ring:
            transitions.setdefault(f"{previous_ring}->{ring}", float(row["time"]))
        previous_ring = ring
        last_ring = ring
    return transitions, last_ring


def _target_report(
    frames: list[dict[str, Any]],
    step_history: list[dict[str, Any]],
    emergency_events: list[dict[str, Any]],
    env: TerminalParallelEnv,
    arrival_id: int,
) -> dict[str, Any]:
    frame_observations = _frame_observations(frames, arrival_id)
    frame_transitions, frame_last_ring = _ring_transitions(frame_observations)
    step_observations = _step_observations(step_history, arrival_id)
    step_transitions, step_last_ring = _ring_transitions(step_observations)

    terminal = [
        row for row in env.energy_outcomes if int(row["arrival_id"]) == int(arrival_id)
    ]
    terminal_row = terminal[-1] if terminal else None
    return {
        "arrival_id": int(arrival_id),
        "uid": None if terminal_row is None else terminal_row.get("uid"),
        "outcome": None if terminal_row is None else terminal_row.get("outcome"),
        "reason": None if terminal_row is None else terminal_row.get("reason"),
        "terminal_time": None if terminal_row is None else terminal_row.get("time"),
        "last_ring_from_steps": step_last_ring,
        "ring_entry_times_from_steps": step_transitions,
        "step_observations": len(step_observations),
        "first_step_time": None if not step_observations else step_observations[0]["time"],
        "last_step_time": None if not step_observations else step_observations[-1]["time"],
        "last_phase_from_steps": None if not step_observations else step_observations[-1]["phase"],
        "last_soc_from_steps": None if not step_observations else step_observations[-1]["soc"],
        "emergency_origins": [
            dict(event) for event in emergency_events
            if int(event["arrival_id"]) == int(arrival_id)
        ],
        "last_observed_frame_ring": frame_last_ring,
        "ring_entry_times_from_stride2_frames": frame_transitions,
        "frame_observations": len(frame_observations),
        "first_observed_frame_time": None if not frame_observations else frame_observations[0]["time"],
        "last_observed_frame_time": None if not frame_observations else frame_observations[-1]["time"],
        "last_observed_frame_phase": None if not frame_observations else frame_observations[-1]["phase"],
        "last_observed_frame_soc": None if not frame_observations else frame_observations[-1]["soc"],
        "frame_scope": "stride2 snapshots; not used for actual emergency origin",
    }


def _o_to_emergency(emergency_events: list[dict[str, Any]]) -> dict[str, Any]:
    arrivals = [
        event for event in emergency_events
        if event.get("before_ring_role") == "outer"
    ]
    return {
        "count": len(arrivals),
        "arrivals": [
            {
                "arrival_id": int(event["arrival_id"]),
                "uid": int(event["uid"]),
                "time": float(event["time"]),
                "before_phase": event.get("before_phase"),
                "before_ring_role": event.get("before_ring_role"),
            }
            for event in arrivals
        ],
        "method": "emergency_departure events with pre-step before_ring_role=outer",
    }


def _install_step_trace(env: TerminalParallelEnv) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    step_history: list[dict[str, Any]] = []
    emergency_events: list[dict[str, Any]] = []
    original_step = env.step

    def traced_step(actions=None, proposal_shortlists=None):
        before_by_uid = {}
        for uid, uav in env.active.items():
            before_by_uid[int(uid)] = {
                "uid": int(uid),
                "arrival_id": int(uav.arrival_id),
                "time": float(env.now),
                "step_count": int(env.step_count),
                "phase": getattr(uav.phase, "value", str(uav.phase)),
                "ring_role": None if uav.ring_role is None else getattr(uav.ring_role, "value", str(uav.ring_role)),
                "soc": float(uav.soc),
            }
            step_history.append(dict(before_by_uid[int(uid)]))
        event_start = len(env.events)
        result = original_step(actions, proposal_shortlists)
        for event in env.events[event_start:]:
            if not event or event[0] != "emergency_departure":
                continue
            uid = int(event[1])
            before = before_by_uid.get(uid)
            if before is None:
                continue
            emergency_events.append({
                "uid": uid,
                "arrival_id": int(before["arrival_id"]),
                "time": float(event[2]),
                "before_phase": before["phase"],
                "before_ring_role": before["ring_role"],
                "before_soc": before["soc"],
                "before_step_count": before["step_count"],
            })
        return result

    env.step = traced_step
    return step_history, emergency_events


def _prepare_replay_output(output_dir: Path) -> None:
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=False)
        return
    current = output_dir / "replay_report.json"
    backup = output_dir / "replay_report_frame_stride2.json"
    if not current.is_file():
        raise FileExistsError(f"refusing output without existing replay report: {output_dir}")
    if backup.exists():
        raise FileExistsError(f"refusing to overwrite preserved report: {backup}")
    current.replace(backup)


def export_run(run_index: int, output_dir: Path) -> dict[str, Any]:
    source = _read_summary(run_index)
    tape = Path(str((source.get("rollout") or {}).get("action_tape", "")))
    if not tape.is_file():
        raise FileNotFoundError(tape)
    config = bounded.build_config()
    env = TerminalParallelEnv(config)
    recorder = SnapshotRecorder(FRAME_STRIDE)
    env.set_step_observer(recorder)
    step_history, emergency_events = _install_step_trace(env)
    replay = SafetyTraceRecorder.replay_action_tape(env, tape)
    exact = bool(replay.get("exact") is True)
    if not exact:
        raise ValueError(f"coverage tape replay was not exact: run{run_index}")
    recorder.capture(env)
    if len(env.completed_uavs) != 24 or len(env.energy_outcomes) != 24:
        raise ValueError(
            f"expected 24 completed/outcome rows, got {len(env.completed_uavs)}/{len(env.energy_outcomes)}"
        )
    seed = int(source["seed"])
    title = f"E04 coverage repair candidate · original failed seed {seed} · run{run_index}"
    _prepare_replay_output(output_dir)
    html_path = output_dir / "recorded_stochastic_policy_replay.html"
    write_playback(html_path, recorder.frames, title)
    report = {
        "stage": "P3_E04_GATE_COVERAGE_REPAIR_RECORDED_POLICY_REPLAY_V1",
        "source": "E04 coverage repair candidate, corresponding old failed seed",
        "policy_role": "coverage_repair_not_old_run_final_policy",
        "run_index": int(run_index),
        "seed": seed,
        "scenario": str(source.get("scenario", "dense24")),
        "source_summary": str(EVALUATION_ROOT / f"run{int(run_index)}" / "summary.json"),
        "action_tape": str(tape),
        "steps_replayed": replay.get("steps_replayed"),
        "steps_expected": replay.get("steps_expected"),
        "exact": exact,
        "title": title,
        "frame_stride": FRAME_STRIDE,
        "playback_html": str(html_path),
        "lifecycle": _lifecycle_from_env(env),
        "target_arrivals": {
            str(arrival): _target_report(recorder.frames, step_history, emergency_events, env, arrival)
            for arrival in TARGET_ARRIVALS[int(run_index)]
        },
        "o_to_emergency": _o_to_emergency(emergency_events),
        "step_trace": {
            "observations": len(step_history),
            "emergency_departures": len(emergency_events),
            "ring_scope": "all active UAVs recorded before every original env.step",
        },
        "energy_outcomes": _jsonable(list(env.energy_outcomes)),
        "safety": _jsonable(getattr(env, "telemetry", {})),
    }
    (output_dir / "replay_report.json").write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-index", type=int, choices=(0, 1, 2))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--schema-check", action="store_true")
    args = parser.parse_args()
    if args.schema_check:
        print(json.dumps(schema_check(), sort_keys=True))
        return 0
    if args.run_index is None or args.output_dir is None:
        parser.error("--run-index and --output-dir are required unless --schema-check is used")
    report = export_run(args.run_index, args.output_dir)
    print(json.dumps({
        "run_index": report["run_index"],
        "exact": report["exact"],
        "steps_replayed": report["steps_replayed"],
        "steps_expected": report["steps_expected"],
        "playback_html": report["playback_html"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
