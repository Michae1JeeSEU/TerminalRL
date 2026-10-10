"""Export exact bounded-gate tapes to the existing playback viewer."""
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


BOUNDED_ROOT = Path("/home/user/training_outputs/terminal/p3_bounded_gate_diagnostic_v1_20260925")
DEFAULT_OUTPUT_ROOT = Path("/home/user/training_outputs/terminal/p3_formal_policy_visualizations_20260925")
FRAME_STRIDE = 2


def _jsonable(value: Any) -> Any:
    return base._jsonable(value)


def _read_summary(run_index: int) -> dict[str, Any]:
    path = BOUNDED_ROOT / f"run{int(run_index)}" / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"summary must be an object: {path}")
    return value


def schema_check() -> dict[str, Any]:
    config = bounded.build_config()
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
        "rejected_release_hold_mode": str(config.v4_2_rejected_release_hold_mode),
    }


def _lifecycle_from_env(env: TerminalParallelEnv, config: Any) -> dict[str, Any]:
    protocol = base._load_yaml_protocol()
    threshold = float(protocol["attributes"]["low_soc_analysis_threshold"])
    rows: list[dict[str, Any]] = []
    for uav in env.completed_uavs:
        initial_soc = float(
            (uav.energy_remaining + uav.energy_consumed) / uav.energy_capacity
        )
        rows.append({
            "uid": int(uav.uid),
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


def export_run(run_index: int, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")
    source = _read_summary(run_index)
    nested = source.get("summary")
    if not isinstance(nested, dict):
        raise ValueError("bounded summary.summary must be an object")
    tape = Path(str(nested["action_tape"]))
    if not tape.is_file():
        raise FileNotFoundError(tape)
    config = bounded.build_config()
    env = TerminalParallelEnv(config)
    recorder = SnapshotRecorder(FRAME_STRIDE)
    env.set_step_observer(recorder)
    replay = SafetyTraceRecorder.replay_action_tape(env, tape)
    exact = bool(replay.get("exact") is True)
    if not exact:
        raise ValueError(f"bounded tape replay was not exact: run{run_index}")
    recorder.capture(env)
    seed = int(source["seed"])
    title = f"run{run_index} bounded .05 canonical-wait seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=False)
    html_path = output_dir / "recorded_stochastic_policy_replay.html"
    write_playback(html_path, recorder.frames, title)
    report = {
        "stage": "P3_BOUNDED_GATE_RECORDED_POLICY_REPLAY_V1",
        "run_index": int(run_index),
        "seed": seed,
        "scenario": str(source.get("scenario", "dense24")),
        "source_summary": str(BOUNDED_ROOT / f"run{int(run_index)}" / "summary.json"),
        "action_tape": str(tape),
        "steps_replayed": replay.get("steps_replayed"),
        "steps_expected": replay.get("steps_expected"),
        "exact": exact,
        "title": title,
        "frame_stride": FRAME_STRIDE,
        "playback_html": str(html_path),
        "lifecycle": _lifecycle_from_env(env, config),
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
