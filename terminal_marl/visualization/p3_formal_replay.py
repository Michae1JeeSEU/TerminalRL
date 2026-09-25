"""Replay one recorded P3 formal rollout into the existing snapshot viewer.

This module replays the requested actions stored in a formal-run action tape.
It does not load or resample the policy.  The checkpoint is bound to the
failed record before the tape is replayed so that the resulting HTML remains
traceable to the recorded rollout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from terminal_marl.core.safety_trace import SafetyTraceRecorder
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training import (
    run_v4_2_p3_controlled_emergency_admission_v5 as admission,
)
from terminal_marl.visualization.snapshot_playback import SnapshotRecorder, write_playback


STAGE = "P3_FORMAL_RECORDED_STOCHASTIC_POLICY_REPLAY_V1"
DEFAULT_FRAME_STRIDE = 2


class ReplayInputError(ValueError):
    """Raised when a formal replay input cannot be bound exactly."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayInputError(message)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"JSON input missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _resolve_artifact(raw: str | Path, run_dir: Path, marker: str) -> Path:
    """Resolve a manifest path, including server absolute paths, exactly once."""
    raw_text = str(raw or "")
    _require(raw_text, f"manifest artifact path is empty ({marker})")
    direct = Path(raw_text)
    if direct.is_file():
        return direct
    if not direct.is_absolute():
        local = run_dir / direct
        if local.is_file():
            return local

    normalized = raw_text.replace("\\", "/")
    parts = PurePosixPath(normalized).parts
    _require(marker in parts, f"manifest path has no {marker} component: {raw_text}")
    relative = Path(*parts[parts.index(marker):])
    matches = tuple(
        candidate for candidate in run_dir.glob(relative.as_posix())
        if candidate.is_file()
    )
    _require(
        len(matches) == 1,
        f"expected exactly one {marker} artifact under {run_dir}, found {len(matches)}: {raw_text}",
    )
    return matches[0]


def _last_failed_record(manifest: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
    records = manifest.get("records")
    _require(isinstance(records, list) and records, "formal manifest has no records")
    raw = records[-1]
    _require(isinstance(raw, Mapping), "last formal record is not an object")
    record = dict(raw)
    failed = (
        record.get("accepted_episode") is False
        or record.get("hard_gate_pass") is False
        or int(record.get("failed_energy", 0) or 0) > 0
        or record.get("full_resolution") is False
    )
    _require(failed, "last formal record is not a failed record")
    return len(records) - 1, record


def _bind_checkpoint(
    manifest: Mapping[str, Any], record: Mapping[str, Any], run_dir: Path,
) -> tuple[Path, str, dict[str, str]]:
    final_raw = manifest.get("final_checkpoint")
    _require(final_raw, "formal manifest final_checkpoint is missing")
    checkpoint = _resolve_artifact(final_raw, run_dir, "checkpoints")

    record_raw = record.get("checkpoint")
    _require(record_raw, "failed record checkpoint is missing")
    record_checkpoint = _resolve_artifact(record_raw, run_dir, "checkpoints")
    _require(
        record_checkpoint.resolve() == checkpoint.resolve(),
        "failed record checkpoint does not match formal final_checkpoint",
    )

    actual_sha = _sha256(checkpoint)
    fields: dict[str, str] = {}
    for key in ("checkpoint_before_sha256", "checkpoint_input_sha256", "checkpoint_sha256"):
        value = record.get(key)
        if value in (None, ""):
            continue
        value = str(value)
        _require(value == actual_sha, f"{key} does not match final checkpoint SHA")
        fields[key] = value
    _require(fields, "failed record has no checkpoint SHA binding")

    manifest_sha = manifest.get("final_checkpoint_sha256")
    if manifest_sha not in (None, ""):
        _require(str(manifest_sha) == actual_sha, "manifest final checkpoint SHA mismatch")
    return checkpoint, actual_sha, fields


def _scenario_config(manifest: Mapping[str, Any], scenario: str) -> Any:
    """Build the admission runner's frozen scenario config and bind its snapshot."""
    config = admission._build_config(str(scenario), admission.base._load_yaml_protocol())
    snapshots = manifest.get("actual_config_snapshots")
    if isinstance(snapshots, Mapping) and scenario in snapshots:
        actual = admission.base._config_snapshot(config)
        _require(
            admission._jsonable(actual) == snapshots[scenario],
            f"scenario configuration snapshot mismatch: {scenario}",
        )
    return config


def _record_outcomes(record: Mapping[str, Any]) -> dict[str, Any]:
    """Copy the formal record's published outcome counters into the report."""
    values: dict[str, int] = {}
    for key in ("requests", "normal", "emergency", "failed_energy", "unresolved"):
        _require(key in record, f"failed record outcome field is missing: {key}")
        try:
            values[key] = int(record[key])
        except (TypeError, ValueError) as exc:
            raise ReplayInputError(f"failed record outcome field is invalid: {key}") from exc
    values["other_failures"] = admission._jsonable(record.get("other_failures") or {})
    raw = record.get("outcomes")
    if isinstance(raw, Mapping) and raw:
        values["raw_outcomes"] = admission._jsonable(raw)
    return values


def _title(run_index: Any, checkpoint: Path, seed: int, scenario: str) -> str:
    return (
        f"run{run_index} {checkpoint.stem} seed{seed} {scenario} "
        "recorded stochastic policy replay"
    )


def replay_run(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    frame_stride: int = DEFAULT_FRAME_STRIDE,
) -> dict[str, Any]:
    """Replay the final failed formal record in one run directory."""
    run_path = Path(run_dir).resolve()
    output_path = Path(output_dir).resolve()
    _require(run_path.is_dir(), f"run directory missing: {run_path}")
    _require(not output_path.exists(), f"output directory already exists: {output_path}")
    _require(
        output_path != run_path and run_path not in output_path.parents,
        "output directory must be independent of the formal run directory",
    )
    _require(int(frame_stride) > 0, "frame_stride must be positive")

    manifest_path = run_path / "manifest.json"
    manifest = _read_json(manifest_path)
    record_index, record = _last_failed_record(manifest)
    checkpoint, checkpoint_sha, checkpoint_fields = _bind_checkpoint(
        manifest, record, run_path,
    )

    tape_raw = record.get("action_tape")
    tape = _resolve_artifact(tape_raw, run_path, "episodes")
    scenario = str(record.get("scenario") or "")
    _require(scenario, "failed record scenario is missing")
    try:
        seed = int(record["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ReplayInputError("failed record seed is missing or invalid") from exc

    config = _scenario_config(manifest, scenario)
    env = TerminalParallelEnv(config)
    recorder = SnapshotRecorder(int(frame_stride))
    env.set_step_observer(recorder)
    replay = SafetyTraceRecorder.replay_action_tape(env, tape)
    exact = bool(replay.get("exact") is True)

    run_index = manifest.get("run_index", run_path.name)
    title = _title(run_index, checkpoint, seed, scenario)
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
        "seed": seed,
        "scenario": scenario,
        "steps_replayed": replay.get("steps_replayed"),
        "steps_expected": replay.get("steps_expected"),
        "exact": exact,
        "original_outcomes": _record_outcomes(record),
        "title": title,
        "frame_stride": int(frame_stride),
        "playback_html": None,
    }

    output_path.mkdir(parents=True, exist_ok=False)
    report_path = output_path / "replay_report.json"
    report_path.write_text(
        json.dumps(admission._jsonable(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if not exact:
        return report

    recorder.capture(env)
    html_path = output_path / "recorded_stochastic_policy_replay.html"
    write_playback(html_path, recorder.frames, title)
    report["playback_html"] = str(html_path)
    report_path.write_text(
        json.dumps(admission._jsonable(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay the final failed P3 formal action tape into snapshot HTML."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = replay_run(args.run_dir, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("exact") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
