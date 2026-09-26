"""Collect pre-step runtime observations from the three source train tapes.

The collector performs one exact action-tape replay per train seed.  It does not
sample a policy or evaluate proposal branches; it only records the live
88-dimensional observations immediately before each replay step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from terminal_marl.core.safety_trace import SafetyTraceRecorder
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training import collect_p3_proposal_head_probe_v1 as collector


OUTPUT_ROOT = Path(
    "/home/user/training_outputs/terminal/p3_runtime_rank_repair_20260926/anchors"
)
FORMAL_ROOT = collector.FORMAL_ROOT
TRAIN_SEEDS = tuple(int(seed) for seed in collector.TRAIN_SEEDS)
OBSERVATION_DIM = 88
STAGE = "P3_RUNTIME_RANK_ANCHORS"


class CollectionError(ValueError):
    """Raised when the source binding or replay result is not exact."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CollectionError(message)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _observation_for_uid(observations: Mapping[Any, Any], uid: int) -> Any:
    if str(uid) in observations:
        return observations[str(uid)]
    if uid in observations:
        return observations[uid]
    raise CollectionError(f"active uid {uid} missing from _observations()")


def _collect_record(
    *,
    record: Mapping[str, Any],
    config: Any,
    observation_rows: list[np.ndarray],
    seed_rows: list[int],
    step_rows: list[int],
    uid_rows: list[int],
) -> dict[str, Any]:
    seed = int(record["seed"])
    tape = Path(str(record["action_tape"]))
    _require(tape.is_file(), f"source action tape missing: {tape}")
    env = TerminalParallelEnv(config)
    original_step = env.step
    row_count_before = len(observation_rows)

    def observed_step(*args: Any, **kwargs: Any) -> Any:
        active = getattr(env, "active", {})
        observations = env._observations()
        _require(isinstance(observations, Mapping), "_observations() must return a mapping")
        active_uids = {int(uid) for uid in active}
        observation_uids = {int(uid) for uid in observations}
        _require(
            active_uids == observation_uids,
            f"active/_observations uid mismatch at seed {seed}, step {env.step_count}",
        )
        for uid in sorted(active_uids):
            value = np.asarray(_observation_for_uid(observations, uid), dtype=np.float32)
            _require(
                value.shape == (OBSERVATION_DIM,),
                f"observation shape changed for seed {seed}, step {env.step_count}, uid {uid}: {value.shape}",
            )
            _require(
                bool(np.isfinite(value).all()),
                f"non-finite observation for seed {seed}, step {env.step_count}, uid {uid}",
            )
            observation_rows.append(value.copy())
            seed_rows.append(seed)
            step_rows.append(int(env.step_count))
            uid_rows.append(uid)
        return original_step(*args, **kwargs)

    env.step = observed_step  # type: ignore[method-assign]
    try:
        replay = SafetyTraceRecorder.replay_action_tape(env, tape)
    finally:
        env.step = original_step  # type: ignore[method-assign]

    _require(bool(replay.get("exact")), f"non-exact replay for seed {seed}")
    _require(
        int(replay.get("steps_replayed", -1)) == int(replay.get("steps_expected", -2)),
        f"replay step count mismatch for seed {seed}",
    )
    source_outcomes = collector._summary_outcomes(tape)
    replay_outcomes = collector._replay_outcomes(env)
    _require(source_outcomes == replay_outcomes, f"outcomes mismatch for seed {seed}")
    return {
        "seed": seed,
        "scenario": str(record.get("scenario", "")),
        "split": str(record.get("split", "train")),
        "action_tape": str(tape),
        "action_tape_sha256": _sha256(tape),
        "steps_expected": int(replay["steps_expected"]),
        "steps_replayed": int(replay["steps_replayed"]),
        "replay_exact": bool(replay["exact"]),
        "outcomes_match": True,
        "rows_count": len(observation_rows) - row_count_before,
    }


def _write_npz_atomic(path: Path, *, observations: np.ndarray, seed: np.ndarray,
                      step: np.ndarray, uid: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, observations=observations, seed=seed, step=step, uid=uid)
    os.replace(temporary, path)


def collect(*, output_dir: Path, formal_root: Path) -> dict[str, Any]:
    _require(not output_dir.exists(), f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    progress: dict[str, Any] = {
        "stage": STAGE,
        "status": "RUNNING",
        "records_expected": len(TRAIN_SEEDS),
        "records_processed": 0,
        "rows_count": 0,
        "processed_seeds": [],
    }
    _write_json_atomic(output_dir / "progress.json", progress)
    observations: list[np.ndarray] = []
    seed_rows: list[int] = []
    step_rows: list[int] = []
    uid_rows: list[int] = []
    record_results: list[dict[str, Any]] = []
    try:
        _, configs, records, source_manifest = collector._load_source_inputs(formal_root)
        selected = [
            dict(row) for row in records
            if int(row.get("seed", -1)) in TRAIN_SEEDS and str(row.get("split")) == "train"
        ]
        selected.sort(key=lambda row: int(row["seed"]))
        _require(
            [int(row["seed"]) for row in selected] == list(TRAIN_SEEDS),
            f"source train seed set changed: {[row.get('seed') for row in selected]}",
        )
        source_checkpoint = Path(str(collector.SOURCE_CHECKPOINT))
        _require(source_checkpoint.is_file(), f"source checkpoint missing: {source_checkpoint}")
        source_checkpoint_sha256 = _sha256(source_checkpoint)
        expected_sha256 = str(getattr(collector, "SOURCE_CHECKPOINT_SHA256", ""))
        if expected_sha256:
            _require(source_checkpoint_sha256 == expected_sha256, "source checkpoint SHA mismatch")

        for row in selected:
            scenario = str(row["scenario"])
            _require(scenario in configs, f"missing source config for scenario {scenario}")
            result = _collect_record(
                record=row,
                config=configs[scenario],
                observation_rows=observations,
                seed_rows=seed_rows,
                step_rows=step_rows,
                uid_rows=uid_rows,
            )
            record_results.append(result)
            progress.update(
                records_processed=len(record_results),
                rows_count=len(observations),
                processed_seeds=[int(item["seed"]) for item in record_results],
                last_seed=int(result["seed"]),
            )
            _write_json_atomic(output_dir / "progress.json", progress)

        _require(bool(observations), "no active actor observations collected")
        observations_array = np.asarray(observations, dtype=np.float32).reshape(-1, OBSERVATION_DIM)
        seed_array = np.asarray(seed_rows, dtype=np.int64)
        step_array = np.asarray(step_rows, dtype=np.int64)
        uid_array = np.asarray(uid_rows, dtype=np.int64)
        _require(
            len(observations_array) == len(seed_array) == len(step_array) == len(uid_array),
            "anchor array lengths differ",
        )
        anchors_path = output_dir / "anchors.npz"
        _write_npz_atomic(
            anchors_path,
            observations=observations_array,
            seed=seed_array,
            step=step_array,
            uid=uid_array,
        )
        rows_by_seed = {
            str(seed): int(np.count_nonzero(seed_array == seed)) for seed in TRAIN_SEEDS
        }
        manifest = {
            "stage": STAGE,
            "status": "COMPLETED",
            "exact": all(bool(item["replay_exact"]) and bool(item["outcomes_match"]) for item in record_results),
            "seeds": list(TRAIN_SEEDS),
            "records_count": len(record_results),
            "records": record_results,
            "rows_count": int(len(observations_array)),
            "rows_by_seed": rows_by_seed,
            "observation_dimension": OBSERVATION_DIM,
            "arrays": ["observations", "seed", "step", "uid"],
            "anchors": str(anchors_path),
            "source_manifest": str(source_manifest),
            "source_checkpoint": str(source_checkpoint),
            "source_checkpoint_sha256": source_checkpoint_sha256,
            "train_seeds": list(TRAIN_SEEDS),
            "validation_tapes_read": False,
        }
        _write_json_atomic(output_dir / "manifest.json", manifest)
        progress.update(status="COMPLETED", rows_count=int(len(observations_array)))
        _write_json_atomic(output_dir / "progress.json", progress)
        return manifest
    except Exception as error:
        progress.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        _write_json_atomic(output_dir / "progress.json", progress)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--formal-root", type=Path, default=FORMAL_ROOT)
    args = parser.parse_args(argv)
    try:
        manifest = collect(output_dir=args.output_dir, formal_root=args.formal_root)
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": manifest["status"], "rows_count": manifest["rows_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
