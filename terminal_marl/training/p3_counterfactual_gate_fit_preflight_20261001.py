"""Readiness-only validation for the registered P3 gate-adapter fit.

This module validates the frozen soft-target JSON and the future replay anchor
manifest.  It has no model, environment, gradient, optimizer, checkpoint, or
sampling path.  A successful result means that the inputs are reviewable; it
does not authorize or perform fitting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "p3_counterfactual_gate_fit_preflight_20261001_v1"
STAGE = "P3_COUNTERFACTUAL_GATE_FIT_PREFLIGHT_20261001_V1"
TARGET_SCHEMA = "p3_counterfactual_gate_targets_20261001_v1"
TARGETS_SHA256 = "4085299501758c55c476fac9676092ace9d9c69f4ea5d3a68b74f786eaaf1473"
ANCHOR_STAGE = "P3_GATE_ANCHOR_STATES_20261001_V1"
ANCHOR_SOURCE_FULL_SHA256 = "30c28c03d26466eabeab2a35c9fc117424648f1a3ae17be6f8d69b6b92cb2195"
EVENT_FULL_SHA256 = "fb89f62339755f7ebcd2bb0f9672438000fe8b754067cd2cdb28bf8407feb3e6"
BEFORE_CHECKPOINT_SHA256 = "3fe7dd1e01d032227521f6fc5c8464fe8d00d98fb0e9f0dbd13411123a27d4ab"
AFTER_CHECKPOINT_SHA256 = "d44990dc7d129573d9145f169df1cbc4f91fa32bc5127b1c58d937260c02a24f"
EXPECTED_CELLS = (1, 2, 3, 4)
EXPECTED_TARGET_KEYS = {1: "307|15", 2: "321|25", 3: "303|20", 4: "365|46"}
EXPECTED_STEPS = {1: 708, 2: 674, 3: 644, 4: 723}
TOTAL_STEPS = sum(EXPECTED_STEPS.values())
OBSERVATION_DIM = 88
FIT_SEED = 9303501
LEARNING_RATE = 1e-4
MAX_UPDATES = 80
KL_CAP = 0.001
TARGET_P_BEFORE_TOL = 1e-6


class PreflightError(ValueError):
    """Raised for malformed or conflicting frozen input metadata."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PreflightError(message)


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise PreflightError(f"{name} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise PreflightError(f"{name} is not finite")
    return result


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: str | Path) -> tuple[dict[str, Any], str]:
    target = Path(path)
    _require(target.is_file(), f"JSON input missing: {target}")
    data = target.read_bytes()
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreflightError(f"invalid JSON input: {target}: {exc}") from exc
    _require(isinstance(payload, dict), f"JSON object required: {target}")
    return dict(payload), hashlib.sha256(data).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any], *, refuse_existing: bool = False) -> None:
    if refuse_existing and path.exists():
        raise PreflightError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _resolve_file(stored: Any, *, bases: Sequence[Path]) -> Path:
    _require(isinstance(stored, str) and stored, "anchor file path is missing")
    raw = Path(stored)
    candidates = [raw]
    prefix = "/home/user/training_outputs/terminal/"
    normalized = stored.replace("\\", "/")
    if normalized.startswith(prefix):
        mirror_root = Path(__file__).resolve().parents[3] / "training_outputs" / "terminal"
        candidates.append(mirror_root / normalized[len(prefix):])
    for base in bases:
        candidates.append(base / raw)
        candidates.append(base / raw.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise PreflightError(f"anchor file missing: {stored}")


def _validate_targets(path: str | Path) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    payload, digest = _read_json(path)
    _require(digest == TARGETS_SHA256, f"target SHA mismatch: {digest}")
    _require(payload.get("schema") == TARGET_SCHEMA, "target schema mismatch")
    _require(payload.get("status") == "TARGETS_READY", "targets are not TARGETS_READY")
    binding = payload.get("input_binding")
    _require(isinstance(binding, Mapping), "target input binding missing")
    _require(binding.get("before_checkpoint_sha256") == BEFORE_CHECKPOINT_SHA256,
             "target BEFORE checkpoint binding mismatch")
    _require(binding.get("after_checkpoint_sha256") == AFTER_CHECKPOINT_SHA256,
             "target AFTER checkpoint binding mismatch")
    raw_cells = payload.get("cells")
    _require(isinstance(raw_cells, list) and len(raw_cells) == 4, "target cell count mismatch")
    cells: dict[int, dict[str, Any]] = {}
    for raw in raw_cells:
        _require(isinstance(raw, Mapping), "target cell is not an object")
        cell = dict(raw)
        cell_id = cell.get("cell_id")
        _require(isinstance(cell_id, int) and cell_id in EXPECTED_CELLS and cell_id not in cells,
                 "target cell ID set mismatch")
        _require(cell.get("known") is True, f"target cell {cell_id} is not known")
        selection = cell.get("selection")
        _require(isinstance(selection, Mapping), f"target cell {cell_id} selection missing")
        _require(selection.get("registration_key") == EXPECTED_TARGET_KEYS[cell_id],
                 f"target cell {cell_id} registration key mismatch")
        context = selection.get("context")
        _require(isinstance(context, Mapping), f"target cell {cell_id} context missing")
        _finite(cell.get("p0"), name=f"target cell {cell_id} p0")
        _finite(cell.get("pstar"), name=f"target cell {cell_id} pstar")
        cells[cell_id] = cell
    _require(sorted(cells) == list(EXPECTED_CELLS), "target cells incomplete")
    return cells, {"path": str(path), "sha256": digest, "schema": payload["schema"]}


def _read_anchor_npz(path: Path) -> dict[str, Any]:
    try:
        loaded = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise PreflightError(f"cannot read anchor NPZ {path}: {exc}") from exc
    with loaded as data:
        required = (
            "world", "step", "uid", "arrival_id", "row_index", "actor_mask",
            "requested_gate", "requested_proposal",
        )
        for key in required:
            _require(key in data, f"anchor NPZ field missing: {key}")
        world = np.asarray(data["world"])
        _require(world.dtype == np.float32 and world.ndim == 2 and world.shape[1] == OBSERVATION_DIM,
                 f"world must be finite float32 rows x {OBSERVATION_DIM}")
        _require(bool(np.isfinite(world).all()), "world contains non-finite values")
        count = len(world)
        arrays = {key: np.asarray(data[key]) for key in required if key != "world"}
        _require(all(value.ndim == 1 for value in arrays.values()), "anchor arrays must be one-dimensional")
        _require(all(len(value) == count for value in arrays.values()), "anchor array lengths differ")
        if "metadata_json" in data:
            metadata_raw = np.asarray(data["metadata_json"])
            _require(metadata_raw.size == 1, "anchor metadata_json must be scalar")
            try:
                metadata = json.loads(str(metadata_raw.reshape(-1)[0]))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise PreflightError(f"anchor metadata_json is invalid: {path}: {exc}") from exc
            _require(isinstance(metadata, Mapping), "anchor metadata_json must be an object")
            arrays["metadata"] = dict(metadata)
        return {"world": world, **arrays}


def _binary_code(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return int(value)
    return -1


def _proposal_code(value: Any) -> int:
    if value is None:
        return -2
    try:
        number = int(value)
    except (TypeError, ValueError):
        return -2
    return number if number == -1 or 0 <= number < 25 else -2


def _row_int(row: Mapping[str, Any], field: str, *, cell_id: int, index: int) -> int:
    value = row.get(field)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise PreflightError(f"anchor cell {cell_id} row {index} field {field} is not integer") from exc


def _anchor_manifest(path: str | Path, target_cells: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    requested = Path(path)
    manifest_path = requested / "full.json" if requested.is_dir() else requested
    manifest, manifest_sha = _read_json(manifest_path)
    _require(manifest.get("stage") == ANCHOR_STAGE, "anchor stage mismatch")
    _require(manifest.get("status") == "FULL_OK", "anchor manifest is not FULL_OK")
    _require(manifest.get("source_full_sha256") == ANCHOR_SOURCE_FULL_SHA256,
             "anchor source full binding mismatch")
    _require(manifest.get("event_full_sha256") == EVENT_FULL_SHA256,
             "anchor event full binding mismatch")
    summary_cells = manifest.get("cells")
    _require(isinstance(summary_cells, list) and len(summary_cells) == len(EXPECTED_CELLS),
             "anchor cell summary missing")
    summaries = {int(item.get("cell_id", -1)): item for item in summary_cells if isinstance(item, Mapping)}
    _require(sorted(summaries) == list(EXPECTED_CELLS), "anchor cell IDs mismatch")
    total_ticks = 0
    total_rows = 0
    cell_reports: dict[int, dict[str, Any]] = {}
    target_rows: dict[int, dict[str, Any]] = {}
    for cell_id in EXPECTED_CELLS:
        summary = summaries[cell_id]
        _require(summary.get("status") == "COMPLETED", f"anchor cell {cell_id} incomplete")
        cell_path = _resolve_file(
            summary.get("cell_json_path"),
            bases=(manifest_path.parent, manifest_path.parent / "cells"),
        )
        cell, cell_sha = _read_json(cell_path)
        _require(summary.get("cell_json_sha256") == cell_sha,
                 f"anchor cell {cell_id} cell JSON SHA mismatch")
        _require(cell.get("stage") == ANCHOR_STAGE and cell.get("status") == "COMPLETED",
                 f"anchor cell {cell_id} schema/status mismatch")
        _require(cell.get("arm") == "before_prefix", f"anchor cell {cell_id} arm mismatch")
        _require(cell.get("replay_exact") is True, f"anchor cell {cell_id} replay is not exact")
        replay = cell.get("replay")
        _require(isinstance(replay, Mapping), f"anchor cell {cell_id} replay metadata missing")
        for replay_flag in ("exact", "config_match", "reset_recipe_match", "reset_state_match",
                            "base_reset_state_match"):
            _require(replay.get(replay_flag) is True,
                     f"anchor cell {cell_id} replay check failed: {replay_flag}")
        if "config_compatibility" in replay:
            _require(replay.get("config_compatibility") in (None, True),
                     f"anchor cell {cell_id} replay config compatibility failed")
        _require(cell.get("controller_installed_and_restored") is True,
                 f"anchor cell {cell_id} controller restoration is not certified")
        _require(cell.get("caller_rng_unchanged_before_restore") is True,
                 f"anchor cell {cell_id} caller RNG pre-restore check failed")
        _require(cell.get("caller_rng_unchanged_after_restore") is True,
                 f"anchor cell {cell_id} caller RNG post-restore check failed")
        source_provenance = cell.get("source_provenance")
        if isinstance(source_provenance, Mapping):
            _require(source_provenance.get("safety_nonzero") is not True,
                     f"anchor cell {cell_id} source safety certificate is nonzero")
            _require(source_provenance.get("lifecycle_valid") is not False,
                     f"anchor cell {cell_id} source lifecycle certificate is invalid")
            binding_checks = source_provenance.get("binding_checks")
            if isinstance(binding_checks, Mapping):
                _require(all(value is not False for value in binding_checks.values()),
                         f"anchor cell {cell_id} source binding certificate failed")
        binding = cell.get("input_binding")
        _require(isinstance(binding, Mapping) and binding.get("full_sha256") == ANCHOR_SOURCE_FULL_SHA256,
                 f"anchor cell {cell_id} source binding mismatch")
        _require(int(binding.get("expected_steps", -1)) == EXPECTED_STEPS[cell_id],
                 f"anchor cell {cell_id} expected steps mismatch")
        rows = cell.get("rows")
        _require(isinstance(rows, list), f"anchor cell {cell_id} rows missing")
        step_metadata = cell.get("step_metadata")
        _require(isinstance(step_metadata, list), f"anchor cell {cell_id} step metadata missing")
        _require(int(cell.get("steps_expected", -1)) == EXPECTED_STEPS[cell_id],
                 f"anchor cell {cell_id} steps_expected mismatch")
        _require(int(cell.get("steps_replayed", -1)) == EXPECTED_STEPS[cell_id],
                 f"anchor cell {cell_id} steps_replayed mismatch")
        _require(len(step_metadata) == EXPECTED_STEPS[cell_id],
                 f"anchor cell {cell_id} step metadata count mismatch")
        _require(summary.get("steps") == EXPECTED_STEPS[cell_id],
                 f"anchor cell {cell_id} manifest tick count mismatch")
        _require(summary.get("rows") == len(rows),
                 f"anchor cell {cell_id} manifest actor row count mismatch")
        npz_info = cell.get("anchors_npz")
        _require(isinstance(npz_info, Mapping), f"anchor cell {cell_id} NPZ binding missing")
        summary_npz_path = _resolve_file(
            summary.get("anchors_npz_path"),
            bases=(manifest_path.parent, manifest_path.parent / "anchors"),
        )
        npz_path = _resolve_file(
            npz_info.get("path"),
            bases=(manifest_path.parent, manifest_path.parent / "anchors"),
        )
        _require(summary_npz_path == npz_path,
                 f"anchor cell {cell_id} manifest/JSON NPZ path mismatch")
        npz_sha = _sha256(npz_path)
        _require(npz_sha == npz_info.get("sha256"), f"anchor cell {cell_id} NPZ SHA mismatch")
        _require(summary.get("anchors_npz_sha256") == npz_sha,
                 f"anchor cell {cell_id} manifest NPZ SHA mismatch")
        arrays = _read_anchor_npz(npz_path)
        _require(len(arrays["world"]) == len(rows), f"anchor cell {cell_id} world row count mismatch")
        if "shape" in npz_info:
            _require(npz_info.get("shape") == [len(rows), OBSERVATION_DIM],
                     f"anchor cell {cell_id} NPZ shape metadata mismatch")
        if "dtype" in npz_info:
            _require(npz_info.get("dtype") == "float32",
                     f"anchor cell {cell_id} NPZ dtype metadata mismatch")
        row_keys: set[str] = set()
        tick_steps: list[int] = []
        last_row_end = 0
        for tick_index, raw_tick in enumerate(step_metadata):
            _require(isinstance(raw_tick, Mapping),
                     f"anchor cell {cell_id} step metadata {tick_index} is not an object")
            tick = dict(raw_tick)
            step = _row_int(tick, "step", cell_id=cell_id, index=tick_index)
            tick_steps.append(step)
            row_start = _row_int(tick, "row_start", cell_id=cell_id, index=tick_index)
            row_end = _row_int(tick, "row_end", cell_id=cell_id, index=tick_index)
            _require(0 <= row_start <= row_end <= len(rows),
                     f"anchor cell {cell_id} step {step} row range is invalid")
            _require(row_start == last_row_end,
                     f"anchor cell {cell_id} step {step} row ranges are not contiguous")
            uids = tick.get("uids")
            _require(isinstance(uids, list), f"anchor cell {cell_id} step {step} UID list missing")
            _require(row_end - row_start == len(uids),
                     f"anchor cell {cell_id} step {step} row range/UID count mismatch")
            try:
                uid_values = [int(uid) for uid in uids]
            except (TypeError, ValueError) as exc:
                raise PreflightError(f"anchor cell {cell_id} step {step} UID list is malformed") from exc
            _require(len(set(uid_values)) == len(uid_values),
                     f"anchor cell {cell_id} step {step} contains duplicate UIDs")
            if "world_shape" in tick:
                _require(tick.get("world_shape") == [len(uid_values), OBSERVATION_DIM],
                         f"anchor cell {cell_id} step {step} world shape metadata mismatch")
            if "observation_hash" in tick:
                _require(isinstance(tick.get("observation_hash"), str) and tick["observation_hash"],
                         f"anchor cell {cell_id} step {step} observation hash missing")
            for offset, uid in enumerate(uid_values):
                row = rows[row_start + offset]
                _require(isinstance(row, Mapping),
                         f"anchor cell {cell_id} actor row {row_start + offset} is not an object")
                row = dict(row)
                index = row_start + offset
                _require(_row_int(row, "row_index", cell_id=cell_id, index=index) == index,
                         f"anchor cell {cell_id} row index mismatch")
                _require(_row_int(row, "world_row_index", cell_id=cell_id, index=index) == index,
                         f"anchor cell {cell_id} world row index mismatch")
                _require(_row_int(row, "step", cell_id=cell_id, index=index) == step,
                         f"anchor cell {cell_id} row/tick step mismatch")
                _require(_row_int(row, "uid", cell_id=cell_id, index=index) == uid,
                         f"anchor cell {cell_id} row/tick UID order mismatch")
                key = f"{step}|{uid}"
                _require(key not in row_keys, f"anchor cell {cell_id} duplicate row key {key}")
                row_keys.add(key)
                if "observation_hash" in tick:
                    _require(row.get("observation_hash") == tick["observation_hash"],
                             f"anchor cell {cell_id} row/tick observation hash mismatch")
                _require(int(arrays["row_index"][index]) == index,
                         f"anchor cell {cell_id} NPZ row index mismatch at {index}")
                _require(int(arrays["step"][index]) == step,
                         f"anchor cell {cell_id} NPZ step mismatch at {index}")
                _require(int(arrays["uid"][index]) == uid,
                         f"anchor cell {cell_id} NPZ UID mismatch at {index}")
                arrival = _row_int(row, "arrival_id", cell_id=cell_id, index=index)
                _require(int(arrays["arrival_id"][index]) == arrival,
                         f"anchor cell {cell_id} NPZ arrival mismatch at {index}")
                _require(int(arrays["actor_mask"][index]) == _binary_code(row.get("actor_mask")),
                         f"anchor cell {cell_id} NPZ actor_mask mismatch at {index}")
                _require(int(arrays["requested_gate"][index]) == _binary_code(row.get("requested_gate")),
                         f"anchor cell {cell_id} NPZ requested_gate mismatch at {index}")
                _require(int(arrays["requested_proposal"][index]) == _proposal_code(row.get("requested_proposal")),
                         f"anchor cell {cell_id} NPZ requested_proposal mismatch at {index}")
                key = f"{step}|{uid}"
                if key == EXPECTED_TARGET_KEYS[cell_id]:
                    target_context = target_cells[cell_id]["selection"]["context"]
                    context = row.get("context")
                    _require(isinstance(context, Mapping), f"anchor target context missing: {key}")
                    for field in ("actor_mask", "arrival_id", "phase", "retry_free", "committed_plan"):
                        _require(context.get(field) == target_context.get(field),
                                 f"target/anchor context mismatch at {key}: {field}")
                    _require(isinstance(row.get("observation_hash"), str) and row["observation_hash"],
                             f"anchor target observation hash missing: {key}")
                    target_rows[cell_id] = {
                        "key": key,
                        "step": step,
                        "uid": uid,
                        "observation_hash": row["observation_hash"],
                        "world_row_index": index,
                        "context": dict(context),
                    }
            last_row_end = row_end
        _require(last_row_end == len(rows),
                 f"anchor cell {cell_id} step metadata does not cover all actor rows")
        _require(tick_steps == list(range(EXPECTED_STEPS[cell_id])),
                 f"anchor cell {cell_id} step metadata tick coverage is not exact")
        total_ticks += len(step_metadata)
        total_rows += len(rows)
        cell_reports[cell_id] = {
            "cell_id": cell_id,
            "cell_json": str(cell_path),
            "cell_json_sha256": cell_sha,
            "source_provenance": cell.get("source_provenance"),
            "npz_path": str(npz_path),
            "npz_sha256": npz_sha,
            "ticks": len(step_metadata),
            "rows": len(rows),
            "world_shape": [int(value) for value in arrays["world"].shape],
        }
    _require(total_ticks == TOTAL_STEPS, f"anchor total ticks {total_ticks} != {TOTAL_STEPS}")
    _require(sorted(target_rows) == list(EXPECTED_CELLS), "one or more target anchor rows are missing")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "source_full_sha256": ANCHOR_SOURCE_FULL_SHA256,
        "event_full_sha256": EVENT_FULL_SHA256,
        "expected_ticks": TOTAL_STEPS,
        "total_ticks": total_ticks,
        "total_rows": total_rows,
        "cells": cell_reports,
        "target_rows": target_rows,
    }


def build_readiness(targets_path: str | Path, anchors_path: str | Path) -> dict[str, Any]:
    targets, target_binding = _validate_targets(targets_path)
    readiness: dict[str, Any] = {
        "schema": SCHEMA,
        "stage": STAGE,
        "phase": "preflight",
        "status": "NOT_READY",
        "targets": target_binding,
        "scope": {
            "checkpoint_sha256": BEFORE_CHECKPOINT_SHA256,
            "trainable_scope": "gate_adapter.* only",
            "critic_frozen": True,
            "proposal_and_temperature_frozen": True,
            "fit_seed": FIT_SEED,
            "learning_rate": LEARNING_RATE,
            "max_updates": MAX_UPDATES,
            "future_optimizer": "Adam",
            "future_weight_decay": 0.0,
            "future_anchor_chunk_size": 512,
            "future_anchor_weighting": "equal overall mean across all anchor rows",
            "objective": "mean four-row BCE(target pstar) + 1.0 * mean Bernoulli KL(candidate || BEFORE) over every anchor row",
            "kl_cap": KL_CAP,
            "rollback": "restore previous valid adapter parameters and stop",
            "PPO_ALLOWED": False,
            "optimizer_used": False,
            "backprop_used": False,
            "model_forward_used": False,
            "candidate_checkpoint_writer": False,
            "heldout_used": False,
        },
        "anchors": {"path": str(anchors_path)},
    }
    try:
        anchors = _anchor_manifest(anchors_path, targets)
    except (FileNotFoundError, KeyError, TypeError, ValueError, PreflightError) as exc:
        readiness["reason"] = "anchors_not_ready"
        readiness["unresolved"] = [str(exc)]
        return readiness
    readiness["status"] = "READY_FOR_ROOT_REVIEW"
    readiness["anchors"] = anchors
    readiness["qualification"] = {
        "review_only": True,
        "fit_authorized": False,
        "candidate_promoted": False,
        "D_labels_are_not_anchor_labels": True,
        "four_states_are_not_a_qualification_population": True,
    }
    return readiness


def bernoulli_kl(probability: float, reference: float) -> float:
    p = _finite(probability, name="probability")
    q = _finite(reference, name="reference")
    _require(0.0 <= p <= 1.0 and 0.0 <= q <= 1.0, "Bernoulli values must be in [0, 1]")
    if p == q:
        return 0.0
    if q in (0.0, 1.0):
        return math.inf
    result = 0.0
    if p > 0.0:
        result += p * math.log(p / q)
    if p < 1.0:
        result += (1.0 - p) * math.log((1.0 - p) / (1.0 - q))
    return result


def binary_cross_entropy(probability: float, target: float) -> float:
    p = min(max(_finite(probability, name="probability"), 1e-15), 1.0 - 1e-15)
    t = _finite(target, name="target")
    _require(0.0 <= t <= 1.0, "BCE target must be in [0, 1]")
    return -(t * math.log(p) + (1.0 - t) * math.log(1.0 - p))


def mean_anchor_kl(candidate: Sequence[float], reference: Sequence[float]) -> float:
    _require(len(candidate) == len(reference) and len(candidate) > 0, "anchor probability lengths mismatch")
    values = [bernoulli_kl(p, q) for p, q in zip(candidate, reference)]
    _require(all(math.isfinite(value) for value in values), "anchor KL contains non-finite value")
    return float(sum(values) / len(values))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--anchors", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        _require(not args.output.exists(), f"refusing to overwrite existing output: {args.output}")
        result = build_readiness(args.targets, args.anchors)
        _write_json(args.output, result, refuse_existing=True)
    except Exception as exc:
        error = {"schema": SCHEMA, "stage": STAGE, "status": "ERROR",
                 "error_type": type(exc).__name__, "message": str(exc)}
        print(json.dumps(error, sort_keys=True))
        return 2
    print(json.dumps({"status": result["status"], "output": str(args.output),
                      "anchors": result.get("anchors", {})}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
