import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from terminal_marl.training import p3_counterfactual_gate_fit_preflight_20261001 as preflight


def _write_json(path: Path, payload: dict) -> str:
    data = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _target_payload() -> dict:
    cells = []
    for cell_id, registration_key in preflight.EXPECTED_TARGET_KEYS.items():
        step, uid = (int(value) for value in registration_key.split("|"))
        cells.append(
            {
                "cell_id": cell_id,
                "known": True,
                "selection": {
                    "registration_key": registration_key,
                    "step": step,
                    "uid": uid,
                    "context": {
                        "actor_mask": True,
                        "arrival_id": 100 + cell_id,
                        "phase": "ring_inner",
                        "retry_free": True,
                        "committed_plan": False,
                    },
                },
                "p0": 0.4 + 0.05 * cell_id,
                "pstar": 0.5 + 0.05 * cell_id,
            }
        )
    return {
        "schema": preflight.TARGET_SCHEMA,
        "status": "TARGETS_READY",
        "input_binding": {
            "before_checkpoint_sha256": preflight.BEFORE_CHECKPOINT_SHA256,
            "after_checkpoint_sha256": preflight.AFTER_CHECKPOINT_SHA256,
        },
        "cells": cells,
    }


def _make_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(
        preflight,
        "EXPECTED_TARGET_KEYS",
        {1: "0|15", 2: "0|25", 3: "0|20", 4: "0|46"},
    )
    path = tmp_path / "targets.json"
    digest = _write_json(path, _target_payload())
    monkeypatch.setattr(preflight, "TARGETS_SHA256", digest)
    return path


def _make_anchor_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    realistic: bool = False,
) -> Path:
    expected_steps = {cell_id: 2 if realistic else 1 for cell_id in preflight.EXPECTED_CELLS}
    monkeypatch.setattr(preflight, "EXPECTED_STEPS", expected_steps)
    monkeypatch.setattr(preflight, "TOTAL_STEPS", sum(expected_steps.values()))
    root = tmp_path / "anchors"
    (root / "cells").mkdir(parents=True)
    summaries = []
    for cell_id, registration_key in preflight.EXPECTED_TARGET_KEYS.items():
        step, uid = (int(value) for value in registration_key.split("|"))
        arrival_id = 100 + cell_id
        if realistic and cell_id == 1:
            tick_uids = [[uid, uid + 1], []]
        elif realistic:
            tick_uids = [[uid], [uid + 1]]
        else:
            tick_uids = [[uid]]
        row_metadata = []
        step_metadata = []
        worlds = []
        steps = []
        uids = []
        arrivals = []
        actor_masks = []
        requested_gates = []
        requested_proposals = []
        for tick_offset, tick_uid_values in enumerate(tick_uids):
            tick_step = step + tick_offset
            row_start = len(row_metadata)
            for tick_uid in tick_uid_values:
                is_target = tick_uid == uid
                row_index = len(row_metadata)
                row_metadata.append(
                    {
                        "row_index": row_index,
                        "world_row_index": row_index,
                        "step": tick_step,
                        "uid": tick_uid,
                        "arrival_id": arrival_id,
                        "observation_hash": f"hash-{cell_id}-{tick_step}",
                        "actor_mask": is_target,
                        "requested_gate": 1 if is_target else None,
                        "requested_proposal": 0 if is_target else -1,
                        "context": {
                            "actor_mask": is_target,
                            "arrival_id": arrival_id,
                            "phase": "ring_inner",
                            "retry_free": True,
                            "committed_plan": False,
                        },
                    }
                )
                worlds.append(np.full(preflight.OBSERVATION_DIM, row_index / 10, dtype=np.float32))
                steps.append(tick_step)
                uids.append(tick_uid)
                arrivals.append(arrival_id)
                actor_masks.append(1 if is_target else 0)
                requested_gates.append(1 if is_target else -1)
                requested_proposals.append(0 if is_target else -1)
            row_end = len(row_metadata)
            step_metadata.append(
                {
                    "step": tick_step,
                    "observation_hash": f"hash-{cell_id}-{tick_step}",
                    "uids": list(tick_uid_values),
                    "row_start": row_start,
                    "row_end": row_end,
                    "world_shape": [len(tick_uid_values), preflight.OBSERVATION_DIM],
                }
            )
        npz_path = root / f"cell_{cell_id:02d}.npz"
        np.savez_compressed(
            npz_path,
            world=np.asarray(worlds, dtype=np.float32).reshape((-1, preflight.OBSERVATION_DIM)),
            step=np.asarray(steps, dtype=np.int64),
            uid=np.asarray(uids, dtype=np.int64),
            arrival_id=np.asarray(arrivals, dtype=np.int64),
            row_index=np.arange(len(row_metadata), dtype=np.int64),
            actor_mask=np.asarray(actor_masks, dtype=np.int8),
            requested_gate=np.asarray(requested_gates, dtype=np.int8),
            requested_proposal=np.asarray(requested_proposals, dtype=np.int8),
            metadata_json=np.asarray(
                json.dumps(
                    {"cell_id": cell_id, "steps_expected": expected_steps[cell_id], "rows": len(row_metadata)},
                    sort_keys=True,
                )
            ),
        )
        cell_payload = {
            "stage": preflight.ANCHOR_STAGE,
            "status": "COMPLETED",
            "arm": "before_prefix",
            "cell_id": cell_id,
            "input_binding": {
                "full_sha256": preflight.ANCHOR_SOURCE_FULL_SHA256,
                "expected_steps": expected_steps[cell_id],
            },
            "replay_exact": True,
            "replay": {
                "exact": True,
                "config_match": True,
                "reset_recipe_match": True,
                "reset_state_match": True,
                "base_reset_state_match": True,
                "config_compatibility": None,
            },
            "controller_installed_and_restored": True,
            "caller_rng_unchanged_before_restore": True,
            "caller_rng_unchanged_after_restore": True,
            "steps_expected": expected_steps[cell_id],
            "steps_replayed": expected_steps[cell_id],
            "rows": row_metadata,
            "step_metadata": step_metadata,
            "empty_world_ticks": [tick for tick in step_metadata if not tick["uids"]],
            "anchors_npz": {
                "path": f"cell_{cell_id:02d}.npz",
                "sha256": preflight._sha256(npz_path),
                "shape": [len(row_metadata), preflight.OBSERVATION_DIM],
                "dtype": "float32",
            },
        }
        cell_path = root / "cells" / f"cell_{cell_id:02d}.json"
        cell_sha = _write_json(cell_path, cell_payload)
        summaries.append(
            {
                "cell_id": cell_id,
                "status": "COMPLETED",
                "rows": len(row_metadata),
                "steps": expected_steps[cell_id],
                "cell_json_path": str(cell_path),
                "cell_json_sha256": cell_sha,
                "anchors_npz_path": str(npz_path),
                "anchors_npz_sha256": preflight._sha256(npz_path),
            }
        )
    _write_json(
        root / "full.json",
        {
            "stage": preflight.ANCHOR_STAGE,
            "status": "FULL_OK",
            "source_full_sha256": preflight.ANCHOR_SOURCE_FULL_SHA256,
            "event_full_sha256": preflight.EVENT_FULL_SHA256,
            "cells": summaries,
        },
    )
    return root


def _refresh_cell_summary(root: Path, cell_id: int) -> None:
    full_path = root / "full.json"
    full, _ = preflight._read_json(full_path)
    cell_path = root / "cells" / f"cell_{cell_id:02d}.json"
    cell, _ = preflight._read_json(cell_path)
    npz_path = root / f"cell_{cell_id:02d}.npz"
    summary = next(item for item in full["cells"] if item["cell_id"] == cell_id)
    summary["cell_json_sha256"] = preflight._sha256(cell_path)
    summary["anchors_npz_sha256"] = preflight._sha256(npz_path)
    _write_json(full_path, full)


def test_math_bce_kl_and_equal_mean() -> None:
    assert preflight.bernoulli_kl(0.5, 0.5) == 0.0
    assert preflight.binary_cross_entropy(0.5, 1.0) == pytest.approx(np.log(2.0))
    expected = (preflight.bernoulli_kl(0.6, 0.5) + preflight.bernoulli_kl(0.4, 0.5)) / 2.0
    assert preflight.mean_anchor_kl([0.6, 0.4], [0.5, 0.5]) == pytest.approx(expected)


def test_missing_anchor_returns_not_ready_without_fabricated_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = _make_target(tmp_path, monkeypatch)
    result = preflight.build_readiness(target, tmp_path / "missing-anchors")
    assert result["status"] == "NOT_READY"
    assert "qualification" not in result
    assert result["scope"]["optimizer_used"] is False
    assert result["scope"]["candidate_checkpoint_writer"] is False
    assert result["unresolved"]
    assert "weights" not in result and "fitted_targets" not in result


def test_complete_fake_anchor_manifest_is_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = _make_target(tmp_path, monkeypatch)
    anchors = _make_anchor_bundle(tmp_path, monkeypatch)
    result = preflight.build_readiness(target, anchors)
    assert result["status"] == "READY_FOR_ROOT_REVIEW"
    assert result["anchors"]["total_rows"] == 4
    assert sorted(result["anchors"]["target_rows"]) == [1, 2, 3, 4]
    assert result["qualification"]["fit_authorized"] is False
    assert result["scope"]["PPO_ALLOWED"] is False
    assert result["scope"]["backprop_used"] is False


def test_world_shape_rejection_is_not_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = _make_target(tmp_path, monkeypatch)
    anchors = _make_anchor_bundle(tmp_path, monkeypatch)
    broken = anchors / "cell_01.npz"
    np.savez_compressed(
        broken,
        world=np.zeros((1, preflight.OBSERVATION_DIM - 1), dtype=np.float32),
        step=np.asarray([0]), uid=np.asarray([15]), arrival_id=np.asarray([101]),
        row_index=np.asarray([0]), actor_mask=np.asarray([True]),
        requested_gate=np.asarray([1]), requested_proposal=np.asarray([0]),
        metadata_json=np.asarray(json.dumps({"cell_id": 1})),
    )
    cell_path = anchors / "cells" / "cell_01.json"
    cell, _ = preflight._read_json(cell_path)
    cell["anchors_npz"]["sha256"] = preflight._sha256(broken)
    cell["anchors_npz"]["shape"] = [1, preflight.OBSERVATION_DIM - 1]
    _write_json(cell_path, cell)
    _refresh_cell_summary(anchors, 1)
    result = preflight.build_readiness(target, anchors)
    assert result["status"] == "NOT_READY"
    assert any("world" in item for item in result["unresolved"])


def test_target_binding_and_key_rejection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = _make_target(tmp_path, monkeypatch)
    payload, _ = preflight._read_json(target)
    payload["cells"][0]["selection"]["registration_key"] = "wrong"
    digest = _write_json(target, payload)
    monkeypatch.setattr(preflight, "TARGETS_SHA256", digest)
    with pytest.raises(preflight.PreflightError, match="registration key"):
        preflight._validate_targets(target)


def test_existing_output_is_refused_and_result_is_strict_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = _make_target(tmp_path, monkeypatch)
    anchors = _make_anchor_bundle(tmp_path, monkeypatch)
    result = preflight.build_readiness(target, anchors)
    json.dumps(result, allow_nan=False)
    output = tmp_path / "result.json"
    output.write_text("{}", encoding="utf-8")
    with pytest.raises(preflight.PreflightError, match="overwrite"):
        preflight._write_json(output, result, refuse_existing=True)


def test_readiness_scope_contains_no_execution_capability(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = _make_target(tmp_path, monkeypatch)
    result = preflight.build_readiness(target, tmp_path / "missing")
    scope = result["scope"]
    assert scope == {
        "checkpoint_sha256": preflight.BEFORE_CHECKPOINT_SHA256,
        "trainable_scope": "gate_adapter.* only",
        "critic_frozen": True,
        "proposal_and_temperature_frozen": True,
        "fit_seed": preflight.FIT_SEED,
        "learning_rate": preflight.LEARNING_RATE,
        "max_updates": preflight.MAX_UPDATES,
        "future_optimizer": "Adam",
        "future_weight_decay": 0.0,
        "future_anchor_chunk_size": 512,
        "future_anchor_weighting": "equal overall mean across all anchor rows",
        "objective": "mean four-row BCE(target pstar) + 1.0 * mean Bernoulli KL(candidate || BEFORE) over every anchor row",
        "kl_cap": preflight.KL_CAP,
        "rollback": "restore previous valid adapter parameters and stop",
        "PPO_ALLOWED": False,
        "optimizer_used": False,
        "backprop_used": False,
        "model_forward_used": False,
        "candidate_checkpoint_writer": False,
        "heldout_used": False,
    }


def test_missing_target_row_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = _make_target(tmp_path, monkeypatch)
    anchors = _make_anchor_bundle(tmp_path, monkeypatch)
    cell_path = anchors / "cells" / "cell_04.json"
    cell, _ = preflight._read_json(cell_path)
    cell["rows"][0]["step"] = 1
    _write_json(cell_path, cell)
    _refresh_cell_summary(anchors, 4)
    result = preflight.build_readiness(target, anchors)
    assert result["status"] == "NOT_READY"
    assert any("row/tick" in item for item in result["unresolved"])


def test_realistic_multi_actor_and_empty_tick_fixture_is_valid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = _make_target(tmp_path, monkeypatch)
    anchors = _make_anchor_bundle(tmp_path, monkeypatch, realistic=True)
    result = preflight.build_readiness(target, anchors)
    assert result["status"] == "READY_FOR_ROOT_REVIEW"
    assert result["anchors"]["total_ticks"] == 8
    assert result["anchors"]["expected_ticks"] == 8
    assert result["anchors"]["total_rows"] == 8
    assert result["anchors"]["cells"][1]["ticks"] == 2
    assert result["anchors"]["cells"][1]["rows"] == 2


def test_tick_coverage_must_start_at_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = _make_target(tmp_path, monkeypatch)
    anchors = _make_anchor_bundle(tmp_path, monkeypatch, realistic=True)
    cell_path = anchors / "cells" / "cell_01.json"
    cell, _ = preflight._read_json(cell_path)
    for tick in cell["step_metadata"]:
        tick["step"] += 1
    for row in cell["rows"]:
        row["step"] += 1
    _write_json(cell_path, cell)
    npz_path = anchors / "cell_01.npz"
    with np.load(npz_path, allow_pickle=False) as loaded:
        arrays = {key: np.asarray(loaded[key]) for key in loaded.files}
    arrays["step"] = arrays["step"] + 1
    np.savez_compressed(npz_path, **arrays)
    cell["anchors_npz"]["sha256"] = preflight._sha256(npz_path)
    _write_json(cell_path, cell)
    _refresh_cell_summary(anchors, 1)
    result = preflight.build_readiness(target, anchors)
    assert result["status"] == "NOT_READY"
    assert any("tick coverage is not exact" in item for item in result["unresolved"])


def test_empty_tick_row_range_and_uid_order_are_required(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = _make_target(tmp_path, monkeypatch)
    anchors = _make_anchor_bundle(tmp_path, monkeypatch, realistic=True)
    cell_path = anchors / "cells" / "cell_01.json"
    cell, _ = preflight._read_json(cell_path)
    cell["step_metadata"][1]["uids"] = [999]
    _write_json(cell_path, cell)
    _refresh_cell_summary(anchors, 1)
    result = preflight.build_readiness(target, anchors)
    assert result["status"] == "NOT_READY"
    assert any("row range/UID count" in item for item in result["unresolved"])


def test_malformed_uid_map_is_not_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = _make_target(tmp_path, monkeypatch)
    anchors = _make_anchor_bundle(tmp_path, monkeypatch, realistic=True)
    cell_path = anchors / "cells" / "cell_02.json"
    cell, _ = preflight._read_json(cell_path)
    cell["step_metadata"][0]["uids"] = ["not-an-integer"]
    _write_json(cell_path, cell)
    _refresh_cell_summary(anchors, 2)
    result = preflight.build_readiness(target, anchors)
    assert result["status"] == "NOT_READY"
    assert any("UID list is malformed" in item for item in result["unresolved"])


def test_source_positive_safety_or_failed_replay_is_not_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = _make_target(tmp_path, monkeypatch)
    anchors = _make_anchor_bundle(tmp_path, monkeypatch, realistic=True)
    cell_path = anchors / "cells" / "cell_01.json"
    cell, _ = preflight._read_json(cell_path)
    cell["source_provenance"] = {
        "safety_nonzero": True,
        "lifecycle_valid": True,
        "binding_checks": {"config_snapshot_unchanged": True},
    }
    _write_json(cell_path, cell)
    _refresh_cell_summary(anchors, 1)
    result = preflight.build_readiness(target, anchors)
    assert result["status"] == "NOT_READY"
    assert any("safety certificate is nonzero" in item for item in result["unresolved"])

    anchors = _make_anchor_bundle(tmp_path / "replay", monkeypatch, realistic=True)
    target = _make_target(tmp_path / "replay", monkeypatch)
    cell_path = anchors / "cells" / "cell_02.json"
    cell, _ = preflight._read_json(cell_path)
    cell["replay"]["config_match"] = False
    _write_json(cell_path, cell)
    _refresh_cell_summary(anchors, 2)
    result = preflight.build_readiness(target, anchors)
    assert result["status"] == "NOT_READY"
    assert any("config_match" in item for item in result["unresolved"])


def test_npz_row_mapping_and_restoration_flags_are_required(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = _make_target(tmp_path, monkeypatch)
    anchors = _make_anchor_bundle(tmp_path, monkeypatch, realistic=True)
    npz_path = anchors / "cell_02.npz"
    with np.load(npz_path, allow_pickle=False) as loaded:
        arrays = {key: np.asarray(loaded[key]) for key in loaded.files}
    arrays["uid"] = arrays["uid"].copy()
    arrays["uid"][0] = 9999
    np.savez_compressed(npz_path, **arrays)
    cell_path = anchors / "cells" / "cell_02.json"
    cell, _ = preflight._read_json(cell_path)
    cell["anchors_npz"]["sha256"] = preflight._sha256(npz_path)
    _write_json(cell_path, cell)
    _refresh_cell_summary(anchors, 2)
    result = preflight.build_readiness(target, anchors)
    assert result["status"] == "NOT_READY"
    assert any("NPZ UID mismatch" in item for item in result["unresolved"])

    anchors = _make_anchor_bundle(tmp_path / "flags", monkeypatch, realistic=True)
    cell_path = anchors / "cells" / "cell_03.json"
    cell, _ = preflight._read_json(cell_path)
    cell["caller_rng_unchanged_after_restore"] = False
    _write_json(cell_path, cell)
    _refresh_cell_summary(anchors, 3)
    target = _make_target(tmp_path / "flags", monkeypatch)
    result = preflight.build_readiness(target, anchors)
    assert result["status"] == "NOT_READY"
    assert any("caller RNG post-restore" in item for item in result["unresolved"])
