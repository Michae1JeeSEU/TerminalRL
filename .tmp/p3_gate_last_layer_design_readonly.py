"""Read-only rank check for the frozen source gate adapter target states."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from terminal_marl.training import fit_p3_counterfactual_gate_adapter_20261001 as fit


SOURCE = "/home/user/training_outputs/terminal/p3_normal_advantage_gate_repair_20260928/fit_smoke/terminal_gate_repair.pt"
TARGETS = "/home/user/training_outputs/terminal/p3_followup_20260930/counterfactual_gate_targets_20261001/targets_v1.json"
READINESS = "/home/user/training_outputs/terminal/p3_followup_20260930/counterfactual_gate_targets_20261001/fit_readiness_v1.json"
OUT = Path(".tmp/p3_gate_last_layer_design_readonly.json")
SEED = 9303501


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def main() -> int:
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite existing output: {OUT}")
    old_threads = torch.get_num_threads()
    caller_rng = fit._snapshot_rng()
    model = None
    payload: dict[str, Any]
    try:
        torch.set_num_threads(1)
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        source_path = fit._resolve_file(SOURCE)
        source_sha_before = fit._sha256(source_path)
        model, metadata, _report, source_sha = fit._load_source_model(SOURCE)
        digest_before = fit._state_digest(model.state_dict())
        targets, readiness, world, target_indices, anchor_reports, readiness_sha, targets_sha = fit._load_frozen_inputs(
            TARGETS, READINESS, None
        )
        target_world, target_values, target_ids = fit._target_rows(targets, world, target_indices)

        adapter = model.gate_adapter
        if not isinstance(adapter, nn.Sequential) or len(adapter) < 2:
            raise RuntimeError("gate_adapter is not a sequential adapter with a final layer")
        last = adapter[-1]
        if not isinstance(last, nn.Linear):
            raise RuntimeError("gate_adapter final module is not nn.Linear")
        with torch.no_grad():
            hidden = adapter[:-1](target_world)
        if hidden.ndim != 2 or hidden.shape[0] != 4:
            raise RuntimeError(f"unexpected hidden shape: {tuple(hidden.shape)}")
        design = torch.cat(
            (hidden.detach().to(dtype=torch.float64), torch.ones((hidden.shape[0], 1), dtype=torch.float64)),
            dim=1,
        )
        singular_values = torch.linalg.svdvals(design)
        max_sv = float(singular_values.max().item()) if singular_values.numel() else 0.0
        rank_tolerance = float(max(design.shape) * torch.finfo(torch.float64).eps * max_sv)
        rank = int(torch.linalg.matrix_rank(design, tol=rank_tolerance).item())

        digest_after = fit._state_digest(model.state_dict())
        source_sha_after = fit._sha256(source_path)
        gate_temperature = getattr(model, "gate_temperature", None)
        if isinstance(gate_temperature, torch.Tensor):
            gate_temperature_value = float(gate_temperature.detach().cpu().item())
            gate_temperature_dtype = str(gate_temperature.dtype)
        else:
            gate_temperature_value = float(gate_temperature)
            gate_temperature_dtype = type(gate_temperature).__name__

        order_binding = []
        for index, cell_id in enumerate(target_ids):
            target = targets[int(cell_id)]
            order_binding.append({
                "row": index,
                "cell_id": int(cell_id),
                "world_row_index": int(target_indices[int(cell_id)]),
                "pstar": float(target["pstar"]),
                "p0": float(target["p0"]),
            })
        payload = {
            "schema": "p3_gate_last_layer_design_readonly_20261002_v1",
            "status": "COMPLETED_READ_ONLY_STRUCTURE_CHECK",
            "source": {
                "checkpoint_path": str(source_path),
                "checkpoint_sha256_before": source_sha_before,
                "checkpoint_sha256_after": source_sha_after,
                "expected_checkpoint_sha256": fit.BEFORE_CHECKPOINT_SHA256,
                "loader_sha256": source_sha,
                "model_digest_before": digest_before,
                "model_digest_after": digest_after,
                "expected_model_digest": fit.BEFORE_MODEL_DIGEST,
                "state_unchanged": bool(digest_before == digest_after),
                "architecture_id": metadata.get("architecture_id"),
                "gate_temperature": gate_temperature_value,
                "gate_temperature_dtype": gate_temperature_dtype,
            },
            "inputs": {
                "targets_sha256": targets_sha,
                "readiness_sha256": readiness_sha,
                "target_ids_order": [int(value) for value in target_ids],
                "target_order_binding": order_binding,
                "anchor_cells": anchor_reports,
                "anchor_rows": int(world.shape[0]),
                "anchor_observation_dim": int(world.shape[1]),
                "fit_seed": SEED,
                "cpu_threads": 1,
            },
            "design": {
                "hidden_source": "model.gate_adapter[:-1](target_world)",
                "hidden_shape": list(hidden.shape),
                "design_matrix_shape": list(design.shape),
                "design_dtype": str(design.dtype),
                "bias_column": "ones",
                "last_linear_weight_shape": list(last.weight.shape),
                "last_linear_bias_shape": list(last.bias.shape),
                "last_linear_weight_dtype": str(last.weight.dtype),
                "last_linear_bias_dtype": str(last.bias.dtype),
                "singular_values_float64": [float(value) for value in singular_values.tolist()],
                "rank_tolerance": rank_tolerance,
                "rank": rank,
            },
            "scope": {
                "no_grad_forward": True,
                "autograd_called": False,
                "optimizer_created": False,
                "weights_assigned": False,
                "candidate_generated": False,
                "objective_or_delta_generated": False,
                "environment_created": False,
                "rollout_called": False,
            },
        }
    except Exception as exc:
        payload = {
            "schema": "p3_gate_last_layer_design_readonly_20261002_v1",
            "status": "FAILED_READ_ONLY_STRUCTURE_CHECK",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    finally:
        fit._restore_rng(caller_rng)
        torch.set_num_threads(old_threads)
    payload["caller_rng_restored"] = fit._rng_equal(caller_rng, fit._snapshot_rng())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload.get("status"),
        "output": str(OUT),
        "design_shape": payload.get("design", {}).get("design_matrix_shape"),
        "rank": payload.get("design", {}).get("rank"),
        "caller_rng_restored": payload.get("caller_rng_restored"),
    }, sort_keys=True))
    return 0 if payload.get("status") == "COMPLETED_READ_ONLY_STRUCTURE_CHECK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
