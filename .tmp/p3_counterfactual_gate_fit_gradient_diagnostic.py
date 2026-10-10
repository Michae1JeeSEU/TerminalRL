"""Read-only gradient diagnostics for the rejected P3 adapter fit.

This script constructs the frozen BEFORE actor, computes source target-BCE and
source anchor-KL gradients with autograd, and reports an algebraic first-Adam
direction without assigning it to the model.  It never creates an optimizer,
environment, candidate checkpoint, or rollout.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.nn import functional as F

from terminal_marl.training import fit_p3_counterfactual_gate_adapter_20261001 as fit


SOURCE = "/home/user/training_outputs/terminal/p3_normal_advantage_gate_repair_20260928/fit_smoke/terminal_gate_repair.pt"
TARGETS = "/home/user/training_outputs/terminal/p3_followup_20260930/counterfactual_gate_targets_20261001/targets_v1.json"
READINESS = "/home/user/training_outputs/terminal/p3_followup_20260930/counterfactual_gate_targets_20261001/fit_readiness_v1.json"
OUT = Path(".tmp/p3_counterfactual_gate_fit_gradient_diagnostic.json")
FIT_ROOT = Path("D:/7_codespace/training_outputs/terminal/p3_followup_20260930/counterfactual_gate_fit_20261001")
FIT_SEED = 9303501
LEARNING_RATE = 1e-4
ANCHOR_CHUNK = 512
ADAPTER_NAMES = (
    "gate_adapter.0.weight",
    "gate_adapter.0.bias",
    "gate_adapter.2.weight",
    "gate_adapter.2.bias",
    "gate_adapter.4.weight",
    "gate_adapter.4.bias",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _finite(value: Any) -> bool:
    try:
        return bool(math.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _grad_stats(gradient: torch.Tensor) -> dict[str, Any]:
    g = gradient.detach().to(dtype=torch.float64).cpu()
    flat = g.reshape(-1)
    return {
        "shape": list(g.shape),
        "max_abs": float(flat.abs().max().item()) if flat.numel() else 0.0,
        "l2": float(torch.linalg.vector_norm(flat).item()) if flat.numel() else 0.0,
        "nonzero_count_exact": int(torch.count_nonzero(flat).item()),
        "nonzero_count_gt_1e-12": int(torch.count_nonzero(flat.abs() > 1e-12).item()),
        "finite": bool(torch.isfinite(flat).all().item()),
    }


def _parameter_scope(model: torch.nn.Module) -> tuple[list[torch.nn.Parameter], dict[int, bool]]:
    params: list[torch.nn.Parameter] = []
    old_flags: dict[int, bool] = {}
    names: list[str] = []
    for name, parameter in model.named_parameters():
        old_flags[id(parameter)] = bool(parameter.requires_grad)
        allowed = name in ADAPTER_NAMES
        parameter.requires_grad_(allowed)
        if allowed:
            names.append(name)
            params.append(parameter)
    if tuple(names) != ADAPTER_NAMES:
        raise RuntimeError(f"unexpected adapter parameter scope: {names}")
    return params, old_flags


def _restore_parameter_flags(model: torch.nn.Module, old_flags: Mapping[int, bool]) -> None:
    for parameter in model.parameters():
        if id(parameter) in old_flags:
            parameter.requires_grad_(bool(old_flags[id(parameter)]))


def _target_gate_rows(
    model: torch.nn.Module,
    target_world: torch.Tensor,
    target_pstar: torch.Tensor,
    params: list[torch.nn.Parameter],
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], torch.Tensor]:
    logits, _ = fit._gate_and_proposal_logits(model, target_world)
    loss_each = F.binary_cross_entropy_with_logits(
        logits.to(dtype=torch.float64), target_pstar.to(dtype=torch.float64), reduction="none"
    )
    loss = loss_each.mean()
    loss_grad = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=False)
    row_grads = [
        torch.autograd.grad(logits[index], params, retain_graph=True, allow_unused=False)
        for index in range(int(logits.shape[0]))
    ]
    return logits, loss_each, list(row_grads), loss_grad


def _anchor_reference(model: torch.nn.Module, world: torch.Tensor) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, int(world.shape[0]), ANCHOR_CHUNK):
            gate, _ = fit._gate_and_proposal_logits(model, world[start : start + ANCHOR_CHUNK])
            parts.append(gate.detach().clone())
    return torch.cat(parts, dim=0)


def _anchor_kl_gradient(
    model: torch.nn.Module,
    world: torch.Tensor,
    reference: torch.Tensor,
    params: list[torch.nn.Parameter],
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    total = int(world.shape[0])
    accumulated = [torch.zeros_like(parameter, dtype=torch.float64) for parameter in params]
    values: list[torch.Tensor] = []
    for start in range(0, total, ANCHOR_CHUNK):
        stop = min(start + ANCHOR_CHUNK, total)
        logits, _ = fit._gate_and_proposal_logits(model, world[start:stop])
        kl = fit._binary_kl_from_logits(logits, reference[start:stop])
        values.append(kl.detach().cpu())
        loss = kl.mean() * (float(stop - start) / float(total))
        gradients = torch.autograd.grad(loss, params, allow_unused=False)
        for index, gradient in enumerate(gradients):
            accumulated[index].add_(gradient.detach().to(dtype=torch.float64).cpu())
    return torch.cat(values, dim=0), accumulated


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
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


def _fit_artifact_binding() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("fit.json", "progress.json", "preflight.json", "fit_v1.log"):
        path = FIT_ROOT / name
        item: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
        if path.is_file():
            item["sha256"] = _sha256(path)
            if path.suffix == ".json":
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(payload, dict):
                        for key in ("status", "stage", "phase", "stop_reason", "accepted_updates", "rejected_updates", "candidate_checkpoint"):
                            if key in payload:
                                item[key] = payload[key]
                except Exception as exc:  # evidence collection keeps raw parse fact
                    item["parse_error"] = f"{type(exc).__name__}: {exc}"
        result[name] = item
    return result


def main() -> int:
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite existing diagnostic: {OUT}")
    old_threads = torch.get_num_threads()
    caller_rng = fit._snapshot_rng()
    model = None
    old_flags: dict[int, bool] = {}
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    try:
        torch.set_num_threads(1)
        random.seed(FIT_SEED)
        np.random.seed(FIT_SEED)
        torch.manual_seed(FIT_SEED)

        source_path = fit._resolve_file(SOURCE)
        source_sha_before = _sha256(source_path)
        model, metadata, source_report, source_sha = fit._load_source_model(SOURCE)
        state_digest_before = fit._state_digest(model.state_dict())
        targets, readiness, world, target_indices, anchor_reports, readiness_sha, target_sha = fit._load_frozen_inputs(
            TARGETS, READINESS, None
        )
        target_world, target_values, target_ids = fit._target_rows(targets, world, target_indices)
        target_pstar = target_values[:, 0]
        target_p0 = target_values[:, 1]

        with torch.no_grad():
            target_source_logits, target_source_q = fit._gate_and_proposal_logits(model, target_world)
            target_source_probs = torch.sigmoid(target_source_logits.to(dtype=torch.float64))
            anchor_reference_logits = _anchor_reference(model, world)
            anchor_reference_q = fit._compute_reference_outputs(model, world)[1]

        target_p0_absdiff = (target_source_probs - target_p0).abs()
        params, old_flags = _parameter_scope(model)
        target_logits, target_bce_each, target_row_grads, target_grads = _target_gate_rows(
            model, target_world, target_pstar, params
        )
        anchor_kl_values, anchor_grads = _anchor_kl_gradient(
            model, world, anchor_reference_logits, params
        )

        first_adam_delta = [
            -LEARNING_RATE * gradient.detach().to(dtype=torch.float64).cpu()
            / (gradient.detach().to(dtype=torch.float64).cpu().abs() + 1e-8)
            for gradient in target_grads
        ]
        first_order = []
        for row_index, row_grads in enumerate(target_row_grads):
            logit_delta = sum(
                float((gradient.detach().to(dtype=torch.float64).cpu() * delta).sum().item())
                for gradient, delta in zip(row_grads, first_adam_delta)
            )
            probability = float(target_source_probs[row_index].item())
            first_order.append({
                "target_id": int(target_ids[row_index]),
                "logit_delta_dot_first_adam_algebra": logit_delta,
                "probability_delta_first_order": logit_delta * probability * (1.0 - probability),
                "label": "first_order_algebra_only_no_parameter_assignment",
            })

        with torch.no_grad():
            after_target_logits, after_target_q = fit._gate_and_proposal_logits(model, target_world)
            after_anchor_logits, after_anchor_q = fit._compute_reference_outputs(model, world)
        state_digest_after = fit._state_digest(model.state_dict())
        source_sha_after = _sha256(source_path)
        q_target_max_absdiff = float((after_target_q - target_source_q).abs().max().item())
        q_anchor_max_absdiff = float((after_anchor_q - anchor_reference_q).abs().max().item())

        result = {
            "schema": "p3_counterfactual_gate_fit_gradient_diagnostic_20261002_v1",
            "status": "COMPLETED_READ_ONLY_GRADIENT_DIAGNOSTIC",
            "stage": "P3_COUNTERFACTUAL_GATE_FIT_GRADIENT_DIAGNOSTIC_20261002_V1",
            "fit_artifacts": _fit_artifact_binding(),
            "source": {
                "checkpoint_path": str(source_path),
                "checkpoint_sha256_before": source_sha_before,
                "checkpoint_sha256_after": source_sha_after,
                "expected_checkpoint_sha256": fit.BEFORE_CHECKPOINT_SHA256,
                "model_digest_before": state_digest_before,
                "model_digest_after": state_digest_after,
                "expected_model_digest": fit.BEFORE_MODEL_DIGEST,
                "state_unchanged": bool(state_digest_before == state_digest_after),
                "metadata_architecture_id": metadata.get("architecture_id"),
                "metadata_temperature": metadata.get("temperature"),
                "source_sha_returned_by_loader": source_sha,
            },
            "inputs": {
                "targets_sha256": target_sha,
                "readiness_sha256": readiness_sha,
                "anchor_rows": int(world.shape[0]),
                "anchor_observation_dim": int(world.shape[1]),
                "anchor_cells": anchor_reports,
                "target_ids": [int(value) for value in target_ids],
                "fit_seed": FIT_SEED,
                "cpu_threads": 1,
            },
            "target": {
                "pstar": [float(value) for value in target_pstar.tolist()],
                "p0_recorded": [float(value) for value in target_p0.tolist()],
                "source_gate_logits": [float(value) for value in target_source_logits.tolist()],
                "source_gate_probabilities": [float(value) for value in target_source_probs.tolist()],
                "bce_per_target": [float(value) for value in target_bce_each.tolist()],
                "bce_mean": float(target_bce_each.mean().item()),
                "source_p0_absdiff_per_target": [float(value) for value in target_p0_absdiff.tolist()],
                "source_p0_absdiff_global_max": float(target_p0_absdiff.max().item()),
                "proposal_q_max_absdiff_after_same_model": q_target_max_absdiff,
                "gradient_bce_by_parameter": {
                    name: _grad_stats(gradient)
                    for name, gradient in zip(ADAPTER_NAMES, target_grads)
                },
                "first_order_algebraic_delta": first_order,
            },
            "anchor": {
                "kl_loss_mean": float(anchor_kl_values.mean().item()),
                "kl_min": float(anchor_kl_values.min().item()),
                "kl_max": float(anchor_kl_values.max().item()),
                "kl_absmax": float(anchor_kl_values.abs().max().item()),
                "gate_logit_absmax_global": float(anchor_reference_logits.abs().max().item()),
                "gate_probability_absmax_global": float(torch.sigmoid(anchor_reference_logits).abs().max().item()),
                "proposal_q_max_absdiff_after_same_model": q_anchor_max_absdiff,
                "gradient_kl_by_parameter": {
                    name: _grad_stats(gradient)
                    for name, gradient in zip(ADAPTER_NAMES, anchor_grads)
                },
            },
            "scope": {
                "optimizer_instance_created": False,
                "optimizer_step_called": False,
                "parameter_assignment": False,
                "candidate_checkpoint_written": False,
                "environment_created": False,
                "rollout_called": False,
                "teacher_used": False,
                "PPO_ALLOWED": False,
                "heldout_used": False,
                "delta_definition": "-1e-4 * g / (abs(g) + 1e-8), tensor-only first-Adam algebra",
            },
        }
        if not all(_finite(value) for value in anchor_kl_values.tolist()):
            raise RuntimeError("non-finite anchor KL value")
    except Exception as exc:
        error = {
            "schema": "p3_counterfactual_gate_fit_gradient_diagnostic_20261002_v1",
            "status": "FAILED_READ_ONLY_GRADIENT_DIAGNOSTIC",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "fit_artifacts": _fit_artifact_binding(),
        }
    finally:
        if model is not None:
            _restore_parameter_flags(model, old_flags)
        fit._restore_rng(caller_rng)
        torch.set_num_threads(old_threads)

    payload = result if result is not None else error
    if payload is None:
        raise RuntimeError("diagnostic produced no payload")
    payload["caller_rng_restored"] = fit._rng_equal(caller_rng, fit._snapshot_rng())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload.get("status"),
        "output": str(OUT),
        "source_state_unchanged": payload.get("source", {}).get("state_unchanged"),
        "caller_rng_restored": payload.get("caller_rng_restored"),
        "anchor_rows": payload.get("inputs", {}).get("anchor_rows"),
    }, sort_keys=True))
    return 0 if payload.get("status") == "COMPLETED_READ_ONLY_GRADIENT_DIAGNOSTIC" else 1


if __name__ == "__main__":
    raise SystemExit(main())
