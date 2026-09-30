"""Experimental 88D to 90D service-context model migration for V4.2.

The migration is an initialization helper only.  It preserves the source
88D computation on the first 88 input columns and zero-initializes the two new
service-context columns.  It does not alter checkpoint loading or carry an
optimizer state across the schema boundary.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .gate_adapter_v4_2 import GateResidualAdapterActorCentralCriticV42
from .gate_bounded_calibrated_v4_2 import (
    BoundedTemperatureCalibratedGateResidualAdapterActorCentralCriticV42,
)
from .gate_calibrated_adapter_v4_2 import (
    TemperatureCalibratedGateResidualAdapterActorCentralCriticV42,
)
from .joint_ppo_v4_2 import HierarchicalActorCentralCriticV42


SOURCE_OBSERVATION_DIM = 88
TARGET_OBSERVATION_DIM = 90
SOURCE_OBSERVATION_SCHEMA = "phaseA_low_obs_v2_1_retry_state"
TARGET_OBSERVATION_SCHEMA = "phaseA_low_obs_v2_4_service_context_90d"
EXPANDED_INPUT_WEIGHT_NAMES = frozenset(
    {
        "actor_encoder.0.weight",
        "critic_encoder.0.weight",
        "gate_adapter.0.weight",
    }
)
_REQUIRED_INPUT_WEIGHT_NAMES = frozenset(
    {
        "actor_encoder.0.weight",
        "critic_encoder.0.weight",
    }
)
_SUPPORTED_MODEL_TYPES = (
    HierarchicalActorCentralCriticV42,
    GateResidualAdapterActorCentralCriticV42,
    TemperatureCalibratedGateResidualAdapterActorCentralCriticV42,
    BoundedTemperatureCalibratedGateResidualAdapterActorCentralCriticV42,
)


def _tensor_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }


def _source_input_widths(state: dict[str, torch.Tensor]) -> dict[str, int]:
    missing = sorted(_REQUIRED_INPUT_WEIGHT_NAMES - set(state))
    if missing:
        raise ValueError(f"source model is missing required input weights: {missing}")
    keys = [
        key for key in EXPANDED_INPUT_WEIGHT_NAMES
        if key in state
    ]
    widths: dict[str, int] = {}
    for key in keys:
        value = state[key]
        if value.ndim != 2:
            raise ValueError(f"input weight must be rank-2: {key}")
        widths[key] = int(value.shape[1])
    if set(widths.values()) != {SOURCE_OBSERVATION_DIM}:
        raise ValueError(
            "source model must use the 88D observation input for every "
            f"present input weight, got {widths}"
        )
    return widths


def _hidden_width(source: nn.Module) -> int:
    try:
        layer = source.actor_encoder[0]
    except (AttributeError, IndexError, TypeError) as exc:
        raise ValueError("source model has no canonical actor_encoder.0 layer") from exc
    if not isinstance(layer, nn.Linear):
        raise ValueError("actor_encoder.0 must be a Linear layer")
    return int(layer.out_features)


def _buffer_float(source: nn.Module, name: str) -> float:
    value = getattr(source, name, None)
    if not isinstance(value, torch.Tensor) or value.numel() != 1:
        raise ValueError(f"source model is missing scalar buffer: {name}")
    scalar = float(value.detach().cpu())
    if not torch.isfinite(torch.tensor(scalar)):
        raise ValueError(f"source model has non-finite buffer: {name}")
    return scalar


def _construct_target(source: nn.Module) -> nn.Module:
    model_type = type(source)
    hidden = _hidden_width(source)
    if model_type is HierarchicalActorCentralCriticV42:
        return model_type(TARGET_OBSERVATION_DIM, hidden=hidden)
    if model_type is GateResidualAdapterActorCentralCriticV42:
        return model_type(TARGET_OBSERVATION_DIM, hidden=hidden)
    if model_type is TemperatureCalibratedGateResidualAdapterActorCentralCriticV42:
        return model_type(
            TARGET_OBSERVATION_DIM,
            temperature=_buffer_float(source, "gate_temperature"),
        )
    if model_type is BoundedTemperatureCalibratedGateResidualAdapterActorCentralCriticV42:
        return model_type(
            TARGET_OBSERVATION_DIM,
            temperature=_buffer_float(source, "gate_temperature"),
            minimum_gate_probability=_buffer_float(source, "minimum_gate_probability"),
        )
    raise TypeError(
        "unsupported service-context migration model type: "
        f"{model_type.__module__}.{model_type.__qualname__}"
    )


def _match_target_tensor_layout(
    target: nn.Module,
    source_state: dict[str, torch.Tensor],
) -> None:
    devices = {value.device for value in source_state.values()}
    if len(devices) != 1:
        raise ValueError(f"source model state must use one device, got {sorted(map(str, devices))}")
    device = next(iter(devices))
    target.to(device=device)
    for module_name, module in target.named_modules():
        for child_name, parameter in module._parameters.items():
            if parameter is None:
                continue
            key = f"{module_name}.{child_name}" if module_name else child_name
            if key not in source_state:
                raise ValueError(f"target parameter is absent from source state: {key}")
            source_value = source_state[key]
            parameter.data = parameter.data.to(device=source_value.device, dtype=source_value.dtype)
        for child_name, buffer in module._buffers.items():
            if buffer is None:
                continue
            key = f"{module_name}.{child_name}" if module_name else child_name
            if key not in source_state:
                raise ValueError(f"target buffer is absent from source state: {key}")
            source_value = source_state[key]
            if source_value.is_floating_point() or source_value.is_complex():
                module._buffers[child_name] = buffer.to(
                    device=source_value.device,
                    dtype=source_value.dtype,
                )
            else:
                module._buffers[child_name] = buffer.to(device=source_value.device)


def _copy_training_and_grad_state(source: nn.Module, target: nn.Module) -> None:
    source_modules = dict(source.named_modules())
    target_modules = dict(target.named_modules())
    if list(source_modules) != list(target_modules):
        raise ValueError("source and target module structures differ")
    for name, source_module in source_modules.items():
        target_modules[name].training = bool(source_module.training)

    source_parameters = dict(source.named_parameters())
    target_parameters = dict(target.named_parameters())
    if list(source_parameters) != list(target_parameters):
        raise ValueError("source and target parameter structures differ")
    for name, source_parameter in source_parameters.items():
        target_parameters[name].requires_grad_(source_parameter.requires_grad)


def migrate_service_context_model(
    source_model: nn.Module,
    *,
    source_observation_schema: str = SOURCE_OBSERVATION_SCHEMA,
) -> tuple[nn.Module, dict[str, Any]]:
    """Create a 90D model with zero-initialized service-context columns.

    The source must be one of the four concrete V4.2 actor/critic model
    classes and the caller must bind it to the V2.1 source checkpoint schema
    through ``source_observation_schema``.  The model object does not carry
    that semantic provenance itself, so a V2.3 88D model is rejected rather
    than treated as a prefix-equivalent source.  The source model, including
    its tensors, mode flags, and gradient flags, is never modified.  The
    returned metadata records that this is an experimental initialization
    migration and that a fresh optimizer is required.
    """
    if source_observation_schema != SOURCE_OBSERVATION_SCHEMA:
        raise ValueError(
            "service-context migration requires source observation schema "
            f"{SOURCE_OBSERVATION_SCHEMA!r}, got {source_observation_schema!r}"
        )
    if type(source_model) not in _SUPPORTED_MODEL_TYPES:
        raise TypeError(
            "source_model must be one of the four supported concrete V4.2 "
            f"model classes, got {type(source_model).__name__}"
        )

    source_state = _tensor_snapshot(source_model)
    _source_input_widths(source_state)
    source_keys = list(source_state)

    # Constructor initialization is isolated from the caller's CPU random
    # stream so the migration cannot alter a later deterministic trajectory.
    with torch.random.fork_rng(devices=[]):
        target = _construct_target(source_model)
    _match_target_tensor_layout(target, source_state)
    target_state = target.state_dict()
    target_keys = list(target_state)
    if source_keys != target_keys:
        missing = sorted(set(source_keys) - set(target_keys))
        unexpected = sorted(set(target_keys) - set(source_keys))
        raise ValueError(
            "source/target state keys differ: "
            f"missing={missing}, unexpected={unexpected}"
        )

    migrated_state: dict[str, torch.Tensor] = {}
    for key in source_keys:
        source_value = source_state[key]
        target_value = target_state[key]
        if key in EXPANDED_INPUT_WEIGHT_NAMES:
            if source_value.ndim != 2 or target_value.ndim != 2:
                raise ValueError(f"expanded input weight must be rank-2: {key}")
            expected_shape = (source_value.shape[0], TARGET_OBSERVATION_DIM)
            if tuple(target_value.shape) != expected_shape:
                raise ValueError(
                    f"target expanded weight shape mismatch for {key}: "
                    f"expected={expected_shape}, got={tuple(target_value.shape)}"
                )
            expanded = torch.zeros_like(target_value)
            expanded[:, :SOURCE_OBSERVATION_DIM].copy_(source_value)
            migrated_state[key] = expanded
        else:
            if tuple(source_value.shape) != tuple(target_value.shape):
                raise ValueError(
                    f"state shape mismatch for {key}: "
                    f"source={tuple(source_value.shape)}, target={tuple(target_value.shape)}"
                )
            migrated_state[key] = source_value.detach().clone()

    target.load_state_dict(migrated_state, strict=True)
    _copy_training_and_grad_state(source_model, target)

    metadata: dict[str, Any] = {
        "migration": "experimental_service_context_v4_2",
        "source_observation_schema": source_observation_schema,
        "target_observation_schema": TARGET_OBSERVATION_SCHEMA,
        "source_observation_dim": SOURCE_OBSERVATION_DIM,
        "target_observation_dim": TARGET_OBSERVATION_DIM,
        "source_model_class": type(source_model).__name__,
        "target_model_class": type(target).__name__,
        "expanded_input_weight_names": sorted(
            key for key in EXPANDED_INPUT_WEIGHT_NAMES if key in source_state
        ),
        "new_columns_initialization": "zeros",
        "state_keys_strictly_equal": True,
        "optimizer_state_migrated": False,
        "optimizer_initialization": "fresh_optimizer_required",
        "initialization_behavior": "88D prefix outputs preserved; appended columns start at zero",
        "qualification_status": "experimental migration only; not qualification",
        "source_schema_contract": (
            "caller-provided V2.1 checkpoint binding; model class and dimension "
            "alone do not establish semantic prefix equivalence"
        ),
    }
    return target, metadata


__all__ = [
    "EXPANDED_INPUT_WEIGHT_NAMES",
    "SOURCE_OBSERVATION_DIM",
    "SOURCE_OBSERVATION_SCHEMA",
    "TARGET_OBSERVATION_DIM",
    "TARGET_OBSERVATION_SCHEMA",
    "migrate_service_context_model",
]
