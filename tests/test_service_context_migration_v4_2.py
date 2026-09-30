from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from terminal_marl.algo.gate_adapter_v4_2 import (
    GateResidualAdapterActorCentralCriticV42,
)
from terminal_marl.algo.gate_bounded_calibrated_v4_2 import (
    BoundedTemperatureCalibratedGateResidualAdapterActorCentralCriticV42,
)
from terminal_marl.algo.gate_calibrated_adapter_v4_2 import (
    TemperatureCalibratedGateResidualAdapterActorCentralCriticV42,
)
from terminal_marl.algo.joint_ppo_v4_2 import HierarchicalActorCentralCriticV42
from terminal_marl.algo.service_context_migration_v4_2 import (
    EXPANDED_INPUT_WEIGHT_NAMES,
    SOURCE_OBSERVATION_DIM,
    TARGET_OBSERVATION_DIM,
    TARGET_OBSERVATION_SCHEMA,
    migrate_service_context_model,
)


MODEL_TYPES = (
    HierarchicalActorCentralCriticV42,
    GateResidualAdapterActorCentralCriticV42,
    TemperatureCalibratedGateResidualAdapterActorCentralCriticV42,
    BoundedTemperatureCalibratedGateResidualAdapterActorCentralCriticV42,
)


def _make_source(model_type):
    torch.manual_seed(20260930)
    if model_type is TemperatureCalibratedGateResidualAdapterActorCentralCriticV42:
        source = model_type(SOURCE_OBSERVATION_DIM, temperature=1.75)
    elif model_type is BoundedTemperatureCalibratedGateResidualAdapterActorCentralCriticV42:
        source = model_type(
            SOURCE_OBSERVATION_DIM,
            temperature=1.75,
            minimum_gate_probability=0.075,
        )
    else:
        source = model_type(SOURCE_OBSERVATION_DIM)
    source = source.to(dtype=torch.float64)
    source.eval()
    # Preserve a non-uniform module-mode pattern and mixed parameter grad flags.
    source.actor_encoder.train()
    for index, parameter in enumerate(source.parameters()):
        parameter.requires_grad_(index % 2 == 0)
    return source


def _snapshot_model(source: nn.Module):
    return {
        "state": {key: value.detach().clone() for key, value in source.state_dict().items()},
        "training": {name: module.training for name, module in source.named_modules()},
        "requires_grad": {
            name: parameter.requires_grad
            for name, parameter in source.named_parameters()
        },
    }


def _assert_close_with_rawdiff(expected: torch.Tensor, actual: torch.Tensor, label: str) -> None:
    rawdiff = float((expected - actual).abs().max().detach().cpu())
    assert rawdiff <= 1e-6, f"{label} rawdiff={rawdiff:.12g}"


def _outputs(model: nn.Module, observations: torch.Tensor, world: torch.Tensor):
    mask = torch.ones(world.shape[:2], dtype=torch.bool, device=world.device)
    gate, proposal = model.distributions(observations)
    value = model.critic(world, mask)
    return gate.logits.detach(), proposal.logits.detach(), value.detach()


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_service_context_migration_preserves_prefix_outputs_and_model_state(model_type):
    source = _make_source(model_type)
    before = _snapshot_model(source)
    rng_before = torch.random.get_rng_state().clone()

    target, metadata = migrate_service_context_model(source)

    assert type(target) is model_type
    assert metadata["source_observation_dim"] == SOURCE_OBSERVATION_DIM
    assert metadata["source_observation_schema"] == "phaseA_low_obs_v2_1_retry_state"
    assert metadata["target_observation_dim"] == TARGET_OBSERVATION_DIM
    assert metadata["target_observation_schema"] == TARGET_OBSERVATION_SCHEMA
    assert metadata["state_keys_strictly_equal"] is True
    assert metadata["optimizer_state_migrated"] is False
    assert metadata["optimizer_initialization"] == "fresh_optimizer_required"
    assert metadata["qualification_status"] == "experimental migration only; not qualification"
    assert torch.equal(torch.random.get_rng_state(), rng_before)

    after = _snapshot_model(source)
    for key in before["state"]:
        assert torch.equal(before["state"][key], after["state"][key]), key
    assert before["training"] == after["training"]
    assert before["requires_grad"] == after["requires_grad"]

    target_state = target.state_dict()
    source_state = source.state_dict()
    assert list(target_state) == list(source_state)
    for key, source_value in source_state.items():
        target_value = target_state[key]
        assert target_value.dtype == source_value.dtype, key
        if key in EXPANDED_INPUT_WEIGHT_NAMES:
            assert target_value.shape[0] == source_value.shape[0]
            assert target_value.shape[1] == TARGET_OBSERVATION_DIM
            assert torch.equal(target_value[:, :SOURCE_OBSERVATION_DIM], source_value), key
            assert torch.count_nonzero(target_value[:, SOURCE_OBSERVATION_DIM:]) == 0, key
        else:
            assert target_value.shape == source_value.shape, key
            assert torch.equal(target_value, source_value), key

    for name, source_module in source.named_modules():
        target_module = target if name == "" else target.get_submodule(name)
        assert target_module.training is source_module.training
    for name, source_parameter in source.named_parameters():
        assert target.get_parameter(name).requires_grad is source_parameter.requires_grad

    observations = torch.randn(4, SOURCE_OBSERVATION_DIM, dtype=torch.float64)
    extra = torch.randn(4, 2, dtype=torch.float64)
    world = torch.randn(2, 3, SOURCE_OBSERVATION_DIM, dtype=torch.float64)
    world_extra = torch.randn(2, 3, 2, dtype=torch.float64)
    source_outputs = _outputs(source, observations, world)
    target_outputs = _outputs(
        target,
        torch.cat((observations, extra), dim=-1),
        torch.cat((world, world_extra), dim=-1),
    )
    for label, expected, actual in zip(
        ("gate_logits", "proposal_logits", "critic_value"),
        source_outputs,
        target_outputs,
    ):
        _assert_close_with_rawdiff(expected, actual, label)

    for buffer_name in ("gate_temperature", "minimum_gate_probability"):
        if hasattr(source, buffer_name):
            assert torch.equal(
                getattr(source, buffer_name).detach(),
                getattr(target, buffer_name).detach(),
            )

    optimizer = torch.optim.Adam(target.parameters())
    assert optimizer.state == {}


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_service_context_migration_rejects_non_88d_sources(model_type):
    if model_type is TemperatureCalibratedGateResidualAdapterActorCentralCriticV42:
        source = model_type(90, temperature=1.75)
    elif model_type is BoundedTemperatureCalibratedGateResidualAdapterActorCentralCriticV42:
        source = model_type(90, temperature=1.75, minimum_gate_probability=0.075)
    else:
        source = model_type(90)
    with pytest.raises(ValueError, match="88D"):
        migrate_service_context_model(source)


def test_service_context_migration_rejects_unknown_state_structure():
    source = _make_source(HierarchicalActorCentralCriticV42)
    source.extra_parameter = nn.Parameter(torch.ones(1, dtype=torch.float64))
    with pytest.raises(ValueError, match="state keys differ"):
        migrate_service_context_model(source)


def test_service_context_migration_rejects_semantically_different_88d_schema():
    source = _make_source(HierarchicalActorCentralCriticV42)
    with pytest.raises(ValueError, match="requires source observation schema"):
        migrate_service_context_model(
            source,
            source_observation_schema="phaseA_low_obs_v2_3_source_bound_88d",
        )


def test_service_context_migration_does_not_accept_optimizer_or_copy_old_state():
    source = _make_source(GateResidualAdapterActorCentralCriticV42)
    old_optimizer = torch.optim.Adam(source.parameters())
    old_optimizer_state = copy.deepcopy(old_optimizer.state_dict())
    target, metadata = migrate_service_context_model(source)
    new_optimizer = torch.optim.Adam(target.parameters())

    assert old_optimizer.state_dict() == old_optimizer_state
    assert new_optimizer.state == {}
    assert metadata["optimizer_state_migrated"] is False
