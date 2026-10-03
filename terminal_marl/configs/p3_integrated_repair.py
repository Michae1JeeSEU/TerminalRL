"""Explicit development configuration for the integrated P3 foundation repair.

This does not promote defaults or authorize training. Historical configurations
and checkpoints retain their original meaning.
"""
from dataclasses import replace

from terminal_marl.core.config import SimConfig

PROFILE = 'p3_integrated_repair_direct_c2_v2_20261003'
PPO_ALLOWED = False
# Used only AFTER a separately approved student-only qualification.
PPO_UPDATE_OPTIONS = {'actor_reduction': 'joint_team', 'entropy_mode': 'gate_only'}


def build_config(*, requests=24, warm_start=12, high_soc=False):
    if not 0 <= warm_start <= requests or requests < 1:
        raise ValueError('require 0 <= warm_start <= requests and requests >= 1')
    schedule = (0.0,)*warm_start + tuple(15.0*(i+1) for i in range(requests-warm_start))
    if not warm_start:
        schedule = tuple(15.0*i for i in range(requests))
    return SimConfig(
        trajectory_schema_version='executable_trajectory_v3',
        action_schema_version='low_action_schema_v4_2_hierarchical',
        observation_schema_version='phaseA_low_obs_v2_4_service_context_90d',
        experiment_mode='fixed_low_student', high_level_enabled=False,
        reset_mode='warm_start_rings' if warm_start else 'flow_from_empty',
        warm_start_uavs=warm_start, warm_start_steady_motion=bool(warm_start),
        warm_start_energy_reachability_check_enabled=bool(warm_start),
        arrival_model='finite_cohort', total_requested_uavs=requests,
        fixed_arrival_schedule=schedule, max_simulation_seconds=1800.0,
        initial_soc_mode='uniform', initial_soc_min=.90 if high_soc else .55,
        initial_soc_max=1.0 if high_soc else .90,
        energy_model_mode='phase_motion_v2', emergency_landing_enabled=True,
        emergency_priority_guard_enabled=True, energy_reachability_guard_enabled=True,
        emergency_priority_mode='committed_route', emergency_trigger_soc=.40,
        route_aware_emergency_enabled=True, executable_admission_energy_enabled=True,
        holding_speed_contract='nominal_until_release',
        final_approach_geometry_mode='continuous_inward_v1',
        final_approach_execution_mode='speed_timing_v1',
        v3_landing_path_mode='direct_c2',
        reconcile_late_merge_slots_enabled=True,
        runtime_proposal_shortlist_k=3, scheduler_mode='execution', teacher_enabled=False,
        v4_2_rejected_release_hold_mode='canonical_wait',
        student_decision_mask_mode='pre_action_forced_wait',
        lz_assignment_mode='nearest_available', release_arbitration_mode='downstream_soc_priority',
        final_queue_capacity=3, task_priority_enabled=False,
        low_level_progress_reward_per_meter=0.0,
        low_level_priority_time_weight=0.0, low_level_soc_urgency_time_weight=2.0,
        low_level_queue_cost_enabled=True, external_queue_step_mode="post_admission",
        low_level_release_rejected_penalty=0.0,
    )


def _legacy_source_config(config):
    return replace(config, trajectory_schema_version='executable_trajectory_v2',
        observation_schema_version='phaseA_low_obs_v2_1_retry_state',
        final_approach_execution_mode='legacy', v3_landing_path_mode='brake_then_curve',
        holding_speed_contract='bounded_residual',
        route_aware_emergency_enabled=False, executable_admission_energy_enabled=False,
        emergency_priority_mode='at_risk_count', low_level_queue_cost_enabled=False,
        external_queue_step_mode='pre_admission')
def load_legacy_source(checkpoint, config):
    """Load the calibrated 88D model without constructing an optimizer."""
    import torch
    from terminal_marl.algo.gate_calibrated_adapter_v4_2 import (
        CALIBRATION_TEMPERATURE,
        TemperatureCalibratedGateResidualAdapterActorCentralCriticV42,
    )
    from terminal_marl.algo.service_context_runtime_v4_2 import ServiceContextInferencePolicyV42
    from terminal_marl.core.checkpoints import load_checkpoint

    with torch.random.fork_rng(devices=[]):
        model = TemperatureCalibratedGateResidualAdapterActorCentralCriticV42(88)
    source = ServiceContextInferencePolicyV42(model, 88, torch.device('cpu'))
    payload = load_checkpoint(checkpoint, source, _legacy_source_config(config), load_optimizer=False)
    if float(model.gate_temperature.item()) != CALIBRATION_TEMPERATURE:
        raise ValueError('candidate temperature changed')
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return source, payload


def initialize_legacy_source(checkpoint, config):
    """Explicit V2/88D -> V3/90D frozen initializer; NOT a qualified V3 policy."""
    from terminal_marl.algo.service_context_runtime_v4_2 import initialize_service_context_policy
    source, payload = load_legacy_source(checkpoint, config)
    policy, metadata = initialize_service_context_policy(source,
        source_checkpoint_metadata=payload['metadata'])
    policy.model.eval()
    for parameter in policy.model.parameters():
        parameter.requires_grad_(False)
    metadata.update(profile=PROFILE, source_trajectory='executable_trajectory_v2',
                    runtime_trajectory=config.trajectory_schema_version,
                    qualification='NOT_QUALIFIED', PPO_ALLOWED=False)
    return policy, metadata
