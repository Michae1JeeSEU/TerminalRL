"""Canonical Phase-A landing-only configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Mapping


class RingRole(str, Enum):
    OUTER = "outer"
    MIDDLE = "middle"
    INNER = "inner"


@dataclass(frozen=True)
class RingSpec:
    role: RingRole
    radius: float
    height: float
    nominal_speed: float


@dataclass
class SimConfig:
    """The only supported simulation schema: ``phaseA_landing_v1``."""

    schema_version: str = "phaseA_landing_v1"
    airspace_schema_version: str = "three_ring_v1"
    vehicle_dynamics_schema_version: str = "performance_constrained_v2"
    trajectory_schema_version: str = "executable_trajectory_v2"
    vehicle_profile_id: str = "research_evtol"
    vehicle_execution_mode: str = "performance_constrained"
    # V4 policy requests one joint categorical scheduling intention.  The
    # decoder alone maps it to the unchanged normalized execution intent.
    action_schema_version: str = "low_action_schema_v4_joint_discrete"
    # V4.2-only execution choice for a rejected RELEASE that remains a ring
    # holder.  Keep the historical requested-intent behavior by default;
    # canonical WAIT is opt-in for new diagnostics and training.
    v4_2_rejected_release_hold_mode: str = "legacy_requested"
    high_action_schema_version: str = "high_airspace_v1_10d"
    observation_schema_version: str = "phaseA_low_obs_v2_topology_87d"
    experiment_mode: str = "fixed_rule"
    high_level_enabled: bool = False
    reset_mode: str = "flow_from_empty"
    warm_start_uavs: int = 0
    # A populated live/debug scene may represent aircraft already established
    # in steady ring flight.  Curriculum/test warm starts retain their legacy
    # zero-speed initialization unless this is explicitly enabled.
    warm_start_steady_motion: bool = False
    # Optional reset-time precondition for phase/motion emergency reachability.
    # The historical warm-start path remains unchanged when disabled.
    warm_start_energy_reachability_check_enabled: bool = False
    # Evaluation-only opt-in for a randomized, separation-safe warm-start
    # layout.  The historical deterministic layout remains the default.
    warm_start_layout_randomized: bool = False
    # ``dt`` remains the policy/environment decision period for compatibility.
    # The physics executor samples the same authoritative plan at a smaller,
    # fixed substep in the performance-constrained mode.
    dt: float = 1.0
    physics_dt: float = 0.2
    safety_sampling_dt: float = 1.0
    trajectory_bezier_samples: int = 41
    trajectory_final_samples: int = 51
    # The default preserves the established braking cubic final-approach
    # geometry.  The continuous inward corridor is an opt-in diagnostic mode
    # for dense inner-ring flow; it reuses the certified emergency corridor
    # geometry while retaining normal final-approach semantics.
    final_approach_geometry_mode: str = "brake_bezier_v1"
    # Opt-in diagnostic repair for ring transfers whose generated endpoint is
    # later than the initially requested merge slot.  Historical runs retain
    # the original fail-closed behavior when disabled.
    reconcile_late_merge_slots_enabled: bool = False
    max_simulation_seconds: float = 3600.0
    max_active_uavs: int = 60
    # Source-bound V2.3 uses an explicit finite communication neighborhood for
    # the five semantic topology slots.  ``None`` preserves legacy V2/V2.1
    # behavior; V2.3 validates that a finite positive range is supplied.
    neighbor_comm_range_m: float | None = None
    safe_distance: float = 80.0
    nmac_distance: float = 40.0
    collision_distance: float = 10.0
    rings: Mapping[RingRole, RingSpec] = field(default_factory=lambda: {
        RingRole.OUTER: RingSpec(RingRole.OUTER, 600.0, 100.0, 22.0),
        RingRole.MIDDLE: RingSpec(RingRole.MIDDLE, 400.0, 70.0, 20.0),
        RingRole.INNER: RingSpec(RingRole.INNER, 200.0, 40.0, 16.0),
    })
    radius_min: float = 120.0
    radius_max: float = 900.0
    height_min: float = 0.0
    height_max: float = 120.0
    min_ring_gap: float = 120.0
    # The operational rings are 100/70/40 m AGL; their 30 m spacing is the
    # canonical structural minimum for this terminal layout.
    min_height_gap: float = 20.0
    max_radius_change_per_update: float = 20.0
    max_height_change_per_update: float = 20.0
    max_speed_profile_change_per_update: float = 2.0
    speed_min: float = 8.0
    speed_max: float = 36.0
    headway_o2m: float = 9.0
    headway_m2i: float = 9.0
    headway_i2f: float = 12.0
    headway_min: float = 3.0
    headway_max: float = 30.0
    max_headway_change_per_update: float = 1.0
    # V2.1 normalizes the remaining retry mask against the maximum headway in
    # the frozen Phase-A design range.  A rejected release waits for
    # ``max(5 s, interface_headway)`` and DynamicAirspace clamps every
    # interface headway at ``headway_max``.  The formal V4.2 V2.1 config
    # freezes this to 30 s; it is not a learned or tuned parameter.
    retry_norm_horizon_seconds: float = 30.0
    metrics_sampling_interval_seconds: float = 20.0
    high_level_decision_interval_seconds: float = 180.0
    geometry_min_dwell_seconds: float = 600.0
    geometry_transition_seconds: float = 20.0
    gate_count: int = 4
    gate_angles_deg: tuple[float, ...] = (0.0, 90.0, 180.0, 270.0)
    approach_threshold_radius: float = 1000.0
    spawn_radius: float = 1300.0
    gate_height: float = 110.0
    gate_min_time_gap_seconds: float = 20.0
    arrival_model: str = "finite_cohort"
    arrival_rate_per_second: float = 0.03
    fixed_arrival_schedule: tuple[float, ...] = ()
    total_requested_uavs: int | None = None
    # Arrival attributes are part of the scenario protocol.  Keeping them in
    # the immutable simulation config makes C0/L0 schedules replayable rather
    # than relying on hidden random-generator defaults.
    gate_probabilities: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)
    mission_probabilities: tuple[float, ...] = (0.45, 0.25, 0.10, 0.20)
    max_external_wait_seconds: float | None = None
    # The normalized energy budget is calibrated to the EH216-S reference
    # maximum flight time (one unit per second at reference power).  Scenario
    # files may still override this explicitly for controlled diagnostics.
    energy_capacity: float = 1500.0
    initial_soc_mode: str = "uniform"
    initial_soc: float | None = None
    initial_soc_min: float | None = None
    initial_soc_max: float | None = None
    soc_warning: float = 0.35
    soc_reserve: float = 0.25
    soc_emergency: float = 0.20
    # Optional P3 urgency shaping.  Zero weights preserve the historical
    # low-level reward values byte-for-byte; V3 protocols opt in explicitly.
    low_level_priority_time_weight: float = 0.0
    low_level_soc_urgency_time_weight: float = 0.0
    low_level_soc_urgency_threshold: float = 0.65
    power_reference: float = 1.0
    power_external_wait: float = 1.0
    power_inbound: float = 1.0
    power_ring_hold: float = 1.0
    power_ring_transfer: float = 1.10
    power_final: float = 0.90
    power_landing: float = 1.05
    # Opt-in phase/motion surrogate. Historical qualification keeps the
    # original normalized power coefficients and episode semantics.
    energy_model_mode: str = "legacy_phase"
    emergency_landing_enabled: bool = False
    # Opt-in V4.2 execution guard that preserves free landing capacity for
    # aircraft approaching the emergency SOC threshold.  Historical
    # configurations retain their runtime behavior when this is disabled.
    emergency_priority_guard_enabled: bool = False
    # Diagnostic-only route-aware candidate energy guard.  This is layered on
    # the existing priority guard and remains disabled for historical runs.
    energy_reachability_guard_enabled: bool = False
    # Opt-in fixed-rule finite-candidate landing-flow coordinator.  The
    # historical V4.2 release and emergency paths remain byte-for-byte
    # selected when this flag is false.
    rule_flow_coordination_enabled: bool = False
    emergency_trigger_soc: float = 0.35
    emergency_energy_margin_seconds: float = 15.0
    phase_motion_nominal_solo_energy: float = 240.9
    low_level_emergency_landing_penalty: float = 8.0
    priority_alpha: float = 0.5
    priority_beta: float = 1.5
    priority_gamma: float = 1.0
    priority_delta: float = 2.0
    priority_wait_reference_seconds: float = 3600.0
    # Scenario-owned arrival priority protocol.  The default retains the
    # historical 1/3/5 distribution; L1-S explicitly freezes it to a
    # homogeneous baseline so a safety witness cannot be confounded by
    # random scheduling priority.
    priority_mode: str = "random_135"
    initial_priority: int | None = None
    # Opt-in task-neutral curriculum switch.  Keep the legacy priority-aware
    # behavior as the default; when disabled, task priority is neutralized in
    # scheduling/reward/observation channels while mission metadata remains
    # available to physical service and compatibility logic.
    task_priority_enabled: bool = True
    num_landing_zones: int = 6
    final_queue_capacity: int = 1
    # Historical protocols retain the lowest-ID/single-pass behavior.  Dense
    # throughput curricula opt into geometry-aware LZ selection and a
    # downstream-first arbitration order without rewriting old evidence.
    lz_assignment_mode: str = "lowest_id"
    release_arbitration_mode: str = "legacy_insertion"
    max_speed_residual_mps: float = 2.0
    max_merge_time_residual_seconds: float = 6.0
    # Deployment interprets one gate decision.  For RELEASE, the default
    # remains one requested proposal; the explicitly qualified bounded mode
    # may certify at most the first three proposals ranked by the same policy
    # forward pass.  This is runtime safety selection, never Teacher repair.
    runtime_proposal_shortlist_k: int = 1
    scheduler_mode: str = "execution"
    teacher_enabled: bool = False
    teacher_mode: str = "teacher_full"
    teacher_release_label_enabled: bool = False
    # Deprecated compatibility knob for archived A6 audits.  New code must
    # select an explicit scheduler mode instead.
    repair_aware_scheduler: bool = False
    prospective_repair_delays_seconds: tuple[float, ...] = (5.0, 10.0, 20.0, 40.0)
    shield_horizon_seconds: float = 30.0
    shield_enabled: bool = True
    # Broad phase may only discard pairs with a geometry-proof clearance;
    # this margin is never used by the narrow-phase hard safety decision.
    shield_broad_phase_margin: float = 20.0
    priority_feature_scale: float = 5.0
    teacher_intervention_penalty: float = 0.10
    teacher_correction_magnitude_penalty: float = 0.05
    shield_intervention_penalty: float = 0.20
    shield_correction_magnitude_penalty: float = 0.10
    shield_id_interventions_per_100_intents: float = 10.0
    shield_ood_interventions_per_100_intents: float = 20.0
    release_threshold: float = 0.0
    low_level_touchdown_reward: float = 25.0
    low_level_time_penalty_per_second: float = 0.01
    low_level_progress_reward_per_meter: float = 0.001
    low_level_release_approved_reward: float = 0.10
    low_level_release_rejected_penalty: float = 0.05
    low_level_collision_penalty: float = 25.0
    low_level_energy_failure_penalty: float = 20.0

    @property
    def vehicle_profile(self):
        """The sole source of vehicle performance limits in main simulation."""
        from .vehicle_profiles import get_vehicle_profile
        return get_vehicle_profile(self.vehicle_profile_id)

    @property
    def min_landing_soc(self) -> float:
        """Minimum SOC for a successful EH216-S touchdown.

        This threshold is owned by the selected vehicle profile.  The
        historical ``soc_reserve`` field remains the queue/admission reserve
        and is intentionally kept as a separate compatibility setting.
        """
        profile_floor = self.vehicle_profile.min_landing_soc
        return self.soc_reserve if profile_floor is None else float(profile_floor)

    # Historical names are compatibility views, never independent fields.
    @property
    def max_speed(self) -> float: return self.vehicle_profile.max_horizontal_speed
    @property
    def max_acceleration(self) -> float: return self.vehicle_profile.max_horizontal_accel
    @property
    def max_turn_rate_deg_s(self) -> float: return self.vehicle_profile.max_yaw_rate_deg_s
    @property
    def max_climb_rate(self) -> float: return self.vehicle_profile.max_climb_rate
    @property
    def max_descent_rate(self) -> float: return self.vehicle_profile.max_descent_rate

    def __post_init__(self) -> None:
        if not isinstance(self.task_priority_enabled, bool):
            raise ValueError("task_priority_enabled must be bool")
        if self.energy_model_mode not in {"legacy_phase", "phase_motion_v2"}:
            raise ValueError("unsupported energy_model_mode")
        if self.final_approach_geometry_mode not in {"brake_bezier_v1", "continuous_inward_v1"}:
            raise ValueError("unsupported final_approach_geometry_mode")
        if self.energy_reachability_guard_enabled and (
            not self.emergency_priority_guard_enabled
            or not self.emergency_landing_enabled
            or self.energy_model_mode != "phase_motion_v2"
            or self.action_schema_version != "low_action_schema_v4_2_hierarchical"
        ):
            raise ValueError(
                "energy_reachability_guard_enabled requires "
                "emergency_priority_guard_enabled=True, emergency_landing_enabled=True, "
                "energy_model_mode='phase_motion_v2', and "
                "action_schema_version='low_action_schema_v4_2_hierarchical'"
            )
        if self.emergency_priority_guard_enabled and (
            not self.emergency_landing_enabled
            or self.energy_model_mode != "phase_motion_v2"
            or self.action_schema_version != "low_action_schema_v4_2_hierarchical"
        ):
            raise ValueError(
                "emergency_priority_guard_enabled requires emergency_landing_enabled, "
                "phase_motion_v2, and low_action_schema_v4_2_hierarchical"
            )
        if not (self.min_landing_soc < self.emergency_trigger_soc <= 1.0):
            raise ValueError("emergency_trigger_soc must exceed the landing SOC floor")
        if (self.emergency_energy_margin_seconds < 0 or self.phase_motion_nominal_solo_energy <= 0
                or self.low_level_emergency_landing_penalty < 0):
            raise ValueError("emergency energy margin and landing penalty must be non-negative")
        if self.schema_version != "phaseA_landing_v1":
            raise ValueError("only phaseA_landing_v1 is supported")
        if self.vehicle_dynamics_schema_version != "performance_constrained_v2" or self.trajectory_schema_version != "executable_trajectory_v2":
            raise ValueError("only the performance-constrained dynamics-v2 schemas are supported")
        if self.vehicle_execution_mode not in {"performance_constrained", "legacy_kinematic"}:
            raise ValueError("unsupported vehicle_execution_mode")
        profile = self.vehicle_profile
        if tuple(self.rings) != (RingRole.OUTER, RingRole.MIDDLE, RingRole.INNER):
            raise ValueError("rings must be OUTER, MIDDLE, INNER in canonical order")
        if self.gate_count != 4 or len(self.gate_angles_deg) != 4:
            raise ValueError("Phase A requires exactly four entry gates")
        if len(self.gate_probabilities) != self.gate_count or any(value < 0.0 for value in self.gate_probabilities) or abs(sum(self.gate_probabilities) - 1.0) > 1e-8:
            raise ValueError("gate_probabilities must be a non-negative distribution over the four gates")
        if len(self.mission_probabilities) != 4 or any(value < 0.0 for value in self.mission_probabilities) or abs(sum(self.mission_probabilities) - 1.0) > 1e-8:
            raise ValueError("mission_probabilities must be a non-negative distribution over the four mission types")
        if self.num_landing_zones != 6:
            raise ValueError("Phase A v1 requires six landing zones")
        if self.lz_assignment_mode not in {"lowest_id", "nearest_available"}:
            raise ValueError("unsupported landing-zone assignment mode")
        if self.release_arbitration_mode not in {
            "legacy_insertion",
            "downstream_priority",
            "downstream_soc_priority",
        }:
            raise ValueError("unsupported release arbitration mode")
        if not 0 <= self.final_queue_capacity <= self.num_landing_zones:
            raise ValueError("final queue capacity must fit the landing-zone set")
        if self.dt <= 0 or self.physics_dt <= 0 or self.safety_sampling_dt <= 0 or self.max_active_uavs <= 0 or self.energy_capacity <= 0:
            raise ValueError("dt, physics_dt, safety_sampling_dt, max_active_uavs and energy_capacity must be positive")
        if not (0.0 < self.collision_distance < self.nmac_distance < self.safe_distance):
            raise ValueError("separation thresholds must satisfy 0 < collision < protected floor < desired distance")
        if self.physics_dt > self.dt or self.safety_sampling_dt > self.dt:
            raise ValueError("physics/safety sampling dt cannot exceed policy dt")
        if self.trajectory_bezier_samples < 3 or self.trajectory_final_samples < 3:
            raise ValueError("trajectory geometric sampling requires at least three points")
        if self.max_speed != profile.max_horizontal_speed:
            raise RuntimeError("vehicle performance must come from the selected profile")
        if self.gate_height > profile.max_operating_altitude_agl_m:
            raise ValueError("entry gate exceeds the selected vehicle AGL operating envelope")
        for ring in self.rings.values():
            if ring.height < self.height_min or ring.height > min(self.height_max, profile.max_operating_altitude_agl_m):
                raise ValueError(f"ring {ring.role.value} height violates the vehicle AGL envelope")
            if ring.nominal_speed > profile.max_horizontal_speed + 1e-9 or ring.nominal_speed ** 2 / ring.radius > profile.max_lateral_accel + 1e-9:
                raise ValueError(f"ring {ring.role.value} nominal speed violates the research eVTOL dynamics envelope")
        if self.initial_soc_mode not in {"uniform", "fixed"}:
            raise ValueError("initial_soc_mode must be uniform or fixed")
        if self.initial_soc_mode == "fixed" and self.initial_soc is None:
            raise ValueError("fixed initial SOC requires initial_soc")
        if self.initial_soc is not None and not 0.0 < self.initial_soc <= 1.0:
            raise ValueError("initial_soc must be in (0, 1]")
        if (self.initial_soc_min is None) != (self.initial_soc_max is None):
            raise ValueError("initial_soc_min and initial_soc_max must be set together")
        if self.initial_soc_min is not None and not (0.0 < self.initial_soc_min <= self.initial_soc_max <= 1.0):
            raise ValueError("initial SOC bounds must satisfy 0 < min <= max <= 1")
        for name in ("low_level_priority_time_weight", "low_level_soc_urgency_time_weight"):
            value = getattr(self, name)
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be finite and non-negative") from exc
            if isinstance(value, bool) or not math.isfinite(numeric_value) or numeric_value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        threshold = self.low_level_soc_urgency_threshold
        try:
            numeric_threshold = float(threshold)
            numeric_reserve = float(self.soc_reserve)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "low_level_soc_urgency_threshold must be finite and satisfy "
                "soc_reserve < threshold <= 1"
            ) from exc
        if (
            isinstance(threshold, bool)
            or not math.isfinite(numeric_threshold)
            or not math.isfinite(numeric_reserve)
            or not (numeric_reserve < numeric_threshold <= 1.0)
        ):
            raise ValueError(
                "low_level_soc_urgency_threshold must be finite and satisfy "
                "soc_reserve < threshold <= 1"
            )
        if self.priority_mode not in {"random_135", "fixed"}:
            raise ValueError("priority_mode must be random_135 or fixed")
        if self.priority_mode == "fixed" and self.initial_priority not in {1, 3, 5}:
            raise ValueError("fixed priority_mode requires initial_priority in {1, 3, 5}")
        if self.priority_mode == "random_135" and self.initial_priority is not None:
            raise ValueError("random_135 priority_mode cannot set initial_priority")
        supported_modes = {
            "fixed_rule", "fixed_teacher_heavy", "fixed_a5_legacy", "fixed_low_student",
            "dynamic_rule", "dynamic_hrl_student",
            # Explicit legacy aliases are kept only so archived tests/results
            # stay replayable.  They are not formal proposal conditions.
            "fixed_low_rl", "learned_high_rule_low", "learned_high_frozen_low",
        }
        if self.experiment_mode not in supported_modes:
            raise ValueError(f"unsupported canonical Phase-A experiment mode: {self.experiment_mode}")
        dynamic_modes = {"dynamic_rule", "dynamic_hrl_student", "learned_high_rule_low", "learned_high_frozen_low"}
        if self.high_level_enabled != (self.experiment_mode in dynamic_modes):
            raise ValueError("high_level_enabled must agree with the selected canonical experiment mode")
        if self.reset_mode not in {"flow_from_empty", "warm_start_rings"}:
            raise ValueError("unsupported reset_mode")
        if not isinstance(self.warm_start_layout_randomized, bool):
            raise ValueError("warm_start_layout_randomized must be bool")
        if self.warm_start_energy_reachability_check_enabled and (
            self.reset_mode != "warm_start_rings"
            or self.energy_model_mode != "phase_motion_v2"
            or not self.emergency_landing_enabled
        ):
            raise ValueError(
                "warm_start_energy_reachability_check_enabled requires "
                "reset_mode='warm_start_rings', energy_model_mode='phase_motion_v2', "
                "and emergency_landing_enabled=True"
            )
        if self.reset_mode == "warm_start_rings" and self.warm_start_uavs > 0 and len(self.fixed_arrival_schedule) < self.warm_start_uavs:
            raise ValueError("safe warm-start requires an explicit fixed_arrival_schedule for every initial UAV")
        if self.observation_schema_version not in {"phaseA_low_obs_v1_41d", "phaseA_low_obs_v2_topology_87d", "phaseA_low_obs_v2_1_retry_state", "phaseA_low_obs_v2_2_scheduler_context_94d", "phaseA_low_obs_v2_3_source_bound_88d"}:
            raise ValueError("unsupported low-level observation schema")
        if self.observation_schema_version == "phaseA_low_obs_v2_3_source_bound_88d":
            if (self.neighbor_comm_range_m is None or isinstance(self.neighbor_comm_range_m, bool)
                    or not math.isfinite(float(self.neighbor_comm_range_m))
                    or float(self.neighbor_comm_range_m) <= 0.0):
                raise ValueError("V2.3 requires a finite positive neighbor_comm_range_m")
        if self.retry_norm_horizon_seconds <= 0.0:
            raise ValueError("retry_norm_horizon_seconds must be positive")
        if self.observation_schema_version in {"phaseA_low_obs_v2_1_retry_state", "phaseA_low_obs_v2_2_scheduler_context_94d", "phaseA_low_obs_v2_3_source_bound_88d"} and self.retry_norm_horizon_seconds < max(5.0, self.headway_max):
            raise ValueError("V2.1+ retry normalization horizon must cover the configured retry/cooldown maximum")
        if self.action_schema_version not in {"low_intent_v3_factorized_discrete", "low_action_schema_v4_joint_discrete", "low_action_schema_v4_1_joint_discrete_26", "low_action_schema_v4_2_hierarchical"}:
            raise ValueError("unsupported low-level action schema")
        if self.v4_2_rejected_release_hold_mode not in {"legacy_requested", "canonical_wait"}:
            raise ValueError("unsupported v4_2_rejected_release_hold_mode")
        if self.scheduler_mode not in {"execution", "teacher_full", "teacher_immediate", "teacher_label_only"}:
            raise ValueError("unsupported scheduler_mode")
        if self.teacher_mode not in {"teacher_full", "teacher_immediate", "teacher_label_only"}:
            raise ValueError("unsupported teacher_mode")
        if self.repair_aware_scheduler:
            # Archived A6 configurations use this bit.  Translate it once at
            # construction instead of allowing it to leak into new paths.
            self.scheduler_mode = "teacher_full"
            self.teacher_enabled = True
        if self.scheduler_mode != "execution":
            self.teacher_enabled = True
        if self.rule_flow_coordination_enabled and (
            self.experiment_mode != "fixed_rule"
            or self.high_level_enabled
            or self.teacher_enabled
            or self.energy_model_mode != "phase_motion_v2"
            or not self.energy_reachability_guard_enabled
        ):
            raise ValueError(
                "rule_flow_coordination_enabled requires fixed_rule, "
                "teacher_enabled=False, energy_model_mode='phase_motion_v2', "
                "energy_reachability_guard_enabled=True, and no high-level dynamic mode"
            )
        if self.rule_flow_coordination_enabled:
            # Fixed-rule flow uses a derived nominal holding contract.  The
            # coordinator and runtime shield therefore share an exact zero
            # speed-residual authority; historical configurations keep their
            # configured default (2 m/s) when this opt-in flag is disabled.
            self.max_speed_residual_mps = 0.0
        if self.runtime_proposal_shortlist_k not in {1, 3}:
            raise ValueError("runtime_proposal_shortlist_k must be 1 or 3")
        if self.runtime_proposal_shortlist_k > 1:
            if self.action_schema_version != "low_action_schema_v4_2_hierarchical":
                raise ValueError("runtime proposal shortlist requires V4.2 hierarchical actions")
            if self.scheduler_mode != "execution" or self.teacher_enabled:
                raise ValueError("runtime proposal shortlist requires Teacher-free execution scheduling")
        if self.experiment_mode in {"fixed_low_student", "dynamic_hrl_student"} and self.teacher_enabled:
            raise ValueError("formal student/deployment modes require teacher_enabled=False")
        if self.experiment_mode in {"fixed_low_student", "dynamic_hrl_student"} and not self.shield_enabled:
            raise ValueError("formal student/deployment modes require shield_enabled=True")
        if self.shield_horizon_seconds <= 0.0 or self.shield_broad_phase_margin < 0.0:
            raise ValueError("shield horizon must be positive and broad-phase margin non-negative")
