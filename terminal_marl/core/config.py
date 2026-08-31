"""Canonical Phase-A landing-only configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
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
    action_schema_version: str = "low_intent_v1_3d"
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
    # ``dt`` remains the policy/environment decision period for compatibility.
    # The physics executor samples the same authoritative plan at a smaller,
    # fixed substep in the performance-constrained mode.
    dt: float = 1.0
    physics_dt: float = 0.2
    safety_sampling_dt: float = 1.0
    trajectory_bezier_samples: int = 41
    trajectory_final_samples: int = 51
    max_simulation_seconds: float = 3600.0
    max_active_uavs: int = 60
    safe_distance: float = 80.0
    nmac_distance: float = 40.0
    collision_distance: float = 20.0
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
    energy_capacity: float = 10000.0
    initial_soc_mode: str = "uniform"
    initial_soc: float | None = None
    initial_soc_min: float | None = None
    initial_soc_max: float | None = None
    soc_warning: float = 0.35
    soc_reserve: float = 0.25
    soc_emergency: float = 0.20
    power_reference: float = 1.0
    power_external_wait: float = 1.0
    power_inbound: float = 1.0
    power_ring_hold: float = 1.0
    power_ring_transfer: float = 1.10
    power_final: float = 0.90
    power_landing: float = 1.05
    priority_alpha: float = 0.5
    priority_beta: float = 1.5
    priority_gamma: float = 1.0
    priority_delta: float = 2.0
    priority_wait_reference_seconds: float = 3600.0
    num_landing_zones: int = 6
    final_queue_capacity: int = 1
    max_speed_residual_mps: float = 2.0
    max_merge_time_residual_seconds: float = 6.0
    # Deployment always interprets exactly one requested action.  Teacher
    # modes are training-only and never valid for formal/student evaluation.
    scheduler_mode: str = "execution"
    teacher_enabled: bool = False
    teacher_mode: str = "teacher_full"
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
        if self.dt <= 0 or self.physics_dt <= 0 or self.safety_sampling_dt <= 0 or self.max_active_uavs <= 0 or self.energy_capacity <= 0:
            raise ValueError("dt, physics_dt, safety_sampling_dt, max_active_uavs and energy_capacity must be positive")
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
        if self.reset_mode == "warm_start_rings" and self.warm_start_uavs > 0 and len(self.fixed_arrival_schedule) < self.warm_start_uavs:
            raise ValueError("safe warm-start requires an explicit fixed_arrival_schedule for every initial UAV")
        if self.observation_schema_version not in {"phaseA_low_obs_v1_41d", "phaseA_low_obs_v2_topology_87d"}:
            raise ValueError("unsupported low-level observation schema")
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
        if self.experiment_mode in {"fixed_low_student", "dynamic_hrl_student"} and self.teacher_enabled:
            raise ValueError("formal student/deployment modes require teacher_enabled=False")
        if self.experiment_mode in {"fixed_low_student", "dynamic_hrl_student"} and not self.shield_enabled:
            raise ValueError("formal student/deployment modes require shield_enabled=True")
        if self.shield_horizon_seconds <= 0.0 or self.shield_broad_phase_margin < 0.0:
            raise ValueError("shield horizon must be positive and broad-phase margin non-negative")
