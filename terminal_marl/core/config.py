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
    action_schema_version: str = "low_intent_v1_3d"
    high_action_schema_version: str = "high_airspace_v1_10d"
    observation_schema_version: str = "phaseA_obs_v1"
    experiment_mode: str = "fixed_rule"
    high_level_enabled: bool = False
    reset_mode: str = "flow_from_empty"
    warm_start_uavs: int = 0
    dt: float = 1.0
    max_simulation_seconds: float = 3600.0
    max_active_uavs: int = 60
    safe_distance: float = 80.0
    nmac_distance: float = 40.0
    collision_distance: float = 20.0
    max_speed: float = 36.0
    max_acceleration: float = 4.0
    max_turn_rate_deg_s: float = 12.0
    max_climb_rate: float = 5.0
    max_descent_rate: float = 7.0
    rings: Mapping[RingRole, RingSpec] = field(default_factory=lambda: {
        RingRole.OUTER: RingSpec(RingRole.OUTER, 600.0, 500.0, 28.0),
        RingRole.MIDDLE: RingSpec(RingRole.MIDDLE, 400.0, 350.0, 24.0),
        RingRole.INNER: RingSpec(RingRole.INNER, 200.0, 200.0, 20.0),
    })
    radius_min: float = 120.0
    radius_max: float = 900.0
    height_min: float = 120.0
    height_max: float = 650.0
    min_ring_gap: float = 120.0
    min_height_gap: float = 80.0
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
    gate_height: float = 600.0
    gate_min_time_gap_seconds: float = 20.0
    arrival_model: str = "finite_cohort"
    arrival_rate_per_second: float = 0.03
    fixed_arrival_schedule: tuple[float, ...] = ()
    total_requested_uavs: int | None = None
    max_external_wait_seconds: float | None = None
    energy_capacity: float = 10000.0
    initial_soc_mode: str = "uniform"
    initial_soc: float | None = None
    soc_warning: float = 0.30
    soc_reserve: float = 0.15
    soc_emergency: float = 0.08
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
    # A6.12 uses only the already-canonical low-level residual authority.  It
    # changes how a rule intent is searched, not the learned action schema.
    repair_aware_scheduler: bool = False
    prospective_repair_delays_seconds: tuple[float, ...] = (5.0, 10.0, 20.0, 40.0)
    release_threshold: float = 0.0
    low_level_touchdown_reward: float = 25.0
    low_level_time_penalty_per_second: float = 0.01
    low_level_progress_reward_per_meter: float = 0.001
    low_level_release_approved_reward: float = 0.10
    low_level_release_rejected_penalty: float = 0.05
    low_level_collision_penalty: float = 25.0
    low_level_energy_failure_penalty: float = 20.0

    def __post_init__(self) -> None:
        if self.schema_version != "phaseA_landing_v1":
            raise ValueError("only phaseA_landing_v1 is supported")
        if tuple(self.rings) != (RingRole.OUTER, RingRole.MIDDLE, RingRole.INNER):
            raise ValueError("rings must be OUTER, MIDDLE, INNER in canonical order")
        if self.gate_count != 4 or len(self.gate_angles_deg) != 4:
            raise ValueError("Phase A requires exactly four entry gates")
        if self.num_landing_zones != 6:
            raise ValueError("Phase A v1 requires six landing zones")
        if self.dt <= 0 or self.max_active_uavs <= 0 or self.energy_capacity <= 0:
            raise ValueError("dt, max_active_uavs and energy_capacity must be positive")
        if self.initial_soc_mode not in {"uniform", "fixed"}:
            raise ValueError("initial_soc_mode must be uniform or fixed")
        if self.initial_soc_mode == "fixed" and self.initial_soc is None:
            raise ValueError("fixed initial SOC requires initial_soc")
        supported_modes = {
            "fixed_rule", "fixed_low_rl", "dynamic_rule",
            "learned_high_rule_low", "learned_high_frozen_low",
        }
        if self.experiment_mode not in supported_modes:
            raise ValueError(f"unsupported canonical Phase-A experiment mode: {self.experiment_mode}")
        dynamic_modes = {"dynamic_rule", "learned_high_rule_low", "learned_high_frozen_low"}
        if self.high_level_enabled != (self.experiment_mode in dynamic_modes):
            raise ValueError("high_level_enabled must agree with the selected canonical experiment mode")
        if self.reset_mode not in {"flow_from_empty", "warm_start_rings"}:
            raise ValueError("unsupported reset_mode")
