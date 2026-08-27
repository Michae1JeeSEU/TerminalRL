"""Frozen research vehicle profiles for the performance-constrained executor.

The main simulation deliberately uses one profile only.  Hardware calibration
belongs to a later Crazyflie validation branch and must never overwrite this
EH216-S research profile.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VehiclePerformanceProfile:
    profile_id: str
    vehicle_model_name: str
    profile_version: str
    parameter_source: str
    mtow_kg: float | None
    max_operating_altitude_agl_m: float | None
    max_operating_altitude_msl_m: float | None
    reference_range_m: float | None
    reference_max_flight_time_s: float | None
    min_landing_soc: float | None
    max_horizontal_speed: float
    never_exceed_speed: float | None
    design_speed: float | None
    min_horizontal_speed: float
    max_horizontal_accel: float
    max_horizontal_decel: float
    max_climb_rate: float
    max_descent_rate: float
    max_vertical_accel_up: float
    max_vertical_accel_down: float
    max_lateral_accel: float
    max_lateral_accel_hard: float | None
    max_yaw_rate_deg_s: float
    max_jerk_xy: float | None = None
    max_jerk_z: float | None = None
    max_yaw_accel_deg_s2: float | None = None
    max_touchdown_horizontal_speed: float = 1.0
    max_touchdown_vertical_speed: float = 1.0


RESEARCH_EVTOL = VehiclePerformanceProfile(
    profile_id="research_evtol", vehicle_model_name="EHang EH216-S",
    profile_version="eh216s_research_v1",
    parameter_source="official_public_values_plus_frozen_engineering_execution_values",
    mtow_kg=620.0, max_operating_altitude_agl_m=120.0,
    max_operating_altitude_msl_m=1000.0, reference_range_m=30_000.0,
    reference_max_flight_time_s=1_500.0, min_landing_soc=0.25,
    max_horizontal_speed=25.0, never_exceed_speed=29.17, design_speed=36.11,
    min_horizontal_speed=8.0, max_horizontal_accel=1.5,
    max_horizontal_decel=1.5, max_climb_rate=4.0, max_descent_rate=6.0,
    max_vertical_accel_up=1.0, max_vertical_accel_down=1.0,
    max_lateral_accel=1.5, max_lateral_accel_hard=1.73,
    max_yaw_rate_deg_s=12.0,
)


def get_vehicle_profile(profile_id: str) -> VehiclePerformanceProfile:
    if profile_id != RESEARCH_EVTOL.profile_id:
        raise ValueError(f"unsupported vehicle profile {profile_id!r}; only research_evtol is valid for main simulation")
    return RESEARCH_EVTOL


@dataclass(frozen=True)
class ScaleConfig:
    """Future hardware scale mapping; intentionally contains no Crazyflie data."""
    length_scale: float
    time_scale: float

    def __post_init__(self) -> None:
        if self.length_scale <= 0.0 or self.time_scale <= 0.0:
            raise ValueError("ScaleConfig values must be positive")
