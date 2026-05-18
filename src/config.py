from dataclasses import dataclass, field


@dataclass
class RingConfig:
    r_min: float = 500.0
    r_max: float = 5000.0
    h_min: float = 50.0
    h_max: float = 1000.0
    v_min: float = 5.0
    v_max: float = 30.0
    # Default initial values (outer -> inner)
    default_radii: tuple[float, float, float] = (4000.0, 2500.0, 1000.0)
    default_heights: tuple[float, float, float] = (800.0, 500.0, 200.0)
    default_speeds: tuple[float, float, float] = (25.0, 20.0, 15.0)


@dataclass
class UAVConfig:
    max_speed: float = 40.0
    descent_speed: float = 25.0
    max_battery: float = 100.0
    battery_critical: float = 0.15
    safe_separation: float = 10.0          # collision danger zone (m)
    near_miss_threshold: float = 50.0      # proximity warning zone (m)
    # Energy consumption rates (per second)
    power_cruise: float = 1.0
    power_descend: float = 0.7
    # Ring-wait constraint
    min_ring_wait: float = 30.0   # minimum seconds to circle on a ring before descending
    # Ring capacity: max UAVs per ring based on ideal spacing
    min_ring_spacing: float = 100.0  # minimum angular arc distance between UAVs on same ring (m)
    # Proximity penalty (soft separation shaping)
    proximity_penalty_threshold: float = 30.0  # distance below which penalty activates (m)
    proximity_penalty_scale: float = 5.0      # penalty = scale / (distance + epsilon)
    # Priority parameters
    priority_alpha: float = 0.5   # wait time scaling
    priority_beta: float = 1.5    # wait time exponent
    priority_gamma: float = 1.0   # battery scaling
    priority_delta: float = 2.0   # battery exponent
    priority_emergency: int = 5   # base_priority >= this gets full ring-wait exemption


@dataclass
class SafetyConfig:
    """Reflexive collision avoidance parameters for SafetyController."""
    # Rule 1: Same-ring angular separation
    angular_separation_threshold: float = 50.0
    # Rule 2: Circling yields to descending
    descend_crossing_lookahead: float = 10.0
    descend_crossing_slowdown: float = 0.85
    # Rule 3: Dual descent divergence
    dual_descend_threshold: float = 50.0
    dual_descend_psi_offset: float = 15.0  # degrees, converted to rad internally
    # Rule 4: Universal proximity brake
    universal_brake_threshold: float = 30.0
    universal_brake_low: float = 0.70
    universal_brake_high: float = 0.85
    # Rule 5: Emergency override
    emergency_yield_slowdown: float = 0.60
    emergency_yield_psi: float = 25.0  # degrees, converted to rad internally


@dataclass
class SimConfig:
    dt_high: float = 1800.0       # high-level interval (s)
    dt_low: float = 2.0           # low-level step (s)
    max_steps: int = 7200         # max env steps per episode (4h @ dt=2s)
    num_landing_zones: int = 6
    entry_angles: tuple[float, float, float, float] = (0.0, 90.0, 180.0, 270.0)


# Mission types → landing zone occupation time after landing
MISSION_TYPE_LZ_OCCUPANCY: dict[int, float] = {
    1: 30.0,    # standard delivery: quick drop-off
    2: 120.0,   # cargo / heavy: needs unloading equipment
    3: 10.0,    # emergency / medical: priority, cleared immediately
    4: 60.0,    # surveillance / data collection: data upload time
}
MISSION_TYPE_NAMES: dict[int, str] = {
    1: "standard",
    2: "cargo",
    3: "emergency",
    4: "surveillance",
}
MISSION_TYPE_WEIGHTS = [0.45, 0.25, 0.10, 0.20]  # spawn probability weights


@dataclass
class EnvConfig:
    ring: RingConfig = field(default_factory=RingConfig)
    uav: UAVConfig = field(default_factory=UAVConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    num_uavs: int = 50
    # Inflow model
    uav_inflow_rate: float = 0.0      # UAVs/hour, 0=no inflow
    max_uavs: int = 100               # max concurrent active UAVs
    # Per-UAV observation
    k_nearest: int = 8                # number of nearest neighbors in local obs
    # Reward weights
    w_energy: float = 0.3
    w_wait: float = 1.0
    w_throughput: float = 2.0
    w_safety: float = 5.0
    w_priority: float = 2.0
    w_battery: float = 10.0
    w_smooth: float = 0.1             # action smoothness penalty
    penalty_conflict: float = 500.0        # collision instant penalty (with termination)
    penalty_near_miss: float = 2.0         # proximity warning penalty
    # Safety controller
    safety_enabled: bool = True            # True = SafetyController active during env.step
