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
    safe_separation: float = 50.0
    # Energy consumption rates (per second)
    power_hover: float = 0.5
    power_cruise: float = 1.0
    power_descend: float = 0.7
    # Priority parameters
    priority_alpha: float = 0.5   # wait time scaling
    priority_beta: float = 1.5    # wait time exponent
    priority_gamma: float = 1.0   # battery scaling
    priority_delta: float = 2.0   # battery exponent


@dataclass
class SimConfig:
    dt_high: float = 1800.0   # high-level interval (s)
    dt_low: float = 2.0       # low-level step (s)
    max_steps: int = 3600     # max env steps per episode
    num_landing_zones: int = 6
    num_waypoints: int = 4    # waypoints per descent path


@dataclass
class EnvConfig:
    ring: RingConfig = field(default_factory=RingConfig)
    uav: UAVConfig = field(default_factory=UAVConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    num_uavs: int = 50
    # Reward weights
    w_energy: float = 0.3
    w_wait: float = 1.0
    w_throughput: float = 2.0
    w_safety: float = 5.0
    w_priority: float = 2.0
    w_battery: float = 10.0
    penalty_conflict: float = 10.0
