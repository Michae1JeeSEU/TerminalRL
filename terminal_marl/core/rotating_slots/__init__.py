"""FT-RS-S1-1.0 rotating slots. No legacy fixed-terminal dependencies."""
from .config import Config, load_config
from .models import KinematicState, CandidateResult, ValidationResult
from .geometry import slot_state, slot_phase, pad_position, ring_angular_speed
from .trajectory import TrajectoryPlan, generate_inbound, generate_transfer, generate_final, make_stable_plan, make_ground_plan
from .energy import evaluate_power, integrate_energy, EnergyProfile, VehicleProfile
from .safety import validate_plan, validate_separation, certified_minimum_distance
