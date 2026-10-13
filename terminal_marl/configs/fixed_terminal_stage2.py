"""Stage2 computational protocol. Stage1 physical/scenario defaults unchanged."""
from dataclasses import dataclass

METHODS = ("fcfs", "soc", "energy_slack", "deadline_slack", "joint_window", "rolling_horizon")


@dataclass(frozen=True)
class SchedulerConfig:
    decision_interval: float = 5.
    normal_search_limit: int = 256
    priority_scan_limit: int = 256
    joint_aircraft_limit: int = 6
    catalogue_scan_limit: int = 24
    candidates_per_aircraft: int = 6
    beam_width: int = 64
    rolling_seconds: float = 20.
    rolling_root_alternatives: int = 4
    oracle_candidates_per_aircraft: int = 4
    oracle_seconds: float = 30.
    oracle_iterations: int = 1000

    def __post_init__(self):
        if any(value <= 0 for value in self.__dict__.values()):
            raise ValueError("scheduler budgets must be positive")


DEFAULT_SCHEDULER_CONFIG = SchedulerConfig()
