"""Frozen mechanism contexts and bounded offline budgets; no learning model."""
from dataclasses import dataclass, replace
from terminal_marl.configs.fixed_terminal_stage2 import DEFAULT_SCHEDULER_CONFIG

REFERENCE_SETTINGS = replace(DEFAULT_SCHEDULER_CONFIG, joint_aircraft_limit=8,
                             catalogue_scan_limit=48, candidates_per_aircraft=8,
                             beam_width=128)
ROOT_SEARCH_KEEP = 12
REMOVE_ONE_LIMIT = 3
OBSERVED_WINDOWS = (20., 60.)


@dataclass(frozen=True)
class Context:
    load: str
    soc: str
    seed: int
    time: float = 0.

    @property
    def key(self):
        return f"{self.load}-{self.soc}-{self.seed}-t{self.time:g}"


FORMAL_CONTEXTS = tuple(Context(load, soc, 101) for load in ("low", "medium", "high")
                        for soc in ("high", "medium", "low")) + (
    Context("medium", "low", 303), Context("high", "high", 202), Context("high", "high", 303),
    Context("low", "medium", 101, 60.), Context("medium", "medium", 101, 60.),
    Context("high", "medium", 101, 60.),
)
