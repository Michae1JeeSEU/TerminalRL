"""Opt-in Stage3 interfaces. Importing these interfaces never imports torch."""
from .schemas import WorkDescriptor
from .advisor import RankResult
from .runtime import AdvisorRuntime, scale_candidate_budget

__all__ = ["WorkDescriptor", "RankResult", "AdvisorRuntime", "scale_candidate_budget"]
