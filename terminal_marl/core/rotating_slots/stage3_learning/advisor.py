"""Ranking is a computation preference, never an execution authorization."""
from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class RankResult:
    ordered_ids: tuple[str, ...]
    model_metadata: Mapping = field(default_factory=dict)
    scores: Mapping = field(default_factory=dict)


class HeuristicAdvisor:
    def rank(self, features, budget):
        return RankResult(tuple(row["descriptor_id"] for row in features), {"model_id": "NO_MODEL"})
