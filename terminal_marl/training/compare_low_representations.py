"""Pre-registered representation comparison manifest; no training execution."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RepresentationCondition:
    name: str
    observation: str
    memory: str
    optimizer: str
    teacher_in_evaluation: bool


REPRESENTATION_PROTOCOL = (
    RepresentationCondition("legacy_a5", "phaseA_low_obs_v1_41d", "none", "ppo", False),
    RepresentationCondition("topology_mlp", "phaseA_low_obs_v2_topology_87d", "none", "ppo", False),
    RepresentationCondition("topology_gru", "phaseA_low_obs_v2_topology_87d", "gru", "ppo", False),
    RepresentationCondition("all_neighbor_recurrent", "all_neighbor_variable_set", "recurrent", "ppo", False),
)
