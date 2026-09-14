"""V4.2 hierarchical encoding of V4.1's unchanged 26 execution intents."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .low_level_joint_action_v4_1 import (JOINT_ACTION_CATALOG_V4_1,
    N_JOINT_ACTIONS_V4_1, joint_index_from_normalized_v4_1)


ACTION_SCHEMA_V4_2 = "low_action_schema_v4_2_hierarchical"
WAIT_GATE = 0
RELEASE_GATE = 1
N_GATE_ACTIONS_V4_2 = 2
N_RELEASE_PROPOSALS_V4_2 = 25
N_EXECUTION_ACTIONS_V4_2 = N_JOINT_ACTIONS_V4_1


@dataclass(frozen=True)
class HierarchicalActionV42:
    """One requested scheduling action, before execution/safety handling."""
    gate: int
    proposal: int | None

    def __post_init__(self) -> None:
        if self.gate == WAIT_GATE and self.proposal is not None:
            raise ValueError("WAIT must not carry a policy proposal")
        if self.gate == RELEASE_GATE and (self.proposal is None or not 0 <= self.proposal < N_RELEASE_PROPOSALS_V4_2):
            raise ValueError("RELEASE requires a proposal in [0, 25)")
        if self.gate not in {WAIT_GATE, RELEASE_GATE}:
            raise ValueError("gate must be WAIT(0) or RELEASE(1)")

    @property
    def joint_index_v4_1(self) -> int:
        return 0 if self.gate == WAIT_GATE else int(self.proposal) + 1

    @property
    def normalized(self) -> np.ndarray:
        return JOINT_ACTION_CATALOG_V4_1[self.joint_index_v4_1].normalized

    def as_dict(self) -> dict[str, object]:
        decoded = JOINT_ACTION_CATALOG_V4_1[self.joint_index_v4_1]
        return {"gate": "WAIT" if self.gate == WAIT_GATE else "RELEASE", "gate_index": self.gate,
                "proposal": self.proposal, "joint_index_v4_1": self.joint_index_v4_1,
                "decoded": decoded.as_dict()}


def hierarchical_from_joint_index_v4_1(index: int) -> HierarchicalActionV42:
    if not isinstance(index, (int, np.integer)) or not 0 <= int(index) < N_EXECUTION_ACTIONS_V4_2:
        raise ValueError("V4.1 execution index must be in [0, 26)")
    return HierarchicalActionV42(WAIT_GATE, None) if int(index) == 0 else HierarchicalActionV42(RELEASE_GATE, int(index) - 1)


def hierarchical_from_normalized_v4_2(action: Any, *, atol: float = 1e-6) -> HierarchicalActionV42:
    return hierarchical_from_joint_index_v4_1(int(joint_index_from_normalized_v4_1(action, atol=atol)))


def decode_hierarchical_actions_v4_2(gates: Any, proposals: Any) -> np.ndarray:
    """Decode parallel gate/proposal values without inventing WAIT proposals."""
    gate_values = np.asarray(gates)
    proposal_values = np.asarray(proposals, dtype=object)
    if gate_values.shape != proposal_values.shape:
        raise ValueError("gate and proposal arrays must have the same shape")
    decoded: list[np.ndarray] = []
    for gate, proposal in zip(gate_values.reshape(-1), proposal_values.reshape(-1)):
        if not np.isfinite(gate) or gate != round(float(gate)):
            raise ValueError("gate values must be finite integer values")
        value = int(gate)
        if value == WAIT_GATE:
            if proposal is not None:
                raise ValueError("WAIT proposal must be null")
            decoded.append(HierarchicalActionV42(value, None).normalized)
        elif value == RELEASE_GATE:
            if proposal is None or not np.isfinite(proposal) or proposal != round(float(proposal)):
                raise ValueError("RELEASE proposal must be a finite integer")
            decoded.append(HierarchicalActionV42(value, int(proposal)).normalized)
        else:
            raise ValueError("gate must be WAIT(0) or RELEASE(1)")
    return np.asarray(decoded, dtype=np.float32).reshape(gate_values.shape + (3,))


def hierarchical_targets_from_v4_1_joint(indices: Any) -> tuple[np.ndarray, np.ndarray]:
    """Pure encoding view: V4.1 Teacher labels are never recomputed or edited."""
    values = np.asarray(indices)
    if not np.all(np.isfinite(values)) or not np.all(values == np.round(values)):
        raise ValueError("joint targets must be finite integer indices")
    values = values.astype(np.int64, copy=False)
    if np.any(values < 0) or np.any(values >= N_EXECUTION_ACTIONS_V4_2):
        raise ValueError("joint targets must be in [0, 26)")
    gates = (values > 0).astype(np.int64)
    # -1 is an internal tensor sentinel only. Serialized views use null.
    proposals = np.where(gates == RELEASE_GATE, values - 1, -1).astype(np.int64)
    return gates, proposals
