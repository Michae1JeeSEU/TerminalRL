"""Low-Level Action V4.1: canonical WAIT plus 25 RELEASE proposals.

V4.1 supersedes the V4 30-category policy catalogue.  Holder speed while
waiting is an execution/safety concern, not a scheduling-policy decision.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


ACTION_SCHEMA_V4_1 = "low_action_schema_v4_1_joint_discrete_26"
RESIDUAL_VALUES = (-1.0, -0.5, 0.0, 0.5, 1.0)


@dataclass(frozen=True)
class JointActionV41:
    index: int
    release_request: float
    speed_residual: float
    merge_time_residual: float

    @property
    def normalized(self) -> np.ndarray:
        return np.asarray((self.release_request, self.speed_residual, self.merge_time_residual), dtype=np.float32)

    @property
    def kind(self) -> str:
        return "RELEASE" if self.release_request > 0.0 else "WAIT"

    def as_dict(self) -> dict[str, object]:
        return {"joint_action_index": self.index, "kind": self.kind,
                "release_request": self.release_request, "speed_residual": self.speed_residual,
                "merge_time_residual": self.merge_time_residual}


# Index zero is the sole policy WAIT.  Its zero residual is a decoding
# canonicalisation, not holder-speed authority.
JOINT_ACTION_CATALOG_V4_1: tuple[JointActionV41, ...] = tuple(
    [JointActionV41(0, 0.0, 0.0, 0.0)] +
    [JointActionV41(1 + speed_index * len(RESIDUAL_VALUES) + merge_index, 1.0, speed, merge)
     for speed_index, speed in enumerate(RESIDUAL_VALUES)
     for merge_index, merge in enumerate(RESIDUAL_VALUES)]
)
N_JOINT_ACTIONS_V4_1 = len(JOINT_ACTION_CATALOG_V4_1)
if N_JOINT_ACTIONS_V4_1 != 26:
    raise RuntimeError("V4.1 joint catalogue must contain 1 WAIT + 25 RELEASE actions")


def joint_action_v4_1(index: int) -> JointActionV41:
    if not isinstance(index, (int, np.integer)) or not 0 <= int(index) < N_JOINT_ACTIONS_V4_1:
        raise ValueError(f"V4.1 joint action must be in [0, {N_JOINT_ACTIONS_V4_1})")
    return JOINT_ACTION_CATALOG_V4_1[int(index)]


def decode_joint_actions_v4_1(indices: Any) -> np.ndarray:
    values = np.asarray(indices)
    if not np.all(np.isfinite(values)) or not np.all(values == np.round(values)):
        raise ValueError("V4.1 joint actions must be finite integer indices")
    values = values.astype(np.int64, copy=False)
    if np.any(values < 0) or np.any(values >= N_JOINT_ACTIONS_V4_1):
        raise ValueError(f"V4.1 joint action indices outside [0, {N_JOINT_ACTIONS_V4_1})")
    return np.asarray([joint_action_v4_1(int(value)).normalized for value in values.reshape(-1)], dtype=np.float32).reshape(values.shape + (3,))


def joint_index_from_normalized_v4_1(action: Any, *, atol: float = 1e-6) -> np.ndarray:
    values = np.asarray(action, dtype=np.float32)
    if values.shape[-1:] != (3,) or not np.all(np.isfinite(values)):
        raise ValueError("V4.1 normalized action must be finite with final dimension 3")
    flat = values.reshape(-1, 3)
    catalog = np.asarray([item.normalized for item in JOINT_ACTION_CATALOG_V4_1], dtype=np.float32)
    distances = np.max(np.abs(flat[:, None, :] - catalog[None, :, :]), axis=-1)
    chosen = distances.argmin(axis=1)
    if np.any(distances[np.arange(len(flat)), chosen] > atol):
        raise ValueError("action is not a V4.1 joint catalogue value")
    return chosen.reshape(values.shape[:-1]).astype(np.int64)
