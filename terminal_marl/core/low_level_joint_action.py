"""Low-Level Action V4: one discrete scheduling-intention catalogue.

V4 deliberately removes V3's invalid independent release target.  A sampled
integer denotes the complete scheduling request; decoding is the sole bridge
to the unchanged normalized execution intent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


ACTION_SCHEMA_V4 = "low_action_schema_v4_joint_discrete"
RESIDUAL_VALUES = (-1.0, -0.5, 0.0, 0.5, 1.0)


@dataclass(frozen=True)
class JointAction:
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


# WAIT keeps a speed request for holder/shield semantics; it has no merge
# proposal, so merge is canonicalised to zero and is not duplicated fivefold.
JOINT_ACTION_CATALOG: tuple[JointAction, ...] = tuple(
    [JointAction(index, 0.0, speed, 0.0) for index, speed in enumerate(RESIDUAL_VALUES)] +
    [JointAction(5 + speed_index * len(RESIDUAL_VALUES) + merge_index, 1.0, speed, merge)
     for speed_index, speed in enumerate(RESIDUAL_VALUES)
     for merge_index, merge in enumerate(RESIDUAL_VALUES)]
)
N_JOINT_ACTIONS = len(JOINT_ACTION_CATALOG)
if N_JOINT_ACTIONS != 30:
    raise RuntimeError("V4 joint catalogue must contain 5 WAIT + 25 RELEASE actions")


def joint_action(index: int) -> JointAction:
    if not isinstance(index, (int, np.integer)) or not 0 <= int(index) < N_JOINT_ACTIONS:
        raise ValueError(f"V4 joint action must be in [0, {N_JOINT_ACTIONS})")
    return JOINT_ACTION_CATALOG[int(index)]


def decode_joint_actions(indices: Any) -> np.ndarray:
    values = np.asarray(indices)
    if not np.all(np.isfinite(values)) or not np.all(values == np.round(values)):
        raise ValueError("V4 joint actions must be finite integer indices")
    values = values.astype(np.int64, copy=False)
    if np.any(values < 0) or np.any(values >= N_JOINT_ACTIONS):
        raise ValueError(f"V4 joint action indices outside [0, {N_JOINT_ACTIONS})")
    flat = values.reshape(-1)
    decoded = np.asarray([joint_action(int(value)).normalized for value in flat], dtype=np.float32)
    return decoded.reshape(values.shape + (3,))


def joint_index_from_normalized(action: Any, *, atol: float = 1e-6) -> np.ndarray:
    values = np.asarray(action, dtype=np.float32)
    if values.shape[-1:] != (3,) or not np.all(np.isfinite(values)):
        raise ValueError("V4 normalized action must be finite with final dimension 3")
    flat = values.reshape(-1, 3)
    catalog = np.asarray([item.normalized for item in JOINT_ACTION_CATALOG], dtype=np.float32)
    distances = np.max(np.abs(flat[:, None, :] - catalog[None, :, :]), axis=-1)
    chosen = distances.argmin(axis=1)
    if np.any(distances[np.arange(len(flat)), chosen] > atol):
        raise ValueError("action is not a V4 joint catalogue value")
    return chosen.reshape(values.shape[:-1]).astype(np.int64)
