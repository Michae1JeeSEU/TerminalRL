"""Low-level Action V3: factorized, normalized scheduling requests.

The policy never emits vehicle units.  Its three categorical factors are
adapted to :class:`LowLevelIntent` only at the environment boundary, where the
existing vehicle/profile configuration maps normalized residuals to m/s and
seconds.  Keeping this conversion here makes the V3 policy portable across
vehicle profiles.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


ACTION_SCHEMA_V3 = "low_intent_v3_factorized_discrete"
RELEASE_VALUES = (0.0, 1.0)  # WAIT, RELEASE
RESIDUAL_VALUES = (-1.0, -0.5, 0.0, 0.5, 1.0)
HEAD_NAMES = ("release", "speed", "merge_time")
HEAD_CARDINALITIES = (2, 5, 5)


def _indices_array(indices: Any) -> np.ndarray:
    values = np.asarray(indices)
    if values.shape[-1:] != (3,):
        raise ValueError(f"V3 policy action must have final dimension 3, got {values.shape}")
    if not np.all(np.isfinite(values)) or not np.all(values == np.round(values)):
        raise ValueError("V3 policy action indices must be finite integers")
    values = values.astype(np.int64, copy=False)
    bounds = np.asarray(HEAD_CARDINALITIES, dtype=np.int64)
    if np.any(values < 0) or np.any(values >= bounds):
        raise ValueError(f"V3 policy action indices outside categorical bounds {HEAD_CARDINALITIES}")
    return values


def indices_to_normalized(indices: Any) -> np.ndarray:
    """Map V3 categorical indices to the public normalized action vector."""
    values = _indices_array(indices)
    catalog = np.asarray((RELEASE_VALUES, RESIDUAL_VALUES, RESIDUAL_VALUES), dtype=object)
    flat = values.reshape(-1, 3)
    mapped = np.asarray([[catalog[head][index] for head, index in enumerate(row)] for row in flat], dtype=np.float32)
    return mapped.reshape(values.shape)


def normalized_to_indices(action: Any, *, atol: float = 1e-6) -> np.ndarray:
    """Return the unique V3 categories for an already-quantized action.

    A non-catalog action is deliberately rejected.  This protects PPO/BC
    provenance from silently treating legacy continuous requests as V3 labels.
    Legacy execution callers may still use ``LowLevelIntent.from_array``.
    """
    values = np.asarray(action, dtype=np.float32)
    if values.shape[-1:] != (3,) or not np.all(np.isfinite(values)):
        raise ValueError("normalized V3 action must be finite with final dimension 3")
    flat = values.reshape(-1, 3)
    result = np.empty_like(flat, dtype=np.int64)
    for head, permitted in enumerate((RELEASE_VALUES, RESIDUAL_VALUES, RESIDUAL_VALUES)):
        options = np.asarray(permitted, dtype=np.float32)
        distances = np.abs(flat[:, head, None] - options[None, :])
        chosen = distances.argmin(axis=1)
        if np.any(distances[np.arange(len(flat)), chosen] > atol):
            raise ValueError(f"action head {HEAD_NAMES[head]} is not a V3 catalog value")
        result[:, head] = chosen
    return result.reshape(values.shape)


def quantize_normalized(action: Any) -> np.ndarray:
    """Explicit nearest-level adapter for a continuous teacher proposal."""
    values = np.asarray(action, dtype=np.float32)
    if values.shape[-1:] != (3,) or not np.all(np.isfinite(values)):
        raise ValueError("teacher action must be finite with final dimension 3")
    flat = values.reshape(-1, 3)
    categories = np.empty_like(flat, dtype=np.int64)
    categories[:, 0] = (flat[:, 0] > 0.0).astype(np.int64)
    for head in (1, 2):
        options = np.asarray(RESIDUAL_VALUES, dtype=np.float32)
        categories[:, head] = np.abs(flat[:, head, None] - options).argmin(axis=1)
    return indices_to_normalized(categories).reshape(values.shape)


@dataclass(frozen=True)
class LowLevelActionV3:
    """One factorized categorical scheduling request and its normalized view."""

    release: int
    speed: int
    merge_time: int

    def __post_init__(self) -> None:
        _indices_array(self.indices)

    @property
    def indices(self) -> np.ndarray:
        return np.asarray((self.release, self.speed, self.merge_time), dtype=np.int64)

    @property
    def normalized(self) -> np.ndarray:
        return indices_to_normalized(self.indices)

    def as_dict(self) -> dict[str, Any]:
        normalized = self.normalized
        return {
            "release_category": "RELEASE" if self.release else "WAIT",
            "release_index": self.release,
            "speed_index": self.speed,
            "speed_normalized": float(normalized[1]),
            "merge_time_index": self.merge_time,
            "merge_time_normalized": float(normalized[2]),
        }

    @classmethod
    def from_indices(cls, indices: Any) -> "LowLevelActionV3":
        values = _indices_array(indices)
        if values.shape != (3,):
            raise ValueError("one LowLevelActionV3 requires exactly three indices")
        return cls(*(int(value) for value in values))

    @classmethod
    def from_normalized(cls, action: Any) -> "LowLevelActionV3":
        return cls.from_indices(normalized_to_indices(action))

    @classmethod
    def quantized_teacher(cls, action: Any) -> "LowLevelActionV3":
        return cls.from_normalized(quantize_normalized(action))
