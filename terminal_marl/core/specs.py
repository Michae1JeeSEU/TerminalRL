
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass(frozen=True)
class BoxSpec:
    shape: Tuple[int, ...]
    low: np.ndarray
    high: np.ndarray
    dtype: str = "float32"


@dataclass(frozen=True)
class DiscreteSpec:
    n: int
    dtype: str = "int64"
