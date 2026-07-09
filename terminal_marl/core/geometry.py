
from __future__ import annotations

import math
from typing import Iterable
import numpy as np


def wrap_angle(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def unit_xy_from_angle(angle_rad: float) -> np.ndarray:
    return np.array([math.cos(angle_rad), math.sin(angle_rad)], dtype=float)


def norm_xy(vec: Iterable[float]) -> float:
    arr = np.asarray(vec, dtype=float)
    return float(np.linalg.norm(arr[:2]))


def distance_3d(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def angle_to_point(from_xyz: np.ndarray, to_xyz: np.ndarray) -> float:
    diff = to_xyz[:2] - from_xyz[:2]
    return math.atan2(diff[1], diff[0])


def rotate90(vec_xy: np.ndarray) -> np.ndarray:
    return np.array([-vec_xy[1], vec_xy[0]], dtype=float)
