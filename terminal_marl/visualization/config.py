"""Runtime-only configuration for the optional terminal viewer."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VisualizationConfig:
    """Settings that never alter canonical simulation configuration."""

    enabled: bool = False
    frame_stride: int = 1
    show_trajectories: bool = False
    show_labels: bool = False
    show_target_geometry: bool = True
    playback_speed: float = 0.0
    trajectory_max_points: int = 20

    def __post_init__(self) -> None:
        if self.frame_stride < 1:
            raise ValueError("frame_stride must be at least 1")
        if self.playback_speed < 0.0:
            raise ValueError("playback_speed must be non-negative")
        if self.trajectory_max_points < 2:
            raise ValueError("trajectory_max_points must be at least two")
