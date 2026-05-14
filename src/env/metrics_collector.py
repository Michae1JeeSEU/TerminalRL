from dataclasses import dataclass, field

import numpy as np


@dataclass
class EpisodeMetrics:
    episode: int = 0
    total_landed: int = 0
    total_failed: int = 0
    total_conflicts: int = 0
    total_priority_violations: float = 0.0
    total_energy: float = 0.0
    avg_wait_time: float = 0.0
    throughput_rate: float = 0.0
    step_count: int = 0


class MetricsCollector:
    """Aggregates per-step metrics and computes episode summaries."""

    def __init__(self):
        self.episode = 0
        self._reset()

    def _reset(self) -> None:
        self._step_energy: list[float] = []
        self._step_wait: list[float] = []
        self._step_conflicts: list[int] = []
        self._step_priority_violations: list[float] = []
        self._landed_this_step = 0
        self._failed_this_step = 0
        self._step_count = 0

    def step(self, total_energy: float, avg_wait: float, num_conflicts: int,
             priority_violations: float, landed: int, failed: int) -> None:
        self._step_energy.append(total_energy)
        self._step_wait.append(avg_wait)
        self._step_conflicts.append(num_conflicts)
        self._step_priority_violations.append(priority_violations)
        self._landed_this_step = landed
        self._failed_this_step = failed
        self._step_count += 1

    def end_episode(self) -> EpisodeMetrics:
        metrics = EpisodeMetrics(
            episode=self.episode,
            total_landed=self._landed_this_step,
            total_failed=self._failed_this_step,
            total_conflicts=sum(self._step_conflicts),
            total_priority_violations=sum(self._step_priority_violations),
            total_energy=float(np.sum(self._step_energy)) if self._step_energy else 0.0,
            avg_wait_time=float(np.mean(self._step_wait)) if self._step_wait else 0.0,
            throughput_rate=(self._landed_this_step / max(self._step_count, 1)),
            step_count=self._step_count,
        )
        self.episode += 1
        self._reset()
        return metrics

    @property
    def latest_energy(self) -> float:
        return self._step_energy[-1] if self._step_energy else 0.0

    @property
    def latest_wait(self) -> float:
        return self._step_wait[-1] if self._step_wait else 0.0

    @property
    def latest_conflicts(self) -> int:
        return self._step_conflicts[-1] if self._step_conflicts else 0
