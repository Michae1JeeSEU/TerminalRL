"""Terminal MARL package.

PettingZoo-like parallel multi-agent environment for a 3D terminal-area UAV arrival problem.
"""

from .core.config import SimConfig, RingSpec
from .config import ExperimentMode
from .envs.parallel_env import TerminalParallelEnv, DEFAULT_EPISODE_METRIC_KEYS

__all__ = [
    "SimConfig",
    "RingSpec",
    "ExperimentMode",
    "TerminalParallelEnv",
    "DEFAULT_EPISODE_METRIC_KEYS",
]
