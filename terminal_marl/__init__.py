"""Canonical Phase-A landing-only terminal management package."""
from .core.config import SimConfig, RingRole
from .envs.parallel_env import TerminalParallelEnv
__all__ = ["SimConfig", "RingRole", "TerminalParallelEnv"]
