"""Frozen synthetic research defaults; no hardware calibration claim."""
from dataclasses import dataclass
from types import MappingProxyType
from terminal_marl.core.fixed_terminal.domain import TerminalConfig


@dataclass(frozen=True)
class LoadProfile:
    initial_fraction: float
    mean_interarrival: float


@dataclass(frozen=True)
class SOCProfile:
    mean: float
    std: float
    lower: float = .05
    upper: float = .98


DEFAULT_CONFIG = TerminalConfig()
LOAD_PROFILES = MappingProxyType({"low": LoadProfile(.25, 35.), "medium": LoadProfile(.6, 18.), "high": LoadProfile(.9, 8.)})
SOC_PROFILES = MappingProxyType({"high": SOCProfile(.85, .08), "medium": SOCProfile(.55, .12), "low": SOCProfile(.30, .10)})
STANDARD_SEEDS = (101, 202, 303)
