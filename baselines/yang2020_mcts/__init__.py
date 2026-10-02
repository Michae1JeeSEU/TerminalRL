"""Bounded, headless execution wrapper for the Yang--Wei 2020 MCTS source."""

from .runtime import CASES, RunOptions, RunResult, run_case

__all__ = ["CASES", "RunOptions", "RunResult", "run_case"]
