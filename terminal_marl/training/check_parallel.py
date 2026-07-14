"""Low-cost local profile for high-level wrapper parallel stepping."""

from __future__ import annotations

import argparse
import json
import platform
import time
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np

from ..config import load_experiment_config
from ..envs.high_level_wrapper import HighLevelHRLWrapper
from ..envs.parallel_env import TerminalParallelEnv
from ..experiments.ablation_runner import apply_demand_profile


def _make_wrapper(config_name: str, demand_level: str, seed: int) -> HighLevelHRLWrapper:
    config = apply_demand_profile(load_experiment_config(config_name), demand_level)
    return HighLevelHRLWrapper(
        TerminalParallelEnv(config),
        low_level_policy="rule-only",
        seed=seed,
    )


def profile_parallel(
    *,
    config_name: str = "dynamic_hrl",
    demand_level: str = "medium",
    n_envs_values: Iterable[int] = (1, 2, 4),
    vec_env_values: Iterable[str] = ("dummy", "subproc"),
    high_level_steps: int = 4,
    output_dir: str | Path | None = None,
) -> list[dict[str, float | int | str | bool]]:
    """Profile tiny high-level stepping workloads without running PPO training."""

    rows: list[dict[str, float | int | str | bool]] = []
    for vec_env in vec_env_values:
        for n_envs in n_envs_values:
            if vec_env == "subproc" and platform.system().lower().startswith("win"):
                warnings.warn(
                    "Skipping subproc profile on Windows local debug to avoid spawn issues.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                rows.append(
                    {
                        "vec_env": vec_env,
                        "n_envs": int(n_envs),
                        "high_level_steps": int(high_level_steps),
                        "wall_time": 0.0,
                        "steps_per_sec": 0.0,
                        "avg_step_time": 0.0,
                        "skipped": True,
                        "reason": "subproc_skipped_on_windows",
                    }
                )
                continue
            wrappers = [
                _make_wrapper(config_name, demand_level, seed=1000 + idx)
                for idx in range(int(n_envs))
            ]
            for idx, wrapper in enumerate(wrappers):
                wrapper.reset(seed=1000 + idx)
            action = np.zeros(8, dtype=np.float32)
            started = time.perf_counter()
            executed = 0
            for _ in range(int(high_level_steps)):
                for wrapper in wrappers:
                    wrapper.step(action)
                    executed += 1
            elapsed = time.perf_counter() - started
            rows.append(
                {
                    "vec_env": vec_env,
                    "n_envs": int(n_envs),
                    "high_level_steps": int(high_level_steps),
                    "wall_time": float(elapsed),
                    "steps_per_sec": float(executed / elapsed) if elapsed > 0.0 else 0.0,
                    "avg_step_time": float(elapsed / max(1, executed)),
                    "skipped": False,
                    "reason": "",
                }
            )
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "parallel_profile.json").write_text(
            json.dumps(rows, indent=2) + "\n",
            encoding="utf-8",
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="dynamic_hrl")
    parser.add_argument("--demand-level", default="medium")
    parser.add_argument("--n-envs", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--vec-env", nargs="+", default=["dummy", "subproc"])
    parser.add_argument("--high-level-steps", type=int, default=4)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    rows = profile_parallel(
        config_name=args.config,
        demand_level=args.demand_level,
        n_envs_values=args.n_envs,
        vec_env_values=args.vec_env,
        high_level_steps=args.high_level_steps,
        output_dir=args.output_dir,
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

