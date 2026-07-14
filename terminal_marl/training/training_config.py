"""YAML-backed configuration for server-side high-level PPO jobs.

The training configuration is intentionally separate from ``SimConfig``:
``SimConfig`` describes the terminal-airspace simulation, while this module
describes how a high-level PPO run should be launched on a workstation or
cluster node.  All presets remain simulation-only and never activate
Crazyflie/Crazyswarm2 hardware paths.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping


TRAINING_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "training"


@dataclass
class PPOTrainingConfig:
    """Server-oriented high-level PPO hyperparameters and run metadata."""

    config: str = "dynamic_hrl"
    demand_level: str = "medium"
    total_timesteps: int = 2048
    n_envs: int = 1
    vec_env: str = "dummy"
    seed_list: tuple[int, ...] = (0,)
    learning_rate: float = 3e-4
    n_steps: int = 64
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    high_level_interval: int | None = None
    low_level_policy: str = "rule-only"
    nominal_delay_mode: str = "analytic"
    output_dir: str = "runs/high_level_ppo"
    device: str = "cpu"
    num_threads: int = 1
    checkpoint_freq: int = 0
    eval_freq: int = 0
    save_replay_traces: bool = False
    fail_if_no_traffic: bool = False
    min_spawned_before_training_check: int = 1
    traffic_check_steps: int = 8
    sanity_check_baseline: bool = False
    max_steps: int | None = None
    simulation_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["seed_list"] = list(self.seed_list)
        return data


def _parse_scalar(raw: str) -> Any:
    value = raw.strip()
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", "~"}:
        return None
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value.strip("\"'")


def _fallback_yaml_mapping(text: str) -> dict[str, Any]:
    """Parse the simple mapping YAML subset used by training presets."""

    data: dict[str, Any] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        content = raw_line.split("#", 1)[0].strip()
        if not content:
            continue
        if ":" not in content:
            raise ValueError(f"expected key: value at line {line_number}")
        key, raw_value = content.split(":", 1)
        data[key.strip()] = _parse_scalar(raw_value)
    return data


def _read_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        data = _fallback_yaml_mapping(text)
    else:
        data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError("training config root must be a YAML mapping")
    return data


def _coerce_seed_list(value: Any) -> tuple[int, ...]:
    if isinstance(value, int):
        return (value,)
    if isinstance(value, str):
        if not value.strip():
            return ()
        return tuple(int(item.strip()) for item in value.split(","))
    if isinstance(value, (list, tuple)):
        return tuple(int(item) for item in value)
    raise ValueError("seed_list must be an int, comma string, or list of ints")


def _coerce_config(data: Mapping[str, Any]) -> PPOTrainingConfig:
    valid_fields = {field.name for field in fields(PPOTrainingConfig)}
    unknown = set(data) - valid_fields
    if unknown:
        raise ValueError(f"unknown training config fields: {', '.join(sorted(unknown))}")
    values = dict(data)
    if "seed_list" in values:
        values["seed_list"] = _coerce_seed_list(values["seed_list"])
    config = PPOTrainingConfig(**values)
    if config.config == "crazyflie_lab":
        raise ValueError("training presets must not target crazyflie_lab hardware mode")
    if not config.simulation_only:
        raise ValueError("training presets must remain simulation_only")
    if config.n_envs <= 0:
        raise ValueError("n_envs must be positive")
    if config.vec_env not in {"dummy", "subproc"}:
        raise ValueError("vec_env must be 'dummy' or 'subproc'")
    if config.device not in {"cpu", "cuda", "auto"}:
        raise ValueError("device must be 'cpu', 'cuda', or 'auto'")
    if config.total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")
    if config.n_steps <= 0 or config.batch_size <= 0:
        raise ValueError("n_steps and batch_size must be positive")
    if not config.seed_list:
        raise ValueError("seed_list must contain at least one seed")
    return config


def load_training_config(path_or_name: str | Path) -> PPOTrainingConfig:
    """Load a PPO training YAML file.

    ``path_or_name`` may be an explicit path or the basename of a preset inside
    ``terminal_marl/configs/training``.
    """

    path = Path(path_or_name)
    if not path.is_file():
        filename = str(path_or_name)
        if not filename.endswith(".yaml"):
            filename = f"{filename}.yaml"
        path = TRAINING_CONFIG_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    return _coerce_config(_read_yaml(path))


def merge_training_overrides(
    base: PPOTrainingConfig,
    overrides: Mapping[str, Any],
) -> PPOTrainingConfig:
    """Return a copy of ``base`` with non-``None`` explicit CLI overrides."""

    data = base.to_dict()
    for key, value in overrides.items():
        if value is not None:
            data[key] = value
    return _coerce_config(data)
