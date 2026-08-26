"""Preflight-only A5-v2 PPO launch contract.

The actual optimizer launch is deliberately absent until the teacher dataset
audit is approved.  Calling :func:`prepare_low_v2_training` is safe: it
validates configuration and records the curriculum metadata only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

from ..core.config import SimConfig
from ..core.observations import LOW_OBS_V2
from .low_v2_protocol import LOW_V2_TEACHER_CURRICULUM, phase_config, validate_low_v2_config


@dataclass(frozen=True)
class LowV2TrainingPreflight:
    observation_dimension: int
    action_dimension: int
    phases: tuple[str, ...]
    configs: tuple[dict, ...]


def prepare_low_v2_training(base: SimConfig | None = None, *, manifest_path: str | Path | None = None) -> LowV2TrainingPreflight:
    """Validate and optionally persist a training manifest; does not train."""
    root = base or SimConfig()
    configs = tuple(phase_config(root, phase) for phase in LOW_V2_TEACHER_CURRICULUM)
    for config, phase in zip(configs, LOW_V2_TEACHER_CURRICULUM):
        validate_low_v2_config(config, for_training=phase.name != "teacher_off")
    result = LowV2TrainingPreflight(LOW_OBS_V2.dimension, 3, tuple(phase.name for phase in LOW_V2_TEACHER_CURRICULUM),
                                    tuple({"experiment_mode": config.experiment_mode, "scheduler_mode": config.scheduler_mode,
                                           "teacher_mode": config.teacher_mode, "teacher_enabled": config.teacher_enabled,
                                           "shield_enabled": config.shield_enabled} for config in configs))
    if manifest_path is not None:
        path = Path(manifest_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return result
