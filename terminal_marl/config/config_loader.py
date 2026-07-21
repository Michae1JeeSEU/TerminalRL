"""Load terminal MARL YAML presets into backward-compatible ``SimConfig`` objects."""

from __future__ import annotations

import ast
import copy
from pathlib import Path
from typing import Any, Mapping

from ..core.config import RingSpec, SimConfig
from .experiment_mode import ExperimentMode


CONFIG_DIRECTORY = Path(__file__).resolve().parents[1] / "configs"


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
    """Parse the mapping-only YAML subset used by project presets.

    PyYAML remains the preferred parser.  This fallback keeps simulation and
    tests usable in minimal environments and intentionally rejects YAML lists,
    anchors, tags, and other advanced constructs.
    """

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        content = raw_line.split("#", 1)[0].rstrip()
        if not content:
            continue
        indent = len(content) - len(content.lstrip(" "))
        if "\t" in raw_line[:indent]:
            raise ValueError(f"tabs are not allowed in YAML indentation (line {line_number})")
        stripped = content.strip()
        if stripped.startswith("- "):
            raise ValueError("fallback YAML parser supports mappings and inline lists only")
        if ":" not in stripped:
            raise ValueError(f"expected key: value at line {line_number}")
        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"empty YAML key at line {line_number}")

        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError(f"invalid YAML indentation at line {line_number}")
        parent = stack[-1][1]
        if key in parent:
            raise ValueError(f"duplicate YAML key {key!r} at line {line_number}")
        if raw_value.strip():
            parent[key] = _parse_scalar(raw_value)
        else:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
    return root


def _read_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        data = _fallback_yaml_mapping(text)
    else:
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("configuration root must be a YAML mapping")
    return data


def _mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def _set_known_fields(config: SimConfig, values: Mapping[str, Any], section: str) -> None:
    for key, value in values.items():
        if not hasattr(config, key):
            raise ValueError(f"unknown SimConfig field {key!r} in {section}")
        setattr(config, key, value)


def _positive_seconds(value: Any, field_name: str) -> float:
    seconds = float(value)
    if seconds <= 0.0:
        raise ValueError(f"{field_name} must be positive")
    return seconds


def _load_rings(data: Mapping[str, Any], config: SimConfig) -> None:
    raw_rings = data.get("rings")
    if raw_rings is None:
        return
    if not isinstance(raw_rings, Mapping) or len(raw_rings) < 2:
        raise ValueError("rings must be a mapping containing at least inner and outer")

    rings: list[RingSpec] = []
    for name, raw_ring in raw_rings.items():
        if not isinstance(raw_ring, Mapping):
            raise ValueError(f"ring {name!r} must be a mapping")
        required = {"index", "radius", "height", "capacity"}
        missing = required - set(raw_ring)
        if missing:
            raise ValueError(f"ring {name!r} is missing: {', '.join(sorted(missing))}")
        speed = raw_ring.get("speed", raw_ring.get("nominal_speed"))
        rings.append(
            RingSpec(
                index=int(raw_ring["index"]),
                radius=float(raw_ring["radius"]),
                capacity=int(raw_ring["capacity"]),
                height=float(raw_ring["height"]),
                speed=None if speed is None else float(speed),
            )
        )
    rings.sort(key=lambda ring: ring.index)
    if len({ring.index for ring in rings}) != len(rings):
        raise ValueError("ring indices must be unique")
    config.rings = rings


def load_sim_config(
    path: str | Path,
    *,
    base: SimConfig | None = None,
) -> SimConfig:
    """Load a YAML preset over a copy of the Python default configuration."""

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    data = _read_yaml(config_path)
    config = copy.deepcopy(base) if base is not None else SimConfig()

    top_level_fields = {
        "name": "config_name",
        "experiment_mode": "experiment_mode",
        "action_mode": "action_mode",
        "simulation_only": "simulation_only",
        "center": "center",
        "dt": "dt",
        "max_steps": "max_steps",
    }
    for source, target in top_level_fields.items():
        if source in data:
            value = data[source]
            if target == "center":
                value = tuple(float(item) for item in value)
                if len(value) != 3:
                    raise ValueError("center must contain three coordinates")
            setattr(config, target, value)

    config.experiment_mode = ExperimentMode.parse(config.experiment_mode).value
    _load_rings(data, config)

    _set_known_fields(config, _mapping(data, "geometry"), "geometry")
    _set_known_fields(config, _mapping(data, "vehicle_limits"), "vehicle_limits")
    safety = dict(_mapping(data, "safety"))
    if "enabled" in safety:
        config.safety_enabled = bool(safety.pop("enabled"))
    if "boundary" in safety:
        boundary = tuple(float(item) for item in safety.pop("boundary"))
        if len(boundary) != 6:
            raise ValueError("safety.boundary must contain six values")
        config.world_boundary = boundary
        config.world_xy_limit = max(abs(boundary[0]), abs(boundary[1]), abs(boundary[2]), abs(boundary[3]))
        config.world_z_limit = float(boundary[5])
    _set_known_fields(config, safety, "safety")
    _set_known_fields(config, _mapping(data, "timing"), "timing")
    _set_known_fields(config, _mapping(data, "arrival_flow"), "arrival_flow")
    _set_known_fields(config, _mapping(data, "reward_weights"), "reward_weights")

    if "control_interval" in data:
        control_interval = float(data["control_interval"])
        if control_interval <= 0.0:
            raise ValueError("control_interval must be positive")
        config.dt = control_interval
    if "high_level_interval" in data:
        interval_seconds = _positive_seconds(data["high_level_interval"], "high_level_interval")
        config.high_level_interval_steps = max(1, int(round(interval_seconds / config.dt)))
        config.decision_interval_seconds = interval_seconds
        config.reconfiguration_dwell_seconds = interval_seconds

    high_level = dict(_mapping(data, "high_level"))
    if "metrics_sampling_interval_seconds" in high_level:
        config.metrics_sampling_interval_seconds = _positive_seconds(
            high_level.pop("metrics_sampling_interval_seconds"),
            "high_level.metrics_sampling_interval_seconds",
        )
    if "decision_interval_seconds" in high_level:
        config.decision_interval_seconds = _positive_seconds(
            high_level.pop("decision_interval_seconds"),
            "high_level.decision_interval_seconds",
        )
        config.high_level_interval_steps = max(
            1,
            int(round(config.decision_interval_seconds / config.dt)),
        )
    if "reconfiguration_dwell_seconds" in high_level:
        config.reconfiguration_dwell_seconds = _positive_seconds(
            high_level.pop("reconfiguration_dwell_seconds"),
            "high_level.reconfiguration_dwell_seconds",
        )
    if "geometry_transition_seconds" in high_level:
        config.geometry_transition_seconds = float(
            high_level.pop("geometry_transition_seconds")
        )
        if config.geometry_transition_seconds < 0.0:
            raise ValueError("high_level.geometry_transition_seconds must be non-negative")
    if high_level:
        unknown = ", ".join(sorted(high_level))
        raise ValueError(f"unknown high_level field(s): {unknown}")

    if config.high_level_interval_steps <= 0:
        raise ValueError("high_level_interval_steps must be positive")
    if config.dt <= 0.0:
        raise ValueError("dt must be positive")
    if config.decision_interval_seconds <= 0.0:
        config.decision_interval_seconds = config.high_level_interval_steps * config.dt
    if config.reconfiguration_dwell_seconds <= 0.0:
        config.reconfiguration_dwell_seconds = config.decision_interval_seconds
    if not config.simulation_only:
        raise ValueError("third-round presets must remain simulation_only")
    if (
        config.experiment_mode == ExperimentMode.CRAZYFLIE_LAB.value
        and not config.safety_enabled
    ):
        raise ValueError("crazyflie_lab cannot disable safety")
    if not config.safety_enabled and not config.simulation_only:
        raise ValueError("safety may only be disabled in simulation_only mode")
    return config


def load_experiment_config(
    name: str,
    *,
    base: SimConfig | None = None,
) -> SimConfig:
    """Load ``terminal_marl/configs/<name>.yaml``."""

    filename = name if name.endswith(".yaml") else f"{name}.yaml"
    return load_sim_config(CONFIG_DIRECTORY / filename, base=base)
