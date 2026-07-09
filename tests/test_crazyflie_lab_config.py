"""Schema and scale checks for the simulation-only Crazyflie preset."""

import ast
from pathlib import Path
import re


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "terminal_marl"
    / "configs"
    / "crazyflie_lab.yaml"
)


def config_text() -> str:
    return CONFIG_PATH.read_text(encoding="utf-8")


def parse_value(raw: str):
    value = raw.strip()
    if value in {"true", "false"}:
        return value == "true"
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def scalar(text: str, key: str):
    match = re.search(rf"(?m)^{re.escape(key)}:\s*(.+)$", text)
    assert match is not None, f"missing config key: {key}"
    return parse_value(match.group(1))


def nested_block(text: str, key: str, indent: int) -> dict:
    prefix = " " * indent
    child_prefix = " " * (indent + 2)
    match = re.search(
        rf"(?m)^{prefix}{re.escape(key)}:\s*$\n"
        rf"(?P<body>(?:^{child_prefix}[^\n]+\n?)+)",
        text,
    )
    assert match is not None, f"missing config block: {key}"
    result = {}
    for line in match.group("body").splitlines():
        name, raw = line.strip().split(":", 1)
        result[name] = parse_value(raw)
    return result


def test_crazyflie_lab_config_exists_and_is_simulation_only() -> None:
    text = config_text()
    assert scalar(text, "name") == "crazyflie_lab"
    assert scalar(text, "experiment_mode") == "crazyflie_lab"
    assert scalar(text, "simulation_only") is True


def test_lab_ring_geometry_is_ordered_and_room_scale() -> None:
    text = config_text()
    outer = nested_block(text, "outer", 2)
    inner = nested_block(text, "inner", 2)

    assert 0.0 < inner["radius"] < outer["radius"] < 2.0
    assert 0.0 < inner["height"] < outer["height"] <= 1.6
    assert inner["index"] == 1
    assert outer["index"] == 2


def test_lab_limits_and_intervals_are_complete() -> None:
    text = config_text()
    limits = nested_block(text, "limits", 0)

    assert 0.0 < limits["speed_min"] < limits["speed_max"]
    assert limits["max_acceleration"] > 0.0
    assert limits["safe_distance"] > limits["nmac_distance"] > limits["collision_distance"]
    assert scalar(text, "dt") > 0.0
    assert scalar(text, "control_interval") >= scalar(text, "dt")
    assert scalar(text, "high_level_interval") > scalar(text, "control_interval")
    assert len(limits["boundary"]) == 6
