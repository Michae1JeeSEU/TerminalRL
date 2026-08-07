"""Audit action semantics and timing for event-gated HRL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import load_experiment_config
from ..control.geometry_event_gate import ACTION_GROUPS
from ..envs.parallel_env import TerminalParallelEnv


ACTION_SEMANTICS = [
    {"index": 0, "name": "outer_radius_increment", "group": "structural_geometry", "unit": "normalized_increment"},
    {"index": 1, "name": "inner_radius_increment", "group": "structural_geometry", "unit": "normalized_increment"},
    {"index": 2, "name": "outer_height_increment", "group": "structural_geometry", "unit": "normalized_increment"},
    {"index": 3, "name": "inner_height_increment", "group": "structural_geometry", "unit": "normalized_increment"},
    {"index": 4, "name": "outer_speed_increment", "group": "operational", "unit": "normalized_increment"},
    {"index": 5, "name": "inner_speed_increment", "group": "operational", "unit": "normalized_increment"},
    {"index": 6, "name": "ring_headway_increment", "group": "operational", "unit": "normalized_increment"},
    {"index": 7, "name": "final_headway_increment", "group": "operational", "unit": "normalized_increment"},
]


def build_audit(config_name: str = "event_gated_hrl") -> dict:
    config = load_experiment_config(config_name)
    env = TerminalParallelEnv(config)
    env.reset(seed=0)
    bounds = env.airspace.bounds
    rate_limits = [
        bounds.max_radius_change_per_update,
        bounds.max_radius_change_per_update,
        bounds.max_height_change_per_update,
        bounds.max_height_change_per_update,
        bounds.max_speed_change_per_update,
        bounds.max_speed_change_per_update,
        bounds.max_headway_change_per_update,
        bounds.max_headway_change_per_update,
    ]
    nominal = env.airspace.state_vector()
    action_semantics = {
        "action_size": 8,
        "semantics": [
            {**item, "rate_limit_per_update": float(rate_limits[item["index"]]), "nominal_value": float(nominal[item["index"]])}
            for item in ACTION_SEMANTICS
        ],
        "structural_geometry_indices": list(ACTION_GROUPS["structural_geometry"]),
        "operational_indices": list(ACTION_GROUPS["operational"]),
        "action_type": "incremental bounded normalized action",
        "bounds": {
            "radius": [bounds.radius_min, bounds.radius_max],
            "height": [bounds.height_min, bounds.height_max],
            "speed": [bounds.speed_min, bounds.speed_max],
            "headway": [bounds.headway_min, bounds.headway_max],
        },
    }
    timing_semantics = {
        "metrics_sampling_interval_seconds": config.metrics_sampling_interval_seconds,
        "high_level_decision_interval_seconds": config.high_level_decision_interval_seconds,
        "high_level_interval_steps": config.high_level_interval_steps,
        "metrics_windows_per_decision": config.high_level_decision_interval_seconds / config.metrics_sampling_interval_seconds,
        "geometry_min_dwell_seconds": config.geometry_min_dwell_seconds,
        "geometry_gate_mode": config.geometry_gate_mode,
        "old_training_interval_override_present": "training presets may still contain high_level_interval; event-gated presets use 180s configs",
    }
    observation = env.get_high_level_observation()
    observation_semantics = {
        "observation_length": int(observation.shape[0]),
        "contains_airspace_state": True,
        "contains_backlog_congestion_conflict": True,
        "contains_arrival_service_information": True,
        "contains_previous_action": True,
        "contains_geometry_deviation": True,
        "contains_dwell_state": True,
        "gate_feature_count": 9,
        "previous_action_feature_count": 8,
    }
    return {
        "config_name": config_name,
        "action_semantics": action_semantics,
        "timing_semantics": timing_semantics,
        "observation_semantics": observation_semantics,
    }


def write_audit(output_dir: str | Path, config_name: str = "event_gated_hrl") -> dict:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    audit = build_audit(config_name)
    (root / "action_semantics.json").write_text(
        json.dumps(audit["action_semantics"], indent=2) + "\n",
        encoding="utf-8",
    )
    (root / "timing_semantics.json").write_text(
        json.dumps(audit["timing_semantics"], indent=2) + "\n",
        encoding="utf-8",
    )
    md = [
        "# Current MDP audit",
        "",
        "- High-level action is an incremental 8D bounded action.",
        "- Structural geometry dimensions: radius/height indices 0-3.",
        "- Operational dimensions: speed/headway indices 4-7.",
        f"- Decision interval: {audit['timing_semantics']['high_level_decision_interval_seconds']} s.",
        f"- Metrics sampling: {audit['timing_semantics']['metrics_sampling_interval_seconds']} s.",
        f"- Geometry dwell: {audit['timing_semantics']['geometry_min_dwell_seconds']} s.",
        f"- Observation length: {audit['observation_semantics']['observation_length']}.",
        "- Observation includes gate state, previous applied action, backlog/congestion/conflict, arrival-service imbalance, geometry deviation, and dwell state.",
    ]
    (root / "current_mdp_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="event_gated_hrl")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_audit(args.output_dir, args.config), indent=2))


if __name__ == "__main__":
    main()
