from dataclasses import replace
import json

import pytest

from terminal_marl.algo.student_ppo_v4_2 import StudentPPOV42
from terminal_marl.training.run_v4_2_hierarchical_s0_qualification import _config
from terminal_marl.training.student_rollout_v4_2 import (
    _accumulate_reward_components,
    rollout,
)


def _short_config():
    return replace(
        _config(),
        observation_schema_version="phaseA_low_obs_v2_1_retry_state",
        experiment_mode="fixed_low_student",
        teacher_enabled=False,
        warm_start_uavs=1,
        total_requested_uavs=1,
        fixed_arrival_schedule=(0.0,),
        max_simulation_seconds=3.0,
    )


def test_accumulator_ignores_per_uav_total_without_double_counting() -> None:
    totals: dict[str, float] = {}
    _accumulate_reward_components(
        totals,
        {"0": {"time": -0.1, "urgency": -0.2, "total": -0.3}},
    )
    assert totals == {"time": pytest.approx(-0.1), "urgency": pytest.approx(-0.2)}
    assert sum(totals.values()) == pytest.approx(-0.3)


def test_rollout_summary_records_post_step_component_totals(tmp_path) -> None:
    steps, result = rollout(
        StudentPPOV42(88),
        _short_config(),
        seed=705,
        max_steps=2,
        output=tmp_path / "reward_totals",
    )

    reward_totals = result["reward_component_totals"]
    assert "total" not in reward_totals
    assert result["reward_total"] == pytest.approx(sum(step.reward for step in steps))
    assert sum(reward_totals.values()) == pytest.approx(result["reward_total"])
    assert result["reward_components_consistent"] is True

    saved = json.loads((tmp_path / "reward_totals" / "summary.json").read_text())
    assert saved["reward_component_totals"] == reward_totals
    assert saved["reward_total"] == pytest.approx(result["reward_total"])
