"""Contract checks for the P2 scenario exposed by the existing 3D viewer."""
from __future__ import annotations

import pytest

from terminal_marl.visualization.ppo_checkpoint_viewer import _p2_scene_config


@pytest.mark.parametrize("scenario,count", (("flow6", 6), ("flow8", 8), ("flow12", 12)))
def test_p2_viewer_uses_natural_flow_and_runtime_shortlist_k3(scenario: str, count: int):
    config = _p2_scene_config(scenario)

    assert config.total_requested_uavs == count
    assert config.fixed_arrival_schedule == tuple(10.0 + 60.0 * index for index in range(count))
    assert config.max_simulation_seconds == config.fixed_arrival_schedule[-1] + 900.0
    assert config.runtime_proposal_shortlist_k == 3
    assert config.high_level_enabled is False
    assert config.teacher_enabled is False
    assert config.scheduler_mode == "execution"
    assert config.shield_enabled is True


def test_p2_viewer_rejects_non_p2_scene():
    with pytest.raises(ValueError, match="P2 profiles require"):
        _p2_scene_config("warm3")
