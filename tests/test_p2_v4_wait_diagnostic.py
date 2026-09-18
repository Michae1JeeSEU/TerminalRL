from __future__ import annotations

from terminal_marl.training.diagnose_v4_2_p2_wait_credit import (
    FORMAL_SEEDS,
    _formal_complete,
    decision_diagnostics,
    diagnostic_config,
)


def _row(step: int, gate: int, teacher: int, *, uid: int = 1, interface: str = "o2m", advantage: float = 1.0):
    return {
        "seed": 771001,
        "step": step,
        "uav_id": uid,
        "interface": interface,
        "requested_gate": gate,
        "teacher_gate": teacher,
        "release_probability": 0.2 if gate == 0 else 0.8,
        "advantage": advantage,
        "release_eligible": True,
        "wait_feature": step / 10.0,
        "headway_slack": 1.0,
        "retry_remaining": 0.0,
        "predicted_conflict": False,
    }


def test_decision_diagnostic_reports_false_wait_streak_and_team_advantage() -> None:
    rows = [
        _row(1, 0, 1, advantage=2.0),
        _row(2, 0, 1, advantage=1.0),
        _row(3, 1, 1, advantage=-1.0),
        _row(4, 1, 0, advantage=-2.0, interface="m2i"),
        _row(5, 0, 0, advantage=0.5, interface="i2f"),
    ]
    result = decision_diagnostics(rows)
    assert result["requested"] == {"WAIT": 3, "RELEASE": 2}
    assert result["teacher"] == {"WAIT": 2, "RELEASE": 3}
    assert result["false_wait_rate"] == 2 / 3
    assert result["false_release_rate"] == 1 / 2
    assert result["false_wait_streaks"]["max"] == 2
    assert result["advantage_by_confusion_cell"]["false_wait"]["mean"] == 1.5
    assert result["interfaces"]["o2m"]["false_WAIT"] == 2


def test_diagnostic_config_preserves_student_authority_and_p2_reward() -> None:
    config = diagnostic_config(6)
    assert config.scheduler_mode == "execution"
    assert config.teacher_enabled and config.teacher_mode == "teacher_label_only"
    assert config.teacher_release_label_enabled is False
    assert config.low_level_progress_reward_per_meter == 0.0
    assert config.total_requested_uavs == 6


def test_formal_shape_requires_two_controllers_and_fresh_seed_block() -> None:
    assert not (set(FORMAL_SEEDS) & set(range(755001, 755007)))
    records = [
        {"controller": controller, "scenario": ("flow6", "flow8", "flow12")[index % 3], "result": {"seed": seed}}
        for controller in ("candidate", "baseline")
        for index, seed in enumerate(FORMAL_SEEDS)
    ]
    assert _formal_complete(records)
    assert not _formal_complete(records[:-1])
