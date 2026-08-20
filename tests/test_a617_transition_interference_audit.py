from terminal_marl.training.a617_transition_interference_audit import (
    MATERIAL_WAITING_AREA,
    PPO_AUTHORIZATION,
    TIMES,
    _classify,
)


def _rows(full_delta: float, freeze_delta: float):
    rows = []
    for timing in TIMES:
        rows.extend([
            {"seed": 0, "timing": timing, "branch": "nominal", "waiting_area": 1_000_000.0},
            {"seed": 0, "timing": timing, "branch": "full_dynamic", "waiting_area": 1_000_000.0 + full_delta},
            {"seed": 0, "timing": timing, "branch": "freeze_only", "waiting_area": 1_000_000.0 + freeze_delta},
        ])
    return rows


def test_a617_sweep_covers_required_transition_actions():
    assert TIMES == (180.0, 240.0, 300.0)


def test_a617_freeze_reproduction_requires_same_sign_material_effect():
    assert _classify(_rows(-MATERIAL_WAITING_AREA, -MATERIAL_WAITING_AREA), (0,)) == ["RECONFIGURATION_FREEZE_INTERFERENCE"]
    assert _classify(_rows(-MATERIAL_WAITING_AREA, MATERIAL_WAITING_AREA), (0,)) == ["TRANSITION_GEOMETRY_PHASE_SHIFT"]


def test_a617_nonmaterial_evidence_is_inconclusive():
    assert _classify(_rows(100.0, 100.0), (0,)) == ["INCONCLUSIVE"]


def test_a617_cannot_authorize_ppo():
    assert PPO_AUTHORIZATION == "STOP"
