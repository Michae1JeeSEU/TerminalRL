from terminal_marl.training.a617_transition_interference_audit import PPO_AUTHORIZATION, _set_freeze_only_overlay
from terminal_marl.training.a619_diagnostic_purity_rerun import MATERIAL_WAITING_AREA, _classify


class _Airspace:
    def affected_interfaces(self):
        return {"o2m"}


class _Env:
    airspace = _Airspace()


def _rows(full_delta: float, freeze_delta: float):
    rows = []
    for seed in (0, 1):
        for timing in (180.0, 240.0, 300.0):
            rows += [
                {"seed": seed, "timing": timing, "branch": "nominal", "waiting_area": 1_000_000.0},
                {"seed": seed, "timing": timing, "branch": "full_dynamic", "waiting_area": 1_000_000.0 + full_delta},
                {"seed": seed, "timing": timing, "branch": "freeze_only", "waiting_area": 1_000_000.0 + freeze_delta},
            ]
    return rows


def test_a619_freeze_overlay_is_removed_instead_of_rebinding_canonical_method():
    env = _Env()
    canonical = env.airspace.affected_interfaces
    _set_freeze_only_overlay(env, True)
    assert env.airspace.affected_interfaces() == {"o2m"}
    _set_freeze_only_overlay(env, False)
    assert "affected_interfaces" not in env.airspace.__dict__
    assert env.airspace.affected_interfaces.__func__ is canonical.__func__


def test_a619_classifies_mixed_freeze_and_geometry_evidence():
    rows = _rows(-MATERIAL_WAITING_AREA, -MATERIAL_WAITING_AREA)
    rows[1]["waiting_area"] = 1_000_000.0 + MATERIAL_WAITING_AREA
    rows[2]["waiting_area"] = 1_000_000.0
    assert _classify(rows) == ["MIXED"]


def test_a619_cannot_authorize_ppo():
    assert PPO_AUTHORIZATION == "STOP"
