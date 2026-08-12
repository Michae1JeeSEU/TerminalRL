from terminal_marl.training.a63_mechanism_screen import CONDITIONS, _positive_leverage, _reproducible, high_config
from terminal_marl.training.a63_dynamic_screen import _safe
from terminal_marl.training.a63_bottleneck_report import build_report


def test_a63_conditions_keep_canonical_10d_action_schema():
    assert set(CONDITIONS) == {"geometry_only", "headway_only", "geometry_headway"}
    assert all(action.shape == (10,) for action in CONDITIONS.values())
    assert high_config(dynamic=True).high_level_enabled
    assert not high_config(dynamic=False).high_level_enabled


def test_a63_leverage_requires_non_regressing_completion_and_material_change():
    nominal = {"completed": 10, "external_queue_area": 100., "service_delay_p95": 50.}
    assert _positive_leverage(nominal, {"completed": 11, "external_queue_area": 110., "service_delay_p95": 55.})["positive"]
    assert not _positive_leverage(nominal, {"completed": 9, "external_queue_area": 60., "service_delay_p95": 30.})["positive"]
    assert _reproducible({"a": 1.}, {"a": 1.})


def test_a63_dynamic_screen_fails_closed_on_missing_safety_metrics():
    assert not _safe({"collisions": 0, "nmac": 0})
    assert not _safe({"collisions": 0, "nmac": 0, "hard_separation_violations": 1})
    assert _safe({"collisions": 0, "nmac": 0, "hard_separation_violations": 0})


def test_a63_bottleneck_report_requires_all_short_screen_artifacts(tmp_path):
    try:
        build_report(tmp_path)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("report must not silently diagnose missing screen data")
