"""Preview-only tests; importing this module never simulates an episode."""
import pytest
from terminal_marl.scripts.evaluate_fixed_terminal_stage5 import compare_continuation_artifact


def test_roundoff_accepted_and_recorded_without_bitwise_claim():
    stored = {"aircraft": [{"position": [-10.029688336419241, 3.]}], "active": True, "count": 1}
    rebuilt = {"aircraft": [{"position": [-10.029688336419243, 3.]}], "active": True, "count": 1}
    audit = compare_continuation_artifact(stored, rebuilt)
    assert audit["exact_match"] is False and audit["float_difference_count"] == 1
    assert audit["max_abs_difference"] == 1.7763568394002505e-15
    assert audit["absolute_tolerance"] == 1e-12 and audit["relative_tolerance"] == 0.
    assert "not_bitwise" in audit["exact_match_definition"]


def test_exact_parsed_values_have_zero_differences():
    value = {"x": [1., 2, True, "same", None], "empty": {}}
    audit = compare_continuation_artifact(value, value)
    assert audit["exact_match"] is True and audit["float_difference_count"] == 0
    assert audit["max_abs_difference"] == 0.


def test_absolute_boundary_accepted_but_exact_config_mode_rejects():
    assert compare_continuation_artifact(0., 1e-12)["float_difference_count"] == 1
    with pytest.raises(ValueError) as error:
        compare_continuation_artifact({"radius": 12.}, {"radius": 12.+1e-13}, "$.snapshot.config", exact=True)
    assert "$.snapshot.config['radius']" in str(error.value)


@pytest.mark.parametrize("left,right", [
    (0., 2e-12), (1e10, 1e10+.01),
    (1, 2), (True, False), (True, 1), (1, 1.), ("a", "b"),
    ([1, 2], [2, 1]), ([1], [1, 2]),
    ({"a": 1}, {"a": 1, "b": 2}),
    ({"a": 1, "b": 2}, {"b": 2, "a": 1}),
    ([1], {"0": 1}), (None, 0),
    (float("nan"), float("nan")), (float("inf"), float("inf")), (0., float("-inf")),
])
def test_nonfloat_changes_over_tolerance_and_nonfinite_rejected_with_path(left, right):
    with pytest.raises(ValueError) as error:
        compare_continuation_artifact({"leaf": [left]}, {"leaf": [right]})
    assert "$['leaf'][0]" in str(error.value)


def test_snapshot_and_plan_audits_are_independent():
    snapshot = compare_continuation_artifact({"position": [0.]}, {"position": [1e-15]}, "$.snapshot")
    plans = compare_continuation_artifact([{"wait_cycles": [0], "end": 3.}], [{"wait_cycles": [0], "end": 3.}], "$.plans['root']")
    assert snapshot["float_difference_count"] == 1 and plans["float_difference_count"] == 0
