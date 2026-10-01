import copy
import json
import math
from pathlib import Path

import pytest

from terminal_marl.training import p3_counterfactual_gate_targets_20261001 as targets


def _branch(*, censored=False, error=None, safety_violation=False):
    return {
        "terminal": True,
        "censored": censored,
        "error": error,
        "safety_violation": safety_violation,
        "termination_classification": "natural_terminal" if not censored and error is None else "error",
    }


def _cell(cell_id, *, p_before=0.4, q=None, branch_overrides=None):
    q = [1.0] * targets.PROPOSAL_COUNT if q is None else q
    release_returns = [float(10 + cell_id + index) for index in range(targets.PROPOSAL_COUNT)]
    wait_return = float(5 + cell_id)
    per_seed = []
    for offset, seed in enumerate(targets.EXPECTED_SEEDS):
        returns = [value + offset * 0.25 for value in release_returns]
        wait = wait_return + offset * 0.1
        delta, _ = targets.weighted_release_delta(q, returns, wait)
        per_seed.append(
            {
                "seed": seed,
                "complete": True,
                "release_returns": returns,
                "wait_return": wait,
                "D": delta,
                "business_metrics": {"wait_seconds": None, "linear_residual": None},
            }
        )
    recorded_d = sum(row["D"] for row in per_seed) / len(per_seed)
    branches = [_branch() for _ in range(targets.BRANCHES_PER_STATE)]
    for index, override in (branch_overrides or {}).items():
        branches[index].update(override)
    return {
        "cell_id": cell_id,
        "status": "COMPLETED",
        "phase": "full",
        "selection": {
            "context": {
                "actor_mask": True,
                "arrival_id": 15,
                "committed_plan": False,
                "now": 1.0 + cell_id,
                "phase": "ring_inner",
                "retry_free": True,
                "soc": 0.5,
                "uid": cell_id,
            }
        },
        "proposal_q_before": q,
        "branch_metrics": branches,
        "all_branches_natural": True,
        "replay": {
            "exact": True,
            "config_match": True,
            "base_reset_state_match": True,
            "reset_recipe_match": True,
            "reset_state_match": True,
            "steps_expected": 100,
            "steps_replayed": 100,
        },
        "source_policy": {"model_unchanged": True, "model_digest": "source"},
        "after_policy": {"model_unchanged": True, "model_digest": "after"},
        "caller_rng_unchanged": True,
        "invariants": {
            "native_policy_sampler_used": False,
            "optimizer_used": False,
            "teacher_used": False,
            "prefix_sampling": True,
            "PPO_ALLOWED": False,
            "D_includes_first_reward": True,
            "controller_max": 25,
            "public_k": 3,
            "max_steps": 1800,
        },
        "source_actual": {"decision": "approved", "last_decision": "approved"},
        "p_before": p_before,
        "p_after": p_before,
        "delta_p": 0.0,
        "local_m": 0.1,
        "linear_residual": None,
        "qualified_D": True,
        "per_seed": per_seed,
        "D": recorded_d,
    }


def _payload(*, probability_bounds=None):
    payload = {
        "stage": targets.STAGE,
        "status": "FULL_OK",
        "phase": "full",
        "PPO_ALLOWED": False,
        "preflight_sha256": targets.EXPECTED_PREFLIGHT_SHA256,
        "pilot_sha256": targets.EXPECTED_PILOT_SHA256,
        "cells": [_cell(cell_id) for cell_id in targets.EXPECTED_CELLS],
    }
    if probability_bounds is not None:
        payload["target_probability_bounds"] = list(probability_bounds)
    return payload


def _build(payload=None):
    return targets.build_targets(
        _payload() if payload is None else payload,
        input_sha256=targets.EXPECTED_INPUT_SHA256,
    )


def test_weighted_delta_normalizes_q_and_uses_all_25_returns():
    q = [2.0] + [1.0] * 24
    returns = [100.0] + [0.0] * 24
    delta, q_sum = targets.weighted_release_delta(q, returns, 5.0)
    assert q_sum == 26.0
    assert delta == pytest.approx(200.0 / 26.0 - 5.0)


def test_target_probability_caps_kl_and_explicit_probability_bounds():
    result = targets._target_probability(
        0.4,
        5000.0,
        probability_bounds=(0.2, 0.45),
    )
    assert 0.2 <= result["pstar"] <= 0.45
    assert result["kl_pstar_to_p0"] <= targets.KL_CAP + 1e-12
    assert result["bounds_capped"] is True
    assert targets._target_probability(0.0, 5000.0)["pstar"] == 0.0
    assert targets.reliability_weight(0.0, 0.0) == 0.0


def test_negative_and_zero_effective_values_move_probability_in_expected_direction():
    negative = targets._target_probability(0.4, -10.0)
    zero = targets._target_probability(0.4, 0.0)
    assert negative["pstar"] < 0.4
    assert zero["pstar"] == pytest.approx(0.4)
    assert zero["target_delta_p"] == pytest.approx(0.0)


def test_high_standard_error_damps_target_magnitude():
    low_uncertainty = targets.reliability_weight(4.0, 1.0)
    high_uncertainty = targets.reliability_weight(4.0, 8.0)
    low_target = targets._target_probability(0.4, low_uncertainty * 4.0)
    high_target = targets._target_probability(0.4, high_uncertainty * 4.0)
    assert high_uncertainty < low_uncertainty
    assert abs(high_target["target_delta_p"]) < abs(low_target["target_delta_p"])


def test_build_targets_known_all_cells_and_schema():
    output = _build()
    assert output["status"] == "TARGETS_READY"
    assert output["known_cell_count"] == 4
    assert [cell["cell_id"] for cell in output["cells"]] == [1, 2, 3, 4]
    for cell in output["cells"]:
        assert len(cell["per_seed"]) == 4
        assert len(cell["per_seed"][0]["release_returns"]) == 25
        assert sum(cell["proposal_q_before_normalized"]) == pytest.approx(1.0)
        assert math.isfinite(cell["mean_D"])
        assert math.isfinite(cell["pstar"])


def test_censored_branch_keeps_cell_and_withholds_qualified_statistics():
    payload = _payload()
    payload["cells"][0]["branch_metrics"][0]["censored"] = True
    output = _build(payload)
    cell = output["cells"][0]
    assert output["status"] == "TARGETS_PARTIAL_UNKNOWN"
    assert cell["known"] is False
    assert cell["pstar"] is None
    assert cell["mean_D"] is None
    assert cell["sample_se_D"] is None
    assert cell["reliability_weight"] is None
    assert cell["effective_D"] is None
    assert "branch_0_censored_or_unknown" in cell["unknown_reasons"]
    assert len(cell["per_seed"]) == 4
    assert sum(cell["proposal_q_before_normalized"]) == pytest.approx(1.0)


def test_missing_seed_returns_keeps_unknown_reason_and_cell():
    payload = _payload()
    payload["cells"][1]["per_seed"][0]["release_returns"] = []
    output = _build(payload)
    cell = output["cells"][1]
    assert cell["known"] is False
    assert cell["mean_D"] is None
    assert cell["reliability_weight"] is None
    assert any("release_returns_missing_or_wrong_count" in reason for reason in cell["unknown_reasons"])
    assert len(cell["per_seed"]) == 4


def test_natural_business_failure_does_not_remove_value_evidence():
    payload = _payload()
    for row in payload["cells"][0]["per_seed"]:
        row["business_metrics"] = {"outcome": "business_failed", "wait_seconds": None}
    output = _build(payload)
    cell = output["cells"][0]
    assert cell["known"] is True
    assert cell["mean_D"] is not None
    assert cell["unknown_business"]["per_seed_business_metrics"][0]["outcome"] == "business_failed"


@pytest.mark.parametrize(
    "override",
    [
        {"safety_violation": True},
        {"error": "runtime_error"},
    ],
)
def test_safety_or_error_branch_withholds_qualified_statistics(override):
    payload = _payload()
    payload["cells"][0]["branch_metrics"][1].update(override)
    output = _build(payload)
    cell = output["cells"][0]
    assert cell["known"] is False
    assert cell["mean_D"] is None
    assert cell["sample_se_D"] is None
    assert cell["reliability_weight"] is None
    assert cell["effective_D"] is None


def test_invalid_q_keeps_raw_seed_rows_as_unknown_evidence():
    payload = _payload()
    payload["cells"][0]["proposal_q_before"][0] = -1.0
    output = _build(payload)
    cell = output["cells"][0]
    assert cell["known"] is False
    assert cell["proposal_q_before_normalized"] is None
    assert len(cell["per_seed"]) == 4
    assert cell["per_seed"][0]["release_returns"]
    assert cell["per_seed"][0]["recomputed_D"] is None
    assert any("negative probability" in reason for reason in cell["unknown_reasons"])


def test_mismatched_binding_phase_or_bounds_is_rejected():
    payload = _payload()
    payload["phase"] = "pilot"
    with pytest.raises(targets.InputValidationError, match="phase mismatch"):
        _build(payload)

    payload = _payload()
    with pytest.raises(targets.InputValidationError, match="input full SHA mismatch"):
        targets.build_targets(payload, input_sha256="0" * 64)

    payload = _payload(probability_bounds=(0.2, 0.8))
    payload["cells"][0]["p_before"] = 0.1
    with pytest.raises(targets.InputValidationError, match="outside explicit bounds"):
        _build(payload)


def test_recorded_d_mismatch_withholds_target_and_statistics():
    payload = _payload()
    payload["cells"][2]["per_seed"][0]["D"] += 1.0
    output = _build(payload)
    cell = output["cells"][2]
    assert cell["known"] is False
    assert cell["mean_D"] is None
    assert cell["sample_se_D"] is None
    assert "seed_9303401_recomputed_D_mismatch" in cell["unknown_reasons"]


def test_cell_mean_mismatch_withholds_post_recompute_statistics():
    payload = _payload()
    payload["cells"][3]["D"] += 1.0
    output = _build(payload)
    cell = output["cells"][3]
    assert cell["known"] is False
    assert cell["mean_D"] is None
    assert cell["sample_se_D"] is None
    assert cell["reliability_weight"] is None
    assert cell["effective_D"] is None
    assert "cell_mean_D_mismatch" in cell["unknown_reasons"]


def test_write_targets_refuses_overwrite_before_reading_input(tmp_path):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    input_path.write_text("{}", encoding="utf-8")
    output_path.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError):
        targets.write_targets(input_path, output_path)
    assert output_path.read_text(encoding="utf-8") == "existing"


def test_serialized_fixture_is_strict_json(tmp_path):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    input_path.write_text(json.dumps(_payload()), encoding="utf-8")
    # The synthetic fixture is intentionally not bound to the real full SHA, so invoke
    # the pure builder for schema and use the writer only for its strict JSON contract.
    output = _build()
    output_path.write_text(json.dumps(output, allow_nan=False), encoding="utf-8")
    decoded = json.loads(output_path.read_text(encoding="utf-8"))
    assert decoded["schema"] == targets.SCHEMA
    assert decoded["known_cell_count"] == 4
