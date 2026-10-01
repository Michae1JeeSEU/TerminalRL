import numpy as np
import pytest

from terminal_marl.training.p3_gate_last_layer_projection_20261002 import (
    HIDDEN_DIM,
    bernoulli_kl_from_logits,
    project_gate_last_layer,
)


def _inputs(*, anchor_rows=32, anchor_scale=0.2):
    rng = np.random.default_rng(20261002)
    H_anchor = rng.normal(scale=anchor_scale, size=(anchor_rows, HIDDEN_DIM))
    H_target = np.zeros((4, HIDDEN_DIM), dtype=np.float64)
    H_target[:, :4] = np.eye(4)
    source_anchor = rng.normal(scale=0.5, size=anchor_rows)
    source_target = np.array([-0.3, 0.2, 0.0, 0.4], dtype=np.float64)
    pstar = np.array([0.75, 0.15, 0.8, 0.25], dtype=np.float64)
    return H_anchor, H_target, source_anchor, source_target, pstar


def test_full_rank_mixed_constraints_are_satisfied_without_cap():
    inputs = _inputs()
    result = project_gate_last_layer(*inputs, kl_cap=10.0)

    assert result["rank"] == 4
    assert result["scale_alpha"] == 1.0
    assert result["bisection_steps"] == 0
    np.testing.assert_allclose(
        result["target_constraint_residual"], 0.0, atol=1.0e-8
    )
    np.testing.assert_allclose(
        result["candidate_target_probability"], inputs[-1], atol=1.0e-12
    )
    assert np.any(result["requested_target_logit_delta"] < 0.0)
    assert np.any(result["requested_target_logit_delta"] > 0.0)


def test_kl_cap_scales_every_target_toward_pstar():
    inputs = _inputs(anchor_scale=1.5)
    result = project_gate_last_layer(*inputs)

    assert 0.0 < result["scale_alpha"] < 1.0
    assert result["max_kl"] <= 0.00095 + 1.0e-10
    source = result["source_target_probability"]
    candidate = result["candidate_target_probability"]
    pstar = inputs[-1]
    assert np.all((candidate - source) * (pstar - source) >= -1.0e-14)
    assert np.any(np.abs(candidate - source) > 0.0)


def test_repeating_each_anchor_row_preserves_equal_mean_projection():
    H_anchor, H_target, source_anchor, source_target, pstar = _inputs()
    base = project_gate_last_layer(
        H_anchor, H_target, source_anchor, source_target, pstar, kl_cap=10.0
    )
    repeated = project_gate_last_layer(
        np.repeat(H_anchor, 3, axis=0),
        H_target,
        np.repeat(source_anchor, 3),
        source_target,
        pstar,
        kl_cap=10.0,
    )
    np.testing.assert_allclose(repeated["direction"], base["direction"], atol=1.0e-12)
    np.testing.assert_allclose(
        repeated["requested_target_logit_delta"], base["requested_target_logit_delta"]
    )
    assert repeated["scale_alpha"] == base["scale_alpha"]


def test_extreme_logits_have_finite_probabilities_and_zero_identity_kl():
    identity = bernoulli_kl_from_logits(
        np.array([-80.0, 80.0, -80.0, 80.0]),
        np.array([-80.0, 80.0, -80.0, 80.0]),
    )
    np.testing.assert_allclose(identity, 0.0, atol=0.0)

    H_anchor, H_target, _, _, pstar = _inputs()
    source_anchor = np.resize(np.array([-80.0, 80.0]), H_anchor.shape[0])
    source_target = np.array([-80.0, 80.0, -80.0, 80.0])
    result = project_gate_last_layer(
        H_anchor, H_target, source_anchor, source_target, pstar
    )
    assert np.all(np.isfinite(result["source_anchor_probability"]))
    assert np.all(np.isfinite(result["candidate_target_probability"]))
    assert np.isfinite(result["max_kl"])


def test_rank_deficient_target_features_are_rejected():
    H_anchor, H_target, source_anchor, source_target, pstar = _inputs()
    H_target[1] = H_target[0]
    with pytest.raises(ValueError, match="rank"):
        project_gate_last_layer(
            H_anchor, H_target, source_anchor, source_target, pstar
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("H_anchor", "nonfinite", "H_anchor"),
        ("source_target_logits", "nonfinite", "source_target_logits"),
        ("pstar", "endpoint", "strictly inside"),
    ],
)
def test_nonfinite_or_endpoint_inputs_are_rejected(field, value, message):
    inputs = list(_inputs())
    if field == "H_anchor":
        inputs[0][0, 0] = np.nan
    elif field == "source_target_logits":
        inputs[3][0] = np.inf
    else:
        inputs[4][0] = 0.0
    with pytest.raises(ValueError, match=message):
        project_gate_last_layer(*inputs)


def test_requested_and_applied_deltas_are_reported_separately():
    inputs = _inputs()
    result = project_gate_last_layer(*inputs)
    np.testing.assert_allclose(
        result["requested_target_logit_delta"],
        np.log(inputs[-1] / (1.0 - inputs[-1])) - inputs[3],
    )
    np.testing.assert_allclose(
        result["applied_target_logit_delta"],
        result["scale_alpha"] * result["requested_target_logit_delta"],
        atol=1.0e-10,
    )
