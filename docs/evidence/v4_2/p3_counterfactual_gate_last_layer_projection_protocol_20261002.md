# P3 counterfactual gate last-layer projection protocol (2026-10-02)

Status: DRAFT_UNADMITTED.

This document records one reviewable closed-form proposal. The mathematical helper and synthetic tests are preparation work; real-data projection solving, model-weight application, candidate generation, and any new performance execution are not authorized by this document. The human approval required for a later execution remains separate.

## Fixed inputs and trainable scope

Use the unchanged source checkpoint `/home/user/training_outputs/terminal/p3_normal_advantage_gate_repair_20260928/fit_smoke/terminal_gate_repair.pt` (SHA256 `3fe7dd1e01d032227521f6fc5c8464fe8d00d98fb0e9f0dbd13411123a27d4ab`), golden model binding digest `3434378719efdb02bf3d41febb52e66797a0d3928756fc0cbbfde056ddd337b0`, target artifact `/home/user/training_outputs/terminal/p3_followup_20260930/counterfactual_gate_targets_20261001/targets_v1.json` (SHA256 `4085299501758c55c476fac9676092ace9d9c69f4ea5d3a68b74f786eaaf1473`), anchor artifact `/home/user/training_outputs/terminal/p3_followup_20260930/gate_anchor_states_20261001/full.json` (SHA256 `1129676804003dd5a569a7723015ef9ceddd8c1cc016cb1ffca02c3cd4318d35`), and fit-readiness artifact `/home/user/training_outputs/terminal/p3_followup_20260930/counterfactual_gate_targets_20261001/fit_readiness_v1.json` (SHA256 `646f95f7d944a58ab2c6a0950b48fd4e1f5f99cc8f838d826ce8b0ca37ee809a`). Compute the projection in CPU float64.

Only gate_adapter.4.weight and gate_adapter.4.bias may change once. Freeze adapter layers 0 and 2, the base gate, proposal head, critic, and calibrated temperature T=2.5. Source q, RNG, and all other model parameters remain unchanged. The read-only structural proof reports H_t shape 4 x 97 with rank 4 and singular values [13.5657020230, 5.8354903501, 2.0122989667, 1.3534608620]; hidden shape is 4 x 96 and the last-layer weight/bias shapes are 2 x 96 and 2 in float32 at T=2.5. Global anchor indices are [4873, 12277, 18971, 26870]. This is structural solvability evidence only; source/RNG were unchanged and no autograd, optimizer, weight update, candidate, or performance result occurred. Structural proof SHA=a03ded5c682ae909c3ea0ab91cc31ba45a29136fd5c3b000b03e992113012127.

The candidate uses the same 28570 anchors and four targets. It is an intrinsic candidate screen, not a business evaluation, P3-resolution claim, default qualification, or PPO admission.

## Closed-form projection

Let H be the source gate_adapter[:-1] observation matrix with a final column of ones, H in R^(N x 97). Let g0 be the calibrated gate-logit difference logit(p) at each anchor. For each target, define d = logit(pstar) - g0_target.

Solve one equality-constrained minimum-displacement projection without changing labels or selecting states:

Q = H_a^T H_a / N + 1e-6 I

X = solve(Q, H_t^T)

v = X solve(H_t X, d)

Require finite values and equality residual at most 1e-8. No state is replaced, reweighted, or selected by business loss.

Apply one scalar correction g' = g0 + alpha H v. Search alpha in [0,1] with exactly 60 fixed bisection steps for the largest alpha satisfying max KL(candidate || source) over all 28570 anchors and the four targets at most 0.00095. If alpha=1 satisfies the numeric reserve, retain alpha=1. The public hard cap remains 0.001 plus 1e-10; the 0.00095 reserve is purely a numeric projection-screen reserve and does not relax that cap.

Encode only in the final gate adapter linear layer:

weight row 1 += (T/2) * alpha * v[:96]
weight row 0 -= (T/2) * alpha * v[:96]
bias row 1 += (T/2) * alpha * v[96]
bias row 0 -= (T/2) * alpha * v[96]

Apply the resulting model weights once as float32. Do not run Adam, PPO, backpropagation, environment interaction, teacher evaluation, held-out evaluation, or an alternate learning-rate/seed/best-of search.

## Hard failure checks

After the one application, a candidate fails with no retry if any condition holds: non-finite output; real maximum KL above `0.001 + 1e-10`; for any target, `abs(p_final - pstar) > abs(p0 - pstar) + 1e-6` (the screen allows non-worsening within tolerance and does not require strict movement); `meanBCE_final >= meanBCE_initial - 1e-10`; or strict reload checks fail for all frozen q, RNG, source, and frozen parameter bindings. On failure, retain the previous valid state and stop.

## Planning output and authorization boundary

The fixed planning output is `/home/user/training_outputs/terminal/p3_followup_20260930/counterfactual_gate_last_layer_projection_20261002`; it does not exist and has not been generated. A later implementation, preflight, ROOT review, and explicit human execution authorization are required before any real-data solve, weight application, candidate generation, or new performance execution.

No candidate is promoted by this protocol. Any later end-to-end scene comparison requires a separate ROOT preregistration. This draft does not claim a P3 solution or a global training cause.
