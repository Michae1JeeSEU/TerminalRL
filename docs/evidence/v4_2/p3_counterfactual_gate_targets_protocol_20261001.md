# P3 counterfactual gate targets protocol — 2026-10-01

This document registers a bounded preparation step for the frozen normal gate-value
artifact. It creates soft target values from already completed JSON records. It does
not run an environment, load a model, perform a forward pass, train a policy, or
change any checkpoint.

## Frozen input and scope

The only accepted input is the completed full artifact with SHA-256
`1280dc0a5697913749f22982cff89aa7f0ff2a8075f4bb05b02331033ff7cdba`, stage
`P3_PREFIX_NORMAL_GATE_VALUE_20261001_V1`, preflight SHA
`a247e7cdbaee8f9686b67cf6506ba2ad5e7d59f5d38df09a0088f7c6629cd5f8`, and pilot SHA
`b5138ee3493fbfeaf22172f0107251a1897175878914727c21bd52e424a85332`. The frozen
policy labels are `BEFORE3fe` (`3fe7dd1e01d032227521f6fc5c8464fe8d00d98fb0e9f0dbd13411123a27d4ab`)
and `AFTERd449` (`d44990dc7d129573d9145f169df1cbc4f91fa32bc5127b1c58d937260c02a24f`).

The input contains four fixed cells, four continuation seeds
`9303401..9303404`, and 26 branches per seed: WAIT plus 25 RELEASE proposals.
Every cell and every seed row is retained. The tool rejects a different stage,
phase, binding, seed set, cell set, or input digest. `PPO_ALLOWED`, optimizer use,
teacher use, held-out data, and weight changes remain false.

## Full-episode value calculation

For each seed, the 25 recorded RELEASE returns are combined with all 25 recorded
proposal weights. The weights are normalized by their standard floating-point sum:

`D_seed = sum(normalized(proposal_q_before_j) * release_return_j) - wait_return`.

The recorded first reward remains inside each full-episode return. The four
recomputed values must match their recorded per-seed `D` values within `1e-10`, and
the cell mean must match its recorded `D` within the same tolerance.

A cell is target-eligible only when every branch is a natural terminal branch with
no censoring, error, or safety violation, the replay and model/RNG invariants pass,
the four seed rows are complete, and the recorded value checks pass. If any such
condition is unknown or fails, the cell remains in the output, its seed evidence is
retained, and its qualified mean, standard error, reliability weight, effective
value, and target are `null`. Missing business fields such as wait or linear
residual are retained as unknown; they are not replaced by zero.

## Uncertainty-softened target

For a known cell, let `mu` be the four-seed sample mean and `s` its sample standard
error (`ddof=1`). The preparation weight is

`w = mu^2 / (mu^2 + s^2)`,

with `w = 0` when both terms are zero. The effective value is `D_eff = w * mu`.
This is a conservative preparation heuristic, not a confidence probability or a
system guarantee.

The fixed sensitivity is `eta = 0.02`. With `p0` equal to the recorded BEFORE
probability, the unconstrained target uses

`logit(p*) = logit(p0) + alpha * eta * D_eff`,

starting with `alpha = 1`. The Bernoulli constraint is

`KL(Bernoulli(p*) || Bernoulli(p0)) <= 0.001`.

If the initial alpha violates the KL cap or an explicitly supplied probability
bound, a fixed 60-step bisection selects the largest feasible alpha in `[0, 1]`.
The default bounds are `[0, 1]`; no probability floor is introduced. Exact endpoint
values `p0 = 0` and `p0 = 1` remain endpoints. An explicit bound must contain `p0`.

The output records each cell's four recomputed seed values, mean, sample standard
error, sign counts, reliability weight, effective value, `p0`, exact observed
probability change, alpha, `p*`, KL, bound/KL cap flags, local state `m` when
present, and unknown business fields. It never sums local state values into a
global gain and does not treat the 4 cells or 416 branches as independent
performance samples.

Any later shared-network fitting would require representative old-policy state
anchors and a separate root-level preregistration and admission decision. These
four fixed cells are not a qualification population and do not provide a global
performance guarantee.

## CLI and output discipline

The pure-Python entry point is:

```text
python -m terminal_marl.training.p3_counterfactual_gate_targets_20261001 \
  --input-full <completed-full.json> \
  --output <new-targets.json>
```

The output path must not exist; an existing output is rejected. JSON serialization
uses strict non-finite-number rejection. The output is a preparation artifact only:
`PPO_ALLOWED=false`, `optimizer_used=false`, `teacher_used=false`, `heldout_used=false`,
and `weights_changed=false`. No fitting, optimizer, PPO admission, teacher run, or
new performance scenario is authorized by this protocol.
