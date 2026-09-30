# P3 balanced proposal-support protocol (2026-10-01)

## Scope and status

This is a frozen off-policy proposal intervention for descriptive coverage of
the registered 24-pair grid. It does not train, use PPO, use a Teacher or
optimizer, consume held-out data, rerun the validated nine-pair control, or
promote a method. The fixed proposal parameters remain `epsilon=0.10`,
`q_floor_min=0.004`, 25 proposal categories, sampled-proposal-first `K=3`,
and the legacy 88D/V2.1, 1800-tick execution contract.

## Inputs and immutable guards

The grid is the registered six-environment by four-keyed-seed set:
`env_seed=877101..877106` crossed with `keyed_policy_seed=9303101..9303104`,
pair IDs 1 through 24. The nine existing proposal-support results for pair IDs
`1,5,9,13,17,21,14,15,16` are imported from
`proposal_support_control_20261001`; the registered v1 `full.json` SHA is
`0178f6cef9083babab207d1fbd0e71cc50b4ebd90ffc2e4601caaeb07a8668b5`.
Each imported pair and arm file, v1 preflight/pilot, source/config/runtime
binding, and all 24 legacy reference arms are hash and seed checked before a
new phase can run. The legacy reference arms are read from
`forced_wait_frozen/arms/pair_NNNN_legacy.json` and are never rewritten.
When rechecking an imported arm against its stored summary, the only numeric
round-trip tolerance is `abs_tol=1e-8` for the finite scalar
`energy_total_consumed`, which is recomputed as an arrival-row sum. All keys,
other values, outcomes, flags, and hashes remain exact; reported values and
the registered v1 full SHA are unchanged.

## Execution

`preflight` creates no environment and records the v1 and 24-arm bindings.
`full` reuses the nine validated summaries and runs each of the other 15 pairs
once through the existing proposal-support `_run_pair` and keyed request
implementation. The gate, actor mask, keyed streams, physics, safety, energy,
emergency, capacity, and reference artifacts stay fixed. A technical or hard
safety error stops the phase after retaining its artifact. Business failures
and censoring remain in the fixed run and keep their metrics unknown where the
full scene is incomplete. There is no retry, resampling, seed choice, or cap
change.

## Reporting and criterion

The output reports all 24 pair summaries and six descriptive environment
profiles, each profile containing four keyed seeds. Profiles are descriptive
groupings, not independent samples. It reports normal/emergency/failed/
diverted/unresolved outcomes, generated/admitted/completed/resolved counts,
wait, elapsed time, energy, raw reward excluding rejection penalty, and the
penalty separately.

The same development criterion as the nine-pair control is evaluated only when
all 24 scenes are natural and technically/hard-safe: 24 admissions per scene,
no new failed/diverted/unresolved arrivals, aggregate normal and successful
completion no lower than the paired legacy reference, aggregate wait no
higher, and strict business improvement. Censored or incomplete values remain
`None` and cannot pass the criterion. This result is not formal P3
qualification.

Implementation: `terminal_marl/training/run_p3_proposal_support_balanced_20261001.py`.
Tests: `tests/test_p3_proposal_support_balanced.py`.
