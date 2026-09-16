# V4.1 Flat-26 vs Hierarchical Release Learnability Diagnostic

## Scope and evidence

This is a newly-run, offline diagnostic.  It is not a V4.2 migration and it
did not modify the Teacher, observations, formal I0-R data, BC objective,
class weights, reward, safety authority, DAgger, or PPO.

Inputs were the current I0-R artifact
`v4_1_i0_learning_coverage_repair_20260910` (dataset SHA-256
`1d06d9e48fb141a80cde20fc22cefcc1e666c1d14d2a3f30b068da85dc83ad9c`), its
independent frozen cohort (SHA-256
`6c5a07503d4c45da8e5318f303c5883b66414ae412e3c84c813cb2ea31dc0409`), and
the failed fresh V4.1 I1 checkpoint.  Each offline probe reduced the train
set to one deterministic, real state per lifecycle event: 240 train events,
156 frozen-cohort events, and 206 real events in the diagnostic-only balanced
subset.  No row was copied or relabelled.

## Flat-26 failure audit

The failed I1 policy put almost all probability on WAIT for both frozen-cohort
target families:

| Teacher family | P(WAIT) | P(RELEASE total) | Release minus WAIT | Best single release |
|---|---:|---:|---:|---:|
| WAIT | 0.9657 | 0.0343 | -0.9315 | 0.0189 |
| RELEASE | 0.9541 | 0.0459 | -0.9082 | 0.0240 |

Its best release proposal was class 13 for every state, including all 72
Teacher-RELEASE states.  This agrees with the recorded all-WAIT I1 failure;
it is not merely an argmax threshold effect.

The exact-observation check found zero same-observation hashes with opposite
WAIT/RELEASE labels.  There were 14 opposite labels sharing a full-state hash;
these are retained as an investigation note because the hash can encompass
different controlled-UAV decisions.  The nearest opposite-label normalized
observation distance was 0.0323 minimum, 1.2361 P50, and 2.5783 P95.  Thus
the evidence does not support an exact Observation V2 contradiction as the
primary explanation.

## Offline probes on the frozen cohort

| Probe | Training CE (before -> after) | Key held-out result |
|---|---:|---|
| A1: natural binary WAIT/RELEASE gate | 0.6894 -> 0.4101 | WAIT recall 0.8690; RELEASE recall 0.8472; macro-F1 0.8581 |
| A2: real-event balanced binary gate | 0.6977 -> 0.4334 | WAIT recall 0.8810; RELEASE recall 0.8333; macro-F1 0.8578 |
| B: conditional 25-way RELEASE proposal | 3.2593 -> 1.1114 | top-1 0.7639; top-3 0.9167; speed agreement 0.9444; merge agreement 0.7639 |
| C: flat 26-way categorical, same A2 events | 3.1153 -> 1.3123 | binary WAIT recall 0.8095; RELEASE recall 0.3472; argmax all class 0 |
| D: offline hierarchical gate + conditional proposal | 3.9607 -> 1.4842 | gate macro-F1 0.8107; proposal top-1/top-3 0.7778/0.9722; decoded joint accuracy 0.7885 |

The proposal tests contain genuine Teacher-RELEASE classes 11–15 and 21.  In
particular, a useful RELEASE classifier is learnable after conditioning on
release; the failure occurs when a single flat softmax must simultaneously
represent the coarse release decision and a fragmented 25-way proposal
distribution.

## Decision

```text
binary_release_separable = true
release_proposal_learnable = true
flat26_learnable = false
hierarchical_probe_learnable = true

ROOT_CAUSE = FLAT_26_FACTORIZATION
recommended_action_schema = V4.2_hierarchical

I0_recollection_allowed = false
I1_allowed = false
PPO_ALLOWED = false
```

The result rules out a third I0 recollection as the next action.  It is also
not authority to change the production action schema: V4.1 flat-26 remains
the active formal schema until a separately authorized V4.2 migration and
qualification are completed.

## Future V4.2 PPO provenance pre-audit

This is a semantic pre-audit only, not an implementation or PPO run.  A
hierarchical requested action must preserve one joint requested-action
log-probability:

```text
WAIT:    log P(WAIT | observation)
RELEASE: log P(RELEASE | observation)
         + log P(release_proposal | observation, RELEASE)
```

The rollout buffer and PPO ratio must use that sampled requested action and
joint log-probability only.  Teacher, execution, applied, and shield actions
must never be substituted into the ratio.  The scheduler must still receive
one decoded requested proposal and must not search the release catalogue.
This design-level provenance check is PASS; implementation qualification is
still required before PPO can ever be enabled.

## Protocol limits

The probes use one fixed diagnostic seed and the frozen cohort is internal
validation rather than an external test.  The balanced event subset is solely
a controlled diagnostic split and is not formal I0 training data.  These
limits constrain the claim to identifying the present learning bottleneck;
they do not qualify V4.2 or authorize I1, DAgger, or PPO.
