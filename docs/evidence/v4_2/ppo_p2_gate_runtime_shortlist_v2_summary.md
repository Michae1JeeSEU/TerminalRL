# P2 Gate Runtime Shortlist Evaluation V2 Summary

> Date: 2026-09-20
>
> Terminal decision: `RUNTIME_SHORTLIST_FREEZE = true`
>
> Downstream authorization: `PPO_ALLOWED = false`; `checkpoint_promotable = false`

## Scope and provenance

This fresh paired evaluation qualifies only the deployment-time bounded shortlist boundary for the
already frozen V4.2 / Observation V2.1 candidate. It performs no training, label collection,
Teacher substitution, checkpoint update, delayed repair, or 25-proposal catalogue scan.

- protocol: `V4_2_P2_GATE_RUNTIME_SHORTLIST_EVALUATION_V2`
- protocol SHA256: `67e61165ef5c056d1b1ebd0a789cafd17f5e0554d07b29128617df3aa8bf6d5d`
- candidate checkpoint SHA256: `4b8be66c100aaaeb5b5699dcce68a96dbe1237964e4c308e7b23b95c62fb8c8c`
- formal seeds: `806001–806006`, fresh and reserved before first reset
- controllers: candidate K=1, candidate K=3, nominal rule
- scenario: six paired flow12 episodes per controller
- manifest SHA256: `6050088d3c28d1531c7e8dde5b828403390bfdeb51dde8bde54e0f2759323905`
- server artifact: `/home/user/training_outputs/terminal/p2_gate_runtime_shortlist_v2_formal_806001_806006`

The predecessor V1 consumed seeds `805001–805006` and is retained as `INVALID_EVALUATOR`.
Its K=3 runtime completed safely and replayed exactly, but the historical likelihood auditor replayed
only rank-1 actions and omitted recorded runtime shortlists. The corrected auditor returns `true` on
the immutable V1 fallback tape. No V1 row or metric entered V2.

## Hard-gate result

All 18 records completed 12/12 requests. Across all controllers:

- safety counters were zero;
- exact environment/trajectory replay passed;
- requested-action likelihood replay passed for both learned controllers;
- no Teacher, labels, optimizer update, reverse transfer, or uncertified execution occurred;
- every K=3 shortlist was request-first, unique, policy-provided, and bounded to three candidates;
- every episode gate passed.

K=3 exercised the changed boundary in every seed:

| Seed | Shortlists supplied | Max checked | Fallback selected | Exhausted |
|---:|---:|---:|---:|---:|
| 806001 | 49 | 3 | 3 | 7 |
| 806002 | 47 | 3 | 5 | 4 |
| 806003 | 39 | 3 | 3 | 2 |
| 806004 | 38 | 3 | 4 | 1 |
| 806005 | 41 | 3 | 3 | 4 |
| 806006 | 40 | 3 | 4 | 2 |

## Paired effectiveness

| Controller | Completion | Weighted wait (s) | Request-to-touchdown (s) |
|---|---:|---:|---:|
| Candidate K=1 | 72/72 | 20.822371 | 327.583333 |
| Candidate K=3 | 72/72 | 11.303291 | 325.527778 |
| Rule | 72/72 | 19.883828 | 328.652778 |

K=3 minus K=1:

- weighted wait: `-9.519080 s`;
- request-to-touchdown: `-2.055556 s`;
- shield actor fraction: `0.0`.

The preregistered non-worse gates pass for both effectiveness metrics, at least one metric is strictly
better, fallback use is nonzero, and every runtime/safety gate passes. Therefore the bounded K=3
runtime boundary is frozen. This result does not by itself authorize PPO, curriculum promotion, or
checkpoint promotion; those require a separate admission review under the frozen boundary.
