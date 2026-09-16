# V4.2 / Observation V2.1 Closure-Protocol Revision

## Frozen protocol

The closure protocol distinguishes deployable state from Teacher-internal candidate computation. This is a protocol revision only: it does not change the V2.1 observation, Teacher labels, Safety authority, action factorization, or runtime execution boundary.

### State-only closure

The state-only set is restricted to current-time information available to a deployed actor and directly represented or reliably identified by the 88D observation: retry remaining, self committed state, phase/ring context, interface frozen, final capacity, and LZ availability.

Each effective state-only predicate requires code-path inventory closure, no unresolved missing feature, and seed/lifecycle/state-hash-disjoint matched-pair separability when an actual pair is available. An LZ-unavailable branch that cannot naturally occur before the preceding final-capacity gate is recorded as `NOT_TRIGGERED_BEFORE_FINAL_CAPACITY_IN_DOMAIN`; no infeasible state may be manufactured to force it.

### Action-conditioned closure

Proposal reservation/headway feasibility, trajectory-dynamics feasibility, SafetyFilter and holder-tube certificates, and Teacher preference among certified proposals remain outside actor observation. Their admissible diagnostic targets are `(obs88, proposal identity) → proposal outcome` and `obs88 → Teacher-induced gate/proposal target`. Actor inputs may not include a certificate, `can_reserve` mask, or 25-proposal scan output.

### Pre-registered criteria

| Component | PASS requirement |
|---|---|
| State-only matched pair | accuracy, macro-F1, WAIT recall, RELEASE recall ≥ 0.80; AUC ≥ 0.85 |
| Gate probe A | same five state-only thresholds |
| Proposal certificate probe B | macro-F1 ≥ 0.80 and AUC ≥ 0.85; fixed-0.5 recalls reported, never threshold-tuned |
| Proposal selection probe C | top-1 ≥ 0.60 and top-3 ≥ 0.80 |
| Deployment boundary | one decoded request only; ExecutionScheduler and SafetyFilter authoritative; no scan/certificate exposure |

Final freeze requires State-only closure PASS, Teacher-supervision closure PASS, schema/replay PASS, no future leakage, and no requested observation or action-factorization revision. A failure is not repaired retrospectively by changing these thresholds.
