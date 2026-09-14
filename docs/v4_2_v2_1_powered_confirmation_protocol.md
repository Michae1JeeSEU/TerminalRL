# V4.2 V2.1 Powered RELEASE Confirmation Protocol

## Status

```text
FROZEN_PREPARATION_ONLY
powered_confirmation_allowed = true
FORMAL_V2_1_I0_RECOLLECTION_ALLOWED = false
I1_ALLOWED = false
DAgger_ALLOWED = false
PPO_ALLOWED = false
```

This prepares a future independent confirmation only. It authorizes neither data collection nor promotion in the current round.

## Pre-registered minimum support

```text
train Teacher-RELEASE >= 100
dev Teacher-RELEASE >= 40
dev retry-active WAIT >= 40
```

Train and dev must be disjoint by seed, lifecycle event ID, and full state hash. All 25 proposal rows from one state remain in that state split.

## Natural-source stratification

- release-ready low traffic
- clear outer release
- middle/interface release
- transition/retry states
- mixed traffic
- resource pressure

Only natural scenario stratification may increase the release-ready share. Row duplication, oversampling, class weighting, synthetic labels, and fabricated certificate outcomes remain prohibited. The actor receives no catalogue masks, certificates, or scan results.

## Frozen closure thresholds

```text
accuracy >= 0.80
macro-F1 >= 0.80
WAIT recall >= 0.80
RELEASE recall >= 0.80
ROC-AUC >= 0.85
```

The future confirmation must separately report direct retry semantics, state-only closure, gate/proposal supervision, actual rejection-mode coverage, and the unchanged execution boundary. Historic V2 I0 cannot be used as a V2.1 training view.
