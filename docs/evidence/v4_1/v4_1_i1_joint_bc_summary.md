# V4.1 I1 Fresh Joint-BC Qualification — FAIL

Fresh `Categorical(26)` actor; CE-only training; no V4-30 checkpoint,
PPO update, DAgger, reward, Observation, or safety change.

| Metric | Before | After train | Frozen cohort |
|---|---:|---:|---:|
| Joint CE | 3.2752 | 0.1647 | 0.7737 |
| Top-1 | 0.0150 | 0.9728 | 0.8450 |
| WAIT recall | 0.0000 | 1.0000 | 1.0000 |
| RELEASE recall | 1.0000 | 0.0000 | 0.0000 |
| Release precision | 0.0272 | 0.0000 | 0.0000 |
| Release macro-F1 | 0.0265 | 0.4931 | 0.4580 |
| Entropy | 3.2518 | 0.1984 | 0.2137 |

Gradient norm was finite (mean 1.1471, max 7.0119), and CE decreased.
However, the frozen cohort prediction is all-WAIT: 200 WAIT / 0 RELEASE and
only joint class `0`. It therefore fails RELEASE recall, multiple-RELEASE
class learning, and the no-all-WAIT collapse gate.

```text
I1_JOINT_BC = FAIL
I2_DAGGER_PREFLIGHT_ALLOWED = false
PPO_ALLOWED = false
```

No class weighting, copied RELEASE data, relabelling, or DAgger round was
used to alter this result. Any subsequent I0 revision must add physically
meaningful RELEASE-expert state sources and recollect rather than balance
this dataset artificially.
