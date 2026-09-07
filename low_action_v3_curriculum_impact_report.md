# Low-Level Action V3 curriculum impact assessment

This assessment is architectural, not a substitute for rerun evidence.

| Stage | Decision | Reason |
|---|---|---|
| Performance Qualification | REVALIDATE | Continuous execution authority is unchanged, but the V3 discrete candidate catalogue must be exercised through the production SafetyFilter. |
| C0 Capacity Calibration | REVALIDATE | Capacity protocol and physical profile are unchanged; verify the V3 request adapter does not alter fixed-rule calibration inputs. |
| L0 Teacher Dataset | REGENERATE | Continuous requested/teacher labels are not valid categorical BC targets. Regenerate with V3 trace schema, unless a separately audited deterministic migration is exact. |
| L1 Full-Teacher PPO | RERUN | V3 actor heads and likelihood are incompatible with continuous checkpoints. Start a fresh V3 policy after V3 L0 and safety qualification. |
| L1b | RERUN | It depends on a qualified V3 L1 policy. Do not enter before L1 pass. |
| L2+ | RERUN | It depends on the frozen low-level V3 checkpoint and must not consume a V2 policy. |

## Preconditions for the next experiment

1. Pass V3 discrete SafetyFilter/Shield integration at fixed `physics_dt=0.2`
   and `safety_sampling_dt=1.0` without increasing action authority.
2. Regenerate and audit a provenance-complete V3 L0 dataset.
3. Run only a short V3 Full-Teacher PPO preflight/pilot with a fresh actor;
   inspect categorical PPO health and requested-action provenance.

No L1b/L2 run is authorized by this report.
