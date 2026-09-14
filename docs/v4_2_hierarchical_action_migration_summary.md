# V4.2 Hierarchical Action Migration

V4.2 replaces only the V4.1 flat-26 actor factorization.  It preserves all
26 executable scheduling intents and retains the existing execution,
trajectory, SafetyFilter, and PredictiveSafetyShield authority.

| Item | Result |
|---|---|
| Old schema | `V4_1_flat26` |
| New schema | `low_action_schema_v4_2_hierarchical` |
| Execution actions | 26 |
| Gate actions | 2 (`WAIT`, `RELEASE`) |
| Conditional RELEASE proposals | 25 |
| WAIT decode | `[WAIT, 0, 0]` |
| RELEASE decode | V4.1 joint index `proposal + 1` exactly |

The shared Observation V2 encoder supplies a two-way gate head and a 25-way
proposal head.  A WAIT never samples a policy proposal. A RELEASE always has
one proposal. The Teacher remains untouched: V4.1 joint label 0 maps to
`(gate=WAIT, proposal=null)` and labels 1–25 map to
`(gate=RELEASE, proposal=joint-1)`.

The V4.1 I0-R source data was retained read-only.  A derived V4.2 target view
was written with 1,500 train and 243 frozen-cohort rows.  It adds only
`gate_target` and nullable `proposal_target`; Observation, semantic Teacher
label, state snapshot/hash, source, seed, and lineage were verified unchanged.
Its train/cohort hashes are respectively
`681119824a739c6ec08b7900d821e4969eadfa85a9a9a307d0e5f618dfcf61fe` and
`d2c7404c33b6f9ffa886798d5da8149fe00e85eb343410bbe4addf2b4edab0f5`.

V4/V4.1 checkpoints and ActionTapes remain historical/read-only and are not
silently migrated. V4.2 formal learning requires a fresh actor.
