# Predicate recipe validation

## Experiment evidence

- **Status / evidence type:** newly run
- **Command:** `python -m terminal_marl.training.run_v4_2_v2_1_predicate_release_reachability --output-dir ~/training_outputs/terminal/v4_2_v2_1_predicate_reachability_qualification_20260913 --max-seeds 10`
- **Environment:** `~/envs/terminal_env`; relevant test suite: 20 passed.
- **Output manifest:** `~/training_outputs/terminal/v4_2_v2_1_predicate_reachability_qualification_20260913/v4_2_v2_1_predicate_release_reachability_manifest.json`, SHA256 `ffd1e74968101c20984bccb341ac85d4531f35a9e4f768a8f290783d5bfbadfd`.
- **Formal-I0 rows created:** 0.

Each of the four recipes produced 10 seed- and lifecycle-independent, predicate-exact witnesses. ActionTape replay was exact for all 40 witness episodes. For every family, collision, NMAC, hard-separation violation, `SafetyInvariantError`, and uncertified execution were all zero.

| Gate | Result |
| --- | --- |
| `FAMILY_CLASSIFIER` | PASS |
| `TRANSITION_LEDGER` | PASS |
| `PREDICATE_RECIPE_VALIDATION` | PASS |
| Dynamic ActionTape replay | PASS (40/40) |
| Teacher determinism | PASS |
| Retry provenance | PASS |
| Safety | PASS |

## Protocol risks

This is a bounded internal simulator qualification, not a claim of global airspace reachability. It uses 10 independent seeds per family, no learning split, no threshold selection, and no sample reweighting. The conclusion is limited to the declared legal recipes and their saved reset provenance.
