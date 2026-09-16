# V4.2 V2.1 Independent Confirmatory Closure Audit

## Experiment evidence

- Status / evidence type: `newly_run`.
- Command: `python -m terminal_marl.training.run_v4_2_v2_1_confirmatory_closure_audit --output-dir <artifact-dir> --target-states 600`.
- Environment: `~/projects/terminal`, `~/envs/terminal_env/bin/python` (Python 3.12.3).
- Output manifest SHA-256: `98e91781e9d501cb37c06986e6734a3bcd17fe767315a5d0fea0fd2456664d56`.

The new bank contains 600 unique semantic states and 600 unique full-state hashes from 62 new seeds (`14001`–`14062`), not the earlier 300-state bank. Sources are balanced across independent interface transition (157), nominal mixed traffic (149), release-ready low traffic (147), and resource pressure (147). The fixed split contains 483 train / 117 dev states from 50 / 12 seeds; seed, lifecycle and state hashes are verified disjoint, and every state's 25 proposals remain on one split.

## State-only evidence

Retry active/inactive states: 265 / 335; committed/uncommitted: 90 / 510; interface frozen/open: 157 / 443; final-capacity blocks: 19. LZ-unavailable was not naturally triggered: the current domain reaches the earlier final-capacity gate first, while LZ availability itself is observed in all 600 states. It is `NOT_TRIGGERED_BEFORE_FINAL_CAPACITY_IN_DOMAIN`, not fabricated.

| Matched-pair predicate | Train/dev pairs | Result |
|---|---:|---|
| retry active vs RELEASE | 25 / 6 | FAIL: accuracy 0.6667, macro-F1 0.6571, WAIT recall 0.5000, RELEASE recall 0.8333, AUC 0.6944 |
| committed vs RELEASE | 25 / 6 | PASS: all metrics 1.0000 |
| interface frozen vs RELEASE | 25 / 6 | PASS: all metrics 1.0000 |
| final capacity vs RELEASE | 15 / 4 | PASS: all metrics 1.0000 |

No state-only feature is unresolved, but the pre-registered retry matched-pair gate fails. Therefore `STATE_ONLY_CLOSURE = FAIL`.

## Teacher-supervision evidence

| Probe | Result |
|---|---|
| A: `obs88 → any certified RELEASE exists` | FAIL: accuracy 0.9744, macro-F1 0.8569, WAIT recall 0.9910, RELEASE recall 0.6667, AUC 0.9635. The frozen RELEASE-recall requirement is 0.80. |
| B: `(obs88, speed/merge identity) → proposal certified` | PASS: macro-F1 0.8229, AUC 0.9927, certified recall 0.5469, rejected recall 0.9971 at fixed 0.5. |
| C: `obs88 → Teacher proposal`, RELEASE-only | PASS: top-1 0.8333, top-3 0.8889, speed agreement 0.8333, merge agreement 1.0000. |

Probe B contains 275/12075 certified train candidates and 64/2925 certified dev candidates. Its fixed-threshold recall is reported as an imbalance limitation; no class weighting, oversampling, or threshold selection was used. Observed rejection strata are holder-tube (273 train / 89 dev; rejected recall 1.0000). Reservation, dynamics, and generic SafetyFilter rejection are `NOT_OBSERVED_IN_DOMAIN` (0 rows each), not substituted with artificial cases.

Because Probe A fails its frozen RELEASE-recall criterion, `TEACHER_SUPERVISION_CLOSURE = FAIL`. The Teacher's internal scan is not given to the actor, but the required three-way closure is not jointly established, so `TEACHER_PRIVILEGED_INFORMATION_RISK = UNRESOLVED` for qualification.

## Safety and provenance boundaries

No formal I0 collection, I1, DAgger, PPO, Teacher/Safety/observation/action modification, class weighting, oversampling, or certificate exposure occurred. V2.1 schema checks pass: 87D prefix identity, retry index 87, finite `[0,1]`, determinism, no future leakage, and exact ActionTape replay. Old V2 checkpoints remain non-resumable for V2.1.

## Protocol risks

This is an internal simulator confirmation, not a deployment generalization claim. The dev split has only six RELEASE states, so release-recall uncertainty is material; results are nevertheless binding because thresholds were frozen before this run. No separate rejection-mode claim is made for categories with zero natural observations.
