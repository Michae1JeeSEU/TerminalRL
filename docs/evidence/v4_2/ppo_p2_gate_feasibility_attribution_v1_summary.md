# P2 Flow12 Feasibility Attribution V1

> Date: 2026-09-19  
> Evidence type: `newly_run`  
> Status: `DIAGNOSTIC_COMPLETE=true`  
> Authorization: diagnostic only; support training, repair, PPO, P3, curriculum, and checkpoint promotion remain false

## Question and protocol

Effectiveness V1 failed only the waiting-gap closure gate (`46.01% < 50%`) and localized most
regression to one dense flow12 seed. Attribution V1 therefore asked whether late flow12 delay was
primarily a remaining gate disagreement or a conditional RELEASE proposal that could not pass the
execution scheduler/SafetyFilter.

The repaired candidate was deterministic and execution-authoritative on six fresh flow12 seeds
`789001–789006`. Teacher ran only as a label sidecar. No optimizer, update, training, or checkpoint
promotion path was present. Aggregation is by actor decision event, split by interface and by
arrival half (`0–5` versus `6–11`).

Protocol SHA256:
`50b2cb7dc40160a54ecfe150122d7bcc3407b638a40e887b20045120e220b1fd`.

## Validity and lifecycle results

All six episodes completed 12/12 requests (72/72 total), with 1,050 actor rows. Every episode passed
safety, exact action-tape replay, exact requested-likelihood, geometry, Teacher label-only, student
authority, and no-training checks. Reverse-transfer samples and safety counters were zero; maximum
inner excursion was `0.01559–0.01922 m`.

Per-seed mean waiting ranged from `7.3629 s` to `23.5989 s`; mean request-to-touchdown ranged from
`324.6667 s` to `348.5 s`. These values describe the diagnostic cohort and are not a replacement
effectiveness test.

## Attribution

| Slice | Actor events | False RELEASE | False WAIT | Proposal match / mismatch when both RELEASE | Accepted / rejected RELEASE |
|---|---:|---:|---:|---:|---:|
| Overall | 1,050 | 23 | 43 | 116 / 166 | 216 / 89 |
| Early arrivals 0–5 | 240 | 1 | 3 | 64 / 58 | 108 / 15 |
| Late arrivals 6–11 | 810 | 22 | 40 | 52 / 108 | 108 / 74 |
| o2m | 609 | 8 | 21 | 54 / 73 | 72 / 63 |
| m2i | 202 | 9 | 22 | 7 / 76 | 72 / 20 |
| i2f | 239 | 6 | 0 | 55 / 17 | 72 / 6 |

Of 89 rejected RELEASE events, 66 occurred where Teacher also selected RELEASE but the student
selected a different proposal. All 66 Teacher-RELEASE rejection events were proposal mismatches;
none was a proposal match. The remaining 23 rejects corresponded exactly to false-RELEASE gate
events. By interface, the Teacher-RELEASE/proposal-mismatch rejects were 55 at o2m, 11 at m2i, and
zero at i2f. The late half contained 74/89 total rejects and 52/66 of the proposal-mismatch rejects.
Primary blockers were predicted separation (72) and no merge gap (17); 89 retry events accumulated
294 retries.

## Interpretation and boundary

The fresh evidence indicates that another gate-only repair is not the highest-value next step.
Gate disagreement remains measurable, but most rejected RELEASE events—and especially the late
o2m/m2i tail—occur when both policies agree on RELEASE and disagree on the conditional proposal.
The next experiment should therefore collect a fresh, split-disjoint flow12 conditional-proposal
support cohort and run an offline learnability audit before any proposal-head update.

This diagnostic does not itself authorize collection rows for training. Its 1,050 rows and all
effectiveness rows remain excluded from training/tuning. The six seeds are consumed and may not be
rerun. Any proposal support, repair, qualification, and effectiveness campaign must use new seeds
and a separately frozen protocol.

Formal manifest/rows/log/marker SHA256:

- `f2382b1a8c5ee486c4246e716fc27b97a34d2948bfe90633e33d0f428a56f35c`
- `b07f51b0354d92943864232b7698a0d53061de22e6593911b10e40a8192c412f`
- `371fd775e823fe347a593fec4ba79ca03217bdfb44ffc38ddbceb382bcbc5161`
- `89e7af446d312c06019d88f2c4b1e8b4424bde8711dae4cba9294550d1ddfa2e`

