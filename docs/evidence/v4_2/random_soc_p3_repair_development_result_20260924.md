# Random-SOC P3 Repair Development Results

Date: 2026-09-24
Protocol: [registered development comparison](random_soc_p3_repair_development_20260924.md)

Status: seven arms completed as development evidence. This is not P3 PASS,
qualification, training admission, PPO admission, or P4 authorization.
`P3_QUALIFIED=false`, `PPO_ALLOWED=false`, and `P4_ALLOWED=false` remain.

## Raw output roots

Server:

```text
/home/user/training_outputs/terminal/random_soc_p3_repair_development_20260924/
```

Local synchronized mirror:

```text
D:/7_codespace/training_outputs/terminal/random_soc_p3_repair_development_20260924/
```

The frozen checkpoint and runtime protocol are unchanged. The seven arms use
the paired P3 scenarios and seeds `844101`, `844102`, and `844103`; the
two-step `844000` dense24 smoke was the preflight. No additional seed or
protocol was introduced.

## Arm results

| scenario | seed | controller | sim s | wall s | normal / emergency / failed-energy | unresolved | terminal | min actual / swept m |
|---|---:|---|---:|---:|---:|---:|---|---:|
| flow6 | 844101 | trained | 556 | 1.228605 | 6 / 0 / 0 | 0 | true | — |
| flow6 | 844101 | nominal_rule | 587 | 1.080217 | 6 / 0 / 0 | 0 | true | — |
| flow12 | 844102 | trained | 1082 | 3.875294 | 11 / 1 / 0 | 0 | true | — |
| flow12 | 844102 | nominal_rule | 1094 | 3.190011 | 12 / 0 / 0 | 0 | true | — |
| dense24 | 844103 | trained | 790 | 26.679016 | 19 / 5 / 0 | 0 | true | 80.049642 / 80.049642 |
| dense24 | 844103 | nominal_rule | 681 | 19.403243 | 19 / 5 / 0 | 0 | true | 80.004074 / 80.000000 |
| dense24 | 844103 | trained_release_only | 689 | 31.085010 | 22 / 2 / 0 | 0 | true | 81.599148 / 81.367757 |

All seven full arms had exit status `0`. The `844000` two-step smoke
completed its server/local preflight with `14` tests passing; it is a smoke
record and is not a full-horizon arm.

Across all seven full arms: `SafetyInvariantError=0`, collision/NMAC/
hard-separation/fail-closed/hash-mismatch counters were zero, weights were
unchanged, Teacher was unused, and PPO was not called. No unreported
intervention counter is converted to zero here.

## Interpretation boundary and current review state

The current review prioritizes a gate-learning repair. The dense24
`trained_release_only` arm is a gate intervention that retained neural
proposal Top3; its two emergency landings are not a trained-policy result.
The trained arm still had five dense24 emergency landings, and the paired
flow12 trained arm had one. Remaining emergency cases are under inspection.
The V4.1 joint Teacher labeler and K3 execution rules for final-pad search and
emergency-pad priority are now aligned; 12 targeted tests pass. No new data
has been collected, and no training or qualification validation has started.

The strict P3 `no_failed_outcomes` check requires every ledger outcome to be
`completed`; `emergency_landed` therefore does not pass that gate even when it
is counted in the service-completed total. Conditional wait statistics do not
override that outcome gate. Safety thresholds, energy floor/reserve, and all
P3 completion gates remain unchanged.

## Additional raw checks

An extra combined legacy P3 check recorded `29 passed, 2 failed`. The raw
record retains the `rewards.py` reward-source hash and `formal PASS=false`;
the old binding was not repaired, and this is not a full-suite PASS.

The local implementation/document delivery is commit `4abacee`; the
repository has no remote and was not pushed. Existing formal/heldout records,
including `829001–829006`, remain untouched.
