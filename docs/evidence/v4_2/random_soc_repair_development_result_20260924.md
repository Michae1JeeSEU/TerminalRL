# Random-SOC Repair Development Comparison Result

Date: 2026-09-24
Protocol: [`random_soc_repair_development_20260924.md`](random_soc_repair_development_20260924.md)
Status: completed as a development comparison only. This is not I1/P3
qualification, PPO admission, or a repair PASS. `P3_QUALIFIED=false`,
`PPO_ALLOWED=false`, and `P4_ALLOWED=false` remain in force.

Raw output roots:

- Server: `/home/user/training_outputs/terminal/random_soc_repair_development_20260924/`
- Local Syncthing mirror: `D:/7_codespace/training_outputs/terminal/random_soc_repair_development_20260924/`

## Preflight and smoke

The first pytest invocation used the server home directory and produced `9
passed, 1 failed` because the protocol loader resolved a repository-relative
path. With the repository working directory set to
`/home/user/projects/terminal`, the required test passed: `10 passed`, exit
`0`. The source files and protocol were then SHA256-matched between local and
server before the paired runs. The `843000` trained smoke completed 2 steps,
60 actor rows, and 59 actor decisions (`47 WAIT / 12 RELEASE`). All 6 shadow
probes preserved the original state; one positive witness was `UID27 WAIT →
nominal RELEASE`, same-step `approved`, and the other 5 probes were negative.

## Paired seed `843101`

| arm | terminal / steps | wall s | normal / emergency / failed-energy / unresolved | WAIT / RELEASE | approved / rejected | min actual / swept m | policy / weights | exit |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| trained | true / 1036 | 66.700964 | 9 / 18 / 3 / 0 | 12861 / 858 | 29 / 829 | 85.970293 / 85.970293 | neural / unchanged | unknown |
| nominal_rule | true / 865 | 40.186961 | 17 / 13 / 0 / 0 | 0 / 7795 | 48 / 7747 | 85.788899 / 85.788899 | no neural / unchanged | 0 |

Both arms recorded zero collisions, NMAC, hard-separation violations, and
fail-closed events. Safety interventions and joint safety corrections were
recorded separately: trained `481 / 30`, nominal `213 / 0`; this is not an
“all safety counters zero” claim. Both arms had `safety_error=null`, no Teacher,
no training/PPO calls, and `PPO_ALLOWED=false`.

The trained arm retained 19 probes, all with `original_state_unchanged=true`,
and 4 approved witnesses: inner-ring WAIT at `t=60`, inner-ring WAIT at
`t=120`, and inner-ring WAIT plus RELEASE at `t=240`. The remaining probes were
`safety_reject` or `safety_wait`.

The paired records show different controller behavior and include existing
same-state safety-feasible WAIT counterexamples. Since the comparison changes
both the requested gate and proposal between arms, it cannot isolate the full
difference to the gate alone. The trained arm still records 13 emergency
landings and 3 failed-energy outcomes. These are observations for the next
repair study, not a cause attribution or a qualification decision.

## Next authorized boundary

The next development work is preparing fresh P3 `flow6`/`flow12`/`dense24`
runs, plus a dense24 forced-gate, frozen-proposal ablation. The code and any
new protocol remain pending; no training or qualification is authorized by
this comparison. Seed `842102` remains unused, and heldout seeds
`829001–829006` remain untouched.
