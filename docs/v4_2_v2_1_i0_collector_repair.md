# V4.2 / Observation V2.1 I0 Collector Repair

## Scope and evidence

This is a `newly_run` collector-only repair. The prior 1,800/300 V2.1 I0
artifact remains `READ_ONLY_FAILED_DIAGNOSTIC_ARTIFACT`; it was neither
appended, filtered, nor used as a training source. No Teacher, Observation,
Safety, loss, reward, I1, DAgger, or PPO behavior changed.

## Dynamic-airspace replay provenance

The four historic replay failures were seed 51004 and 61004
(`transition_retry_wait`), and seed 51007 and 61007 (`interface_headway`).
Their old tape headers omitted dynamic-airspace state. For each, the audit
found five dynamic-airspace field differences and one event-scheduler
difference between generic reset and the collector's pre-decision setup.

The collector now records a deterministic `reset_recipe` with its hash,
base-reset state hash, dynamic-airspace state hash, arrival RNG-state hash,
the declared high-level geometry action, its gate, and its setup result. The
ActionTape state now includes actual/target ring geometry, speed bias,
headways, occupancy/committed counts, transition start/end and affected
interfaces, scheduler executed-release timing, landing-zone state, event
queue, and arrival generator state. Replay applies the recipe after seed reset
before the first low-level observation and fails closed on an unknown recipe.

| Dynamic source family | Independent tapes | Exact tapes |
| --- | ---: | ---: |
| `transition_retry_wait` | 10 | 10 |
| `interface_headway` | 10 | 10 |

```text
DYNAMIC_SOURCE_REPLAY_CLOSURE = PASS
exact_replay_rate = 1.0
PRELIGHT_ACTIONTAPE_EXACT_REPLAY = PASS
```

Every verified tape matched base reset, recipe hash, post-recipe reset state,
pre-step observation, requested category, Teacher target, executed action,
and post-state hash.

## Natural RELEASE coverage finding

The new-seed preflight used scenario-level source strata and semantic event
retention. It did not choose rows by Teacher label, duplicate rows, or alter
Teacher ranking. It retained 601 unique decision events from 1,938 raw
decision evaluations: 492 WAIT and 109 RELEASE events, with 109 unique
RELEASE lifecycles.

| RELEASE family | Preflight target | Independent lifecycles | Result |
| --- | ---: | ---: | --- |
| clear outer | 15 | 0 | fail |
| middle/interface | 15 | 8 | fail |
| post-conflict | 10 | 6 | fail |
| post-WAIT | 10 | 0 | fail |
| other valid RELEASE | — | 83 | diagnostic only |

The natural Teacher support observed was joint classes `[11, 12, 13, 14]`,
speed values `[-0.5, 0.5]`, and merge values `[-1.0, 0.5, 1.0]`. This is
evidence of the current scenario recipes' reachable support, not grounds to
lower the preregistered 8-joint-class / 4-speed / 5-merge gate. A separate
reachable-action support audit is required before any proposed gate change.

## Protocol decision

The replay blocker is repaired; the remaining blocker is scenario-manifold
coverage. The next collector change must target legal initial spacing,
occupancy, relative progress, headway age, reservation context, arrival
timing, lifecycle phase, and recovery state to produce the missing families
naturally. It must not collect “seven more releases,” copy rows, balance
classes, or relabel. Formal recollection remains blocked.
