# Normal planning with a future execution boundary — phase 1

This is an opt-in interface implementation, `planning_effective_state_v1`. The
default synchronous callback and its full snapshot-version stale check remain
available. Phase 1 does not change candidate ranking, CP budgets, dynamics,
geometry, safety certification, energy parameters, emergency policy, or the
five-second normal planning heartbeat. No formal closed-loop comparison or
training has been run for this interface.

## Timing and scope

`PlanningTiming` freezes three absolute simulation times before work begins:

| Field | Initial default |
| --- | --- |
| `snapshot_time` | revealed state when the task starts |
| `effective_time` | `snapshot_time + 30 s` |
| `ready_deadline` | `effective_time - 1 s` |

The lead and preparation margin are independent configuration inputs. They are
not filled in from measured latency. Production work is measured with
`perf_counter`; tests inject durations while retaining real trajectory
generation, certification, resources, energy integration, and world execution.

The wrapper accepts the existing `NormalSchedulerCallback`: EnergySlack,
`rh_cp_sat`, and an advisor-equipped callback use the same future port. This
phase does not modify their search algorithms or train an advisor. For example:

```python
normal = NormalSchedulerCallback(method="rh_cp_sat")
world.decision_callback = LatencyAwareScheduler(
    normal, lead_seconds=30., preparation_margin=1.)
```

The executable prefix in this mode consists only of selected first maneuvers
whose original `start_time == effective_time` (numerical comparison tolerance
`1e-8 s`). Other suffixes and delayed maneuvers remain tentative predictions.
They are not shifted or partially extracted from an already submitted group.
An active maneuver still in progress at the boundary cannot be replaced. An
existing soft commitment at the boundary also excludes that aircraft from this
new executable prefix. The legacy synchronous prefix retains its existing
five-second window.

## Visible projection and task lifecycle

`project_visible_snapshot` consumes only the immutable revealed snapshot and
approved active/soft trajectories. It samples their existing `state(t)` and
integrates their energy portions, including known holding gaps and endpoint
continuations. It predicts known maneuver completion, versions, touchdown and
service completion. SOC is bounded conservatively downward and peak use upward
by the numerical energy allowances. Unknown/nonfinite energy and exhausted
bounds reject the projection. This function has no manifest, does not clone or
advance a world, and does not forecast unknown arrivals or emergency actions.
Already revealed upstream requests remain upstream in the projection.

The projected snapshot time is the fixed effective time. The future port buffers
the callback's certified proposal and normal-path evidence; it returns
`DEFERRED`, never an actual accepted commit. There is one normal task. Further
normal triggers set one dirty flag instead of adding tasks. A later ordinary
trigger takes a new snapshot after completion; no queued snapshots are replayed.

The finite executor lifecycle is:

1. Measure snapshot/projection/callback work and schedule its result event.
2. At delivery, use current revealed state to project the same fixed effective
   time, run the existing whole-prefix trajectory/resource/separation/energy
   certification, and prepare a world-owned transaction packet.
3. Measure preparation and the ready dependency guard. Higher-priority service
   postpones the pending result/preparation/ready delivery. Reject if readiness
   exceeds the original deadline.
   The deadline does not cancel computation for free: an overdue task stays
   pending until its measured result/preparation/ready work actually returns;
   ordinary triggers only coalesce, and an expired result starts no preparation.
   A rejected/no-prefix task transfers any unconsumed priority handoff service
   to the CPU queue so a subsequent task cannot reuse that occupied time.
4. Register a world-issued `ready_at` receipt. `PREPARED` denotes a registered
   future execution token; before its recorded `ready_time`, it does not mean
   that readiness has already occurred. No proposed resource occupancy is
   installed at this point.
5. At the effective event, require the timely readiness receipt and
   `world.time >= ready_at`, check its dependencies and state bounds, then
   atomically install and start the entire nonempty prefix. A failed member
   rejects the entire packet. No trajectory is translated, truncated, or applied
   to an individual aircraft after another member fails.

Actual successful joining means **`COMMITTED` with `submitted > 0`**. Certified
KEEP/no executable prefix is separately recorded as `NO_EXECUTABLE_PREFIX`.
Pending, prepared, empty-prefix, rejected, and committed outcomes must remain
separate when reporting success fractions. The original callback's submitted
hashes/counts are updated only by actual executor acknowledgement.

## Certification receipt and atomic publication

The packet registry is owned by the world. Public publication takes an issued
token, not a caller-constructed evaluation or arbitrary prepared calendars.
The registry records measured preparation eligibility, the ready completion
time, and its original deadline. A token without a timely ready receipt cannot
publish, including after checkpoint restoration.

Preparation precomputes natural commitment-transition signatures and the
effective p/v/a and energy bounds. The effective-time guard performs cached
field/hash/occupancy comparisons and bound checks; it does not regenerate
motion, integrate energy, search a reference, or run trajectory/separation
validation. Numerical refinement and full transaction certification occur in
measured preparation. Publication merges the affected aircraft's reservation
delta into current calendars, retains unrelated transactions and past intervals,
and performs the normal-start state/event side effects with the certified energy
forecast. It cannot silently cancel an existing own soft commitment.

This first implementation conservatively treats **all live traffic** as a
separation dependency. New admitted traffic can therefore reject a packet even
when it might be geometrically unrelated. It does not claim finely localized
dependency reuse. It checks future occupancy semantics, approved trajectory and
release identities, emergency eligibility/declaration/pending state, actual
p/v/a, energy-basis conservation, and certified SOC/peak bounds. Natural time
progression, a known active completion, known service release, calendar
target-to-occupied representation changes, and pure monitor estimate refreshes
are not independently treated as global staleness. Unrelated already-cleared
history is excluded from the traffic receipt.

Publication is an atomic executor-event abstraction. Its lightweight dependency
guard and write bookkeeping have separately measured wall/CPU costs; they are
**not** backdated as if they had delayed an action already executed. Their CPU
occupancy enters subsequent online service. The dispatch measurement includes
the actual work. This abstraction does not justify treating publication as free
or moving complex validation into the effective event.

## Shared online service and limitations

The explicit emergency mode is
`existing_priority_response_cpu_charged`. Existing monitor, emergency,
admission, and approved-soft-start work continues while normal work is pending.
Their measured service occupies the same normal computation timeline. Events,
arrivals, releases and emergency handling retain priority before a normal
delivery/publication at the same simulation time.

This interface changes normal planning only. Existing shared emergency action
timestamps and response rules are preserved; it does not implement a future
latency handoff for every control action. World integration, state export, and
rendering are not charged as normal computation latency. A finite completion
handoff charges a priority dispatch once to the next normal stage, rather than
creating an endless chain of artificial completion wakes. Ready-registration
bookkeeping can qualify a future `ready_time` without another wake solely for
renaming its state; it must still meet the fixed deadline and publication guard.

`LatencyAwareScheduler.state_dict` and the Stage2 checkpoint codec preserve the
pending event, timing, proposal, packet registry and readiness qualification.
The original generic deterministic-driver checkpoint restriction is unchanged.
As with the existing Stage2 restore API, the runner reattaches the restored
scheduler as the world's callback. Events record task start, coalescing,
preparation, packet registration, and final actual acknowledgement; dispatch
timings retain the common service measurements.

## Local bounded evidence

`tests/test_rotating_slots_planning_latency.py` exercises nine tests with real
worlds/certification, including these three small scenarios:

- Injected 10-second normal work and 2-second preparation: old holding motion
  remains continuous with decreasing SOC; no early proposed pad/soft occupancy;
  actual joint start occurs at the preselected 30-second boundary.
- During pending normal work, a fixture reveals emergency eligibility at t=5.
  The unmodified common emergency handler chooses, certifies and commits its
  pad-0 emergency trajectory. The affected normal group is rejected.
- Work exceeding the fixed ready deadline and insufficient normal reserve both
  leave the expired/uncertified normal action unexecuted.

The same file checks revealed-only projection, fixed inflight endpoints,
one-member invalidation without partial apply, natural completion/calendar
rebuild and estimate refresh, explicit-vs-legacy prefix semantics, CPU-service
deadline accounting, unissued/unready tokens, and an actual Stage2 checkpoint
JSON roundtrip containing a pending packet and wrapper state. Targeted legacy
tests cover synchronous latency, exact-plan execution, stale atomic commits,
detached certification, prefix reuse, and the existing checkpoint path.

These are implementation checks, not H/L quality or latency-benefit evidence.
Primary review is required before phase 2 or any formal server run.

Observed local checks on 2026-10-26: scoped compile returned RC0; the final nine
latency tests passed in 10.81 s, with isolated output under
`artifacts/planning_latency_tests_20261026_final`. Nine selected existing
synchronous/execution/stale/checkpoint tests passed in 2.91 s, under
`artifacts/planning_latency_compat_20261026_a`. No full-repository test suite or
formal experiment was run.

The later approved-soft ordering regression extends the same natural-transition
case: at the join boundary an old approved soft maneuver starts before a new
normal delivery. Its expected release graph is prepared from known active/soft
commitments and reservations at their critical times. Known completed supports
are normalized on both sides; unknown graph keys/hashes still reject. The
single regression passed in 2.38 s; the original nine interface tests plus nine
selected compatibility cases then passed together (18 tests, 6.97 s).

## Preregistered short H/L protocol — implemented, not launched

This section fixes the next run before its results are observed. It is one pair,
not a seed search or a performance claim. The dedicated entry point is
`scripts/stage3/run_latency_interface.py`; it does not invoke the earlier 600s
driver. The three small-world interface scenarios above precede this pair.

| Item | Fixed value |
| --- | --- |
| Physical root | original `s3r23`, seed `7324`, `medium24_rate150_v1` |
| Initial population and manifest | 24 airborne, 3 in service; frozen 141 requests over 3600 s, rate 150/hour |
| Horizon and order | H then L, each 120 simulated seconds, serial CPU12 |
| Workload | exactly two fresh child jobs; no retry, training, labels, new seeds or heldout signal measurement |
| Per-arm hard wall watchdog | 900 s, identical for H and L |
| New-run evaluation cap | 1800 s total; an unavailable full second-arm cap returns to primary, never shortens L's cap |
| Normal timing | lead 30 s, preparation margin 1 s, heartbeat 5 s |
| Planning budgets | candidate active 6 s, CP 8 s, hint 3 s, callback soft 30 s/watchdog 60 s; 32 attempts/aircraft, 1024 total |
| Boundary output | every 5 s |
| Full checkpoints | 0, 60 and 120 s; terminal checkpoint on a partial/caught failure |
| Endpoint | `settle_only_at_end=True`; dispatch/delivery exactly at 120 s is censored and remains pending |

H means heuristic candidate ordering with the `rh_cp_sat` CP-SAT backend and the
same progressive catalog interface as L. It is not a pure heuristic baseline.
L uses the existing frozen model. Both are created through
`evaluation._scheduler`, set `ranking_scope='comparable_group_v1'`, then use the
same `LatencyAwareScheduler`. The wrapper itself is passed to `world.advance`;
it is not hidden inside a lambda that would lose its event protocol.

The accepted phase-2 implementation compares candidates only within the current
aircraft/current-parent group, reuses a valid ranked batch for at most three
learned consumptions, and skips singleton model ranking. Immediate coverage uses
the explicit effective time, so effective+1 s is not immediate. The existing
five-second resource-window bucket remains unchanged. The implementation agent
reported 19 new plus 59 existing checks passed (78 total), and primary reviewed
the diff/log before this run was authorized for preparation. Those checks do not
establish an online quality or speed benefit.

The model input archive remains read-only:

- Parent: `/home/user/training_outputs/terminal/stage3_online_v2_20261025`.
- Model SHA256: `dba25b3393a5580bdc3971d9adf7d6e3cff6360404fdb08bc8fbb6f0b36e3b48`.
- Normalizer SHA256: `1e06b066dad2d657acbe16de6706b180dd4f4ed8ba1bc106cefcc2a3d4f4e7be`.
- Original t0 parent: `/home/user/training_outputs/terminal/stage3_learning_v1_0_20261023/roots/s3r23`.
- New output: `/home/user/training_outputs/terminal/stage3_latency_interface_20261026`.

Preparation copies only the model trio, original binding/source metadata and
the fixed root's required t0 inputs. It preserves the original model binding and
fingerprint byte-for-byte. `execution_scope.json` separately records the new
interface/ranking/protocol, actual execution source commit and source inventory,
CPU, driver hash and original physical input identities. New checkpoints use
the new execution source identity and save the wrapper/pending packet state;
t0 restore still uses its original cold callback and identity. No fingerprint
field is rewritten to make old weights accept a new scope.

The primary supplies `--source-freeze` from a newly frozen execution checkout;
the server does not use Git or `freeze_local`. The inventory must cover existing
`campaign.source_files()` plus `tests/test_rotating_slots_planning_latency.py`,
`tests/test_rotating_slots_latency_campaign.py`, and
`tests/test_rotating_slots_latency_ranking.py`. Source hashes, prepared inputs,
fixed protocol, copied binding, weights and normalizer are checked after
preparation and before execution. The earlier frozen96c checkout is not this
new execution checkout.

After primary review and freezing, the three stages are:

```text
python scripts/stage3/run_latency_interface.py prepare --checkout NEW_FROZEN_CHECKOUT --source-freeze PRIMARY_SOURCE_FREEZE --output NEW_OUTPUT
python scripts/stage3/run_latency_interface.py run --checkout NEW_FROZEN_CHECKOUT --output NEW_OUTPUT
python scripts/stage3/run_latency_interface.py summarize --checkout NEW_FROZEN_CHECKOUT --output NEW_OUTPUT
```

`job` is an internal child stage requiring a parent-owned running 900s watchdog
receipt. Exclusive output/start markers prevent a second run. The parent checks
CPU12 availability, preserves each RC/log/receipt, pauses its new ledger in
`finally`, and stops on any failed or partial arm. This new ledger does not reset
or modify the preceding online-v2/600s ledgers.

Normal/catchable failures save the independent observed terminal, partial
result, lifecycle/scheduler/dispatch records and authoritative replay when
possible. A hard 900s process kill cannot execute Python `finally`: the parent
retains the last genuinely completed boundary and records
`UNAVAILABLE_HARD_TIMEOUT` for an unobserved terminal/replay. It never labels the
last boundary as the state at process kill and does not add grace or retry.

Summary uses the largest boundary actually completed by both arms, plus their
independent terminal observations. Saved quality includes normal/emergency
landings, failures, pending/admitted aircraft, cohorts, energy and holding. Raw
actual scheduler records feed `aggregate_decisions`; wrapper lifecycle records
are a separate population. The join rate is committed nonempty groups divided
by finished nonempty proposed groups; pending and no-prefix counts are separate.
Raw rejection reasons accompany explicit dependency/stale and energy categories.

Full normal-planning work per task sums measured snapshot/projection/callback
work, preparation (already including ready guard), ready registration,
publication bookkeeping and normal evidence. Ready guard and service-side
publication are not added a second time. Total/P95 and component observations
are reported. All-task total work retains actual computation already performed
for pending tasks; its distribution includes those partial observations. Only
`finished_task_full_planning_work` supplies the P95 for complete finished tasks.
Full-task old-5s
overruns, original callback budget misses and ready-deadline rejection counts
remain separate. Actual dispatch wall/CPU and task-level work are also separate.
The retained `full_online_wall_s` field is a job-elapsed diagnostic including
setup and previous boundary/checkpoint/log I/O; it is not pure online planning.
`world.advance` minus dispatch is labeled integration/execution/recording
overhead, not pure physics or planning delay. Terminal/job/receipt clocks retain
their different replay-export and process-start boundaries.

No outcome or formal run has been produced for this protocol by this task.
The runner's local checks use small JSON/process-receipt fixtures only, with no
real closed-loop rollout, solver, model load or inference.
Compile/help returned RC0. The six runner fixture tests passed in 0.43 s
(`artifacts/latency_campaign_tests_20261026_summary`), covering immutable model
binding/input preparation, exact pair/caps, wrapper-preserving boundary and
checkpoint serialization, caught/kill timeout evidence, pending-aware rates,
and saved-output common-boundary/README summarization.
