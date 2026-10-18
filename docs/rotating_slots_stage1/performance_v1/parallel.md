# R0 reference-pad execution adapter

`terminal_marl/core/rotating_slots/performance/parallel.py` implements the six
independent pad bodies of R0 `_reference`. It does not replace
`_update_emergency`: each original query is reached at its original position in
that loop. A cache hit runs the original hit branch; a miss submits exactly six
pads, collects in pad order, applies the original `min` tie rule, and uses the
original exact-cache store. A currently occupied pad is not filtered.

The scoped `install_parallel_reference(world, pool)` context patches the class
method with an instance identity guard and restores it on exit. It adds no
fields to the authoritative world, so checkpoint field inventories do not
contain executor handles. Other world instances use the saved original method.
No worker receives parent aircraft, requests, calendars, active plans or events.
Only immutable query/configuration/identity input crosses the boundary.

Construct `ReferencePool` with an explicit `config`,
`expected_core_root` pointing to the frozen `rotating_slots` directory,
`expected_core_manifest` mapping every immediate core `.py` filename to its
SHA256, and `semantic_identity` containing `policy_id`,
`effective_rules_hash`, `candidate_policy_hash`, `numerical_policy_hash`.
The policy must be `upstream_capacity_only_pad_arrival_v1_20261018`.
Also supply `process_budget`, `core_budget`, and
`exclusive_process_accounting=True`. Worker counts are integers from 1 to 6.
A pool with one worker includes IPC and is not the serial baseline.

Use the pool context outside the binding context:

```python
with ReferencePool(4, config=world.config,
                   expected_core_root=frozen_core,
                   expected_core_manifest=frozen_core_hashes,
                   semantic_identity=verified_semantic_identity,
                   process_budget=5, core_budget=4,
                   exclusive_process_accounting=True) as pool:
    with install_parallel_reference(world, pool):
        world.advance(original_target)
accounting = pool.summary()
```

This is an API example, not a qualification-run command. A frozen runner must
bootstrap its R0 imports before importing core modules. The helper directory
can be supplied externally through the frozen rotating-slots package path.
Spawn imports this same helper file under the checked flat name `parallel`;
the helper has no top-level core imports. Initializers verify the byte manifest
and actual imported core roots, then independently reload and check the frozen
configuration/profile identity. They reject an already imported different core;
they never overwrite `sys.modules` to disguise a mixed implementation.

Windows uses `spawn`. Linux uses `fork` to avoid Python spawn's additional
resource-tracker process and preserve the strict one-parent plus six-worker
limit. Linux startup requires exactly one active Python thread and no existing
multiprocessing children. Create the pool before heartbeat/sampler threads.
Workers start eagerly through bounded ready tasks; the RSS sampler starts only
after processes have been created. This Linux choice must be recorded as the
actual execution backend; Windows spawn tests do not establish Linux evidence.

The parent and workers constrain native libraries to one thread with
`threadpoolctl` and environment limits. Linux requires `threadpoolctl` because
fork inherits already loaded libraries. The limits are restored in the parent
after shutdown. `process_budget` must cover parent plus workers; `core_budget`
must cover the worker allocation (the parent shares that allocation).
Separate jobs still require an externally planned aggregate allocation.

`benchmark` is the default and does not construct cProfile or return function
tables. `profile` requires an explicit directory and a finite
`max_profile_tasks` (default six, maximum 1000). It writes raw per-pad `.prof`
files using `time.process_time` as the profiler timer and refuses to overwrite
existing files or exceed the task bound.
Profile timings cannot enter uninstrumented speed comparisons.

Template construction and fallback/instantiation counts report actual worker
work. Worker-local template/cache histories may change work counts, never
canonical representatives or output. Canonical durations return to the parent
table, actual counts are aggregated into the existing reference counts, and the
last fallback diagnostic comes from the last fallback pad in original order.
The existing `world.cpu` fields cover parent work only. Use the separate
performance resource summary for full pool accounting; do not add nested world
timers or completed-task durations and call that process-tree CPU.

`process_metrics.py` begins its timing before pool startup and finalizes after
join and native-limit restoration. Under the explicit exclusive-child context,
Linux uses the `RUSAGE_CHILDREN` delta after all owned workers are joined.
Windows retains OS handles while workers are alive and reads lifetime
`GetProcessTimes` after exit. Both include initialization, IPC, serialization,
failed/partial work and shutdown. Unsupported or failed OS accounting leaves
CPU unavailable/incomplete; no task-only substitute is used. The exclusive
declaration requires callers to avoid unrelated subprocess creation/reaping
during the measurement, because Linux child accounting is process-wide.

RSS sweeps read the parent and all registered live worker RSS in the same
sweep, recording per-PID values, elapsed time and sweep duration. The default
interval is 0.1 wall seconds (configurable from 0.01 to 5). Only 512 recent
samples, the sample count and cumulative sampled maximum are retained in memory.
An optional `rss_series_path` writes the full series to a new append-only JSONL
file. Registering already known processes does not add extra query-rate samples.
The reported sampled
simultaneous maximum is a lower bound: reads within a sweep are sequential,
transient peaks between sweeps can be missed, and no finite upper error bound
is asserted. It is never a sum of per-process lifetime peaks. OS read failures
and exited/inaccessible PIDs remain explicit in the samples.

The separate `sum_lifetime_peak_upper_bound_bytes` is conservatively above
simultaneous RSS and is explicitly not a simultaneous measurement. Linux uses
the parent's OS lifetime high-water mark plus worker count times the maximum
reaped-child high-water mark (past children can make this loose). Windows reads
exited workers' retained OS handles where the OS provides peak working sets;
otherwise the upper bound is unavailable. An upper bound below the memory limit
can prove compliance; a sampled lower bound above it proves failure. A sampled
lower bound below the limit with no sufficiently low upper bound is unresolved
and must not pass the memory gate.

Exceptions, process death, serialization failures and query/source/profile
mismatches propagate as implementation failures. No result is changed to
`None`, `NOT_FOUND`, or a physical failure. Failure and cancellation terminate
only processes owned by this executor, join them, and restore the binding.
Normal close joins all workers as well. No global process-name termination is
used.

The bounded test file is `tests/test_rotating_slots_performance_parallel.py`.
It checks real direct-query payload/qE and emergency-loop equivalence for
1/2/4/6 workers, ties and out-of-order completion, cache hits/off/restarts,
parent mutation isolation, identity/error/death propagation, profile bounds,
benchmark profiler exclusion, and normal/error/cancel lifecycle. It does not
claim P4 qualification, six-window performance, full-traffic equivalence, or
P5/P6 acceptance. Linux execution and source-isolated runner evidence must be
collected separately from the local Windows tests.
