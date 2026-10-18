# Scalar induced-root reuse — local functional validation

Status: applied and locally tested on 2026-10-18. The scoped batch passed 70 tests
with one opt-in common-window integration test skipped. Common-state workload
measurements and performance qualification for this change remain pending.

The change memoizes the scalar `energy._root(vp, vn, rhs, bisections=80)`.
Its original function declaration, defaults, arithmetic, branch test, bracketing
and bisection loop remain unchanged under a decorator. `_root_uncached` exposes
the original function through `__wrapped__`; the decorator computes misses using
the original bound function. The source-based numerical policy projection still
observes the unchanged original `_root` body.

Only finite built-in `float` arguments and a positive built-in `int` iteration
count are eligible. The key contains each argument's `float.hex()` representation,
the exact iteration count and the explicit algorithm policy string
`scalar-induced-positive-branch-r0-bisection-v1`. Signed zeros remain distinct.
Other numeric types and nonfinite inputs bypass reuse and retain the original
evaluation or exception behavior. The cache stores immutable scalar floats or
`None`; it does not store a `PowerResult`, accept a failed motion input, normalize
physical arguments, or use a profile hash as a substitute for actual arguments.

The cache defaults to enabled and uses deterministic FIFO eviction, at most
32768 entries and at most 8 MiB of estimated owned bytes. The estimate includes
the current dictionary allocation, each key tuple, all immutable key fields and
each value. Shared objects are counted repeatedly, making the estimate
conservative relative to independent recursive accounting with deduplication.
Oversized entries are recomputed. Eviction only removes derived results.

`root_cache_clear()` empties entries and resets performance counters.
`root_cache_set_enabled(bool)` controls reads and insertion without changing
numerical policy; disabling retains existing entries. `root_cache_stats()`
returns detached logical-request, actual-compute, hit, eviction, entry and byte
counts together with capacities and policy. These are process-local performance
data and are not world state or checkpoint data. Worker reset can explicitly call
the clear API. A newly started interpreter begins with an empty cache.

The common-window cold observer reads these statistics when available, reporting
an absent R0 cache as absent and empty. It continues requiring the existing named
caches and never clears a warm module to manufacture cold-start evidence.

The targeted test file uses independently verified R0 source in a fresh
`-I -B` child and OPT in a separate fresh `-I -B` child. The shared query covers
all 83 slots at `0`, `300.137` and `720.94458` using actual velocity and
centripetal acceleration, full `PowerResult` fields and interval `power_bounds`,
31 varied finite scalar fixtures plus signed-zero, grazing and unavailable-branch
cases, forged identical profile hashes with distinct actual configuration data,
and cache off/on/repeat exact hexadecimal comparison. Other tests cover immutable
stored types, different numeric arguments and iteration counts, unsupported
numeric fallback, failed power inputs that never call the root, FIFO churn under
default and reduced caps, independent owned-byte accounting, reset, disable,
unchanged root AST/defaults and semantic fingerprint, and optional cold-cache
observation.

Observed local validation:

```text
venv/Scripts/python.exe -B -m pytest
  tests/test_rotating_slots_performance_root_reuse.py
  tests/test_rotating_slots_performance_energy_queries.py
  tests/test_rotating_slots_physics.py
  tests/test_rotating_slots_performance_windows.py
  -q -p no:cacheprovider
  --basetemp artifacts/rotating_slots/root_reuse_tests_20261018a
```

The process exited 0: `70 passed, 1 skipped in 14.57s` (wrapper wall 15.162s).
The skipped test is `test_real_cold_window_fixture_profile_and_benchmark_same_records`
at `tests/test_rotating_slots_performance_windows.py:305`, guarded by
`RUN_WINDOW_INTEGRATION != "1"`, with reason `requires stable implementation
fingerprint`. It is an opt-in fixture, not a platform skip. After source and
documentation stabilize, the Linux combined suite must set
`RUN_WINDOW_INTEGRATION=1` explicitly. The executed root tests confirmed the frozen original body/default
AST and semantic fingerprint, default 40000-request churn and reduced caps with
independent recursive owned-byte accounting, unsupported-input fallback,
no-root power failures, signed-zero keys, disable/reset and optional cold
observation. A profiling assertion directly observed four original scalar
function evaluations with the cache disabled and one with it enabled across four
identical requests. These are function-level observations, not workload speed.

The independent frozen R0 child used commit
`1c5afd37c891a0ed4776f64367fc7264db3c1c54`, source digest
`cd76b4c14b0a48d6d2a71ec7c63ce77010e6ea019b545807635b926eb2984891`.
All compared hexadecimal scalar roots, full power results, bounds and forged-hash
configuration cases matched R0. For the 83-slot, 249-state, 41-scalar-fixture
query, the first enabled pass recorded 805 logical requests, 528 actual original
evaluations and 277 hits. After the repeat, totals were 1610 logical requests,
528 actual evaluations and 1082 hits. Both snapshots held 528 entries,
230583 estimated bytes and zero evictions. The separate churn tests checked the
32768-entry and 8388608-byte limits; their passing assertions do not measure
whole-process RSS or establish aggregate cache-budget qualification.

The existing independent energy-query regression also retained exact semantic
and repeated-query matches: 747 stable queries, 18 descent bindings/54 segments,
30 arbitrary intervals and 42 remaining-energy times per pass. Raw stdout,
stderr, numeric exit, launch metadata, R0/OPT results and summary are retained
under `artifacts/rotating_slots/root_reuse_tests_20261018a*`.

Next, primary review and the authorized real common-state performance window
remain necessary before a workload performance claim. Full P3 qualification and
resource acceptance remain incomplete.

No changes were made to `evaluate_power`, `_power_array`, `power_bounds`,
quadrature, integration, peak-duration arithmetic, configuration or candidate
policy. The existing energy-query evidence remains applicable as a regression
target; this patch does not introduce a second energy formula or peak cache.
