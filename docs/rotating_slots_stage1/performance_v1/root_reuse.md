# Scalar induced-root reuse — local functional validation

Status: root reuse is applied. The current revision uses OrderedDict FIFO with
65536 entries / 32 MiB, and peak reuse is also applied. This revision passed its
local batch; Linux validation and a new actual common-window measurement remain
pending. Prior
root-only evidence below used the former 32768-entry / 8 MiB plain-dict cache.

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
65536 entries and at most 32 MiB of estimated owned bytes. The estimate includes
the actual current OrderedDict allocation, each key tuple, all immutable key fields and
each value. Shared objects are counted repeatedly, making the estimate
conservative relative to independent recursive accounting with deduplication.
Oversized entries are recomputed. FIFO eviction uses `popitem(last=False)`; hits
do not move an entry. Empty-container sizing also uses the actual OrderedDict
type. Eviction only removes derived results.

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

Prior root-only local validation (former plain-dict / 8 MiB configuration):

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

Prior Linux validation and actual common-window observation:

The primary agent reviewed the Linux `root_paired` combined test record:
`120 passed in 13.82s`, with 120 collected node IDs. Its metadata test-count field
is zero; that contradiction is retained rather than rewriting the metadata or
using zero as the executed count. This record predates the current OrderedDict,
larger root budget and peak reuse revision.

`paired_root_diagnostic_20261018a`, W71-A, completed all six cold jobs with matching
six-member ReplayBundle hashes and all owned processes ended. R0 rollout wall
seconds were `10.69648504399811`, `10.610744668003463`, `10.633883430004062`;
OPT seconds were `8.689243458997225`, `8.809021413995652`, `8.801069499000732`.
Median wall ratio was `1.208249001`, status `NOT_QUALIFIED`. Each of the three OPT
runs recorded 72814 logical root requests, 51543 actual computations, 21271 hits,
33987 evictions, 17556 retained entries and 8388538 estimated bytes under the old
8 MiB cap. The initial root-cache configuration did not show an additional speed
gain. These observations motivated bounded capacity tuning; they do not isolate
its speed effect.

The current root budget is 65536 entries / 32 MiB and both caches use OrderedDict
FIFO. Current root tests require at least 90000 distinct requests with actual
eviction, FIFO order unchanged on a hit and independent owned-byte accounting.
The peak budget stays 4096 entries / 16 MiB. Local batch `energy_cache_tests_20261018b` exited 0:
`116 passed, 3 skipped in 58.28s` (wrapper 58.941s). It ran root reuse, peak reuse,
existing energy queries, physics, windows and paired tests in the local venv
`-I -B` interpreter, setting `dont_write_bytecode=True` before pytest import and
an explicit project import path. Exact command, logs, exit and evidence are under
`artifacts/rotating_slots/energy_cache_tests_20261018b*`.

The first batch, `energy_cache_tests_20261018a`, is retained with exit 1:
`115 passed, 3 skipped, 1 error in 49.35s`. The primary agent diagnosed only the
synthetic unresolved-node test trace: it treated returned endpoint tuples as
Interval objects and called `.pair()`. The sole test fix serializes actual
`v/a` endpoint values through `exact([v,a])`, retaining float hex, synthetic
labelling and the three-node R0/OPT trace. No kernel, budget or policy changed
between the two batches. The independent root and existing energy comparisons
remained exact; the completed b batch also verified the peak oracle.

Current root query counters were 805 logical / 528 computes / 277 hits initially,
then 1610 / 528 / 1082 after repeat, retaining 528 entries / 255735 estimated
bytes under the new cap. The 90000-distinct-key churn, actual eviction, OrderedDict
FIFO hit-order and independent owned-byte assertions passed. The three skipped
tests are the opt-in window fixture and two actual Linux paired checks. These
local results do not execute those Linux/opt-in integrations.
Neither the new capacity nor peak reuse has been measured in a real common-state
window. Primary review, Linux checks and the separately authorized measurement
remain pending. Full P3 qualification and resource acceptance remain incomplete.

No changes were made to `evaluate_power`, `_power_array`, `power_bounds`,
quadrature, integration, peak-duration arithmetic, configuration or candidate
policy. The existing energy-query evidence remains applicable as a regression
target. The root component introduces no second energy formula; the separately
documented peak cache reuses exact duplicate top-level queries.
