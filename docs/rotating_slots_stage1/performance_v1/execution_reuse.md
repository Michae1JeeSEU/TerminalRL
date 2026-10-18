# Interval-local execution query reuse

**LOCAL CHECKS PASSED. NOT_QUALIFIED / HUMAN_PENDING.**

This implementation changes `safety.py`, `world.py`, adds
`tests/test_rotating_slots_performance_execution_reuse.py`, and adjusts two
existing P2 monkeypatch wrapper signatures to accept the private keyword.
Existing assertions and pair-order expectations remain intact. The patch was
prepared separately and applied only after the primary dialogue confirmed the
previous diagnostic's completion. No deployment or server file is changed.

The private `_ExecutionQueryCache` is constructed from the existing
`bases` only when the world's `cache_enabled` is true. It lives in one
`_check_execution` invocation, including that invocation's failure bisections.
No world/checkpoint field, module cache, wrapper plan or eager numerical
precomputation is introduced. The default validator and audit calls have no
cache. The old mock-holder path without `cache_enabled` defaults to false;
actual worlds always have this field and retain cache use when enabled.

Only exact `TrajectoryPlan` instances present in that base set are eligible.
The cache retains strong references to them. Query keys contain object identity,
operation, exact `float.hex()` times and derivative index. Signed zeros and
adjacent floats remain distinct; non-float times, unknown custom plans and
non-immutable results bypass caching and call the original methods. The
original immutable KinematicState or nested-tuple value is returned without
copying or rewriting numerical expressions.

FIFO eviction is deterministic. Hard maximums are 4096 entries and an 8 MiB
cache-owned byte estimate; lower private limits permit bounded churn tests.
The estimate charges binding/container metadata, keys, returned values, a
4096-byte fixed object/counter allowance and 512 bytes of per-entry bookkeeping;
original authoritative plan payloads are
referenced, not copied. If bindings alone exceed the budget, all query caching
is disabled for that local object. Oversized values recompute without storage.
This estimate is not a complete process RSS qualification. Local hit/miss,
eviction/bypass counters are test observations, not physics or physical world
state, and are not required to match R0's internal computation counts.

Only `state(mid)` and `bounds(a,b,k)` calls in `validate_separation.check` use
the optional keyword-only `_query_cache`. Constant-vertical and canonical
stable structural proofs, pair order, recursion, every arithmetic expression,
certificate details/minima, UNKNOWN behavior, bisection limits and failure
energy/event order remain unchanged. Each bisection end creates an exact new
key; a prior SAFE result is never reused as a certificate or minimum.

The isolated-R0 tests compare complete SAFE/UNSAFE/UNKNOWN details,
mixed transfer/final/ground/junction queries, first-failure order and short
real execution records. Separate tests cover twenty moving function-level
plans and 190 pairs, actual original method call reductions, cache-off/default
behavior, repeated fresh local caches, entry/byte churn, strong references,
custom-plan bypass, signed zero/adjacent times, and object/version separation.
An independent recursive `sys.getsizeof` test accounts for owned cache objects,
attribute/counter/binding dictionaries, entry records, keys and numerical values
(excluding authoritative plan objects) after population and churn, and requires
the estimate to cover that measured accounting. Private smaller budgets have
an 8192-byte floor above the fixed metadata allowance. Observed accounting is
recorded below.
The twenty moving plans are function-level geometry fixtures, not approved
traffic. These tests do not establish a speedup.

The oracle query imports the unchanged fixed R0 export in a separate
`-I -B` interpreter and compares tagged `float.hex()` results to a separate OPT
child. Both set `sys.dont_write_bytecode=True` before imports. Cache counters
are stored separately from semantic result comparisons. Formal corpus,
threshold, complete-traffic and performance acceptance remain outside this
local change.

## Actual local validation

The authorized command ran only the execution-reuse, existing P2, physics,
world and Cartesian-component test files, using local
`venv/Scripts/python.exe -B -m pytest`, `-q -x -p no:cacheprovider` and
`--basetemp=artifacts/rotating_slots/execution_reuse_tests_20261018a`.
Result: **131 passed in 110.84s**, exit code 0. `pytest_process.json` records
the complete argv and return code; `numericexitcode.txt` also records 0.
All oracle/OPT children use `-I -B` with bytecode writes disabled before imports.

In the new isolated comparison, complete cached/default SAFE, UNSAFE and UNKNOWN
results, mixed-plan/junction inputs, controlled pair/minimum/first-failure
behavior, and actual 0.4-second execution output match R0 exactly.
`execution_reuse0/provenance.json` records `semantic_exact_match: true`.
Both result files have SHA256
`3371acf9468382ae89fe1cf3fb24a2f38896d52c50bf1c5402257ee5db4d5e25`;
the provenance SHA256 is
`bc39eeb22903917e7ed8c4d9297f7327f75e3678751da2207fbbea5837af16a0`.
The tested new test source SHA256 is
`e5578702c93922a577d1cff86ff1b4c6a0525834163ca8fcd3de22cb82d95439`.

Twenty moving function-level plans produce 190 original pairs. Actual original
method calls fall from 380 to 20 for state and 760 to 40 for bounds. Both fresh
cache repetitions retain exactly the uncached results, with 60 query entries;
each observes 360 state hits and 720 bounds hits. This is a bounded method-count
observation, not a measured window speedup.

| Observed cache | Entries | Estimated owned bytes | Independent recursive owned bytes |
|---|---:|---:|---:|
| First fresh populated cache | 60 | 80716 | 52757 |
| Second fresh populated cache | 60 | 80716 | 52749 |
| Default-limit churn | 4096 | 5505459 | 4216913 |
| 8192-byte-limit churn | 2 | 7214 | 4472 |

Default churn records five evictions and the small-byte-limit case records
eight. The independent accounting includes owned objects, attribute and counter
dictionaries, bindings, entry records, keys and returned values, excluding
authoritative plan objects. The estimate covers that observed accounting and
stays within its limit. These figures do not qualify full process RSS.

Evidence under the new basetemp includes
`test_twenty_moving_plans_reduc0/method_counts_and_owned_bytes.json` (SHA256
`0691e05fc9bfa9720580481a43b73dde9256df44b02e032743f0e35d4c6b9b6e`) and
`test_entry_and_estimated_byte_0/entry_and_byte_churn.json` (SHA256
`e7663e1d7d7c2e80ee715e4fa70b747368ea88b87153d53ed54dfc29da496ba0`).
The merged pytest log is
`artifacts/rotating_slots/execution_reuse_tests_20261018a.log`, SHA256
`d6728adfc0f71b30958d6698c52ba4c0b2114c4a030489faafac96dbae22fb45`.
No server execution, deployment-manifest change or performance qualification
was performed for this implementation.

`artifacts/rotating_slots/execution_reuse_patch_20261018a/run_linux_tests.ps1`
is prepared for use after Git delivery. In the same invocation it computes
SHA256 for four core files and the five selected test files, reads local HEAD
as a runtime label, and Base64-encodes the request without manual hash copying.
Its `-LocalCheckOnly` path returned `LOCAL_SOURCE_GATE_READY`, nine hashes and
`server_executed: false`; readiness output is saved beside the script. The
embedded Python was parsed without execution. The unexecuted remote branch
uses fresh `-I -B`, bytecode disabled before imports, numerical threads 1 and
nice target 10, and records before/after source/R0 evidence, logs, UTC and exit
status under the fixed new server output
`/home/user/training_outputs/terminal/rotating_slots_perf_v1/execution_reuse_linux_tests_20261018a`.
Expected collection is 131 tests. No Linux run or passing result is claimed
by this preparation.
