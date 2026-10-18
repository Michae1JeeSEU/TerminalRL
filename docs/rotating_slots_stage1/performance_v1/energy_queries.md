# Energy query characterization before P3

Status: **NOT_QUALIFIED / HUMAN_PENDING**. This is bounded L0/L1 evidence for
contract §§10, 14 and 15.1 and the subjects of C01–C04. The test names and
coverage below do not award catalogue PASS, performance qualification or any
P3 optimization acceptance. No existing source, test, script or deployment
file was modified for this task; only this document and
`tests/test_rotating_slots_performance_energy_queries.py` were added.

## Oracle and comparisons

The same explicit query program runs in two fresh interpreters. Expected
results come from the fixed 24-source-file R0 export at
`artifacts/rotating_slots/performance_reference_r0`, commit
`1c5afd37c891a0ed4776f64367fc7264db3c1c54`, source SHA256
`cd76b4c14b0a48d6d2a71ec7c63ce77010e6ea019b545807635b926eb2984891`.
`performance.provenance.run_reference` checks export bytes and module paths;
the test additionally verifies every returned R0 source-file hash. OPT runs
in a separate `-I -B` child. Both set `sys.dont_write_bytecode=True` before
core imports. Each child has a 600-second timeout.

All floats, including signed zero, are serialized with `float.hex()`. NaN and
positive/negative infinity have explicit tags. Integers, booleans, missing
fields, statuses, reasons and complete candidate payload/hash fields retain
their original types and values. Results are compared exactly without
rounding or OPT-generated golden values. Cache hit/miss observations are
diagnostic metadata, outside this physical-equivalence comparison. All returned energy/domain results
in the executed queries were valid; no unexpected-error artifact was produced.

## Explicit query scope

| Subject | Actual inputs per first/repeated pass |
|---|---|
| C01 subject: stable slots | All 83 slots (37 outer, 28 middle, 18 inner), absolute starts `0`, `300.137`, `720.9445824868908`, lengths `0.031`, `0.2`, `10`: 747 queries |
| C02 subject: fixed descent | Six pads × the same three absolute starts: 18 bindings, 54 actual segments; full three-segment, nongrid within-segment and both junction-crossing energy queries |
| C03 subject: arbitrary intervals | Three generated accepted plans: outer→middle transfer, inner normal final and outer emergency final; 30 queries total |
| C04 subject: remaining energy | The same three immutable accepted plan objects; 42 now values in total, including the `0.2` progression and nongrid/boundary offsets |

Stable plans retain their real slot metadata and authoritative payload/hash
and cover `[0,1000]`; the test reuses each plan across its queries. Every query
records the original Cartesian midpoint state, scalar power with its actual
nonzero centripetal acceleration, an explicitly identified zero-acceleration
power control, and complete integrated energy/SOC/peak/error results. The
centripetal term is never removed from the queried plan or integrator.

Observed R0 battery-power ranges across these stable queries were
`68078.00959046128–68078.0095904613 W` (outer),
`74812.87009118882 W` (middle) and `92773.1278485505 W` (inner).
The small outer variation is retained exactly in the saved outputs; the test
does not replace all absolute-state scalar values with a single ring constant.

Descent inputs contain actual pad IDs, absolute times, all polynomial segment
metadata and endpoint/midpoint states with scalar power/domain results. Their
initial pad positions, touchdown positions and C2 joins are checked. Energy
queries use the clearly labelled `DescentSegmentsOnly` function-level
container, whose `intervals()` yields these real segments. This container is
not a `TrajectoryPlan`, an approved full final or a bypass certificate for
future runtime use. It isolates segment-level numerical behavior; it does
not test full-final admission/resource binding.

The generated candidates are independently accepted in R0 and OPT. Their
complete candidate results and plan payloads/hashes are included in explicit
inputs. Arbitrary queries include full duration, zero length at start/end,
length shorter than `0.2`, within-cell and cross-cell intervals, nongrid
`0.137`/`0.371` offsets, trailing partial intervals and every final segment
junction. `max_dt=0.2` is unchanged for all calls; no endpoint is rounded.

C04 uses the same accepted immutable `TrajectoryPlan` representation throughout
each series, with explicit full, used, remaining and incremental intervals.
It does not perform a world/resource commit. Direct remaining energy is always
queried from R0's original integrator. The output separately records
`full−used`, partition residuals, accumulated incremental energy, differences
from direct used energy and each original reported integration error. No
mathematical exact-sum assertion or new error guarantee is imposed on R0.
The plan payload/hash remains unchanged after its complete series.

Observed maximum absolute R0 residuals in joules are:

| Plan | used + remaining − full | cumulative increments − direct used | direct remaining − (full − used) |
|---|---:|---:|---:|
| Transfer | `0.0` | `7.275957614183426e-12` | `1.1823431123048067e-10` |
| Normal final | `1.862645149230957e-09` | `3.725290298461914e-09` | `2.6338966563344e-09` |
| Outer emergency final | `3.725290298461914e-09` | `3.725290298461914e-09` | `4.496541805565357e-09` |

These are characterization values, not optimization tolerances or proofs that
subtracting independently conservative prefixes is safe. Full per-query
`error_j`, peak duration, validity, status and reason are retained alongside
each residual.

## Cache observation and result

The energy `_segment_table` and `_integrable_plan` caches start empty in each
fresh process and are explicitly cleared before candidate construction. They
remain empty immediately before the first query pass; candidate construction
may already have warmed existing safety-validation certificates. After the
first pass, `_segment_table` has 6 entries (6 misses/438 hits) and
`_integrable_plan` has 86 entries (86 misses/820 hits). After repeating exactly
the same query sequence, entries/misses remain unchanged and hits become
882/1726 respectively. Both passes have exactly equal returned results, and
R0/OPT cache observations also match exactly.

This observes only these existing energy cache paths in this process sequence.
It does not claim complete cache on/off equivalence, future P3 cache behavior,
eviction, invalidation or memory-budget coverage.

The actual local command was:

```powershell
.\venv\Scripts\python.exe -B -m pytest tests/test_rotating_slots_performance_energy_queries.py -q -x -p no:cacheprovider --basetemp=artifacts/rotating_slots/energy_query_tests_20261018a
```

Historical result before the review amendment below: **8 passed in 20.01s**,
exit code 0, test source SHA256
`8adf1ef1f4dd3d0e6c612bf4cffb9cbb81f4bb02f3f12a31831e2f46d2f4c61a`.
Only this new test file was run. These `20261018a` artifacts remain unchanged.
Artifacts are under
`artifacts/rotating_slots/energy_query_tests_20261018a/energy_queries0`:

- `explicit_inputs.json`: 273425 bytes, SHA256 `6f93dd18dff85aead51274919eb220fabe3c275f92e66e91cf2d18d78abf3826`.
- `r0_results.json` and `opt_results.json`: each 2658780 bytes, identical SHA256 `8b7d582de31759303e9e87be479e3903f43f69be208c331ad39d78f3eb3d6cdb`.
- `provenance.json`: SHA256 `6635c7d48ca99b003625b6782388e91577004da067638739d6c5ae12bf59cdc2`; includes R0 PID/import paths/file hashes, OPT source hashes, query/test hashes, counts and exact-comparison results.
- `observed_summary.json`: actual coverage, cache counts, power ranges, residual summaries and recorded pytest outcome.
- `opt_stdout.log`, `opt_stderr.log`, `opt_numericexitcode.txt`: isolated OPT subprocess output and exit evidence.

The subsequent review amendment removes equality of R0/OPT cache hit/miss
counts from acceptance: contract §4 permits different internal computation and
cache behavior. The new provenance field `semantic_exact_match` compares all
query inputs, coverage counts and physical/candidate results while excluding
`cache_observation`; `cache_counts_required_equal` is explicitly false. Cache
observations remain recorded in full. The replacement cache check independently
requires each implementation's named caches to start empty, have nonnegative
integer counters and remain within their declared entry limits. First/repeated
returned results still must match exactly in each implementation, independently
of cache counters.

The primary dialogue authorized the revised test after concurrent trajectory
component work finished. The same isolated local command, using the new
`artifacts/rotating_slots/energy_query_tests_20261018b` basetemp, reported
**8 passed in 9.78s**, completion exit code 0; the OPT child also exited 0.
The tested source SHA256 is
`78899d18fabd73be7c16ead130ed36cffe4ec52014cec7c3d43be660da86d080`.
The new `energy_queries0/provenance.json` has SHA256
`e4e4ce54baae49ceb0d7de6a49f1258692d4cd33efb1918dd562c5ea01ace358`
and records `semantic_exact_match: true`, `cache_counts_required_equal: false`
and `first_repeat_exact_match: true`.

The `20261018b` R0 result bytes match the historical `20261018a` R0 result
bytes exactly; both current R0 and OPT results retain SHA256
`8b7d582de31759303e9e87be479e3903f43f69be208c331ad39d78f3eb3d6cdb`.
Current explicit inputs retain SHA256
`6f93dd18dff85aead51274919eb220fabe3c275f92e66e91cf2d18d78abf3826`.
The current provenance separately records the tested OPT trajectory source
hash following that component change. The merged pytest output log is
`artifacts/rotating_slots/energy_query_tests_20261018b.log`, SHA256
`fe42c34b0c408953d0a77c157dfdaef12e9895cdc6e5771360ceadf4edcb9368`;
the new basetemp also contains `pytest_result.json` and `numericexitcode.txt`.
Historical `20261018a` files were not overwritten. No additional test file or
server run was executed for this revision.

## Remaining scope

This fixed, nonrandom characterization does not cover 20 shapes per maneuver,
inbound or middle→inner transfer, middle/inner emergency sources, phase wrap
and both sides of all 24 phase-bin boundaries, comprehensive success/rejection
cases, all abstract peak-threshold crossing shapes, or C04 reserve/qE/event
threshold behavior. It performs no traffic loop, scheduler change, server
execution, performance measurement, new optimization or complete catalogue
qualification. Those absent cases are not marked PASS.
