# Exact duplicate top-level peak queries — prepared patch

Status: APPLIED locally after primary review. Current root reuse and the existing
final-cache observer are retained. Local tests for this revision passed as recorded below;
Linux validation and actual common-window measurements remain pending. There is
no measured speedup for peak reuse, full C07 claim or P3 qualification.

This change implements bounded memoization of exact duplicate calls to
`energy._peak_measure` at built-in integer `depth == 0`. A decorator retains the
current P2 named function body, default, arithmetic, left-before-right recursion,
depth limit 26 and unresolved width cutoff `1e-7`. A miss calls the bound original
function. Every recursive call with positive depth bypasses lookup and storage;
there is no subtree certificate, interval-containment rule or conservative-prefix
subtraction. `_peak_measure_uncached` exposes the original through `__wrapped__`.

Only exact `Segment` and `Config` instances with finite built-in float `a`, `b`
and threshold are eligible. The key includes all six actual segment fields,
the entire current `Config.to_dict()` payload, exact interval endpoints,
threshold, depth and the fixed policy string `peak-r0-order-depth26-width1e-7-v1`.
Recursive type tags distinguish integers, booleans, floats, strings, None, lists,
tuples and dictionaries. Floats use `float.hex()`, retaining signed zero; dictionary
order is preserved. Unsupported or nonfinite payloads fall back to the original
function. Neither a profile hash nor object identity replaces actual payload
construction. An object modified through `object.__setattr__` receives a new key
on its next request.

The process-local cache stores immutable float answers only. It defaults to
enabled, with deterministic OrderedDict FIFO eviction, at most 4096 entries and at most
16 MiB estimated owned bytes. The estimate includes actual current OrderedDict allocation
and the recursive sizes of stored tuple/scalar keys and float values, counting
shared immutable objects repeatedly. Empty-container sizing also uses the actual OrderedDict type. FIFO uses
`popitem(last=False)` and hits do not reorder entries. Oversized answers are recomputed. No
exception is stored. `peak_cache_clear()` resets derived entries and performance
counters; `peak_cache_set_enabled(bool)` disables lookup and insertion while
retaining existing entries.

`peak_cache_stats()` separates top-level logical requests, actual original
top-level evaluations and hits. It also reports evaluations made at other depth
values, total original node evaluations, evictions, entries, estimated bytes and
capacities. Counters and answers are performance state and do not enter semantic
world state or checkpoints. The common-window cold observer reads the optional
peak-cache statistics; absence in R0 is valid and empty, and observation never
clears warm entries. The existing final-cache observer obtains the same optional
`energy._PEAK_CACHE` key by a read-only call after stopped rollout timing and
record flush, before ReplayBundle export. Its scope, failure reporting and
separate original cold evidence are retained without replacing its current body.

The paired consumer explicitly permits the reviewed optional peak-cache key in
addition to the existing root key. If peak is present in cold evidence, its
`entries` must be the built-in integer zero and `present` must be a bool. Explicit
absence with zero entries and older R0 payloads that omit the key remain valid.
Warm entries and any unknown extra cache key are rejected; unrelated paired
workload, identity, source, deadline, replay and qualification checks are unchanged.

The independent R0/OPT child-process query compares hexadecimal peak
answers with caching disabled, enabled and repeated. Inputs cover all six pads
at three absolute times, actual fixed descents, transfer and final non-grid
queries, queries spanning segment joins, intervals certified entirely above or
below a supplied threshold and a physical scalar-power threshold crossing.
Controlled narrow hover/tangent and synthetic unresolved-node fixtures are
explicitly labelled as function-level fixtures rather than approved trajectories.
The unresolved fixture compares the original recursion inputs and answer.

Other tests cover original-function call counts, unchanged current P2
peak AST/defaults and frozen semantic fingerprint, actual configuration changes
behind forged identical hashes, segment/configuration mutation, every segment
field and interval/threshold invalidation, signed-zero/type/order key distinctions,
unsupported inputs and exceptions, default-cap churn, smaller entry caps,
oversized entries, independent recursive owned-byte accounting, disable/reset
and optional cold observation. The completed local batch is recorded below.
The test also verifies that final-cache observation retains warm peak
counters and entries unchanged. New paired fixtures cover both optional keys
together, explicit/omitted R0 absence, warm and unknown keys, missing fields and
incorrect bool/int field types. These additional assertions passed in that batch.

The reviewed peak implementation was applied after the prior real root-only
six-job diagnostic finished. The implementation also changes both cache
containers to OrderedDict; the root budget is now 65536 entries / 32 MiB while
the peak budget remains 4096 / 16 MiB. The earlier root-only diagnostic does not
measure this revision or isolate the benefit of capacity changes.

The requested local batch covers root reuse, peak reuse, existing energy queries,
physics, windows and paired orchestration. Linux-only and opt-in window tests are
not implicitly executed by a local ordinary batch. Linux validation and an
actual common-state window remain required before a workload speed claim. Root
and peak numeric formulas, `_segment_table`, joule integration, validation
entrypoints, the original 80 root bisections, depth-26/width-1e-7 peak policy and
left-before-right recursion remain unchanged.


Local validation on 2026-10-18 used the venv `-I -B` interpreter, with bytecode
writing disabled before pytest import and an explicit project import path. The
same six files (root reuse, peak reuse, energy queries, physics, windows, paired)
ran with `-q -p no:cacheprovider` and new basetemp
`artifacts/rotating_slots/energy_cache_tests_20261018b`. Result: exit 0,
`116 passed, 3 skipped in 58.28s` (wrapper 58.941s). The three skips are the
opt-in window fixture and two Linux-only paired checks. Linux validation and
actual common-window performance for this revision remain unexecuted.

The retained first batch `energy_cache_tests_20261018a` exited 1 with
`115 passed, 3 skipped, 1 error in 49.35s`. Its synthetic unresolved-node trace
called `.pair()` on actual endpoint tuples in frozen R0. The primary agent
identified this as a fixture serialization error. Only that test line changed to
`trace.append(exact([v,a]))`; float-hex encoding, the synthetic label and three
nodes remain. No energy, cache budget, numerical policy or kernel change was made
to obtain the successful b result. Both attempts' raw evidence remains available.

The completed independent R0/OPT comparison is exact for all recorded peak
results, including 6 pads / 18 descent bindings / 54 descent segments, 195 direct
queries, 6 cross-segment queries and the three-node controlled unresolved trace.
The first enabled pass recorded 412 logical requests / 206 original top-level
computations / 206 hits, 76 non-top-level evaluations, 97 evictions and 109
retained entries / 16743851 estimated bytes. After repeat, cumulative totals were
824 / 412 / 412, 152 non-top-level evaluations and 303 evictions, with the same
109 entries / 16743851 bytes. Every immediate duplicate matched exactly; the full
matrix legitimately exceeded the byte cap and was recomputed as FIFO evicted
entries. This is function-level reuse evidence, not a workload speed claim or
whole-process RSS measurement. The direct original-call, FIFO hit-order,
independent owned-byte, cap, unsupported-input, cold/end observer and paired
whitelist assertions also passed.
