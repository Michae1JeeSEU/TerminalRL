# Exact duplicate top-level peak queries — default disabled

Status: implementation retained, default `_PEAK_CACHE_ENABLED=False`. Explicit
`peak_cache_set_enabled(True)` remains available for scoped comparison; tests
that exercise caching enable it explicitly. Default execution calls the original
peak evaluator without building exact keys or traversing owned-byte estimates.
No original peak arithmetic, left-before-right recursion, depth-26 limit or
width-1e-7 conservative policy is changed.

The primary agent chose default disablement after the actual
`paired_energy_cache_diagnostic_20261018a` result: R0 median rollout wall
10.654058980995615s, OPT median 11.831673675995262s, median ratio 0.9004693.
All six jobs across three repetitions retained equal replay bytes. This revision
showed a measured slowdown; it is not a physical failure. The older root-only
revision's medians 10.63388343/8.801069499s and ratio 1.208249 are a different
revision and are not a final result for this configuration.

The separate `peak_cache_profile_20261018a` recorded exclusive CPU 85.1131s.
`_peak_owned_bytes` had 13,493,867 calls / cumulative 27.5598s; key construction
had cumulative 2.5593s; configuration `Config.to_dict` had 2422 primitive/total calls / cumulative
2.759218356s (`config.py:54` is the source location, not a call count).
These cumulative scopes are nested and must not be added. Profile data is not a
speed measurement. Only 6 of 2422 peak-cache requests hit in that observation.
Default disablement removes the measured cache-bookkeeping path while retaining
an explicitly enabled implementation for review; its new measured effect is not
yet established.

When enabled, only exact duplicate builtin-depth-zero queries with exact Segment
and Config types and finite builtin float interval/threshold values can reuse a
result. Keys contain every actual segment field, entire current Config.to_dict
payload, signed-zero-preserving/type-tagged values, exact endpoints/threshold,
depth and fixed numerical policy. No profile hash or object identity replaces
actual payloads. Recursive depths bypass storage; immutable float answers alone
are cached, without exception caching, subtree reuse or prefix subtraction.
The OrderedDict FIFO remains bounded by 4096 entries / 16 MiB estimated bytes;
actual container sizes are included, oversized inputs recompute, hits do not
reorder entries. Clear and enable/disable touch performance state only.

Historical local validation remains in `energy_cache_tests_20261018a/b`.
The first batch's error was only a synthetic tuple-trace `.pair()` assumption;
the fixture was corrected to `exact([v,a])` without a kernel/policy change.
The b batch passed 116 tests with three skips in 58.28s. Its enabled-cache oracle
matched independent R0 for 195 direct and six cross-segment queries, six pads,
54 descent segments and the labelled three-node synthetic trace. Those results
predate the current Bernstein application and default-disable revision.

The current actual-source test adds a fresh `-I -B` default-off check: it forbids
key/owned-byte helpers, calls real `_peak_measure` and compares exact hex output
with the original evaluator. The existing explicitly enabled tests remain.
The current actual-source batch exited 0: `164 passed, 3 skipped in 74.06s`
(wrapper 74.672s), under `artifacts/rotating_slots/final_kernel_tests_20261019a*`.
The fresh default-off child observed two logical requests / two original
evaluations, zero hits and zero entries, with key/owned-byte helpers forbidden.
Its ordinary and narrow-tangent answers matched the original evaluator as exact
hex. Raw evidence is
`test_fresh_default_disabled_by0/default_disabled.stdout.json` under that basetemp.
Explicitly enabled peak and the unchanged independent R0 matrix also remained
exact. The three skips are the opt-in window fixture and two Linux-only paired
checks; Linux integration and actual common-window measurements follow separately.

Per `user_scope_amendment_20261019.md`, this is one bounded kernel closeout,
not another search for a required speed ratio. Physics, safety, equivalence,
complete records and real resource observations remain required. Historical
statuses stay unchanged; no original-contract `PERF_ACCEPTED` is claimed.
