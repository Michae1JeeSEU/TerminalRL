# Bernstein coefficient precomputation

Status: APPLIED as this round's last kernel candidate. The actual-source target
batch passed 164 tests with three skips in 74.06s, exit 0; artifact history below
is kept distinct from that application result.
The scope follows `user_scope_amendment_20261019.md`: complete one round and
report measured effects, without another optimization loop to reach 2×.

The original `poly_bounds` body remains verbatim as `_poly_bounds_uncached`,
including successive derivatives, left-associated product/division and sum order,
NumPy error arithmetic, nextafter padding and return type. The public
`lru_cache(maxsize=16384)` remains. Only builtin tuple coefficients of at most
128 finite builtin int/float values, finite builtin float endpoints/scale and
builtin int derivatives 0 through 3 use metadata; unsupported cases call the
original function. The public LRU's original signed-zero aliases are retained.

Coefficient-only metadata stores original derivative coefficients,
`c[j]*comb(j,i)` products, integer combination tables and `sum(abs(c))`. Its key
contains the policy, derivative and each coefficient's builtin type/float.hex;
integer values also disambiguate distinct ints with equal float conversion.
Interval powers are computed once per exponent on a public LRU miss. No Horner
rewrite, polynomial shift, new interval certificate or physical policy is added.

The OrderedDict FIFO is bounded by 4096 entries / 16 MiB estimated owned bytes.
Sizing includes the actual cache object/container, recursive record/key sizes
and 4096 bytes of scalar/bookkeeping allowance; shared references are deliberately
overcounted. Eviction uses `popitem(last=False)` and hits do not reorder entries.
Oversized metadata falls back to the original computation. Metadata clear/stats/
enable APIs leave the interval LRU intact; disabling metadata does not discard
already cached interval answers. Owned-byte accounting is not RSS qualification.

The read-only window observer explicitly reports
`trajectory._BERNSTEIN_METADATA`; R0 absence is `present=false, entries=0`.
The existing final observer body reads the same named cache after rollout and
record flush, before replay export, preserving separate cold evidence. Paired
cold evidence accepts this known key only with builtin int zero entries and bool
present, while rejecting warm/unknown keys, missing fields and wrong field types.
Worker accelerator-reset integration remains a separate task; this change does
not edit parallel, process_metrics or runner.

Historical candidate evidence is retained in
`artifacts/rotating_slots/bernstein_precompute_patch_20261018a` and `...18b`.
The a/c run passed 8 tests in 5.67s; b passed the same eight tests in 7.12s and
nine requested metadata-whitelist fixtures in 3.25s. Both numeric candidates kept
9380 queries plus repeats, exactly matching independent fixed R0 hex results:
`2fa6c61a9a600fe0d5a227efa8b228bab35c877e9a1e3c3c8f452453bc57636e`.
Canonical same-object/import-path assertions were checked. Earlier a bootstrap
results initialized the old package first and remain historical, not candidate
integration evidence. No applied-source test result is inferred from these runs.

The b candidate's estimate/independent observed bytes were 12015/5919 populated
and 6821/2561 after 79 FIFO evictions under an 8649-byte fixture budget; its
oversized fixture retained zero entries. These are local retained-object checks,
not traffic performance or full resource acceptance.

The current actual-source batch targets Bernstein, peak/root reuse, P2,
Cartesian component, execution reuse, energy queries, windows and paired tests.
It retains the original numerical matrix. Current Linux/integration and real
common-window measurement remain pending; completed outcomes will be reported
without inventing the original contract's `PERF_ACCEPTED` status.


Current applied-source validation

`final_kernel_tests_20261019a` ran the nine listed target files through venv
`-I -B`, with bytecode disabled before pytest import, an explicit project path
and no artifact override or `RUN_WINDOW_INTEGRATION`. Exit was 0:
`164 passed, 3 skipped in 74.06s` (wrapper 74.672s). All 9380 queries and repeats
matched fixed R0, with the same result SHA256 above. Bootstrap evidence confirmed
actual `terminal_marl/core/rotating_slots/trajectory.py` and all canonical
same-object assertions. Root, explicitly enabled peak and existing energy
oracle comparisons were also exact. The fresh peak-default-off child verified
original results with key/owned-byte helpers forbidden.

Actual command/stdout/stderr/numeric exit are retained at
`artifacts/rotating_slots/final_kernel_tests_20261019a.*`; Bernstein oracle and
import evidence are in its `bernstein_equivalence0/provenance.json`, root evidence
in `root_reuse0/evidence.json`, and default-off output in
`test_fresh_default_disabled_by0/default_disabled.stdout.json`.
The three skips are the opt-in window fixture and two Linux-only paired checks.
Current Linux joint integration and real common-window effects remain unmeasured;
this local result does not complete the remaining lifecycle/traffic/observation
work or manufacture a contract performance acceptance label.
