# A4 minimal quality-gap diagnostics

This entrypoint is `scripts/stage3/run_selective_quality_gap_a4.py`. It executes
one explicit factor for one case, under a 300-second parent watchdog. It never
runs a matrix, advances physics, publishes a proposal, retries, or modifies old
results. Its fresh server output is
`~/training_outputs/terminal/rs_selective_quality_a4_20261030`. Its separate
7200-second child-wall ledger shared with common initialization, audits, and
formal arms starts at zero and references the historical
1374.067592 seconds without crediting or resetting any old ledger.

All actions require a reviewed `--source-freeze` and an available physical
`--cpu`. Example shape (one action only):

```bash
python scripts/stage3/run_selective_quality_gap_a4.py baseline --case R1 --cpu 14 --source-freeze /absolute/reviewed/source.json
```

Cases are the old v2 high-arm physical checkpoints 0000, 0065, 0210, and 0235,
named R1, R4, R12, and P2. Historical source times are 0, 60,
206.03508481464814, and 233.65165837993845. Each new baseline uses its actual
checkpoint boundary and lead20, an empty candidate provider, saved selector
history, `opportunity_reference_v4`, and all three search switches false. The
baseline archive freezes final scope/catalog/global frame/actual selection and
callback reference state, and preserves original wall/CPU/memory observations.
This is a common-boundary cold diagnostic, not the original warm input replay.

The explicit factors and dependencies are:

| Factor | Cases | Input | Only changed |
|---|---|---|---|
| baseline | all four | old checkpoint | normal callback, once |
| augment | all four | baseline archive | bounded explicit single-valid new columns |
| catalog_cp8 | all four | original catalog | offline fixed-catalog control |
| augment_cp8 | all four | augmented catalog | candidate additions |
| catalog_cp24 | R1, P2 | original catalog | solver allocation 8 to 24 seconds |
| offset1, offset2 | P2 | original projected snapshot/frame and saved history | selector offset |

`augment` allows six seconds including column validation. R1 tries only
`initial_inner_02`, only P0/P1, with duration attempts alternating P0/P1,
at most one retained column per pad and two in
total, within the baseline group. R4 and R12 select the first hash-sorted
existing immediate partial for `initial_middle_02` and `initial_middle_01`.
P2 prioritizes an actually selected immediate partial with frozen virtual pad2;
otherwise it records the explicit fallback to a selected partial, then to an
existing partial in the same group. Each completion preserves that exact
partial prefix and adds at most one full chain. It calls the existing bounded
reference suffix generator and never changes score, geometry, or scope.
Missing candidates, bounded generation failures, and UNKNOWN validations are
observations without a physical-infeasibility claim.

An addition must pass single-column validation. It is retained as a
`TENTATIVE_SINGLE_VALID_COLUMN`, even when replacing its aircraft's original
actual selection conflicts with another original choice. That exact-baseline
joint/prefix probe is recorded separately and does not filter candidate
coverage. The final augmented solver combination must pass joint and prefix
certification; an uncertified objective is descriptive and cannot count as
improvement evidence. New catalog demands are rebuilt against the same frozen frame,
including every fixed background demand and every original column. Reference
cache state is cloned from the baseline for augmentation; validation contexts
are independent and empty. Both original and augmented fixed-catalog solves
use solver8, hint3, the quality profile's 64-round maximum, and identical 60-second validation
and search allowances. The solver24 factor changes only solver allocation;
its candidate hash is identical to solver8. Optimality is restricted to the
fixed observed catalog.

Every factor is a separately billed `OFFLINE_DIAGNOSTIC` job. Its `binding.json`
contains source and timing identity; outputs are exclusively created, so a
reused path or job cannot silently overwrite results. Nonfinite reservation
ends are encoded explicitly; archived executable inputs use the existing
whitelisted hexadecimal-float codec with three named diagnostic dataclasses.

Decoded Section diagnostics can change a frame's repr-based identity while
its codec content is identical. Archive loading first verifies payload and
candidate hashes, the saved scope-to-downstream frame binding, and every
snapshot/time binding. It then refreshes only the scope's objective-frame ID
and downstream frame ID to the decoded frame identity. New outputs record
`archive_identity_rebinding`, including saved/decoded IDs and canonical frame
SHA; frame values and columns stay unchanged, and validation starts in a new
context without reused ACCEPTED results. Short-continuation inputs may differ
from their baseline source manifest only in the quality-gap/short-continuation
scripts and their two tests. Their source differences are recorded; every
other file's presence and hash, including core/config files, must match.
Ledger source revisions remain an explicit primary action.

Short continuation has an external entrypoint and is never launched by a
diagnostic factor. `run_selective_short_continuation_a4.py --stage prepare`
prepares only R4/R12, taking the smallest-hash existing immediate partial for
`initial_middle_02`/`initial_middle_01`. It replaces only that aircraft's actual
baseline choice and preserves every other selected column. Independent offline
batch/prefix checks must certify both choices, and the forced combination must
have a strictly worse frozen downstream score; otherwise the pair is
`NOT_APPLICABLE`, without another UID or case search. Offline discovery and
certification costs are separate from online performance evidence.

After primary review, `--stage run --branch baseline_choice` or
`--stage run --branch forced_partial` runs one branch under its own 300-second
child cap in the same A4 ledger. The reviewed source freeze must include this
script and its tests and match the diagnostic `a4_source_hash`. Outputs are
new-only under `diagnostic_continuations/<case>/<branch>`; preparation writes
the sibling `preparation/prepared.json` and cannot publish an action.

Both branches restore the same SHA-bound physical checkpoint and executed
selector history, with cold candidate/reference/previous search state. A narrow
first-callback adapter revalidates the chosen combination through the real
wrapper FuturePort instead of invoking CP. It compares the complete source and
effective codec inputs, excluding only each top-level generated `version`;
timing, energy, aircraft, resources, commitments, trajectories and configuration
must all match. It records old/new versions and canonical hashes, refreshes the
scope/frame identity links, and performs new single/batch/prefix certification.
Only the certified immediate prefix is deferred to the unchanged executor's
preparation and final authorization. Actual validation/preparation costs count
against the lead20 ready deadline at source+19; missed work is rejected without
moving the effective boundary.

After the intervention both branches clear search products again and delegate
to the same `opportunity_reference_v4`, three-switches-false, old-score normal
policy with lead20 and a five-second heartbeat. Execution/history continue from
actual events; already started plans and physical energy ledgers remain intact.
The endpoint is the absolute source time plus 120 seconds. Full replay,
terminal increments, per-ring holding, first certificates, actual submission
acknowledgements, touchdown/trajectory records and frozen objective comparisons
are saved. An initial action that was not delivered is explicitly marked and
cannot count as a valid action contrast. Replay seed comes from the checkpoint
manifest linked by its identity hash and must equal the fixed high-flow 7325.
Implementing or preparing this entrypoint does not itself run either branch.
