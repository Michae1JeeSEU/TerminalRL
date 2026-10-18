# Six-window corpus freezing

`scripts/server/freeze_rotating_slots_performance_inputs.py` is a separate helper
for performance contract §15.2. The existing runner still exposes only
`preflight`, `capture`, and `micro`. This helper does not run paired windows,
qualify implementation equivalence, or report speedups.

Run it only after all four source attempts have completed and their parent
`<case>.exitcode.txt` files contain zero. Use a new output directory, separate
from both the capture campaign and frozen reference:

```bash
python -B scripts/server/freeze_rotating_slots_performance_inputs.py \
  --campaign "$HOME/training_outputs/terminal/rotating_slots_perf_v1/capture_b1153a3_20261018b" \
  --reference artifacts/rotating_slots/performance_reference_r0 \
  --output "$HOME/training_outputs/terminal/rotating_slots_perf_v1/corpus_b1153a3_unique"
```

The four cases are `rate180_seed71`, `rate60_seed23`, `rate120_seed11`, and
`rate180_seed11`. Accepted terminal statuses are `SUSPENDED_CAPTURE`,
`PHYSICAL_FAILURE`, and `COMPLETED_EPISODE`. Missing files, a running attempt,
timeouts, implementation failures, incomplete accounting, or inconsistent source
identities fail with exit code 2. A suspended capture must have reached its
required requested milestone plus 60 simulated seconds; truncation is not a
physical stop.

Selection follows `W71-A`, `W71-B`, `W71-C`, `W60`, `W120`, `W180`, with preferred
requested milestones 300, 900, 1200, 600, 600, and 540 seconds. The recorded
natural cut for a preferred pinned request is selected only if at least 60
actual simulated seconds remain through the observed terminal stop. Only a real
physical failure permits substitution: choose the latest saved natural corpus
cut at least 60 seconds before that stop, then move backward to maintain at
least 30 seconds between cuts with the same workload seed already selected in
the fixed order, including seed 11 across rates 120 and 180. A preferred cut
conflicting with this spacing is unavailable; it cannot trigger substitution
unless the physical stop also makes the preferred 60-second suffix unavailable.
Missing a required pin after its 60-second suffix was observed is also missing
evidence and does not permit substitution.
No wall timing enters selection. Insufficient distinct cuts yield `UNAVAILABLE`
and an `INCOMPLETE` manifest, with exit code 2.

Before publishing a complete manifest, the helper verifies the fixed R0 export,
semantic and physical configuration identities, full 3600-second generated
arrival workload, original 3600/5400 invocation targets, R0 serial/parallel
labels, terminal result/ledger/source consistency, and retained checkpoint
index/sidecar hashes. An isolated R0 process invokes its actual
`ReplayBundle.validate`, checks regenerated manifest/commit/state bounds, loads
each selected checkpoint with the exact captured implementation identity, checks
the decoded world time and cursor, and cold-restores it using the existing
checkpoint implementation. Restore verifies the source stream prefix offset and
hash, frame boundaries/counts, and pending frame against the saved world.
It does not advance the restored world.

`inputs/<window>/` contains byte-for-byte copies of the selected checkpoint and
runner sidecar. These bytes deliberately retain their original absolute source
attempt/stream paths. Preserve the original campaign's immutable `states.jsonl`
streams: each manifest entry records the prefix offset/hash and pending frame
metadata. Copying a checkpoint does not make it self-contained. Linux paths in
synced captures are not silently mapped to Windows paths; execute the utility
where those original paths exist. Existing runner resume rules are unchanged;
this helper introduces no migration or path-rewrite bypass.

`validation/validation_request.json` stores the isolated validation inputs, so
the subprocess command line remains bounded even for a large campaign. Its hash,
isolated R0 provenance/import paths, new collector file hashes, argv, interpreter,
and start/finish times are recorded separately from captured source identity.
`validation/<window>/` retains the new restore stream used for verification.
Source indices, pins, streams, and capture files are never written.

`corpus_manifest.json` is atomically published only after validation and copies
finish. Its `manifest_sha256` hashes the canonical JSON object excluding that
field, using the existing provenance digest function. `FROZEN` requires all six
windows; it always retains `NOT_QUALIFIED` and `HUMAN_PENDING`. A validation or
copy error leaves `freeze_report.json` with `INCOMPLETE` and no final manifest.
Existing output directories are refused, including incomplete prior attempts.

The targeted test file covers fixed-order selection and exact 60/30-second
boundaries, unavailable cases, source status/hash/policy/cursor/stream negatives,
atomic completion and no overwrite, bounded command-line size, and an actual
frozen R0 replay plus checkpoint restore at 0.2 seconds within a 0.4-second
fixture. That compact fixture is not a six-window campaign or equivalence proof.

Observed local check (exit 0): **28 passed in 27.04 seconds**.

```powershell
venv/Scripts/python.exe -B -m pytest tests/test_rotating_slots_performance_corpus.py -q --basetemp=artifacts/rotating_slots/performance_corpus_tests_20261018e
```

Linux component validation also completed on the real terminal
`rate60_seed23` attempt: `read_source` and `validate_reference_inputs` accepted
the selected checkpoint at 600 simulated seconds and the complete observed
replay prefix from 0 through 660 seconds. Cold restoration performed no advance,
and the source `states.jsonl` bytes remained unchanged. The component result is
`VALID`, exit 0, at
`/home/user/training_outputs/terminal/rotating_slots_perf_v1/corpus_component_w60_20261018d/component_result.json`.
The six-window corpus has not yet been frozen. The 28 tests and this component
check do not constitute completion of the contract's 70 acceptance cases.

During validation, hashes for the frozen reference's 24 source files remained
unchanged. Ten extra `.pyc` files with 07:32 inspection-time timestamps were
quarantined with hash guards; the causative argv was unavailable. The audit is
`/home/user/training_outputs/terminal/rotating_slots_perf_v1/reference_bytecode_quarantine_20261018a/quarantine_evidence.json`.
All original 514 deployment files remained unchanged.

Every future inspection or import subprocess must explicitly use both `-I` and
`-B` and set `sys.dont_write_bytecode = True` before imports. Do not rely on the
parent interpreter's flags or environment to prevent child bytecode writes.
