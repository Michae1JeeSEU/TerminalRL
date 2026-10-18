# Prepared final-pass deployment and hash-only migration

Preparation only. The default command parses the generator and its native child,
verifies the fixed 24-file R0 export and frozen corpus canonical digest, and reads
the unchanged checkpoint-helper hash. It generates no deployment/approval, runs
no checkpoint restore or advance, and performs no SSH or server execution.

```powershell
& 'D:\7_codespace\terminal\venv\Scripts\python.exe' -I -B artifacts/rotating_slots/final_pass_deployment_20261019a/prepare_deployment.py --self-check
```

After sources are frozen and the primary has reviewed the actual successful
`final_pass_linux_tests_20261019c` Linux result and the finalized matrix
orchestration script, the exact generation command is:

```powershell
& 'D:\7_codespace\terminal\venv\Scripts\python.exe' -I -B artifacts/rotating_slots/final_pass_deployment_20261019a/prepare_deployment.py --generate-after-primary-approval --primary-approval-note 'Primary reviewed the fresh final Linux suite, its exact source inventory, numerical oracles, window/lifecycle evidence and finalized matrix orchestration script; approve implementation-hash migration only for the single report-only final pass.'
```

The approval flag and a nonempty note are mandatory. Local HEAD is read freshly
as a label at generation time. The source bytes and inventory are checked
separately; the label does not assert correspondence to Git blobs. Generation
requires actual synced Linux `run_metadata.json`, numeric/pytest exits of zero,
the real collection/nodeids and successful source-finish verification. There is
no fixed test-count target. Fresh Bernstein/root/peak/energy/P2 oracle outputs,
actual cold-window fixture and worker-lifecycle JSONs plus their process/log
siblings are included by their observed paths and SHA256. Missing outputs fail
without inventing pass evidence. All files in the tested source inventory must
still have their exact tested bytes locally. The production inventory must
equal that suite inventory after excluding only its explicitly named test and
harness extras. The generator and exact reviewed
`artifacts/rotating_slots/final_pass_window_matrix_20261019a/run_matrix.py` are
separately hashed, explicit deployment tooling extras. The latter allows the
actual matrix entrypoint to pass the existing provenance check. Its AST/hash
readiness is reported separately; neither script is claimed as part of the
14-file numerical Linux suite. Generation requires the matrix script to exist.

The current real-window fixture must contain the exact output identities
`benchmark`, `profile`, `parallel` and `migration`. All four result and cold-cache
reports are retained, including the parallel result's observed worker startup,
cold caches, complete startup coverage, joined shutdown and CPU evidence.
This is the actual four-variant fixture protocol, not a target for future pytest
collection counts. A skipped/platform-dependent test prevents generation and
returns the raw suite-log tail for primary review; no skip is converted to pass.

Successful generation writes new `deployment.json`, `R0/preflight.json`,
`OPT/preflight.json`, each child's `preflight_process.json`,
`R0_migration_approval.json`, `OPT_migration_approval.json`, and
`artifact_manifest.json` in this directory. Existing generated paths cause a
refusal; historical artifacts are preserved. A failed generation may leave
partial evidence, which must be reviewed by the primary before further action.
The script never overwrites or silently retries those outputs.

The two preflight processes use native argv with `-I -B`, existing
`runner.bootstrap`, `windows.bind_window_entrypoint` and `_verify_finish`.
Their argv/stdout/stderr/exit and exact target/checkpoint/shim/helper identities
are recorded. They do not restore or advance an actual source checkpoint.
Each approval uses existing `window_migration_approval_v1`, registering the two
frozen source implementation identities against the exact target for that
kernel. Only the implementation SHA may differ: semantic/config identity,
checkpoint helper and both observation-shim hashes must remain identical.
Existing corpus validation retains its cursor checks. No historical
energy-only or unchanged-trajectory assumption is applied.

Supporting evidence paths remain the original absolute Linux paths under
`/home/user/training_outputs/terminal/rotating_slots_perf_v1`; the manifest
records the corresponding local Syncthing paths only for reading/hash evidence.
Frozen corpus/checkpoint/stream paths are never rewritten. Actual Linux
`validate_approval` will consume those exact server evidence bytes later.

`qualified: true` is issued only by the explicit approved generation branch and
qualifies implementation-hash migration only. The user's amendment makes speed
targets report-only for one final pass. Every artifact retains
`NOT_QUALIFIED` and `HUMAN_PENDING`; this step establishes no performance or
visual acceptance and launches no real window. The artifact manifest records
the exact expected next-launch inputs and digests for primary review.

The current required suite output is c. Historical Linux outputs a and b remain
retained: a exited1 with 195 passed/one tracker assertion failure; b had 263
collected, 223 passed/one remote local-only manifest-creation setup error in
41.65s. Neither can authorize generation. Harness-only allowed paths remain explicit
existing artifacts/rotating_slots/final_pass_linux_tests_20261019a/run_tests.py
and run_linux_tests.ps1. Evidence-directory names and harness source paths are
intentionally distinct. No c result or migration approval has been generated by
this preparation update.

The fresh c suite supplies runner tests with an immutable local deployment
metadata JSON through Syncthing. Its path, file SHA, expected commit and exact
modules={} production map are verified separately before/after tests and recorded
in runner_fixture_preflight.json/runner_fixture_finish.json. The generator checks
those records against the local metadata bytes and current production baseline,
and retains both records as supporting evidence. The JSON does not enter source
extra_paths or production discovery. No server manifest creation is enabled and
no SSH environment variable is removed. LocalCheckOnly may create only this
separately identified metadata artifact, with identical-existing-byte checks.
