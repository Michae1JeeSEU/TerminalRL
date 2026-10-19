# Fixed R0 G09 recovery B evidence

This one-off local recovery preserves the interrupted original92 run and failed recovery A. It neither changes the kernel nor pursues a performance threshold. The fixed40 jobs remain counts10/30/60/83 × stable/inflight_attempt × repeats0..4, duration.2 and original command `run_stage1_tests G09 actualbenchmark`. No other case is rerun, no old output is overwritten, and the R0 fixed24-source export remains unchanged.

## A remains a failed run, with two completed physical jobs

`run_20261019a/summary.json` ended at UTC08:00:03.502393 as `RECOVERY_ERROR`, driver exit1, supervisor exit1 and both fresh job children exit1. Original `diagnostics._diagnostic_job` had already written COMPLETED results and valid archives for:

| Persisted A result | Original wall seconds | Original CPU seconds |
| --- | --- | --- |
| benchmark_n60_stable_r2 | 200.37625250034034 | 198.390625 |
| benchmark_n60_stable_r3 | 199.71602859999985 | 197.484375 |

The primary diagnosis is a validation representation error: in-memory `metrics.pad_busy_seconds` has integer pad keys; original `qualification.plain` serializes Mapping keys as strings and tuples as lists. Comparing raw in-memory metrics directly with loaded archive JSON therefore failed after the physical job completed. This is not a physical difference. The corrected JOB reads original `_diagnostic_job`'s real written `result.json`, requires exact equality to original R0 `qualification.plain(raw_row)` for the entire result, then validates that serialized row against the archive. No rounding, dropped comparison or physical formula change is used.

A's parent `run_reference` export postcheck was reached before its nonzero-returncode error, and A's recorded source/reference/old-input before/after maps are equal. Its children stopped at query `checked` line30; their final post-import module audit was not reached. The B inspector records the actual A child-start PIDs/flags/argv, request job/source identity, exit1 and that limitation. Current fresh strict R0 read-only validation can verify the persisted artifacts now; it does not retroactively turn those child exits into0 or claim their missing final import audit completed.

## B fixed inputs and remaining work

B requires exactly22 original completed rows plus exactly the above two persisted A rows, then plans only16 missing jobs: n60 stable repeat4, n60 inflight repeats0–4, n83 stable/inflight repeats0–4. The two A jobs are not recomputed. A row/job/count/source mismatch stops without choosing substitutes or retrying. Original and reused inputs have separate roots, hashes and origin labels; they are never described as24 rows from the original run.

One fresh `run_reference(modules=())` R0 inspector validates all24 serialized rows, exact job dictionaries/provenance/source maps, ReplayBundle integrity/metrics, positive CPU/wall, actual airborne/inflight counts. It also verifies A's failure records, matching before/after maps, actual child startup and original request identities. Its bounded no-simulation checks use both real A rows: persisted metrics equal archive metrics, reconstructed integer-pad-key metrics fail direct equality but pass original `plain`, and changing actual `physical_airborne_count` is still rejected by the archive-metric comparison.

`--inspect-only` validates/writes a plan without simulating. Only explicit `--run-missing` runs the fixed16 jobs, each as a fresh isolated `-I -B` R0 child, six native-thread variables1, original serial/cProfile behavior and1800s query budget. At most two children run concurrently; a failed wave stops later waves while an already started companion is awaited. Requests travel by JSON file and query code is bounded; no monkeypatch or import/export-check bypass is installed.

After all16 complete, `diagnostics.json` takes original22 + reused2 + new16 in the original40 order, preserving each row's real archive path/provenance/timing. Required schema/protocol is retained for the original G09 consumer, with explicit `G09_BENCHMARK_ONLY_RECOVERY`, no original traffic matrix and no claim of original diagnostics completion. A fresh R0 child consumes it through original `qualification.run_catalogue(...cases=['G09'],workers=1)` and must actually report G09 PASS. The outer fallback G09 archive is not the40 benchmark replays. The original92 run remains interrupted and A remains `RECOVERY_ERROR`.

Before/after snapshots separately protect current source/script, exact R0, original91 case results +22 result/archives, and A's two result/archives plus A summary/source/reference/old-input/child failure evidence. Explicit resolved-root checks precede hashing. These are current observed snapshots, not invented old terminal states. No physical files are copied. Unknown child exit remains null with no numeric child file; successful helper return proves0 but does not expose successful stderr. No hard-timeout complete process-tree CPU or formal speed claim is made.

## Historical B preparation checks and launcher

The following readonly command was executed for this fix; its output now exists and must not be overwritten. It did not launch missing jobs:

```powershell
& 'D:\7_codespace\terminal\venv\Scripts\python.exe' -I -B 'D:\7_codespace\terminal\artifacts\rotating_slots\final_pass_reference_g09_recovery_20261019a\recover_g09.py' --inspect-only --workers 2 --output 'D:\7_codespace\terminal\artifacts\rotating_slots\final_pass_reference_g09_recovery_20261019a\inspect_20261019b'
```

`--help` writes nothing; output must be a new owned child. `launch_recovery.ps1 -LocalCheckOnly` only checks fixed paths, fresh `run_20261019b`, hashes and syntax. The prepared normal branch uses independent hidden PowerShell `-Supervise`, then hidden Python `-I -B --run-missing --workers 2`, redirected stdout/stderr and separate actual child/supervisor/internal exit records. Its evidence prefix is `launch_20261019b`; it never precreates the driver's output root. The original A output and launch files stay untouched. Neither normal launcher nor `--run-missing` is executed as part of this fix.

Historical `inspect_20261019a` validated22+18 with the A driver, before its physical run; its raw evidence remains unchanged. B readiness/readonly results are recorded below only after execution. Human status remains `HUMAN_PENDING`.

## Historical B read-only result

AST, `--help` and B `-LocalCheckOnly` completed with exit0; the latter started no process and confirmed fresh `run_20261019b`. The actual B inspection ran UTC08:16:32.698391–08:16:50.618983, outer17.906s, parent PID54712 and fresh strict R0 child4136, both exit0. `inspect_20261019b/summary.json` reports `INSPECTED_22_ORIGINAL_2_REUSED_16_MISSING`, success=true, original_completed22, reused_completed2, validated_existing24, missing_count16 and new_completed0. Source, reference, original input and reused-A input before/after hashes all remained equal.

`inspect_20261019b/plan.json` records both real A rows' archive metrics equal; reconstructed integer-pad-key memory metrics compare directly false but original R0 `plain` compares true. For each row, modifying actual `physical_airborne_count` was rejected specifically as `ARCHIVE_METRICS_MISMATCH`. All24 rows passed the current strict R0 provenance/import/config/archive/positive-CPU/wall/count checks. The plan separately records A's actual child PIDs36936/29500, previous child exits1, completed parent export postcheck and absent child final post-import audit. Current revalidation does not retroactively change A's failed atomic child status.

Raw B argv/PID/UTC/stdout/stderr/numeric exit use `inspect_20261019b.*`; strict R0 evidence/startup and positive/negative results are under `inspect_20261019b/children/inspect/` and `plan.json`. No physical job was recalculated, no B missing job or final G09 consumption ran, no background supervisor was started, and no Git/server command was executed during this fix. Formal B launch remains separate from these completed read-only checks.

## Completed B recovery (2026-10-19)

B is now terminal: [summary](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_reference_g09_recovery_20261019a/run_20261019b/summary.json) records `G09_RECOVERY_CHECKED`, success=true, ended UTC 09:11:00.297361 and driver wall 2774.797 s. It used 22 original + 2 persisted A + 16 newly completed B jobs; only the missing 16 were run. [diagnostics.json](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_reference_g09_recovery_20261019a/run_20261019b/diagnostics.json) retains all 40 in their original order and the explicit benchmark-only recovery scope. Source, reference, old-input and reused-input before/after maps all remain equal.

[Supervisor completion](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_reference_g09_recovery_20261019a/launch_20261019b.completion.json) records UTC 08:24:40.1782478–09:11:00.3838812, 2780.2038314 s, driver PID 1868 and supervisor PID 36424, both exit 0; [driver numeric exit](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_reference_g09_recovery_20261019a/run_20261019b/numericexitcode.txt), [external child exit](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_reference_g09_recovery_20261019a/launch_20261019b.child_numeric_exit.txt) and [supervisor exit](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_reference_g09_recovery_20261019a/launch_20261019b.supervisor_numeric_exit.txt) are 0. The final original-R0 consumer child PID 20400 exited 0; its [G09 result](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_reference_g09_recovery_20261019a/run_20261019b/check_G09/G09/result.json) is PASS with 202 assertions. These are G09-only recovery facts: the original 92-case run remains interrupted, A remains `RECOVERY_ERROR`, and the original V failures remain unchanged. They do not establish all 70 added catalogue items or human acceptance; `HUMAN_PENDING` remains.
