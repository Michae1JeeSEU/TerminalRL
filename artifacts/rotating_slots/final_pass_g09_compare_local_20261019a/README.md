# Fixed G09 read-only comparison evidence

This local one-off tool compares the actual 40 original G09 benchmark jobs. It never starts a simulation, resumes recovery, changes physical inputs or launches B. Existing R0/OPT inputs remain read-only. Only fresh output children of this artifact directory are accepted.

The fixed protocol is counts 10/30/60/83 × stable/inflight_attempt × repeats 0..4, duration .2 and `run_stage1_tests G09 actualbenchmark`. Every protocol field is checked individually; only the actual output paths differ and are separately recorded. Full mode reads the ordered R0 B report and requires exactly 22 original + 2 persisted A + 16 new B rows, matching each real result.json. It never substitutes outer fallback analytic archives, globs arbitrary attempts, or changes source labels to the current documentation HEAD.

R0 must be the byte-verified 24-source export, commit `1c5afd37c891a0ed4776f64367fc7264db3c1c54`, source digest `cd76b4c14b0a48d6d2a71ec7c63ce77010e6ea019b545807635b926eb2984891`. OPT retains its actual `afb61940886add31e472738bf9ba654199c9fa1c` label. The OPT preflight 96-file before map, real `core_hashes_after.json`/`core_hashes_comparison.json`, and each row's source map are checked against the comparison-time source files, then those actual source hashes are checked again after comparison. `completion_observed.json` is terminal observation evidence; its UTC is not presented as precise completion time or used to calculate speed.

Full mode requires OPT's actual exit 0, 94 terminal case results, G09 PASS and 40 COMPLETED rows in benchmark_results.json. R0 B must have `G09_RECOVERY_CHECKED`, internal/driver/supervisor exit 0, exact 22/2/16 counts, unchanged recorded source/input maps, strict final R0 consumer child evidence and original G09 PASS. Missing/incomplete terminal prerequisites yield `NOT_READY`/exit 2 with no simulation. Source/protocol/physical differences yield specific evidence and exit 1; they are not ignored or diagnosed by the tool.

Both sides' 80 real job ReplayBundles are validated in one isolated `provenance.run_reference(modules=())` R0 query with a file request. It verifies archive/result serialized metrics, provenance, actual airborne/inflight/simulated/state counts and positive timings. Each pair compares true bytes and SHA256 of config.yaml plus states/events/rejections/trajectories/manifest/metrics (seven files); index/provenance are individually validated, not required to be equal between different sources. Read input files and the fixed R0 export are checked before/after. All Python processes use `-I -B`, dontwrite before project imports and six native-thread variables 1.

`comparison.json` retains differences, 40 pairs/80 validations/280 member comparisons and eight groups of five original wall/CPU samples with median and nearest-rank P95. Any observed R0/OPT ratio is labeled `diagnostic_profiled_observation_not_formal_speedup`: these are cProfile-on .2 sim jobs with concurrency contention; R0 combines 22 original worker-history samples and 18 fresh A/B samples, while OPT used its original worker. They are not six-window cold-cache or full-traffic qualification. Human status remains `HUMAN_PENDING`.

The original diagnostic RSS fields may contain 0 with `psutil.NOT_INSTALLED`; that is unavailable measurement, not real zero memory. This tool reports original wall/CPU samples and does not turn that RSS value into a memory claim.

## Historical preparation commands

At the preparation stage, the one-pair self-check and terminal-readiness gate check ran; the full 40-pair comparison had not yet run. The completed comparison is recorded below. The self-check reads just existing n10 stable repeat 0 on each side, validates two archives and seven members, computes/serializes one-sample timing summaries, and rejects in-memory protocol mutation/missing-ID/duplicate-ID negatives. It does not depend on B finishing and does not claim the formal 40 mode ran. The self-check output below now exists and must not be overwritten.

```powershell
& 'D:\7_codespace\terminal\venv\Scripts\python.exe' -I -B 'D:\7_codespace\terminal\artifacts\rotating_slots\final_pass_g09_compare_local_20261019a\compare_g09.py' --self-check --output 'D:\7_codespace\terminal\artifacts\rotating_slots\final_pass_g09_compare_local_20261019a\self_check_20261019a'
```

The following command was prepared then and subsequently executed after primary review. Its output now exists and must not be overwritten (the earlier readiness check used a separate output):

```powershell
& 'D:\7_codespace\terminal\venv\Scripts\python.exe' -I -B 'D:\7_codespace\terminal\artifacts\rotating_slots\final_pass_g09_compare_local_20261019a\compare_g09.py' --compare --output 'D:\7_codespace\terminal\artifacts\rotating_slots\final_pass_g09_compare_local_20261019a\compare_20261019a'
```

`--help` writes nothing. No original case PASS or performance acceptance is assigned by this tool.

## Historical preparation evidence

AST and `--help` completed with exit 0. The real one-pair self-check ran UTC 08:40:17.455346–08:40:20.250630 (outer 2.797 s), parent PID 17364 / strict R0 validation child 48960, exit 0. `self_check_20261019a/comparison.json` records `SELF_CHECK_RECORDED`, 1 pair/2 validations/7 member comparisons, no differences, and `formal_40_comparison_executed=false`. Changed-protocol, missing-ID and duplicate-ID in-memory negatives were each rejected. The one group has sample_size=1; its actual raw wall/CPU and median/nearest-rank-P95 fields were serialized, without treating n=1 as five repeats or a formal speedup.

All 154 read input-file hashes, actual OPT before/after/current source checks and the fixed 24-member R0 export remained unchanged. Detailed R0 import/source evidence is `self_check_20261019a/reference_evidence.json`; input maps are `input_hashes_before.json` and `input_hashes_after.json`. Comparison SHA256 is `69cd4c4caf67790ad3d912415ca16410034b26efe68cd82931ead58fe3b3799f`. Exact outer argv/PID/UTC/stdout/stderr/numeric exit are retained with prefix `self_check_20261019a.*`. No formal 40 comparison, simulation, recovery launch, Git operation or server command was executed during that preparation stage.

The separate readiness check ran UTC 08:44:21.561656–08:44:23.258060, child PID 46892, outer 1.688 s. Its actual child exit was 2 (the enclosing PowerShell tool reported exit 1); both driver and outer numeric-exit artifacts record 2. `readiness_check_20261019a/comparison.json` records `NOT_READY` because `run_20261019b/summary.json` was absent, after the OPT terminal/ordered-job checks. Pair/validation/member counts are all 0 and `formal_40_comparison_executed=false`; input hashes, final OPT source hashes and the fixed R0 export remained unchanged. The stderr log is empty. This is a missing-prerequisite gate result, not a completed 40-pair comparison. Exact argv/PID/UTC/logs are retained with prefix `readiness_check_20261019a.*`.

## Completed 40-job comparison (2026-10-19)

[Comparison](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_g09_compare_local_20261019a/compare_20261019a/comparison.json) records `COMPARISON_RECORDED`, `formal_40_comparison_executed=true`, 40 pairs / 80 ReplayBundle validations / 280 config-plus-physical-member comparisons, all bytes equal and no differences. All 1127 read-input hashes, the 24-source R0 export and 96 OPT source files remained unchanged before/after. The [outer argv/timestamps](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_g09_compare_local_20261019a/compare_20261019a.argv.json) record UTC 09:14:31.9371977–09:14:43.3106283; [wrapper numeric exit](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_g09_compare_local_20261019a/compare_20261019a.numericexit.txt) and [driver numeric exit](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_g09_compare_local_20261019a/compare_20261019a/numericexitcode.txt) are 0. Internal comparison wall was 10.375 s.

Each row below has five original profiled observations. Time values are seconds and ratios are dimensionless, displayed to three decimals; the JSON retains raw values and nearest-rank P95. The ratio column is R0 median wall / OPT median wall, labeled `diagnostic_profiled_observation_not_formal_speedup`.

| Count | Mode | R0 wall median | OPT wall median | R0 CPU median | OPT CPU median | Observed wall ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 10 | stable | 24.433 | 25.347 | 24.250 | 25.094 | 0.964 |
| 10 | inflight_attempt | 29.509 | 30.725 | 29.203 | 30.359 | 0.960 |
| 30 | stable | 67.443 | 70.579 | 66.922 | 69.547 | 0.956 |
| 30 | inflight_attempt | 74.560 | 79.336 | 73.906 | 78.281 | 0.940 |
| 60 | stable | 197.764 | 189.184 | 195.266 | 187.672 | 1.045 |
| 60 | inflight_attempt | 197.050 | 195.645 | 194.859 | 193.812 | 1.007 |
| 83 | stable | 264.273 | 273.741 | 260.734 | 270.516 | 0.965 |
| 83 | inflight_attempt | 546.280 | 537.928 | 533.750 | 531.625 | 1.016 |

These cProfile-on .2 sim-second jobs include concurrent contention and different process histories: 22 original R0 worker-history samples plus 18 fresh A/B samples versus OPT worker reuse. Their ratios are reported observations, not six-window cold-cache speed measurements or performance thresholds; no further tuning is implied. RSS values recorded as 0 because psutil was unavailable are missing measurements, not zero memory. The original interrupted R0 run and A failure remain preserved; G09 comparison does not establish the complete original 92-case run, all 70 added catalogue items or human acceptance. `HUMAN_PENDING` remains.
