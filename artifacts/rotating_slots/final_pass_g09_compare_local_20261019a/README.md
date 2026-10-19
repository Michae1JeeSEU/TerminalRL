# Fixed G09 read-only comparison preparation

This local one-off tool compares the actual 40 original G09 benchmark jobs. It never starts a simulation, resumes recovery, changes physical inputs or launches B. Existing R0/OPT inputs remain read-only. Only fresh output children of this artifact directory are accepted.

The fixed protocol is counts 10/30/60/83 × stable/inflight_attempt × repeats 0..4, duration .2 and `run_stage1_tests G09 actualbenchmark`. Every protocol field is checked individually; only the actual output paths differ and are separately recorded. Full mode reads the ordered R0 B report and requires exactly 22 original + 2 persisted A + 16 new B rows, matching each real result.json. It never substitutes outer fallback analytic archives, globs arbitrary attempts, or changes source labels to the current documentation HEAD.

R0 must be the byte-verified 24-source export, commit `1c5afd37c891a0ed4776f64367fc7264db3c1c54`, source digest `cd76b4c14b0a48d6d2a71ec7c63ce77010e6ea019b545807635b926eb2984891`. OPT retains its actual `afb61940886add31e472738bf9ba654199c9fa1c` label. The OPT preflight 96-file before map, real `core_hashes_after.json`/`core_hashes_comparison.json`, and each row's source map are checked against the comparison-time source files, then those actual source hashes are checked again after comparison. `completion_observed.json` is terminal observation evidence; its UTC is not presented as precise completion time or used to calculate speed.

Full mode requires OPT's actual exit 0, 94 terminal case results, G09 PASS and 40 COMPLETED rows in benchmark_results.json. R0 B must have `G09_RECOVERY_CHECKED`, internal/driver/supervisor exit 0, exact 22/2/16 counts, unchanged recorded source/input maps, strict final R0 consumer child evidence and original G09 PASS. Missing/incomplete terminal prerequisites yield `NOT_READY`/exit 2 with no simulation. Source/protocol/physical differences yield specific evidence and exit 1; they are not ignored or diagnosed by the tool.

Both sides' 80 real job ReplayBundles are validated in one isolated `provenance.run_reference(modules=())` R0 query with a file request. It verifies archive/result serialized metrics, provenance, actual airborne/inflight/simulated/state counts and positive timings. Each pair compares true bytes and SHA256 of config.yaml plus states/events/rejections/trajectories/manifest/metrics (seven files); index/provenance are individually validated, not required to be equal between different sources. Read input files and the fixed R0 export are checked before/after. All Python processes use `-I -B`, dontwrite before project imports and six native-thread variables 1.

`comparison.json` retains differences, 40 pairs/80 validations/280 member comparisons and eight groups of five original wall/CPU samples with median and nearest-rank P95. Any observed R0/OPT ratio is labeled `diagnostic_profiled_observation_not_formal_speedup`: these are cProfile-on .2 sim jobs with concurrency contention; R0 combines 22 original worker-history samples and 18 fresh A/B samples, while OPT used its original worker. They are not six-window cold-cache or full-traffic qualification. Human status remains `HUMAN_PENDING`.

The original diagnostic RSS fields may contain 0 with `psutil.NOT_INSTALLED`; that is unavailable measurement, not real zero memory. This tool reports original wall/CPU samples and does not turn that RSS value into a memory claim.

## Commands

The one-pair self-check and the terminal-readiness gate check were executed during preparation; the full 40-pair comparison was not executed. It reads just existing n10 stable repeat 0 on each side, validates two archives and seven members, computes/serializes one-sample timing summaries, and rejects in-memory protocol mutation/missing-ID/duplicate-ID negatives. It does not depend on B finishing and does not claim the formal 40 mode ran. The self-check output below now exists and must not be overwritten.

```powershell
& 'D:\7_codespace\terminal\venv\Scripts\python.exe' -I -B 'D:\7_codespace\terminal\artifacts\rotating_slots\final_pass_g09_compare_local_20261019a\compare_g09.py' --self-check --output 'D:\7_codespace\terminal\artifacts\rotating_slots\final_pass_g09_compare_local_20261019a\self_check_20261019a'
```

Prepared full-comparison output command, **not executed during preparation** and subject to primary review (the readiness check used its own separate output):

```powershell
& 'D:\7_codespace\terminal\venv\Scripts\python.exe' -I -B 'D:\7_codespace\terminal\artifacts\rotating_slots\final_pass_g09_compare_local_20261019a\compare_g09.py' --compare --output 'D:\7_codespace\terminal\artifacts\rotating_slots\final_pass_g09_compare_local_20261019a\compare_20261019a'
```

`--help` writes nothing. No original case PASS or performance acceptance is assigned by this tool.

## Actual preparation evidence

AST and `--help` completed with exit 0. The real one-pair self-check ran UTC 08:40:17.455346–08:40:20.250630 (outer 2.797 s), parent PID 17364 / strict R0 validation child 48960, exit 0. `self_check_20261019a/comparison.json` records `SELF_CHECK_RECORDED`, 1 pair/2 validations/7 member comparisons, no differences, and `formal_40_comparison_executed=false`. Changed-protocol, missing-ID and duplicate-ID in-memory negatives were each rejected. The one group has sample_size=1; its actual raw wall/CPU and median/nearest-rank-P95 fields were serialized, without treating n=1 as five repeats or a formal speedup.

All 154 read input-file hashes, actual OPT before/after/current source checks and the fixed 24-member R0 export remained unchanged. Detailed R0 import/source evidence is `self_check_20261019a/reference_evidence.json`; input maps are `input_hashes_before.json` and `input_hashes_after.json`. Comparison SHA256 is `69cd4c4caf67790ad3d912415ca16410034b26efe68cd82931ead58fe3b3799f`. Exact outer argv/PID/UTC/stdout/stderr/numeric exit are retained with prefix `self_check_20261019a.*`. No formal 40 comparison, simulation, recovery launch, Git operation or server command was executed.

The separate readiness check ran UTC 08:44:21.561656–08:44:23.258060, child PID 46892, outer 1.688 s. Its actual child exit was 2 (the enclosing PowerShell tool reported exit 1); both driver and outer numeric-exit artifacts record 2. `readiness_check_20261019a/comparison.json` records `NOT_READY` because `run_20261019b/summary.json` was absent, after the OPT terminal/ordered-job checks. Pair/validation/member counts are all 0 and `formal_40_comparison_executed=false`; input hashes, final OPT source hashes and the fixed R0 export remained unchanged. The stderr log is empty. This is a missing-prerequisite gate result, not a completed 40-pair comparison. Exact argv/PID/UTC/logs are retained with prefix `readiness_check_20261019a.*`.
