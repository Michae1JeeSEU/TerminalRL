# One-off R0 visual scenario material mapping

This script leaves the original R0 V01–V11 failures and the fixed 24-source export unchanged. It does not call `run_visual`, import UI inside R0, relabel old results, replace physical functions, or create an alternative viewer. A successful extraction is `R0_SCENARIO_MATERIAL`, never an original V-case PASS. Human acceptance remains `HUMAN_PENDING`.

The fixed six material groups are V01=`threshold_crossing`, V02=`three_laps_controls`, V04=`full_lifecycle`, V06=`six_pads`, V07=`all_ring_emergencies`, V08=`busy_failure` then `busy_release`. Only the first four need new R0 scenario execution. V07 reads the existing R0 F01 `replay_paths`; V08 reads F09 then F08 paths from `final_pass_reference_regression_local_20261019a`. Those source archives retain their actual paths/identities. V03/V11 map to V02, V05/V09 to V01, and V10 to V04; aliases do not launch more cases.

Fresh groups use the existing `provenance.run_reference(modules=(), timeout=1800)` once each, with fresh isolated `-I -B` interpreters, native threads1, fixed export byte/import checks and original `fixtures.execute_scenario`, `qualification.Evidence/provenance`, `save_replays`. No fallback archive is generated after a scenario exception. `threshold_replay` and `three_laps_replay` remain explicit analytic fixtures from the original scenario, not flight outcomes. World failures that the original physical assertions expect are retained in the scenario result.

At most two independent R0 children run together. The outer thread pool only runs the subprocess helper; kernels are serial. A failed wave stops later waves; an already active companion child is allowed to finish. Reuse never starts a simulation. The common current viewer side only validates/loads archives and queries `ReplayController` at start+.137, end, event jumps and the original query point. It does not instantiate a World or render UI.

Each R0/OPT pair compares actual bytes of `config.yaml` plus six physical members: states, events, rejections, trajectories, manifest, metrics. All member comparisons, differing-file records, first differing byte and raw paths are retained, without rounding or repairing differences. Fresh scenario relative replay names must match; reused V08 paths are mapped in F09→F08 order to OPT V08's recorded replay list, without renaming files. A path-count or bytes difference stops rather than regenerating/retrying. Parent source maps and fixed R0 export are checked before/after; viewer archive files are checked before/after each read.

Only V01 smoke is currently authorized:

```powershell
& 'D:\7_codespace\terminal\venv\Scripts\python.exe' -I -B 'D:\7_codespace\terminal\artifacts\rotating_slots\final_pass_reference_visual_20261019a\run_reference_visual.py' --output 'D:\7_codespace\terminal\artifacts\rotating_slots\final_pass_reference_visual_20261019a\_smoke' --cases V01 --workers 1
```

`--help` writes nothing. Output must be a new child directory of this artifact directory. The default case list is the exact six groups; do not execute it until primary review/authorization. Do not rerun all92 or all11 V cases.

Evidence: `invocation.json`, source/reference before maps, each fresh child's exact `child_argv.json`, original `reference_evidence.json`, captured scenario stdout, actual known child numeric exit, process timing/error, scenario assertions/environment/results, comparison JSON and final summary/numeric exit. The existing helper does not expose successful child stderr; no empty stderr evidence is fabricated. A helper-reported failed returncode or timeout text/output is preserved; unknown child exit is null and has no numeric child file. Main success is an explicit conjunction of selected-case count, material completion, viewer/bytes comparisons and unchanged source/reference; it directly returns numeric0 only for that conjunction. This is material preparation and does not sign human acceptance or performance qualification.

## Executed smoke only

On 2026-10-19, the exact V01 command above completed from 07:26:26.372500 to 07:26:35.791793 UTC (outer wall9.406s). Parent PID29888 and the isolated R0 child PID30048 both returned0. The original `threshold_crossing` executed five physical/fixture assertions, all true, and returned the actual `_smoke/V01/threshold_replay` path. Two common-viewer bundle validations, both seek/event snapshot round-trips and all seven member byte comparisons succeeded, with no recorded differences. Parent source and the fixed24-file R0 export were unchanged before/after. The R0 evidence records16 imported modules and source digest `cd76b4c14b0a48d6d2a71ec7c63ce77010e6ea019b545807635b926eb2984891`.

Raw outer argv/PID/time/exit/stdout/stderr are `_smoke.launch.json`, `_smoke.process.json`, `_smoke.numericexitcode.txt`, `_smoke.stdout.log`, `_smoke.stderr.log`. Detailed results are `_smoke/summary.json` (SHA256 `8223ff4a82a353aa61acd554f9b473d1807fd26b186538345cbdff89f6b0dd88`) and `_smoke/V01/comparison.json` (SHA256 `188f7e20d38a5cef1a5ea2d50aeb147cddf0dd1cbf51c5976d30f122dd245a5a`). Script SHA at this run was `6e5b23b588a49de5db5005a9a845765caeffc565994af7ae53c1427e75dbc1da`. AST and `--help` checks also completed successfully. No other group, UI, full V suite or full92 rerun was executed; original V failures remain unchanged and human status remains pending.

After this smoke, the startup environment list was completed with `BLIS_NUM_THREADS=1`, bringing it to six native-thread variables. Physical generation and comparison logic did not change. Only AST and the exact six-variable list were checked after this addition; the physical smoke was not rerun. Its original script fingerprint above remains the identity of the executed evidence.
