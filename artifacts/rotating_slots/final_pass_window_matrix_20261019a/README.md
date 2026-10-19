# Fixed final window matrices — parallel and sustained runs recorded

The fixed parallel and sustained matrices have completed; primary selected workers 6. run_matrix.py is a thin fixed
driver using existing performance.paired.collect and the existing real window
CLI. There is no new systemd wrapper, scheduler, automatic backend selection,
retry, tuning loop or numerical performance acceptance threshold.

## 实际固定矩阵结果（2026-10-19）

[实际 completion](D:/7_codespace/training_outputs/terminal/rotating_slots_perf_v1/final_pass_parallel_matrix_20261019a/completion.json) 为 FIXED_MATRIX_FINISHED，24/24 项 COMPLETED_WINDOW、collector exit 0、CPU complete/worker join，六成员 hash 和 exact timeline 与原 serial baseline 相同；[source finish](D:/7_codespace/training_outputs/terminal/rotating_slots_perf_v1/final_pass_parallel_matrix_20261019a/source_finish.json) 记录源未变，[launch numeric exit](D:/7_codespace/training_outputs/terminal/rotating_slots_perf_v1/final_pass_parallel_matrix_20261019a_launch/numericexitcode.txt) 为 0。相对 OPT serial 的六窗 rollout 几何比（0/1/2/4/6）为 1 / 0.9943168158489465 / 1.2547395282256242 / 1.4362188916975185 / 1.5418654506300054。每个并行配置 n=1，baseline 原三样本保留，不声称统计显著。

主对话选择 workers 6，六窗本轮 rollout 均最快；逐窗 kernel+workers CPU / OPT serial 比为 0.9915–1.0836，最大 kernel+workers RSS sweep 为 3460562944 bytes。collector+kernel 与 kernel+workers RSS 重复包含父进程，不能相加；sweep/HWM 非严格上界。相对 R0 serial 的同六窗几何比为 1.5311947907087373，与相对 OPT serial 的约 1.54 区分。数值仅报告，不追历史门槛。

本更新快照中 sustained 六窗 × 三重复共 18/18 已完成：[completion](D:/7_codespace/training_outputs/terminal/rotating_slots_perf_v1/final_pass_sustained_20261019c/completion.json) 为 FIXED_MATRIX_FINISHED/error=null，同窗三份六成员 hash 与 exact timeline 相同，冷启动/六 worker startup及join/CPU complete/7200 guard DISARMED、physical failed=false。[source finish](D:/7_codespace/training_outputs/terminal/rotating_slots_perf_v1/final_pass_sustained_20261019c/source_finish.json) 为源未变，[launch exit](D:/7_codespace/training_outputs/terminal/rotating_slots_perf_v1/final_pass_sustained_20261019c_launch/numericexitcode.txt) 为 0。各样本先 wall/actual-sim、再窗口中位与六窗几何平均，得到 4.8062896750216 wall秒/sim秒；W71-B/C 高于原每窗≤10历史数字，按用户修订只报告，不调优或宣称全部性能资格通过。

持续作业完整 kernel+workers CPU 为 17077.706637 s，collector 最终快照为 20.48264844 s，后者不与逐 job collector 重复相加且不含最终 write/print 尾；kernel+workers RSS sweep 最大 3613859840 bytes（约 3.365669 GiB），非严格上界。[完整 60 sim R0/OPT oracle](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_sustained_oracle_20261019a/README.md) 仍未得结论，原准备版本为 873a888，预检校验器修复后待重新预检及运行，未记为已启动；OPT三重复一致不能替代该对照。traffic 未运行，70映射与八类人工观察仍 pending/HUMAN_PENDING。下方模板及 historical synthetic self-check 不作为待测项目的执行证据。

Modes are plan, self-check, parallel, sustained and summarize. The script must
be an explicit deployment.extra_paths member before the deployment is finalized;
it uses existing verify_deployment before and after, honoring all expected
extras, and records its own SHA separately. The real window CLI independently
validates R0/corpus/migration/runtime identity. Preparation changes no source,
physical configuration, tests, server or Git state.

parallel is exactly the six frozen windows in contract order, each at worker
counts 1/2/4/6 once, 10 simulated seconds, OPT benchmark: 24 new jobs. Serial
backend 0 is separately read from all three initial OPT trials in an already
completed six-window paired --report-only campaign; all R0 trials remain retained
and verified by paired.summarize. No baseline is rerun or selected by fastest
sample. Each parallel window/backend has n=1, so no empirical tail inference is
made. Medians with n=1 retain that sole sample. The printed 0/1/2/4/6 table shows
validity and geometric OPT-serial rollout ratios over the actually present
windows; incomplete coverage remains explicit. No backend is chosen by code.

sustained requires an explicit primary-selected --inner-workers 0/1/2/4/6. It
runs exactly six windows x three repetitions x 60 simulated seconds: 18 fresh
jobs. It compares each frozen input/cursor/start identity and compares the three
60s replay/timeline results within a window. It does not compare complete 10s
and 60s replay hashes or divide 10s wall times by 60s wall times. A separate full
60s R0/OPT oracle comparison remains pending; this driver does not add that matrix.
Each raw result retains nominal duration and its actual simulated delta. Each
window reports median rollout-wall/actual-sim and end-to-end-wall/actual-sim
computed per sample before taking the median. A backend's geometric mean of
these window rollout medians includes an explicit covered-window count; this is
available for sustained without inventing an equal-duration serial comparison.

Each collect call sequentially joins its one fresh -I -B kernel. The kernel uses
the first max(1,W) CPUs from six explicit primary-approved physical P-core IDs;
the kernel parent shares W cores with its W workers and receives W+1 process
budget. The extra matrix/paired collector process is separately reported. The
collector sets native numerical threads to one and nice10. Physical core IDs
are checked for uniqueness and topology is recorded; no core/backend is chosen
automatically. Each window has an independent Linux deadline, default 1800 wall
seconds. A matching PERFORMANCE_TIMEOUT is retained separately and the remaining
fixed jobs continue. Integrity/physical/implementation errors stop the matrix
without fallback. Exit codes are 0 for all completed windows, 1 for a finished
fixed matrix containing timeouts, and 2 for an integrity/source/physical stop.

Raw collector records, actual argv/stdout/stderr/deadlines/RSS series and all
window outputs remain in each job directory. Completion checks include cold
start, all requested worker startups/joins and complete successful kernel CPU.
Ten-second results must match all six R0/OPT serial replay members and exact
start/stop/delta float.hex values. Sixty-second repetitions match one another.
Rollout and end-to-end wall, rollout-parent CPU, full kernel+workers CPU and
collector CPU are separate fields. Validation CPU is separately observed. The
matrix process CPU field covers through its final snapshot, excludes the final
write/print tail, and must not be added to overlapping per-job collector CPU.
Timeout paths retain observed reaped CPU with incomplete accounting labels.
Collector+kernel RSS and kernel+worker RSS estimates remain separate; lifetime
HWM and sequential PID sweep values are not strict bounds or RSS-limit PASS.

All real input paths, the exact commit/digests, six CPU IDs and a new output
under /home/user/training_outputs/terminal are required. New experiment output
must not already exist or overlap frozen input/reference/baseline directories.
The following are argument templates, not executed commands or real result paths;
replace every angle-bracket value with primary-reviewed existing input paths.

```bash
/home/user/envs/terminal_env/bin/python -I -B artifacts/rotating_slots/final_pass_window_matrix_20261019a/run_matrix.py parallel \
  --corpus-manifest '<CORPUS_JSON>' --corpus-sha256 '<CORPUS_SHA256>' \
  --reference '<R0_EXPORT_DIRECTORY>' --deployment-manifest '<DEPLOYMENT_JSON>' \
  --implementation-commit '<EXACT_COMMIT>' --r0-migration-approval '<R0_APPROVAL_JSON>' \
  --opt-migration-approval '<OPT_APPROVAL_JSON>' --serial-campaign '<COMPLETED_REPORT_ONLY_CAMPAIGN>' \
  --cpus 0 2 4 6 8 10 --output '<NEW_TRAINING_OUTPUT>' --budget 1800
```

Use the same explicit arguments with `plan --plan-for parallel` to inspect all
24 argv arrays without execution. For the sustained plan use
`plan --plan-for sustained --inner-workers <PRIMARY_CHOICE>`; actual sustained
uses `sustained --inner-workers <PRIMARY_CHOICE>`. CPU IDs shown above are a
template requiring the primary's actual machine approval. summarize needs only
`summarize --output '<EXISTING_MATRIX_OUTPUT>'`; it verifies retained evidence
hashes and prints a fresh table, without another simulation or overwriting data.

The local self-check needs no physical inputs or helper imports:

```powershell
.\venv\Scripts\python.exe -I -B artifacts/rotating_slots/final_pass_window_matrix_20261019a/run_matrix.py self-check
```

It checks 24/18 plan counts, order, unique cases, fixed CLI argv and aggregation
using explicitly synthetic times. It writes a new self_check.json beside the
script (or a new --self-check-output path) and refuses to overwrite it. Those
values are not physics or performance observations. Plan prints only, self-check
writes only its own artifact, and summarize reads an existing output; none runs
a kernel. Actual launch requires primary review and authorization.
The sustained aggregate self-check includes 60.2/60.4/60.6 actual simulated
seconds and deliberately varying wall times, checking the median of per-sample
ratios rather than nominal-60 division or a ratio of medians.
