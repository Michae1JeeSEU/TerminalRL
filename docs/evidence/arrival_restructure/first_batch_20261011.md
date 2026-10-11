# Arrival restructure：2026-10-11 第一批证据

状态：服务器 smoke、small reference 与 full pilot 均为 COMPLETED；9/9 pilot arms 均为 natural_terminal，整体第一批已完成。
本记录不把 smoke、进程启动或目录存在写为 full-cycle completion。
协议与研究边界见 [研究重构计划](../../research_restructure_plan.md)。

## 源码与运行位置

- 原 checkout：`D:/7_codespace/terminal`；分支 `codex/v4-2-v2-1-formal-i1`。
- 原冻结 HEAD：`99c4ffb5af5c106a1aefedb99dc1fe72ed884c2b`；tag：
  `codex/frozen-pre-restructure-20261011`。原工作目录的 36 个 `.tmp` tracked deletions 保持不动。
- 新 worktree：`D:/7_codespace/terminal/.worktrees/arrival-restructure`；分支 `codex/arrival-restructure`。
- 文档提交：`af328a7`；实现与本批运行 source commit：`59e3f24652c7480087b8110a832105bbd010bea1`。
- 服务器工作目录：`/home/user/projects/terminal/.worktrees/arrival-restructure`，由 Syncthing 同步。
- 产物根：`/home/user/training_outputs/terminal/arrival_restructure_20261011_v2`；
  本地同步根：`D:/7_codespace/training_outputs/terminal/arrival_restructure_20261011_v2`。
- 启动前 5 个 production 文件、3 个新增 test 文件与研究计划的 9 个 SHA256 均与本地一致。
  Python 3.12.3 / NumPy 2.5.1，项目 imports 均解析到新 worktree。

## Protocol V2

| 用途 | 场景或 N | development seed |
|---|---|---|
| 2-tick smoke | flow6_mixed / dense24_fixed / dense24_mixed | 896091 / 896092 / 896093 |
| full pilot | 同上三个场景 | 896101 / 896102 / 896103 |
| small reference | N=3 / 4 / 5 / 6 | 896113 / 896114 / 896115 / 896116 |

Pilot 为三个场景 × `rule_nominal_k3 / energy_slack / rolling_reorder`，共 9 arms。
Reference 每个 N 在完整 n! lexicographic permutation 索引中等距取 6 个；若初始
energy order 未包含，则替换最后一个。每个排列各运行 all-release / serial-release，
即每 N 实际 12、合计 48 rollouts；每 N 24 permutations / 48 candidates 仍只是硬上限。
V2 在首个 full rollout 前登记；缩小实际登记范围仅依据 dense 2-tick 性能观察，
未使用 full outcome 数据挑选候选。Small reference 为 t=0 warm start、SOC=0.8；
pilot 保留原课程 demand/SOC；二者不能混为同一需求条件。

每 rollout 按 native time cap、2400 ticks、7200 wall seconds 中先到者停止；
未自然闭合即 CENSORED/unknown。Smoke 固定 2 ticks。完成 smoke 后允许 pilot 9 workers
与 reference 15 workers 并发，总计最多 24，无 nested pools；四个 BLAS/OpenMP 线程变量均为 1。
Rolling 设置为 horizon=4 ticks、max_orders=4、replan_interval=10、planning_wall_seconds=60。
Planning cap 在 native 调用间协作检查，不能抢占进行中的 native step，不是实时上界。

## 控制与证据边界

Legacy 保留课程原 `downstream_soc_priority` 仲裁；新控制器使用 opt-in `external_dispatch_v1`。
因此 legacy/new 对照含 control-authority 差异，不自动归因为算法优越性。
Energy-slack 使用 detached normal-route 估计；rolling 只重排同一 nominal K3 请求。
Reference serial variant 可改变 WAIT/release，单独登记，不能归因为 order-only benefit。
Committed flight、scheduler slots、pad/LZ、服务、trajectory certification、SafetyFilter、
PredictiveSafetyShield、reserve 与 energy emergency takeover 继续由原生执行层负责。
Task-neutral priority、物理 mission/service metadata 与 normal/emergency outcome 分开保留。

在线调度不读取尚未到达的 future request；这只是访问边界，不构成经过验证的
partial-observation policy 或 POMDP 性能结论。Bounded candidate search 不证明全局最优、
物理不可行性、完整 recursive feasibility 或连续优化最优性。
所有 seeds 属 development，Q untouched，`PPO_ALLOWED=false`；本批无 RL 训练或 admission。

## 验证记录（newly_run；本轮已有日志）

- 新增 core / authority / evaluation suites：15 / 23 / 13，组合 `51 passed / 5.61 s`。
- 既有 17 文件定向回归：`172 passed / 4 baseline mock failures / 2 skipped`。
- Worktree 的 2 个 YAML CRLF 与 ignored Yang setup 条件对应的 8 个环境失败已处理；
  不把这些初始环境失败计为新增实现回归。
- 全量 collection 仍有 3 个既有 `L1SafetyGateError` 缺失导入错误；未宣称全仓通过。
- 本地日志：`tmp_restructure_checks/verified_new.log`、`verified_prepared.log`、
  `regression_A.log` / `regression_B.log` / `regression_C.log` 与相应 baseline logs。

## 服务器 smoke（newly_run）

9/9 arms 为 `CENSORED`、`stop_reason=tick_cap`、`ticks=2`、`integrity_ok=true`。
9/9 independent replay 为 `MATCH`、`matched_steps=2`、`whole_cycle=false`。
Python PID 3222397 在终态检查时已不存在；`smoke.exit` 为 `0`，`smoke.log` 为空。
`smoke/summary.json` 已生成。该结果仅覆盖两步输出与重放，不证明完整到达闭环。

| 场景 | controller | elapsed s | first tick s | planning s | native checks s |
|---|---|---:|---:|---:|---:|
| flow6_mixed | rule_nominal_k3 | 0.015 | 0.012 | <0.001 | <0.001 |
| flow6_mixed | energy_slack | 0.015 | 0.012 | <0.001 | <0.001 |
| flow6_mixed | rolling_reorder | 0.016 | 0.013 | 0.001 | <0.001 |
| dense24_fixed | rule_nominal_k3 | 19.481 | 19.395 | <0.001 | 18.597 |
| dense24_fixed | energy_slack | 23.754 | 22.237 | 7.991 | 14.889 |
| dense24_fixed | rolling_reorder | 85.386 | 84.482 | 71.068 | 13.436 |
| dense24_mixed | rule_nominal_k3 | 17.683 | 17.607 | <0.001 | 16.796 |
| dense24_mixed | energy_slack | 26.007 | 24.131 | 8.369 | 16.776 |
| dense24_mixed | rolling_reorder | 79.220 | 77.123 | 62.042 | 16.245 |

Dense independent replay elapsed 为 14.035–18.943 s；其余精确值在 summary。
Flow6 在两步内尚无请求，故其 smoke 不覆盖 RELEASE 分支。

## 命令与产物语义

以下为实际使用的 module CLI；工作目录与 interpreter 如上，output 必须是新目录。
Smoke、reference 与 pilot 的终态均已核验，为 COMPLETED；9/9 pilot arms 均为 natural_terminal，整体第一批已完成，后续 M4+ 阶段仍未完成。

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/home/user/envs/terminal_env/bin/python
OUT=/home/user/training_outputs/terminal/arrival_restructure_20261011_v2
SHA=59e3f24652c7480087b8110a832105bbd010bea1
$PY -m terminal_marl.scripts.evaluate_arrival_restructure smoke --workers 9 --source-commit "$SHA" --output "$OUT/smoke"
$PY -u -m terminal_marl.scripts.evaluate_arrival_restructure pilot --workers 9 --source-commit "$SHA" --output "$OUT/pilot"
$PY -u -m terminal_marl.scripts.evaluate_arrival_restructure reference --workers 15 --source-commit "$SHA" --output "$OUT/reference"
```

每 arm 保存 `protocol.json`、`config.json`、`initial_state.json`、`steps.jsonl`、
`progress.json`、`result.json`；根目录保存 manifest/summary，wrapper 日志与 PID/exit 独立于 arm。
Tape 保存 requested actions、K3 shortlist、release order 与执行 provenance。
Independent replay 使用相同 config/seed，在 fresh env 逐步比较 canonical state、checked
trajectory hashes、ledger、service/completed、energy/outcomes；仅规范化 opaque UUID，
不只比较最终 totals。Reference 选择有效完整 resolved rollout，按 F/D/U、E、wait、
makespan、energy 排序；只有 N+E=完整请求分母且 F=D=U=not_yet_requested=0，
并且 whole-cycle replay MATCH，才记 joint completion witness；E=0 才记 normal witness。
Failure/diversion rollout 即使自然闭合，也不能记为 joint completion witness。
Censored wait 标为 `accrued_partial`，保留 expected/actually_requested/known_unresolved/
not_yet_requested 分母；不得将 accrued mean 写成完整 full-cycle mean。

## Small reference 完成（newly_run）

`reference/summary.json` 的 48/48 candidates 均 COMPLETED、full_valid=true、integrity_ok=true；
每个 N 登记的 12 个 candidates 均已评估。四个 selected candidates 均为 normal joint
completion witness，independent replay MATCH whole-cycle。`reference.exit.json` 为 exit=0、
wrapper_error=null，整个 reference job elapsed=176.723 s。
这形成了实际 N=3–6 的正常联合可执行到达参考；候选最优仅限各自登记集合。
48 个 rollouts 是四个固定 cohort 的重复 variants，不能当作 216 个独立请求样本。

| N | selected / variant | static UID priority | N/E/F/D/U | total / mean wait s | makespan s | energy | min landing SOC |
|---|---|---|---|---:|---:|---:|---:|
| 3 | n3_candidate00 / all_release | 0,1,2 | 3/0/0/0/0 | 1.356493 / 0.452164 | 361 | 524.470613 | 0.595202 |
| 4 | n4_candidate00 / all_release | 0,1,2,3 | 4/0/0/0/0 | 2.208747 / 0.552187 | 486 | 951.106604 | 0.515576 |
| 5 | n5_candidate06 / all_release | 2,4,3,1,0 | 5/0/0/0/0 | 2.861763 / 0.572353 | 329 | 778.081913 | 0.656810 |
| 6 | n6_candidate10 / all_release | 2,0,1,3,4,5 | 6/0/0/0/0 | 2.198270 / 0.366378 | 409 | 1148.482514 | 0.612497 |

| N | planning / native / elapsed s | replay matched ticks | replay elapsed s |
|---|---:|---:|---:|
| 3 | 0.008 / 11.532 / 14.596 | 361 | 4.988 |
| 4 | 0.010 / 15.516 / 19.737 | 486 | 9.230 |
| 5 | 0.006 / 8.413 / 10.888 | 329 | 8.359 |
| 6 | 0.010 / 36.216 / 41.834 | 409 | 14.450 |

48 个 candidates 的 collisions、NMAC、hard separation violations 与 checked/executed
trajectory hash mismatches 均为 0。完整候选状态与原生 safety telemetry 保存在
[first_batch_metrics.json](first_batch_metrics.json) 的列式数据中；该文件不含 full configs 或
per-arrival 列表。选择项均为完整请求已生成并闭合，wait_measurement=full_cycle。

## Full pilot

当前 9 个 pilot arm 的 `result.json` 均为 COMPLETED，均为 `stop_reason=natural_terminal`，无缺失 arm。
下表列出全部结果，mean wait 统一使用所有请求的 `full_cycle` mean；9/9 pilot arms 的
`full_valid=true`、`integrity_ok=true`，整体第一批已完成。
9/9 rows 均为 `expected=actual`，且 `not_yet_requested=F=D=U=0`；collisions、NMAC、
hard separation violations 与 checked/executed trajectory hash mismatches 均为 0。

| 场景 | controller | N/E/F/D/U | mean wait s (full_cycle) | makespan s | energy | planning s | native checks s | elapsed s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| flow6_mixed | rule_nominal_k3 | 6/0/0/0/0 | 5.516609 | 620 | 1602.602733 | 0.008571 | 27.213616 | 30.343842 |
| flow6_mixed | energy_slack | 6/0/0/0/0 | 5.516609 | 620 | 1602.602733 | 11.036761 | 28.871277 | 43.341506 |
| flow6_mixed | rolling_reorder | 6/0/0/0/0 | 5.516609 | 620 | 1602.602733 | 59.282734 | 44.372945 | 108.708629 |
| dense24_fixed | rule_nominal_k3 | 20/4/0/0/0 | 150.753313 | 1033 | 11479.979349 | 0.026959 | 1128.450500 | 1146.588428 |
| dense24_fixed | energy_slack | 20/4/0/0/0 | 116.561504 | 895 | 9731.381042 | 276.273059 | 658.518434 | 951.832903 |
| dense24_fixed | rolling_reorder | 20/4/0/0/0 | 116.561504 | 895 | 9731.381042 | 1097.971167 | 598.715611 | 1711.051417 |
| dense24_mixed | rule_nominal_k3 | 17/7/0/0/0 | 134.476602 | 1096 | 10544.692712 | 0.027355 | 1005.089996 | 1024.282020 |
| dense24_mixed | energy_slack | 18/6/0/0/0 | 156.419245 | 943 | 10889.707947 | 450.967195 | 990.251744 | 1458.310198 |
| dense24_mixed | rolling_reorder | 22/2/0/0/0 | 156.058133 | 943 | 10739.785309 | 1553.755312 | 753.196573 | 2321.910362 |

flow6_mixed 的三个 controller 结果一致；`energy_slack` 与 `rolling_reorder` 的 planning
时间更长。dense24_fixed 中，`energy_slack` 与旧 `rule_nominal_k3` 同为 20N4E；相对旧 rule，
平均等待减少 22.680635%、总能耗减少 15.231720%、makespan 减少 13.359148%。该差异同时
包含 authority 变化，不能称为普遍优势。

mixed energy 相对旧 rule 为 18N6E vs 17N7E，但全请求 mean wait 由 134.476602 增到
156.419245 秒（+16.317072%）、能耗由 10544.692712 增到 10889.707947（+3.271933%），
makespan 由 1096 降到 943（-13.959854%）、max wait 由 526.477268 降到 396.316857
（-24.722893%），不能只用少一次应急认定全面改善。`min_landing_soc` 分别为 0.166228
（mixed energy）和 0.195310（旧 rule），此处只列结果原值。

本批 wall/planning timings 均采自在同一服务器并发运行，不是独占 CPU 的性能基准。
dense24_fixed 的 `rolling_reorder` 与 `energy_slack` 的 N/E/F/D/U、total/mean/max wait、
makespan、energy、min_landing_soc 完全相同；planning 分别为 1097.971167 s 与 276.273059 s，
额外搜索没有在这个开发 case 改善这些结局指标。
固定 rolling 的完整 tape 记录为 895 ticks、195 replanned、fallback=0、evaluated=224、
timeout=3、native_rollout_steps=897；836 个非空 order 与 `energy_order` 逐项相同，
另有 59 个双空数组，缺字段/null=0；planning 最大值为 84.014637 s。

在同一 external authority 下，mixed rolling 相对 `energy_slack` 少 4 次 E（6→2），
N 由 18 增至 22；mean wait -0.230862%、energy -1.376737%、makespan 相同，
max wait +13.475395%、planning 为 3.445384 倍，`min_landing_soc` 由 0.166228
降至 0.144597。相对旧 rule 为 17N7E→22N2E，mean wait +16.048539%、energy +1.850150%、
max wait -14.579005%、makespan 1096→943；legacy 与 external authority 不同，不能称全面改善或
泛化，且每个 pilot 场景仅有一个 development seed。

`pilot.exit.json` 为 exit=0、wrapper_error=null、job elapsed=2322.026033 s，完成时间为
`2026-10-11T03:38:53.558601+00:00`；父子进程 3223082–3223092 均已退出，monitor
`arrival-reconstruction-runs` 已删除。5 个 production 文件本地/服务器 SHA256 完全一致且源文件
无 staged/unstaged diff；pilot summary/exit/log 本地/服务器 SHA256 一致，log=0 bytes；
summary SHA256 为 `52e0d888dfcdc3e25f33f8c3ca7821ab4b4b15cf9ebb1984a726c1ee91b7a471`，
exit SHA256 为 `aa2c02fccf938a0411eb2688029dc4e73d9227f5ad3b3b236993c83aa1b8c626`。
Small reference 不建立 learned policy、控制器性能优势、全局最优或泛化结论。
