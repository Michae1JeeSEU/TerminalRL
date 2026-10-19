# 2026-10-19 单轮性能结果

本轮六窗 OPT 六 worker 相对 R0 serial 的几何加速为 **1.5312×**（相对 OPT serial 为 **1.5419×**），每个并行配置 n=1。六个名义 10 sim 冷窗口的单核 rollout 未出现整体提速：R0/OPT 耗时中位数比的几何平均为 **0.9930793832**，对应 OPT 几何耗时增加 **0.70%**。各窗有节约也有退化；本报告如实保留首次完整组，不追加调参或择优重跑。用户已将原合同的 2 倍速度等数字门槛修订为报告指标；原数字及历史字段保留作对照，物理、安全、数值和离散等价要求不变。人工状态仍为 **HUMAN_PENDING**。

## 实现与观测范围

本轮实现包含单物理区间的基础轨迹复用、同一 polar jet 复用、Cartesian 单导数计算、root 查询复用及 Bernstein 系数公共子表达式预计算；执行期 state/bounds 复用局限于当前调用。peak cache 默认关闭（[energy.py](D:/7_codespace/terminal/terminal_marl/core/rotating_slots/energy.py)）。这些实现不构成对本轮快慢差异的原因归因。内核定稿标识为 `7a9f5ed`，原单核报告编辑时 HEAD 为 `4c7cbb4`；实际服务器 campaign 使用部署 `4623e38d9b13238f6ca886229c603a5626285698`，不以文档 HEAD 替换运行身份。

[campaign.json](D:/7_codespace/training_outputs/terminal/rotating_slots_perf_v1/final_pass_paired_20261019a/campaign.json) 记录 UTC 09:11:25.100906–10:32:38.045015，CPU 0、nice 10、原生数值线程 1、benchmark 模式。六个冻结输入各按 AB/BA/AB 首组顺序执行，R0/OPT 各三次，共 36 个独立冷进程；36 份结果均为 `COMPLETED_WINDOW`，命名加速缓存初始为空。各窗六份六物理成员 hash 与精确时间线一致，组内等价检查完整；[summary.json](D:/7_codespace/training_outputs/terminal/rotating_slots_perf_v1/final_pass_paired_20261019a/summary.json) 为 `COMPLETED_PAIRED`、errors 空、`INTEGRITY_COMPLETE_REPORT_ONLY`，[外层退出记录](D:/7_codespace/training_outputs/terminal/rotating_slots_perf_v1/final_pass_paired_20261019a_launch/numericexitcode.txt) 为 0。这只覆盖窗口内推进，不代表完整 traffic。

## 名义 10 sim 冷窗口结果

rollout wall 单位为秒，每个实现 n=3。变化率定义为 `(OPT/R0−1)×100%`，正值为耗时增加；比值为 `R0/OPT`，大于 1 表示本窗节约。表中显示值四舍五入，计算使用原始值。

| 窗口 | R0 wall 中位数 | OPT wall 中位数 | R0/OPT | OPT 耗时变化 |
| --- | ---: | ---: | ---: | ---: |
| W71-A | 10.657301 | 9.969478 | 1.068993 | -6.45% |
| W71-B | 321.189467 | 317.237109 | 1.012459 | -1.23% |
| W71-C | 296.833857 | 297.801739 | 0.996750 | +0.33% |
| W60 | 2.261781 | 2.518502 | 0.898066 | +11.35% |
| W120 | 49.661780 | 48.407157 | 1.025918 | -2.53% |
| W180 | 103.674803 | 107.430765 | 0.965038 | +3.62% |

W71-B 在原推进逻辑的自然切点结束，实际 delta 为 `10.055417513109433` sim 秒、stop 为 `731.0000000000002`；两实现和各重复一致。其余五窗实际 delta 均为 10 sim 秒。不能把 36 次均描述为严格 10 sim。W60 耗时增加 11.35%，W180 增加 3.62%；这些均保留为本轮观察。摘要的 nearest-rank P95 每侧仅 n=3，等于该侧三个样本的最大值，不能作为可靠尾部估计或显著性证据。

## CPU 与 RSS 的测量边界

下表为现有摘要的生命周期 CPU 中位数，单位为秒。kernel 子进程均已 join；collector 单列，不能藏入 rollout 主表。每列各自取中位数，不能把两列中位数之和当成总 CPU 中位数。

| 窗口 | R0 joined kernel CPU | OPT joined kernel CPU | R0 collector CPU | OPT collector CPU |
| --- | ---: | ---: | ---: | ---: |
| W71-A | 13.637 | 13.021 | 0.065 | 0.065 |
| W71-B | 327.775 | 323.940 | 0.581 | 0.665 |
| W71-C | 303.652 | 304.186 | 0.542 | 0.609 |
| W60 | 3.685 | 3.963 | 0.033 | 0.034 |
| W120 | 53.107 | 51.824 | 0.159 | 0.165 |
| W180 | 108.097 | 111.770 | 0.245 | 0.261 |

campaign 最终 accounting snapshot 的 36 个 joined kernel CPU 合计为 4855.666958 s，collector 生命周期 CPU 为 11.001992356 s，合计 4866.668950356001 s；collector 最后的 campaign/summary JSON 写出及 stdout 尾部发生在该快照之后，未计入。结果中的完整性字段适用于已 join 的 kernel/owned children，不扩展为未回收后代的完整 CPU 证明。RSS 为 PID sweep 与 lifetime HWM 估计，采样并非严格同时，HWM 也不证明瞬时总 RSS 上界；原始 RSS 序列和估计字段保留在各 case 输出中，本轮不宣称严格资源资格。

## 已完成的相关验证

- [Linux c 联合日志](D:/7_codespace/training_outputs/terminal/rotating_slots_perf_v1/final_pass_linux_tests_20261019c/stdout.log)：14 个测试文件实际收集 263 项，263 passed、无 skip，退出 0；包含真实四种 window 路径、隔离 R0 oracle 与 Linux worker 生命周期证据。这是测试执行数量，不能换算成新增 70 个 catalogue ID 全 PASS。
- [OPT 原 94 项结果](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_regression_local_20261019a/results.json)：93 PASS、V12 HUMAN_PENDING。[79 case 对照](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_regression_compare_local_20261019a/comparison.json) 为 99 对 replay、198 次 validate、693 个配置及物理成员无差异；[G04/G05 只读复核](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_period_compare_local_20261019b/review_summary.json) 另有 6 对、12 次 validate、42 个成员相同。合计 81 个功能 case、105 对、735 个成员；不含 G09 或 V。
- [独立 G09 对照](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_g09_compare_local_20261019a/compare_20261019a/comparison.json)：40 对真实 job、80 次 validate、280 个配置及物理成员 byte 相同；R0 B 已由原 R0 G09 消费检查 PASS。原 cProfile-on、.2 sim、并发争用及 R0 原 22 个 worker 历史样本加 18 个 fresh 样本、对比 OPT worker 复用的差异均保留；比值仅为 `diagnostic_profiled_observation_not_formal_speedup`，不混入上述六窗速度统计。原 G09 因缺 psutil 留下的 RSS 0 是缺测，不是零内存。

原 R0 92 项运行仍为中断，整体退出及原因未知；原 V01–V11 在 UI 导入时、早于场景执行的 FAIL 保留。后来生成或复用 R0 物理归档并由共同 viewer 读取的材料，不把原失败改写为 PASS。[视觉作业单](D:/7_codespace/terminal/docs/rotating_slots_stage1/performance_v1/visual_review_20261019.md) 的最低八类观察均仍为 HUMAN_PENDING，无研究者签收。

## 固定并行矩阵与后端选择

[matrix.json](D:/7_codespace/training_outputs/terminal/rotating_slots_perf_v1/final_pass_parallel_matrix_20261019a/matrix.json) 记录六窗 × workers 1/2/4/6 × 一次，共 24 项，UTC 10:39:21.583816 开始；[completion.json](D:/7_codespace/training_outputs/terminal/rotating_slots_perf_v1/final_pass_parallel_matrix_20261019a/completion.json) 于 UTC 11:22:22.939906 记录 `FIXED_MATRIX_FINISHED`、error=null。24 项均 COMPLETED_WINDOW、collector exit 0、CPU complete、worker 已 join；六个 replay 成员 hash 和精确时间线逐项与对应 serial baseline 一致。[source_finish.json](D:/7_codespace/training_outputs/terminal/rotating_slots_perf_v1/final_pass_parallel_matrix_20261019a/source_finish.json) 为 source_unchanged=true，[外层退出记录](D:/7_codespace/training_outputs/terminal/rotating_slots_perf_v1/final_pass_parallel_matrix_20261019a_launch/numericexitcode.txt) 为 0。名义 10 sim，W71-B 仍为原自然切点 delta 10.055417513109433。

每个并行窗口/后端只有 n=1，下表保留其唯一 rollout wall，单位秒；backend 0 是原 paired 的三个 OPT 样本中位数，未重跑或取最快样本。不能据此声称统计显著或可靠尾部表现。

| 窗口 | backend 0（n=3） | workers 1 | workers 2 | workers 4 | workers 6 |
| --- | ---: | ---: | ---: | ---: | ---: |
| W71-A | 9.969478 | 10.077452 | 7.809977 | 6.919017 | 6.348986 |
| W71-B | 317.237109 | 314.287778 | 259.676937 | 233.661106 | 221.022280 |
| W71-C | 297.801739 | 293.613607 | 240.590218 | 212.796601 | 200.076586 |
| W60 | 2.518502 | 2.599916 | 2.191083 | 2.032475 | 1.974911 |
| W120 | 48.407157 | 49.184469 | 38.753543 | 32.850127 | 30.015411 |
| W180 | 107.430765 | 107.342986 | 76.297678 | 61.190291 | 55.164113 |

汇总以相同六窗为覆盖范围。rollout 比值为 OPT serial / 当前 backend；CPU 比值为当前 kernel+workers / OPT serial 的逐窗比值几何平均。RSS 列为所读 kernel+workers PID sweep 样本中的最大值，单位 bytes；0 后端取原 18 个 OPT 运行，其余各取六次运行，不是严格上界。

| 后端 | 相对 OPT serial rollout 几何比 | kernel+workers CPU 几何比 | 最大 RSS sweep（bytes） |
| --- | ---: | ---: | ---: |
| 0 | 1.000000 | 1.000000 | 743387136 |
| 1 | 0.994317 | 1.003522 | 1212960768 |
| 2 | 1.254740 | 1.019640 | 1683415040 |
| 4 | 1.436219 | 1.026539 | 2624593920 |
| 6 | 1.541865 | 1.023715 | 3460562944 |

主对话已选择 **workers 6**：本轮六窗 rollout 均最快，相对 OPT serial 的几何比为 `1.5418654506300054`；对应 kernel+workers CPU 的逐窗比值范围为 0.9915–1.0836。同六窗 R0 serial / OPT workers 6 的几何比为 `1.5311947907087373`，与相对 OPT serial 的约 1.54 倍区分。这是固定单轮观测上的主对话选择，不是代码自动选型或新调参循环；原单核几何比 0.9930793832 及各窗退化仍保留。

workers 6 最大 kernel+workers RSS sweep 为 3460562944 bytes。collector+kernel 与 kernel+workers 两种 RSS 口径重复包含 kernel 父进程，不能相加或把不同时刻峰值拼成总量。矩阵记录完整 joined kernel CPU 合计 3259.455650 s，矩阵 collector 至最终快照 CPU 为 7.954110060 s；后者与逐 job collector CPU 重叠，不相加，且最终 write/print 尾部仍未计入。RSS sweep/HWM 估计不构成严格资源上界或资源资格通过。

## 此快照仍待测的范围

本次文档更新采用固定并行 24 jobs 已完成、主对话选择后端 6 后的快照：sustained 18 jobs 尚未启动，独立完整 60 sim 的 R0/OPT oracle 仍 pending，15 case / 30 traffic 未运行；均无本报告可引用结果。不能据上述名义 10 sim 结果推断持续表现或完整 traffic 结论；八类人工观察仍 HUMAN_PENDING。历史 `required_geometric_speedup=2.0`、`historical_component_gate=NOT_PASSED`、`performance_status=NOT_QUALIFIED` 字段按原值保存；按用户修订，本轮只报告数字，不以追过这些历史数字门槛追加优化。是否最终签收由主对话在其余既定证据和人工观察完成后决定。
