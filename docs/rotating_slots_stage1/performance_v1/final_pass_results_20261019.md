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

## 六 worker 持续窗口实测

[sustained completion](D:/7_codespace/training_outputs/terminal/rotating_slots_perf_v1/final_pass_sustained_20261019c/completion.json) 记录 `FIXED_MATRIX_FINISHED`、error=null，六窗各三次，共 18/18 COMPLETED_WINDOW；同窗三份六物理成员 hash 和精确时间线相同。全部为冷启动，六 worker startup/join、CPU complete、7200 wall 秒 guard 正常 DISARMED，physical failed=false、failure=null。[source finish](D:/7_codespace/training_outputs/terminal/rotating_slots_perf_v1/final_pass_sustained_20261019c/source_finish.json) 为 source_unchanged=true，[外层退出记录](D:/7_codespace/training_outputs/terminal/rotating_slots_perf_v1/final_pass_sustained_20261019c_launch/numericexitcode.txt) 为 0。matrix 起始 UTC 2026-10-19 11:41:36.329447，completion 结束 UTC 15:02:52.360019（北京时间 23:02:52.360019）；时区表示不改变同一时刻，不将这些 UTC 时刻之差作为 monotonic wall 测量。

下表 rollout wall 为原三次观测（秒，显示三位小数），delta 为每次实际 sim 秒；W71-A 的原值为 60.199999999999704，表中简写 60.2。先对每个样本计算 wall/actual-delta，再取同窗中位数，未用名义 60 秒代替实际 delta。

| 窗口 | 三次 rollout wall（秒） | 实际 delta（sim秒） | median wall/actual-sim |
| --- | --- | ---: | ---: |
| W71-A | 139.802 / 141.207 / 140.000 | 60.2 | 2.325580626 |
| W71-B | 1427.486 / 1453.426 / 1427.200 | 60.05541751310943 | 23.769485495 |
| W71-C | 1549.346 / 1533.150 / 1542.699 | 60 | 25.711647918 |
| W60 | 10.073 / 10.060 / 10.116 | 60 | 0.167875445 |
| W120 | 380.679 / 379.167 / 378.391 | 60 | 6.319450778 |
| W180 | 481.800 / 490.527 / 491.665 | 60 | 8.175457167 |

六个窗口上述中位数的几何平均为 **4.8062896750216 wall秒/sim秒**。原每窗 ≤10 wall秒/sim秒的历史数字对照中，W71-B、W71-C 高于 10；按用户修订只报告实测，不追加调优，也不称全部性能资格 PASS。同窗三重复相同只说明 OPT6 重复结果一致，不能替代完整 60 sim R0/OPT 对照；此处不计算 10 sim/60 sim 速度比。

完整 joined kernel+workers CPU 合计 **17077.706637 s**，collector 至最终快照 CPU 为 **20.48264844 s**；后者与逐 job collector 重叠，不能重复相加，最终 write/print 尾部未计。最大 kernel+workers RSS sweep 为 **3613859840 bytes（约 3.365669 GiB）**，仍是采样/HWM 估计，非严格资源上界。

## 旧单轮 60 sim oracle 的中断终态

[oracle completion](D:/7_codespace/training_outputs/terminal/rotating_slots_perf_v1/final_pass_sustained_oracle_20261019a/completion.json) 于服务器 UTC 2026-10-19 17:49:38.124928 记录 **STOPPED_ON_SOURCE_ERROR**；[根退出码](D:/7_codespace/training_outputs/terminal/rotating_slots_perf_v1/final_pass_sustained_oracle_20261019a/numericexitcode.txt) 与 [launch 退出码](D:/7_codespace/training_outputs/terminal/rotating_slots_perf_v1/final_pass_sustained_oracle_20261019a_launch/numericexitcode.txt) 均为 2，outer error 为 `ValueError: outer fixed file inventory changed`。[终态采集记录](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_sustained_oracle_20261019a_progress_2.json) 的本地采集区间为 UTC 18:11:41.7505638–18:11:42.1517773，服务器观测为 18:12:30.666815，native SSH exit 0、匹配进程 0；这些采集时刻不冒充运行结束时刻或 monotonic wall。

| 窗口 | 既有记录状态 | R0 rollout wall（秒） | 对照范围/限制 |
| --- | --- | ---: | --- |
| W71-A | [ORACLE_EXACT](D:/7_codespace/training_outputs/terminal/rotating_slots_perf_v1/final_pass_sustained_oracle_20261019a/windows/W71-A/R0/row.json) | 175.01734341998235 | 三份OPT、18成员及float.hex时间线同；R0仅n=1 |
| W71-B | [ORACLE_EXACT](D:/7_codespace/training_outputs/terminal/rotating_slots_perf_v1/final_pass_sustained_oracle_20261019a/windows/W71-B/R0/row.json) | 1994.411032767006 | 三份OPT、18成员及float.hex时间线同；R0仅n=1 |
| W71-C | [ORACLE_EXACT](D:/7_codespace/training_outputs/terminal/rotating_slots_perf_v1/final_pass_sustained_oracle_20261019a/windows/W71-C/R0/row.json) | 2175.6804914230015 | 三份OPT、18成员及float.hex时间线同；R0仅n=1 |
| W60 | [ORACLE_EXACT](D:/7_codespace/training_outputs/terminal/rotating_slots_perf_v1/final_pass_sustained_oracle_20261019a/windows/W60/R0/row.json) | 12.90339039202081 | 三份OPT、18成员及float.hex时间线同；R0仅n=1 |
| W120 | [IMPLEMENTATION_FAILURE](D:/7_codespace/training_outputs/terminal/rotating_slots_perf_v1/final_pass_sustained_oracle_20261019a/windows/W120/R0/attempt/result.json)；collector exit 1、无row | 不作合格样本 | `ProvenanceError: DEPLOYMENT_BYTES_CHANGED_DURING_RUN: terminal_marl/core/rotating_slots/audit.py` |
| W180 | NOT_RUN | — | 无result/row |

四窗合计 12 pairs / 72 个物理成员字节及 SHA 相同，精确时间线相同；这不是六窗完整 oracle 或正式速度重复表。W120 虽记录 actual delta=60 和 rollout=596.62691618799 s，因上述 IMPLEMENTATION_FAILURE 不纳入合格样本。`source_finish.json` 实际缺失，不补造来源结束证明；[小型事实摘要](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_sustained_oracle_20261019a/interruption_review_20261020.json) 分开记录原始产物、终态采集及主对话后续源码比较的来源。

主对话按原 523-source map 对后续本地源码作只读比较，记录 7 个差异：`audit.py`、`energy.py`、`performance/checkpoint.py`、`replay.py`、`safety.py`、`trajectory.py`、`world.py`（均在 `terminal_marl/core/rotating_slots/` 下）。`c59aa6b` / `e343c77897169ff390c5026ed77c27aa9ea4ba8a` 是观察到的后续工作状态；不据此断言哪条写命令或提交时刻造成运行变化，也不评价新实现。原 deployment manifest 保持不变，后续 HEAD 不冒用原部署 `4623e38d9b13238f6ca886229c603a5626285698`。此前已审 36 paired / 24 parallel / 18 OPT sustained 及单核 0.993079、六worker相对R0 1.5312 的事实保留，本次中断不撤销这些先前证据。

## 此快照仍待测的范围

本次快照中，独立 60 sim oracle 已中断，仅四窗有既有对照，六窗完整结论仍缺。15 case / 30 jobs traffic 为 NOT_RUN；主对话原预定 R0 4 workers / OPT 6 workers，但未 launch。共享源码已变且版本方向待用户确认，本轮不再推进旧 traffic；主对话已删除旧 automation-2（工具成功），这不表示自动工作全部完成。70 映射仍是旧快照；94 新索引的 EVIDENCE_INDEX_REVIEWED 只表示索引忠实性已复核，不是全部准则 PASS。八类人工观察仍 HUMAN_PENDING；相对后续优化版本的选择留待用户决定，本报告不代作选择。历史 `required_geometric_speedup=2.0`、`historical_component_gate=NOT_PASSED`、`performance_status=NOT_QUALIFIED` 字段按原值保存；按用户修订，本轮只报告数字，不以追过这些历史数字门槛追加优化。是否最终签收由主对话在其余既定证据和人工观察完成后决定。
