# 局部流量恢复 R3：实现、四臂结果与验收（2026-10-31）

限定实现与机制测试通过，四臂真实执行、终态导出及独立审计已闭环；本轮不作为整体调度服务质量改进验收或推荐。中负荷两臂 treatment 正常着陆均由 R2 的 7 降到 3，本轮四臂实际 M→I 补给均为 0。现有证据确认了未来选中动作与当前可执行前缀脱节的残留，但不足以把全部回归唯一归因于 energy 或 G。

| 验收层次 | 结论 | 证据边界 |
| --- | --- | --- |
| 限定实现、机制小测试 | 通过 | 含受影响动作撤回、standby、修复、能量与 checkpoint 路径的定向测试 |
| 本次四臂真实执行与审计 | 闭环完成 | 四臂结果 COMPLETE、t=320，独立审计全部 PASS；高负荷为完整终态 checkpoint 的只读重导出 |
| 整体服务质量改进 | 不通过 | 中负荷明显退化，四臂无实际 M→I 补给；四臂没有 standby / repaired 实际交付样本 |

## 数据、源绑定与比较口径

本地可直接检查：[compact_metrics.json](D:/7_codespace/terminal/artifacts/rs_local_recovery_serializationfix_20261031/results/compact_metrics.json)、[compact_metrics.csv](D:/7_codespace/terminal/artifacts/rs_local_recovery_serializationfix_20261031/results/compact_metrics.csv)、[audit_receipts.json](D:/7_codespace/terminal/artifacts/rs_local_recovery_serializationfix_20261031/results/audit_receipts.json)、[transfer_manifest.json](D:/7_codespace/terminal/artifacts/rs_local_recovery_serializationfix_20261031/results/transfer_manifest.json)。小型表与回执经 SSH 只读取回，三份数据的本地 SHA-256 均与服务器一致；compact 为 v2，SHA-256 `b95c566767d76644f7be026c47bb941246645cda459d67dc35bead2946c1e30c`。

完整 `summary.json` / `comparison.csv` 位于服务器 `/home/user/training_outputs/terminal/rs_local_recovery_serializationfix_20261031/final_report`，约 8.2 / 7.7 MB。大回放和 checkpoint 的 Syncthing 同步尚未齐全，不能据本地目录存在就称本地 replay 已可完整打开。下列数值取自已校验的小型报告，取适当小数位；完整精度见 JSON。

算法源提交为 [`ead8b8e7162c8e33efe27d640eeaa4f492f0c0a6`](https://github.com/Michae1JeeSEU/TerminalRL/commit/ead8b8e7162c8e33efe27d640eeaa4f492f0c0a6)，[原源冻结文件](D:/7_codespace/terminal/artifacts/rs_local_recovery_serializationfix_20261031/source_freeze.json) 的 source hash 为 `b8fdcd254723fbd40b358ef9e8673008be7ba065cae960816323c439bb0dc552`。导出工具提交 `34dde002` 不改变原算法源、source commit 或运行 binding。

MF / MK 表示 s3r23 的 FULL_H / RULE_LOCAL_K8，HF / HK 表示 s3r24 对应两臂。R2 基准为只读 `rs_service_opportunity_supply_r2_20261030`，已有 [R2 summary](D:/7_codespace/terminal/artifacts/rs_service_opportunity_supply_r2_20261030/results/summary.json) 与 [R2 CSV](D:/7_codespace/terminal/artifacts/rs_service_opportunity_supply_r2_20261030/results/comparison.csv) 不覆盖。本轮中负荷取原根 `rs_local_recovery_serializationfix_20261031/arms/s3r23`，高负荷取该根 `terminal_export/arms/s3r24`；组合报告逐臂保留 input root 和 export lineage。旧轮缺少的候选/局部恢复遥测保持 null，不补零。

## 正常着陆、补给与物理空闲

窗口均为绝对 t=20–320。正常着陆区分 treatment 和 common 初始化继承 FINAL；增量能耗只读取 `increment.cumulative_delta.energy_kwh`。

| 臂 | treatment 正常 R2→R3 | 继承正常 R2→R3 | 增量能耗 kWh R2→R3 | M→I 实际启动/完成 R2→R3 |
| --- | ---: | ---: | ---: | --- |
| MF | 7→3 | 5→5 | 157.916→157.888 | 7/7→0/0 |
| MK | 7→3 | 5→5 | 158.226→157.870 | 7/7→0/0 |
| HF | 7→7 | 3→3 | 241.933→226.639 | 6/6→0/0 |
| HK | 6→7 | 3→3 | 238.540→226.538 | 6/6→0/0 |

两轮四臂的应急着陆与失败均为 0。较低能耗与较少实际服务/转移同时出现，不能单独作为调度质量改善证据。本轮没有实际 M→I 轨迹可与候选预测服务窗对齐；候选 forecast 不等于真实启动、真实服务或可执行认证。

| 臂 | 总 pad 空闲（pad·s）R2→R3 | 最长全 pad 同时空闲 s R2→R3 | 上游有需求且 inner 库存为空最长 s R2→R3 |
| --- | ---: | ---: | ---: |
| MF | 787.991→1027.000 | 10.000→101.053 | 40.000→209.553 |
| MK | 755.643→1027.000 | 10.000→98.385 | 40.000→206.885 |
| HF | 955.280→983.794 | 10.000→10.000 | 0.000→154.706 |
| HK | 1081.661→970.000 | 33.626→15.279 | 0.000→128.779 |

inner 库存为空指已准入上游需求存在时缺少 HOLD_INNER 库存，是库存断供观测；它不证明所有物理空闲都能避免，也不把未准入未来队列作为需求。可避免性仍需有效联合 witness，缺少 witness 保持未知。

逐 pad 物理空闲秒数如下；总量是六个 pad 的累加，并非模拟时长。

| 轮次/臂 | pad 0 | pad 1 | pad 2 | pad 3 | pad 4 | pad 5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R2 MF | 109.500 | 93.000 | 83.000 | 93.000 | 231.500 | 177.991 |
| R3 MF | 153.000 | 166.500 | 156.500 | 166.500 | 158.000 | 226.500 |
| R2 MK | 109.500 | 93.000 | 83.000 | 93.000 | 220.952 | 156.191 |
| R3 MK | 153.000 | 166.500 | 156.500 | 166.500 | 158.000 | 226.500 |
| R2 HF | 226.500 | 251.640 | 163.140 | 93.000 | 98.000 | 123.000 |
| R3 HF | 123.000 | 153.000 | 216.500 | 226.500 | 128.000 | 136.794 |
| R2 HK | 166.500 | 226.500 | 213.408 | 115.126 | 141.500 | 218.626 |
| R3 HK | 153.000 | 166.500 | 143.000 | 226.500 | 141.500 | 139.500 |

各环 holding 的时间与能量如下，每格为“机群累加秒 / kWh”；能量从保存的 holding joule 除以 3,600,000，不是总增量能耗。

| 轮次/臂 | outer | middle | inner |
| --- | ---: | ---: | ---: |
| R2 MF | 4248.2 / 80.336 | 1166.0 / 24.230 | 730.7 / 18.831 |
| R3 MF | 4248.2 / 80.336 | 2400.0 / 49.875 | 180.9 / 4.662 |
| R2 MK | 4202.7 / 79.475 | 1184.3 / 24.612 | 668.4 / 17.224 |
| R3 MK | 4248.2 / 80.336 | 2400.0 / 49.875 | 186.2 / 4.799 |
| R2 HF | 6009.9 / 113.650 | 2503.7 / 52.030 | 1832.8 / 47.233 |
| R3 HF | 6010.2 / 113.657 | 3600.0 / 74.813 | 553.2 / 14.257 |
| R2 HK | 6009.5 / 113.643 | 2751.2 / 57.173 | 1480.0 / 38.140 |
| R3 HK | 6006.4 / 113.584 | 3600.0 / 74.813 | 555.9 / 14.326 |

## 候选参与、认证与实际交付

关键候选按 candidate hash 去重；实际 CP 参与要求有 `actual_solve_events`。主参与率分母为进入 catalog 的合格关键候选，而不是所有生成后可能被单机安全检查拒绝的候选。R2 无这些遥测。

| R3 臂 | 生成关键候选 | 单机通过 | catalog 合格 / 实际 CP 参与 | 主参与率 | 局部证书复用 | 包络证明 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MF | 97 | 66 | 66 / 66 | 100% | 51 | 2 |
| MK | 117 | 90 | 90 / 90 | 100% | 77 | 3 |
| HF | 93 | 65 | 65 / 65 | 100% | 36 | 1 |
| HK | 98 | 65 | 65 / 65 | 100% | 39 | 1 |

以下两种交付率必须同时看。非空 join 排除无前缀任务及 pending；全部 finished 分母保留已完成的空前缀任务。

| 轮次/臂 | committed 非空 | finished 非空 | 全部 finished | 非空 join | committed 非空 / 全部 finished |
| --- | ---: | ---: | ---: | ---: | ---: |
| R2 MF | 8 | 12 | 14 | 66.7% | 57.1% |
| R3 MF | 3 | 3 | 22 | 100.0% | 13.6% |
| R2 MK | 10 | 12 | 13 | 83.3% | 76.9% |
| R3 MK | 3 | 3 | 31 | 100.0% | 9.7% |
| R2 HF | 10 | 13 | 15 | 76.9% | 66.7% |
| R3 HF | 6 | 6 | 19 | 100.0% | 31.6% |
| R2 HK | 8 | 13 | 14 | 61.5% | 57.1% |
| R3 HK | 6 | 7 | 21 | 85.7% | 28.6% |

| R3 臂 | original 交付 task/plan | standby task/plan | repaired task/plan | 观察到局部变化后的整批拒绝 | 显式局部联合冲突 | deadline 类别 |
| --- | --- | --- | --- | ---: | ---: | --- |
| MF | 3 / 3 | 0 / 0 | 0 / 0 | 0 | 0 | 无 |
| MK | 3 / 3 | 0 / 0 | 0 / 0 | 0 | 0 | 无 |
| HF | 6 / 7 | 0 / 0 | 0 / 0 | 0 | 0 | READY_DEADLINE_MISSED ×1 |
| HK | 6 / 7 | 0 / 0 | 0 / 0 | 0 | 0 | READY_DEADLINE_MISSED ×1 |

本轮合格关键候选进入 CP、局部证书复用和原方案交付均有真实运行观测；standby / repaired 线上交付全为 0，没有这条路径的四臂实际交付样本。受影响动作撤回、保留 standby、repair 的保障证据来自定向小场景测试，不能据此泛称所有线上状态都得到保障。局部整批拒绝为 0 也不等于所有搜索失败或空前缀已解决。

callback 首次非空搜索认证的 wall elapsed 分布如下，沿用 compact 的 P95 口径；n 为有该记录的 callback 数。这是搜索内部时间，不是模拟时刻、ready receipt、实际交付或服务时间。

| 轮次/臂 | n | 中位数 s | P95 s |
| --- | ---: | ---: | ---: |
| R2 MF | 13 | 5.146 | 6.756 |
| R3 MF | 3 | 3.916 | 4.704 |
| R2 MK | 14 | 3.538 | 5.543 |
| R3 MK | 3 | 4.021 | 4.269 |
| R2 HF | 14 | 5.549 | 7.237 |
| R3 HF | 6 | 7.591 | 10.084 |
| R2 HK | 15 | 3.552 | 4.373 |
| R3 HK | 7 | 5.635 | 10.312 |

## 已核对的残留机制

`stage2_planning._executable_start` 在存在 `planning_timing` 时要求动作 `start == effective`，当前 prefix 只收可执行的 first action。中负荷 FULL_H callback index 2 的 `snapshot=50.289792377`、`effective=70.289792377`；末层 phase 5 在同一 final catalog `e858f7aed6609261d840a1204c77bad03c0e03a9f6e28d94fb0ebc20605fc550` 返回 OPTIMAL，`canonical_selected_integer_key` 与 callback `downstream_integer_key` 均为 `[0,-6,0,1768,2923386228,35987]`。

该解选择 `initial_middle_02` 的 TRANSFER `90.3→130.3`，但 callback return 为 `certified=true`、`prefix=[]`，`empty_reason=REASONABLE_WAIT`，范围为 `observed_bounded_catalog`。预测中的未来转移可被选中，却不进入当前可执行前缀；搜索认证不等于交付或实际启动。该残留不能归为未进入 CP 或局部取消，现有证据也不足以归因于能耗权重，或解释为全部回归的唯一根因。附近 P5 状态回放不是精确历史复现；真实 CP 的 P5 单测通过，两类证据保持区分。

本轮维持既定 scope，不继续改六目标、能耗、lead20、安全、queue baseline 或 high coverage。上游库存断供与预测/执行脱节是保留的科学问题，不在本轮追加新实验或协议调整。

## 实现边界、版本与修改文件

恢复顺序为保守识别并撤下受影响动作 → 认证并登记保留动作组成的 standby → 有界修复原受影响子集。standby 登记前，无法由包络证明安全的变化流量对先归入受影响集合。包络依据实际时间区间连续界，不以环编号替代几何。变化的交通/能量绑定被跟踪并重新认证；fallback 有限能量检查覆盖到下一次 lead 调整。恢复候选在同一新鲜 frame 中按当前可执行前缀比较，入口要求 world 已签发 packet；prepare 前及搜索阶段的失败不保证被覆盖。

FULL_H / RULE_LOCAL_K8 scope、六项 `service_opportunity_v1` 目标、energy、lead20 和安全约束未改；`stage2_service`、downstream、frame 定义未改。common 种子、checkpoint 与配套文件逐字节复用，s3r23 / s3r24 SHA-256 分别为 `0f67e40052c6d80ba35efebaf1e47fb113a1aca7046332d20b0c44f8cc29efc7` / `255f3ce5a3c45b01d3fb59ce5069ce39dec73b420735759ecdd6ce510f5eda77`。CPU 映射 MF/MK/HF/HK=16/17/18/19，五项数值库线程均为字面值 1，smoke t=20–45、正式 t=20–320。prepare/arm/audit/smoke 上限分别为 120/1200/300/120 秒。

已推送的实现提交：

| 提交 | 内容 |
| --- | --- |
| [208f5c9](https://github.com/Michae1JeeSEU/TerminalRL/commit/208f5c9c7200d0cbcc2fed7038ce555f43f2d0ed) | 局部恢复、候选 CP 参与遥测、driver/report 与验证 |
| [cfcee48](https://github.com/Michae1JeeSEU/TerminalRL/commit/cfcee480a95f72a7b622ce281cd7da84e70c937f) | 冷 common 恢复兼容及补充验证 |
| [ead8b8e](https://github.com/Michae1JeeSEU/TerminalRL/commit/ead8b8e7162c8e33efe27d640eeaa4f492f0c0a6) | packet 序列化与预算链续跑修复，正式算法冻结版本 |
| [34dde002](https://github.com/Michae1JeeSEU/TerminalRL/commit/34dde00290d0166d19943b3b480e284d9e8659b2) | 终态只读导出、独立审计与组合报告工具 |

本轮实际修改文件按模块列出：

- 核心：[local_traffic_recovery.py](D:/7_codespace/terminal/terminal_marl/core/rotating_slots/local_traffic_recovery.py)、[planning_latency.py](D:/7_codespace/terminal/terminal_marl/core/rotating_slots/planning_latency.py)、[selective_replanning/local_problem.py](D:/7_codespace/terminal/terminal_marl/core/rotating_slots/selective_replanning/local_problem.py)、[stage2_planning.py](D:/7_codespace/terminal/terminal_marl/core/rotating_slots/stage2_planning.py)、[world.py](D:/7_codespace/terminal/terminal_marl/core/rotating_slots/world.py)。
- 执行/报告：[run_service_opportunity_comparison.py](D:/7_codespace/terminal/scripts/stage3/run_service_opportunity_comparison.py)、[run_selective_common_initial_a4.py](D:/7_codespace/terminal/scripts/stage3/run_selective_common_initial_a4.py)、[report_service_opportunity_comparison.py](D:/7_codespace/terminal/scripts/stage3/report_service_opportunity_comparison.py)、[reproduce_local_recovery_round.py](D:/7_codespace/terminal/scripts/stage3/reproduce_local_recovery_round.py)、[export_local_recovery_terminal.py](D:/7_codespace/terminal/scripts/stage3/export_local_recovery_terminal.py)。
- 测试：[test_local_traffic_recovery.py](D:/7_codespace/terminal/tests/test_local_traffic_recovery.py)、[test_service_opportunity_search.py](D:/7_codespace/terminal/tests/test_service_opportunity_search.py)、[test_service_opportunity_campaign.py](D:/7_codespace/terminal/tests/test_service_opportunity_campaign.py)、[test_selective_common_initial_a4.py](D:/7_codespace/terminal/tests/test_selective_common_initial_a4.py)、[test_local_recovery_terminal_export.py](D:/7_codespace/terminal/tests/test_local_recovery_terminal_export.py)。
- 记录：[本文](D:/7_codespace/terminal/docs/rotating_slots_stage3/local_recovery_20261031.md)。最终文档提交由主对话交付时报告，不预写未来提交。

独立批次分别为原集成 175 passed、冷恢复兼容 92 passed、序列化修复 53 passed、预算 32 passed、导出工具服务器 11 passed，批次重叠不能相加为唯一测试数。物理恢复定向测试 16 passed，覆盖 lead20、释放依赖级联、自然转换复用、horizon 延长、修复中 checkpoint、低 SOC / 重复 SOC 变化及部分修复 fallback。附近状态复现只读解码 checkpoint，不接入归档 sink、不推进、不 publish；哈希不同明确标记。早期错误 cwd exit 4、174 pass / 1 归档兼容失败及 NUMEXPR 设置错误的单 callback 诊断均保留，后者不作为可比较性能证据。

## 终态导出、审计与实际费用

原 R3 smoke 因 cold wrapper traffic policy 绑定不匹配失败，未进入 episode/callback；修复只在无 pending、无签发 packet 的冷切换处绑定新策略，通用 load_state_dict 仍严格校验。原 R3 支出 8.607537163 秒。restorefix 两臂 checkpoint mappingproxy 错误 exit 1、两臂获授权 SIGTERM −15，全部保留；其支出 966.798145137 秒。

当前轮中负荷原结果 COMPLETE。高负荷原 job 仍为 BUDGET_LIMITED：HF / HK child wall 为 1200.038645871 / 1200.002831140 秒、exit −9、stop reason JOB_WALL_LIMIT；原回执没有被改为 exit 0。两份 checkpoint_0300 envelope 有效、payload time=320、relative=300。末端导出被 watchdog 截断，算法窗口已完成；relative 295 的 online 约 347/343 秒，relative 300 为 361.444/354.993 秒，边界 output I/O 为 771.595/793.408 秒，不能把整段 1200 秒都算作算法计算。

| 高负荷臂 | 源 checkpoint_0300 SHA-256 | 新导出 child wall s | 新 audit child wall s |
| --- | --- | ---: | ---: |
| HF | `9dda718fa7216fd86e7cdba01c2cbbced45b52349638ff37b86b142c9218213d` | 31.113461422 | 23.202444780 |
| HK | `d0c0c4bda1293a6a20fea483d189e17fa5ec080319384fa6aada35414b77d140` | 31.823257511 | 23.079554789 |

高负荷从原 t=320 checkpoint 串行只导出，advance/callback/resume 均为 0，pending 状态保留；新数据只写 terminal_export。导出 cap180、audit cap300，继续计入原共享账本。receipt 内部 export wall 为 HF 29.288 / HK 30.010 秒；上表 child wall 含启动/退出开销，两者不混用。中负荷 audit child wall 为 MF 9.808759005 / MK 9.959467473 秒。

本轮四臂的 [审计回执](D:/7_codespace/terminal/artifacts/rs_local_recovery_serializationfix_20261031/results/audit_receipts.json) 均为 PASS：continuous separation CERTIFIED、resource/C2 PASS、PVA CHECKED、independent PASS、proposal nonexecution PASS、full_a20_covered=true；所列 issue counts 为 0。R2 标准路径下 MF/MK 审计 PASS，HF/HK 未读到对应独立审计结果，在 compact 保持 null，不冒填 PASS，也不改写旧报告历史。

| 费用层次 | 实际秒数 |
| --- | ---: |
| 原始总 cap | 7200.000000000 |
| 两轮前驱累计已收费 | 975.405682300 |
| 本 serializationfix 根承接 cap | 6224.594317700 |
| 本根 settled（含导出/审计） | 4944.495376240 |
| 全 R3 累计收费 | 5919.901058540 |
| 剩余 | 1280.098941460 |

费用从上一实际 cap 扣减 settled，历史费用链、账本哈希、失败/中止费用均保留，不返还或重置为新 7200 秒。最终值见本地 audit_receipts 的 ledger_summary。

人工回放仍为 HUMAN_PENDING，原大回放尚在同步；这不作为本轮已授权实现、测试与报告交付的前置条件。尚未完成的是人工回放与上述科学残留的后续研究。本轮验收结论维持：限定机制实现通过、四臂执行审计闭环完成，整体调度服务质量改进不予验收或推荐。
