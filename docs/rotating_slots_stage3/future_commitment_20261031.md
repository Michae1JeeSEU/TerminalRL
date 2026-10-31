# 绝对时刻 first-action 预约 R4：结果与验收（2026-10-31）

本次原始两个固定负荷 seed、四臂各一次闭环中，未来 first-action 执行承诺的核心标准达到：43 个已发布、窗内到期且未取消的未来预约均实际启动，所有已记录启动与原绝对 start 在 `1e-7 s` 内一致；未解释的到期未启动及显式 missed start 均为 0。四臂正常运行至 t=320、relative=300、exit=0，独立安全审计全部 PASS。这不是重复运行的统计估计；结论限定本次运行，不是所有状态的普遍保障，也不能把全部服务差异唯一归因于预约机制、G 或 energy。

MF/MK 为 s3r23 的 FULL_H/K8，HF/HK 为 s3r24 对应两臂。R3 是已有结果，R4 是本轮实际运行；继承 FINAL 与 treatment 正常着陆分别统计。原始比较、逐 pad 空闲、holding 各环、个体预约及未发布 occurrence 见 [compact JSON](D:/7_codespace/terminal/artifacts/rs_future_commitment_20261031/results/compact_metrics.json) / [CSV](D:/7_codespace/terminal/artifacts/rs_future_commitment_20261031/results/compact_metrics.csv)。

## 四臂服务与执行证据

| 臂 | treatment 正常 R3→R4 | 继承正常 R3→R4 | 增量能耗 kWh R3→R4 | 总 pad 空闲（pad·s）R3→R4 |
| --- | ---: | ---: | ---: | ---: |
| MF | 3→9 | 5→5 | 157.888→157.093 | 1027.000→738.219 |
| MK | 3→9 | 5→5 | 157.870→158.499 | 1027.000→732.817 |
| HF | 7→10 | 3→3 | 226.639→230.857 | 983.794→725.664 |
| HK | 7→9 | 3→3 | 226.538→229.802 | 970.000→926.322 |

R4 最长全 pad 同时空闲均为 10 s，R3 依次为 101.053/98.385/10.000/15.279 s。已准入上游有需求且 HOLD_INNER 库存为空的最长段，R4 均为 0，R3 为 209.553/206.885/154.706/128.779 s。库存断供、物理空闲和“可避免空闲”是不同概念，不能把所有空闲自动判为可避免。两轮四臂实际应急和失败均为 0；energy 只取 `increment.cumulative_delta.energy_kwh`。

M→I 下表按 first-action trajectory hash 去重，后链 suffix 独立保存；R3 实际 M→I 启动/完成四臂均为 0。delivery certified 来自 world `reservation_prepared`，不是搜索内部 certified。

| R4 臂 | generated / actual CP | 最终选中 | delivery certified | published | started | completed | 已发布取消 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MF | 38 / 33 | 9 | 9 | 9 | 9 | 9 | 0 |
| MK | 27 / 18 | 9 | 9 | 9 | 9 | 9 | 0 |
| HF | 32 / 31 | 11 | 9 | 9 | 9 | 9 | 0 |
| HK | 28 / 25 | 16 | 11 | 11 | 7 | 4 | 2 |

合格关键候选实际 CP 参与为 MF74/74、MK69/70、HF60/60、HK57/59，不能概称本轮全部100%。下表涵盖所有动作类型的未来预约，定义为绝对 `start > effective + 1e-7`，与 M→I 子集不同；按 reservation ID 去重。

| R4 臂 | 未来 published | start≤320 | 已发布取消 | started / 准点 | 窗外 pending | replacement / 明确延后 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MF | 9 | 9 | 1 | 8 / 8 | 0 | 1 / 0 |
| MK | 16 | 16 | 3 | 13 / 13 | 0 | 3 / 1 |
| HF | 11 | 11 | 0 | 11 / 11 | 0 | 0 / 0 |
| HK | 17 | 13 | 2 | 11 / 11 | 4 | 2 / 2 |

53 个未来 published = 43 个实际启动 + 6 个已发布取消 + 4 个窗外尚未到期。全部43个启动准点，没有缺失分类/身份/时间元数据或 malformed event；窗外 pending 和已取消 ID 均不计 missed。未发布 pending discard 另计，MK为3，其余0；superseded pending 四臂均0。替换/延后只按显式旧新 ID 关联，不能由同机不同 start 猜测。

交付率保留两种分母：R4 committed 非空 / finished 非空依次为12/12、14/15、10/11、10/12；committed 非空 / 全部 finished 为12/14、14/15、10/11、10/14，即85.7%/93.3%/90.9%/71.4%。R3后一口径为13.6%/9.7%/31.6%/28.6%。未完成任务不在 finished 分母内；较高 join 不能掩盖空前缀或窗口末端 pending。

## 未发布 occurrence 与保留限制

逐 callback/action 的最终选中未发布为 MF0、MK3、HF2、HK8。其中 callback/group 的 `READY_DEADLINE_MISSED` 关联 occurrence 数为0/3/1/5，不能当作9个独立任务或逐轨迹的物理失败原因。其余4个 reason=null，来自 HF callback11 的1个与 HK callback14 的3个：t=320 时任务仍 COMPUTING，result/preparation/ready/finished 均未到达，没有实际 published delivery certificate；这不是已发布预约“消失”。

只读终态证据见 [pending_terminal_observations.json](D:/7_codespace/terminal/artifacts/rs_future_commitment_20261031/pending_terminal_observations.json)。checkpoint 由 delegated server worker 经 Stage2Codec 解码并核验 digest，主对话复核；没有 advance/callback。下表取实际 scheduler.pending；online service 是累计调度服务，不是预约认证单项成本。

| 臂/callback | snapshot→effective | pending RESULT due | pending online s | emergency 类 wall / CPU s | admission wall s | 新预约 revalidation wall s |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| HF/11 | 244.332→264.332 | 341.886 | 85.672 | 79.698 / 79.683 | 5.749 | 0 |
| HK/14 | 287.001→307.001 | 322.714 | 21.813 | 16.742 / 16.741 | 4.515 | 0.443 |

这些 due 均晚于观察末端320。对应时间段 emergency 类35/17次、admission35/17次；已记录服务主要是既有 emergency 检查/处理和 admission，HF该段没有新预约 revalidation。CPU分类名 emergency 不代表发生实际应急事件，实际应急仍为0。不能把85秒累计在线服务称为预约认证成本。执行承诺在本次运行有效，但晚期规划仍受在线服务排队限制，本轮不改算法或扩大研究范围。

## 固定实现与比较协议

一个交付单元认证 `effective 等待 → 原绝对 start 的 first action → end`，不平移时间；后链仍为预测。world 的 soft/registry、依赖、delta guard 和有限 fallback 能量共同维护预约。prepared/actual delivery selection binding 分开记录；中间 CP selected 与最终选择分开，逐 occurrence 区分新发布、已有有效预约、未发布及未知，避免同 hash 历史发布掩盖后续情况。

主取消只计已 published 预约；PENDING_DELIVERY discard、未选 repair/revalidation 的 superseded 独立统计。跨飞机 redirect/非局部替换返回 `NONLOCAL_LATENCY_TRANSACTION`；active 动作不能从该接口续交第二 first action，合法旧约保留。终态清除大证书，保留紧凑身份/时刻/reason。cold handoff 无 pending/issued packet，保留物理、历史、CPU时钟；只绑定 resumed 字典，不给只读 policy property 赋值。

相对 `c1a3d1d`，energy、stage2_service、stage2_downstream、scope/selectors无diff。六目标、几何、安全、search15/reserve4/ready19/lead20不变；CPU16–19、每进程五项数值库线程均1。common从原A4逐字节复用，s3r23/s3r24 SHA分别为 `0f67e40052c6d80ba35efebaf1e47fb113a1aca7046332d20b0c44f8cc29efc7` / `255f3ce5a3c45b01d3fb59ce5069ce39dec73b420735759ecdd6ce510f5eda77`。smoke t20–45，formal t20–320；prepare/smoke/audit cap120/120/300，formal watchdog1500只容忍I/O。R4独立总cap7200，失败费用不重置，未扩展 queue baseline/high coverage。

`--old-summary` 只读选择 R3 final_report 的 new 四臂，验证唯一性并标 `preexisting_artifact`；R4为 `newly_run`。本轮四臂均正常导出，无需 R4 terminal_export；原R3 baseline 的 export lineage保留。四 smoke均COMPLETE/exit0，实际rh_cp_sat，scope19/8/34/8；decoded world预约2/3/3/2，各1 RESERVED、其余EXECUTING，首任务t40发布，未来start53.5晚于smoke末端。smoke只验证其当时发布/保留/checkpoint，不冒充未来完成证据。

## 审计、成本与数据

四臂 saved audit 全PASS：continuous CERTIFIED、resource/C2 PASS、PVA CHECKED、independent PASS、proposal nonexecution PASS、full_a20=true；人工回放 HUMAN_PENDING，不作为本轮交付前置要求。formal收费 wall MF/MK/HF/HK为961.724/885.891/1102.182/1086.582秒。audit内部wall为13.172/12.678/15.237/14.459秒，实际账本child wall为14.609/14.274/16.847/16.035秒，两者不混用。最终账本 settled **4276.508360909764 s**，剩余 **2923.491639090236 s**。首次报告命令cwd错误，未运行或生成文件，随后纠正；没有重跑实验。

本地 [extraction_manifest.json](D:/7_codespace/terminal/artifacts/rs_future_commitment_20261031/results/extraction_manifest.json) 记录输入与输出SHA，JSON/CSV八行逐字段一致。源报告镜像为 `D:\7_codespace\training_outputs\terminal\rs_future_commitment_20261031\final_report\summary.json`，11,917,191 bytes，SHA `021130dbe2719aed371ebe12d51baadff54ef06042d498afa0841529946c1c80`。服务器完整报告位于 `/home/user/training_outputs/terminal/rs_future_commitment_20261031/final_report`。完整future遥测保留在 [详细JSON](D:/7_codespace/terminal/artifacts/rs_future_commitment_20261031/results/future_reservation_details.json)，属于大体积未跟踪证据；原始summary/replay不作为本轮小型提交。逐ID生命周期、未发布原因及task关联均在compact中，null保持未知。

## 验证、版本与文件范围

独立服务器批次：core80 passed/0skip、真实solver14/0skip、integration78/0skip、最终receipt诊断28/0skip；组间重叠，不能加总为200独立测试。本地77pass/3skip已由服务器补验，legacy55pass单列，physical13包含core。接入证据 [integration_final.xml](D:/7_codespace/terminal/artifacts/rs_future_commitment_20261031/tests/integration_final.xml)；提取器仅用小型合成fixture验证并对真实结果做一次manifest/CSV一致性核对，没有新实验。

定向物理回归中，effective=70.289792、绝对start=90.3的动作实际启动并完成，projection=90.447识别其为EXECUTING；另覆盖已实现的重复重规划、原子替换、timeout、低SOC、资源、冲突及混合即时/未来动作检查，不外推为所有状态的保障。

已推送实现：[核心f7bc9c64](https://github.com/Michae1JeeSEU/TerminalRL/commit/f7bc9c64bbcc7d9d5b3ce3f35dccb6d8d634d7b6)、[脚本a1e6d1a4](https://github.com/Michae1JeeSEU/TerminalRL/commit/a1e6d1a4e157e494261fbf5519e85f72627af8da)、[最终receipt e3ae9a96](https://github.com/Michae1JeeSEU/TerminalRL/commit/e3ae9a96bd46a0d7ff548bcc3dd0e1ed22ff42f0)。[源freeze](D:/7_codespace/terminal/artifacts/rs_future_commitment_20261031/source_freeze.json) 绑定e3ae9a96，1249文件，source hash `39818d0a3e0e6187c22241e8fe2021946d34fe6b24c30eaebfd2d608f3288cc7`；rawfreeze未提交。最终文档/数据提交待实际产生后由主对话报告。

准确文件分组如下：

- 核心：[future_commitment.py](D:/7_codespace/terminal/terminal_marl/core/rotating_slots/future_commitment.py)、[local_traffic_recovery.py](D:/7_codespace/terminal/terminal_marl/core/rotating_slots/local_traffic_recovery.py)、[planning_latency.py](D:/7_codespace/terminal/terminal_marl/core/rotating_slots/planning_latency.py)、[stage2_campaign.py](D:/7_codespace/terminal/terminal_marl/core/rotating_slots/stage2_campaign.py)、[stage2_planning.py](D:/7_codespace/terminal/terminal_marl/core/rotating_slots/stage2_planning.py)、[world.py](D:/7_codespace/terminal/terminal_marl/core/rotating_slots/world.py)。
- 执行/报告：[run_service_opportunity_comparison.py](D:/7_codespace/terminal/scripts/stage3/run_service_opportunity_comparison.py)、[run_selective_common_initial_a4.py](D:/7_codespace/terminal/scripts/stage3/run_selective_common_initial_a4.py)、[report_service_opportunity_comparison.py](D:/7_codespace/terminal/scripts/stage3/report_service_opportunity_comparison.py)、[export_local_recovery_terminal.py](D:/7_codespace/terminal/scripts/stage3/export_local_recovery_terminal.py)。
- 测试：[test_rotating_slots_future_commitment.py](D:/7_codespace/terminal/tests/test_rotating_slots_future_commitment.py)、[test_future_commitment_solver.py](D:/7_codespace/terminal/tests/test_future_commitment_solver.py)、[test_rotating_slots_latency_handoff.py](D:/7_codespace/terminal/tests/test_rotating_slots_latency_handoff.py)、[test_rotating_slots_planning_latency.py](D:/7_codespace/terminal/tests/test_rotating_slots_planning_latency.py)、[test_service_opportunity_campaign.py](D:/7_codespace/terminal/tests/test_service_opportunity_campaign.py)、[test_selective_common_initial_a4.py](D:/7_codespace/terminal/tests/test_selective_common_initial_a4.py)、[test_local_recovery_terminal_export.py](D:/7_codespace/terminal/tests/test_local_recovery_terminal_export.py)。
- 本次小型交付：[本文](D:/7_codespace/terminal/docs/rotating_slots_stage3/future_commitment_20261031.md)、[artifact-only提取器](D:/7_codespace/terminal/artifacts/rs_future_commitment_20261031/extract_compact_results.py)、compact JSON/CSV、extraction manifest及pending终态观察JSON；提取验证见 [validate_extractor.py](D:/7_codespace/terminal/artifacts/rs_future_commitment_20261031/tests/validate_extractor.py) 与 `tests/extractor_validation_20261031b`。

本轮核心执行标准在这些运行中达到，服务与M→I观察较R3改善；不泛化到其他状态，不把物理空闲自动判为可避免，也不在本轮继续修改目标、energy、lead、安全或扩大研究范围。
