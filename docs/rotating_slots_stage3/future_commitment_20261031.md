# 绝对时刻 first-action 预约 R4：冻结协议与接入记录

本轮检验已选中的未来 first action 是否能在 lead20 发布执行承诺，并在原始绝对 start 启动。尚未运行服务器 smoke 或正式实验，不预写服务质量、补给或安全结果。R3 的中负荷回归和预测/执行前缀残留保留为只读比较背景。

## 固定范围与运行预算

新独立输出根为 `$HOME/training_outputs/terminal/rs_future_commitment_20261031`。R4 的 child wall 总 cap 为 7200 秒，不承接 R3 剩余额度；本轮失败、中止或超时仍计费，已尝试 job ID 不自动重试。prepare / smoke / audit 上限保持 120 / 120 / 300 秒，formal 每臂 watchdog 为 1500 秒。1500 仅容忍 checkpoint/replay I/O，不改变算法 search15、reserve4、ready19、lead20。

两 seed、四臂、CPU16–19、每进程五项数值库线程字面值 1、smoke t=20–45、formal t=20–320 不变。FULL_H / RULE_LOCAL_K8 的 scope、六项 `service_opportunity_v1` 目标、energy、安全及几何参数不变，不扩大 queue baseline 或 high coverage。

common 从原 `rs_selective_quality_a4_20261030/common_initial` 逐字节复制，不重新初始化；s3r23 / s3r24 SHA-256 分别为 `0f67e40052c6d80ba35efebaf1e47fb113a1aca7046332d20b0c44f8cc29efc7` / `255f3ce5a3c45b01d3fb59ce5069ce39dec73b420735759ecdd6ce510f5eda77`。cold handoff 必须无 pending、无签发 packet，保留物理、执行历史和 CPU 时钟；只在 resumed 状态字典中绑定新 wrapper 的 delivery policy，并记录两侧元数据，不给只读 property 赋值，也不弱化通用 loader。

## R3 baseline 与结果路径

报告的 `--old-summary` 明确读取服务器 R3 `rs_local_recovery_serializationfix_20261031/final_report/summary.json` 的 `campaigns.new`，验证四个唯一 COMPLETE 臂与窗口，保留已有 export lineage，标记 `preexisting_artifact` 并记录源 summary SHA。R2 用法 `--old-root` 保持兼容；不能把 R3 原根中缺少末端结果的高负荷臂当成完整 baseline，也不能把 compact 小型表误作完整 summary。

R4 四臂通常直接从新根取数。若 watchdog 仅截断末端 I/O，而完整 checkpoint0300 已有效达到 t=320，可在原账本内另计费，只读恢复并导出到该根 `terminal_export`。原 job 仍保留 BUDGET_LIMITED；不 advance、resume 或重新调用 callback/solver。组合报告仅替换实际存在的逐臂 export，不预设高负荷一定需要重导出。export 同根串行运行，cap180；audit cap300。服务器报告只能写入本轮 training_outputs 的全新报告目录，本地只能 fresh artifacts。

## 遥测口径

`future_reservation_metrics` 用 first-action trajectory hash 联接 reservation ID，预约按 ID 去重。候选动作元数据和最终选择读取 callback 有序 actions / `final_selected_first_actions`；中间 `selected_events` 单列，不等于最终选择。预测 suffix 独立列出，不计入当前 first-action M→I 漏斗；缺少分类元数据的数量在主要摘要显式列出，不能悄悄降低分母。

链条为 generated → actual CP → final selected → delivery certified → published → started → completed / cancelled。`reservation_prepared` 仅在 world 完整等待/动作能量、联合连续安全、资源与依赖认证成功后发出，可作 delivery certificate；搜索 `certified_events` 单独保留。publication/start/completion 时间只能来自实际 event。混合即时 FINAL 和未来 TRANSFER 按每机、每条轨迹/预约计数，不用 task COMMITTED 代替。

除按 hash 去重的漏斗，另保留逐 callback/action 最终选择 occurrence，区分本轮实际发布、已有有效预约、未观察到发布、窗口外尚未到期及绑定未知。这样同一 hash 早前发布不能掩盖后续轮的漏交付；已有有效预约也不能算 missing。流失是阶段事件“未观察到”，不是自动的因果归因。callback/joint-group 的 WAIT、stop 和 actual delivery reason 与 candidate rejection、reservation cancel/guard reason 分开标注作用范围。

预约记录保留 effective、绝对 start/end 和所有事件时刻。主取消指标只计已 published 预约的取消；`previous_state=PENDING_DELIVERY,published=False` 的 discard 另列 pending cancellation，未选 repair/重验证候选的 `reservation_pending_superseded` 另列，均不能当作已预约取消。同一 ID 的旧 pending discard 不抹掉之后成功发布的有效预约。已取消 ID 不计未启动遗漏；窗口外未来预约单列 pending，不能计 missed。错过 start 采用显式 guard/cancel 证据。replace 从 published 的 `replaced_reservation_ids` 和 cancelled 的 `replacement_reservation_id` 去重联接，延后要求替换后绝对 start 更晚，不根据同机不同时间猜测。完整新协议下可观测无事件为 0；旧协议或不支持字段为 null/unknown。逐 occurrence 同时保留 `delivery_selection_binding` / `actual_delivery_selection_binding`。

## 输出检查与待办

本地测试输出在 [tests](D:/7_codespace/terminal/artifacts/rs_future_commitment_20261031/tests)。接入测试包含混合即时/未来动作、预测 suffix 分离、最终选中未发布、同 hash 多轮选择和 existing reservation、窗口外 pending、显式替换/取消去重、错过启动、JSON/CSV 一致以及 R3 new 四臂 baseline 选择。保留原 landings、idle、库存断供、holding、能耗、应急/失败与审计指标。

core 字段冻结后，本地 campaign/common/export 三模块最终集成为 78 passed，证据为 [integration_final.xml](D:/7_codespace/terminal/artifacts/rs_future_commitment_20261031/tests/integration_final.xml)。新增 pending discard / superseded 用例验证它们不计已发布取消，同一 ID 后续成功发布仍可保持有效预约。该批次与核心定向测试分别记录，不相加为唯一测试总数。

当前尚待主对话集成复核、源提交与新 freeze、服务器集成测试、25 秒 smoke、正式四臂及独立审计。报告没有任何 R4 正式结果，也不把本地 schema 测试作为线上保障或服务质量收益证据。
