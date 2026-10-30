# 局部流量恢复 R3：实现与验证记录（2026-10-31）

实现已冻结并推送，服务器集成测试通过。正式比较、25 秒 smoke 和人工回放审查仍待完成；目前没有吞吐、能耗或空闲改善结论。

源代码提交为 [`208f5c9c7200d0cbcc2fed7038ce555f43f2d0ed`](https://github.com/Michae1JeeSEU/TerminalRL/commit/208f5c9c7200d0cbcc2fed7038ce555f43f2d0ed)。[源冻结文件](D:/7_codespace/terminal/artifacts/rs_local_recovery_20261031/source_freeze.json) 覆盖 1244 个文件，源哈希为 `833aca8abf16fc6b5e87b6b43bb7dd01065c2a69a217a1acaed2af4db7c110d7`。

## 实现边界

恢复顺序为：保守识别并撤下受影响动作 → 认证并登记保留动作组成的 standby → 在预算内修复原受影响子集。standby 登记前，无法由包络证明安全的变化流量对先归入受影响集合。包络依据实际时间区间上的连续界进行证明，不以环编号代替几何检查。交通或能量绑定发生变化时跟踪变更并重新认证；fallback 的有限能量检查覆盖到下一次 lead 调整。

FULL_H / RULE_LOCAL_K8 的范围、lead20、安全检查、六项 `service_opportunity_v1` 目标及能量模型保持不变；`stage2_service`、downstream 和 frame 的定义未修改。恢复比较在同一新鲜 frame 中对当前可执行前缀计分。恢复入口要求已有 world 签发的 packet，因此这次实现不能保证解决 prepare 之前或搜索阶段的所有失败。未来上游队列和 baseline 模型仍在范围外。

附近物理状态复现只读解码 checkpoint，不接入归档 sink，不推进世界，不 publish。若源、状态或轨迹哈希与归档事件不同，必须标为附近状态复现，不能称为精确历史回放。

## 冻结比较协议

新服务器输出目录为 `$HOME/training_outputs/terminal/rs_local_recovery_20261031`，上一轮账本只读引用 `rs_service_opportunity_supply_r2_20261030/repair_budget_ledger.json`。原始 `rs_selective_quality_a4_20261030/common_initial` 的种子、checkpoint 和配套文件逐字节复用；两个 common prepare 已完成，checkpoint 哈希与前轮一致：

| Root | Common checkpoint SHA-256 | FULL_H CPU | RULE_LOCAL_K8 CPU |
| --- | --- | --- | --- |
| s3r23 | `0f67e40052c6d80ba35efebaf1e47fb113a1aca7046332d20b0c44f8cc29efc7` | 16 | 17 |
| s3r24 | `255f3ce5a3c45b01d3fb59ce5069ce39dec73b420735759ecdd6ce510f5eda77` | 18 | 19 |

五项数值库线程设置均为字面值 `1`。全轮 child wall 上限为 7200 秒，prepare / arm / audit / smoke 分别为 120 / 1200 / 300 / 120 秒。smoke 单独写入 `smoke_only`，模拟 t=20–45；正式窗口仍为 t=20–320。现有 driver 每次只运行指定阶段和单臂。

只读报告使用 [report_service_opportunity_comparison.py](D:/7_codespace/terminal/scripts/stage3/report_service_opportunity_comparison.py)，old root 为 `D:\7_codespace\training_outputs\terminal\rs_service_opportunity_supply_r2_20261030`，new root 为 `D:\7_codespace\training_outputs\terminal\rs_local_recovery_20261031`。已有 [R2 summary](D:/7_codespace/terminal/artifacts/rs_service_opportunity_supply_r2_20261030/results/summary.json) 和 [R2 CSV](D:/7_codespace/terminal/artifacts/rs_service_opportunity_supply_r2_20261030/results/comparison.csv) 保留原样；新报告写入不与输入重叠的全新本地 artifacts 目录。

报告保留 treatment / inherited 正常着陆、逐 pad 物理空闲、全 pad 同时空闲最长段、上游已有需求时 inner 库存为空、实际 M→I 启动/完成、增量能耗、应急、失败及审计证据。`increment_energy_kwh` 只取 `increment.cumulative_delta.energy_kwh`。物理空闲不能直接判为可避免空闲。

关键候选按 candidate hash 去重，分别报告生成、单机检查通过、进入 catalog 和实际 CP 参与数量。主参与率分母是进入 catalog 的关键候选，实际参与要求存在 `actual_solve_events`；生成候选分母另列。M→I 实际时刻按轨迹哈希匹配预测服务时刻；预测值不是实际服务或认证。局部恢复报告包含受影响/保留动作、撤回与级联、真实检查及冲突、证书复用、观察到局部变化后的整批拒绝、original / standby / repaired 交付和超时类别。重复日志副本去重，失败修复的状态与原因独立保留；缺失旧字段为 null，不补零。

## 已完成验证与待办

服务器 `integrated_r3_verified`：175 passed、0 skipped，耗时 82.56 秒，PID `413599`；UTC `2026-10-30T18:12:37Z` 至 `18:13:59Z`（北京时间 10 月 31 日）。日志和回执镜像在 [本地 R3 输出目录](D:/7_codespace/training_outputs/terminal/rs_local_recovery_20261031)。此前错误 cwd 的启动以 exit 4 结束、未运行测试；随后一次为 174 passed、1 项归档兼容失败，修正临时测试缓存后完成上述验证。失败回执保留。

新增 14 项物理恢复用例通过，覆盖 lead20、释放依赖级联、自然状态转换复用、horizon 延长、修复期间 checkpoint、低 SOC fallback、连续 SOC 变化，以及部分修复后剩余动作的 fallback。driver / report 定向测试为 22 passed，覆盖账本、字节复用、窗口边界、关键候选去重、增量能耗、实际回放取数及 deadline 后仅 guard 记录受影响动作的拒绝。

尚待完成：smoke、正式四臂执行、R2/R3 数值比较、最终 replay audit 和人工回放审查。后续结果在这些步骤完成后补充。

## 冷恢复兼容修复（正式运行前）

初次 R3 高压 smoke 两臂均在 wrapper 恢复时因 traffic policy 绑定不匹配而失败，未进入 episode / callback。历史 common 保存的是 legacy 初始化器，新 treatment 构造器根据 service profile 选择局部恢复策略。修复仅在 `restore_common` 的冷切换处允许使用新 wrapper 策略：必须同时满足无 pending normal work、world 中无签发 packet；保留物理状态、CPU 时钟与执行历史，并记录 initializer / method traffic policy。通用 `LatencyAwareScheduler.load_state_dict` 的绑定检查保持严格。

原 R3 输出、失败回执、账本和源冻结文件保留。后续输出使用 `rs_local_recovery_restorefix_20261031`，其 child wall 上限严格为 `7200 - 原 R3 settled_child_wall_s`，要求原任务全部终态，并记录原账本哈希、已支出及兼容修复原因。当前镜像原账本四项任务均为终态，已支出 `8.607537163188681` 秒；不重置这部分预算。新根仍使用相同 common 字节、CPU16–19、单线程和 t=20–45 smoke，另建提交后的源冻结文件。

本地 `-B` 定向验证通过 46 项，涵盖冷 legacy→local 切换的物理/时钟一致、pending / packet 拒绝、通用绑定拒绝和剩余预算承接。服务器复核、新 smoke 和正式结果仍待完成。NUMEXPR 设置错误的一次 K8 单 callback 诊断不作为可比较性能证据。
