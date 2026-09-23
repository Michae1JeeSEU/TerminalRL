# Emergency reachability preflight（2026-09-23）

## 范围与状态

本页记录两项局部证据：

1. RESERVED LZ 的 release ETA 生命周期修复；
2. 单机 `phase_motion_v2` 应急路线可达性预检。

两项证据都不改变已冻结的 formal 结果，不构成 qualification，也不授权 PPO。

## Opt-in warm-start energy precondition

已加入默认关闭的 `warm_start_energy_reachability_check_enabled`。启用后，warm-start
reset 会在每架 UAV 完成 ring 状态与 steady dynamics 初始化、进入 `active` 前，计算其
当前 ring 到兼容 LZ 的最低单机可执行应急路线 SOC 要求。若采样 SOC 低于该要求，reset
直接报出 uid、ring、初始 SOC 和要求 SOC；实现不裁剪、不重采样，也不把该 UAV 静默分流
到上游请求。

在当前样本中，`.35` OUTER 被拒绝，`.41` 的 OUTER/MIDDLE/INNER 三环小样本通过。该
检查是单机 reset-time 能量前置条件，不包含拥堵、等待、merge slot、LZ service 竞争或
多机时间耦合，因此不提供拥堵场景的可达性保证。关闭该选项时历史 warm-start cohort
保持原有语义。

## RESERVED LZ release ETA 修复

此前 RESERVED LZ 的 `service_remaining` 为 `0`，因此 `earliest_release_time()` 会错误返回当前时刻。当前实现增加 `expected_release_at`：

- 只有 SafetyFilter 通过并且 `_commit()` 成功写入 landing/emergency plan 后，才记录
  `plan.end_time + dt + mission_service_seconds`；
- RESERVED 且 ETA 未知时返回 `inf`；
- RESERVED 且 ETA 已知时返回 `max(now, expected_release_at)`；
- OCCUPIED 时返回 `now + service_remaining`；
- FREE 时返回 `now`；
- release、touchdown 和 service complete 会清理该字段；
- ETA 只接受匹配 RESERVED owner、有限且不早于 `reservation_start` 的值。

定向验证命令覆盖 `tests/test_landing_zone.py`、`tests/test_emergency_energy_v2.py` 和
`tests/test_emergency_priority_guard.py`，结果为 `29 passed`。

## 单机 phase-motion 应急可达性预检

样本固定为：

- energy model：`phase_motion_v2`；
- energy capacity：`1500`；
- emergency trigger：`SOC=.40`；
- emergency energy margin：`15 s`；
- preferred safety separation：`80 m`；
- ring states：OUTER、MIDDLE、INNER；
- 每个 ring 取 4 个角度；
- 6 个 LZ；
- 总计 `3 × 4 × 6 = 36` 条单机 route。

36 条 route 均完成生成并通过 SafetyFilter 认证。按 ring、角度和 LZ 变化得到的最低所需初始 SOC 范围为：

| ring | minimum required initial SOC |
|---|---:|
| OUTER | `.356081–.360308` |
| MIDDLE | `.330841–.332932` |
| INNER | `.295625–.296596` |

在采样的 ring/angle 状态中，`.35` 的 OUTER case 不满足能量可达性；`.40` 和 `.4106` 在全部采样状态中满足该预检的能量条件。

## Standalone candidate energy/LZ budget

新增的独立预算模块可对一个已生成的 candidate trajectory 计算：candidate 能耗、LZ
reported release ETA、ring hold 等待能耗、可执行 emergency fallback route 能耗、landing
reserve 与 configured margin，并返回所选 LZ、等待时间和能量 slack。它同时支持 final
candidate 的 matching free/owned LZ 检查，以及 transfer candidate 的 RESERVED/ OCCUPIED
释放时刻预算。

在 2026-09-23 预检记录时，该模块尚未接入 admission 或 release gate；截至 2026-09-24，
admission pad ETA 与 candidate route/等待/reserve/margin budget 已在 opt-in exploratory
链中接入。fallback route 仍只是基于 reported pad release ETA 的可执行能量下界，未作
SafetyFilter certification 或未来可行性保证。

warm-start、budget、emergency 定向测试共 `24 passed`。

该结果是隔离单机样本，不包含拥堵、等待、merge slot、LZ service 竞争或多机时间耦合；没有据此声明全局无失败，也没有据此改变 formal hard gates。在本页所记录的 2026-09-23 预检时，emergency-priority guard 仍是 opt-in FAIL，`PPO_ALLOWED=false`；截至 2026-09-24，guard 已在独立 opt-in exploratory 链中运行，但 exploratory 结果不构成 qualification，`PPO_ALLOWED=false` 保持不变。

## 2026-09-24 状态更新

后续 exploratory 已使用当前 opt-in V2.3 source-bound 88D、理想 `320 m` 通信范围、warm-start
solo SOC check、admission pad ETA budget 和 candidate route/等待/reserve/margin budget。
SafetyFilter 仍保持原 `80/40/10 m` 契约，fallback future route 仍未作未来安全认证；本页的
单机预检和 budget 证据不应与 exploratory 的多机终态混合。配对结果与性能边界见
[Emergency reachability guard exploratory（2026-09-24）](emergency_reachability_guard_exploratory_20260924.md)。
