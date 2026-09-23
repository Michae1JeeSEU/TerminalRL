# Emergency reachability preflight（2026-09-23）

## 范围与状态

本页记录两项局部证据：

1. RESERVED LZ 的 release ETA 生命周期修复；
2. 单机 `phase_motion_v2` 应急路线可达性预检。

两项证据都不改变已冻结的 formal 结果，不构成 qualification，也不授权 PPO。

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

该结果是隔离单机样本，不包含拥堵、等待、merge slot、LZ service 竞争或多机时间耦合；没有据此声明全局无失败，也没有据此改变 formal hard gates。当前 emergency-priority guard 仍是 opt-in FAIL，`PPO_ALLOWED=false`。后续性能与 source-bound 观察继续独立进行。
