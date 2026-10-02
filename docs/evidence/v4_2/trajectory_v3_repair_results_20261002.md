# 轨迹 V3 修复结果与审核记录（2026-10-02）

状态：用户已同意本轮重大修复。`executable_trajectory_v3` 已以 opt-in 方式实现，当前范围限定为 static geometry；V2 默认路径与 50 条历史 hash 回放保留。本文记录已运行证据和资格边界，不将 P3 标记为已解决或已准入。

## 已实施的实现合同

- V3 使用同一条 quintic 曲线产生 position、velocity、acceleration，并提供全曲线解析的速度、加速度和 altitude bounds。
- 执行前增加 live initial-state guard 与 ring-handoff guard；有限 prefix、同曲线 Safety/runtime error budget 和当前阈值、SOC、reward、动作 authority 继续沿用既有合同。
- V2 default、legacy replay 与 50 条历史 hash 对照保留；V3 只作为 opt-in 物理执行基础。dynamic geometry 尚未 qualification，当前没有真实 Crazyflie/Crazyswarm2 hardware adapter 可核验。

## Newly-run V3 运动学诊断

四 fixture 诊断没有 environment step、training 或 optimizer 操作。修正后的 artifact 为：

`training_outputs/terminal/p3_foundation_trajectory_diagnostic_v3_corrected_20261002.json`，SHA256=`380da099c4595a4370d979f7d9730b782267a48c6de8cb53aa70e11c88e72305`；状态 `KINEMATIC_CONSISTENCY_PASS`，`finding_count=0`，`technical_error_count=0`。

| mode | reservation slot | duration (s) | canonical departure (s) | trajectory hash grouping |
|---|---:|---:|---:|---|
| `legacy` | 0 | 104.74858412577643 | 24.622751480485736 | `c1c127…` |
| `legacy` | 30 | 104.74858412577643 | 24.622751480485736 | `c1c127…` |
| `speed_timing_v1` | 0 | 104.74858412577643 | 24.622751480485736 | `c1c127…` |
| `speed_timing_v1` | 30 | 108.25511970686497 | 30.0 | `c6bb6c…` |

四 fixture 的 position-sampled 最大水平加速度为 `1.2796908 m/s²`，最大垂直加速度绝对值不超过 `0.8205116 m/s²`；同一 artifact 的解析上界为水平 `1.2800000 m/s²`、垂直 `0.8243633 m/s²`。解析导数误差最大值分别为 position→velocity `3.392528924450744e-9`、velocity→acceleration `7.549545766316612e-8`；velocity integral 与 acceleration integral 误差最大值分别为 `4.440892098500626e-14` 与 `9.992007221626409e-15`。解析 altitude 范围约为 `0–40 m`（min=`4.440892098500626e-16`，max=`40.0`）。

这些数值属于 `newly_run` isolated diagnostic。历史 V2 fixture 曾出现 node validator 返回空结果但 position-derived derivative 超出 profile 的记录；V3 artifact 的解析/积分/采样复核为当前物理实现的单独证据，不能替代 closed-loop qualification。

## 已确认的基础实现事实

本轮复核确认了三个 foundation defects：早期 final action proposal 没有进入 physical trajectory（首阶段已实现 opt-in 修复路径）；position/velocity 的执行一致性与 node-only validator 曾不一致；ring braking 的 endpoint-only load 存在 handoff 覆盖缺口。它们足以影响 credit/evidence 的可解释性，但历史 P3 中的因果份额尚未测量，也没有据此确定唯一根因。

既有 critic、larger-batch、gate-fit 等尝试保留为 controlled diagnostic evidence；它们不能单独修复未执行的 proposal 或不一致的物理轨迹。`gamma=lambda=1`、full-MC 等旧结果仍按原 cohort 边界解释。

## 回归、预检与准入边界

- `2×25` 条 V2 historical hash comparison 保持一致；本轮 newly-run safety/dynamics regression 记录为 `123 passed / 1 skipped`、`89.26 s`，持久化日志 SHA 前缀/后缀为 `3272f8…7010`。Unified 首次运行的失败 provenance 保留；最终 retry exit `0`，为 `298 passed / 1 skipped`、`99.21 s`，日志 `/home/user/training_outputs/terminal/trajectory_v3_repair_20261002/unified_named_regression_retry.log`，SHA=`606b5cc5278da7ff3a2637f28099e945c9080b2e5a826e9faa91a8643e94e485`。
- 初始 V3 preflight 记录保留，日志 SHA=`9df70d920006af0879e48355995e5c518b03a176239006014eb1ffc51ed286a5`；随后 retry 为 `CHECK_ONLY_OK`、exit `0`、`arm_count=8`。成功 artifact 为 `/home/user/training_outputs/terminal/p3_foundation_repair_v3_20261002/preflight.json`，SHA=`8f21a8203efbdefcdbb88e9e820ce8c1a7a096179e7244f8cce52eddefc6fc40`。其中 source/load schema=`executable_trajectory_v2`、runtime=`executable_trajectory_v3`、transfer=`frozen_v2_source_checkpoint_to_v3_runtime`、`source_checkpoint_is_v3_trained=false`，source SHA=`3fe7dd1e01d032227521f6fc5c8464fe8d00d98fb0e9f0dbd13411123a27d4ab`。
- loader-specific tests 为 `12 passed`、`0.78 s`，日志 SHA=`f7ffb14763b8aedeb310e1f3b397d195656f6d09f825197e8efb5aa698bc959b`。Unified 首次运行观察到 `297 passed / 1 failed / 1 skipped`；唯一失败为 altitude exact assertion（actual=`1.9999999999999911`，expected=`2.0`），该 provenance 保留。Focused diagnostic 为 `6 passed`、`0.17 s`；仅两处 assertion 改为 `np.testing.assert_allclose(atol=1e-12, rtol=0)`，无 production change。最终 retry 已 exit `0`。
- 首个 compact smoke 为三 source 各 `2 ticks`、`SMOKE_OK`，runtime V3；energy coverage valid，repaired88/90 requested action tape 与 metrics/hash 完全相等。对应 log/JSON SHA=`5560975f3039b9308097eb3adcb7e1586e07fa18cf43834b2ea67878bfafcab6`，wall time=`162.1 s`。compact output 缺少 arm validity、safety/error 与 load/runtime binding 字段，主审尚未接受为 complete smoke。
- Strengthened smoke retry exit `2`、wall=`162.297 s`，日志 `/home/user/training_outputs/terminal/trajectory_v3_repair_20261002/v3_smoke_verified_retry.log`，SHA=`ad384d78593bfe95d289fa4be39b1fef7ecad5499d2f1c39943496b71b3e4885`；错误为 `two-tick smoke arm has a hard safety issue`，失败输出仍缺少上述具体字段。reporter 使用的 `uncertified_joint_safety_action` 是 trace-dump reason，不是 runtime counter；实际初始化/递增 counter 为 `joint_safety_fail_closed`。字段缺失保持 `None/reject`，不弱化安全规则。evaluator-specific tests 为 `14 passed`、`0.77 s`，日志 SHA=`d5943a9aaa948b3f3e93f328500c1913034f75d2c4c784d5076771172a62e62e`；corrected retry pending，frozen flags 保持 false。
- 最终 corrected V3 smoke 已被接受：log=`/home/user/training_outputs/terminal/trajectory_v3_repair_20261002/v3_smoke_counter_mapping_retry.log`，JSON=`/home/user/training_outputs/terminal/p3_foundation_repair_v3_20261002/smoke_verified.json`，SHA=`967426e3fa6bc24e85847ea3e1da9352ef1e473e01197a2aeda26689a8ec3710`；exit `0`、wall=`161.610 s`、`SMOKE_OK`。三臂各 2 ticks 且 `censored=true`，validity valid、`technical_errors=[]`、error/lifecycle error 为 null；safety 全部 none、`uncertified=0`、source=`joint_safety_fail_closed`、invariant=`0`，无 collision/NMAC/protected/hard events，energy coverage valid。source/load/runtime binding 为 V2/V2/V3，使用同一 fixed source SHA；repaired88/90 action tape 与 metrics 最大 diff 为 `0`，hash/outcome/request identity 相等，final trace count=`1/2/2`，per-arm wall=`38.9024/42.4635/42.6102 s`。
- corrected evaluator tests 为 `16 passed`、`0.76 s`，日志 SHA=`14ebcecee0f92e2d789387eb3b1dc42b070b8fca462e2c1406cf81f9c279b7e8`。
- Core implementation commit 为 `f1dfb1034d6517fb1b5197035dc92e3eabe482ba`（24 files）；remote-v 为空，未 push。
- 八臂 zero-training matrix 尚未启动；下一步先做相同两 tick 性能剖析，核对计算热点，再启动已登记八臂无训练对照。当前没有 held-out、training、optimizer step、checkpoint write 或 default promotion，`PPO_ALLOWED=false`。因此 P3 仍未 solved/admitted。

上述结果区分 `newly_run` 与 `preexisting_artifact`，不构成完整 audit 或业务性能结论。相关协议见 [P3 foundation repair protocol](p3_foundation_repair_protocol_20261002.md)；基础链路记录见 [P3 foundation review](p3_foundation_review_20261002.md)。
