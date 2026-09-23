# 应急优先级 Guard V1 配对试验（2026-09-23）

## 结论

本轮评估了一个 opt-in 的 emergency-priority guard：guard 默认关闭；开启后，低 SOC UAV 的环间 RELEASE 可能被 `emergency_energy_guard` 拒绝，并在最终放行时优先保留可用降落点。试验使用相同 seed 对比 rule/original K3 的 guard off/on，未进行训练、优化器更新、Teacher 或 label 收集。

8/8 formal records 均到达终态，request accounting、exact action replay、SafetyFilter、安全计数、Teacher-free、无 labels/updates 和 K=3 shortlist 检查均通过。聚合结果为 `28 normal + 51 emergency + 16 upstream-divert + 1 failed_energy`。guard-on 的 `zero_failed_energy` gate 失败，因此本轮 `PASS=false`、`PPO_ALLOWED=false`。

本轮结果不能证明预留降落点能够保证低 SOC UAV 零失败，也不能证明 PPO 能够提供该保证。固定 `.40` 投影切换条件对环间计划形成硬阻断，存在低 SOC UAV 长时间等待的死锁风险；因此 Guard V1 不晋级，保持默认关闭。

## 冻结协议与 provenance

- 协议：[emergency_priority_guard_pilot_v1.yaml](../../../terminal_marl/configs/low_v4_2/emergency_priority_guard_pilot_v1.yaml)，SHA256 `28815c8632b7130563c9e733aa7b3fdb0dd5fc116d232ce88e5efd63114d2fc4`。
- runner：[diagnose_emergency_priority_guard_pilot_v1.py](../../../terminal_marl/training/diagnose_emergency_priority_guard_pilot_v1.py)，SHA256 `c76752b0cd417baa6475199379454b20dc2f34c0e98ae2ee5a7bed3a8ebacc8e`。
- frozen original K3 checkpoint：SHA256 `4b8be66c100aaaeb5b5699dcce68a96dbe1237964e4c308e7b23b95c62fb8c8c`。
- formal matrix：fresh seeds `840401/840402` × `rule/original_k3` × `guard off/on`，共 8 条记录；未重跑 predecessor seeds `840301/840302`。
- formal manifest：[manifest.json](../../../../training_outputs/terminal/emergency_priority_guard_pilot_v1_20260923/manifest.json)，SHA256 `af111076ef98870a1cee930b3396d7555f68be7111cd72c743a9054595da2510`。
- 所有 formal case 的 `PPO_ALLOWED=false`，`guard_telemetry_available=false`；guard 开关只改变显式 opt-in execution guard。

## Formal 结果

表中四个结果依次为 `normal / emergency / upstream-divert / failed_energy`。

| controller | seed | guard off | guard on |
|---|---:|---:|---:|
| rule | 840401 | `2 / 8 / 2 / 0` | `1 / 8 / 2 / 1` |
| original K3 | 840401 | `1 / 9 / 2 / 0` | `1 / 9 / 2 / 0` |
| rule | 840402 | `5 / 5 / 2 / 0` | `7 / 3 / 2 / 0` |
| original K3 | 840402 | `6 / 4 / 2 / 0` | `5 / 5 / 2 / 0` |

全部 8 条记录的终态、请求记账、exact replay、SafetyFilter 安全计数和 runtime K=3 检查通过。安全事件计数为 `0`；failed-energy 只在 guard-on 聚合中出现 `1` 次。

## 失败 witness

只读回放中，rule seed `840401` 的 arrival `1` / uid `1` 初始 SOC 为 `0.4317193579`。

- guard-on：647 次 `o2m` release 尝试全部记录为 `emergency_energy_guard`；该 UAV 在 `t=647 s` 外环电量耗尽，尾部状态显示 6 个 LZ 均为 free。
- guard-off：`t=0` 的 `o2m` release 获得 approved；该 UAV 在 `t=274 s` 应急降落，touchdown 时 SOC 为 `0.3221628884`。

这组对照事实表明，固定 `.40` 的预测能量投影拒绝可能把低 SOC UAV 留在环上，即使后续存在可用降落点；它不构成对放宽 SafetyFilter、取消安全约束或直接启动 PPO 的授权。

## 下一步允许动作

下一轮应移除这条固定 `.40` 的环间硬阻断，改为带证书的可达能量预算和应急替代路径，并继续由现有 SafetyFilter 做最终认证。修复后须另行冻结协议，使用全新 seeds 验证；不得重跑或复用 `840401/840402` 的 formal 结论。完成前保持 `PPO_ALLOWED=false`，Guard V1 保持默认关闭。
