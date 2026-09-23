# 应急优先级 Guard V2 配对诊断（2026-09-23）

## 结论边界

本轮是 Guard V2 的独立 fresh-seed 配对诊断，比较 rule/original K3 在 guard off/on 下的固定场景结果。它没有训练、优化器更新、Teacher 或 PPO 授权；V2 结果不能写成已经修复，也不能作为 checkpoint promotion 依据。

8/8 formal records 均到达终态，request accounting、exact action replay、SafetyFilter 安全计数和 Teacher-free 检查通过。聚合结果为 `33 normal + 33 emergency + 24 upstream-divert + 6 failed_energy`；guard-on 有 `4` 次 failed energy，guard-off 有 `2` 次。`guard_on_zero_failed_energy` gate 失败，aggregate `PASS=false`，`PPO_ALLOWED=false`。

单纯调整能量阈值和抢留 pad 都没有实现零 failed-energy。当前证据指向计划承诺时的路线、时间和电量可达性与应急认证、资源时隙之间的耦合；不能靠当前 PPO 的 reward 设计保证零失败。因此 V2 不晋级，guard 默认关闭。

## 冻结协议与 provenance

- 协议：[emergency_priority_guard_pilot_v2.yaml](../../../terminal_marl/configs/low_v4_2/emergency_priority_guard_pilot_v2.yaml)，SHA256 `151ae69248626d7f1afcfa8e0fb0c5e0193f86552fd620167ec054bddaa8943f`。
- runner：[diagnose_emergency_priority_guard_pilot_v2.py](../../../terminal_marl/training/diagnose_emergency_priority_guard_pilot_v2.py)，SHA256 `a873cd8bcfdda636f20dc366b5ea540a1a4ec5f703531170c80a703c17e74a44`。
- frozen original K3 checkpoint：SHA256 `4b8be66c100aaaeb5b5699dcce68a96dbe1237964e4c308e7b23b95c62fb8c8c`。
- formal matrix：fresh seeds `840501/840502` × `rule/original_k3` × `guard off/on`，共 8 条记录；未重跑 V1 的 `840401/840402`。
- formal manifest：[manifest.json](../../../../training_outputs/terminal/emergency_priority_guard_pilot_v2_20260923/manifest.json)，SHA256 `bfbdd421ad8fe72a084b42a24ff28e1389796acd898175c48f4917d4c9fdc440`。
- protocol 的 source bindings 固定了当前 `config.py`、`uav.py`、`parallel_env.py` 和 V1 runner 哈希；V2 保持 `PPO_ALLOWED=false`。

## Formal 结果

表中四个结果依次为 `normal / emergency / upstream-divert / failed_energy`。

| controller | seed | guard off | guard on |
|---|---:|---:|---:|
| rule | 840501 | `5 / 5 / 2 / 0` | `6 / 4 / 2 / 0` |
| original K3 | 840501 | `5 / 5 / 2 / 0` | `3 / 5 / 2 / 2` |
| rule | 840502 | `3 / 4 / 4 / 1` | `5 / 3 / 4 / 0` |
| original K3 | 840502 | `3 / 4 / 4 / 1` | `3 / 3 / 4 / 2` |

全部 8 条记录均通过终态、请求记账、exact replay、安全计数为零和 Teacher-free；guard-on 的 failed-energy hard gate 仍失败。结果只用于诊断 Guard V2 的资源与能量语义。

## 只读 failure evidence

guard-on 的 original K3 失败目标为：seed `840501` 的 id `1`、id `6`，以及 seed `840502` 的 id `1`、id `2`。它们的 release history 多次出现 `emergency_energy_guard`，对应重复次数分别为 `320 / 214 / 382 / 242`；最后窗口显示 6 个 LZ 均为 free。

guard-off 的 seed `840502` id `1` 因长时间 `safety_wait` 失败。该事实说明“保留 pad”本身不能解决已承诺路径的能量和时间可达性；它也不证明应放宽 `80/40/10 m` SafetyFilter。

当前 `_warm_start()` 直接注入显式 SOC `.35–.55`，没有经过 admission check；该压力场景包含低于单机无拥堵名义下限 `.4106` 的初始状态。因此这些状态必须单独审视初始可达性，不能把它们直接当作常规随机 SOC 下限的证据。

## 待实现的下一步方案

以下是待实现方案，不是已完成修复：

1. 在每架 UAV 的计划承诺前建立可达能量/时间预算，预算同时检查候选应急路径、LZ earliest usable/service 时隙和 `15 s` 余量。不满足正常准入条件时上游分流；warm-start 的低 SOC stress 状态必须存在可达的应急或替代点，否则标记为先天不可行。
2. 只有在存在可认证的 abort/hover/返环轨迹、足够 SOC、原子切换和释放 LZ 条件时，才允许抢占已分配但尚未降落的健康 UAV。SafetyFilter 的 `80/40/10 m` 条件保持不变；无法证明安全时，需要物理备降点或应急通道。
3. 规则层和安全层通过后，再扩展共享 policy 观测，包括自身剩余可达时间/能量、其他 UAV 最紧迫 deadline、每个 pad 的可用时刻/预留，以及当前 `i2f` 占用的影响，并加入非法动作 mask。训练和资格评估分开处理 normal 与 stress，`failed_energy=0` 作为硬门槛；PPO 继续禁用。
