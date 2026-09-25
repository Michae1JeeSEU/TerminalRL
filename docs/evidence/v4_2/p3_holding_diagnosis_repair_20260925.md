# P3 holding 诊断与修复候选（2026-09-25）

状态：`DRAFT_COVERAGE_FIT_ANCHOR_EVAL_COMPLETE`。本页把
restart1 formal 的失败、holding diagnostic、策略分布 probe、forced-gate 反事实、coverage fit
和 closed-loop anchor evaluation 放在同一证据链中。它记录诊断与候选修复，不把候选修复写成
训练通过、P3 qualification 或容量结论。

## 1. 证据边界

原始 formal 结果见
[P3 V5 restart1 result](p3_controlled_emergency_formal_v5_result_20260925.md)。输出根为：

```text
/home/user/training_outputs/terminal/p3_controlled_emergency_formal_v5_restart1_20260925/
```

holding diagnostic 的三个 target replay 结果位于：

```text
/home/user/training_outputs/terminal/p3_holding_diagnostic_20260925/run0/target_replay_diagnostic.json
/home/user/training_outputs/terminal/p3_holding_diagnostic_20260925/run1/target_replay_diagnostic.json
/home/user/training_outputs/terminal/p3_holding_diagnostic_20260925/run2/target_replay_diagnostic.json
```

12 个 `state_only_shadow_probes` 的 E04/final gate 分布结果位于：

```text
/home/user/training_outputs/terminal/p3_policy_probe_20260925/policy_probe.json
```

本地同步文件为
[policy_probe.json](D:/7_codespace/training_outputs/terminal/p3_policy_probe_20260925/policy_probe.json)
和
[samples_v2.json](D:/7_codespace/training_outputs/terminal/p3_policy_probe_20260925/samples_v2.json)。
probe 使用 strict model load、`load_optimizer=false`、`torch.no_grad()`，没有环境重放、动作采样
或策略更新。

## 2. 已观察到的两层问题

### 2.1 gate 分布的 WAIT 偏置

正式结果的三条 `failed_energy` 分别对应 UID `4`、`271`、`108`；UID `230` 的
`emergency_landed` 是受控应急着陆，不是 `failed_energy`。12 个 state-only probe 中，只有
1 个 probe 存在 certified RELEASE candidate，其余 11 个的 `selected_label` 为 WAIT。

| run / UID | time (s) | certified RELEASE | E04 `pRelease` | final `pRelease` |
|---|---:|:---:|---:|---:|
| run0 / 4 | 120 | yes | 0.0031371510 | 0.0001113526 |
| run0 / 4 | 200 | no | 0.4001095593 | 0.0000903587 |
| run0 / 4 | 280 | no | 0.0000313337 | 0.0000008618 |
| run1 / 230 | 300 | no | 0.0073521989 | 0.0000827922 |
| run1 / 230 | 500 | no | 0.0000093706 | 0.0000006175 |
| run1 / 271 | 280 | no | 3.8666e-13 | 5.5350e-14 |
| run1 / 271 | 330 | no | 1.8567e-13 | 4.3852e-15 |
| run1 / 271 | 390 | no | 0.0000956482 | 0.0000037638 |
| run2 / 108 | 200 | no | 0.0000060124 | 0.0000006366 |
| run2 / 108 | 300 | no | 3.0732e-11 | 9.0312e-13 |
| run2 / 108 | 440 | no | 0.0000355101 | 0.0000019697 |
| run2 / 108 | 480 | no | 0.0308923516 | 0.0119285313 |

E04 的 12 点平均 `pRelease` 为 `0.0368057613`，final 为 `0.0010184070`。这组 probe
说明 gate 在 E04 之后进一步偏向 WAIT；它不是对整段 trajectory 的重新采样，也不是公平的
性能比较。proposal top-3 在抽查点保持原 proposal head 的输出；例如 run0/UID4/time120
的 top-3 index 为 `6/14/12`，E04 与 final logits 均为
`30.5874/29.6076/25.1046`。

### 2.2 安全/资源可达性与能量耗尽

gate 偏置之外，实际执行链仍会因安全认证、资源占用和能量可达性阻挡 RELEASE。holding
diagnostic 记录中，UID108 在 `t=850` 的剩余能量为 `138`，而包含 reserve 与路径需求的
required energy 约为 `286`；energy guard 在该状态拒绝继续执行是当前边界内的正确行为，不能
通过放宽能量 guard 处理。

因此，现有证据对应两个同时存在的层面：gate 过度 WAIT（并且 E04 之后进一步加重），以及
真实安全/资源阻挡使部分等待最终进入能量耗尽。证据不支持“空域已经空了，只是被其他 UAV
拦截”的单一解释；也不能把结果归因成 SOC 单因素问题。`emergency_landed` 仍是可接受的带
代价降级结果，`failed_energy` 仍是硬失败。

## 3. forced-gate 反事实

对三个原失败 seed 使用 `controller=trained_release_only`、原 final checkpoint、neural K=3
proposal、完整 safety/energy execution chain 的 bounded diagnostic 结果为：

| run | seed | steps | normal | emergency_landed | failed_energy | unresolved | safety error |
|---|---:|---:|---:|---:|---:|---:|---|
| run0 | 850218 | 727 | 21 | 3 | 0 | 0 | null |
| run1 | 851032 | 787 | 20 | 4 | 0 | 0 | null |
| run2 | 852090 | 759 | 21 | 3 | 0 | 0 | null |

输出根为：

```text
/home/user/training_outputs/terminal/p3_force_release_probe_20260925/
```

三个 run 均记录 `gate_override=true`、`neural_proposal_used=true`、weights unchanged、
`PPO_ALLOWED=false`。这是 deterministic forced-gate + neural K=3 的反事实诊断，不是训练策略
成绩，也不提供 stochastic PPO qualification 证据；它不能证明 deterministic argmax 下的
starvation 已经解决。

## 4. bounded stochastic repaired replay

对原三个失败 seed 使用 `epsilon=.05` 的 bounded gate、canonical WAIT 和原安全/能量执行链，
完成了 bounded stochastic development replay。结果根为：

```text
/home/user/training_outputs/terminal/p3_bounded_gate_diagnostic_v1_20260925/run0/
/home/user/training_outputs/terminal/p3_bounded_gate_diagnostic_v1_20260925/run1/
/home/user/training_outputs/terminal/p3_bounded_gate_diagnostic_v1_20260925/run2/
```

| run | seed | 原始 normal / emergency / failed_energy | bounded `.05` + canonical normal / emergency / failed_energy | steps | resolved | safety hard failures | target arrival outcome |
|---|---:|---:|---:|---:|---:|---:|---|
| run0 | 850218 | 16 / 7 / 1 | 16 / 8 / 0 | 802 | 24/24 | 0 | arrival4 `emergency_landed` |
| run1 | 851032 | 15 / 8 / 1 | 15 / 9 / 0 | 825 | 24/24 | 0 | arrival18 `normal`; arrival20 `emergency_landed` |
| run2 | 852090 | 18 / 5 / 1 | 21 / 3 / 0 | 754 | 24/24 | 0 | arrival16 `emergency_landed` |

原始 lifecycle wait 的分母为每个失败 run 的 `23` 个 completed UAV，repaired replay 为
`24` 个；两者不能直接按同一分母比较。该 bounded replay 仅消除了已知的三例能量耗尽，未解决
目标的长期等待；它不作为最终策略，deterministic argmax 保持不变。

## 5. 修复候选

### 修复 A：受控 proposal fallback

在 proposal 被拒绝的 opt-in 诊断路径中，可回退到 canonical WAIT，同时保留 safety shield。
默认 legacy 路径不改变；该候选不放宽安全认证，不把 WAIT 误标为 RELEASE，也不改变 SOC 或
priority 排序语义。

### 修复 B：bounded stochastic gate calibration

新增 `terminal_marl/algo/gate_bounded_calibrated_v4_2.py`，提供独立 architecture：

```text
v4_2_hierarchical_gate_temperature_bounded_probability_residual_adapter_v4
```

它从旧 calibrated model 严格复制 actor encoder、gate head、gate adapter、proposal head、
critic 和 `gate_temperature=2.5`，只增加 persistent
`minimum_gate_probability` buffer。默认 `epsilon=.05`，gate 分布为：

```text
p_bounded = epsilon + (1 - 2 * epsilon) * p_calibrated
```

随后通过 `torch.distributions.Categorical(probs=p_bounded)` 供 rollout、log-prob 和 PPO
复用同一个实际分布。proposal head 完全不变；不使用 forced RELEASE，不做 SOC 排序，也不
宣称 deterministic argmax 已被修复。`epsilon=0` 保留 identity 测试入口。

bounded stochastic replay 结果见第 4 节；它是同三失败 seed 的开发诊断，不能替代正式 stochastic
training 或 PPO evidence。相关测试记录为 `10 + 14` 项，其中两组有重叠、不可相加，另有既有
相关测试 `38` 项。energy 相关的 `12 pass / 4 fail` 在修复前的当前 HEAD worktree 也能复现，
不把它归因于本次 bounded calibration 修复。

## 6. coverage gate-adapter fit 与 closed-loop anchor evaluation

### 6.1 coverage fit

coverage fit 从 E04 checkpoint 开始，只训练 `gate_adapter`，固定 `20` epochs、`lr=1e-4`、
batch `64`、label smoothing `.1`；proposal、encoder、critic 和 temperature 保持冻结。输入
规模为 coverage `1062` 行、old anchor `1529` 行、dev `1335` 行，run0/1 的 original 与
bounded labels 用于 coverage，旧 dev 与 run2 两组 labels 用于 dev；这些 labels 来自离线
state-only Teacher 采集。输出 checkpoint 为：

```text
/home/user/training_outputs/terminal/p3_gate_coverage_repair_v2_20260925/fit853401/p3_gate_coverage_repair_v2.pt
SHA256 03c0acb0d561219cbe1dc8dac88cd4df696ade4e4691ff0d4db1d4a72ed5e3ba
```

源 E04 SHA256 为
`e04e5b4a8f8365b52a41f66c96ad1f66a09a554ea64118fcab8040d64175d353`。checkpoint roundtrip exact，
`non_adapter_state_bitwise_unchanged=true`、`proposal_logits_bitwise_unchanged=true`，未保存或复制
optimizer state；报告中的 six exact-label checks 通过，服务器 prefix exact 检查通过。coverage
fit 是 diagnostic-only，`PPO_ALLOWED=false`、`heldout_consumed=false`、`checkpoint_promotable=false`。

报告字段顺序为 `NLL / RELEASE recall / WAIT recall / macro-F1`，before → after：

| 分组 | rows | before | after |
|---|---:|---|---|
| old anchor | 1529 | `0.7552645 / .8969359 / .7589744 / .7582381` | `0.7185643 / .8412256 / .7709402 / .7487538` |
| coverage | 1062 | `1.1720254 / .6390244 / .7549592 / .6499394` | `0.8915852 / .7609756 / .7584597 / .6923857` |
| pooled dev | 1335 | `1.2067959 / .7439024 / .7735849 / .7215775` | `1.0571464 / .7560976 / .7567031 / .7141853` |

dev 分源结果完整保留如下；old flow 的 WAIT recall 变化不能被 pooled dev 隐去：

| dev source | rows | before | after |
|---|---:|---|---|
| old flow6 | 78 | `0.7019802 / .8461538 / .8974359 / .8717105` | `0.7900042 / .8461538 / .7435897 / .7943309` |
| old flow12 | 69 | `0.3190412 / .9361702 / .9090909 / .9175627` | `0.4193009 / .9361702 / .8181818 / .8817625` |
| old dense24 | 665 | `1.1438805 / .7261146 / .7618110 / .7033963` | `.9917443 / .7579618 / .7598425 / .7131581` |
| original run2 | 312 | `1.8540663 / .6086957 / .7406015 / .6053676` | `1.5888426 / .6086957 / .7180451 / .5900247` |
| bounded run2 | 211 | `.9249050 / .6410256 / .8139535 / .6898645` | `.7844042 / .6153846 / .8023256 / .6720381` |

### 6.2 已知 failure seed 的 coverage closed-loop replay

使用上述 coverage checkpoint、原 V5 canonical 配置、`K=3`、stochastic rollout 和 Teacher-free
执行链，在同一 formal failure scenes 上完成三条回放。结果根为：

```text
/home/user/training_outputs/terminal/p3_gate_coverage_repair_v2_20260925/evaluation/run0/
/home/user/training_outputs/terminal/p3_gate_coverage_repair_v2_20260925/evaluation/run1/
/home/user/training_outputs/terminal/p3_gate_coverage_repair_v2_20260925/evaluation/run2/
```

| run / seed | steps | normal | emergency_landed | failed_energy | unresolved | requests | safety hard counters |
|---|---:|---:|---:|---:|---:|---:|---:|
| run0 / `850218` | 810 | 22 | 2 | 0 | 0 | 24 | 0 |
| run1 / `851032` | 739 | 19 | 5 | 0 | 0 | 24 | 0 |
| run2 / `852090` | 781 | 19 | 5 | 0 | 0 | 24 | 0 |

三条均 `full_resolution=true`、proposal exact；仅 gate_adapter 训练改变，非 adapter 参数与
proposal logits 逐 tensor 不变，rollout 前后整个 checkpoint 不变。在线 evaluation 为
Teacher=false，且未调用 PPO 或 heldout。原 formal failure scenes 的合计为 `49 normal + 20 emergency_landed + 3 failed_energy`；
coverage replay 为 `60 normal + 12 emergency_landed + 0 failed_energy`。因此已知三条 energy
failure 在这组三 seed 复现中消失，但 emergency 仍保留为受控降级结果；这不是所有等待/盘旋问题
都已消失的证据，也不是 P3 graduation 或 independent qualification。

三条 coverage replay 的 action-tape 可视化回放均 exact：run0/run1/run2 分别为 `810/810`、
`739/739`、`781/781` steps，且每条为 `24/24` requests resolved。可视化入口为：

| run | 本地产物 |
|---|---|
| run0 | [recorded replay](D:/7_codespace/training_outputs/terminal/p3_formal_policy_visualizations_20260925/coverage_run0/recorded_stochastic_policy_replay.html) |
| run1 | [recorded replay](D:/7_codespace/training_outputs/terminal/p3_formal_policy_visualizations_20260925/coverage_run1/recorded_stochastic_policy_replay.html) |
| run2 | [recorded replay](D:/7_codespace/training_outputs/terminal/p3_formal_policy_visualizations_20260925/coverage_run2/recorded_stochastic_policy_replay.html) |

基于记录中最后一个非空 `ring_role` 的 stride-2 派生统计，run0 arrival4 UID4 的末帧为 inner
且 `completed`；run1 arrival18 UID286 为 `completed`、arrival20 UID272 为 `emergency_landed`
且末帧为 inner。此前由该抽样得到的 run2 arrival16 UID134 “末帧为 outer”及 emergency origin
计数 `1/0/1` 已撤回；抽样漏掉了瞬时 middle 状态，不能作为实际 origin 结论，也不构成
outer→inner 直接跳环的证据。该旧报告备份为 `replay_report_frame_stride2.json`，最终统计使用
full-step 事件。

三条 replay 的 `step_trace.emergency_departures` 计数为 run0/run1/run2 的 `2/5/5`，与
`emergency_landed` outcome 完全对应；全 72 个 development targets 的
`before_ring_role=outer` emergency-origin 计数为 `0`。full-step 事件给出的四个重点 target 为：

| run / arrival / UID | full-step lifecycle | 终态 |
|---|---|---|
| run0 / 4 / UID4 | outer→middle `t=205`；middle→inner `t=266` | normal |
| run1 / 18 / UID286 | outer→middle `t=448`；middle→inner `t=558` | normal |
| run1 / 20 / UID272 | outer→middle `t=353`；middle→inner `t=420`；inner emergency `t=487` | `emergency_landed` |
| run2 / 16 / UID134 | release 发起 `t=357`；outer→middle 到达 `t=421`；middle emergency `t=421` | `emergency_landed` |

保留的全 step exact 诊断为：

```text
D:/7_codespace/training_outputs/terminal/p3_gate_coverage_repair_v2_20260925/run2_remaining_outer_diagnostic.json
```

该诊断显示 run2 arrival16 UID134 在 `t=357`、SOC `.41377` 获准 O2M；`t=421` pre-step phase
为 `ring_middle`、SOC `.37669`，随后完成受控 emergency，终态 SOC `.30926`。六个采样点
`173/213/253/293/333/421` 的 state-only Teacher 标签均为 WAIT 且无 certified RELEASE；这是
定点采样，不能推广为整条 trajectory。实际 ring 阶段记录为 `11` 次 safety reject 与 `1` 次
approved，`t=357` 未额外查询 Teacher。该目标确实尝试并切入 middle 后完成受控 emergency，不是
此前所写的持续外环盘旋耗尽问题；后续 formal PPO 仍需 short pilot。

priority-weighted wait 为 `131.1688/157.2021/147.9678`；low-SOC wait 样本数为 `10/10/8`，
均值为 `111.0077/85.1909/99.9963 s`。原 formal 的 wait 分母 `23` 与 coverage replay 的
`24` 不按同一分母比较。full-step lifecycle 事实已补齐，但不能把这三条回放写成所有目标均已正常。

### 6.3 old development seed anchor closed-loop replay

同一 canonical 配置下，对旧开发 seed `flow6=846201`、`flow12=846202` 分别加载 E04 与
coverage checkpoint，使用 stochastic、`collect_labels=false`、无 PPO 的 closed-loop rollout。
输出根为：

```text
/home/user/training_outputs/terminal/p3_gate_coverage_repair_v2_20260925/anchor_evaluation/
```

| checkpoint | scenario / seed | steps | normal / emergency_landed / failed_energy | unresolved | safety hard counters |
|---|---|---:|---:|---:|---:|
| E04 | flow6 / `846201` | 627 | `6 / 0 / 0` | 0 | 0 |
| E04 | flow12 / `846202` | 976 | `12 / 0 / 0` | 0 | 0 |
| coverage | flow6 / `846201` | 627 | `6 / 0 / 0` | 0 | 0 |
| coverage | flow12 / `846202` | 976 | `12 / 0 / 0` | 0 | 0 |

四条回放均 terminal/full-resolution，两个 checkpoint 的 SHA 在 rollout 前后保持不变。它们复用
已知 development seed，不能作为独立 qualification cohort。run2 的 bad-CR 软件现场、原 formal
failure roots、manifest 与 tapes 均保留 immutable；本地镜像 hash 差异不改变服务器 prefix exact
校验结果。

## 7. 当前结论与边界

coverage fit 与三条 failure-scene replay 已完成，但只能作为开发诊断：已知三条 energy failure
在该复现集合中消失，emergency 仍存在，old-flow/dev 的分类指标有分组退化，不能把 coverage
checkpoint 直接推广为 P3 策略，也不能宣称所有盘旋已根除。定向 gate 修复已完成，本轮不要求
重训全部既有课程，不放宽 SafetyFilter，不加入 SOC 专用规则。

下一步只保留一个 bounded mixed PPO pilot 入口：需要先由主审绑定 fresh optimizer、mixed old
anchors、anti-collapse 监测和 priority/old-course 指标；它不是 formal 3×240 restart，也不是
P3/P4 qualification。本轮 target/lifecycle/replay 事实已补齐；dense30 与同负荷随机环数量/安全
非均匀角度评估尚未启动，heldout 未消费。

相关检查记录为合并 `33 PASS`、既有 runtime `38 PASS`；energy 相关 `12 pass / 4 fail` 可在
修复前当前 HEAD worktree 复现。冻结的场景、reward、安全边界和旧 checkpoint 保持不变。
