# P3 Controlled-Emergency PPO Admission V5

状态：已注册，供恢复后的 entry、pilot 和正式大规模训练准入使用。本文只冻结
准入路线与证据边界，不启动实验，不把 `emergency_landed` 追溯改写为旧协议中的
`completed`，也不为历史结果制造新的 PASS。

## 1. 目标和结果语义

本版本把 P3 的训练目标从“应急必须为零”调整为：在完整安全与能量底线下，学习
priority、SOC、任务类型、资源占用和等待之间的权衡。高密度下，经过同一
SafetyFilter/SafetyShield 认证、没有能量耗尽并完成应急着陆的账本结果
`emergency_landed` 是带代价的降级服务，可以进入学习反馈；它本身不阻止 PPO entry。

账本必须区分：

| 结果 | 训练与准入语义 |
|---|---|
| `completed` | 正常完成，进入主要性能统计 |
| `emergency_landed` | 安全认证且能量未耗尽的应急完成；报告事件代价、priority、low-SOC 和全请求等待 |
| `failed_energy` | 硬失败，停止当前阶段的技术准入 |
| unsafe / 未认证执行 | 硬失败，包含 SafetyFilter/shield fail-closed、joint fail-closed、间隔、collision/NMAC 或执行 hash 违约；普通 shield correction/intervention 可非零，但必须按既有 actor-intervention 上限报告 |
| `unresolved` | entry/evaluation 的完整请求未闭合；formal resolution failure 是硬失败。pilot 的 horizon 截断必须另行报告，不把训练中的正常时间上限误写成普遍安全失败 |

`emergency_landed` 的比例和等待差异是 regression/performance report。相对初始策略的
“新增 emergency 数量”以及 priority-weighted wait 差异均为报告性回归标记，不是阻止开始
学习的自动硬门槛；本协议不要求在 PPO 学习之前先获得性能改善。后续独立 qualification
protocol 可以把 dense `<=10%`、sparse `<=5%` 作为 provisional research targets，
但这些目标尚未验证，既不是本协议的 admission threshold，也不能为既有结果追溯 PASS。

## 2. 冻结的环境、策略和训练边界

所有 V5 阶段保持：

- 车辆与能量：`research_evtol/eh216s_research_v2`、`energy_capacity=1500`、
  `min_landing_soc=.10`、`soc_reserve=.25`、`emergency_trigger=.40`；
- 安全：desired/planning separation `80 m`、protected floor `40 m`、collision event
  floor `<10 m`，保留原 SafetyFilter、predictive shield、headway、reservation、LZ/final
  capacity 和 route guard；shield correction/intervention 不要求为零，但 actor takeover/intervention
  fraction 仍须满足既有 `<=.05` 上限，shield/joint fail-closed 事件必须为零；
- 策略：Observation V2.1 `88D`、V4.2 hierarchical action、`K=3` runtime shortlist、
  `late-slot OFF`、`Teacher=false`、`downstream_priority`；
- 需求：initial SOC `Uniform(.55,.90)`、priority `{1,3,5}`、既有 mission mixture、
  `flow6`/`flow12`/`dense24`；
- 记录：requested gate/proposal 的 log-probability、execution-applied action、shield
  action、observation/post-state hash、ActionTape exact replay 和完整请求分母都必须保留。

低层策略定义不变：actor 只请求 `WAIT` 或 `RELEASE` 以及条件性的 speed×timing proposal；
轨迹生成器和安全层决定是否可执行，应急路线不是新增 actor action。Teacher、执行调度器或
shield 的动作不能替代 requested action 进入 PPO ratio。V5 不启用 SOC-only ordering，
也不把 gate-adapter-only 监督修复当作最终架构。

已有 I0/I1/P1/P2 能力不从头重训。旧课程策略作为 mixed old anchors 做回归检查，保留
自然流、低负荷和已学习任务分布；本轮不重置 I0，也不消费旧的 qualification/heldout
cohort。历史 canonical `829001–829006` 保持 untouched。

## 3. canonical reference 的边界

已注册的 canonical comparison 使用历史初始 checkpoint `4b8be66c...` 与 gate repair
candidate `e04e5b4a...`，沿用原有历史 all-complete/`no_failed_outcomes` gate。该比较
用于确认训练起点和记录修复差异，不把新 `emergency_landed` 语义回写到历史结果，也不授权
PPO。选择后续训练 checkpoint 时，先按安全与完整请求闭合筛选，再选 emergency 数量更少者，
再选 priority-weighted wait 更低者；完全相同才使用 canonical tie-break。该开发选择规则不
使用 heldout 作调参。

`848101/848102` 的 canonical 与 repair 四个 arm 均 exit `0`、full-resolved、zero-energy 且
安全硬计数为零；aggregate 为 canonical `31 normal + 17 emergency`、repair
`40 normal + 8 emergency`。repair 在两组 priority-weighted wait 与 low-SOC wait 上均更优，
因此 V5 stochastic entry 与 pilot 的冻结初始化来源为 repair candidate E04：

```text
SHA256: e04e5b4a8f8365b52a41f66c96ad1f66a09a554ea64118fcab8040d64175d353
path: /home/user/training_outputs/terminal/random_soc_gate_repair_v1_20260925/fit846401/random_soc_gate_repair_v1.pt
```

旧 canonical `no_failed_outcomes` gate 的历史 false 保持不变；该 development selection 不构成
P3 qualification。V5 entry/pilot 从上述 E04 开始，仍须按本协议的技术 gate 与后续主代理 review
执行。

## 4. Fresh seed registry and execution order

所有 V5 seeds 在本协议冻结后消费一次；不得 reset、rerun、替换或追加未登记 seed。输出写入
服务器 `/home/user/training_outputs/terminal`，源码只在本地同步工作树维护。

### 4.1 Smoke

`849000`：只用选定策略运行一个 dense24 短 smoke，执行 `2` steps，但绑定完整 `1800 s`
horizon 配置，确认 88D、Teacher off、weights unchanged、finite 输出、ActionTape 字段和
安全计数可落盘。Smoke 不是性能或资格结论。

### 4.2 Stochastic no-update entry cohort

以下 entry arms 使用冻结候选策略，不执行 PPO update：

| seed | 场景 |
|---:|---|
| `849101` | `flow6` |
| `849102` | `flow12` |
| `849103` | `dense24` |

entry 要求完整回合运行并完成全部请求终态核算，确认 requested likelihood、exact replay、finite/terminal accounting、
Teacher/labels/optimizer off、weights unchanged、proposal/temperature frozen，以及
zero-energy 和所有安全硬计数为零。`emergency_landed` 可报告并进入性能摘要；formal resolution
failure、`failed_energy` 或任何 unsafe/unresolved 仍阻止技术 entry。entry 不等同于课程毕业。

### 4.3 Bounded PPO pilot

只有 entry 的技术检查完成后才按下列顺序运行。pilot 共 `6` 个 registered arms，每个 arm
运行一个 full-cycle episode 并执行一次 PPO update，总计恰好 `6` 次 update：

| seed | 场景 |
|---:|---|
| `849201` | `flow6` |
| `849202` | `dense24` |
| `849203` | `flow12` |
| `849204` | `dense24` |
| `849205` | `flow6` |
| `849206` | `dense24` |

pilot 使用 full-cycle episode；安全/软件硬失败立即停止后续注册 arm，普通性能结果保留并
报告。所有已注册的 `6` 次 update 必须有效并满足 PPO technical gates；“至少一次 accepted
update”是最低存在性检查，不允许用一次成功更新掩盖其余 invalid update。

固定训练超参数：Adam `3e-5`、`epochs=2`、`batch_size=32`、`gamma=.99`、GAE `=.95`、
clip `=.1`、entropy coefficient `=.001`、target KL `=.01`、maximum KL `=.05`。
只训练 `gate_adapter` 与 `critic`；`actor_encoder`、`gate_head`、`proposal_head`、
`gate_temperature` 冻结。PPO ratio 和 entropy 只使用 requested action 的 factorized
log-probability。

沿用既有 priority time weight `1.0`、SOC urgency weight `2.0`、threshold `.65`，先不混入
新的奖励权重或 SOC 排序。学生 rollout summary 已持久化 `reward_component_totals` 和
`reward_total`，并校验组件和与总回报一致；累计字段使用实际的
`time/progress/release/touchdown/failure/teacher/shield/urgency/emergency` keys，排除每个
UAV 的 `total` 避免重复计算。`urgency` 仍是 priority 与 SOC 的合并项，当前不虚构独立的
priority/SOC 分解。对应实现为 `4d34672`，定向测试 `9 passed`。

### 4.4 Same-initial-state paired development

每个 paired seed 在 pilot update 之前先记录 `before`，在全部 `6` 次 pilot updates 完成后
再记录 `after`。同一 paired record 内，before/after 使用同一组明确保存的初始状态；这是
有意的两个 policy reset，不是重复消费 seed，也不使用历史 heldout：

| seed | 场景 |
|---:|---|
| `849301` | `flow6` |
| `849302` | `flow12` |
| `849303` | `dense24` |

before/after 使用同一组明确记录的初始状态做 paired development，便于比较 update 前后
的安全、完整回合与全部请求终态结果、`emergency_landed`、全请求等待、priority-weighted wait 和
low-SOC wait。它们是 development seeds，不是历史 heldout，也不替代后续独立 qualification。

## 5. Technical gates and authorization

### 5.1 Entry gate

entry 需要：

1. `849101–849103` 均完成完整回合与全部请求终态核算，或按协议记录完整 resolution；
2. zero-energy、SafetyFilter/shield、collision/NMAC、hard separation、execution hash、
   finite state、exact ActionTape replay 和 requested likelihood 检查通过；普通 shield correction
   可非零，但 actor intervention fraction `<=.05` 且 shield/joint fail-closed 为零；
3. `Teacher=false`、无 labels、无 optimizer/update、weights unchanged、K3 shortlist
   没有扩展成 25 项 catalogue scan；
4. schema、checkpoint lineage、restore metadata 与 `80/40/10` 安全边界一致。

entry 通过只设置 `PPO_ENTRY_ALLOWED=true` 的候选状态，不设置 `FORMAL_P3_ALLOWED`。

### 5.2 Pilot technical gate

pilot technical PASS 必须同时确认：

- 6 个 registered updates 都被接受且 `approx_kl` 在 target/max 纪律内；
- 至少一个真实 accepted update、PPO requested-action likelihood 正确、proposal/temperature/
  actor encoder/gate head freeze 正确；
- exact checkpoint restore 后权重、optimizer/step metadata 与保存点一致；
- before/after development 都有完整回合的 finite/replay/likelihood/safety/zero-energy
  结果，且完整请求分母已报告；
- 不因 `emergency_landed` 本身停止学习，同时不容忍 `failed_energy`、unsafe 或正式 resolution
  failure。

runner 应输出 `PILOT_TECHNICAL_PASS`、`performance_review_required=true`、
`FORMAL_P3_ALLOWED=false`。技术 PASS 不自动代表性能改善或课程结业。

### 5.3 Formal large-scale authorization

只有在 pilot technical PASS、paired development 完整落盘，并由主代理审查 balanced
performance evidence 后，才可把 `FORMAL_P3_ALLOWED` 置为 true，并准备正式大规模 P3 的
训练命令。本次目标止于 readiness 与可复核的启动准备，不在本协议内启动正式大规模 job。审查
至少覆盖 normal/emergency_landed、failed-energy、priority/low-SOC/全请求等待、每类负荷、
安全/完整性计数和初始/最终 checkpoint lineage；等待差异与 emergency regression 是报告
依据，不预设必须严格改善。之后仍需另行冻结 formal train/dev/heldout qualification
protocol；本 V5 不把 development 结果直接称为 P3 qualification。

当前状态：`PPO_ENTRY_ALLOWED=false`、`FORMAL_P3_ALLOWED=false`、`P3_QUALIFIED=false`、
`P4_ALLOWED=false`。本协议完成准入证据后，才允许进入正式大规模 P3 训练；不提前启动 P4
或高层 A6。
