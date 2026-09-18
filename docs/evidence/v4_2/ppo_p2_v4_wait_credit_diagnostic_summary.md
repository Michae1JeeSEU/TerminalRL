# P2 V4 WAIT / credit diagnostic summary

> 日期：2026-09-18
>
> 证据类型：`newly_run`（fresh stochastic diagnostic）+ `preexisting_artifact`
>（冻结 observations 上的离线 checkpoint 对比）
>
> 终态：`DIAGNOSTIC_COMPLETE=false`；`PPO_ALLOWED=false`；不得课程升级

## 1. 运行边界

正式诊断按预注册协议使用 fresh seeds 771001–771006，N6/N8/N12循环两次；candidate与
checkpoint24 baseline均为stochastic student behavior。Teacher仅以`teacher_label_only`
sidecar产生审计标签，scheduler保持`execution`，不替换请求动作；没有模型或optimizer更新。

- 协议提交：`93d44b2`
- WAIT诊断 manifest SHA256：
  `b290e2d5f69749e1109bff2c5f348458732a22fbe8c0ce34c15ce98880eefc93`
- formal reservation SHA256：
  `40ffe819690d8708ed3eedcf9c9c086b7d670eae3f6304fe05e97e3ed4a018cc`
- 离线checkpoint审计提交：`d3bb666`
- 离线审计结果 SHA256：
  `64c34bfbbeece10c6308c7b9cffc43b58bdb6000ec64357c82bb51943facabb3`

## 2. Formal diagnostic 终态

12/12 records均生成，但资格完整性 gate 未闭合：candidate完成50/52，baseline完成48/52；
baseline seed771001为4/6，candidate与baseline seed771006均为10/12，均到达冻结horizon。
candidate seed771004只观察到Teacher RELEASE，未满足单回合Teacher双类覆盖。所有回合的
collision、NMAC、hard separation、uncertified execution、SafetyInvariantError和shield均为0。

因此该run是保留的fresh diagnostic FAIL；它不追溯否定V4 deterministic heldout hard-gate
PASS，但进一步阻止P3升级，并证明stochastic训练分布仍存在排空尾部风险。

| 聚合指标 | candidate | checkpoint24 baseline |
|---|---:|---:|
| actor decisions | 3,777 | 5,820 |
| requested WAIT / RELEASE | 2,935 / 842 | 4,599 / 1,221 |
| Teacher WAIT / RELEASE | 1,853 / 1,924 | 3,622 / 2,198 |
| false-WAIT / Teacher RELEASE | 1,601 / 1,924 | 1,775 / 2,198 |
| false-WAIT rate | 83.21% | 80.76% |
| false-RELEASE rate | 28.01% | 22.03% |
| Brier / binary NLL | 0.5469 / 6.4692 | 0.4281 / 4.9686 |
| false-WAIT streak median / p90 / max | 8 / 42 / 76 | 8 / 37.8 / 82 |

candidate六个seed的false-WAIT率均在75%–90%；不是单seed异常。按interface，candidate
false-WAIT/Teacher RELEASE为o2m 594/700（84.86%）、m2i 503/669（75.19%）、i2f
504/555（90.81%）。三个interface均存在，i2f最严重。

## 3. Team credit 证据

这里的advantage是与正式kernel一致的raw team GAE；同一team step的值会在PPO actor
objective中重复给该step的全部actor decision。正负率是raw值的描述，正式更新会做统一
仿射标准化，但WAIT与RELEASE的相对排序不变。

| candidate cell | N | raw GAE mean | positive rate |
|---|---:|---:|---:|
| requested WAIT | 2,935 | +2.1719 | 53.46% |
| requested RELEASE | 842 | -1.0538 | 23.16% |
| false-WAIT（Teacher RELEASE） | 1,601 | +2.6025 | 61.46% |
| true RELEASE | 323 | -0.0996 | 36.84% |

六个candidate seed中，requested WAIT的平均raw team GAE均高于requested RELEASE。
baseline也保持相同方向。false-WAIT状态全部`release_eligible=1`、retry remaining为0，
headway slack中位数为1；candidate在这些状态上的RELEASE概率中位数仅
`6.59e-7`、p90仅`0.0148`。这排除了“主要因为retry mask或首次release后的安全重试”
的解释，并显示当前team-time credit会把由其他UAV事件产生的正回报共同赋给WAIT。

原V1 manifest中的`predicted_conflict_rate`辅助字段实际读取的是
`nearest_predicted_conflict` topology slot的presence bit，不是认证后的冲突谓词；不得用它
做冲突归因。实现已更名为`nearest_conflict_slot_present_rate`，核心gate、概率、Teacher
标签、GAE与streak结果不受该辅助命名修正影响，正式run不重写。

## 4. I3 → checkpoint24 → V4 离线对比

离线审计在同一批9,597个冻结actor observations上重新评分三个checkpoint，不启动环境、
不调用Teacher、不训练。combined结果为：

| checkpoint | false-WAIT | false-RELEASE | Brier | binary NLL |
|---|---:|---:|---:|---:|
| I3 DAgger | 82.85% | 21.88% | 0.4636 | 5.5487 |
| checkpoint24 | 83.11% | 23.00% | 0.4703 | 5.5765 |
| P2 V4 checkpoint8 | 80.37% | 27.07% | 0.4826 | 5.5940 |

I3并不优于两个PPO checkpoint的全部核心指标，因此
`supports_PPO_gate_forgetting_hypothesis=false`。V4把combined false-WAIT降低约2.7个百分点，
但以更高false-RELEASE和更差整体calibration为代价。三者在Teacher RELEASE状态上的平均
RELEASE概率（0.177/0.175/0.199）均低于各自在Teacher WAIT状态上的均值
（0.221/0.230/0.272）。

## 5. 结论与允许的下一步

当前证据更支持“P2自然流状态上的gate support/calibration缺口 + team-step信用分配失配”，
而不是单纯PPO遗忘、KL漂移或SafetyFilter拒绝。下一轮不能继续盲加PPO steps，也不能
放宽安全门槛。

允许的修复顺序：

1. 预注册fresh P2自然流Teacher-label-only support collection，训练/资格seed严格分离；
2. 先证明新状态上的WAIT/RELEASE可学习性与seed-disjoint gate calibration；
3. 做CE/DAgger gate repair并保留旧D1锚点，防止已有语义退化；
4. student-only stochastic qualification必须完成全部请求、三个interface覆盖、安全/shield为0，
   并显著降低false-WAIT；
5. 只有该阶段PASS后，才审查是否需要带明确local gate credit或监督锚点的PPO；原team-GAE
   actor objective不得不加诊断地继续扩训。
