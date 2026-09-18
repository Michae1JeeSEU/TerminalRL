# P2 natural-flow gate support collection protocol V2

> 状态：执行前预注册；runner提交`3a60c9f`；尚未消费V2 formal seeds
>
> 协议SHA256：`14e235ec3402671a493aea2669135ace53fc7c97e4226ef36e1dada0910b6e43`
>
> 授权边界：只收集fresh support；PASS最多授权offline learnability audit

## 1. V1终态与唯一变化

V1按冻结协议产生15/15 records、7,616个actor rows，数据覆盖与安全均通过，但seed772009
记录3次SafetyFilter干预，违反V1的`shield_zero`，所以manifest SHA
`3f376b68209dbd7a68db058b8344204138dae3ea107cd2678d2f4966312fd979`
保持`SUPPORT_COLLECTION=FAIL`。V1 rows不得训练或并入V2。

V2唯一语义变化是把每回合shield gate改为
`shield_actor_fraction <= 0.05`。该阈值逐字继承早于V1的I3 student-only qualification，
I3 protocol SHA为`21d1761860ba4636c811b9a7739629e12fafe12dbf2ee9bd92ae6decfbd4f387`；
不是根据V1的3/959结果拟合。Safety全部计数仍必须为0，Teacher仍不得替换student请求，
replay、logprob、schema、class/interface和数量gate全部不变。

## 2. Fresh seeds与绑定

| 用途 | seeds | 场景 |
|---|---|---|
| train support | 775001–775009 | flow6/flow8/flow12循环三次 |
| dev support | 776001–776006 | flow6/flow8/flow12循环两次 |
| future qualification | 774001–774006 | 本轮不运行 |
| smoke | 775000、776000 | flow6，各160 steps |

formal前检查三组互斥并全部未被historical registry消费；V2一次性marker为
`p2_gate_support_collection_v2_consumed.json`。smoke不创建marker。未来资格seeds保持untouched，
本轮不得reset或写consumed marker。

除V1 FAIL manifest外，V2继续绑定：

- candidate SHA：`5e6811b8349d875fb007943476aec62d93550a74ffaf4ae46614c5c188ab004f`；
- 8,552-row D1 SHA：`270e54ef6ed0c36f535e2a0f25cef7cf9fdd6c91016ee295e7e336fb800cacb0`；
- D1 manifest SHA：`a6319c291e5c91e155fd1c7fb20ae95dcdd4973733abfe83fc10c5e6006f472c`；
- WAIT diagnostic manifest SHA：
  `b290e2d5f69749e1109bff2c5f348458732a22fbe8c0ce34c15ce98880eefc93`。

所有旧artifact仅做只读hash/终态绑定，任何旧row都不得混入V2。

## 3. 不变hard gates

每回合必须到达terminal或冻结horizon、Safety全部计数为0、ActionTape replay精确、requested
logprob精确、Teacher label-only、student request authoritative、无训练，并满足继承的5%
shield-actor fraction上限。train/dev阈值保持V1：

| split gate | train | dev |
|---|---:|---:|
| rows | 3,000–30,000 | 1,800–20,000 |
| 每个Teacher class最少 | 600 | 350 |
| 每个Teacher class比例 | ≥20% | ≥20% |
| 每个interface总数 | ≥500 | ≥250 |
| 每个interface × class | ≥120 | ≥60 |

还要求有限88D Observation V2.1、decision-event全局唯一、train/dev seed与event完全不相交、
全部计划episode执行闭合。请求完成率保留为描述；后续student-only qualification仍须全部排空。

## 4. 终态授权

只有formal全部gate通过时：

```text
SUPPORT_COLLECTION=PASS
SUPPORT_FROZEN=true
LEARNABILITY_AUDIT_ALLOWED=true
```

smoke与任何FAIL均不授权后续审计；无论结果如何，本阶段固定
`GATE_REPAIR_ALLOWED=false`、`PPO_ALLOWED=false`、`curriculum_promotion_allowed=false`。
