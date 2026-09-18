# P2 natural-flow gate support collection protocol V1

> 状态：已执行，`SUPPORT_COLLECTION=FAIL`；runner 提交 `4166249`
>
> 协议 SHA256：`8e614001e2475c2e1821036b88aeea93aa14afec468320eb7fd5792afca82aac`
>
> 授权边界：本阶段只收集和冻结 Teacher-label-only support；即使 PASS 也不授权 gate repair、PPO 或课程升级

## 1. 假设与目的

fresh WAIT/credit 诊断显示，V4 candidate 在 P2 自然流 Teacher RELEASE 状态上的
false-WAIT 为83.21%，且 I3/checkpoint24/V4 三个 checkpoint 均存在同类缺口。当前单一可检验
假设是：现有 D1 缺少 N6/N8/N12 自然流 student-visited gate 状态支持；在决定训练形式前，
必须先冻结训练/开发 seed 分离的新 support，并验证 WAIT/RELEASE、三个 interface 和 88D schema
覆盖。收集本身不证明 gate 可学习，也不改变任何模型。

## 2. 冻结 lineage

- 行为策略：P2 V4 `checkpoint_8.pt`，SHA256
  `5e6811b8349d875fb007943476aec62d93550a74ffaf4ae46614c5c188ab004f`；
- 旧 88D D1 锚点：8,552 rows，dataset SHA256
  `270e54ef6ed0c36f535e2a0f25cef7cf9fdd6c91016ee295e7e336fb800cacb0`，
  manifest SHA256 `a6319c291e5c91e155fd1c7fb20ae95dcdd4973733abfe83fc10c5e6006f472c`；
- 前置 WAIT diagnostic manifest SHA256：
  `b290e2d5f69749e1109bff2c5f348458732a22fbe8c0ce34c15ce98880eefc93`；
- D1 与 diagnostic 只做只读 hash 绑定；其 rows 不混入本轮 support 文件。

## 3. Seed、场景与行为

| 用途 | seeds | 场景 | 行为 |
|---|---|---|---|
| train support | 772001–772009 | flow6/flow8/flow12 循环三次 | stochastic student authoritative |
| dev support | 773001–773006 | flow6/flow8/flow12 循环两次 | stochastic student authoritative |
| future qualification | 774001–774006 | 本轮不运行 | 仅冻结，后续 student-only 使用 |
| smoke | 772000、773000 | flow6，各160 steps | 不预留 formal seeds |

三组 formal seeds 必须互斥，并在第一次 reset 前全部通过 historical registry 未消费检查。
formal 仅为 train/dev 创建一次性 consumed marker；future qualification seeds 不得 reset、不得写
consumed marker。Teacher 使用 `teacher_label_only` sidecar，scheduler 保持 `execution`；Teacher
不替换请求动作，禁止任何模型或 optimizer update。

## 4. 数据与 hard gates

只保留 `actor_mask=true` 且 phase 属于 o2m/m2i/i2f 的行，分别写入不可覆盖的
`train_support.jsonl` 和 `dev_support.jsonl`。每个保留行必须是有限 88D Observation V2.1，
保留原 Teacher/requested action provenance，并增加 split、scenario、interface。

每回合必须满足：到达 terminal 或冻结 horizon、全部安全计数为0、shield为0、ActionTape精确
复放、requested logprob精确、label与actor行存在、Teacher label-only、student执行权威且无训练。
出现安全或收集异常时立即停止，未完成的计划 episode 使 formal FAIL。

| split gate | train | dev |
|---|---:|---:|
| rows | 3,000–30,000 | 1,800–20,000 |
| 每个 Teacher class 最少 | 600 | 350 |
| 每个 Teacher class 比例 | ≥20% | ≥20% |
| 每个 interface 总数 | ≥500 | ≥250 |
| 每个 interface × class | ≥120 | ≥60 |

此外要求 decision-event ID 全局唯一、train/dev seed和event完全不相交、所有固定 episode 执行
闭合。请求完成率只作描述，不把当前 candidate 的排空缺口伪装为 support 收集失败；后续
student-only qualification 仍必须全部排空。

## 5. 终态与后续授权

只有非 smoke 且全部 episode、split、schema、identity、safety/replay gates 通过时，才写：

```text
SUPPORT_COLLECTION=PASS
SUPPORT_FROZEN=true
LEARNABILITY_AUDIT_ALLOWED=true
```

无论结果如何，本阶段固定：

```text
GATE_REPAIR_ALLOWED=false
PPO_ALLOWED=false
curriculum_promotion_allowed=false
```

PASS 后下一步只能预注册离线 learnability audit；该审计必须仅用 train support 拟合，并在
冻结 dev support 与旧 D1 anchor 上报告 gate recall、macro-F1、Brier/NLL 和 interface 分解。

实际终态见 [P2 gate support collection V1 summary](ppo_p2_gate_support_collection_v1_summary.md)。
