# P2 gate support → repair → student qualification summary

> 日期：2026-09-18
>
> 终态：`P2_GATE_STUDENT_QUALIFICATION=FAIL`
>
> 授权边界：`P2_EFFECTIVENESS_EVALUATION_ALLOWED=false`，`PPO_ALLOWED=false`，
> `P3_ALLOWED=false`，`curriculum_promotion_allowed=false`，`checkpoint_promotable=false`

本报告把已完成的 P2 gate support、离线 learnability、gate repair/calibration 和 fresh
student-only qualification 串成一条证据链。各阶段的失败与通过均保留；本次 qualification
的 fresh seeds 已消费，不能重跑或用于调参。

## 1. Evidence inventory and boundaries

| 阶段 | 证据类型 | 终态 / 作用 |
|---|---|---|
| V2 support collection | `newly_run` | PASS；只收集 support，不授权训练 |
| gate learnability audit | `newly_run` | PASS；只证明离线可学习性，不是闭环 qualification |
| residual repair V1/V2 | `newly_run` | V1 数值失败；V2 保留唯一 support-dev NLL 失败 |
| fixed-temperature calibration V3 | `newly_run` | PASS；无训练/optimizer/environment/Teacher，只授权 fresh qualification |
| student-only qualification | `newly_run` | formal 6/6 完成后终态 FAIL |
| qualification finalization | `recovery_only` | 仅从 immutable completed artifacts 恢复最终 manifest；未重跑 rollout/reset |

所有 formal qualification 结论均来自 fresh stochastic student-authoritative rollout；Teacher
只作为 label-only sidecar。没有用 Teacher action 替换 student requested action，也没有训练、
optimizer update 或 PPO 更新。

## 2. Preexisting and newly run inputs

### 2.1 V2 support and D1 bindings

- V2 support collection：train `78/78`、dev `52/52`；train rows `5,627`、dev rows
  `2,009`；support manifest SHA256
  `09402508bc02155c1369de477dc320d5ca596768d4de6461f4a0526f2c7d9304`。
- train/dev support file hashes在本摘要中不展开（当前交接资料只保留前缀
  `80e6ef...` / `260471...`）。
- learnability audit 为 `3/3` fresh seeds PASS；其 manifest 仅以
  `e2affddd...` 前缀引用。该 PASS 是离线 gate learnability 证据，不等价于环境中的
  stochastic qualification。
- D1 作为既有 report-only anchor，未被 qualification rollout 当作训练数据或行为来源。

### 2.2 Repair and calibration chain

- Repair V1：non-finite repair loss，保留为历史 FAIL。
- Repair V2：使用 float64 logits-space anchor KL 后完成稳定 formal run，但 support-dev
  仍只失败于 binary NLL：

  | 数据集 | macro-F1 | WAIT recall | RELEASE recall | Brier | binary NLL |
  |---|---:|---:|---:|---:|---:|
  | support-dev | 0.859 | 0.884 | 0.880 | 0.109 | 0.969 |
  | D1 anchor | 0.971 | — | — | 0.013 | 0.0466 |

  V2 repair manifest SHA 以 `6586154a...` 前缀记录，qualification checkpoint SHA 以
  `91509526...` 前缀记录；两者均为 qualification-only、non-promotable，不授权 PPO。
- V3 fixed temperature calibration 选定 `T=2.5`。该阶段无训练、optimizer、环境 rollout
  或 Teacher call，并通过离线 metric/invariant gates。calibration manifest SHA256
  `a784ffa753d4588d9a0b997a8ee913010c128abfbf49c96b5e790f5bf6ae7272`；calibrated
  checkpoint SHA256
  `61e03e02e3fb204d2db4e814097543905f51e1a3772626fdac1d7bc02ab8ca6b`。V3 只授权下面的
  fresh student-only qualification。

## 3. Preregistered qualification protocol

Qualification protocol SHA256 为
`8064bebd115cad253e22ea019f678bb0a5c9085bb8a47989319fd9956a54aa65`。
formal seeds 为 `774001–774006`，scenario 顺序为 `flow6/flow8/flow12` 循环两次；全部
使用 stochastic student-authoritative behavior，Teacher 只保留 label-only sidecar。

每个 formal episode 预注册要求包括：完成数量等于请求数量、terminal 且最终 queue 为 0、
在 horizon 内、所有 safety counts 为 0、actor shield fraction ≤5%、exact action-tape
replay、requested logprob exact、Teacher label-only、student authoritative、actor rows
非空，以及无训练/optimizer/policy update。聚合要求包括：52/52 requested UAVs、两类
requested gate、unique decision IDs、finite 88D rows、overall/interface sampled
false-WAIT/false-RELEASE gates 和概率 metric gates。

## 4. Formal rollout and recovery-only finalization

六个 preregistered episodes 均已完成，合计 `52/52` requested UAVs，保留 `983` actor rows。
每回合的 completion、safety、shield、replay、requested-logprob 和 provenance gates 均
为 true。formal rollout 结束后仅在最终 JSON 序列化阶段遇到 reservation registry 中 set
值的 serializer 错误；没有在该错误前后重复任何 formal seed。

随后由 recovery-only finalization 从既有 summary、labels 和 action tapes 重建 episode
records，并只为 replay/logprob exact 检查实例化 replay environment；没有 rollout、formal
seed reset 或 marker 写入。恢复提交为 `b1c2f7c`，最终 qualification manifest SHA256
`c781ffc1230ee6c86db0a127ae8d17b014571c3b391feb27bf06643b9e5ed3c8`；one-shot marker
SHA256 `921cb108a23e9125da007c987eed10250a6e26fb95b589e43c1d1bbc0286799e`。

恢复 provenance：`recovered_from_completed_artifacts=true`，
`formal_rollouts_repeated=false`，`formal_seed_resets=0`。历史事实仍记录
`environment_rollout=true`；该恢复过程不改变 marker。

## 5. Terminal qualification gates

| gate / metric | 结果 | 预注册阈值 |
|---|---:|---:|
| sampled false-WAIT | `106/318 = 0.333333` | ≤ 0.30 — FAIL |
| sampled false-RELEASE | `29/665 = 0.043609` | ≤ 0.30 — PASS |
| macro-F1 | 0.815534 | ≥ 0.75 — PASS |
| WAIT recall | 0.960902 | ≥ 0.65 — PASS |
| RELEASE recall | 0.625786 | ≥ 0.75 — FAIL |
| Brier | 0.121740 | ≤ 0.20 — PASS |
| binary NLL | 0.524499 | ≤ 0.70 — PASS |
| per-interface probability gates | all three PASS | macro-F1 ≥ 0.65 — PASS |
| row integrity | PASS | finite 88D / unique IDs — PASS |

因此 terminal FAIL 由两个 preregistered gates 触发：overall sampled false-WAIT rate
超过 0.30，以及 probability RELEASE recall 低于 0.75。该结论不是由 recovery serialization
异常造成；recovery 只补写了缺失 manifest，未改变 episode 或 metric evidence。

## 6. Authorization boundary and next allowed work

本次 qualification 不授权 effectiveness evaluation、PPO、P3、curriculum promotion 或
checkpoint promotion。fresh seeds `774001–774006` 已消费，必须作为 immutable FAIL evidence
保留，不得 rerun、reset 或用于 threshold/model tuning；当前没有 server task remains。

下一步如需继续，必须在任何新 reset 前预注册 seed-disjoint 的 repair/qualification protocol，
重点针对 RELEASE recall 和 overall false-WAIT robustness，同时保留 safety gates、D1 anchor
和 no-original-team-GAE-PPO 边界。Effectiveness/P3 仍 blocked。
