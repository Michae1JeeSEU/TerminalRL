# P2 gate support → learnability → repair → student qualification summary

> 日期：2026-09-19
>
> 终态：`P2_GATE_STUDENT_QUALIFICATION=PASS`（Qualification V2；旧 Qualification V1 仍为 immutable FAIL）
>
> 当前授权边界：`P2_EFFECTIVENESS_EVALUATION_ALLOWED=true`；`PPO_ALLOWED=false`，
> `P3_ALLOWED=false`，`curriculum_promotion_allowed=false`，`checkpoint_promotable=false`

本报告把已完成的 P2 gate support successor、离线 learnability、anchored gate repair 和
fresh student-only qualification 串成一条证据链。历史 V1/V2 失败与新 successor PASS 均保留；
formal seeds `774001–774006` 与 `782001–782006` 已分别消费，不能重跑、reset 或用于调参。
本摘要区分 preexisting artifacts、newly run evidence 和 recovery-only serialization finalization。

代码证据提交链为 support/audit 历史 `18663b3`、`8ed0669`、`b7eaf85`、`c28efd5`、`dfa727f`，
anchored repair V3 `8a59312`，Qualification V2 `de8047b`，post-terminal CLI serialization fix
`d486bab`。Git 无 remote，以上提交未 push。

## 1. Evidence inventory and boundaries

| 阶段 | 证据类型 | 终态 / 作用 |
|---|---|---|
| V2 support collection | `preexisting_artifact` | PASS；只收集 support，不授权训练 |
| V3/V4/V5 calibrated-candidate support successor | `newly_run` | V3/V4 保留完整安全 episodes并逐步补足 rows；V5 最终 PASS，冻结 train/dev `3,786/2,415` |
| gate learnability audit V2 | `newly_run` | PASS；3/3 fresh probes；只证明离线可学习性，不是闭环 qualification |
| residual repair V1/V2 | `preexisting_artifact` | V1 数值失败；V2 保留唯一 support-dev NLL 失败 |
| anchored residual repair V3 | `newly_run` | PASS；仅 gate_adapter 六参数更新，T/proposal/critic/common state invariant |
| fixed-temperature calibration V3 | `preexisting_artifact` | PASS；无训练/optimizer/environment/Teacher，只授权 fresh qualification |
| student-only qualification V1 | `preexisting_artifact` | formal `774001–774006` 终态 FAIL；seeds 不得重跑 |
| student-only qualification V2 | `newly_run` | formal `782001–782006` 6/6 完成并 PASS；仅授权 P2 effectiveness evaluation |
| V2 post-terminal CLI serialization fix | `newly_run` | 仅修复完整结果打印的 set 序列化；不改变已落盘 formal evidence，无 rerun/reset |
| qualification finalization V1 | `recovery_only` | 旧 V1 仅从 immutable completed artifacts 恢复最终 manifest；未重跑 rollout/reset |

Qualification V1/V2 的 formal 结论均来自 fresh stochastic student-authoritative rollout；Teacher
只作为 label-only sidecar。没有用 Teacher action 替换 student requested action，也没有训练、
optimizer update 或 PPO 更新。learnability/repair/calibration 均没有 environment rollout；
V2 audit 与 repair V3 没有 Teacher 调用。所有数字均标注为已有产物或本轮 newly_run 证据，
不得把 Qualification PASS 表述为 PPO 准入。

## 2. Preexisting and newly run inputs

### 2.1 V2 support and D1 bindings

- V2 support collection：train `78/78`、dev `52/52`；train rows `5,627`、dev rows
  `2,009`；support manifest SHA256
  `09402508bc02155c1369de477dc320d5ca596768d4de6461f4a0526f2c7d9304`。
- train/dev support file hashes在本摘要中不展开（当前交接资料只保留前缀
  `80e6ef...` / `260471...`）。
- 原 support V2 后的 learnability audit 为 `3/3` fresh seeds PASS；其 manifest 在历史交接资料中
  仅以 `e2affddd...` 前缀引用。该 PASS 是离线 gate learnability 证据，不等价于环境中的
  stochastic qualification；最终 V5 successor support 上的 audit V2 以完整 SHA 在 §2.3 记录。
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

### 2.3 Calibrated-candidate support successor and anchored repair

- Support V3 为 calibrated candidate 的 on-policy state-distribution round：15/15 episodes
  安全完成，但 train/dev rows 仅 `1,150/908`，因此不改变 V2 support 的冻结证据，也不混入
  旧 support 或失败 qualification rows。
- V4 supplement 保留 V3 rows 并新增 27 个安全 episodes，combined train/dev 为
  `3,786/1,787`；除 dev cardinality 少 13 外，其余 support gates 通过。V5 dev-only supplement
  再新增 3 个安全 episodes，最终冻结 train/dev 为 `3,786/2,415`，全部 unchanged V2 support
  gates PASS。V5 manifest SHA256 为
  `100448260298b13cb33ce301267a44f9503a9cc1baea5c93f6121bff22fae960`，train/dev support
  SHA256 为 `40002db70c2e0d5d181b53b2f23290c65954f6bbe6282be1066b6f558e6153d4` /
  `da106e0d862c8e435748514abb10c4e0e5c500fb3bbd73f03cbfa6aa186a261b`。
- Learnability audit V2 在 V5 support 上 PASS，manifest SHA256 为
  `5a72549d05c4733c2163512bf30cfe28d601c38e3ddfe23dc8291ec2f17b3679`；3/3 fresh probes
  通过 support/dev，但 unanchored adapter probes 在 D1 降级，说明后续 repair 需要 D1 anchor。
  该 audit 没有 environment rollout、Teacher call 或 formal policy training。
- Anchored repair V3 PASS：protocol SHA256 `f8a84632fea21e6789c8a40c0a22938cb5f0bdfcdb4ec9efd5e48304867778e8`，
  manifest SHA256 `e1e0883e355676593ab4d561aeb46fb37656629ec514ad6f01e12c3bbd17d410`，checkpoint
  SHA256 `4b8be66c100aaaeb5b5699dcce68a96dbe1237964e4c308e7b23b95c62fb8c8c`。仅 gate_adapter
  六参数改变；temperature `T=2.5`、proposal logits、critic 和 common state 均 bitwise invariant。
  repair V3 仅授权 fresh Qualification V2。

Repair V3 final offline metrics：

| 数据集 | macro-F1 | WAIT recall | RELEASE recall | binary NLL | Brier |
|---|---:|---:|---:|---:|---:|
| support-dev | 0.951223 | 0.959234 | 0.951005 | 0.175879 | 0.034016 |
| D1 anchor | 0.981455 | 0.987511 | 0.996016 | 0.041551 | 0.010236 |

## 3. Historical Qualification V1 protocol

Qualification V1 protocol SHA256 为
`8064bebd115cad253e22ea019f678bb0a5c9085bb8a47989319fd9956a54aa65`。
formal seeds 为 `774001–774006`，scenario 顺序为 `flow6/flow8/flow12` 循环两次；全部
使用 stochastic student-authoritative behavior，Teacher 只保留 label-only sidecar。

每个 formal episode 预注册要求包括：完成数量等于请求数量、terminal 且最终 queue 为 0、
在 horizon 内、所有 safety counts 为 0、actor shield fraction ≤5%、exact action-tape
replay、requested logprob exact、Teacher label-only、student authoritative、actor rows
非空，以及无训练/optimizer/policy update。聚合要求包括：52/52 requested UAVs、两类
requested gate、unique decision IDs、finite 88D rows、overall/interface sampled
false-WAIT/false-RELEASE gates 和概率 metric gates。

## 3.1 2026-09-19 successor Qualification V2 protocol

Qualification V2 protocol SHA256 为
`7ab9e2a84b9aea8da2ea975d98b4867de4a45bd5725bc72019b2e95ddbb75094`。它绑定 PASS 的 anchored
repair V3（repair manifest SHA256 `e1e0883e355676593ab4d561aeb46fb37656629ec514ad6f01e12c3bbd17d410`、
checkpoint SHA256 `4b8be66c100aaaeb5b5699dcce68a96dbe1237964e4c308e7b23b95c62fb8c8c`），使用 fresh
formal seeds `782001–782006`，scenario 顺序同为 `flow6/flow8/flow12` 循环两次。旧
`774001–774006` 明确禁止重用；V2 seeds 已消费 exactly once。

Qualification V2 保持 stochastic student-authoritative rollout，Teacher 仅 label-only sidecar，
无 training/optimizer/policy update。Smoke seed `782000` 仅执行 160-step safe execution-path
probe，不期待 completion、不创建 formal marker、不产生 authorization。

## 4. Historical Qualification V1 formal rollout and recovery-only finalization

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

## 4.1 2026-09-19 Qualification V2 formal rollout

V2 的六个 preregistered episodes（`782001–782006`）均完成，合计 `52/52` requested UAVs，
保留 `621` actor rows；每回合 completion、safety、shield、replay、requested-logprob 和
student-authority/Teacher-label-only provenance gates 均为 true。形式化 sampled action 指标为
false-WAIT `18/206 = 0.0873786`、false-RELEASE `16/415 = 0.0385542`，均通过预注册阈值。

Smoke seed `782000` 的结果与既有 smoke 语义一致：执行 160-step safe execution-path probe，
不期待 completion，不创建 formal marker，也不产生 authorization。

同一 fresh actor rows 上的 probability metrics 为：macro-F1 `0.9532325`、WAIT recall
`0.9590361`、RELEASE recall `0.9563107`、binary NLL `0.1034486`、Brier `0.0314997`；
每个 interface gate 均通过。因此 Qualification V2 终态为 PASS，仅
`P2_EFFECTIVENESS_EVALUATION_ALLOWED=true`，PPO/P3/curriculum/checkpoint promotion 仍为 false。

V2 formal manifest SHA256 为 `7cd4f02c88dc953fe1545c613424e8c961c295d4f8c099456d6697f300088feb`，
one-shot marker SHA256 为 `43fafcd99e22d48bffb6c2389767a265c2376b05f95a070ae3d823f4066949d9`。形式化过程确实先写出了完整 PASS manifest 与 one-shot marker；随后 CLI 在打印完整 result 时
遇到 reservation historical registry 中 set 值的 JSON 序列化错误。该错误发生在 formal evidence
已落盘之后，不改变 manifest/marker/metrics，也没有 rerun 或 formal seed reset。修复提交
`d486bab` 将 normal/finalize-existing CLI 打印统一经 `_jsonable`，但 Qualification V2 artifact
的 source runner hash 仍是 pre-fix `b91e0d3794495bb768cf80ab266f1584b0af01addd2b0998c9c07e30af8d4a82`；
不能声称 formal rollout 运行了 patched source。

## 5. Historical Qualification V1 terminal gates

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

## 5.1 Qualification V2 terminal gates

| gate / metric | 结果 | 预注册阈值 |
|---|---:|---:|
| sampled false-WAIT | `18/206 = 0.0873786` | ≤ 0.30 — PASS |
| sampled false-RELEASE | `16/415 = 0.0385542` | ≤ 0.30 — PASS |
| macro-F1 | 0.9532325 | ≥ 0.75 — PASS |
| WAIT recall | 0.9590361 | ≥ 0.65 — PASS |
| RELEASE recall | 0.9563107 | ≥ 0.75 — PASS |
| Brier | 0.0314997 | ≤ 0.20 — PASS |
| binary NLL | 0.1034486 | ≤ 0.70 — PASS |
| per-interface probability gates | all three PASS | macro-F1 ≥ 0.65 — PASS |
| row integrity / episode completeness | 6/6, 52/52, 621 rows — PASS | finite 88D / unique IDs — PASS |

V2 因所有 preregistered episode、sampled action、probability、row integrity 和 authorization
gates 均通过而终态 PASS；该 PASS 只打开 P2 effectiveness evaluation，不打开 PPO/P3 或
curriculum/checkpoint promotion。

## 6. Authorization boundary and next allowed work

Qualification V1 不授权 effectiveness evaluation、PPO、P3、curriculum promotion 或
checkpoint promotion；fresh seeds `774001–774006` 已消费，必须作为 immutable FAIL evidence
保留，不得 rerun、reset 或用于 threshold/model tuning。Qualification V2 则仅授权
`P2_EFFECTIVENESS_EVALUATION_ALLOWED=true`；`PPO_ALLOWED=false`、`P3_ALLOWED=false`、
`curriculum_promotion_allowed=false`、`checkpoint_promotable=false`。V2 seeds `782001–782006`
同样已消费，不能重跑或调参。当前没有 server task remains。

下一步必须在任何新 reset 前预注册 fresh paired P2 effectiveness evaluation：student-only、
Teacher disabled，并冻结 safety/completion/replay/provenance gates 与 rule comparison。只有
effectiveness evidence 通过后，才可进行 evidence-based admission review，再考虑 PPO/P3；
Qualification PASS 本身不等价于 PPO 准入。
