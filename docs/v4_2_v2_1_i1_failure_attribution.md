# V4.2 / Observation V2.1 I1 Formal Failure Attribution

## Final decision

```text
training_fit_sufficient = true

shared_gate_error_fraction = 0.293785
shared_proposal_error_fraction = 0.169492

release_support_gap = true
proposal_11_support_gap = true
proposal_14_support_gap = true

conservative_release_retention_bias = true
proposal_boundary_ambiguity = false
new_observation_aliasing = false

ROOT_CAUSE = I0_TO_Q_RELEASE_MANIFOLD_SUPPORT_GAP

i0_training_data_revision_required = true
training_protocol_revision_required = false
architecture_revision_required = false
observation_revision_required = false

V4_2_V2_1_I1 = FAIL
CANONICAL_I1_CHECKPOINT = none
V4_2_V2_1_I2_DAGGER_PREFLIGHT_ALLOWED = false
PPO_ALLOWED = false
```

本诊断不追溯改变正式 I1 结果。三个 checkpoint 均保持
`READ_ONLY_FAILED_QUALIFICATION_ARTIFACT`，Q cohort 已永久转为
`CONSUMED_DIAGNOSTIC_REFERENCE`。

## Evidence inventory

| Evidence | Type | Use |
|---|---|---|
| Formal I0-R, 2020 rows, SHA `6bcbc178...` | `preexisting_artifact` | 训练支持流形 |
| Consumed Q, 1124 rows, SHA `8d1e80bb...` | `preexisting_artifact` | 失败归因，不再作 untouched qualification |
| Checkpoints 4202/4203/4204 | `preexisting_artifact` | 只读推理 |
| Per-state consensus, 88D support distances, context TV, sidecar audit | `newly_run` | 根因证据 |

没有运行 retraining、checkpoint selection、阈值修改、数据修改、重权、过采样、
Teacher/Observation/Safety/architecture 修改。

## Training fit vs Q generalization

| Seed | Train gate F1 | Q gate F1 | Train RELEASE recall | Q RELEASE recall | Train proposal top-1 | Q proposal top-1 | Train/Q decoded |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4202 | 0.8976 | 0.7860 | 0.7804 | 0.5254 | 0.9872 | 0.8079 | 0.9297 / 0.8968 |
| 4203 | 0.9029 | 0.8219 | 0.8614 | 0.6893 | 0.9872 | 0.8192 | 0.9287 / 0.8977 |
| 4204 | 0.8884 | 0.7934 | 0.7484 | 0.5480 | 0.9893 | 0.8023 | 0.9257 / 0.8995 |

三个 seed 的训练 proposal top-1 均接近 0.99，训练 gate macro-F1 为
0.888–0.903；Q 上 RELEASE recall 和 proposal top-1 同时显著下降。因此证据支持
generalization/support gap，不支持训练本身学不动。

## Three-seed error consensus

Q 中有 177 个 Teacher-RELEASE rows。Gate 共识分解：

- 52 个被三个 seed 全部错判为 WAIT，shared false-negative rate = 29.38%。
- 23 个被两个 seed 错判，17 个被一个 seed 错判，85 个三个 seed 全对。
- 全部 1124 rows 的 three-seed gate disagreement rate 仅 6.14%。

Proposal 共识分解：

- 30/177 被三个 seed 全部预测错，shared error rate = 16.95%。
- 5 个被两个 seed 预测错，1 个被一个 seed 预测错，141 个三个 seed 全对。
- Teacher-RELEASE 上 three-seed proposal disagreement rate 为 5.08%。

错误高度跨 seed 重合，不符合主要由 optimization variance 造成的模式。

## RELEASE support-distance audit

距离使用冻结的 normalized Observation V2.1 88D Euclidean metric；kNN 固定
`k=10`，局部半径固定为 `0.05/0.10/0.25`。

| Group | Nearest train RELEASE median | Nearest train WAIT median | Same-proposal median | kNN RELEASE fraction median |
|---|---:|---:|---:|---:|
| RELEASE correct in all 3 | 0.5246 | 1.2525 | 0.7340 | 0.70 |
| RELEASE missed in all 3 | 1.3935 | 1.1757 | 1.4142 | 0.35 |

共同 false-WAIT 状态到训练 RELEASE 的中位距离为全对组的 2.66 倍，且其最近
WAIT 比最近 RELEASE 更近。预注册的三个 support-gap comparison 全部触发。

按 family，`post_conflict` 的 nearest RELEASE median 为 1.4094、kNN RELEASE
fraction median 为 0.40；训练仅含 12 个 post-conflict RELEASE lifecycles，来自
2 seeds/episodes，而 Q 含 60 lifecycles、10 seeds。`post_WAIT` 的 Q 状态也偏向训练
未充分覆盖的 high-traffic 和 ring-inner contexts。

## Proposal-specific context support

Proposal 11 不是简单 row-count 不足：训练有 51 lifecycles，Q 有 26，但 Q 中
24/26 位于 N=12 traffic，而训练只有 28/51；traffic distribution total variation =
0.3741。因此判为 `TRAIN_SUPPORT_CONTEXT_COVERAGE`。

Proposal 14 同样是 context gap：训练有 39 lifecycles、Q 有 22；训练只有 2/39
middle-ring，而 Q 有 10/22。其 family、phase/ring、traffic total variation 分别为
0.5748、0.4033、0.3881。训练中的 post-WAIT proposal-14 仅 2 rows，而 Q 为 10。

Proposal-error sidecar 中，可计算 certified alternative rank gap 的错误仅 28.57%
属于相邻 rank near-tie，故 `PROPOSAL_BOUNDARY_AMBIGUITY = false`。主要问题是明确
context 的训练支持不足，而非 Teacher 排序普遍模糊。

## Scenario attribution and conservative bias

最稳定的 strata 是 `clear_outer_low_traffic`（三个 seed 全指标 1.0）。主要 gate
弱点集中在：

- `post_WAIT_release`: RELEASE recall 0.095 / 0.381 / 0.238；
- `transition_retry`: 0.143 / 0.429 / 0.286；
- `conflict_recovery`: 0.238 / 0.714 / 0.381。

同时 WAIT recall 保持很高，且 retry-active false RELEASE rate 为 0。这不是 all-WAIT
collapse，而是对 inhibition/retry/transition 后 WAIT→RELEASE 边界的保守保持偏置：
模型会正确 WAIT，但在约束解除后仍倾向继续 WAIT。

## Near-aliasing conclusion

formal I0-R + Q 的 exact observation-hash opposite-gate conflict 为 0。局部混合标签
与较低局部 RELEASE 支持同时出现，但本诊断没有找到一个明确、Teacher-decisive、
actor 不可见的 state-only 变量。因此不能仅因边界困难重开 Observation；
`new_observation_aliasing = false`。

## Prospective I0-R2 repair specification

下一阶段应先预注册 fresh I0-R2 collector protocol，以全新独立 expert episodes 在
scenario level 增加：

- post-WAIT、retry-expired/transition、post-conflict RELEASE boundaries；
- high-traffic proposal-11 contexts；
- middle/interface、post-WAIT、post-conflict proposal-14 contexts；
- 对上述 contexts 的独立 seeds、episodes 和 lifecycles，而非重复已有 rows。

禁止复制 consumed Q、复制旧 I0、按 label 过采样、class weighting 或修改 Teacher。
任何 I0-R2 或训练协议改变后，必须预注册 qualification protocol V2，采集并冻结新的
untouched Q2，再训练 fresh multi-seed actors，并只评估 Q2 一次。

## Protocol risks

- 88D Euclidean distance 等权处理各 normalized coordinate，是诊断 metric，不是因果证明。
- support-gap cutoffs 是本轮固定诊断规则，不是新的 qualification threshold。
- 本轮没有 causal intervention；根因结论由 training/Q gap、跨 seed 共识、邻域支持和
  context distribution 四组一致证据共同支持。
