# P3 proposal-head probe V1（2026-09-26，预注册协议）

## 状态与目的

- 状态：`REGISTERED_PENDING_SMOKE`。本文件只登记 proposal-only 开发试验；smoke、collection、fit、eval 均尚未在本协议下启动。
- 目的：在冻结 gate 与安全边界的条件下，检查 proposal head 是否能覆盖实际 K=3 执行所需的 speed×timing proposal。
- 该试验只用于开发筛选，不能单独证明 late-slot 或 gate failure 已修复，也不构成 P3/P4 qualification；不追求 `emergency_landed=0`。

## 固定 source 与可训练范围

source 固定为 V7 使用的 `03c0` checkpoint，不从失败的 formal run0 续训：

```text
path: /home/user/training_outputs/terminal/p3_gate_coverage_repair_v2_20260925/fit853401/p3_gate_coverage_repair_v2.pt
sha256: 03c0acb0d561219cbe1dc8dac88cd4df696ade4e4691ff0d4db1d4a72ed5e3ba
```

仅优化 `proposal_head.weight` 与 `proposal_head.bias`，共 `2425` 个参数。以下参数逐参数冻结：`88D encoder`、`gate_head`、`gate_adapter`、temperature 与 critic。相同输入的 gate logits 必须 bitwise 不变；架构、WAIT/RELEASE 语义以及 `25` 个 speed×timing proposal 的定义保持不变。不得加入在线 Teacher、强制 RELEASE、probability floor、UID/SOC override 或 reward 改动。

## 数据与固定状态标签

只读使用既有 source baseline 六个 development tape：

```text
/home/user/training_outputs/terminal/p3_adaptive_kl_formal_v7_20260926/run1/dev_before_training
```

seed `873101–873103` 为 train，`873104–873106` 为 validation；这些 seed 已用于开发分析，不称为 untouched qualification cohort。已有六个 witness 状态不进入 fit。

每个 tape 固定 `32` 个 slot，slot `i=0..31` 的目标步为：

```text
floor((N - 1) * i / 32)
```

到达目标步后，取该 slot 中第一个 runtime-eligible actor state；每个 step 至多取一个 slot。候选按 `(arrival_id, uid)` 排序，以 `slot_index modulo actor_count` 选择；缺失记录不替换、不补样本、不换 seed。runtime eligibility 沿用冻结 student runtime 规则。

每个标签由同一状态的 deepcopy 产生：late-slot 保持 OFF，使用 state-only Teacher 审查全部 `26` 个候选；`C` 是经过原 SafetyFilter 与 energy 认证的 `25` 个 RELEASE 子集。`C` 为空的 row 仍保留作 KL anchor，但不生成 WAIT 标签。每行保存 `obs(88D)`、source proposal logits、gate logits、SOC、priority、phase 与 `C`。最多 `192` rows。

## Collection 与输出边界

正式输出根固定为：

```text
/home/user/training_outputs/terminal/p3_proposal_head_probe_v1_20260926
```

其 `collection/`、`fit/`、`eval/` 子目录均必须拒绝已存在的 output/control，禁止覆盖既有产物。smoke 使用独立 sibling root：

```text
/home/user/training_outputs/terminal/p3_proposal_head_probe_v1_20260926_smoke
```

在 full collection 前，只允许运行一次 source 首个 tape 的首个 actor prefix smoke。smoke 必须验证 prefix exact、schema 可序列化、`C` 映射覆盖、RNG/live environment/model/config 不变；smoke 通过并提交本协议后，才允许按上述固定 seed 完整采集。missing 与 errors 必须保留在报告分母和错误清单。

## 单次 fit 规则

fit 只允许一次，固定：

```text
seed=890401
optimizer=Adam
learning_rate=0.001
batch_size=32
max_epochs=40
gradient_clip_norm=1
```

对非空 `C` 的 row，set loss 为：

```text
L_set = mean(-logsumexp(logp[C]))
```

总 loss 为 `L_set + 1 * KL(ref || candidate)`；KL 在全部 train rows 上取均值，`ref` 为 source proposal distribution。概率计算使用 log-space，避免 tiny probability underflow。每个 optimizer step 后，对全部 train anchor rows 的 `KL(ref || candidate)` 均值执行 retention 上限 `<= .05`；超限则回滚该 step 的 model 与 optimizer，停止本次 fit，并保留最后已接受参数。该 `.05` 是本 proposal 试验的 retention 上限，不改变 V7 gate/reference caps；不得自动降 LR、换 seed 或增加 epochs。

每个 epoch 必须记录 `C` coverage、`qK3`、KL、参数变化以及 accepted/rejected step。validation 只报告，不用于 checkpoint 选择或调参；若没有实际非零 proposal 更新，fit 记为无有效学习。

## Frozen eval 与开发筛选门槛

eval 前必须执行 checkpoint save/reload，并验证 frozen gate、gate logits 与 candidate proposal 的 roundtrip；随后 candidate 在六个 development seed 上各运行一次 stochastic full episode，source baseline 复用既有同 seed 结果。train-used 三 seed 与 validation 三 seed 分开报告，合计 `84` requests。

报告必须包含全请求 lifecycle 的 priority-weighted wait、`initialSOC<=.65` low-SOC wait、normal、`emergency_landed`、`failed_energy`、other/unresolved 与 safety counters；缺失请求不得从分母删除。

仅当以下条件均满足时，才记录为该开发筛选的通过候选：

1. 新增 safety、energy、other、unresolved 均为零；
2. overall 与 validation 的 normal 数量均不低于 source；
3. overall 与 validation 的 priority-weighted wait、low-SOC wait 均不超过 source 的 `1.05×`（空组记 `NA`，不记零）；
4. 至少有开发集 K3 coverage 改善，或闭环 normal/wait 改善。

这些只是开发筛选条件，不是 P3 qualification。若出现退化或无法学习，保留完整结果交 primary 诊断，不自动重复试验。

## 明确排除项

本协议不消费 `880xxx`、`854xxx`、`829xxx` 等 heldout，不重启 V7，不解冻 gate/encoder/critic，不改变 late OFF、canonical WAIT、K=3、安全、能量或 reward 语义，也不自动接入 formal 或 fit 后续训练。正式 proposal 适配是否继续须由 primary 结合本结果另行审查。
