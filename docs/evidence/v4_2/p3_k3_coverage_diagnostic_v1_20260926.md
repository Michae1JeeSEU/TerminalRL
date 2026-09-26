# P3 K3 coverage diagnostic V1（2026-09-26，预注册协议）

## 状态与目的

- 状态：`REGISTERED_PENDING_SMOKE`。本文件只注册诊断边界；当前未运行 smoke 或 full diagnostic。
- 目的：区分 late-slot 约束、实际 K3 执行可行性与门控抑制分别对已存在状态的限制。
- 结果用途仅为诊断。它不验证训练标签最优性，不验证泛化，不产生新的训练标签，也不作为吞吐量或 qualification 的预设判据。

## 固定输入与实验边界

只读使用 source `03c0` 的六个 development tape，输入目录为：

```text
/home/user/training_outputs/terminal/p3_adaptive_kl_formal_v7_20260926/run1/dev_before_training
```

使用已登记 seeds `873101–873106`。每个 tape 的步数记为 `N`，目标步固定为：

```text
floor((N - 1) * f),  f ∈ [0, 0.15, 0.30, 0.45, 0.60, 0.75]
```

不添加、不替换、不补 seed。安全边界、能量检查、formal YAML、source policy、checkpoint 与 proposal 定义保持不变；不执行 fit、PPO、新 seed、heldout 或 formal 重跑。诊断结果不得自动接入 fit，也不得触发 formal 恢复。

输出根固定为：

```text
/home/user/training_outputs/terminal/p3_k3_coverage_diagnostic_v1_20260926
```

full 与 smoke 均必须拒绝已存在的 output/control，禁止覆盖既有产物。smoke 使用独立根，不能覆盖既有输出：

```text
/home/user/training_outputs/terminal/p3_k3_coverage_diagnostic_v1_20260926_smoke
```

## 状态抽样规则

对每个 tape 的六个目标步按时间顺序处理，每个 step 最多抽取一个 slot。抽样状态必须是目标 step 及其之后、该 slot 中第一个存在 runtime actor 的 pre-action 状态；如果目标位置与下一个 slot 重叠，至少推进到下一个 step 后再取样。

runtime actor eligibility 严格复用 student runtime：

- 没有 `committed_plan`；
- 当前环为 `O`、`M` 或 `I`；
- 没有 `future_release_reservation`。

候选按 `(arrival_id, uid)` 排序，按 `slot_index modulo actor_count` 选择。缺失状态或缺失 actor 不替换、不补样本、不更换 seed；missing 与 errors 必须进入报告分母和错误清单。

run0 dev80 的 seed `873106`、arrival `15` 的 `steps=136/152/200/250/297/500` 六个已诊断状态属于独立 witness 层，不并入六个 tape 的总体比例。主体最多 `36` 个 tape states，另加最多 `6` 个 witness states，共最多 `42` states；每个 state 做 off/on 两次，最多 `84` 个 shadow catalogue。

## 每状态 off/on 诊断

每个状态使用 `deepcopy`，只切换 `reconcile_late_merge_slots_enabled` 的 off/on 值。两种副本均执行 state-only Teacher 的全 `26` 候选审查，从中提取 `25` 个经过相同 SafetyFilter 与 energy checks 的 `RELEASE certified` 候选集合。

本流程不使用在线 Teacher，不改变原 tape、实际策略动作、checkpoint、live environment 或 RNG。off/on 结果只描述候选可行性差异；catalogue 可行不等于闭环成功，也不等于实际执行的 K3 shortlist 成功。

## 四个 checkpoint 的前向比较

在相同 live `88D` observation 上，对以下四个冻结 checkpoint 做 inference-only 前向：

1. source `03c0`；
2. run0 episode `80`；
3. run1 episode `240`；
4. run2 episode `240`。

每个 state 与每个 checkpoint 保存：

- `pRelease`；
- proposal logits；
- 按实际 sampled proposal 规则得到的 `qK3`；
- `pRelease * qK3`；
- 已测的 `k3_certified_mass`。

`pRelease * qK3` 只表示该单状态下尝试到 certified K3 proposal 的概率近似，不表示闭环成功概率。各 policy 的 proposal 若不同，保留各自的 `qK3`，不假定 proposal 相同。`qK3` 不作为最优 RELEASE 标签，不预设会提升吞吐量。

## Smoke 门槛

完整诊断前必须先执行一次有界 smoke：只读取 source 的第一个 tape，运行到第一个有效 sample，然后停止。smoke 必须验证：

- 输出序列化可读且 schema 稳定；
- `25` 个 proposal 的映射完整；
- tape prefix exact；
- config、RNG、live environment 与 model 均未改变；
- smoke 不创建或覆盖 full 输出。

只有 smoke 通过且本协议已提交后，才允许执行 full diagnostic。当前不宣称 smoke 已通过。

## Full 诊断完整性与报告

full 运行必须对六个 source tape、独立 witness 层和四个 checkpoint 保持同一输入规则；每个 tape 必须 exact 且 outcomes 一致，模型必须保持不变。所有 missing、validation error、运行 error 和未取到 actor 的状态均保留并报告，不静默删除。

结果按以下分组报告样本数、catalogue feasible 数、`qK3` 分布、gate probability 与 `pRelease*qK3`：

- 场景；
- 环 `O/M/I`；
- `SOC <= .65` 与 `SOC > .65`；
- priority 分组。

witness 层单独报告，不并入主体总体比例。报告只用于 primary 的后续诊断审查；不得自动生成 fit 数据、更新 PPO、修改 SafetyFilter/energy 边界、开启 late flag 晋级或重跑 formal。

## 预注册边界

本协议接受关注 normal、priority、low-SOC 与 energy 结果，并允许受控 emergency 非零。它不设定 emergency 消失、`qK3` 单调提升或吞吐量改善为通过条件，也不把任一单状态 catalogue 可行性解释为整体策略性能。

`P3_QUALIFIED=false`、`P4_ALLOWED=false`；原 heldout 保持 untouched。完整诊断结束后，primary 另行决定是否注册有界 gate 修复；本协议本身不授权训练或策略晋级。