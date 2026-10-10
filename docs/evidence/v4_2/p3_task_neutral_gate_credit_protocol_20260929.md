# P3 Task-neutral Gate Credit Diagnostic Protocol

日期：2026-09-29。状态：implementation blocked，awaiting explicit user authorization；服务器已确认无相关活动进程。
自动审批拒绝新增诊断模块，理由是其被视为需直接授权的实质代码新增；未拆分 patch 重试。这不是测试失败，
worker 未写入新模块或测试，未执行 check-only/full，尚未 launch。本协议是一次有界的只读 gate-credit 诊断，
用于从已完成的 task-neutral critic-refresh comparison 轨迹重建 WAIT/RELEASE 的
优势与概率变化；接受标准是事实重建正确，不预设性能改善，也不把相关性解释为根因。

## 1. 输出、固定输入与预算

输出根为：

```text
/home/user/training_outputs/terminal/p3_task_neutral_gate_credit_20260929
```

输入固定为 comparison ROOT 下 refreshed 臂 batch02 的 `24` 条完整 tapes：

```text
/home/user/training_outputs/terminal/p3_task_neutral_critic_refresh_comparison_20260929/full/refresh/training/batch_02/training/episode_01..episode_24
```

对应 env seeds 为 `881125..881148`，policy seeds 为 `881225..881248`；实际 `24` tapes 共
`18792` steps（各 tape steps：`780,766,942,780,729,819,747,750,799,803,730,838,817,768,682,758,714,676,768,797,870,756,719,984`），
每条最多 `1800` steps，总 replay 上限 `43200` steps，原始请求总数 `576`。collection checkpoint 固定为：

```text
/home/user/training_outputs/terminal/p3_task_neutral_critic_refresh_comparison_20260929/full/refresh/training/batch_01/policy.pt
```

collection checkpoint SHA256 为
`1637a15d73184617a1e386cd29cbcdbe29c0679b1414f5fb413ec7a9fbef3805`；同状态 control batch01
checkpoint（`control/training/batch_01`）SHA256 为
`c7e77764f05473cfaff0732b992a4a27c936822a636b9d7f8828cdcc77be97b8`。

同状态推理使用 source `3fe7`、comparison ROOT 下的 `control/training/batch_01`、
`refresh/training/batch_01` 和 `refresh/training/batch_02` checkpoints；refresh batch02 的最终
checkpoint SHA256 为
`370344b6523f250bd6816bbfd6ef51bd065c4aea1352b21d8910adc1275e942c`。不得用最终模型解释
collection 时的 episode24 failure；该 failure 发生在 batch02 update 之前。

本诊断为 fresh `check_only`/`full` 两个目录；check-only 必须为 `0` runtime，主审审阅后才允许
一次 full。不得重跑、改预算或更换输入。整个过程为 `0` 新 policy sampling、`0` PPO、`0`
optimizer、`0` Teacher、`0` heldout。

## 2. 重建内容

按实际 full-batch credit 语义恢复 `gamma=1`、`lambda=1`、`reward_scale=.01`，以完整的 `24`
episode batch 做 actor-world 归一化，不跨 episode 拼接边界。检查每条 episode 的 MC return、
terminal 边界、pre-step SOC/phase/UID 对齐；终态标签只用于事后分组，不作为输入或事前筛选。

按 WAIT/RELEASE、SOC `0/.2/.4/.6/1` 分组，并按 phase、normal/emergency/failed 分组，报告：

- advantage 与正优势比例；
- `pRelease` 饱和情况；
- 各 checkpoint 的概率变化与 delta。

局部代理 `A_norm * (gate - p_before)` 只表示独立 logit 方向，不能解释为共享参数更新，也不能
解释为因果动作价值。

## 3. 保留的失败证据与输出

保留 episode24 arrival `14`、UID `57` 的 trace 及完整 capture cache，避免重复采样。该条记录
的失败事实沿用 comparison report：standard、SOC `.553655769→0`、wait `674.2165` s、terminal
`892`、`ring_inner`、`energy_depleted`；不因其存在而删除或重跑任何 tape。

fake full 输出必须先通过本地单测与序列化检查，再执行 server check-only；full 输出须保存完整
replay、episode 边界、MC、分组统计和 capture-cache 引用。报告仅用于判断后续是否需要有依据的
有限修复，不证明根因或统计显著性。

## 4. 边界

本协议不改变 task-neutral、reward、安全或 checkpoint 语义，`global PPO_ALLOWED=false`，formal、
heldout、P4 和 Teacher 边界保持不变。当前仅实现中，尚未 launch；后续新 PPO 或训练预算须另行
授权。
