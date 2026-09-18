# P2 V4 target-KL 训练与 heldout 评估终态

> 日期：2026-09-18
>
> 证据类型：`newly_run`（训练、正式 heldout 评估与既有 action tape 只读分解）
>
> 适用范围：固定几何、自然入流 N6/N8/N12、V4.2 / Observation V2.1 低层策略

## 1. 协议与绑定

- 恢复点：P2 V2 最后合格 `checkpoint_5.pt`，SHA256
  `8a8466f61119f7f5b5f780249d26f218ee9b8ae3e4aededc5c7f5fe2e51b201b`。
- target-KL：每次 optimizer step 后在完整 rollout requested-action 分布上计算 sampled KL；
  阈值 `0.01`，超限或非有限值时同时回滚模型与 Adam，训练阶段要求至少一个接受更新、
  至多一个拒绝更新。原 `approx_kl <= 0.05` gate 保留。
- qualification seeds：766001–766006；entry seeds：767001–767002；正式训练 seeds：
  769001–769008；heldout seeds：755001–755006。
- 正式 heldout 只运行一次，固定最终 `checkpoint_8.pt`，与 checkpoint24 baseline、nominal rule
  在相同最新生成器下配对；deterministic、无更新、无 Teacher。
- 代码提交：`c9d5b38`（target-KL 训练保护）与 `c030858`（V4 heldout 评估入口）。

## 2. 训练证据

资格 6/6 PASS；入口 2/2 PASS，入口 `PPO_ALLOWED=true`。正式训练 8/8 回合 PASS，
共完成 66/66 架；safety、shield、逆环向均为 0，最大连续内侧偏差
`0.019706256981635306 m`。

正式训练执行 468 个 optimizer step，468 个接受、0 个拒绝；各回合
`max_accepted_rollout_kl` 的最大值为 `0.006534156040536751 < 0.01`。这说明新保护在本轮
未通过放宽门槛取得 PASS，也没有发生回滚后继续掩盖超限的情况。

- 最终 checkpoint：`p2_resume_v4_20260918/train/checkpoint_8.pt`
- checkpoint SHA256：
  `5e6811b8349d875fb007943476aec62d93550a74ffaf4ae46614c5c188ab004f`
- train/root manifest SHA256：
  `73502ed0f956e517c47dd14ea6bfcc411cb927eba43c4a409fdee119fa7c82f0`

## 3. Heldout 评估证据

18/18 个 paired records 全 gate PASS；三种控制器均完成 52/52 架，所有 safety 计数、
shield 与 actor shield 均为 0，geometry、exact replay、requested likelihood（适用时）均通过。
`training_called=false`、`teacher_enabled=false`、`deterministic=true`、candidate/baseline
`label_rows=0`。

| 控制器 | 加权平均等待 / s | request-to-touchdown / s | 六场景平均完成时间 / s |
|---|---:|---:|---:|
| V4 candidate | 55.1096 | 355.8077 | 957.1667 |
| checkpoint24 baseline | 65.0183 | 361.3269 | 962.1667 |
| nominal rule | 16.1842 | 319.3077 | 939.5000 |

candidate 相对 baseline：等待减少 `9.9087 s`（约 15.2%），request-to-touchdown
减少 `5.5192 s`，平均完成时间减少 `5.0 s`。但 candidate 相对 rule：等待仍增加
`38.9254 s`，request-to-touchdown 增加 `36.5 s`，平均完成时间增加 `17.6667 s`。
因此正式评估的安全与完成 hard gates 为 PASS，且相对学习基线有改善；但低层策略尚不能
认定满意，不能进入 P3。

- heldout manifest SHA256：
  `167d22cb29d8ef6d08275cb0ff4cbe270c0d9005611a176b85d81fd64730f104`
- 一次性 seed reservation SHA256：
  `00f8eb2854841ab468108abee25d3aa2b52801a515f3bba8cb32547eb45031ff`
- progress SHA256：
  `67968286c94333336104d2fe1af9e204f369259e8113f169197dbae338a72fea`

## 4. Action tape 只读分解

从 18 个正式 tape 的最终 `release_state` 汇总全部 156 个 release lifecycle/控制器。
该分解不启动新 rollout，也不改变已消费 heldout 结论。

| 控制器 | actor decisions | WAIT | RELEASE | WAIT 占比 | lifecycle 平均审批延迟 / s | retries |
|---|---:|---:|---:|---:|---:|---:|
| candidate | 2,851 | 2,506 | 345 | 87.90% | 6.73 | 189 |
| baseline | 3,370 | 3,019 | 351 | 89.58% | 8.21 | 195 |
| rule | 不适用 | 不适用 | 不适用 | 不适用 | 4.43 | 691 |

candidate 的接口平均审批延迟分别为 o2m `5.65 s`、m2i `9.88 s`、i2f `4.65 s`；
baseline 分别为 `6.35 s`、`13.25 s`、`5.02 s`。candidate 相比 baseline 同时减少了
actor WAIT（513 次）和 release 后审批延迟，和 heldout 的等待改善方向一致。

每架 UAV 有三个 release lifecycle。按 lifecycle 均值折算，candidate 相对 rule 的
release 首次尝试后审批差约 `3 × (6.73 - 4.43) = 6.9 s/UAV`，显著小于正式评估中的
平均等待差 `38.9254 s/UAV`。rule 每步重复尝试的控制逻辑与 learned actor 不同，因而其
retry 次数不能与 actor 的 retry 次数直接作策略优劣比较；但现有数据足以排除“全部差距
主要来自首次尝试后的 SafetyFilter 重试”这一解释。当前首要待诊断对象是首次 RELEASE
尝试前的高 WAIT 倾向及其状态/优势信号，而不是放宽安全门槛。

## 5. Verification Status

- 训练协议与结果：**VERIFIED**（manifest、checkpoint 与哈希均存在）。
- heldout safety/completion gates：**VERIFIED PASS**。
- 相对 checkpoint24 效率改善：**VERIFIED on six paired seeds**，不外推统计显著性。
- “满意低层策略”与进入 P3：**NOT VERIFIED / BLOCKED**，原因是相对 rule 的等待与
  request-to-touchdown 差距仍大。
- 当前下一步：仅基于既有 tape 定位 pre-attempt WAIT 的 phase/interface/state 条件；若需
  新一轮训练，必须另行预注册 repair hypothesis、fresh seeds、预算与效果 gate。
