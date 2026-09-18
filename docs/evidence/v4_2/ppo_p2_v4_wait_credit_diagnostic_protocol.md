# P2 V4 WAIT / credit diagnostic protocol

> 状态：已执行，`DIAGNOSTIC_COMPLETE=false`；diagnostic-only；不授权 PPO 或课程升级

P2 V4 正式 heldout 已证明 candidate 相对 checkpoint24 baseline 有改善，但平均等待仍比
nominal rule 高38.9254s。已消费 action tape 只保留每回合末120秒，能够排除“差距主要来自
首次release尝试后的SafetyFilter重试”，却不足以恢复全回合pre-attempt WAIT、policy概率和
team advantage。不得重跑755001–755006，因此使用fresh diagnostic cohort补齐所需证据。

## 冻结设计

- candidate：P2 V4 `checkpoint_8.pt`，SHA256
  `5e6811b8349d875fb007943476aec62d93550a74ffaf4ae46614c5c188ab004f`；
- baseline：checkpoint24，SHA256
  `6420df9a8562135756819459c7682ec897d47acd64e841b75a21c697298ea14f`；
- formal seeds：771001–771006，场景按flow6/flow8/flow12循环两次；smoke seed771000；
- stochastic student behavior，用于计算on-policy team GAE；不更新模型或optimizer；
- Teacher仅以`teacher_label_only` sidecar输出WAIT/RELEASE审计标签，scheduler保持`execution`，
  Teacher不进入`env.step`，不替换student行为；
- 物理、Observation V2.1、V4.2 action、P2 progress reward=0、安全栈与自然流配置保持不变。

## 输出与判读

按controller、interface和confusion cell输出：requested/Teacher gate分布、false-WAIT率、
RELEASE概率/calibration、team GAE分布、WAIT与false-WAIT streak，以及release eligibility、
headway slack、retry remaining、wait feature和predicted conflict上下文。

诊断只回答以下分叉：

1. false-WAIT是否集中在Teacher已判RELEASE的状态；
2. false-WAIT是否伴随正team advantage，提示team-time credit对WAIT提供错误强化信号；
3. 问题是否集中于特定interface或retry/headway上下文；
4. candidate相对baseline的概率、误差和streak是否同步改善。

完整输出必须同时满足：无训练调用、student行为权威、Teacher label-only、全部请求完成、
安全与shield计数全0、requested与Teacher两类gate均出现、三个interface均覆盖。即使全部满足，也只标记
`DIAGNOSTIC_COMPLETE=true`；`PPO_ALLOWED=false`、`curriculum_promotion_allowed=false`。

实际终态与归因见 [P2 V4 WAIT / credit diagnostic summary](ppo_p2_v4_wait_credit_diagnostic_summary.md)。
