# P3 adaptive-KL V7 准备记录（2026-09-26）

本页记录 V7 formal runner、协议、adaptive probe 终态和 formal 启动边界。它不替代 P3 qualification，也不把 probe checkpoint 自动晋级为正式模型。

## 当前状态

- 用户目标是完成有效的三组独立 `240` episode formal PPO，并在训练结束、模型冻结后对同一场景集合执行评估。
- 当前 adaptive probe 为 run0/run2 各 `8` 个回合，输出根为 `/home/user/training_outputs/terminal/p3_adaptive_kl_probe_v1_20260926/`，两组均 `8 UPDATED / 0 SKIPPED_KL / 0 HARD_FAILED`、`8/8` restore exact；accepted minibatches 为 run0/run2 `298/332`，最终 reference KL 为 `.0077864/.0133243`。
- probe 训练诊断各为 `132` requests，合计 `264`：run0 `110 normal + 22 emergency_landed`，run2 `117 normal + 15 emergency_landed`，合计 `227 normal + 37 emergency_landed`，`0 failed_energy`、`0 unresolved`；before/after development 均 full-resolution 且 hard failure 为 false。dense24 paired 为 run0 emergency `1→5`、priority wait `+15.6374s`、low-SOC wait `+40.9181s`，run2 emergency `3→1`、priority wait `+5.8564s`、low-SOC wait `+2.5737s`；稀疏 flow wait 有改善。该结果支持 primary 的有效更新审查，但不表示性能全面改善，也不晋级 probe checkpoint。
- primary 已允许从同一 03c0 source checkpoint、fresh Adam 启动 `3×240` formal。formal 输出根为 `/home/user/training_outputs/terminal/p3_adaptive_kl_formal_v7_20260926/`；run0/run1/run2 当前 `RUNNING`，child PID 为 `2641745/2641747/2641746`。control 路径为 `/home/user/training_outputs/terminal/p3_adaptive_kl_formal_v7_20260926/_control/runN/{pid,status.json,exitcode,...}`。三 run 的首回合 checkpoint 已确认：`processed/updated/skipped=1/1/0`、`ppo_called=true`、安全计数为零，gate delta 分别为 `.0245436/.0346563/.0341213`，checkpoint 不可 promotion，`P3/P4=false`。最新服务器采样快照显示三 run 均到 `3 processed / 3 UPDATED / 0 skipped / 0 hard`，首 checkpoint 的 model/Adam restore 与 RNG restore 均为 true；各 run 首个 flow6 均为 `6 normal / 0 emergency_landed / 0 failed_energy / 0 unresolved`。这些是采样快照，不宣称当前训练状态恒定。
- heartbeat `p3-v7` 每 `20` 分钟持续跟进到 formal training 与后续 `64` episode evaluation 完成；用户已授权该持续跟进，异常或终态通知，不自动重启。
- 冻结协议文件为 [`p3_adaptive_kl_v7.yaml`](../../../terminal_marl/configs/low_v4_2/p3_adaptive_kl_v7.yaml)，本地 SHA256 为 `d2bbd3a3361abb0589099fd2096b8f404f61c34bf94db10a958de99fc3f14a53`。

## 固定边界与候选正则

reference gate 的候选软惩罚为

```text
L_ref = 0.1 * K + 400 * relu(K - 0.02)^2
```

其中 `K` 为固定 03c0 reference gate distribution 到当前 gate distribution 的 weighted KL。reference hard cap `.05`、rollout-KL target `.01`、fresh Adam、proposal 子图冻结、SafetyFilter 和能量 hard gates 均保持原 V6 边界。此候选不增加 probability floor，不强制 RELEASE，不按 UID/SOC 覆盖动作，也不改变 proposal 或温度定义。

V7 adaptive probe 使用 `run_p3_adaptive_kl_probe_v1`：run0 seeds `860080–860087`，run2 seeds `862049–862056`；前四个回合用于既有停止点诊断重现，后四个为原计划中尚未消费的诊断回合。probe 是 admission evidence，不能作为独立 formal 训练或 qualification 样本。

本次 probe 已完成上述计划。它保留了原 03c0 reference、proposal 冻结和安全/能量边界；primary 依据有效更新机制允许新的 formal，但保留 dense24 priority/low-SOC 等待退化作为性能审查事实，不作普遍性能改善或 P3 结业结论。

formal 计划从同一 03c0 source checkpoint、每 run fresh Adam 开始，沿用 `[flow6, dense24, flow12, dense24]×60`：run0 `870001–870240`、run1 `871001–871240`、run2 `872001–872240`。development seeds 为 `873101–873106`，按 flow6/flow12/dense24 重复两次，在 processed episode `40/80/120/160/200/240` 做 paired full-resolution evaluation；heldout `854001–854012` 与历史 `829001–829006` 不消费。

训练后 evaluation 仅在 formal 完成并冻结模型后执行：同一场景集合使用 seeds `880101–880112` 对比 03c0 与各最终 checkpoint；另登记 dense30 no-inflow 和 same-load random dense24 stress。random layout opt-in 默认 `false`，只用于后续 evaluation，不反向改变训练协议。

## 实现与检查

V7 runner 已实现 checkpoint 的 model/Adam/reference/RNG 精确恢复，保存 processed/updated/skipped counters、per-update gate delta、最终 checkpoint SHA、配置快照和实际 wall time；development cadence 以 processed episode 计数，避免 skip 后重复同一评估路径。形式上仍区分 `UPDATED`、安全 `SKIPPED_KL` 与 hard failure；连续安全 skip 的 stalled 规则不等同于 P3 qualification。

本地定向集成测试覆盖 adaptive penalty、adaptive probe、recovery、V7 formal、random layout、GateReference、Student PPO、旧 KL guard 和 runtime shortlist，共 `69 passed`。V7 专项测试为 `7 passed`。evaluation runner 提交为 `5292945`；服务器 evaluation 定向测试为 `7 passed`，source-only 两步 smoke 为 `SMOKE_COMPLETED`。formal server preflight/test 为 `7 passed`（约 `1.38s`），三 run check-only 均为 `CHECK_ONLY_OK`，且每 run 已生成 predev summary/tapes、source/admission/cmdline 校验通过。另运行旧 SharedPPO 路径得到 `8 passed / 1 failed`；失败原始错误为：

```text
TypeError: SharedPPO.__init__() got an unexpected keyword argument 'policy_distribution'
```

该旧路径错误仅作原始事实记录，未据此归因 V7 或本轮 adaptive-KL 实现。原 V6 formal、bounded recovery、受限 emergency 与既有开发证据保持历史含义，不被本页替换；当前仍不能声称 formal 已完成、策略已晋级或 P3/P4 已通过。
