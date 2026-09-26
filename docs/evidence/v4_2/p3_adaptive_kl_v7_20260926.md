# P3 adaptive-KL V7 准备记录（2026-09-26）

本页记录 V7 formal runner、协议和 admission 边界的准备事实。它不替代外部 admission report，也不把正在运行的 bounded probe 当作 formal 结果或 P3 qualification。

## 当前状态

- 用户目标是完成有效的三组独立 `240` episode formal PPO，并在训练结束、模型冻结后对同一场景集合执行评估。
- 当前 adaptive probe 为 run0/run2 各 `8` 个回合，输出根为 `/home/user/training_outputs/terminal/p3_adaptive_kl_probe_v1_20260926/`，正在运行；目前没有 probe 终态、formal admission 或 P3 pass 结论。
- formal runner 尚未准入。计划输出根为 `/home/user/training_outputs/terminal/p3_adaptive_kl_formal_v7_20260926/`，formal 仍需 primary 绑定 `P3_ADAPTIVE_KL_FORMAL_ADMISSION_V7` report。
- 冻结协议文件为 [`p3_adaptive_kl_v7.yaml`](../../../terminal_marl/configs/low_v4_2/p3_adaptive_kl_v7.yaml)，本地 SHA256 为 `d2bbd3a3361abb0589099fd2096b8f404f61c34bf94db10a958de99fc3f14a53`。

## 固定边界与候选正则

reference gate 的候选软惩罚为

```text
L_ref = 0.1 * K + 400 * relu(K - 0.02)^2
```

其中 `K` 为固定 03c0 reference gate distribution 到当前 gate distribution 的 weighted KL。reference hard cap `.05`、rollout-KL target `.01`、fresh Adam、proposal 子图冻结、SafetyFilter 和能量 hard gates 均保持原 V6 边界。此候选不增加 probability floor，不强制 RELEASE，不按 UID/SOC 覆盖动作，也不改变 proposal 或温度定义。

V7 adaptive probe 使用 `run_p3_adaptive_kl_probe_v1`：run0 seeds `860080–860087`，run2 seeds `862049–862056`；前四个回合用于既有停止点诊断重现，后四个为原计划中尚未消费的诊断回合。probe 是 admission evidence，不能作为独立 formal 训练或 qualification 样本。

formal 计划从同一 03c0 source checkpoint、每 run fresh Adam 开始，沿用 `[flow6, dense24, flow12, dense24]×60`：run0 `870001–870240`、run1 `871001–871240`、run2 `872001–872240`。development seeds 为 `873101–873106`，按 flow6/flow12/dense24 重复两次，在 processed episode `40/80/120/160/200/240` 做 paired full-resolution evaluation；heldout `854001–854012` 与历史 `829001–829006` 不消费。

训练后 evaluation 仅在 formal 完成并冻结模型后执行：同一场景集合使用 seeds `880101–880112` 对比 03c0 与各最终 checkpoint；另登记 dense30 no-inflow 和 same-load random dense24 stress。random layout opt-in 默认 `false`，只用于后续 evaluation，不反向改变训练协议。

## 实现与检查

V7 runner 已实现 checkpoint 的 model/Adam/reference/RNG 精确恢复，保存 processed/updated/skipped counters、per-update gate delta、最终 checkpoint SHA、配置快照和实际 wall time；development cadence 以 processed episode 计数，避免 skip 后重复同一评估路径。形式上仍区分 `UPDATED`、安全 `SKIPPED_KL` 与 hard failure；连续安全 skip 的 stalled 规则不等同于 P3 qualification。

本地定向集成测试覆盖 adaptive penalty、adaptive probe、recovery、V7 formal、random layout、GateReference、Student PPO、旧 KL guard 和 runtime shortlist，共 `69 passed`。V7 专项测试为 `7 passed`。另运行旧 SharedPPO 路径得到 `8 passed / 1 failed`；失败原始错误为：

```text
TypeError: SharedPPO.__init__() got an unexpected keyword argument 'policy_distribution'
```

该旧路径错误仅作原始事实记录，未据此归因 V7 或本轮 adaptive-KL 实现。原 V6 formal、bounded recovery、受限 emergency 与既有开发证据保持历史含义，不被本页替换；当前仍不能声称 formal 已完成、策略已晋级或 P3/P4 已通过。
