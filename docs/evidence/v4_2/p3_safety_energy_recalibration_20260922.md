# P3 安全接管、EH216-S 能量与 dense24 敏感性诊断（2026-09-22）

## 边界与实现

本轮更改了仿真语义，**没有**重写 P1/P2/P3 的历史训练或资格结果。当前主线的
`80 m` 是首选运行间隔，`40 m` 是需要 swept-segment 与递归后备证明的保护下限，
`<10 m` 是 collision event 的研究代理阈值。只有 bounded runtime shield 能证明
实际执行轨迹全程 `>=40 m`，才允许在 `40--80 m` 缓冲区接管并继续整个 episode；
接管即使数值修正为零也进入 shield reward cost。无法取得证明仍在物理执行前
fail-closed。速度可行性计算已取消在速度上限处预先耗尽侧向加速度的错误钳制，
改为按目标方向检查连续与有限步加速度。

能量参考按 EH216-S profile 的 `1500 s` 最大飞行时间映射为 `1500` 个归一化单位
（参考功率每秒 1 单位），phase 功率倍率仍是工程近似，而非实测功耗模型。
touchdown 必须保有 `SOC >= 0.25`；此前即达下限的 UAV 记
`DIVERTED_LOW_SOC`，不是完成。公开飞行时间只支持量级校准，不支持声称
不同 phase 的实际功耗已经标定。机型公开参数来源：
[EHang EH216-S 产品页](https://www.ehang.com/ehang216s/)及
[中国民航局 EH216-S 专用条件](https://www.caac.gov.cn/XXGK/XXGK/BZGF/ZYTJHHM/202202/P020220222611863596239.pdf)；
`80/40/10 m` 均非上述资料公布的机型法定间隔。

## 新种子诊断

脚本：`terminal_marl/training/diagnose_v4_2_p3_safety_energy_sensitivity_v1.py`。
冻结 original K3：SHA256
`4b8be66c100aaaeb5b5699dcce68a96dbe1237964e4c308e7b23b95c62fb8c8c`。
场景沿用 P3 V4 dense24：初始 12 架分布在 O/M/I 环 `[6,4,2]`，后续 12 架
每 15 秒流入，初始 SOC `.55--.90`，全程上限 1800 秒。除首选间隔
`60/80/100 m` 外，保护下限 40 m、collision 10 m、能量 1500、场景及控制器
均保持配对一致。seed `830101` 的 80 m 单例是启动前 pilot，不计入下表的
双种子矩阵。所有种子仅供本次探索性诊断，不得当作新的 P3 qualification。

| seed | 首选间隔 | rule 完成/24；完成者平均等待/请求至落地(s) | original K3 完成/24；完成者平均等待/请求至落地(s) |
|---:|---:|---:|---:|
| 830102 | 60 m | 24；45.465 / 233.125 | 24；82.031 / 268.292 |
| 830102 | 80 m | 24；63.027 / 244.958 | 23；97.603 / 277.348 |
| 830102 | 100 m | 22；200.503 / 360.864 | 22；196.686 / 374.500 |
| 830103 | 60 m | 23；96.128 / 274.391 | 24；77.344 / 263.042 |
| 830103 | 80 m | 23；122.090 / 298.826 | 18；247.478 / 419.778 |
| 830103 | 100 m | 20；231.402 / 396.400 | 22；185.416 / 359.955 |

12/12 cases、288/288 requests 均有确定终态：269 完成，19 低 SOC 分流；
全部 12 条 action tape exact replay，`SafetyInvariantError`、`<40 m`
hard-separation、NMAC、collision 和 uncertified execution 均为 0。
全矩阵 6 次 shield use、0 次实际 `40--80 m` buffer entry，因此矩阵支持
“新实现下这些回合没有失守”，**不能**单靠这些回合证明缓冲接管在真实密集
冲突中频繁或有效；该语义另有非 mock 的双机环境步进回归测试。
`80 m` 下 original K3 双种子只完成 `41/48`，rule 为 `47/48`；
完成者等待不能掩盖未完成。100 m 较 60 m 在这两个种子上明显增加等待和低
SOC 分流；只有两个种子，不能据此推断总体效果或选定正式阈值。

个体抽查：seed `830102`、80 m 的 original K3 中，UAV 13 于请求
30 s、入场 35 s 后在 `t=471 s` 从内环以 SOC `.249805` 分流；当时
内/外等待分别约 247.088/6 s，初始 SOC 由能量账本约为 `.5541`。
同 seed 的 rule 为 24/24，说明至少此例不能简单归为“初始电量天然
不可达”；调度等待同样是重要影响因素。seed `830103`、80 m 的 K3
有 6 架低 SOC 分流，初始 SOC 范围约 `.628--.772`，rule 仅 1 架分流。

证据：服务器 `~/training_outputs/terminal/p3_safety_energy_sensitivity_dense24_830102_830103_20260921/manifest.json`，
本地 Syncthing 镜像 `D:/7_codespace/training_outputs/terminal/p3_safety_energy_sensitivity_dense24_830102_830103_20260921/manifest.json`，
manifest SHA256 `b2fa3272122fcd88d80263c228288280e0ca09c34d73dacf615ea623ee034424`。
pilot seed `830101` original K3 为 24/24、841 秒、最小 swept separation
82.228 m、0 低 SOC 分流和 0 安全异常；pilot manifest 中 requests 字段因
诊断脚本早期汇总缺失显示 0，raw summary 与 tape 均为 24，故只作附加探索
证据，不进入上表或总计。脚本已增加 requests 字段回退并有回归测试。

为定向检查旧 seed `828001` 的第 533 秒故障，另用**进程内临时**
`energy_capacity=10000` 隔离能源新语义，保持原始 K3、80 m，运行 600 秒。
该诊断到 600 秒为 15/24 完成、9 架未解决；第 533 秒附近未再发生
`SafetyInvariantError`，最小 swept separation 84.394 m，0 shield、0
buffer entry、0 hard/NMAC/collision，600/600 exact replay。它说明更新后的
系统可越过原故障时刻，**不是**旧执行轨迹的逐状态重放，也不证明后续全程
24/24 完成。证据位于
`~/training_outputs/terminal/p3_828001_safety_repro_legacy_energy_20260922/manifest.json`；
旧 seed 已暴露，只用于事后故障复现，不能作新资格评估。

## 对课程与资格的影响

- P1/P2/P3 的旧结果保持其原协议下的历史证据，**不因代码更新而追溯重写**；
  但速度、安全、能量和 reward 语义已经改变，旧 checkpoint 不能凭旧 gate
  自动晋级到当前 MDP。P1/P2 应先在新语义、互不重叠的新种子上重新资格/效果
  评估；只有明显掉 gate 或效果退化时才需要重训，不应先盲目全量重训。
- P3 V3 final heldout 失败、Q3 完成率失败、V4 正式 PPO 0 次更新仍是终态。
  旧 V4 协议的 828xxx 开发种子已暴露；不能恢复旧进程，不能把本次诊断升级
  为正式 P3 资格。`P3_QUALIFIED=false`，`P4_ALLOWED=false`。
- 现有 P3 dense24 的 `.55--.90` 初始 SOC 加 25% 落地 reserve，与长尾等待
  共同导致完整完成门槛经常失败。下一轮应先区分“保证全员可达”的高 SOC
  dense24 资格场景和保留 `.55--.90` 的低 SOC 压力场景；后者报告完成率、
  分流、等待与安全，并对比同种子的 rule；不能把高 SOC 场景通过当作
  低 SOC 调度问题已解决，也不得以完成者条件均值替代全体绩效。两类场景的
  fresh seeds、上限、比较对象及 hard gates 均需在任何 P3 训练前另行冻结。

本报告是探索性诊断与下一轮协议输入，`PPO_ALLOWED=false`，不授权 checkpoint
promotion 或 P4/A6。
