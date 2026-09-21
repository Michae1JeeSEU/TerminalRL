# P3 dense24 长尾验证与 V4 完成率保护训练

状态：2026-09-21。此页记录 Q3 失败之后的新诊断与预注册训练；诊断 PASS 不代表策略合格，V4 formal PASS 也只授权独立 heldout。

## 为什么不能只延长 episode

对 Q3 已暴露的 dense24 seed `826006`，以相同策略、固定场景和安全约束将上限由 1080 秒延至 1800 秒。训练候选 checkpoint 5 仍为 `13/24`，最后一次 touchdown 在第 380 秒，结束时 11 架仍在环上；原始 K3 为 `24/24`，第 911 秒终止；rule 为 `22/24`。三者均 exact replay、安全错误计数为 0。诊断 manifest SHA256 为 `efd8e4fab6467954c6419570016b8889074a561902f8680fe76e58906fd5f3f9`，输出目录 `p3_q3_horizon_extension_v1_20260921`。这证明当前 learned 长尾并非仅由 1080 秒截断造成，也不能归咎于 1 秒决策步长：同一时间分辨率和安全过滤下，原始 K3 能完成。

完成者平均等待 `41.486 s`（trained）与 `183.914 s`（original K3）不能直接比较优劣，因为前者遗漏了 11 架的等待，是删失样本。策略导致的早期合流顺序、后续无可认证间隔与稀疏 RELEASE 请求共同构成当前观测到的停滞；现有证据不支持放宽 SafetyFilter，也尚不足以唯一归因到某一个早期动作。

## 终止与比较规则

episode 在所有请求完成时立即终止，未完成时到有限上限终止并显式判为吞吐未达标；不采用无限延长直到全部完成。只有完整完成、无安全违例且回放一致的回合，才用 request-to-touchdown、priority-weighted wait、低 SOC wait 和最后 touchdown 时间比较效率。未完成回合必须同时报告 `completed/requests`、积压数和删失状态，不能拿完成者的低等待抵消未完成。

## V4 预注册训练

协议 `terminal_marl/configs/low_v4_2/p3_mixed_priority_soc_training_v4.yaml`，SHA256 `50fac4e44171239e3e09458987808c4a6567cfc36178181b7906e8c18ad19a54`，代码提交 `25cec79`。从原始 K3 checkpoint SHA256 `4b8be66c100aaaeb5b5699dcce68a96dbe1237964e4c308e7b23b95c62fb8c8c` 和全新 Adam 开始；保持 V3 的 88D/K3、轨迹、安全、奖励与 1 秒决策步长。flow6/flow12 上限仍为 1210/1570 秒，dense24 唯一场景改动为上限 1800 秒，三环初始 6/4/2 架、后续再到达 12 架不变。

- entry：`827001/827002`，短 dense24/flow6，更新丢弃。
- formal：`827101–827106`，dense24/flow6/dense24/flow12/dense24/dense24；完整上限、随机策略，每回合完成率仅报告，安全、回放、PPO 更新仍为 hard gate。
- dev：`828001/828002`，每次更新后确定性 dense24、完整 1800 秒；先记录原始 K3 基线。仅首个两回合均 24/24、全部安全与完整性通过、priority-weighted wait 严格优于基线、低 SOC wait 不劣于基线的检查点可被选中，并立即停止后续训练。基线若不能 24/24，当前协议停止，不能用删失等待充当基准。
- untouched heldout：`829001–829006`，预留但训练脚本绝不评估。即使 V4 formal PASS，`P3_QUALIFIED=false`、`P4_ALLOWED=false`，须再经一次性独立评估。

此协议没有改动安全阈值或以 Teacher 更新 PPO。若 V4 仍不能选出合格检查点，应基于新证据另行预注册修复，不改写 V4 或重用这些 seeds。

首次启动在历史种子扫描和预约标记落盘之后、首次 environment reset 之前，因 `set` 无法 JSON 序列化而退出；输出仅有 `registry_progress.json`，没有 entry/formal 产物。预约标记 SHA256 为 `cdbba071cd4bf659399acb4c71ad172e9e57d99a45653b9ac152d22e920f351b`。后续仅允许通过校验此标记、原输出目录唯一文件和冻结种子清单的 `--resume-pre-reset` 路径继续同一次协议，不删除标记、不新建种子或把这次失败算成训练回合。

恢复执行后，entry 两个短回合及其 episode gates 全 PASS。formal 首先执行原始 K3 的 dense24 开发基线，seed `828001` 在第 533 秒、完成 `12/24` 时出现 `SafetyInvariantError: No certified bounded joint backup action; physics execution blocked for components ((13, 28),)`；碰撞、NMAC、hard separation 和 uncertified execution 计数仍为零，是 fail-closed 阻止了后续物理执行。runner 随后在 likelihood exact replay 同一安全异常处退出，未写 root manifest；formal PPO 更新 **0 次**、未产生检查点，dev 第二种子与全部 heldout 均未评估。恢复日志 `p3_mixed_priority_soc_training_v4_20260921.resume.log`，服务器进程已退出。V4 不得追溯算作 PASS 或通过同一种子重启；完整原因须基于已有动作带另作只读归因。

只读检查现有动作带 `safety_trace_seed_828001_step_533.json`（SHA256 `76da558666497fe7610ed1ead69c1ff41c4a3033b8a22dd784c71e0420f77a39`）：最后 60 个 step 共 509 个 actor 状态，WAIT 请求 492、RELEASE 请求 17，仅 1 次获批；10 次 safety reject、4 次 safety wait、83 次 shield 介入。13 个主阻塞标记为 `PREDICTED_SEPARATION`，1 个为 `NO_MERGE_GAP`。失败当步无人请求 RELEASE，UID13/28 均在内环、未 committed、请求残差均为零；当前预测最小间隔约 `79.48 m`，但递归门槛为 `backup_existence`，没有找到下一执行状态可认证的有界联合备份。动作带止于失败 step，没有 post-state；这表明失败是面向未来可恢复性的认证失败，不能把当前距离或当步放行动作单独解释为根因，也不能据此直接降低安全约束。summary SHA256 `6b300859be373a64f554869dcd7c9bd05d21111e6ad3596c3bb5e1f00b4b7217`。
