# P3 V3 混合 PPO、末期安全边界与 fresh Q3

状态：2026-09-21。所有服务器结果位于 `~/training_outputs/terminal`，本地 Syncthing 镜像为 `D:/7_codespace/training_outputs/terminal`。本页区分训练 gate、一次性 heldout、暴露种子诊断和后续 fresh Q3；不把诊断中的好指标当作资格结果。

## 协议与训练

P3 保持 88D V4.2/K=3 bounded runtime、固定三环几何和 P2R dense24 资源语义。课程混合从空流入的 flow6/flow12 与初始三环有机、继续到达的 dense24；priority 取 `1/3/5`，初始 SOC 覆盖约 `.55–.90`。urgency 时间项为 priority 权重 `1.0`、SOC 权重 `2.0`、SOC 阈值 `.65`，旧配置默认关闭；完整 horizon 为 flow6 `1210 s`、flow12 `1570 s`、dense24 `1080 s`。训练从 original K3 SHA256 `4b8be66c100aaaeb5b5699dcce68a96dbe1237964e4c308e7b23b95c62fb8c8c` 重新开始，不承接 V2 失败模型。

- V2（无 urgency shaping）训练 formal 8/8 PASS，离线 coverage audit PASS；一次性 heldout 18/18 完整性通过，但 dense low-SOC wait 从 initial `151.0523 s` 恶化为 trained `191.8083 s`，且 dense priority wait 略恶化，因此未晋级。训练输出 `p3_mixed_priority_soc_training_v2_20260921`；heldout manifest SHA256 `fdfdaefb4ae6ae0e72eb016301b20990543ab63227e9348201308f907921bae6`。
- V3 protocol SHA256 `c69dc4b308e97652936c8da5bb6a47ef61a73a4f3cb599de8f39cb703f3c4d25`；entry 2/2、formal 12/12 的安全、回放、likelihood、更新和恢复 gate 全通过，194 个 lifecycle rows 覆盖 priority `1/3/5`、SOC `.55193–.89944`。训练 root manifest SHA256 `38703fa2af89a56f2ac46551f3c6006bf750547a7768f707990c7873e8209b76`，final checkpoint 12 SHA256 `0fb4acf9b05ba2fa8f5700e64eca6c2b9fc6174bb091c4e12ee4c1ea3942fb6b`。训练通过只授权一次性 heldout，不是部署/课程升级。

## Final heldout 失败与暴露种子诊断

V3 heldout protocol SHA256 `bdcd4be571555b0f512af30e8bf548cc11948b3347e18b552093ae1997e42e7a`，seeds `825001–825006`，trained/initial/rule 配对。前 15/18 records 完整性通过；第16条 trained/dense24/825006 在 684 秒、21/24 完成时触发 `SafetyInvariantError: No certified bounded joint backup action; physics execution blocked for components ((35, 118),)`。此前碰撞、NMAC、hard-separation、uncertified execution 均为0，但这个安全不变量本身是 hard failure；未运行剩余两条，未生成完整 heldout manifest，checkpoint 12 不可晋级。失败 summary/action tape SHA256 分别为 `080c7799a235fba1ca1c4b79d29f5bd0e00e44eec616cca1c5c3d6fb75dff3ec` / `97bfb66b974cfd4e11012733c71cb8c16936a20b359d9af3bbd43d4970a50d6f`。

已暴露的 825006 只用于安全边界归因。冻结诊断协议 SHA256 `1bd3af56afc3b7db5199125a2ef5ca6c1205913966d19c653e4cfd901a118269`，对 original K3 与 12 个 V3 检查点直接 deterministic rollout。13/13 完成、diagnostic integrity PASS；index 0–11 均安全且 24/24 完成，只有 checkpoint 12 再现同一安全错误、21/24。manifest SHA256 `e2be6b10f9cad30262bf68c3b9b0d57e73d9d5408f08cb7fe77f9e9f0ed40dfa`，其 `checkpoint_promotable/P3_QUALIFIED/P4_ALLOWED` 全为 false。checkpoint 4–6 在此单个暴露 seed 上的低 SOC wait 均约 `58.427 s`、priority-weighted wait 约 `80.76–80.81 s`；这些只用于形成下一轮候选，不是泛化证据。

## 事后早停候选与 fresh Q3

在看过上述暴露诊断后明确选定 checkpoint 5（SHA256 `19c31195dd881e7e4fde91530aff00375f2787754569491749474699b7f06b2d`）：它所在正式 dense24 回合 24/24 完成、安全0，且处于诊断的 4–6 效率平台。选择为 **posthoc**，没有改写 V3 final heldout；新的 Q3 是对此固定候选的全新种子确认。

Q3 protocol SHA256 `9dad9f8287d02ef1ac584553fab35c23089a08f2c0d940472778cb1ccb367dc8`；历史 registry 在第一次 reset 前扫描 4,333 个文件、2,177 个历史 seeds，`826001–826006` 交集为零。六个 seeds 按 flow6/flow12/dense24 重复两轮，trained/initial/rule 共 18 条；full horizon、urgency 配置和安全/回放/likelihood/geometry/Teacher isolation 与旧 paired heldout 同门槛。Q3 marker SHA256 `1181432fb82702c3480cee8e12b97cd07d7722c3f88ef3296a92b22519922923`，manifest SHA256 `2ad3c6e52d52a2431f86284cd604911aed99329bbc9af897fbbb723548309e8c`。

18/18 episode integrity 和全部安全计数通过；sparse 两控制器均 100% 完成，trained priority-weighted wait `6.237 s` vs initial `6.738 s`。dense 第一组 826003 三控制器均 24/24，但第二组 826006 trained `13/24`、initial `24/24`、rule `22/24`；trained dense 聚合完成率 `.771` vs initial `1.0`，触发 completion non-worse 与 strict-improvement 两个 hard decision gates，故 `PASS=false`、`P3_QUALIFIED=false`、`checkpoint_promotable=false`、`P4_ALLOWED=false`。

826006 trained 在 1080 秒末仍有 11 架活动（inner 7、middle 4），6 个降落点全空、无 reservation；最后 touchdown 在 380 秒，整个回合 23 次 shield、459 次 shortlist exhausted。initial 在 911 秒前 24/24 完成；rule 在 1080 秒为 22/24。trained 的 dense priority-weighted wait `105.457 s` vs initial `194.826 s` 只对已完成的 lifecycle rows 聚合；826006 trained 仅 13 rows、priority 权重分母35，而 initial 24 rows、分母72。未完成者的等待未计入这个数字，不能解释为吞吐改善或安全效率达标。

只读检查 action tape 的末段 `960–1079 s`：11 架的 1,320 个 actor-step 中 WAIT 请求1,222次、RELEASE 请求98次；98次中63次三项 shortlist 用尽后 `safety_reject`、35次 `safety_wait`，**批准0次、切环/进近 phase transition 0次**。UID88全段未请求 RELEASE，但其余10架均有请求。末态累计 release intents 的 `primary_blocker` 有 `NO_MERGE_GAP` 8个、`PREDICTED_SEPARATION` 2个；无最终 headway/final-capacity blocker。这说明长尾不是“所有无人机完全不请求”或 LZ 直接不足，而是这条 learned 轨迹形成了请求也无法被当前 K3 安全认证的拥塞状态；初始 K3 在同 seed 完成24/24，因而不是这个初始场景必然不可达。`action_tape.applied_action`/`executed_mapped_action` 表示提交给执行路径的意图，并不等于 `last_release_decision=approved`，判断是否推进必须看后者、shortlist rank 和 phase transition。哪些较早决策将系统推入该状态，现有末段快照不足以逐步归因，不能在未做新协议审查前直接改动安全阈值。

随后在与原评估一致的 Linux 环境对 1080 步原始动作带及 Top-3 shortlist 作逐步 exact replay，reset/config/observation/post-state 和执行映射均核对通过；此运行不重新采样策略或消费新 seed。诊断协议 SHA256 `49d43d565127e9cc01a6fa35f8ead1613ff8a3e9e7c02c6f9b38ca318a8e747c`，manifest SHA256 `9202e7154a34bc065624a7173a1f86897ebb92b6a775d223d459ad4e22708d3b`；`PASS=true` 只表示诊断回放完整，晋级标记全为 false。按半开时间段汇总：

| 秒段 | 新获批 RELEASE | safety reject / retry wait | 新完成 | 主阻塞来源 |
|---|---:|---:|---:|---|
| 0–180 | 26 | 31 / 3 | 3 | predicted separation 34 |
| 180–380 | 23 | 45 / 7 | 7 | no merge gap 6、separation 46 |
| 380–600 | 0 | 122 / 56 | 3 | no merge gap 87、separation 91 |
| 600–800 | 0 | 111 / 52 | 0 | no merge gap 88、separation 75 |
| 800–960 | 0 | 87 / 37 | 0 | no merge gap 68、separation 56 |
| 960–1080 | 0 | 63 / 35 | 0 | no merge gap 56、separation 42 |

因此放行获批累计值自 `t=360 s` 的49次直到结束不再增长；380秒后那3架完成是先前已获批航迹的排空。全回合累计 `NO_MERGE_GAP=305`、`PREDICTED_SEPARATION=344`、reject459、retry wait190。末态 11 架 priority `1/3/5` 与 SOC 均有分布（最低末态 SOC约 `.475`），不能把滞留简单归为低优先级或不可完成低电量。这个时间线把下一轮修复目标限定为**避免形成早期不再有可认证 merge/final 的状态**，而不是在后期放松安全认证或只增加 landing-zone 容量。具体应采用何种训练目标/课程和可重复 gate，需在新的训练协议中冻结，并由 fresh dev/heldout 验证。

## 约束与后续

V2、V3 final heldout、frontier、Q3 均保留不可变终态。`825006` 与 `826001–826006` 已用于选择/确认，不可拿来训练、重新选择检查点、调阈值或再作资格评估。下一轮应先明确后半段 dense24 停滞机制，并在新协议中提前冻结训练/开发/heldout种子、完整 cycle、未完成惩罚与完成率、安全不变量及与 initial/rule 的配对阈值；任何新 PPO 都必须重新进入 fresh gate。P4 交通压力、P5 几何泛化与 A6 动态空域仍无授权。
