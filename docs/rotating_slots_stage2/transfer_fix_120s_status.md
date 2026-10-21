# Stage2 跨环候选修复：120秒成对pilot进行中

用户已人工观察旧3600秒回放，并明确不接受其调度行为。旧实现的人工状态更新为
**HUMAN_REJECTED_BEHAVIOR**；这与已执行轨迹的数值审计通过是不同判断。
本次修复实现和独立120秒成对pilot均为 **IN_PROGRESS**，新回放仍待人工观察；本文不预填结果或新版本哈希。

## 旧回放与只读反例

两方法旧回放的前120秒均无O→M或M→I开始/完成事件，仅4次正常final开始、2次接地。
实际非空正常提交位于0、80、100秒，均来自初始inner飞机。全3600秒的跨环开始/完成分别为：
Slack O→M 9/9、M→I 5/5；CP O→M 13/13、M→I 9/9。
首个O→M约769.978534秒开始，首个M→I约814.978534秒开始。
旧质量、耗时、轨迹、原审计FAIL及校正后PASS全部保留，见[旧完整窗口记录](repaired_campaign_status.md)。

真实busy t0快照的[只读probe](D:/7_codespace/training_outputs/terminal/stage2_transfer_diagnosis_20261021/probe_before.json)
记录`read_only=true`、`state_unchanged=true`，39架初态为14 outer、12 middle、10 inner及3架地面服务。
以下独立动作均通过原`evaluate_planning_snapshot`前缀检查：

| 飞机 | 合法目标 | 轨迹时长秒 |
|---|---|---:|
| initial_outer_00 | middle槽1 / 26 / 27 | 40 / 80 / 45 |
| initial_middle_00 | inner槽17 | 40 |
| initial_inner_00 | pad5 | 48.5 |

这些是分别验证的物理反例，不是联合提交或新算法候选结果。其他目标确有RESOURCE_CONFLICT及HARD_SEPARATION拒绝，
不能据此宣称安全约束与动作可用性无关。

## 主对话确认的实现缺陷与人工结论

本次排查基于修复前源码`f653a09`；旧正式运行的来源仍以原记录及指纹为准。
主对话确认：候选供应按目标ID优先穷举duration，未扫描的parent又被deferred子树压住；
共用catalog实际只覆盖低号槽位/pad，3秒预算或64次尝试在覆盖其余目标前耗尽。
这是算法候选供应/展开实现缺陷，不是PyVista显示问题，也不是CP求解时间不足：旧全窗口cp_solve仅累计3.469秒。
旧数值审计PASS只说明已执行轨迹通过相应账本/安全检查，不能证明调度行为合理，不能替代用户的行为验收。

pad空闲不等于飞机可以立即接地。旧t0两架已发起正常final，直到63.5秒才接地，正常进近本身需要时间。
但pad5整轮没有normal，且初始inner00在t0存在通过原检查的pad5动作；其长期闲置不能仅用进近耗时解释。

主对话已用Git log及blame核实：`stage2_planning.py`及其`_paths`、`_successors`、`get_columns`中的
target/duration顺序、deferred展开和预算处理均由新增基线提交`6d7d20d8`引入，此前历史没有该文件。
目前定位为本次共享候选模块的实现缺陷，不是此前仿真性能优化提交引入的切环回归；现行安全核可以认证上述合法动作，
3秒/64次预算放大了枚举覆盖问题。此结论不推广为任何其他性能优化均无问题。
此前验收缺少busy场景实际切环行为指标；单机多步测试和已执行轨迹的安全audit不足以覆盖这一验收缺口。

## 首轮覆盖修复失败与已授权算法修订

首轮修复保留原合同的几何排序启发式，验证记录为 **36 passed、2 skipped、2 failed**；
失败为T02/T16的isolated outer没有MOVE，不能记作修复通过。
[本地紧凑probe](../../artifacts/rotating_slots_stage2_20261021/transfer_diagnosis/probe_after_local_compact.json)
保留该轮事实：busy catalog耗时3.061001400463283秒，36架可决策飞机各3–5次尝试，仅3架inner有MOVE，outer/middle均无MOVE。
Slack选出的前缀为inner00→pad5、inner03→pad1、inner07→pad3，三条均为0–48.5秒；
`selected_certified=true`、`prefix_evaluation.accepted=true`。此probe保持`state_unchanged=true`，不是120秒运行结果。

主对话进一步确认：弱几何剩余时长低估逐阶段完成所需工作，统一best-first在64次内扩展许多partial而仍无完整链；
全duration描述符队列本身也消耗时间。主对话已向用户明确说明，并授权修订两方法共用的候选算法：
采用惰性target/start及duration游标；排序下界按原canonical径向/垂直约束和既有20秒起、5秒步长网格，
求逐阶段必要时长，叠加13.5秒下降，再与原几何下界取max。

这明确修订原合同§7.5的排序启发式，不声称该公式仍原样保持；冻结原合同文件不直接改写。
两方法共享此修订，物理、安全、能源、目标函数、seed及预算均不变，所有动作仍须通过原认证。
排序下界仅用于候选展开顺序，不能作为能源估计或可行性证书。
本次候选策略标识更新为`shared_multistep_columns_v2_geometry_screened`，用于区分上述启发式及枚举器修订；
v2最终本地回归 **40 passed、2 skipped，19.43秒**；服务器回归 **101 passed、1 skipped，19.42秒**，
真实CP的T09/T15通过。主对话已复核初始core差异，安全认证未改。
`accepted_first_target`计数已修正：只有目录`check_single`接受才消耗每个首目标2条的名额；
物理target/start的2个时长策略不变。

服务器默认busy冷启动catalog耗时3.157秒，inner有10条MOVE列、覆盖5架飞机，outer/middle仍无MOVE，原预算未变。
这一观测尚未证明busy连续切环；单次初始冷启动也不等于全120秒连续重规划行为。
主对话将按用户原授权启动120秒paired pilot验证该行为；当前尚未启动，不填写运行结果或修复成功结论。
整体仍 **IN_PROGRESS**，源码freeze及推送待Git交付流程完成，本文不预填新提交哈希。

## 本次授权与记账边界

本次仅授权修复后的**独立120秒paired pilot**，不跑3600秒，不与旧窗口拼接。
保持原seed、manifest、繁忙初态、几何、能源、安全阈值及算法预算；不新增实验组或扩大审计框架。
核心修改及上述回归已形成；最终源码交付、120秒运行产物和人工观察尚待完成。
预计输出根目录为`/home/user/training_outputs/terminal/stage2_transfer_fix_120s_20261021`，
同步本地为`D:\7_codespace\training_outputs\terminal\stage2_transfer_fix_120s_20261021`；此处不代表已启动或已产出结果。

累计既有计账为 **20569.8436471秒**（含此前保守600秒后处理）。本轮准备拟保守计600秒，
涵盖截至目前的诊断及测试，服务器实际prepare另计；这是待落实的记账口径，不是实测600秒或已启动记录。
本次成本接续计入，
不重置原10小时额度、34200秒计算截止或36000秒总上限，不重复计旧准备费用。
