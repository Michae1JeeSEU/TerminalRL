# Stage2 跨环候选修复：v3的120秒pilot完成，人工待观察

用户已人工观察旧3600秒回放，并明确不接受其调度行为。旧实现的人工状态更新为
**HUMAN_REJECTED_BEHAVIOR**；这与已执行轨迹的数值审计通过是不同判断。
v2的独立120秒成对pilot已结束，行为修复 **NOT ACCEPTED**；其实际结果保留在下文。
v3已完成独立120秒成对pilot，主对话接受其展示两类实际切环，作为本次指定短窗口修复验证；
这不代表人工验收、最优性、实时保证或3600秒性能。新回放人工状态仍 **HUMAN_PENDING**。
最新实际结果列于前部；v2未获接受及v3运行前开发记录保留在后部历史记录。

## v3实际120秒结果

运行源码为`bede731fdf871829bf92d792b3c9450a6317167f`，source hash为
`5751ce59d0a828fdbc08a4c1041d0437b62ba774cc763061da3d82fd11448eea`，
候选ID为`shared_multistep_columns_v3_reference_seed`。
在v2几何筛选及惰性枚举基础上，v3先按环交错、环内能量优先顺序完成各机参考链种子，再展开其他候选；
仍由原认证检查动作，未改变物理、能源、安全阈值、目标函数、seed或预算。
源码提交`b5201eb`和`bede731`均已推送，旧结果和失败probe均保留。

实际根目录为`D:\7_codespace\training_outputs\terminal\stage2_transfer_fix_120s_v3_20261021`。
UTC **2026-10-21T10:29:38Z–10:35:23Z**，wall345.036秒、退出0，共同120秒，
campaign为 **PARTIAL_REQUESTED_WINDOW**。双方无`error.json`或`invalid.json`，守恒true，`terminated_invalid=false`。

| 120秒指标 | Energy-slack | RH-CP-SAT |
|---|---:|---:|
| normal / emergency / failure | 4 / 0 / 0 | 6 / 0 / 0 |
| O→M开始 / 完成 | 3 / 2 | 3 / 3 |
| M→I开始 / 完成 | 6 / 5 | 7 / 7 |
| 正常final开始 / 接地 | 6 / 4 | 7 / 6 |
| admitted / 空中pending / 上游pending | 44 / 40 / 0 | 44 / 38 / 0 |
| 总终端能量J / C_legacy | 361869009.687559 / 4690.505771389637 | 358116408.1101329 / 4628.595822453194 |
| 审计status / issue_count | PASS / 0 | PASS / 0 |

两方法首个O→M、M→I均在t0开始、t40首次完成；首个正常final均在t0开始、t48.5接地。
计数使用实际`maneuver_started`及轨迹哈希，完成使用`transfer_completed`/`touchdown`，包含t120精确边界。
按fixture原始ring识别，Slack接地为initial_inner_03（48.5）、initial_middle_03（88.5）、
initial_middle_00（93.5）、initial_middle_06（98.5）；CP为initial_inner_03（48.5）、initial_inner_01（58.5）、
initial_inner_07/initial_middle_00/initial_middle_05（均93.5）、initial_inner_02（119.590051）。
这些均为正常接地；初始outer及新请求在此窗口双方均未接地。Slack的initial_outer_04/00已开始final，
结束时刻分别128.5/133.5，截止仍在途，不计成功或失败。

| 全部callback/dispatch的wall统计 | Energy-slack | RH-CP-SAT |
|---|---:|---:|
| N | 57 | 59 |
| 正常P95 / max秒 | 4.798884 / 5.198976 | 5.914836 / 6.050298 |
| 正常>5秒 | 1/57（1.7544%） | 37/59（62.7119%） |
| 完整dispatch P95 / max秒 | 5.261256 / 5.736105 | 6.059527 / 6.710729 |
| 完整dispatch>5秒 | 9/57（15.7895%） | 44/59（74.5763%） |
| 正常CPU累计秒 | 251.206351 | 301.868329 |
| 完整dispatch CPU累计秒 | 261.077644 | 311.642776 |
| noop / 非空调用 / 提交前缀数 / fallback | 49 / 8 / 15 / 0 | 51 / 8 / 17 / 0 |
| CANDIDATE_WORK_CUTOFF / 目录不完整调用 | 57 / 57 | 59 / 59 |
| SOFT_BUDGET标签调用 | 0 | 34 |
| 决策status | FEASIBLE57 | LEX_LIMITED55；CATALOG_LEX_OPTIMAL4 |
| raw CP阶段status | 不适用 | OPTIMAL328 |

最多的已记录子阶段wall为reference_seed_generation 162.287/170.180秒（Slack/CP）；
Slack selection为69.874秒，CP hint construction为73.453秒、selected validation为33.609秒、cp_solve为0.152751秒。
正常callback整体wall为251.450/302.302秒，公共准入9.928/9.752秒、公共应急0.021939/0.023142秒。
阶段有包含关系，不能重复相加；这些是测得成本，不推断全局最优、实时保证或长窗口性能。
截断说明有限目录未穷尽，不能把未找到称为全局不可行；328个OPTIMAL为有限模型阶段状态。

两份[Slack审计](D:/7_codespace/training_outputs/terminal/stage2_transfer_fix_120s_v3_20261021/energy_slack/audit_0120.json)、
[CP审计](D:/7_codespace/training_outputs/terminal/stage2_transfer_fix_120s_v3_20261021/rh_cp_sat/audit_0120.json)
均PASS0、continuous_separation=CERTIFIED、physical_outcome=IN_PROGRESS，分别154/157帧，5968/6002能量区间，1794/2018 pair runs。
审计物理标签仅描述此短窗口，不代表运行仍进行。
[comparison_curves.png](D:/7_codespace/training_outputs/terminal/stage2_transfer_fix_120s_v3_20261021/comparison_curves.png)
及双方`replay_0120`均存在，主对话已检查曲线和审计。campaign及summary记录核心0/2和四库线程1；
worker PID文件为空，不能由该文件核验启动时的实际affinity，本记录不混同配置记录与启动进程观测。

服务器108 passed、1 skipped的qlog不含逐项记录；结合环境锁OR-Tools9.15.6755及测试分支，
唯一skip为已安装依赖时不适用的缺依赖负向测试，T09/T15实际CP测试未因缺依赖跳过。
主对话接受本次120秒pilot展示两类切环，人工仍HUMAN_PENDING；旧回放的HUMAN_REJECTED_BEHAVIOR不改写。

累计费用以较保守的`campaign.campaign_wall_s=22180.690032155544`秒作为后续carry，
`budget.elapsed_floor_s=22180.688587903976`单列保留；按campaign末值计算剩余计算额度为 **12019.309967844456秒（约12019.31秒）**。
此前21488.54562318558加本次准备302.4508186278399，得到启动floor21790.99644181342；
300秒credit已含于准备费用，不重复加。恢复文件中的12409.003558秒是启动时剩余，不能当作结束剩余；原10小时预算不重置。

### 本地人工回放命令（尚未执行）

以下路径、Python入口及viewer的`.run()`接口已核验；命令将打开人工窗口，不自动代签观察状态。
在`D:\7_codespace\terminal`的PowerShell分别执行：

```powershell
.\venv\Scripts\python.exe -c "from terminal_marl.visualization.rotating_slots_replay import ReplayController,RotatingSlotsReplayViewer; RotatingSlotsReplayViewer(ReplayController(r'D:\7_codespace\training_outputs\terminal\stage2_transfer_fix_120s_v3_20261021\energy_slack\replay_0120')).run()"
.\venv\Scripts\python.exe -c "from terminal_marl.visualization.rotating_slots_replay import ReplayController,RotatingSlotsReplayViewer; RotatingSlotsReplayViewer(ReplayController(r'D:\7_codespace\training_outputs\terminal\stage2_transfer_fix_120s_v3_20261021\rh_cp_sat\replay_0120')).run()"
```

### 本次修复文件范围

`b5201eb`及`bede731`合计涉及以下文件，服务器针对性回归为108 passed、1 skipped，不代表全仓库回归：

- [stage2_planning.py](../../terminal_marl/core/rotating_slots/stage2_planning.py)、[trajectory.py](../../terminal_marl/core/rotating_slots/trajectory.py)、[stage2_campaign.py](../../terminal_marl/core/rotating_slots/stage2_campaign.py)：候选供应、几何筛选/排序、完整参考链种子及策略指纹。
- [probe_stage2_normal_actions.py](../../scripts/diagnostics/probe_stage2_normal_actions.py)：只读候选与合法动作probe。
- [test_stage2_candidate_coverage.py](../../tests/test_stage2_candidate_coverage.py)、[test_stage2_normal_action_probe.py](../../tests/test_stage2_normal_action_probe.py)：覆盖与probe针对性测试。
- [repaired_campaign_status.md](repaired_campaign_status.md)、本文件：旧人工复查和修复过程/短窗口结果。

## 历史记录（保留原观测与运行前状态）

以下为各阶段当时记录，未启动、待验证等表述不代表上方v3 pilot的当前状态。

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

## 本次授权与记账边界（v2运行前历史）

本次仅授权修复后的**独立120秒paired pilot**，不跑3600秒，不与旧窗口拼接。
保持原seed、manifest、繁忙初态、几何、能源、安全阈值及算法预算；不新增实验组或扩大审计框架。
核心修改及上述回归已形成；最终源码交付、120秒运行产物和人工观察尚待完成。
预计输出根目录为`/home/user/training_outputs/terminal/stage2_transfer_fix_120s_20261021`，
同步本地为`D:\7_codespace\training_outputs\terminal\stage2_transfer_fix_120s_20261021`；此处不代表已启动或已产出结果。

累计既有计账为 **20569.8436471秒**（含此前保守600秒后处理）。本轮准备拟保守计600秒，
涵盖截至目前的诊断及测试，服务器实际prepare另计；这是待落实的记账口径，不是实测600秒或已启动记录。
本次成本接续计入，
不重置原10小时额度、34200秒计算截止或36000秒总上限，不重复计旧准备费用。

## v2的120秒pilot实际结果：行为修复未获接受

以下为运行后的当前记录；上文“尚未启动”“待freeze/推送”等属于运行前历史。
候选v2运行源码为`b5201eb98deb53ca948c671ad894bae422f6fbac`，source hash为
`b1121234ddb21c151b61ae2ccb23b39d9a210a047627888ae1e0254c7c4af294`，
策略ID为`shared_multistep_columns_v2_geometry_screened`，见实际[provenance](D:/7_codespace/training_outputs/terminal/stage2_transfer_fix_120s_20261021/provenance.json)。
运行UTC **2026-10-21T10:00:09Z–10:05:02Z**，退出0，共同窗口120秒，campaign状态为 **PARTIAL_REQUESTED_WINDOW**，
不是完整3600秒结果。实际产物位于上述`stage2_transfer_fix_120s_20261021`输出目录。

| 120秒指标 | Energy-slack | RH-CP-SAT |
|---|---:|---:|
| normal / emergency / failure | 6 / 0 / 0 | 6 / 0 / 0 |
| O→M开始 / 完成 | 0 / 0 | 0 / 0 |
| M→I开始 / 完成 | 3 / 3 | 4 / 3 |
| 正常final开始 / 接地 | 7 / 6 | 7 / 6 |
| 正常callback P95秒 | 4.62477 | 5.43019 |
| 完整dispatch P95秒 | 5.17203 | 6.01185 |
| 独立审计status / issue_count | PASS / 0 | PASS / 0 |

移动计数来自`replay_0120`的events与trajectories哈希关联：开始使用`maneuver_started`，
跨环完成使用`transfer_completed`，接地使用`touchdown`；normal/emergency/failure及时间分位数来自双方`summary_0120.json`。
两份[Slack审计](D:/7_codespace/training_outputs/terminal/stage2_transfer_fix_120s_20261021/energy_slack/audit_0120.json)、
[CP审计](D:/7_codespace/training_outputs/terminal/stage2_transfer_fix_120s_20261021/rh_cp_sat/audit_0120.json)
均为连续间隔CERTIFIED、physical_outcome=IN_PROGRESS；该物理标签描述120秒窗口，不代表运行进程仍在进行。

主对话根据双方O→M仍为0/0，判定本轮**行为修复NOT ACCEPTED**。审计PASS不代替行为验收，
上述耗时也不构成实时保证。新120秒回放人工状态仍 **HUMAN_PENDING**，旧3600秒回放的人工结论仍为
**HUMAN_REJECTED_BEHAVIOR**。
代码来源归因保持前述结论：共享候选搜索由基线`6d7d20d8`引入，目前没有证据表明此前仿真加速禁止了安全切环，
但没有穷尽排除全部历史性能优化问题。

本轮结束的campaign累计记账为 **21488.54562318558秒**；后续工作必须继承该累计值，不重置原10小时预算。
实际[budget](D:/7_codespace/training_outputs/terminal/stage2_transfer_fix_120s_20261021/budget.json)
仍保留34200秒计算截止和36000秒总额。
主对话正在实施早期参考链种子修订，尚未验证；该后续实现仍 **IN_PROGRESS**，此处不预填下一版结果、提交或成功结论。

## v3早期参考链种子（中间开发历史）

主对话反馈的primitive round-robin种子中间probe未满足完整参考链覆盖：本地outer/middle/inner有MOVE的飞机数为
**0/0/5**，服务器为 **0/10/5**，服务器候选阶段约3.08秒。这些是中间probe观察，不是最终v3或120秒成对运行结果。

当前修订将种子pass改为每机completion-first，在原每机64次尝试及共享3秒候选预算内优先完成该机参考链；
飞机顺序采用确定性的outer、middle、inner交错，各环内按能量优先级排序。
这是两方法共用候选排序的显式修订，不声称严格保持原合同枚举顺序；Energy-slack自身的求解选择优先级不变。
其他算法规则、物理、安全、能源及尝试预算保持不变，动作仍须原认证；上述中间probe不构成最终v3验证，后续证据见下，pilot结果尚未形成。

## v3最终本地与服务器证据（运行前历史）

主对话已复核最终实现；本地针对性验证为 **47 passed、2 skipped，23秒**。服务器回归已完成，pilot尚未启动。
一次冷启动[focused本地probe](../../artifacts/rotating_slots_stage2_20261021/transfer_diagnosis/probe_complete_chain_focused_local.json)
的catalog耗时3.1557秒，outer/middle/inner分别有 **3/1/2** 条MOVE，均为完整链；物理尝试合计36次，单机最大10次，
记录1次RESOURCE_CONFLICT拒绝。这是单次有限目录证据，不是120秒连续行为验证。

使用同一catalog执行一次Energy-slack选择，见[selected artifact](../../artifacts/rotating_slots_stage2_20261021/transfer_diagnosis/probe_complete_chain_focused_local_selected.json)：
选出outer/middle/inner各1条，`certified=true`且原前缀检查接受，总budget elapsed为4.7385秒。
实际选择前缀为outer01→middle槽1（0–40秒）、middle00→inner槽17（0–40秒）、inner00→pad5（0–48.5秒）。
这些本地结果不预判服务器验证或pilot成功，人工观察仍HUMAN_PENDING。

服务器最终回归为 **108 passed、1 skipped，pytest20.82秒、wall21.0388秒，退出0**，
UTC2026-10-21T10:22:39Z–10:23:00Z，见[log](D:/7_codespace/training_outputs/terminal/stage2_transfer_diagnosis_20261021/verification_v3.log)及
[meta](D:/7_codespace/training_outputs/terminal/stage2_transfer_diagnosis_20261021/verification_v3.meta.json)。
T09/T15单项执行确认仍待核对，不由该总数推造单项结果或最优性。
服务器[focused probe](D:/7_codespace/training_outputs/terminal/stage2_transfer_diagnosis_20261021/probe_complete_chain_focused_server.json)
已完成，`read_only=true`、`state_unchanged=true`；catalog耗时3.01019秒，完整MOVE为outer5、middle3、inner4，
单机最大10次尝试，仍触发原3秒CANDIDATE_WORK_CUTOFF。单机检查为12 SAFE、3拒绝：middle01/02为RESOURCE_CONFLICT，
inner02为HARD_SEPARATION。以上均不是已启动pilot或120秒连续行为结果。

拟用独立输出目录`stage2_transfer_fix_120s_v3_20261021`，仍仅运行120秒；继承先前累计 **21488.54562318558秒**。
当前针对性诊断/测试拟保守记300秒准备credit，另加实际prepare费用，不重置原预算。
初始T0约5分钟，依据上一轮相同120秒pilot的293.79秒wall；拟两worker、物理核0/2、数值库线程1。
上述目录、费用及执行配置均为待启动记录，不表示进程已正常启动。
