# 安全进展与服务情景修订：单次 360 秒验证

记录日期：2026-10-22。当前状态：用户请求的单次 360 秒验证已完成，退出码 0。两方法均正常落地 20 架、应急落地 0、能量失败 0，仍有 41 架空中待完成，各有 2 架已声明应急且尚未落地。独立审计均 PASS，回放加载与 seek 检查通过；人工验收仍为 `HUMAN_PENDING`。

## 本轮授权与边界

用户要求修复长期正常着陆偏低、优化逻辑及服务容量不足，并在修复后执行单次 360 秒验证。旧 1620 秒结果未通过用户的性能验收，旧证据继续保留。本轮不恢复或清零旧预算，也不追加种子、重试、搜索或消融矩阵。

新输出根目录为 `/home/user/training_outputs/terminal/stage2_progress_service_360s_v4_20261022`，通过既有 Syncthing 链路同步到 `D:/7_codespace/training_outputs/terminal/stage2_progress_service_360s_v4_20261022`。单轮真实总预算为 3600 秒，其中归档保留 300 秒，计算截止为累计 3300 秒；本轮测试、诊断及准备的实耗计入该预算。旧累计 34312.8967 秒预算记录独立保留，不与新预算拼接或改写。预算只在新 root 首次 prepare 时初始化，run/resume 不重新授时。

## 显式服务情景

保持原规范文件与 `load_config()` 默认基线。新命名情景 `service_half_v1` 只修改完整规范 payload 中的三个服务时长：

| 服务类别 | 比例 | 原情景时长（秒） | 新情景时长（秒） |
|---|---:|---:|---:|
| short | 0.2 | 60 | 30 |
| standard | 0.6 | 120 | 60 |
| long | 0.2 | 240 | 120 |

原服务均值为 132 秒；计入每架 13.5 秒垂直下降占用后，6 pad 的理想吞吐为 `6×3600/(132+13.5)=148.45` 架/小时，低于名义来流 180 架/小时。新服务均值为 66 秒，对应理想吞吐为 `6×3600/(66+13.5)=271.70` 架/小时。这是资源占用的理想计算，未包含安全冲突、进近调度、初始占用及有限窗口影响，不能用作实际吞吐结论。

服务减半是用户授权的仿真情景修订，尚未验证为现实地面周转时间。新情景保持几何、SOC、arrival seed 与随机流、服务类别概率、energy/safety 参数不变；初始地面剩余服务时间由 30/60/90 秒同比缩为 15/30/45 秒。前 360 秒冻结清单实际新增 25 架，折合该短窗口 250 架/小时，另有 36 架初始空中积压，因此仍不能承诺窗口内全部落地。

新完整配置为 `new10.16/fixed_terminal_rotating_slots_stage2_service_half_v1.yaml`，语义 hash 为 `5582b671279c0162dd3f6f72da12368a5b2ecffd12d97f5d59eaf945a47ecf87`。campaign 同时冻结配置字节、情景名及配置 hash，运行、checkpoint 和回放使用同一情景。

## 已有快照诊断证据

对旧 `energy_slack/checkpoint_0240.json` 的只读快照诊断发现，`initial_outer_05`、`initial_outer_06` 的候选目录均无 MOVE，但已有分别可接受的 O→M 前缀：240–390 秒到 middle slot 6，以及 240–280 秒到 middle slot 12。接受前缀不等于已找到完整落地 reference。

`artifacts/rotating_slots_stage2_20261022/diagnosis270/objective_0240_two_witnesses.json` 记录的旧目标为：

| 飞机 | KEEP key | 安全 O→M witness key |
|---|---|---|
| initial_outer_05 | `(0,0,3,901,16338722.192824317)` | `(0,0,3,901,16874104.34934795)` |
| initial_outer_06 | `(0,0,3,901,16338722.192824317)` | `(0,0,3,901,17579361.31654674)` |

KEEP 与两个 witness 的 reference 均为 UNKNOWN。前四个分量相同，进展前缀因真实耗能更大而劣于 KEEP；主对话据此确认该比较中的目标退化，不能解释为安全内核拒绝这两个 witness。该文件标记 `state_unchanged=true`；诊断使用当前工作树读取旧状态，未作原旧代码的精确恢复或运行接续。

后续 300 秒只读探针也保留了候选外的安全前缀证据：`energy_slack_0300_probe.json` 对 outer_02→M3、outer_06→M12 找到 300–340 秒前缀，但整个探针达到墙钟上限并标记 partial，outer_08 搜索未完成；不能把该截断解释为不可行。`rh_cp_sat_0300_probe.json` 标记 completed，对 outer_02→M3、outer_03→M5 找到 300–340 秒前缀，对 outer_05→M8 找到 300–375 秒前缀。两文件均标记状态及输入 checkpoint 未变；这些诊断 witness 不是算法当时实际持有的候选或已执行动作。

## 本轮优化条款修订

本轮依据用户修正优化逻辑的授权，对原合同 §9.2 的 UNKNOWN 尾项作显式修订，原合同文件本身不修改。新的候选策略标识为 `shared_multistep_columns_v4_safe_progress`，目标标识为 `reliability_throughput_progress_tail_v2`。

修订保持 F、-L、U、已知 reference 的 tau、真实能耗及应急语义。UNKNOWN 仍保持 UNKNOWN 和 U=3；其 tau 改为 `901 + _remaining_lower(末端环)`，复用既有正常剩余时间下界作为启发式，且仍大于 900、劣于 known。该值不是已知 reference，也不是预测或保证的到达时间。

已实现的即时 progress 阶段使用候选墙钟预算及尝试次数预算的前半部分，有界轮转分配给飞机，优先筛查不依赖当前资源释放的目标。该优先顺序不排除后续多步候选的联合释放；完整链候选仍保留。partial 候选不再隐式生成 H 末端 reference；H 末端仅在精确缓存命中时记为 KNOWN，否则保持 UNKNOWN。主对话已检查这些实现改动。

这轮同时修改候选/目标实现与服务情景，任何总性能变化均不能单独归因于算法，不另扩大为消融矩阵。下列旧 checkpoint 的同状态候选前后对比独立检查实现修复，仍使用旧基线状态，未将半服务情景混入该比较。

## 修后同状态输出与当前验证范围

三个输出均位于 `artifacts/rotating_slots_stage2_20261022/diagnosis270/`。表中动作是实际选中且送入原 `evaluate` 的即时 prefix，不是把诊断 witness 直接加入算法；三项 `selected.evaluation.status` 均为 ACCEPTED。

| 输出文件 | 来源快照 | 本地实际选择器 | 选中 O→M / M→I | normal 墙钟（秒） | 原 evaluate | 状态 / 输入 checkpoint 未变 |
|---|---|---|---:|---:|---|---|
| energy_slack_0240_after_safe_progress.json | EnergySlack，240 秒 | EnergySlack | 2 / 1 | 6.1620 | ACCEPTED | true / true |
| energy_slack_0300_after_safe_progress.json | EnergySlack，300 秒 | EnergySlack | 2 / 2 | 6.3484 | ACCEPTED | true / true |
| rh_cp_sat_0300_after_safe_progress.json | RH-CP-SAT，300 秒 | EnergySlack hint 选择路径 | 3 / 2 | 6.9058 | ACCEPTED | true / true |

第三行读取 RH-CP-SAT 方法产生的旧状态，但本地实际调用 EnergySlack 选择器，并非本地 CP-SAT 求解。三个文件的 scope 都明确记录 `no local CP solver execution`；本地 venv 缺少 `ortools`。服务器实际 CP 测试的独立结果见下文，不回填为这三个本地快照输出的求解证据。

三次 normal 墙钟均超过 5 秒软预算，当前尚未达到该耗时目标。输出还存在 `incomplete` 候选覆盖记录；这些结果只表明该次有界目录与选择路径产生了可接受的进展动作，不能声称覆盖全部候选、全部飞机均可进展或已经证明全局最优。

本地测试分开记录：规划等新增/相关测试 80 passed、2 skipped，最后 safe-progress 针对性批次 12 passed；服务情景及既有 physics/replay/performance checkpoint 兼容批次为 84 passed、14.07 秒。服务批次验证配置、资源时长和序列化兼容性，不能替代实际 CP 求解或正式 360 秒运行验收。

服务器验证目录为 `/home/user/training_outputs/terminal/stage2_progress_service_v4_validation_20261022`。主对话确认服务器批次为 119 passed、1 skipped、2 deselected，墙钟 33.674934149 秒，实际 T09/T15 CP 测试通过。JUnit 中跳过的是 `test_cp_dependency_missing_is_explicit`，message 为 `OR-Tools available`：服务器已安装 OR-Tools，因此缺失依赖分支测试跳过；两个依赖本地 provenance/Git 的测试在服务器特意排除，已在本地通过。此处记录测试结果，不等于正式 360 秒实验完成。

## 冻结版本与正式运行初验

修复提交 `1e205b92156f0c8fd72e1ccd6d58a6e847db3fbe` 已成功推送。冻结 source hash 为 `5405da7efd082467aa180cb8d3c370bbea83c0388ae06e813f232fdc38d9cc01`。主对话已核对服务 profile、manifest、source 和 config 与冻结身份匹配；运行期间不修改冻结源码。

新 root 的 prepare 仅执行一次，退出码 0，墙钟约 2.852 秒。初始预算累计下界为 702.4992141723633 秒，包含保守准备 credit 700 秒：规划诊断及测试 450 秒、服务测试 85.81 秒、服务器测试约 33.675 秒，其余覆盖短检查及包装开销的保守余量。预算为总计 3600 秒、计算截止 3300 秒、归档保留 300 秒；原旧预算保持独立。

工作目录为 `/home/user/projects/terminal`，实际启动命令为：

```bash
/home/user/envs/terminal_stage2_v1/bin/python -u /home/user/projects/terminal/scripts/server/run_rotating_slots_stage2.py run --output /home/user/training_outputs/terminal/stage2_progress_service_360s_v4_20261022 --target 360
```

服务器启动时间为 `2026-10-21T17:35:30.392775+00:00`。当时 wrapper PID 122325、runner PID 122336；两个 worker PID 为 122339（CPU 0）和 122340（CPU 2），绑定不同物理核，线程数及数值库线程环境均为 1。运行日志为 root 下 `campaign.log`，计时为 `campaign.timing.txt`；终态退出码见下文。

UTC 17:37:04 的启动初验观察到 runner 已运行约 93.6 秒，两方法最新状态时间均为 23.466796875 秒，两个 `states.jsonl` 与 checkpoint 均非空，未见 failed/invalid/error。当时共同 60 秒边界尚未到达，状态为 `RUNNING`；这是历史启动检查，不是最终结果。

初始预计墙钟 T0 取 1800 秒，依据旧 360 秒实际约 1082 秒及新增调度开销作保守估计，并非完成保证。该初始估计和上述启动初验保留，后续观察单独记录如下。

## 历史共同 60 秒边界与调度记录

该次本地读取窗口为 UTC `2026-10-21T17:38:45.7611370Z` 至 `2026-10-21T17:39:11.2291497Z`；服务器观察时间为同日 UTC `17:39:47Z`，runner 累计运行约 257 秒。两组时间分别保留，不用本地读取时间差替代服务器进程经过时间。

当时两方法均已发布 `summary_0060.json` 和对应 checkpoint；checkpoint payload 时间为 60 秒，且存在 boundary summary。states 时间分别为 EnergySlack 60.126953125 秒、RH-CP-SAT 60.60028076171875 秒。当时共同完成边界为 60 秒，尚未完成目标 360 秒；下表仅为该次历史快照。

| 方法 | 正常落地 | 应急落地 | 已声明应急 | 能量失败 | 空中待完成 | 上游等待 |
|---|---:|---:|---:|---:|---:|---:|
| EnergySlack | 3 | 0 | 0 | 0 | 38 | 0 |
| RH-CP-SAT | 3 | 0 | 0 | 0 | 38 | 0 |

该次检查未见 invalid 或 truncated 证据，当时真实终态 `exit_code.txt` 尚未生成，运行继续。持久预算下界为 1144.7631485462189 秒，包含 prepare 后经过的墙钟；该累计预算与 runner 的约 257 秒经过时间口径不同，不能用后者覆盖前者。

以共同进度 `p=60/360` 和服务器 runner 经过时间约 257 秒粗估，总耗时 `T≈257/p=1542` 秒，剩余 `R≈1285` 秒。启动开销与后续调度密度会变化，该估计不是完成保证，亦不替代总预算约束。

主对话当时通过工具创建当前 thread 的 heartbeat `stage2-360`，状态为 ACTIVE，约 20 分钟后复查，设置时扣除观察后约 1 分钟；等待期间未保留睡眠监控 worker。任务完成后，主对话已通过工具成功删除该 heartbeat。

## 360 秒终态与结果

运行结束于 UTC `2026-10-21T18:02:09.415931+00:00`，runner 墙钟 1599.0231325626373 秒，`exit_code.txt=0`。共同完成边界、checkpoint payload 及最终 state 均为 360 秒，未见 invalid/truncated。原 campaign 的 `PARTIAL_REQUESTED_WINDOW` 与 `target_s=3600` 是 runner 对非 3600 秒窗口的标签，不能解释为用户本次 360 秒任务失败；原始文件保持不变。持久预算下界为 2502.5826683044434 / 3600 秒，计算截止 3300 秒，未以 runner 耗时改写累计预算。

同窗结果如下，初始地面 3 架不计入本轮 offered/成功。新旧每方法均对应 36 架初始空中和 25 架新增请求。

| 360 秒情景 / 方法 | 正常落地 | 应急落地 | 能量失败 | 空中待完成 | 上游待完成 | 初始 / 新增成功 |
|---|---:|---:|---:|---:|---:|---:|
| 旧 v3 / Slack | 9 | 1 | 0 | 49 | 2 | 10 / 0 |
| 旧 v3 / CP | 10 | 1 | 0 | 50 | 0 | 11 / 0 |
| 新 v4 / Slack | 20 | 0 | 0 | 41 | 0 | 20 / 0 |
| 新 v4 / CP | 20 | 0 | 0 | 41 | 0 | 19 / 1 |

新两方法均满足 `61=20+0+0+41+0`，即 offered 中 32.79% 已正常落地。已落地者 100% 正常不等于全部飞机完成；双方仍有 2 架已声明应急且未落地，不能写成无应急压力。`20/(360/3600)=200` 架/小时只是短窗口观测，不是稳态容量。旧同窗能量失败也为 0，本轮不能据此宣称长期能源失败已解决；41 架未完成及应急尾项仍未结束。

| 情景 / 方法 | 总成本 C_legacy | 应急成本 | 能量（kWh） | O→M 开始 / 完成 | M→I 开始 / 完成 | 完成切环事件总数 |
|---|---:|---:|---:|---:|---:|---:|
| 旧 v3 / Slack | 18100.58 | 2400 | 327.71 | — | — | 15 |
| 旧 v3 / CP | 19032.31 | 3600 | 323.33 | — | — | 18 |
| 新 v4 / Slack | 14454.67 | 0 | 326.86 | 35 / 33 | 30 / 29 | 62 |
| 新 v4 / CP | 14247.85 | 0 | 321.49 | 35 / 33 | 29 / 28 | 61 |

切环开始与完成来自实际 state 转换，完成合计与 events 的 62/61 一致。下表为紧凑状态及 pad 占用；hold 顺序为 O/M/I，切环为 O→M/M→I，当前 pad 包含正在服务或垂降的有效占用，仅未来预约不计入当前。

| 时刻 / 方法 | HOLD O/M/I | 切环 O→M/M→I | 已正常落地 | 当前 pad（服务+垂降） | 仅未来预约 pad |
|---|---:|---:|---:|---:|---:|
| 240 / Slack | 2/14/15 | 3/2 | 12 | 4（3+1） | 2 |
| 240 / CP | 2/12/13 | 3/5 | 12 | 3（2+1） | 3 |
| 270 / Slack | 2/16/16 | 1/1 | 14 | 5（4+1） | 1 |
| 270 / CP | 2/12/16 | 3/2 | 16 | 5（5+0） | 1 |
| 300 / Slack | 3/13/13 | 4/4 | 15 | 5（4+1） | 1 |
| 300 / CP | 3/11/14 | 5/3 | 17 | 5（5+0） | 1 |
| 360 / Slack | 2/15/17 | 2/1 | 20 | 6（5+1） | 0 |
| 360 / CP | 2/16/15 | 2/1 | 20 | 4（3+1） | 2 |

270 秒两方法 outer HOLD 均仅 2 架，对比旧状态约 27 架，积压已向 middle/inner 移动，不能宣称所有瓶颈消除。全窗 pad 有效占用利用率分别为 73.55% / 74.13%，分母为 `6×360`；未来预约不视为正在服务。主对话已打开检查 [comparison_curves.png](../../../training_outputs/terminal/stage2_progress_service_360s_v4_20261022/comparison_curves.png)，图中 “Pads occupied” 仅统计 LANDED_SERVICE，不包含垂降或未来预约，与上表当前占用口径不同。

| 情景 / 方法 | normal mean / p95（秒） | normal >5 秒 | 完整 dispatch mean / p95（秒） | dispatch >5 秒 | no-op / fallback |
|---|---:|---:|---:|---:|---:|
| 旧 v3 / Slack | 4.2288 / 4.7153 | 2/172 | 5.1667 / 7.3568 | 70/172 | 152 / 0 |
| 旧 v3 / CP | 4.9919 / 5.6723 | 104/174 | 5.7005 / 7.4440 | 138/174 | 152 / 0 |
| 新 v4 / Slack | 5.0912 / 5.6498 | 203/265 | 5.3722 / 6.3127 | 236/265 | 210 / 0 |
| 新 v4 / CP | 5.1701 / 5.5778 | 258/266 | 5.3611 / 6.3262 | 260/266 | 207 / 0 |

新 normal p95 约 5.65/5.58 秒，仍不满足 5 秒软预算。CP 的 266 次决策均为 LEX_LIMITED，其中 242 次 solver list 为空，仅 24 次决策包含共 122 次 OPTIMAL phase solve；phase 最优不代表完整滚动方案全局最优。81 SAFE / 41 UNSAFE 是中间候选校验，最终 266 次 commit 均 ACCEPTED。fallback=0 不表示每轮执行了 CP。

主对话读取实现确认：先构建 EnergySlack hint，再处理 resource conflict，随后进入 CP；`search_available` 受共享软时间限制。候选和 hint 开销压缩 CP 阶段，可解释两方法结果接近。所有调用都记录 CANDIDATE_WORK_CUTOFF（Slack 265、CP 266 次），SOFT_BUDGET 分别为 Slack 181、CP 254 次，候选覆盖并不完整。

独立审计两份均 PASS、`issues=[]`、continuous separation CERTIFIED。实际 ReplayController 加载两份回放并 seek 0/240/270/300/360，10 次均到达准确 controller/render 时间，实体状态、坐标及 SOC 可用，检查退出 0、墙钟 21.29185 秒；index 字节及 HUMAN_PENDING 均未改变。旧 120 秒人工确认不代替本轮人工验收。

完整证据见 [事实 JSON](../../artifacts/rotating_slots_stage2_20261022/v4_final_0360_facts.json)、[回放加载检查](../../artifacts/rotating_slots_stage2_20261022/v4_replay_load_check.json) 与 [结果根目录](../../../training_outputs/terminal/stage2_progress_service_360s_v4_20261022/)。算法与服务情景同时变化，总改善不能单独归因算法；30/60/120 秒仍只是授权仿真假设。

## 完成后本地回放

两份 `replay_0360` 已完成、同步并通过实际 controller 加载检查。先切换目录，再任选一条 `-c` 命令独立打开相应回放；人工视觉验收仍待用户完成。

```powershell
Set-Location "D:/7_codespace/terminal"
.\venv\Scripts\python.exe -c "from terminal_marl.visualization.rotating_slots_replay import ReplayController,RotatingSlotsReplayViewer; RotatingSlotsReplayViewer(ReplayController(r'D:\7_codespace\training_outputs\terminal\stage2_progress_service_360s_v4_20261022\energy_slack\replay_0360')).run()"
.\venv\Scripts\python.exe -c "from terminal_marl.visualization.rotating_slots_replay import ReplayController,RotatingSlotsReplayViewer; RotatingSlotsReplayViewer(ReplayController(r'D:\7_codespace\training_outputs\terminal\stage2_progress_service_360s_v4_20261022\rh_cp_sat\replay_0360')).run()"
```

## Git 交付与人工验收

代码修复提交 `1e205b92156f0c8fd72e1ccd6d58a6e847db3fbe` 已推送。本轮最终文档与两份事实 JSON 一并纳入结果交付，具体提交号以 Git 历史和交付消息为准。运行、审计和回放加载检查已闭环；人工验收：`HUMAN_PENDING`。
