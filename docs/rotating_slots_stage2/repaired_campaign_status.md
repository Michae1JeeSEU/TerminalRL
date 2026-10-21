# Stage2 修正版成对运行：完整3600秒，新增审计通过

双方已在同一物化manifest下完成 **3600仿真秒**，持久化状态为 **COMPLETED_3600**。
运行UTC **2026-10-21T02:02:25Z–06:07:33Z**，退出0，运行段14708秒；heartbeat已由工具删除。
原始3600秒审计分别有12/5项PEAK_DELTA；离线审计修复测试25项通过，双方原回放的新增审计均 **PASS、0问题**。
原始FAIL及campaign的archive_status保留，新增报告不覆盖旧记录。
全部回放已导出，人工PyVista观察仍为 **HUMAN_PENDING**。

| 状态维度 | 当前记录 |
|---|---|
| implementation | 修正版有界实现已交付；离线审计修复测试25 passed |
| correctness | 运行前修复验证218 passed、1 skipped；离线审计修复测试25 passed；新增3600秒审计双方PASS、0问题，原FAIL保留 |
| experiment | **COMPLETED_3600**；不是预算截尾，也不拼接旧轮840秒 |
| real_time_observation | **MEASURED_NOT_CERTIFIED** |
| human_review | **HUMAN_PENDING** |

## 同窗口质量

以下读取`summary_3600.json`、`comparison.json`及决策记录；comparison实际存在且内容与双方summary一致。
双方`truncated=false`指完整仿真窗口，不代表候选目录完整。

| 3600秒指标 | Energy-slack | RH-CP-SAT |
|---|---:|---:|
| F；battery/peak | 44；44/0 | 50；50/0 |
| 实际接地L；normal/emergency | 104；10/94 | 101；13/88 |
| admitted / 空中pending / 上游pending | 184 / 36 / 1 | 185 / 34 / 0 |
| offered / 已释放新请求 / 分流 | 208 / 172 / 23 | 208 / 172 / 23 |
| 总终端能量J | 8238478687.834279 | 8306223526.819811 |
| emergency+reroute费用 | 353400 | 321000 |
| failure费用 | 44000000 | 50000000 |
| emergency费用 | 339600 | 307200 |
| reroute费用 | 13800 | 13800 |
| C_legacy | 44469372.970866226 | 50437991.2783183 |
| 累计空中时间秒 | 114413.10370852177 | 115305.68859427767 |
| 累计上游等待秒 | 1559.867157703016 | 1685.5897240254183 |
| 累计空中+上游时间秒 | 115972.97086622479 | 116991.27831830308 |
| 首次失败仿真秒 | 661.6249818546746 | 661.6249818546747 |
| 首次失败前/后接地 | 15 / 89 | 15 / 86 |
| 守恒 | true | true |

failure、emergency、reroute单项已从原始审计的`derived_costs`复核，不是仅由总额相减推算。

主键分别为
`(44,-104,353400,8238478687.834279,115972.97086622479)`和
`(50,-101,321000,8306223526.819811,116991.27831830308)`。
按合同先比较F，本单轮记录中Energy-slack的44次失败低于CP的50次，总接地104也高于101；
CP正常接地13高于10不能覆盖可靠性优先级，不能据此称CP整体胜出。新增审计的primary_key和守恒与上述原指标一致。

| cohort（admitted / success / failed / air / upstream / reroute） | Energy-slack | RH-CP-SAT |
|---|---|---|
| initial_airborne | 36 / 20 / 16 / 0 / 0 / 0 | 36 / 20 / 16 / 0 / 0 / 0 |
| new_requests | 148 / 84 / 28 / 36 / 1 / 23 | 149 / 81 / 34 / 34 / 0 / 23 |

初始cohort成功/失败/未完成计数相同；这些质量计数的差异来自新请求，不声称全部内部轨迹、时间或能耗相同。
初始三架地面服务不计新接地；pending不计失败。
本轮采用`post_failure_model=abstract_isolation_no_cascade`，有损失的吞吐不能描述为安全运营容量。
单seed固定繁忙初态的结果不推广为容量标定、统计显著性或普遍优越性。

## 正常与完整dispatch时间

下表均为wall秒、保留全部调用；正常callback包括其子阶段，完整dispatch还含公共应急/准入。

| 正常callback | Energy-slack | RH-CP-SAT |
|---|---:|---:|
| N / mean / P50 | 1525 / 3.579257 / 3.640883 | 1528 / 3.898538 / 3.987769 |
| P95 / P99 / max | 4.503760 / 4.858799 / 5.230875 | 5.018676 / 5.329692 / 5.758796 |
| >5秒 | 12/1525（0.7869%） | 82/1528（5.3665%） |
| CPU累计秒 | 5454.779438 | 5945.216105 |

| 完整dispatch | Energy-slack | RH-CP-SAT |
|---|---:|---:|
| N / mean / P50 | 1525 / 7.417651 / 6.672949 | 1528 / 8.015882 / 7.200841 |
| P95 / P99 / max | 14.309991 / 18.143511 / 19.777291 | 14.653985 / 18.208067 / 19.802141 |
| >5秒 | 1147/1525（75.2131%） | 1297/1528（84.8822%） |
| CPU累计秒 | 11305.696943 | 12222.393890 |

非空提交另按`submitted>0`筛选，并按相同时刻关联dispatch；不是把全部ACCEPTED都当作移动。
summary的`nonempty_normal_timing`按有可决策飞机筛选，本轮仍含全部1525/1528次，不等于下面的非空提交口径。

| 非空提交调用：N；mean/P50/P95/P99/max；>5秒 | Energy-slack | RH-CP-SAT |
|---|---|---|
| 正常wall | 22；3.958931/4.192373/4.869122/5.155659/5.230875；1/22（4.55%） | 33；4.693314/4.874189/5.571381/5.712169/5.720237；12/33（36.36%） |
| 完整dispatch wall | 22；6.347381/5.618990/10.953783/11.675526/11.848255；15/22（68.18%） | 33；7.950980/7.787544/12.370701/12.428573/12.428806；28/33（84.85%） |

上述非空提交调用的CPU累计：正常87.015133/154.703136秒，完整dispatch139.557840/262.033236秒（Slack/CP）。

## 候选、前缀与阶段成本

逐行汇总`decisions.jsonl`；同一调用可有多个截断标签，列数和cache_hits总计不是独特物理方案数量。

| 记录 | Energy-slack | RH-CP-SAT |
|---|---:|---:|
| 决策 / noop / fallback | 1525 / 1503 / 2 | 1528 / 1495 / 0 |
| 非空提交调用 / 前缀总数 | 22 / 23 | 33 / 35 |
| commit_status ACCEPTED（含空前缀） | 1525 | 1528 |
| 至少一机目录不完整的调用 | 1525/1525 | 1528/1528 |
| 列数sum/min/max/mean | 67379/10/105/44.182951 | 67840/10/109/44.397906 |
| trajectory_recomputes / cache_hits合计 | 1575473 / 205277 | 1562120 / 228627 |
| CANDIDATE_WORK_CUTOFF | 1300 | 1274 |
| AIRCRAFT_ATTEMPT_LIMIT | 1428 | 1431 |
| GLOBAL_ATTEMPT_LIMIT | 53 | 51 |
| SOFT_BUDGET / MOVING_COLUMN_LIMIT | 9 / 11 | 78 / 48 |
| stale / watchdog | 0 / 0 | 0 / 0 |
| 决策status | FEASIBLE1523；PREFIX_NOT_EXECUTABLE2 | CATALOG_LEX_OPTIMAL1235；LEX_LIMITED293 |
| raw CP阶段status | 不适用 | OPTIMAL8609 |
| CP候选认证 | 不适用 | SAFE8062；UNSAFE547 |
| CP prefix认证 | 不适用 | ACCEPTED8059；NOT_CHECKED547；REJECTED3 |

有限目录每次均有不完整飞机，不能把未找到或截断称为全局不可行。
8609条OPTIMAL阶段状态及CATALOG_LEX_OPTIMAL不等于连续物理空间或全部未来需求的全局最优；
阶段认证包含被拒暂定组合，不能用solver状态代替前缀检查。
没有候选级accepted/rejected总字段，不能以尝试数减保留列数推造拒绝数量。

| 已记录阶段：wall累计 / CPU累计秒 | Energy-slack | RH-CP-SAT |
|---|---:|---:|
| 公共应急 | 5102.655108 / 5100.372668 | 5497.353932 / 5484.816727 |
| 公共准入 | 750.886037 / 750.538949 | 793.937516 / 792.355265 |
| 正常callback整体 | 5458.366920 / 5454.779438 | 5956.965562 / 5945.216105 |
| snapshot/refinement | 12.661520 / 12.661833 | 14.027807 / 14.027034 |
| candidate generation | 4713.827580 / 4710.750039 | 4654.956578 / 4645.902463 |
| selection | 668.409671 / 667.986367 | 2.008640 / 1.997160 |
| hint construction | 不适用 | 672.588466 / 671.469823 |
| model build | 不适用 | 219.742462 / 219.275871 |
| resource conflict build | 不适用 | 0.644902 / 0.644735 |
| cp_solve | 不适用 | 3.469170 / 3.451521 |
| selected validation | 16.595389 / 16.594044 | 349.762427 / 348.819242 |
| commit | 44.969434 / 44.887558 | 37.570334 / 37.437728 |
| fallback | 0.013498 / 0.013500 | 不适用 |

这些层级非独立可加：正常callback已含其子阶段，不能再与candidate等简单相加。
公共应急字段合并monitor/rescue，未分别记录。当前主要耗时记录在候选生成和公共应急；
CP的cp_solve仅累计3.469秒，不能将其完整正常耗时5956.966秒归因于CP求解器慢。
统计为逻辑时钟实验的实际计算成本，不是延迟已注入运动的控制安全验证。

全窗口实际曲线为
[comparison_curves.png](D:/7_codespace/training_outputs/terminal/stage2_repaired_campaign_v1_20261021/comparison_curves.png)。
主对话已目视检查；完整回放均已导出，但图中normal planning不能替代完整dispatch上表，PyVista仍HUMAN_PENDING。

## 原始审计、取证及校正状态

原始`audit_3600.json`保留：Energy-slack **FAIL12项**、CP **FAIL5项**，全部为PEAK_DELTA；
其他连续间隔为CERTIFIED，summary守恒均true；原FAIL保留。
主对话已核对17项只读取证：按历史实际轨迹重算fine累计，全部等于newpeak，误差分解残差0；
其中6项涉及历史精化、11项为当前区间fine/coarse差异，3项为负修正。
证据为`artifacts/rotating_slots_stage2_20261021/peak_delta_evidence.json`，只读取证wall为21.316秒。
主对话确认原当步差分审计未计既有累计误差预算；此诊断不改run代码、轨迹或账本，不需要重跑仿真。

离线audit修复检查既有整计划`peak_error_s`累计包络，保留1e-5，逐帧检查累计量并检查ground冻结；
不声称重建了precision-switch事件。新增`test_rotating_slots_audit_peak_precision.py`及既有audit、Stage2 audit测试
共 **25 passed**，pytest16.51秒、wall17.2622878秒，UTC07:15:16.703396–07:15:33.965684。
含首次红阶段和fixture修正在内的五轮测试wall共46.1162688秒。
主对话已复核实际diff；离线`audit.py` SHA256为
`ecff4f5f251ef950b641feab2e1f151ad06e9fb53141a1473acabc9c0a8eb833`。
离线审计修复验证提交为`4d0603472651a87b781264276f0f6fbb7edc1ea0`，已提交并推送；与本轮运行源码分列。
**修复测试=25 passed，新增再审计=双方PASS、issue_count=0**。运行源码仍为下述2b045be，仿真、energy、world及planner未改。

| 新增3600秒审计 | Energy-slack | RH-CP-SAT |
|---|---:|---:|
| continuous_separation / physical_outcome | CERTIFIED / ENERGY_LOSSES_CONTINUED | CERTIFIED / ENERGY_LOSSES_CONTINUED |
| state_frames / energy_intervals / continuous_pair_runs | 4480 / 142468 / 15069 | 4480 / 143679 / 15456 |
| peak envelope检查数 / 最大宽度秒 | 588037 / 0.01800537109375 | 588568 / 0.0185394287109375 |
| 原PEAK_DELTA mismatch记录数 | 12 | 5 |

报告为[Slack新增审计](D:/7_codespace/training_outputs/terminal/stage2_repaired_campaign_v1_20261021/energy_slack/audit_3600_peak_precision.json)、
[CP新增审计](D:/7_codespace/training_outputs/terminal/stage2_repaired_campaign_v1_20261021/rh_cp_sat/audit_3600_peak_precision.json)；
原[Slack审计](D:/7_codespace/training_outputs/terminal/stage2_repaired_campaign_v1_20261021/energy_slack/audit_3600.json)和
[CP审计](D:/7_codespace/training_outputs/terminal/stage2_repaired_campaign_v1_20261021/rh_cp_sat/audit_3600.json)仍保留FAIL。
再审计UTC07:19:11.665786–07:20:08.186603，并行wall56.520806330秒；两进程退出0、核心0/2、数值库线程1，代码SHA匹配4d06034。
原audit与replay.json前后hash相同，`inputs_unchanged=true`；没有重新推进仿真。
独立账本检查共用冻结数值内核，不是对现实能源模型的独立经验oracle。

## 来源与预算终态

运行源码提交`2b045bef2b5f6b2673669899fe58a5f068d58a94`，source hash
`01749a6cf72239e9f557d004fcf297d39a21fa22346d26f9588b98d9b7653c6c`，
指纹文件`artifacts/rotating_slots_stage2_20261021/repaired_source_provenance.json`。
输出根目录为`/home/user/training_outputs/terminal/stage2_repaired_campaign_v1_20261021`，
本地为`D:\7_codespace\training_outputs\terminal\stage2_repaired_campaign_v1_20261021`。
旧轮[INVALID_STOP840记录](first_campaign_status.md)保持独立，不拼接为本轮结果。

新旧manifest、fixture、environment.lock逐字节一致，场景/SOC/种子/算法预算未改；manifest仍172条，SHA256
`10b97225bde7992af59c79857b0c387bde69ec3b1153e36867a9d42406ad4bc6`。
完整保留旧已耗4812.677915491629秒carry，旧准备不重复计费；恢复验证以已知140.347秒保守记300秒，
新轮自身准备300.55827703222167秒，旧新准备合计901.1101518273063秒，小于一小时准备上限。

| 累计预算 | 秒数/口径 |
|---|---|
| 已结束campaign累计（含旧carry） | 19969.843647099653 |
| 本轮后处理已知wall | 123.9528251；取证21.31575 + 五轮测试46.1162688 + 并行再审56.5208063，不重复加两个child wall |
| 本轮后处理保守计账 | 600；含上述工作及短只读预检，不是实测600秒 |
| 最终保守总计账 | 20569.8436471（约5.714小时） |
| 原计算截止 / 总上限 | 34200 / 36000，不变 |

原campaign/budget文件保留终态，不重置；结束后的人工阅读/编码等待不当作自动运行。
600秒为原10小时内后处理额度，不是新实验预算。

## 首60秒及启动历史

UTC02:04:18的启动核验已确认wrapper99045、runner99053、tracker99056、workers99057/99058，
物理核心0/2、四库线程1，首个共同60秒文件存在。双方当时F0/L0、空中41、上游0、20决策/19noop/0fallback，
正常P95为4.5017359/5.0432798秒，完整dispatch P95为5.0589809/5.7453142秒。
两份60秒审计PASS0项、CERTIFIED、IN_PROGRESS仅覆盖历史60秒，不替代原始3600秒审计。
初始T0约5小时只是旧840秒用4033 wall秒的粗估；运行已结束，不保留ETA或ACTIVE heartbeat承诺。
