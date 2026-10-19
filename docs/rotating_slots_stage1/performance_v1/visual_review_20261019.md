# 既有 PyVista 人工作业单（2026-10-19）

人工状态：**HUMAN_PENDING**。本文准备材料与操作，不记录研究者已经观看，也不签署 `VISUAL_ACCEPTED`。只有主对话决定最终签收。性能数字按[用户范围修订](D:/7_codespace/terminal/docs/rotating_slots_stage1/performance_v1/user_scope_amendment_20261019.md)单轮报告，内核定稿不再为数值门槛调优。

依据[合同 §17–18、H02–H05](D:/7_codespace/terminal/rotating_slots_performance_contract_v1_0/rotating_slots_simulation_performance_contract_v1_0.md:546)，使用现有 `RotatingSlotsReplayViewer` / `Terminal3DViewer` 和[真实入口](D:/7_codespace/terminal/terminal_marl/scripts/open_stage1_replay.py:6)。截图、自动比较和原94项的 G04/G05 结果不替代交互观察，也不是新增性能 catalogue 同名条目的签收。

## 已有成对材料与定位

R0 根：[final_pass_period_reference_local_20261019a](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_period_reference_local_20261019a)。OPT 根：[final_pass_regression_local_20261019a](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_regression_local_20261019a)。两侧均有 G04、G05 的 `replay_00/01/02`：

- G04：`replay_00/01/02` 分别为 physics dt `.2/.1/.05`。真实脚本为 inner 槽0飞机 `a` 的 normal final 到 pad0，另有 pad3 的 `other_ground`（初始服务剩余300s）。事件为 t=0 commit/start、t=29.21084213256836 source clear、t=143.5 touchdown；归档至约203.5s。这不是上游等待、外环应急或“同一未来pad服务释放前起飞”的场景。
- G05：`replay_00/01/02` 分别为 scheduler period `1/5/10s`，同一请求 `r` 在 t=0 release/admit，t=35 inbound complete，t=75、115 transfer complete，t=183.5 touchdown，t=243.5 service complete；归档至约300s。两侧 `replay_00/rejections.jsonl` 各有30条拒绝，首条 t=0 为 generation search exhausted / `NOT_FOUND`，包含各约束 reason counts；不能解释成全局不可行证明。

参数对应使用[a/input_settings_mapping.json](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_period_compare_local_20261019a/input_settings_mapping.json)。a 的[比较摘要](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_period_compare_local_20261019a/comparison_summary.json)记录12次bundle校验和42文件相同，但[a numeric exit](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_period_compare_local_20261019a/numeric_exit.txt)为1；主对话已确认是包装器退出逻辑错误，保留该失败，不称a进程成功。[b/review_summary.json](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_period_compare_local_20261019b/review_summary.json)的只读复核为 success=true、12校验、42文件相同、3组PNG bytes及pixels相同，[b completion](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_period_compare_local_20261019b/completion.json)及[numeric exit](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_period_compare_local_20261019b/numeric_exit.txt)为0。b 的 `input_settings_mapping` 具有 null observations_value，不用它证明参数对应。

已有 t=30、G04 dt=.2 三视图截图位于 [R0 截图目录](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_visual_review_local_20261019a/R0_G04_dt02_t30)与[OPT 截图目录](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_visual_review_local_20261019a/OPT_G04_dt02_t30)，文件名均为 `top_down.png`、`side.png`、`3d.png`。[像素比较记录](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_visual_review_local_20261019a/OPT_G04_dt02_t30/pixel_comparison.json)提供每图bytes/pixel hash，图像尺寸1280×860；这些事实仍不代表人工已观察。

身份材料从各 bundle 的 `replay.json`、`config.yaml`、`trajectories.json` 和对应 case 的 `environment.json`读取。已读 G04/replay_00：R0 code_commit=`1c5afd37c891a0ed4776f64367fc7264db3c1c54`，OPT code_commit=`afb61940886add31e472738bf9ba654199c9fa1c`；共同 config_hash=`0c7cce0c0c019d50f7bba5aecd5e8340925d0d16867f8f06eb51e89102343d53`，config bytes SHA256=`2e49138901459a13d815bfe10d141457344026d35f82683b1c8170e9d726540e`。归档声明的身份与实际观察时使用的viewer源码身份须分别填写，不能只写当前HEAD。

## 六组场景材料（独立于原V结果）

[materials summary](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_reference_visual_20261019a/materials_20261019a/summary.json)记录 `R0_SCENARIO_MATERIAL`、success=true、numeric0，source/reference/original_V_failures_unchanged均为true。[外层completion](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_reference_visual_20261019a/launch_20261019a.completion.json)记录 UTC07:32:03.3051019–07:33:51.3698495、108.0647476s、exit0；这是材料作业耗时，不是模拟速度。四组V01/V02/V04/V06为fresh R0子进程且child exit0；V07只读复用R0 F01的三个回放，V08按F09再F08复用各一个回放，未改原身份/路径。共9对、18次validate、63个config加六physical成员bytes比较，无差异；全部seek round-trip和只读检查一致。这是共同既有viewer的材料证据，不把原V01–V11改成PASS，也不代表R0原92整组成功。

固定24文件R0导出没有visualization，原V01–V11在UI导入时失败，早于场景执行；失败路径最后生成的fallback `analytic_replay`不是这些视觉场景材料，原失败保留。本轮四组直接调用R0原`execute_scenario`，后两组复用已真实执行的R0物理归档，均不改R0或注入当前kernel。

场景与实际路径列表以各comparison的`R0_paths/OPT_paths`为准：[V01](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_reference_visual_20261019a/materials_20261019a/V01/comparison.json)、[V02](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_reference_visual_20261019a/materials_20261019a/V02/comparison.json)、[V04](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_reference_visual_20261019a/materials_20261019a/V04/comparison.json)、[V06](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_reference_visual_20261019a/materials_20261019a/V06/comparison.json)、[V07](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_reference_visual_20261019a/materials_20261019a/V07/comparison.json)、[V08](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_reference_visual_20261019a/materials_20261019a/V08/comparison.json)。V02保留`three_laps_replay`（三圈解析稳定fixture），V06为真实六pad着陆/服务材料；V01保留`threshold_replay`（诊断拒绝fixture），不能当作执行不安全飞行。V03/V11映V02，V05/V09映V01，V10映V04；别名不增加运行数或验收项。

## 真实 checkpoint 冷恢复接缝材料

固定 paired `W71-A/group-1/repeat-1` 的 [R0 bundle](D:/7_codespace/training_outputs/terminal/rotating_slots_perf_v1/final_pass_paired_20261019a/windows/W71-A/group-1/repeat-1/R0/attempt/replay) / [OPT bundle](D:/7_codespace/training_outputs/terminal/rotating_slots_perf_v1/final_pass_paired_20261019a/windows/W71-A/group-1/repeat-1/OPT/attempt/replay) 均为真实 checkpoint 冷恢复后的完整本地归档，覆盖 0–310s、各 1613 帧；cut 为 300s，最后小于 cut / 正好 cut / 首个大于 cut 的真实帧分别是 `299.957955557834 / 300.0 / 300.2`，不是假定等距邻帧。源/运行 cursor 均为 original_target=3600、phase=loop_start、outer_phase=arrival_window；migration_id 分别为 `final-pass-20261019a-4623e38d9b13-r0-36562efb3e1f` 和 `final-pass-20261019a-4623e38d9b13-opt-36562efb3e1f`。来源为各 attempt 的 input_identity/result/cold_cache，Linux checkpoint/stream 路径按原证据保留，未映射或恢复到 Windows。

[材料摘要](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_checkpoint_visual_20261019a/summary.json) 记录两次 bundle validate/load、三个时刻的 R0/OPT snapshot 相同及各时刻 seek 返回一致；[七成员比较](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_checkpoint_visual_20261019a/physical_member_comparison.json) 的 config 加六物理成员 bytes 相同。[R0 t300 三视图](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_checkpoint_visual_20261019a/R0_t300) / [OPT t300 三视图](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_checkpoint_visual_20261019a/OPT_t300) 共六张非空 1280×860 PNG；三组 bytes/pixels 均相同。现有 viewer 的两个 fresh `-I -B` stdin 进程使用 `--time 300 --off-screen --screenshots`，均实际退出 0，stdout/stderr/argv/UTC 和输入及 viewer 源前后 hash 保留在材料目录；44 个被读文件未变。以上是只读材料生成，不是人工验收或速度测量。实测 t300 静态3D图中状态文字与部分模型重叠；人工仍需在既有viewer交互检查，未据截图判定可读性通过。

## 两个可直接执行的本地命令

在 PowerShell 中分别执行；下列交互命令会打开可见窗口，尚未运行。接缝材料已另用同一入口离屏截图，未启动可见GUI。使用项目现有 venv/PyVista；入口依赖说明见[requirements-visualization.txt](D:/7_codespace/terminal/requirements-visualization.txt)。脚本经stdin交给 `-I -B`，在任何项目导入前设置 `dont_write_bytecode` 和显式 `sys.path`，避免隔离模式直接 `-m` 找不到本地包。两命令使用相同既有viewer分别读取归档，不启动仿真。

R0，V04真实正常全流程，初始 t=35（gate0入场完成/跨环开始）：

```powershell
@'
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, r"D:\7_codespace\terminal")
import runpy
sys.argv = ["terminal_marl.scripts.open_stage1_replay",
    r"D:\7_codespace\terminal\artifacts\rotating_slots\final_pass_reference_visual_20261019a\materials_20261019a\V04\replay_00",
    "--time", "35"]
runpy.run_module("terminal_marl.scripts.open_stage1_replay", run_name="__main__")
'@ | & 'D:\7_codespace\terminal\venv\Scripts\python.exe' -I -B -
```

OPT，同一脚本设置与初始时刻：

```powershell
@'
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, r"D:\7_codespace\terminal")
import runpy
sys.argv = ["terminal_marl.scripts.open_stage1_replay",
    r"D:\7_codespace\terminal\artifacts\rotating_slots\final_pass_regression_local_20261019a\V04\replay_00",
    "--time", "35"]
runpy.run_module("terminal_marl.scripts.open_stage1_replay", run_name="__main__")
'@ | & 'D:\7_codespace\terminal\venv\Scripts\python.exe' -I -B -
```

查看其他材料时，只将两命令的bundle路径替换为下表同一对R0/OPT实际路径，并将`--time`设为同一定位时刻；查看接缝时替换为上节 paired 两侧 bundle，初始 `--time` 设为 `300`，再定位 `299.957955557834` 与 `300.2`；F08复用源的`replay_00`对应OPT V08的`replay_01`，不能只靠同名目录配对。G04/G05历史三档回放与t30截图仍保留在上节路径。交互使用Space暂停、Right单步、p/n跳事件、时间滑条seek、1/2/3三视图、f跟随、t轨迹、l槽标签、d详情；查看定位时刻及相邻帧，记录实际事件和差异，关闭窗口结束交互。

## 最低八类人工记录

材料状态只描述当前输入。`AVAILABLE`不表示已观察，`MATERIAL_PENDING`不虚构回放链接。各行人工状态均为 `HUMAN_PENDING`；观察结论和异常由研究者实际填写。

| 类别 / 关联条目 | 现有材料与真实定位 | 材料状态及未覆盖事实 | 人工状态 | 观察结论 / 异常（留空） |
| --- | --- | --- | --- | --- |
| 正常入场与跨环 / H02 | [R0](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_reference_visual_20261019a/materials_20261019a/V04/replay_00) / [OPT](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_regression_local_20261019a/V04/replay_00)；gate0 t=0准入、35入场完成、75/115跨环完成、183.5接地、243.5服务完成 | AVAILABLE；真实四门正常全流程；共同查看三视图和事件轴 | HUMAN_PENDING | |
| 有上游等待的准入 / H02 | [R0 E14](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_reference_regression_local_20261019a/E14/replay_00) / [OPT E14](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_regression_local_20261019a/E14/replay_00)；rerouted_request t0 WAITING/AWAITING_DECISION，t143.5仍WAITING，t144.50000000000577 CENSORED_UPSTREAM，从未admit | AVAILABLE（partial scripted waiting only）；upstream_wait_time=144.50000000000577，auto_emergency=False、not_a_research_baseline=True、scripted_conservation，物理failure终止；等待后准入未覆盖。C05是OUTER_CAPACITY立即reroute，不能替代本行 | HUMAN_PENDING | |
| 服务未结束前出发至未来pad / H02 | [R0](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_reference_regression_local_20261019a/F08/replay_00) / [OPT](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_regression_local_20261019a/V08/replay_01)（R0 F08→OPT V08第二对）；rescue t0 commit/start，pad0–5 t=.35服务结束，source clear29.21084213256836；DESCENT130–143.5，t143.5 emergency touchdown pad0 | AVAILABLE；记录service_end=263.5，归档到约250，未执行至该清场时刻。是否正确显示未来committed RESERVED须人看；不虚构protected event | HUMAN_PENDING | |
| 外环应急穿越内部空域 / H03 | [R0](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_reference_regression_local_20261019a/F01/replay_00) / [OPT](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_regression_local_20261019a/V07/replay_00)（R0 F01外环→OPT V07第一对）；a t0出发，source clear40.13527965545654，DESCENT195–208.5，touchdown208.5 | AVAILABLE；V07另两对是middle/inner。核对真实位置、应急源环、SOC/功率/peak及代价，不能只看截图 | HUMAN_PENDING | |
| 繁忙资源下救援等待及失败 / H03 | [R0](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_reference_regression_local_20261019a/F09/replay_00) / [OPT](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_regression_local_20261019a/V08/replay_00)（R0 F09→OPT V08第一对）；rescue初始HOLD_INNER/EMERGENCY_WAITING；SOC_INSUFFICIENT t0共12条，约2/5/10/15s各6条；t19.117346166184838 ENERGY_FAILURE、SOC0 | AVAILABLE；这是物理能量失败，非预算timeout；对照等待状态、拒绝原行与终止事件 | HUMAN_PENDING | |
| checkpoint续跑接缝 / H04 | [R0](D:/7_codespace/training_outputs/terminal/rotating_slots_perf_v1/final_pass_paired_20261019a/windows/W71-A/group-1/repeat-1/R0/attempt/replay) / [OPT](D:/7_codespace/training_outputs/terminal/rotating_slots_perf_v1/final_pass_paired_20261019a/windows/W71-A/group-1/repeat-1/OPT/attempt/replay)；W71-A repeat1，真实cut=300，邻帧299.957955557834/300/300.2；[定位与离屏材料](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_checkpoint_visual_20261019a/summary.json) | AVAILABLE；真实冷恢复归档含接缝前历史及恢复后推进，三个时刻snapshot/seek只读相同，t300三视图已生成；连续性仍须研究者实际观察 | HUMAN_PENDING | |
| 原始候选拒绝的解释 / H03 | [R0](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_reference_regression_local_20261019a/F09/replay_00) / [OPT](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_regression_local_20261019a/V08/replay_00)的SOC_INSUFFICIENT原行与ENERGY_FAILURE；另见[R0](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_reference_visual_20261019a/materials_20261019a/V01/threshold_replay) / [OPT](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_regression_local_20261019a/V01/threshold_replay)的t0 diagnostic_crossing_rejected/HARD_SEPARATION | AVAILABLE；分别核对资源繁忙时的能量拒绝与诊断安全拒绝。G05 generation NOT_FOUND材料仍在；搜索耗尽不等于全局不可行 | HUMAN_PENDING | |
| 同一对飞机80m附近连续距离 / H03 | [R0](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_reference_visual_20261019a/materials_20261019a/V01/threshold_replay) / [OPT](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_regression_local_20261019a/V01/threshold_replay)；cross_a/cross_b，t0诊断拒绝，witness t=.05，distance79.999m，bounds79.99899999999624–79.99900000000007m | AVAILABLE（diagnostic fixture）；fixture_only / diagnostic_only_not_executed，未执行unsafe飞行；连续证书与原始拒绝详情共同查看，渲染帧本身不证明连续安全 | HUMAN_PENDING | |

查看每一可用对时，三视图都核对真实三维位置、权威轨迹、当前/未来slot与pad预约、SOC/功率/peak、源环与拒绝原因；有相应场景才检查上游队列、服务倒计时和未来committed pad的 `RESERVED`。缺少的显示/材料如实记录，不据截图相同填通过。

## 研究者填写区

observer：________________　实际观察UTC起止：________________

实际viewer代码身份/文件hash：________________

R0 / OPT归档代码身份与source hash：________________

配置hash / 配置bytes hash / 参照包与被查看bundle文件hash：________________

实际查看case、replay编号、时刻/事件、三视图及使用的交互操作：________________

逐类结论、异常、截图或日志路径：________________

主对话最终复核与签收：________________　当前仍为 **HUMAN_PENDING**。

viewer只读归档；不重新生成机动、不在seek时推进world，不平滑掉轨迹或时间差异。观察时长单独记录，不计模拟速度；服务器benchmark关闭viewer。本轮已运行既有PyVista离屏截图，未启动可见GUI、仿真或新物理实验；实际人工观察尚未发生，自动一致性比较不替代人工签收。
