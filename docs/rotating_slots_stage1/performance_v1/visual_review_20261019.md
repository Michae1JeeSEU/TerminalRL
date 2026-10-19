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

## 两个可直接执行的本地命令

在 PowerShell 中分别执行；命令会打开交互窗口，本文编写时未运行。使用项目现有 venv/PyVista；入口依赖说明见[requirements-visualization.txt](D:/7_codespace/terminal/requirements-visualization.txt)。脚本经stdin交给 `-I -B`，在任何项目导入前设置 `dont_write_bytecode` 和显式 `sys.path`，避免隔离模式直接 `-m` 找不到本地包。两命令使用相同既有viewer分别读取归档，不启动仿真。

R0，G04 dt=.2，初始 t=30：

```powershell
@'
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, r"D:\7_codespace\terminal")
import runpy
sys.argv = ["terminal_marl.scripts.open_stage1_replay",
    r"D:\7_codespace\terminal\artifacts\rotating_slots\final_pass_period_reference_local_20261019a\G04\replay_00",
    "--time", "30"]
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
    r"D:\7_codespace\terminal\artifacts\rotating_slots\final_pass_regression_local_20261019a\G04\replay_00",
    "--time", "30"]
runpy.run_module("terminal_marl.scripts.open_stage1_replay", run_name="__main__")
'@ | & 'D:\7_codespace\terminal\venv\Scripts\python.exe' -I -B -
```

观察正常入场/跨环时，将两命令中的 `G04\replay_00` 同时改为 `G05\replay_00`，初始 `--time` 改为 `35`；仍保持相同case、replay编号和时刻。另外两档分别改为 `replay_01/02`，不能将不同dt/period当作R0/OPT一对。交互使用 Space暂停、Right单步、p/n跳事件、时间滑条seek、1/2/3切换俯视/侧视/三维、f跟随、t轨迹、l槽标签、d详情；共同查看指定时刻以及相邻帧，记录实际看到的事件和差异。窗口关闭才结束交互。

## 最低八类人工记录

材料状态只描述当前输入。`AVAILABLE`不表示已观察，`MATERIAL_PENDING`不虚构回放链接。各行人工状态均为 `HUMAN_PENDING`；观察结论和异常由研究者实际填写。

| 类别 / 关联条目 | 现有材料与真实定位 | 材料状态及未覆盖事实 | 人工状态 | 观察结论 / 异常（留空） |
| --- | --- | --- | --- | --- |
| 正常入场与跨环 / H02 | [R0 G05/00](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_period_reference_local_20261019a/G05/replay_00)、[OPT G05/00](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_regression_local_20261019a/G05/replay_00)；r在0/35/75/115/183.5/243.5s的事件 | AVAILABLE；观察入场、两次跨环、final及服务，原始事件轴不重生成 | HUMAN_PENDING | |
| 有上游等待的准入 / H02 | 本作业单已核实的G05在t=0立即admit | MATERIAL_PENDING；未定位该类别的成对材料，不能将立即准入当等待 | HUMAN_PENDING | |
| 服务未结束前出发至未来pad / H02 | G04有other_ground在pad3，a去pad0 | MATERIAL_PENDING；这不证明同一未来pad的保护下降区间及RESERVED显示 | HUMAN_PENDING | |
| 外环应急穿越内部空域 / H03 | G04是inner normal final；G05正常单请求 | MATERIAL_PENDING；未定位外环应急成对事件/回放 | HUMAN_PENDING | |
| 繁忙资源下救援等待及失败 / H03 | 已核实G04/G05无此救援等待失败事件 | MATERIAL_PENDING；不以普通ground服务或正常着陆替代 | HUMAN_PENDING | |
| checkpoint续跑接缝 / H04 | 已有G04/G05回放未标为本类别的成对续跑接缝材料 | MATERIAL_PENDING；需实际接缝定位及前后seek/单步/跳事件/flush对照，当前不造时间 | HUMAN_PENDING | |
| 原始候选拒绝的解释 / H03 | [R0 G05拒绝](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_period_reference_local_20261019a/G05/replay_00/rejections.jsonl)、[OPT G05拒绝](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_regression_local_20261019a/G05/replay_00/rejections.jsonl)；t=0，r，generation NOT_FOUND，原reason_counts | AVAILABLE（局部）；对照详情overlay与原行；不把搜索耗尽说成物理失败/全局不可行，完整应急失败解释仍缺 | HUMAN_PENDING | |
| 同一对飞机80m附近连续距离 / H03 | G04的连续minimum证书见[observations](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_regression_local_20261019a/G04/observations.json)：126.3420680482248–126.34283293216514m，witness 58.366943359375s | MATERIAL_PENDING；该距离不是80m附近；不能仅凭渲染帧证明连续80m证书 | HUMAN_PENDING | |

查看每一可用对时，三视图都核对真实三维位置、权威轨迹、当前/未来slot与pad预约、SOC/功率/peak、源环与拒绝原因；有相应场景才检查上游队列、服务倒计时和未来committed pad的 `RESERVED`。缺少的显示/材料如实记录，不据截图相同填通过。

## 研究者填写区

observer：________________　实际观察UTC起止：________________

实际viewer代码身份/文件hash：________________

R0 / OPT归档代码身份与source hash：________________

配置hash / 配置bytes hash / 参照包与被查看bundle文件hash：________________

实际查看case、replay编号、时刻/事件、三视图及使用的交互操作：________________

逐类结论、异常、截图或日志路径：________________

主对话最终复核与签收：________________　当前仍为 **HUMAN_PENDING**。

viewer只读归档；不重新生成机动、不在seek时推进world，不平滑掉轨迹或时间差异。观察时长单独记录，不计模拟速度；服务器benchmark关闭viewer。本文未启动UI、仿真或新实验，未以自动一致性比较替代人工签收。
