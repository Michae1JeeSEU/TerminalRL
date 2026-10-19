# 固定六窗完整 60 sim R0/OPT 对照准备

`check_sustained_oracle.py` 只补既定完整窗口数值/离散对照，不寻优。正式运行顺序固定为 W71-A、W71-B、W71-C、W60、W120、W180，每窗一个 fresh R0_SERIAL、0 worker、名义 60 sim；不重跑 OPT，也不生成三重复速度主表。CPU 0、nice 10、外层设置七个原生线程变量均为 1（含 OMP_THREAD_LIMIT）。冻结 collector 的 p.THREADS 只观测六字段：OMP/OPENBLAS/MKL/NUMEXPR/VECLIB/BLIS；检查器严格要求这六个已记录值均为字符串 1，不虚称记录了第七项。原 collector 为每个 `-I -B` kernel 独立提供 7200 wall 秒 Linux guard。没有后台启动器、retry 或新调度器。

正式入口使用原 window CLI，固定 corpus、R0 导出、R0 migration approval 和部署 `4623e38d9b13238f6ca886229c603a5626285698`；`--kernel R0 --duration 60 --mode benchmark --inner-workers 0 --core-budget 1 --process-budget 1`。新输出仅为 `/home/user/training_outputs/terminal/rotating_slots_perf_v1/final_pass_sustained_oracle_20261019a`，拒绝覆盖。新增 driver 不在冻结 523-source map/extra_paths 中；外层复用 traffic_pass 的 `modules={}` 固定文件核验并单列自身 SHA，不改部署。原 window 的运行时导入、配置、semantics、helper、shim、迁移及来源结束校验照常执行。

`run` 首先要求 sustained c 真正 18/18 终态、外层 exit 0、source 未变，六窗各三份 OPT6 结果均冷启动、worker 父身份/初始化/join/完整 CPU 有证，并核对原始 row、archive index/成员、输入身份、精确时间线与同窗三重复一致。缺项或异常在启动 R0 前返回非零，不给出成功结论。每个 R0 完成后，核验其实际 R0 replay 模块路径与原 window 的 `ReplayBundle.validate`/source-finish 执行记录，再与同窗全部三个 OPT 结果逐字节/SHA 比较六物理成员及 start/stop/actual-delta 的 float.hex；自然延期不量化或裁切。

输出每窗 row、原 collector 日志/guard/实际 argv、原 window 结果/回放，以及最终 completion/numeric exit。未知异常、物理或来源错误、语义差异均停止，保留已写证据交主对话；匹配的 PERFORMANCE_TIMEOUT 保留并继续剩余固定窗口，最终为不完整，不能成为 `SIX_WINDOW_60SIM_ORACLE_EXACT`。无超时且六窗全部相同才记录该限定状态；仍为 HUMAN_PENDING，不宣称全合同或速度资格通过。

CPU 保留 joined kernel 与 collector 各自口径；外层 CPU 与逐 job collector 重叠，不能相加，最终 write/print 尾部未计。timeout 不宣称完整后代 CPU。RSS PID sweep/HWM 为估计，不是严格上界。原始 OPT 输出、输入/来源文件在操作前后只读核对，不复制或伪造 checkpoint。原 window CLI 在其选定内核中执行 ReplayBundle.validate；外层不导入当前物理 world 生成额外场景，也不把 OPT validator 用于 R0。

## 历史准备与运行接口

本地准备阶段只执行 AST/help/self-check；后续服务器旧单轮实际运行及中断终态见下节。self-check 的小文件明确为 SYNTHETIC_BYTE_COMPARISON_ONLY，不是有效 ReplayBundle，不调用物理验证 API。

```powershell
& 'D:\7_codespace\terminal\venv\Scripts\python.exe' -I -B 'D:\7_codespace\terminal\artifacts\rotating_slots\final_pass_sustained_oracle_20261019a\check_sustained_oracle.py' self-check
```

默认 `plan` 只打印固定六条 CLI 与 7200 预算，不导入项目 helper、不启动进程。自检覆盖顺序/六条 R0 命令、六成员相等、一 ULP 时间差异、成员字节差异、缺第三份 OPT 拒绝，以及 timeout 不能成为全部相同。自检输出目录拒绝覆盖，额外自检须显式 `--self-check-output` 指定同 artifact 下新目录。

以下为旧单轮的 Linux 运行接口记录；本轮已中断，当前不重新执行，不能在本地运行：

```bash
/home/user/envs/terminal_env/bin/python -I -B /home/user/projects/terminal/artifacts/rotating_slots/final_pass_sustained_oracle_20261019a/check_sustained_oracle.py run
```

同一 Linux 入口以 `summarize` 只读检查保存的 row evidence SHA 并打印既有结果，不重跑 kernel、不改写汇总。缺完整 OPT 终态时，不应启动正式 `run`。是否实际启动及最终签收由主对话决定。

## 历史预检校验器修复（2026-10-20）

历史 `preflight_20261020a.json` 保留，未 launch、无 R0 物理样本；原检查器将七项启动设置当成六项 collector 观测要求。按主对话限定修复，环境观测现在严格与原 `p.THREADS` 六字段相等；来源、语义、affinity、cold 及其他校验保持。外层仍设置七变量全为 1。旧 self-check 保留；新 `self_check_20261020b/self_check.json` 绑定当前 driver SHA，六字段全 1 接受，每个字段缺失或非 1 共 12 个负例均拒绝，并检查七项启动设置不变。实际已完成 W71-A repeat1 的六字段只读核对见同目录 `actual_thread_observation.json`；它不是新的物理执行。AST/help/局部自检均 exit 0。这是当时的准备快照；随后旧单轮已实际运行并中断，未形成六窗完整对照结论。

## 旧单轮实际终态

[completion](D:/7_codespace/training_outputs/terminal/rotating_slots_perf_v1/final_pass_sustained_oracle_20261019a/completion.json) 为 STOPPED_ON_SOURCE_ERROR，server finished UTC 2026-10-19 17:49:38.124928，根与 launch exit 2，outer error 为 `outer fixed file inventory changed`。A/B/C/W60 的既有 row 为 ORACLE_EXACT，各与三份 OPT 的六成员及精确时间线相同，共四窗 / 12 pairs / 72 member comparisons；R0各窗仅n=1，不是正式速度重复表。W120 result 为 IMPLEMENTATION_FAILURE，报 `ProvenanceError: DEPLOYMENT_BYTES_CHANGED_DURING_RUN: terminal_marl/core/rotating_slots/audit.py`，collector exit1、无row；已观察到的delta60/rollout596.62691618799不作合格样本。W180 NOT_RUN。source_finish.json 缺失，不补造。

[终态采集](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_sustained_oracle_20261019a_progress_2.json) 记录 local UTC 18:11:41.7505638–18:11:42.1517773 / server UTC 18:12:30.666815、native SSH0、匹配进程0。[interruption_review](D:/7_codespace/terminal/artifacts/rotating_slots/final_pass_sustained_oracle_20261019a/interruption_review_20261020.json) 引用原始产物 SHA，并将主对话后续7文件差异记录与原始运行证据分开；不归因具体写命令/提交时刻、不评价后续优化。原manifest未改，c59aa6b/e343c77不替换旧4623e38部署身份。现有summarize接口确实存在，本轮未调用。

既有36 paired、24 parallel和18 OPT sustained结果保留；本次不能称六窗oracle完成。traffic15case/30jobs未运行，原预定R0 4/OPT6但未launch；共享源码已变且版本方向待用户确认，因此本轮不再推进旧 traffic。主已成功删除旧automation-2，不等于自动工作全部完成。70映射仍旧快照，94索引EVIDENCE_INDEX_REVIEWED不代表最终准则验收；HUMAN_PENDING保留，后续版本选择由用户决定。
