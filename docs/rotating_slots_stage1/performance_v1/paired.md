# 串行共同窗口 paired component gate

`scripts/server/run_rotating_slots_performance_paired.py` 提供 `paired` 与
`summarize` 两个真实入口。该功能只实现合同 §16.1 的六个 10 sim 秒串行窗口
component gate；即使该 component 返回 `PASS`，整体仍为
`NOT_QUALIFIED / HUMAN_PENDING`。不代表 60 秒持续窗口、完整 15 项 traffic、
并行后端、独立事后安全审计或人工观察已完成。

## 执行输入和顺序

`paired` 必须指定已有的 `--corpus-manifest`、canonical `--corpus-sha256`、
`--reference`、`--deployment-manifest`、当前 `--implementation-commit`、
`--r0-migration-approval`、`--opt-migration-approval`、`--logical-cpu` 和全新
`--output`。只支持真实 Linux 新进程 `python -I -B`，不支持覆写已有 campaign。
stdout 输出完整 summary，campaign 内也保存 `summary.json`。

默认严格按 W71-A、W71-B、W71-C、W60、W120、W180 顺序，每窗口三次配对：

| repeat | 实际启动顺序 |
| --- | --- |
| 1 | R0，OPT |
| 2 | OPT，R0 |
| 3 | R0，OPT |

初始矩阵为 36 个 job。每个 job 是独立 serial kernel child，无进程池，固定
10 sim 秒、benchmark 模式、完整原 driver/记录和冷恢复；不启用 cProfile。
`--diagnostic-windows` 可显式选子集，仍按固定六 ID 顺序执行。该选项无论选了
几个窗口都标为 diagnostic，不能通过六窗口 component gate。

父 collector 先用已有 corpus/provenance/contract/approval validators 检查冻结
选择、deployment、固定 24 文件加 manifest 的 R0、语义和精确迁移 pair，读取
真实 supporting evidence。父进程不导入任何 `terminal_marl` core。每个 child
重新经过实际 window CLI 的 `runner.bootstrap`，在恢复/advance 前验证实际
target execution；paired 不建立新的物理计算公式或来源旁路。

CPU 选择为同一个显式 logical CPU，记录允许 CPU 集、package/core/thread
siblings、nice、native 线程设置、load、可用内存、UTC、所有实际命令和 PID。
child 固定 nice 10、所有六个 native 库线程变量为 1；不改变其他进程、现有
任务、策略、需求或 frozen source 绝对路径。

## 生命周期、timeout 与成本

collector 同时只等待一个 kernel child。复用已审查诊断 launcher 的 fresh
child/collect 结构及 `LinuxWallDeadline`，对 child CLI 工作设置独立 7200 秒
kernel SIGALRM，未安装 Python timeout callback。只有实际 wait 得到
`-SIGALRM`，且唯一 ARMED record 的 PID/signal/deadline 匹配，才分类为
`PERFORMANCE_TIMEOUT`。普通非零、缺结果、错误环境或结果差异都会停止后续
job 并保留已有输出。Popen 失败不伪造 numeric exit code。

每个 slot 的 `collector/` 与新的 `attempt/` 是兄弟目录。预建日志目录不会使
真实 window CLI 的输出目录预先存在；不移动源记录或已生成结果。日志保留
stdout/stderr、实际 numeric exit、guard records、runtime environment 和 RSS
采样。恢复与窗口计算仍由现有 window CLI 执行。

`OwnedProcessMetrics` 从 spawn 前到 wait/join 后只拥有当前一个 child。
完整 joined child CPU 与 collector `time.process_time` CPU 分列，并另记
post-join 结果验证的父 CPU 与包括它的 tree CPU。campaign 用最终 accounting
snapshot 的绝对 `time.process_time()` 包含 collector 启动/import 成本，另列
从 paired/preflight 入口开始的 run interval delta、全部 joined child CPU及
完整 collector CPU 与 child CPU 之和。最后 campaign/summary JSON 序列化和
stdout 输出位于该 snapshot 之后，明确不在此累计内；未知 child CPU 保留为
null 与 incomplete，不用零冒充完整成本。child rollout
CPU 单列，不能替代完整 child 生命周期；不累加嵌套 `world.cpu` 或 profile
累计值。采样 RSS 是父+当前 child 的同时 sweep 下界；OS lifetime HWM 的和
只作为声明的上界，不能称为实际同时峰值。此前子进程可能使 HWM 上界更松。
这些资源原值不代表其他完整合同资源门槛已经获得资格。

## 结果核验、噪声与汇总

每个实际结果必须为 `COMPLETED_WINDOW`，保持同一个 frozen checkpoint cut、
原 cursor/target/workload/semantic identity、冷 cache 证据与 benchmark serial
后端。开始、实际停止和 sim delta 的 `float.hex` 必须与同组各 job 精确一致，
不是只比较名义 10 秒。六个 ReplayBundle 数据成员
`states/events/rejections.jsonl`、`trajectories.json`、`manifest.json`、
`metrics.json` 的 SHA 必须一致，index、result 和 input identity 的 SHA 也
保留并重查。实际 `ReplayBundle.validate` 已由每次 window CLI 执行，不用
record/replay 代替 checkpoint 或额外改变物理步。

冷缓存检查必须包含原五个数值 cache。仅额外允许已知的
`energy._ROOT_CACHE` 观察项：`present` 为 bool 且 `entries=0`；R0 可明确报告
`present=false`，OPT 可报告 `present=true`。未出现该可选项仍兼容；warm 或
其他未审查 cache key 均拒绝，不把任意额外项作为自动获批的缓存。

R0 或 OPT 三次 rollout wall 的 `max/min > 1.25` 时整组六项标记 `NOISY`，
保存 load/memory 复查，再完整重做这一组六项一次。两组全部保留；固定使用
第二组而不选择较快组。第二组仍 NOISY 时不可用于通过 gate，不继续重试。
非噪声组不允许额外 retry。

每窗口保存三次 R0/OPT 原值、median、nearest-rank P95（n=3，即最大值）、
最大值、按 repeat 配对的 wall ratio，以及单列 rollout CPU 比率。六窗口
median wall speedup 的几何平均必须 `>= 2.0`，且每个窗口满足
`O <= max(1.10*B, B+0.05s)`。不更改阈值，不挑最好 repeat 或最好 group。

`summarize --campaign PATH` 只读取已持久化文件，可以指定全新的 `--output`
保存另一个 JSON；默认仅打印。它按固定 slot 检查 coverage、重复/意外项、
错误、guard/environment/CPU 标签及 result/input/replay/collector evidence
hash，重新计算噪声与最终 component gate。缺失、错误 campaign 或数据变化
返回 `INVALID_OR_INCOMPLETE`，diagnostic subset 永不 PASS。

## 验证范围

新增测试覆盖固定顺序、36 项 coverage、缺失/额外 collector、hash 变化、
profile/cold/CPU/environment/guard/timeline 错误、噪声阈值、一次整组 retry、
second NOISY 和速度/逐窗阈值。汇总测试使用明确 synthetic metadata/data
fixture，只验证聚合与拒绝逻辑，不声称真实 traffic 等价。

小型真实 collector 测试创建 `python -I -B` 的纯 stdlib/helper child，验证
实际退出码、隔离/native1、完整 child CPU/RSS 记录及无 child 时不造 numeric
exit。实际 SIGALRM 测试仅在 Linux 执行；Windows 明确 skip。它们不运行真实
window，不构成性能或六窗口 traffic 证据。现有 window executor 的真实
service-events fixture 仍保留其已披露 admission seam 限制。

Linux 另有一个真正执行 `paired()` 的 tiny orchestration 测试：单窗口
diagnostic 顺序启动六个实际 `-I -B` child，经过真实 kernel guard、完整
collector CPU/RSS、持久化与离线 summarize。该测试仅替换 preflight 和内部
`window_command` 为明确标记的 synthetic 结果 emitter，不经过真实 physics
或 CLI admission，不把 fixture 中的 wall 常量当作性能数据。生产命令构造
没有新增 emitter/绕过验证的 CLI 选项。正式全六窗之前仍需真实冻结 W71-A
六作业 diagnostic subset 验证实际 window CLI 链路。

历史准备阶段只把新模块预加载到测试进程，没有 apply tracked source。Windows
本地最后一次 draft 测试结果为 `25 passed, 2 skipped in 11.19s`，日志位于
`D:/7_codespace/terminal/artifacts/rotating_slots/paired_patch_20261018a/local_tests_20261018c.log`。
两项 skip 分别是实际 Linux SIGALRM 和 Linux six-child orchestration；不能
据 Windows 结果声称它们已经执行。

主对话复核后，本轮已应用四个新文件。本地通过 venv Python `-I -B`、导入前
禁写 bytecode、显式项目 import 路径运行 paired 测试、
`test_rotating_slots_performance_parallel.py::test_resource_samples_are_bounded_and_full_series_is_explicit`
及 deadline 测试，结果为 `41 passed, 6 skipped in 12.74s`，退出码 0。
新 basetemp 是 `artifacts/rotating_slots/paired_tests_20261018a`，日志是
`artifacts/rotating_slots/paired_tests_20261018a.log`。六项 skip 为 paired 的
两项 Linux 集成检查及 deadline 的四项 Linux 检查。Linux 联合检查和真实
W71-A 六作业 diagnostic subset 尚未执行。新的 helper/入口会改变部署指纹，
实际运行必须重新生成与其精确匹配的 deployment 和已登记 migration pair，
不能沿用旧 identity。
