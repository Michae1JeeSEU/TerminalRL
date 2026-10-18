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

`--report-only` 按 `user_scope_amendment_20261019.md` 保存
`report_only_single_group_v1` 策略：每窗只执行首个 AB/BA/AB 六项，噪声为观察值，
不会自动追加第二组。summary 把旧阈值结论保存为 `historical_component_gate`，
当前 `component_gate=REPORT_ONLY`；完整性、语义和缺失证据错误仍返回失败。
`summarize` 消费保存的策略，report-only 不要求 noisy retry，额外第二组仍拒绝。
不带此 flag 或旧 campaign 缺少策略字段时，保留历史一次 noise retry 规则和 gate。

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
累计值。采样 RSS 与 OS lifetime HWM sum 都是估计，不声明严格上下界或实际
同时峰值；collector+kernel 的采样与 kernel+workers 的采样分别保留，不把未
对齐时刻的 peaks 相加。此前子进程也可能影响 lifetime HWM。
这些资源原值不代表其他完整合同资源门槛已经获得资格。

`collect()` 保留原 `logical_cpu` 调用，并允许显式 `cpu_affinity` 列表；child
检查请求为实际 allowed CPU 的子集，固定 native1/nice10/fresh `-I -B`。
kind=`window` 只运行已知 windows CLI 的 `window`，kind=`runner` 只运行已知
runner CLI 的 `traffic`；都要求唯一 `--output`，不接受任意脚本。
这一 callable 支持后续固定串并行 job；本 paired campaign 的默认命令仍为 serial。
collector 自身额外一个进程，与 kernel 加 worker 数分列，不新增进程数量 gate。
`collector_uses_pool=false` 和 legacy `no_pool=true` 都仅描述 collector；实际
kernel 的 pool 从 `inner_worker_count` 观察，未成功 spawn 时 kernel 与 kernel 加
worker 的实际进程数量均为 0。
对于真实 kernel，exit0 之外还读取自己指定的 result：资源 `cpu_complete=true`，
并行还需 `closed_and_joined=true`，才声明 descendant CPU 完整。Linux wait 后
RUSAGE_CHILDREN 包含由 kernel 已回收的 workers；异常、硬超时或缺失 join 证据
保留已回收 CPU 的 lower-bound 观测，完整 CPU 为 null/incomplete，不能声称拥有
未回收 orphan CPU。已知 stdlib 小 fixture 不创建 descendants，单独标明该范围。

## 结果核验、噪声与汇总

每个实际结果必须为 `COMPLETED_WINDOW`，保持同一个 frozen checkpoint cut、
原 cursor/target/workload/semantic identity、冷 cache 证据与 benchmark serial
后端。开始、实际停止和 sim delta 的 `float.hex` 必须与同组各 job 精确一致，
不是只比较名义 10 秒。六个 ReplayBundle 数据成员
`states/events/rejections.jsonl`、`trajectories.json`、`manifest.json`、
`metrics.json` 的 SHA 必须一致，index、result 和 input identity 的 SHA 也
保留并重查。实际 `ReplayBundle.validate` 已由每次 window CLI 执行，不用
record/replay 代替 checkpoint 或额外改变物理步。

冷缓存检查必须包含原五个数值 cache。仅额外允许已审查的
`energy._ROOT_CACHE`、`energy._PEAK_CACHE` 与已命名的
`trajectory._BERNSTEIN_METADATA` 观察项：`present` 为 bool 且 `entries=0`；
peak 和 metadata 的 entries 必须是 built-in int，不接受 bool 冒充整数零。
R0 可明确报告 `present=false`，OPT 可报告 `present=true`。未出现可选项仍兼容；
warm 或其他未审查 cache key 均拒绝，不把任意额外项作为自动获批的缓存。

历史默认策略中，R0 或 OPT 三次 rollout wall 的 `max/min > 1.25` 时整组六项标记 `NOISY`，
保存 load/memory 复查，再完整重做这一组六项一次。两组全部保留；固定使用
第二组而不选择较快组。第二组仍 NOISY 时不可用于通过旧 gate，不继续重试。
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

2026-10-19 本轮窗口/collector 集成本地 batch 为
`84 passed, 3 skipped in 21.47s`，numeric exit 0（wrapper 22.117351s）。
执行 windows、paired 两文件和既有真实 pool1 reference/emergency 等价小样本，
fresh venv `-I -B` 且导入 pytest 前禁 bytecode。新检查覆盖 known runner/window CLI、
显式 affinity、pool join/resource 证据、缺结果时保留 CPU lower bound、report-only
首组噪声与 saved policy，以及拒绝额外 retry。聚合与 completion evidence fixtures
均明确 synthetic，不是 Linux 或真实 traffic 的性能结论。
实际命令和 PID 在 `artifacts/rotating_slots/window_collector_tests_20261019a.launch.json`，
stdout/stderr/numeric exit 同前缀；basetemp 为同前缀目录。未设置
`RUN_WINDOW_INTEGRATION`；三项 skip 为 opt-in window 和两个 Linux-only paired 检查。
Linux 联合窗口/guard/orchestration、完整 traffic 及串并行测量尚未由本批次执行。
随后修正 failed-spawn 的 kernel 加 worker 总数为 0，并新增明确的
`collector_uses_pool=false` 标签；仅复跑该既有 failed-spawn 测试，
`1 passed in 0.22s`、numeric exit 0。原始命令/stdout/stderr/exit 保留于
`artifacts/rotating_slots/collector_failed_spawn_tests_20261019a.*`，没有重复完整 batch。

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
collector 调用 seam：真实 stdlib child 结束后由测试父进程复制明确 synthetic
结果，不经过真实 physics
或 CLI admission，不把 fixture 中的 wall 常量当作性能数据。生产命令构造
没有新增 emitter/绕过验证的 CLI 选项。首次正式 W71-A AB/BA/AB 组同时提供实际
window CLI 链路检查，遇完整性错误即停止，不另增一轮 diagnostic campaign。

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
两项 Linux 集成检查及 deadline 的四项 Linux 检查。在该次本地检查时 Linux 联合检查和真实
W71-A 六作业 diagnostic subset 尚未执行。新的 helper/入口会改变部署指纹，
实际运行必须重新生成与其精确匹配的 deployment 和已登记 migration pair，
不能沿用旧 identity。


后续主对话已复核旧 root-only 版本的 Linux `root_paired` 联合检查：stdout 为
`120 passed in 13.82s`，nodeids 有 120 项；metadata 中测试计数字段却为 0。
保留这项矛盾，不改写原 metadata，也不把 0 当作实际执行数量。该记录发生在本轮
OrderedDict/root 扩容及 peak cache 应用之前，不代表当前新版本已完成 Linux
检查。新 known-PEAK 白名单及对应测试纳入本轮本地六文件 batch；实际结果另行
记录，Linux/opt-in 集成仍需显式执行。


本轮本地六文件复测已退出 0：`116 passed, 3 skipped in 58.28s`，使用 venv
`-I -B`，在导入 pytest 前禁写 bytecode 并显式设置项目 import 路径；basetemp
为 `artifacts/rotating_slots/energy_cache_tests_20261018b`，相同前缀外置保存实际
命令、stdout/stderr、exit 和独立 R0/OPT 证据。范围为 root/peak reuse、energy
queries、physics、windows、paired；三项 skip 为 opt-in window 与两项 Linux
paired 检查。新的 known-PEAK whitelist、warm/unknown 拒绝和字段类型检查已在
该本地批次执行；当前版本 Linux/实际 common-window 测量仍未完成。

首次 `energy_cache_tests_20261018a` 的 `115 passed, 3 skipped, 1 error in 49.35s`
及完整日志保留。主对话诊断错误仅在 synthetic unresolved trace 的 tuple 序列化，
将 `.pair()` 调用改为 `exact([v,a])` 后完整重跑；没有为测试改变 kernel、缓存预算
或物理/数值政策。旧 Linux 120/nodeids120/metadata0 的矛盾记录继续保留。


Bernstein metadata 候选阶段只增加上述 known-key 观察和严格白名单；warm、unknown、
missing 和字段类型负向 fixtures 已准备，不把任意新增 cache key 自动批准。
现有 final-cache observer 保持原函数体，读取同一只读快照；worker reset 接入仍为
后续待定范围，本补丁不修改 parallel。artifact 隔离候选测试不能替代应用版联合
检查、真实 common-window 测量或 C/P 资格。


本次 metadata 白名单的九个 synthetic 候选 case 已通过 artifact-only canonical
paired import 执行：`9 passed, 36 deselected in 3.25s`，退出 0。显式路径检查确认
仅 paired 使用命名 artifact，其余 helper 来自实际当前树；实际源码未应用。
metadata 数值候选仍沿用原 9380 查询矩阵，八个候选测试 `8 passed in 7.12s`，
不把这些隔离结果当作应用版 physics/window、真实测量或 C/P 资格。


当前状态：Bernstein known-key 观察及严格白名单已应用；最终 observer 原函数体保留。
peak cache 默认关闭，仍有显式启用入口；其观察项存在但冷 entry 为零仍符合原检查。
上述 artifact 候选测试是历史隔离证据，不冒充当前应用版验证。当前目标本地 batch
覆盖 Bernstein、peak/root、P2、Cartesian、execution reuse、energy queries、
windows 与 paired，实际结果见下段；`RUN_WINDOW_INTEGRATION` 本次不设置，
Linux 联合集成和实际测量后续单独核验。

本轮范围按 `user_scope_amendment_20261019.md`：完成一次集成、测量与回归并报告
收益/退化和未覆盖项；原程序 gate/历史状态保留为观察值，不因未达到 2× 再反复
寻优，也不伪写原合同 `PERF_ACCEPTED`。物理、安全、等价和完整记录要求不变。


当前应用版目标 batch 已退出 0：`164 passed, 3 skipped in 74.06s`，wrapper
74.672s；九文件范围及实际命令见
`artifacts/rotating_slots/final_kernel_tests_20261019a.launch.json`，stdout/exit
在同前缀外置文件。未设置 artifact override，Bernstein 9380 查询及 repeat 使用
实际 trajectory 源码并与独立 R0 精确一致；fresh 默认-off peak 检查也通过。
三项 skip 是 opt-in window fixture 和两项 Linux-only paired 检查；本次没有执行
Linux 联合集成、真实串并行 traffic 或新的 common-window 测量，生命周期集成及
后续一次测量/回归/人工观察工作仍由各自真实证据验收。
