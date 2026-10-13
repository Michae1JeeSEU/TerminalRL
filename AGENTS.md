# AGENTS.md

## 一般工作流与模型分工

本节是本项目后续对话的默认执行约定；用户当次明确指定模型或流程时，以当次要求为准。本项目的模型分工优先于通用的“所有执行任务均交给 luna_worker”约定。

| 工作 | 默认模型与推理强度 | 职责 |
|---|---|---|
| 核心研究及项目代码的新增、修改和修复，包括算法、实验脚本、测试及与实现配套的研究文档 | GPT-6 Astra，`model=gpt-6-astra`，`reasoning_effort=high` | 按主对话确定的范围完成实现和相关验证，提交可核验的改动与证据 |
| 服务器任务启动验证和进度监控 | GPT-5.6 Luna，`model=gpt-5.6-luna`，`reasoning_effort=high` | 执行已确定的命令，收集进程、日志、产物、进度和时间证据，按下文监控规则返回主对话 |
| 意图理解、研究范围、方案选择、故障诊断、实现复核、结果解释和最终验收 | 主对话当前模型 | 保留决策权，独立审查子任务证据并接续原定工作 |

1. **先定范围再委派**：主对话先读取适用说明和当前状态，明确目标、允许修改的文件、输入、验收条件和停止条件。任务应具体、有边界、可验证；独立任务可并行，依赖任务及重叠写入串行。所有 worker 都须知道存在其他协作者，不得覆盖或撤销他人改动。
2. **显式选择模型**：核心研究与代码任务交给 Astra high 实现；服务器监控交给 Luna high。若运行时通过 `spawn_agent` 指定模型，使用支持模型覆盖的 `fork_turns=none`，并提供完整任务背景；不要使用固定为其他模型或推理强度的自定义角色来冒充上述配置。纯机械的文件整理、指定测试及日志收集等可继续交给通用 luna_worker，但服务器监控仍必须使用 Luna high。
3. **主对话必须复核实现**：Astra 完成后，主对话检查实际 diff、需求与接口语义、适用的测试/小样本输出，以及研究约束是否保持。不能仅凭 worker 的“已完成”或测试摘要直接验收。出现问题由主对话诊断，再向 Astra 下达有边界的修复任务；复核通过后才能启动依赖该实现的正式服务器任务。
4. **报告只提供执行证据**：worker 使用 `status`、`actions`、`artifacts`、`checks`、`deviations`、`unresolved` 六项报告，给出实际命令、文件路径、PID/job ID、退出码、测试结果和观测值。将观测事实与缺失信息分开；超出范围、出现异常或需调整方案时返回主对话，不自行扩展研究、解释失败原因或改变协议。
5. **完成闭环**：主对话核验结果后，继续已授权的分析、文档更新和版本交付。涉及文件修改时按 Git 准则检查、验证、提交并推送；仅监控或等待不会自动授权新实验或协议调整。
6. **如实处理工具限制**：`AGENTS.md` 约定工作流，不能直接切换主对话模型或改变运行时固定角色。指定模型、推理强度或 heartbeat 工具不可用时，明确报告限制和未完成环节，不默默替换模型、不宣称自动切换或定时接续已成功。

## 阶段6当前范围（2026-10-13）

用户已授权[阶段6学习辅助联合调度](docs/fixed_terminal_stage6_learning_assisted_scheduling_plan.md)，当前进入implementation/preflight，正式264episode和30同状态面板（15jobs、240方法行）尚未启动。复用阶段5冻结MLP rank/GB，无需重训；固定物理、93维特征、pool14和FCFS闭合J。八方法比较采用TOTAL B3→B5→B8→full及已到达机群正常完成上界证书，证书不保证未来到达或全J最优。质量条件epsilon_normal=0与同状态非退化计算减少10%门槛预先冻结，A/B/C判断等待正式证据。

当前依据为[方法](docs/arrival_restructure/stage6_learning_assisted_scheduling.md)、[协议](docs/experiments/fixed_terminal_stage6_protocol.md)、[运行状态](docs/evidence/fixed_terminal_stage6/run_status.md)。阶段1–5历史事实与结论保持；下文“下一阶段另定范围”等表述属于阶段5结束时的历史状态，本节记录后续明确授权。阶段6仍不开展PPO、actor、teacher动作模仿、offline RL、GNN/attention或改变几何。

## 项目简介与研究主线

本项目当前研究固定多环终端区内，在随机剩余电量（SOC）和安全间隔约束下，对多 UAV 的向内跨环转移、最后进近及着陆 pad 资源进行联合调度，提高正常着陆效率并控制应急代价。

阶段1–5冻结研究已完成。阶段5存在离线排序信号和部分在线收益，但MLP/GB在增加正常完成时增加一次能源失败，同状态计时没有明显加速，尚未达到不退化系统质量并明显降算的B类门槛，也无长期价值C类证据；不笼统宣称学习完全无效或普遍无必要。保留beamB5/B8/full工程参考与rank/GB实验性结果，下一阶段另定范围。学习只近似当前已到达机群的完整联合批次闭合价值，确定性候选生成/安全/能源/资源/应急层继续负责约束；不自动开展actor/PPO、teacher动作模仿、offline RL、GNN/attention或几何改变。

## 主代码路线

- 新主线为 `terminal_marl/core/fixed_terminal/`、`terminal_marl/envs/fixed_terminal_env.py`、`terminal_marl/configs/fixed_terminal_stage1.py`、`terminal_marl/scripts/evaluate_fixed_terminal_stage1.py` 与 `tests/test_fixed_terminal*.py`。
- 旧 `src/`、动态空域、RL 和 arrival 调度代码仅保留为历史资产，不作为新核心的运行依赖。
- 旧兼容性目标与动态空域优先级不约束当前研究重构；避免为兼容旧训练接口污染新问题定义。

## 文档索引

| 文档 | 路径 | 用途 |
|---|---|---|
| 文档入口 | [docs/README.md](docs/README.md) | 当前文档权威关系与历史边界 |
| 阶段化方案 | [docs/fixed_terminal_reconstruction_plan_stage1.md](docs/fixed_terminal_reconstruction_plan_stage1.md) | 用户指定的研究范围和顺序 |
| 问题定义 | [docs/fixed_terminal_problem_definition.md](docs/fixed_terminal_problem_definition.md) | 几何、运动、容量、能源、资源、应急与指标 |
| 当前状态 | [docs/CURRENT_STATUS.md](docs/CURRENT_STATUS.md) | 实际实现、验证进度与允许的下一步 |
| 研究定位 | [docs/research_positioning.md](docs/research_positioning.md) | 研究问题与结论范围 |
| 需求 | [docs/requirements.md](docs/requirements.md) | 必須满足的模型行为 |
| 设计 | [docs/design_spec.md](docs/design_spec.md) | 独立核心职责与数据流 |
| 技术规范 | [docs/technical_spec.md](docs/technical_spec.md) | 参数、接口、测试与输出 |
| 执行计划 | [docs/execution_plan.md](docs/execution_plan.md) | 阶段4已完成工作与后续范围边界 |
| 阶段2方法 | [docs/arrival_restructure/stage2_traditional_baselines.md](docs/arrival_restructure/stage2_traditional_baselines.md) | 六方法、联合批次、滚动信息边界与 oracle |
| 阶段2协议 | [docs/experiments/fixed_terminal_stage2_protocol.md](docs/experiments/fixed_terminal_stage2_protocol.md) | 冻结预算、162场比较和12个小实例 |
| 阶段5结果 | [docs/evidence/fixed_terminal_stage5/results.md](docs/evidence/fixed_terminal_stage5/results.md) | 离线信号、在线质量代价与同状态计算结论 |
| 阶段5方法 | [docs/arrival_restructure/stage5_learning_validation.md](docs/arrival_restructure/stage5_learning_validation.md) | 联合批次筛选、TOTAL预算与证据关系 |
| 阶段5协议 | [docs/experiments/fixed_terminal_stage5_protocol.md](docs/experiments/fixed_terminal_stage5_protocol.md) | 四模型、冻结数据及质量/计时评价矩阵 |
| 阶段4方法 | [docs/arrival_restructure/stage4_learning_scheduler.md](docs/arrival_restructure/stage4_learning_scheduler.md) | 学习筛选、确定性约束和闭合参考 |
| 阶段4协议 | [docs/experiments/fixed_terminal_stage4_protocol.md](docs/experiments/fixed_terminal_stage4_protocol.md) | 数据分组、训练/筛选预算与完整链评价 |
| 阶段3协议 | [docs/experiments/fixed_terminal_stage3_protocol.md](docs/experiments/fixed_terminal_stage3_protocol.md) | 15context、根池、累计J与配对机制核对 |
| 学习问题定义 | [docs/fixed_terminal_learning_problem_definition.md](docs/fixed_terminal_learning_problem_definition.md) | 已核对的受约束窗口/联合价值及结论边界 |

## 开发原则

1. 固定环数量、半径、高度、节点、路径与 pad；不得根据方法或运行结果改变几何及容量。
2. 调度问题联合包含各环等待、向内跨环、最后进近和 pad 窗口；不能仅实现初始放行排序。
3. 安全间隔、运动学可执行性、能源储备和资源可行性由独立确定性层执行，不能交给未来学习模型。
4. 区分几何容量与运营容量；检查连续轨迹间隔，而非仅采样时刻距离。
5. 按运动模式累计归一化 SOC 消耗；明确其合成研究假设，不宣称实机能耗标定或未建模的加速度动态。
6. 应急只在已定义有限正常可行域为空且应急路径、储备和资源可用时成立。搜索预算耗尽返回未知，不得冒充正常不可行。
7. 外部请求的准入、等待时间和能耗均明确建模，全部请求纳入结局统计，保留失败、应急和截尾。
8. 到达与 SOC 使用独立可复现随机流；SOC 使用拒绝采样截断正态。pilot 前冻结参数，不按结果调场景。
9. 新核心配套针对性测试；先小样本核对输出，再运行场景。需求、语义或接口变化时同步更新文档。
10. 保留旧实验和文档，但不将其结果当作新模型的有效性证据。核心不导入 ROS2/Crazyswarm2。

阶段4完整结果与结论见[阶段4结果与运行证据](docs/evidence/fixed_terminal_stage4/results.md)。

## 近期优先任务

以 [docs/CURRENT_STATUS.md](docs/CURRENT_STATUS.md) 为滚动入口。[阶段5结果](docs/evidence/fixed_terminal_stage5/results.md)记录已完成的研究验收：正式test360与修复后continuation9均完成，四模型hash不变，严格汇总已完成。保留当前研究结论，后续研究另定范围，不自动新增实验、扩模型或重复打包。32上限未实际扩大候选池，continuation九场均仅HOLD、pair_count0，不证明无排序反转或无continuation偏差。固定93维表示、种子、预算和物理保持不变；部署输入不得含未到达draw、seed/profile/method标签或真实结局。阶段1–4历史结论、证据与用户阶段1–5计划保留原样。

## 本地-服务器同步准则

1. 本地工作目录 `D:\7_codespace\terminal` 已通过 Syncthing 与服务器
   `~/projects/terminal` 实时同步。代码、脚本和文档修改只应在本地完成。
2. 不需要、也不应手动把本地文件打包上传到服务器再解压；除非用户明确要求，
   不使用 `scp`、`rsync`、手工 zip/tar 传输源码来更新服务器项目目录。
3. Git 操作只在本地仓库执行；每次修改前检查 `git status`、当前分支和 `git worktree list`，确认实际承载授权分支的 checkout。当前 `codex/arrival-restructure` 分支位于 `D:\7_codespace\terminal`，嵌套 worktree `D:\7_codespace\terminal\.worktrees\arrival-restructure` 为 detached HEAD，不应沿用旧路径假定其仍有当前分支上游。保留历史冻结 checkout 和他人改动；服务器侧仅作为运行环境，不承担源码提交职责。
4. 服务器只通过 `ssh lab-server` 执行运行、测试、评估、查看日志和读取结果等命令。
5. 服务器输出统一写入 `$HOME/training_outputs/terminal`；该目录会通过 Syncthing
   同步到本地 `D:\7_codespace\training_outputs\terminal`。不要把 runs、训练输出或评估
   产物写入 `~/projects/terminal`。
6. 若发现服务器代码尚未同步到本地提交，应先等待或提示用户检查 Syncthing 状态，
   不要自行采用手动打包覆盖服务器代码的方式绕过同步链路。

## 常用命令

```bash
# 运行新增测试
python -m pytest tests/test_fixed_terminal_domain.py tests/test_fixed_terminal_safety.py tests/test_fixed_terminal_env.py -q

# 全仓库历史/兼容性测试（可选，不是阶段1独立入口要求）
python -m pytest tests/ -v

# 查看当前项目结构
find terminal_marl -maxdepth 3 -type f | sort
```

## 长耗时诊断与服务器任务准则

### 预检、并行与启动

1. **先验输出闭环**：对新建或大幅修改的诊断/仿真，提交服务器前必须以小样本或单测验证输入、索引、字段名、聚合逻辑和序列化输出一致；不得把未经输出路径验证的脚本直接投入长运行。
2. **优先多核 CPU 并行**：提交服务器任务前检查可用 CPU、当前负载、内存及脚本已有的并行参数。优先按独立 seed、场景、episode 或数据分片并行，明确设置 worker 数，尽可能使用实际可用的多核资源；不影响其他任务、不超出内存。控制进程数与每进程数值库线程数，避免嵌套并行过度抢占 CPU。单任务串行有必要时，记录具体原因。
3. **并行不改变实验协议**：保持冻结种子、随机流、数据划分、模型预算、几何和物理参数不变。各 worker 独立写入输出，汇总时核对预期任务数、缺失、重复和失败；先用小样本验证并行与串行的任务覆盖和结果语义一致，再启动正式任务。
4. **记录运行与预估**：主对话确定命令、并行度和初始预计总耗时 `T0`，记录估计依据、开始时间、总工作量、日志路径、输出目录和 PID/job ID。无法预估时先用小样本或首次有效进度建立估计，不启动高频轮询。
5. **启动后再汇报**：由 Luna high 检查进程、日志和至少一个预期中间产物确实正常生成，确认后主对话才报告任务已正常启动。命令已提交、进程存在或进度到 100% 都不能单独作为任务成功的证据。

### Luna high 短期监控

1. **查询预算**：正常进度查询间隔约为初始预计总耗时的 `T0 / 4`，力争每个任务的主动进度查询总计不超过四次。启动验证不计入该预算；不要每次都按“剩余耗时 / 4”缩短间隔，造成无限轮询。异常检查和 heartbeat 唤醒后的复查按各自规则执行，不为满足查询次数而忽略失败。
2. **基于有效进度估时**：每次查询记录累计运行时间 `t`、已完成/总工作量、进度比例 `p`、进程状态和最近产物。工作量近似均匀且 `0 < p < 1` 时，预计总耗时 `T = t / p`，预计剩余耗时 `R = T - t`。多阶段或任务耗时不均时用当前阶段吞吐及剩余工作量估计，不把启动阶段或不可比的百分比直接线性外推。
3. **超过一小时立即交接**：初始可靠估计或任何一次有效进度推算得到 `T > 60 min` 时，Luna high 汇报观测证据并结束持续监控，由主对话转入 heartbeat；服务器任务继续运行。若查询预算已用完而任务仍正常运行，也按最新预计剩余耗时转入 heartbeat，不继续驻留轮询。
4. **异常返回主对话**：进程退出、日志报错、预期产物缺失或进度无法可靠估计时，Luna high 返回原始证据，主对话决定检查或处置。`p=0` 时不能使用除法估时；缺少可靠估计时主对话选择并说明有依据的下一次复查时间。

### heartbeat 唤醒与接续

1. **按剩余耗时唤醒**：使用当前对话的 heartbeat 工具设置下一次唤醒，间隔取最新预计剩余耗时 `R`；创建时扣除从估时观测到设置调度已经过去的时间。比如运行 15 min 完成 10%，推算 `T=150 min`、`R=135 min`：结束 Luna 持续监控，约 135 min 后唤醒本对话复查。heartbeat 等待期间不保留一个睡眠中的监控 worker。
2. **保存接续上下文**：heartbeat 提示须包含原任务目标、已授权的后续工作、命令/PID/job ID、服务器/日志/输出路径、最近进度及观测时间、估时依据，并要求唤醒后检查完成状态和产物。通过工具创建或更新，并确认工具成功；复用同一任务的 heartbeat，避免重复调度，不用长时间 sleep 或另建 cron 代替对话 heartbeat。
3. **已完成则继续原任务**：唤醒后核对退出状态、完成标记及预期结果完整性；确认完成后读取结果，由主对话分析、复核并继续原先已授权的工作，不只发送“已完成”通知就停止。任务完成或失败时删除对应 heartbeat，避免后续重复唤醒。
4. **未完成则重估再调度**：任务仍在正常运行时按最新进度重算剩余耗时，更新同一个 heartbeat 的下一次唤醒时间；若工具只支持重复间隔，每次复查后都更新间隔，并在任务终态删除。未变化且无异常时保持静默；失败、异常停滞、缺失预期输出或需要用户决定时及时报告，不反复延后掩盖问题。
5. **工具受限不得假报**：若 heartbeat 不支持所需间隔、更新或自动接续，主对话明确说明实际能力及未安排的部分。不得把创建失败、普通提醒或未验证的调度当作已具备自动接续。

## Git 版本管理准则

1. 每次完成一轮有意义的文件修改或功能迭代后，必须先检查 `git status` 和 `git diff`，确认只包含本轮相关改动。
2. 提交前必须运行与改动范围匹配的测试或检查；若测试无法运行或未通过，必须在交付说明中明确记录，不得伪装为已验证。
3. 使用小而清晰的提交，提交信息应说明改动目的；不得把无关改动混入同一提交。
4. 禁止提交密码、令牌、私钥、个人凭据、大型临时产物或本地环境文件；必要时先更新 `.gitignore`。
5. 不得覆盖或丢弃用户已有的未提交改动。暂存时应显式指定本轮文件，避免直接使用 `git add .`。
6. 每次在本地完成一轮文件修改并通过与改动范围匹配的验证后，必须将本轮相关改动提交并推送到远程 GitHub 仓库 `https://github.com/Michae1JeeSEU/TerminalRL.git` 的当前分支上游（当前重构分支为 `origin/codex/arrival-restructure`）。此项授权已由用户持续授予，后续无需每轮重新询问；不得只停留在本地修改或提交状态。若认证、网络或上游分支配置导致推送失败，必须如实报告失败及原因，不得声称已完成推送。
7. 未经用户明确授权，不得强制推送、改写公共提交历史、删除远程分支或执行其他破坏性 Git 操作。
8. 最终交付时必须报告：修改文件、测试结果、提交哈希，以及是否已成功推送。

推荐流程：

```bash
git status --short
git diff -- <本轮文件>
python -m pytest <相关测试> -v
git add <本轮文件>
git commit -m "<type>: <本轮改动摘要>"
git push
```
