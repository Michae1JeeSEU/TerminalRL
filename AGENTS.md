# AGENTS.md

## 项目简介与研究主线

本项目当前研究固定多环终端区内，在随机剩余电量（SOC）和安全间隔约束下，对多 UAV 的向内跨环转移、最后进近及着陆 pad 资源进行联合调度，提高正常着陆效率并控制应急代价。

阶段1已完成固定问题合同与内部验证；用户已授权阶段2传统调度比较。已完成 FCFS、SOC、能源余量、deadline余量、联合窗口和滚动优化的162场冻结比较，以及12个小规模有限目录 MILP参考。固定物理合同不变，不引入 RL、IL、teacher、网络或实机实验。阶段2主审结论为学习必要性尚未建立，不自动进入学习研究。

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
| 执行计划 | [docs/execution_plan.md](docs/execution_plan.md) | 阶段2实现、预检、比较与审查 |
| 阶段2方法 | [docs/arrival_restructure/stage2_traditional_baselines.md](docs/arrival_restructure/stage2_traditional_baselines.md) | 六方法、联合批次、滚动信息边界与 oracle |
| 阶段2协议 | [docs/experiments/fixed_terminal_stage2_protocol.md](docs/experiments/fixed_terminal_stage2_protocol.md) | 冻结预算、162场比较和12个小实例 |

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

## 近期优先任务

以 [docs/CURRENT_STATUS.md](docs/CURRENT_STATUS.md) 为唯一滚动进度入口。阶段2六方法、原子联合可行性、有限目录oracle、47项测试及162+12冻结实验已完成，结果见 [阶段2证据](docs/evidence/fixed_terminal_stage2/results.md)。本轮未建立学习必要性证据，保留负结果与完整交付，不启动第三阶段训练。方法不得读取尚未到达请求；已承诺完整计划保持不变。阶段1证据保留，不恢复旧 I0/I1/I2、PPO、M1–M4 或动态几何路线。

## 本地-服务器同步准则

1. 本地工作目录 `D:\7_codespace\terminal` 已通过 Syncthing 与服务器
   `~/projects/terminal` 实时同步。代码、脚本和文档修改只应在本地完成。
2. 不需要、也不应手动把本地文件打包上传到服务器再解压；除非用户明确要求，
   不使用 `scp`、`rsync`、手工 zip/tar 传输源码来更新服务器项目目录。
3. Git 操作只在本地仓库执行：当前授权的重构分支以本地嵌套 worktree
   `D:\7_codespace\terminal\.worktrees\arrival-restructure` 为 Git 路径；原始冻结 checkout
   保持不动。服务器侧仅作为运行环境，不承担源码提交职责。
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

1. **先验输出闭环**：对新建或大幅修改的诊断/仿真，提交服务器前必须以小样本或单测验证输入、索引、字段名、聚合逻辑和序列化输出一致；不得把未经输出路径验证的脚本直接投入长运行。
2. **启动后再汇报**：服务器任务启动后，先等待并检查进程、日志和至少一个预期中间产物确实正常生成；不能只因命令已提交就宣称任务正常或结束对该任务的跟进。
3. **自适应监控**：预计耗时较长的服务器任务应自动创建 heartbeat 监控。检查间隔应按预计运行时间设定，避免过密或过久；若实际耗时明显偏离预估，应及时调整间隔。未变化时静默，完成、失败、缺失预期输出或需要用户决定时立即报告，并在任务终态删除 heartbeat。

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
