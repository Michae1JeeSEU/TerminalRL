# AGENTS.md

## 项目简介

本项目研究城市低空交通起降场终端区中的动态空域构型调节与多 UAV 降落流协同管理。当前路线已重构为：

```text
Dynamic terminal airspace reconfiguration
+ Multi-UAV landing-flow coordination
+ Safety-filtered executable trajectory generation
+ Crazyflie / Lighthouse / Crazyswarm2 scaled validation
```

本项目不应被实现或描述为“RL 直接控制无人机飞行动作”的底层飞控问题。强化学习应输出空域调节、放行、排序、调速和轨迹意图。真实飞行动作由轨迹生成器、Crazyswarm2 和 Crazyflie 板载控制器执行。

## 论文主线

推荐论文主线：

> 面向城市低空起降场终端区，研究如何根据动态入场交通流实时调节环形终端区空域构型，并协同决策多架 UAV 的等待、切环、最终进近和降落意图，从而在安全间隔和轨迹可执行性约束下提升降落吞吐量、降低等待延误和冲突风险。

核心 gaps：

1. 固定终端区空域结构难以适应动态入场交通流；
2. 空域构型设计与单机降落动作缺少跨层协同；
3. 学习型终端区控制缺少面向真实无人机平台的轨迹可执行约束。

## 主代码路线

- `terminal_marl/` 是后续唯一主开发线。
- `src/` 是 legacy HRL/Gym 原型，仅作参考，不在其中继续新增主要功能。
- 新功能优先放在 `terminal_marl/core/`、`terminal_marl/envs/`、`terminal_marl/hardware/`、`terminal_marl/configs/` 和 `terminal_marl/scripts/`。

## 文档索引

| 文档 | 路径 | 用途 |
|---|---|---|
| 研究定位 | [docs/research_positioning.md](docs/research_positioning.md) | Q1 创新主线、gap、贡献点和实验体系 |
| 需求文档 | [docs/requirements.md](docs/requirements.md) | 重构后的研究目标、动作语义、实验平台适配需求 |
| 设计规范 | [docs/design_spec.md](docs/design_spec.md) | 动态空域、轨迹生成器、安全过滤、Crazyswarm2 adapter 架构 |
| 技术规范 | [docs/technical_spec.md](docs/technical_spec.md) | 依赖、接口、配置、动作/观测、测试和日志规范 |
| 执行计划 | [docs/execution_plan.md](docs/execution_plan.md) | Phase A–J 的开发路线 |
| Codex 任务 | [docs/CODEX_NEXT_STEPS.md](docs/CODEX_NEXT_STEPS.md) | 下一轮 Codex 具体执行任务 |

## 开发原则

1. **主线明确**：只在 `terminal_marl/` 上推进新架构，`src/` 不再作为主开发对象。
2. **创新聚焦**：必须服务于动态空域构型与多 UAV 降落流协同，不要退化成普通多机避障或固定环降落控制。
3. **轨迹意图优先**：低层策略输出 release/speed/timing/slot residual，不直接输出自由 `ψ, γ, V`。
4. **安全优先**：所有推进、切环、final approach 和 landing 必须经过 VTAC/SafetyFilter。
5. **硬件解耦**：`core/` 不 import ROS2/Crazyswarm2；硬件接口放在 `terminal_marl/hardware/` 并采用懒加载。
6. **先仿真后实物**：先完成 rule-only、residual policy 和 dynamic airspace 的仿真/dry-run，再接入真实 Crazyflie。
7. **实验可消融**：必须支持 fixed/dynamic、rule/RL、with/without safety、direct-action baseline 等模式。
8. **兼容旧脚本**：新增 action mode 或 wrapper 时，不应破坏当前已有训练/评估脚本。
9. **测试驱动**：每个新增核心模块必须配套 `tests/`。
10. **文档同步**：需求、架构或接口变化时，同步更新 `docs/`。

## 近期优先任务

按顺序执行：

```text
1. terminal_marl/core/dynamic_airspace.py
2. tests/test_dynamic_airspace.py
3. terminal_marl/core/trajectory_generator.py
4. tests/test_trajectory_generator.py
5. ExperimentMode / fixed_dynamic switches
6. crazyflie_lab 配置骨架
7. SafetyFilter 骨架
8. intention_residual action_mode
```

## 本地-服务器同步准则

1. 本地工作目录 `D:\7_codespace\terminal` 已通过 Syncthing 与服务器
   `~/projects/terminal` 实时同步。代码、脚本和文档修改只应在本地完成。
2. 不需要、也不应手动把本地文件打包上传到服务器再解压；除非用户明确要求，
   不使用 `scp`、`rsync`、手工 zip/tar 传输源码来更新服务器项目目录。
3. Git 操作只在本地仓库执行：检查、暂存、提交和推送均以本地
   `D:\7_codespace\terminal` 为准。服务器侧仅作为运行环境，不承担源码提交职责。
4. 服务器只通过 `ssh lab-server` 执行运行、测试、评估、查看日志和读取结果等命令。
5. 服务器输出统一写入 `$HOME/training_outputs/terminal`；该目录会通过 Syncthing
   同步到本地 `D:\7_codespace\training_outputs\terminal`。不要把 runs、训练输出或评估
   产物写入 `~/projects/terminal`。
6. 若发现服务器代码尚未同步到本地提交，应先等待或提示用户检查 Syncthing 状态，
   不要自行采用手动打包覆盖服务器代码的方式绕过同步链路。

## 常用命令

```bash
# 运行新增测试
python -m pytest tests/test_dynamic_airspace.py tests/test_trajectory_generator.py -v

# 运行全部测试
python -m pytest tests/ -v

# 查看当前项目结构
find terminal_marl -maxdepth 3 -type f | sort
```

## Git 版本管理准则

1. 每次完成一轮有意义的文件修改或功能迭代后，必须先检查 `git status` 和 `git diff`，确认只包含本轮相关改动。
2. 提交前必须运行与改动范围匹配的测试或检查；若测试无法运行或未通过，必须在交付说明中明确记录，不得伪装为已验证。
3. 使用小而清晰的提交，提交信息应说明改动目的；不得把无关改动混入同一提交。
4. 禁止提交密码、令牌、私钥、个人凭据、大型临时产物或本地环境文件；必要时先更新 `.gitignore`。
5. 不得覆盖或丢弃用户已有的未提交改动。暂存时应显式指定本轮文件，避免直接使用 `git add .`。
6. 每轮迭代完成并成功提交后，应推送到当前分支对应的远程上游分支。若尚未配置远程仓库、上游分支或认证失败，应停止推送并明确告知用户所缺信息，不得声称已上传。
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
