# CLAUDE.md

## 项目简介

低空交通终端区降落：动态空域构型与 UAV 控制协同的层次强化学习（HRL）研究项目。

## 文档索引

| 文档 | 路径 | 用途 |
|------|------|------|
| 需求文档 | [docs/requirements.md](docs/requirements.md) | 项目需求、空域结构、动作空间、目标约束 |
| 技术规范 | [docs/technical_spec.md](docs/technical_spec.md) | 技术选型、依赖、代码规范、接口约定 |
| 设计规范 | [docs/design_spec.md](docs/design_spec.md) | 系统架构、模块设计、接口定义 |
| 执行计划 | [docs/execution_plan.md](docs/execution_plan.md) | 分阶段任务拆解、里程碑、验收标准 |

## 开发日志

- 目录：[dev_logs/](dev_logs/)
- 命名规则：`YYYY-MM-DD.md`
- 每次会话结束时，自动更新当日日志，记录：完成事项 / 待办事项 / 遇到的问题 / 明日计划

## 工作原则

1. **增量推进**：严格按 [execution_plan.md](docs/execution_plan.md) 逐 Phase 执行，一个 Phase 完成确认后再启动下一个
2. **遇问题即停**：设计方案或实现过程中遇到不确定的点，先向用户提问，不自行猜测
3. **先设计后编码**：每个 Phase 开始前先明确输入/输出/完成标准
4. **稳定优先**：不追求一次性完成大量功能，确保每一步稳定可运行
5. **文档同步**：需求或设计变更时，同步更新 docs/ 下的对应文档
6. **Git 版本管理**：每次成功的修改或每个 Phase 完成后，自动执行 git commit，记录版本差异。提交信息需清晰描述变更内容、原因及影响范围，保持原子化提交（一个逻辑变更一个 commit）

## 常用操作

```bash
# Python 环境：项目 venv
./venv/Scripts/python.exe -m pytest tests/ -v
./venv/Scripts/python.exe -m src.train
./venv/Scripts/python.exe -m src.train_ppo

# 代码格式化
./venv/Scripts/python.exe -m black src/ -q
./venv/Scripts/python.exe -m isort src/ -q

# 启动 TensorBoard
./venv/Scripts/tensorboard.exe --logdir logs/
```
