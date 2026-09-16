# Terminal MARL 文档入口

本目录只维护一套当前有效的项目说明。项目当前研究范围是 **Phase A —
Landing-only terminal-area operations**：动态三环终端空域构型、多 UAV 降落流协同、
安全过滤的可执行轨迹，以及后续 Crazyflie 缩尺验证。强化学习负责交通管理和轨迹意图，
不直接输出姿态、电机或自由飞行动作。

> 当前工程与实验进度请先看 [CURRENT_STATUS.md](CURRENT_STATUS.md)。历史 handoff、
> 单次实验报告和旧版本方案均不再作为实现入口。

## 推荐阅读顺序

### 只需快速接手项目

1. [研究定位](research_positioning.md)：研究问题、创新边界与论文主线；
2. [当前状态](CURRENT_STATUS.md)：已冻结 schema、已完成证据、当前阻塞和下一步；
3. [设计规范](design_spec.md)：系统模块、数据流与职责边界；
4. [低层策略专项规范](reference/low_level_policy.md)：当前 V4.2 / Observation V2.1 语义。

### 需要修改代码或实验

1. [需求文档](requirements.md)：系统必须满足的行为和约束；
2. [设计规范](design_spec.md)：模块边界与核心对象；
3. [技术规范](technical_spec.md)：配置、接口、schema、测试与日志；
4. [执行计划](execution_plan.md)：依赖顺序与阶段 gate；
5. [当前状态](CURRENT_STATUS.md)：确定现在允许做什么、哪些阶段仍被阻塞。

## 当前有效文档

| 文档 | 职责 | 不负责 |
|---|---|---|
| [research_positioning.md](research_positioning.md) | 研究范围、gap、贡献、RQ、论文叙事 | 实现进度 |
| [requirements.md](requirements.md) | 不可变业务语义与系统需求 | 文件级实现 |
| [design_spec.md](design_spec.md) | 架构、模块职责、核心数据流 | 历史迁移 |
| [technical_spec.md](technical_spec.md) | 当前 schema、API、配置、日志、测试 | 实验结果判定 |
| [execution_plan.md](execution_plan.md) | 阶段依赖、gate、整体开发顺序 | 当前完成度 |
| [CURRENT_STATUS.md](CURRENT_STATUS.md) | 当前完成度、阻塞、最近正式证据、下一步 | 改写规范 |

发生冲突时，先按“研究定位 → 需求 → 设计 → 技术”的顺序确定意图；
`CURRENT_STATUS.md` 只决定当前阶段和授权边界。若冲突仍无法消除，应先同步修订相关主规范，
不得用 `evidence/` 或 `history/` 中的旧文本覆盖当前规范。

## 专项参考

| 文档 | 内容 |
|---|---|
| [reference/low_level_policy.md](reference/low_level_policy.md) | V4.2 hierarchical action、88D Observation V2.1、Teacher/Execution/Safety 边界与训练 gate |
| [reference/reconfiguration_high_level.md](reference/reconfiguration_high_level.md) | 动态重构状态机、affected-set freeze、高层 reward 与 qualification |
| [reference/vehicle_dynamics.md](reference/vehicle_dynamics.md) | EH216-S 性能约束动力学与可执行轨迹 |

专项参考必须服从根目录主规范。它们用于补充细节，不建立第二套项目语义。

## 证据与历史材料

```text
docs/
├── evidence/          # 正式 protocol、qualification、diagnostic、manifest
│   ├── dynamics/
│   ├── pre_v4/
│   ├── v4/
│   ├── v4_1/
│   └── v4_2/
└── history/           # 已替代的计划、迁移说明和日期型 handoff
    ├── handoffs/
    ├── migrations/
    └── plans/          # 包含已完成/被替代的可视化开发总领等过程计划
```

- `evidence/` 中的文档是不可追溯改写的实验记录。它们可支持结论，但不自动升级当前状态；
- `history/` 仅用于解释设计如何演化，禁止作为新实现或新训练的依据；
- 旧 checkpoint、旧 action/observation schema 和已消费 qualification cohort 只能按各自记录中的限制使用。

索引见 [evidence/README.md](evidence/README.md) 与
[history/README.md](history/README.md)。

## 文档维护规则

1. 研究范围或贡献变化：同步更新研究定位、需求、设计和技术规范；
2. schema 或接口变化：同步更新技术规范、相关专项规范、测试和 `CURRENT_STATUS.md`；
3. 阶段完成、失败或阻塞：新增一份 `evidence/<version>/` 记录，并滚动更新
   `CURRENT_STATUS.md`，不再创建新的日期型 handoff；
4. 单次诊断、预注册 protocol、run summary、manifest 不放在 `docs/` 根目录；
5. 已被替代的方案移动到 `history/`，并在历史索引中注明替代它的当前文档；
6. 所有文档链接使用仓库相对路径，移动文件后必须运行链接检查。
