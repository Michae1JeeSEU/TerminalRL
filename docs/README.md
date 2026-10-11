# Terminal MARL 文档入口

本目录只维护一套当前有效的项目说明。项目当前研究范围是 **Phase A —
Landing-only terminal-area operations**：动态三环终端空域构型、多 UAV 降落流协同、
安全过滤的可执行轨迹，以及后续 Crazyflie 缩尺验证。强化学习负责交通管理和轨迹意图，
不直接输出姿态、电机或自由飞行动作。

本轮低层主课程采用 task-neutral 语义：正常降落、吞吐、完整周期等待和尾部 starvation 监测，
保留随机到达、SOC 差异、能量与安全约束、环/路径可执行性和资源预约。`mission_type` 与
`base_priority` 接口字段继续保留；`task_priority_enabled=false` 时，base 的有效值统一为
`1`，mission 只表示物理服务/兼容性 metadata，policy mission one-hot 固定为 `STANDARD`，
88D schema 不变。服务时长 `30/120/10/60 s` 保持以维持负荷。能量应急 takeover 始终保留，
与后续独立 emergency mission 鲁棒性场景分开。完整课程评估待运行，状态以
[CURRENT_STATUS](CURRENT_STATUS.md) 为准；旧 source/candidate 与
formal/heldout 记录均按历史协议解释，不能据此宣称稳定收益或自动放行 PPO。

> **2026-10-11 研究重构入口**：当前研究顺序、第一批 bounded traditional joint reference、
> external_dispatch_v1、local reorder、seed/预算和 IL/RL measured-value gate 见
> [research_restructure_plan.md](research_restructure_plan.md)。该计划取代旧的“先取得低层 RL
> 成功再进入其他研究”排序；本页的任务语义、schema、安全和历史证据边界继续有效。第一批
> 标记为 IN_PROGRESS，尚无结果；阶段授权仍由 [CURRENT_STATUS.md](CURRENT_STATUS.md) 决定。

## 2026-09-24 历史集成摘要（不定义当前课程）

历史 opt-in `fixed_rule_dense30_coordinated` 使用 downstream/deadline landing-flow coordinator；
最近五场、`150` UAV、`143 normal / 7 emergency / 0 failed-energy` 记录均为规则控制器，且沿用
历史 `.25` landing floor，不是训练策略验证。轨迹与安全链保留完整动力学和 SafetyFilter 检查，
`validate_plan` 的批量等价路径将 100-step 测量从 `45.319964 s` 降至 `36.037378 s`；该数字是
函数/prefix 证据，不能外推完整仿真吞吐。viewer 已支持 30 点云拾取、30 Hz position-only 更新、
标签缓存和离屏回归；PyVista `0.48.4` / VTK `9.6.2` 的 60-frame benchmark 为 `3.2452403 s`
到 `0.6678959 s`，同样不代表 runtime FPS，专项合计 `35 passed`。

该历史诊断采用 `eh216s_research_v2`、归一化容量 `1500` 与 `min_landing_soc=.10`；
`soc_reserve=.25` 仍是独立的 queue/admission reserve，其它阈值不变。已有 `.25` 实验和 checkpoint
只作历史证据。研究边界仍是 landing-only：RL 输出交通管理/降落意图，不直接控制飞行动作；在低层
当前 `PPO_ALLOWED=false`；完成必要策略修复并通过 PPO admission 后再决定 PPO，低层 static
qualification 后才进入 A6 高层动态空域。

> 当前工程与实验进度请先看 [CURRENT_STATUS.md](CURRENT_STATUS.md)。历史 handoff、
> 单次实验报告和旧版本方案均不再作为实现入口。

## 推荐阅读顺序

### 只需快速接手项目

1. [研究定位](research_positioning.md)：研究问题、创新边界与论文主线；
2. [研究重构计划](research_restructure_plan.md)：当前证据顺序、第一批 pilot 和后续学习/几何路线；
3. [当前状态](CURRENT_STATUS.md)：已冻结 schema、已完成证据、当前阻塞和下一步；
4. [设计规范](design_spec.md)：系统模块、数据流与职责边界；
5. [低层策略专项规范](reference/low_level_policy.md)：当前 V4.2 / Observation V2.1 语义。

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
| [research_restructure_plan.md](research_restructure_plan.md) | 当前研究排序、bounded reference、pilot 与后续 measured-value gate | 当前完成度与 qualification 判定 |
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
| [reference/visualization_console.md](reference/visualization_console.md) | 本地可视化 console、实时/暂停控制、snapshot 回放与单文件离线导出 |
| [literature_and_baseline_review_20261002.md](baselines/literature_and_baseline_review_20261002.md) / [waltz2024_comparison.md](baselines/waltz2024_comparison.md) | 文献与 Waltz 2024 对比基线参考；用于研究基线设计，不代表当前 qualification 状态 |
| [common_arrival_execution_20261003.md](baselines/common_arrival_execution_20261003.md) | 外部基线共同执行桥接、运行与指标说明；三环/自由航向公平性协议及正式实验边界 |

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

本轮研究与 P3 复核的逐项判定、文档修复状态、已存在证据与待运行实验见
[research review repairs 2026-09-30](evidence/v4_2/research_review_repairs_20260930.md)。该页不替代
`CURRENT_STATUS.md`，也不把待验证设计或未运行实验写成 PASS。

## 文档维护规则

1. 研究范围或贡献变化：同步更新研究定位、需求、设计和技术规范；
2. schema 或接口变化：同步更新技术规范、相关专项规范、测试和 `CURRENT_STATUS.md`；
3. 阶段完成、失败或阻塞：新增一份 `evidence/<version>/` 记录，并滚动更新
   `CURRENT_STATUS.md`，不再创建新的日期型 handoff；
4. 单次诊断、预注册 protocol、run summary、manifest 不放在 `docs/` 根目录；
5. 已被替代的方案移动到 `history/`，并在历史索引中注明替代它的当前文档；
6. 所有文档链接使用仓库相对路径，移动文件后必须运行链接检查。


## 2026-10-03 当前修复入口

新P3实现修复及旧R2主动停止见 [统一修复记录](evidence/v4_2/p3_integrated_repair_20261003.md)。上方88D叙述保留为历史课程；新开发profile使用90D并等待重新qualification，当前不允许PPO。
