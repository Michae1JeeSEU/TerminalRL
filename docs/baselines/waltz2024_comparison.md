# Waltz、Okhrin、Schultz（2024）基线复现与公平比较设计

本文档把 Waltz、Okhrin、Schultz 的 *Self-organized free-flight arrival for urban air mobility* 作为本项目的外部对比基线。论文信息以 [TR Part C DOI 10.1016/j.trc.2024.104806](https://doi.org/10.1016/j.trc.2024.104806)、[arXiv v2](https://arxiv.org/html/2404.03710v2) 和[作者提供的官方 GitHub 仓库](https://github.com/MarWaltz/TUD_RL)为准。

这里需要把两个工作阶段分开：一是按论文和仓库语义完成原始实现的无训练复现，二是将已有 checkpoint 迁移到本项目的共同物理执行后端，作为未来的对比实验。当前文档没有声称复现已经完成，也没有声称任何策略已经优于另一策略。用户要求的当前工作不包含重新训练；任何 adapted/retrained 版本都必须在以后单独授权。

配套目录索引见 [Waltz 2024 baseline README](../../baselines/waltz2024/README.md)。该索引统一指向后续保留的源码快照、环境和场景入口。

## 1. 基线命名与比较范围

| 名称 | 作用 | 可以支持的结论 |
|---|---|---|
| `waltz_original` | 保留论文/仓库原始 decoder、观测、动作和场景语义的复现；只用于 reference validation | 是否能够运行并重现原始语义和公开实验入口 |
| `waltz_zero_shot_common` | 将未来可加载的原始 checkpoint 接入共同物理执行后端；不重新训练 | 在共同场景中的端到端结果，包含原策略与本项目执行层之间的接口差异 |
| `waltz_zero_shot_common_shielded` | 在上一行共同后端上使用同一安全执行层，并单独报告 shield intervention rate | 加入相同执行安全层后的端到端结果，以及安全接管代价 |
| adapted/retrained | 未来若需要改变观测、动作、LZ 数量或场景，再单独注册并授权 | 只能作为另一个明确标记的实验，不得回写成 `waltz_original` |

`waltz_original` 不把 heading 输出映射成 `WAIT/RELEASE`，也不强行塞进本项目的三环构型；否则它就不再是原始基线。原策略与本项目策略具有不同的决策权限时，结果是系统级端到端比较，不能据此证明“意图学习”本身更强。为支持动态空域和调度的因果解释，现有 `fixed_rule`、`fixed_low_rl`、`dynamic_rule`、`dynamic_hrl` 以及强调度对照仍需保留。

当前基线复现不会改变 `PPO_ALLOWED=false`，不会推进 A6 动态空域训练，也不会把真实 Crazyflie/Crazyswarm2 硬件标记为已通过 qualification。

## 2. 原始论文语义的复现边界

`waltz_original` 的第一目标是忠实执行公开代码和论文描述。以下是复现时需要保持的语义清单；数值以论文/仓库核对结果为准，尚未核对出的歧义不能静默改写。

| 项目 | 原始语义 |
|---|---|
| 飞行几何 | 同一高度的 circular free flight；圆形空域外半径 `R=800`，VTOL 半径 `200` |
| 实际动作 | 一个连续标量 heading delta，每 `1 s` 最多 `5°`；不是离散的放行意图 |
| 速度 | `Uniform(10,16)`；波次中的基准速度为 `13` |
| 距离事件 | incident 距离 `100`，collision 距离 `10` |
| 资源阻塞 | 单个 VTOL 被阻塞 `60 s` |
| 选择规则 | 按距中心的最近规则选择，`sigma∈{-1,+1}` |
| actor 输入 | own `3` 项，加上每个邻居 `6` 项；保留 descending distance、`h=2` 等原始语义 |
| 几何约束 | 原始 free-flight 不施加本项目三环 enforcement |

### 2.1 论文参数与公开实现的差异

后续复现需要把论文文字和公开实现逐项对齐。当前已识别的源文件是官方仓库中的 [`tud_rl/envs/_envs/UAM_Modular.py`](https://github.com/MarWaltz/TUD_RL/blob/HEAD/tud_rl/envs/_envs/UAM_Modular.py)；下表只记录待复现时需要保留和核对的静态事实，不是运行结果。

| 项目 | 论文/目标描述 | 公开实现中的源值 | 复现处理 |
|---|---:|---:|---|
| 圆形空域外半径 / source spawn radius | `R=800` | `Destination.spawn_radius=1000` | `waltz_original` 按固定源码复现；论文 `800` 与源码 `1000` 的差异单独记录 |
| observation normalization denominator | `d_scale=1000` | 原生 `_get_state` 以 `spawn_radius` 作为距离分母，当前值为 `1000` | 若物理 spawn radius 改为 `800`，原生 `_get_state` 分母也会变成 `800`，违反论文 `d_scale=1000`；要实现物理 `R=800`，必须把 physical spawn `800` 与 observation denominator `1000` 解耦，两者不需要相同 |
| respawn radius | 未以同一表述给出 | `1200` | 按源码复现并在普通 YAML/config/log 中记录 |
| VTOL radius | `200` | `200` | 保持一致并核对单位 |

因此，“按论文 R800 运行”与“按 GitHub 公共实现运行”是两个需要显式命名的核对对象，不能把单独修改 `spawn_radius` 的结果冒充完整复现。源码的 `spawn_radius=1000` 是出生/观测归一化相关参数，不是硬性的圆形 `R=800` 边界；若共同比较层要使用物理 `R=800`，应保留论文要求的 `d_scale=1000`，显式解耦物理 spawn 与观测分母，并在场景 YAML/config 中冻结；这属于比较场景设定，不应悄悄改写 `waltz_original`。

原始 decoder、own `3` + 每邻居 `6` 的观测、CPA/history 字段都要保留。若迁移时必须改变 normalization，先做维数、字段顺序、单位和范围的 dimensional-consistency check；仅凭 Crazyflie 尺度测试不能声称 EH216 的 zero-shot 成功。不能为了提高结果而盲目改变 `dt`、最大转向或其他动力学限制。

适配器还必须明确记录坐标约定：源实现使用 North/East 坐标，heading `0°` 指向 North 并按顺时针增加；映射到本项目坐标系时，转换公式和单位必须显式实现并测试。history 按 UID 持久化，即使 ordered-neighbor slot 发生变化也不能重置；只有新 vehicle 创建时才初始化 history。缺失邻居继续使用源语义的 NaN padding 和 ghost-neighbor 占位，不能改成未声明的零填充或重新解释 actor 输入。

原始实验入口还需要单独核对下列输入：波次规模为 `4`，发生在 `t=0,30,60`，速度 `13`；15 秒到达间隔、速度 `Uniform(10,16)` 和 `±20°` 扰动；安全 gate 检查距离 `300 m`；噪声等级 `10/20/100`。公开材料的 §6.2 没有单独规定南侧 native Poisson 输入的速度分布和航向扰动。当前本地 `poisson` 扩展明确继承 `study` 的 `Uniform(10,16)` 与 `±20°` 参数，并将这一假设写入配置和日志。

该本地扩展的南侧输入为四个簇，簇起始时刻为 `0/120/240/360 s`；每簇飞机数独立采样 `Poisson(λ=5)`，同簇相邻飞机间隔 `10 s`。非空簇的跨度为 `10×(count−1) s`，空簇不产生飞机；这既不是每簇固定四架，也不是连续时间齐次 Poisson 到达过程，也不声称是已发布的原始 exact generator。

arXiv 文字列出的 finite `N` 为 `{5,10,15,25,30}`，没有列出 `20`；但 `180` 次运行的计数看起来暗示共有六个取值，dev 配置中的 `sim_study_N=20` 也表明 `20` 是受支持的 case。后续协议可以把 `{5,10,15,20,25,30}` 作为显式的六值方案，但不能断言 arXiv 的遗漏已经被证明是排版错误；原始复现仍需把采用的取值和来源写入普通配置/日志，不得静默修正论文文字。

## 3. 共同物理执行层与语义适配

未来共同场景的输入由一份预先生成的、策略无关的 request tape 提供。每条记录至少包含 `id`、`time`、`gate`、`speed`、`SOC` 和 `service type`。所有策略读取同一份 tape，不因策略内部随机数消耗不同而漂移；被阻塞的 gate 不得导致请求丢失或 gate teleport。external queue/admission 规则对所有策略相同；若运行中重新分配 gate，另行记录 `reassigned_gate`，不得覆盖原始请求字段。

所有策略使用相同的 airspace footprint envelope、允许高度区间、车辆动力学、`dt=1 s`、physics substeps、最大速度/转向/加速度约束，并保持 `a_lat=v·ω=v²κ`（`ω` 为角速度、`κ` 为曲率）相关可行性约束。LZ 数量、位置和各 service time 在场景 YAML 中冻结；最终降落使用共同的 landing backend。完成时间取 service release，而不是进入 `R=200` 附近的 touchdown/entry 时刻。

除了动力学和资源，还必须冻结各方法可获得的 observable information 与 prediction budget。原始 actor 没有 SOC、service type 或 multi-LZ 字段；Bridge tier 的充足能量、均匀 `60 s` service 和单 LZ 用于减少任务语义迁移，Research tier 使用共同 guards/scheduler，新增 resource interface 必须标记为 adapted。保留原始 checkpoint 而不重训属于 OOD zero-shot transfer，单凭该结果不能证明同任务 learned-algorithm superiority；不得只给 proposed method 额外 prediction 或 oracle information。

共同后端的当前代码默认值也必须从源码冻结，而不能从旧文档默认值推断：[`terminal_marl/core/config.py`](../../terminal_marl/core/config.py) 的 `SimConfig` 使用三环半径 `600/400/200`、高度 `100/70/40`、`safe_distance=80`、`collision_distance=10` 和 `num_landing_zones=6`；相应的场景样例见 [`terminal_marl/configs/phase_a_fixed_rule.yaml`](../../terminal_marl/configs/phase_a_fixed_rule.yaml)。这些是当前实现事实，不是已经注册的 Waltz 比较协议。正式比较必须把最终采用的半径、高度、LZ 数量/位置、gate 高度、`dt`、physics substeps 和安全阈值写入普通场景 YAML/config/log，避免把 stale requirements 中的 gate height 或其他默认值带入实验。

原始方法是单高度 free-flight，而本项目 proposed method 使用三维环和阶段化高度。即使给两者相同的 altitude envelope，也不等于可控性和 accessible volume 相同；这是 system architecture difference。凡执行条件允许，应分别做 common-altitude 和 common-geometry ablation，并明确标记其适用范围。ablation 不把三环语义强行注入原始方法；两套系统决策权限不同的端到端结果，也不能直接作为 intent-learning superiority 的证据。

原始策略的 `sigma` 仍根据共同执行层提供的可用性和 nearest-distance 规则计算。把一个原始单资源模型推广到多个 LZ、引入 LZ assignment 或其他资源接口时，必须标记为 adapted/generalized 版本；它不再属于 `waltz_original`。同理，原始 heading 动作不转换为 `WAIT/RELEASE`，本项目的 gate、ring 和 high-level intent 也不回填到原始 actor 的输出空间。

共同安全执行层如果会拒绝、截断、延迟或替换模型输出，比较前先报告它的限制和已知接管语义。`waltz_zero_shot_common_shielded` 必须报告 intervention 的次数、比例和原因；不能把 shield 后的轨迹直接当作原策略未经修改的输出。

为了保持资源和能量比较公平，不能给原始方法一个单 pad 场景、给本项目方法六个 pad；也不能让原始方法完全没有 SOC/energy guards、再把本项目的 energy failure 单独计入。EH216 与 Crazyflie 的尺度转换、能量模型和硬件可执行性必须作为共同后端的显式配置，而非结果出来后再调整。

## 4. 两层比较场景

### 4.1 Bridge tier

Bridge tier 只用于连接原始自由飞行语义和共同执行后端，暂不引入场景复杂度差异。两种方法共享：一个 LZ、`60 s` 均匀 service、充足能量、task-neutral 规则、`R=800`、四个 arrival gates，以及相同的动力学和高度预算。一个 LZ 与四个 gates 是同一场景内的资源和入口设置，不是把原始单 VTOL 与本项目多 LZ 直接比较。

### 4.2 Research tier

Research tier 才用于本项目的研究场景。所有方法共享 YAML 中冻结的 LZ 数量、位置、动力学和高度预算；不能预先假定本项目代码中的默认 LZ 数量就是公平值。service 类别及时间按下表设定：

| service type | service time | probability |
|---|---:|---:|
| 类别 1 | `30 s` | `0.45` |
| 类别 2 | `120 s` | `0.25` |
| 类别 3 | `10 s` | `0.10` |
| 类别 4 | `60 s` | `0.20` |

`task_priority_enabled=false`，但 SOC/energy guards 对所有方法共同开启。service type 只提供物理服务时长和兼容性元数据，不转化为任务偏好排序。实际值必须写入场景 YAML/config，并在普通运行日志中保留；不要把当前实现中的未注册默认值当作协议。

负荷率应相对于同一个独立测得的 fixed-rule 参考容量 `μ_ref` 定义，而不能按每个方法自己的容量归一化。`0.5 μ_ref`、`0.8 μ_ref`、`1.1 μ_ref` 可作为 low/medium/overload 的设计建议；`1200 s` 高负荷加 `1200 s` 低负荷、dwell `600 s` 也只是 burst 设计建议。它们不是已经注册的 formal protocol，最终数值须在测试前一次冻结。

## 5. 运行与统计规则

有限 cohort 必须运行到每个请求的 terminal state；未完成请求不能从分母中删除，必须作为 censored/unresolved 明确记录。continuous-flow 实验使用相同的 fixed observation window 和相同 warm-up 规则。吞吐量以每小时 service completions 计，同时报告 backlog change，避免只用完成者的等待时间替代流量结果。

碰撞事件继续按 `10` 报告；另设一个所有策略共用的 separation threshold，在测试前从 `80` 或 `100` 中选定并写进 YAML。安全距离使用 swept trajectory 的最小距离，而不能只比较离散 endpoint。每条结果至少保留以下类别和指标：

| 范畴 | 必须报告 |
|---|---|
| 安全与资源 | collision `10`、共同 separation threshold、forbidden entry、resource conflict、shield intervention 次数/比例/原因 |
| 终态 | `normal`、`emergency`、`failure`、`diversion`、`unresolved/censored` |
| 流量与延误 | 每小时 service completions、backlog change、所有请求的 request-to-service delay、external wait、internal wait、`p95`、最大值、makespan |
| 资源与能量 | service utilization、能量消耗、完成时 SOC、energy guard/diversion 记录 |

配对评估使用相同的 request tape、gate、service metadata 和初始能量抽样；推荐 `30` 个相互独立的 evaluation episodes，以便与论文报告方式对齐。checkpoint 只有单一训练 seed 时，只能报告该 checkpoint 的推理结果，不能据此声称训练 seed 方差。baseline development seeds 与之后的正式注册/held-out 测试分开，不能访问 Q2 test，也不能事后用 Q2 选 checkpoint。

不同环境的 raw reward 不做横向比较；只比较共同定义的物理、资源、延误和安全指标。不得 cherry-pick episode、负荷率、checkpoint 或过滤后的完成者集合。

## 6. 当前状态与后续边界

当前已保留官方仓库的 main 与 dev 源码快照，并完成静态环境/场景入口核对；另行检查了 v1.1 release，但未下载其资产。dev 快照包含 [`UAM_Modular_Validation.py`](../../baselines/waltz2024/upstream_dev/tud_rl/envs/_envs/UAM_Modular_Validation.py) 中的 `entry_check`、`noise` 参数，以及 [`uam_scenario_eval.py`](../../baselines/waltz2024/upstream_dev/tud_rl/common/uam_scenario_eval.py) 场景评估入口。静态搜索没有找到 Poisson 过程生成器，也没有找到 Crazyswarm2 UAM field controller。当前 main/dev 源码及另行检查的 v1.1 release 中没有已确认的 UAM-specific actor checkpoint；用户也确认目前没有可提供的权重，因此 `waltz_zero_shot_common` 和 `waltz_zero_shot_common_shielded` 目前只是冻结的未来命名。

按当前授权，下一步范围先停留在环境和场景语义；Policy IV 仍 blocked。2026-10-02 的有限测试证据（`evidence_type=newly_run`）位于服务器 `/home/user/training_outputs/terminal/waltz2024/verification_20261002`，并同步到本地 `D:/7_codespace/training_outputs/terminal/waltz2024/verification_20261002`；本地汇总校验文件为 [`checks.json`](../../../training_outputs/terminal/waltz2024/verification_20261002/checks.json)，服务器对应 `/home/user/training_outputs/terminal/waltz2024/verification_20261002/checks.json`。`pytest_final.stdout` 记录 17 passed、14 warnings、2.03 s、退出码 0，覆盖前 65 步 wave/study 源码与包装器状态/动力学/history、一组截断 JSON/CSV schema 与调度边界、不兼容 checkpoint 拒绝，以及真实 CLI 缺少 actor 时退出码 2 且不创建输出目录。九个 `diagnostic-zero` case（`wave_upstream`、`study30_off`、`study30_on`、`study5_noise10/20/100`、`poisson_off`、`poisson_on`、`wave_paper800`）也均退出码 0，`complete=true`、`truncated=false`、`active=0`、`unspawned=0`，request/spawn conservation、CSV `(time,uid)` 唯一性和 `sigma∈{-1,+1}` 均通过，并生成各自 chart。`complete=true` 仅表示有限调度已消费且没有 active vehicle，不证明 service completion、landing success、安全性或性能；这些 case 使用 constant-zero heading，输出标记为 `status=runtime_diagnostic`、`policy_verified=false`、`official_policy_claim=false`。例如 `wave_upstream` 为 12 个请求、2 个 VTOL entry 和 10 个 outside removal，`study30_off/on` 均为 30 个 outside removal。未提供权重和所需输入前，不宣称 actor inference、zero-shot 迁移或 reproduction pass；本文档没有新训练命令、没有 checkpoint 结果，也没有任何性能结论。

只有在 `waltz_original` 的运行语义完成核对并取得可用权重后，才可另行准备两个共同后端条目。若需要修改动作、观测、LZ 数量、SOC/energy 语义或训练权重，应新建明确的 adapted/retrained 条目并单独授权；这不属于当前基线复现。该基线工作本身不改变 `PPO_ALLOWED=false`、A6 或 hardware 的资格状态，也不构成“当前研究方法已胜出”的结论。
