# 2026-10-18 准入等待、pad 到达与回放显示修订

用户授权两项研究规则变更：外环有容量但暂时找不到安全入场轨迹时，在终端区外等待择机入场，只有容量不足才分流；应急与普通进近采用一致的 pad 到达口径，可以在既有服务结束前出发，在资源释放后进入受保护下降区间。新策略标识为 `upstream_capacity_only_pad_arrival_v1_20261018`。这两项明确授权优先于旧合同中相冲突的执行口径，`new10.16/` 冻结合同、YAML、94 项目录及旧结果不覆写。

本记录区分旧回放事实、授权后的模型定义和新实现的有界反事实结果。旧 f67 的测试、benchmark、traffic 与可视化记录继续按原版本解释；它们不能转记为新规则通过。新规则没有据此获得 94 项全通过或 `STAGE1_ACCEPTED`。

## 旧实际观察

来源为旧 `traffic_rate180_seed11` 的真实归档，其 measured commit 为 `f67b5c8462433ba701191c493a37e4a8e472334e`、源码集合 hash 为 `110be32211d37e82772d806d86ae599d0a58a32e01af3465be836a58de902750`。只读提取见 [facts.json](../../artifacts/rotating_slots/user_observations_20261018/facts.json)，旧规则原状态重建见 [probe.json](../../artifacts/rotating_slots/user_observations_20261018/probe.json)，静态执行口径见 [code_facts.md](../../artifacts/rotating_slots/user_observations_20261018/code_facts.md)。这些文件保留各自观测时点；code_facts 中当时的磁盘 parity 不代表修改后的当前磁盘。

- `req_000001` 在 `16.25545875046598 s` 的 7 个实际入场候选均因与已在入场的 `req_000000` 冲突而记录 `HARD_SEPARATION`。这不是外环 37 槽容量已满的证据。
- `req_000008` 在 `102.49614261797981 s` 的前 6 个候选因与 `req_000003` / `req_000005` 的交通冲突被拒，最后 slot 19 候选的单机 freeflow 参考筛查又超出累计峰值 120 s。旧 reroute 最终仅保留最后一个 `INITIAL_ENERGY_SCREEN`，掩盖了前面的实际拒绝类别；不能用这个最终 reason 代替完整候选序列。
- `540 s` 时 P5 的 `req_000003` 服务尚未结束，释放时刻为 `544.2753377237965 s`。旧应急规则要求出发时 pad 当前为空，`req_000009` / `req_000010` 的 P5 候选因此被 `PAD_NOT_CURRENTLY_FREE` 拒绝。该拒绝发生在资源/后续检查之前，不能把它解释为新到达口径下也必然不可行。
- 普通 `req_000026`（旧 viewer 的 UID26）在 `540 s` 已真实启动 `FINAL_NORMAL`；`580 s` 开始垂降、`593.5 s` 接地、`653.5 s` 服务结束。其 P5 committed 日历区间为 `[580, 653.5)`，轨迹 hash 为 `b2be8deca9e8923a25732008fa50b16f3d68c206f0351cfd5ab7693b6921e9ab`。`550 s` 地面没有服务飞机，不等于 P5 没有已承诺的未来资源；UID26 此时已开始飞行，不能按“未开始 soft”撤销。

旧 controller 仅用 `reservation.start <= t` 决定是否显示预约，因而将 550 s P5 错标为 FREE。显示已修为：尚未结束的未来 committed 预约也显示 RESERVED；同时分别给出实际 flight/maneuver start、下降、接地和 pad 占用/服务结束时间。580 s 是下降/资源区间开始，不是该航段起飞时刻。当前 service 的 owner/倒计时与后续预约分别显示，不以未来 owner 的区间终点冒充当前服务结束。SOFT 明确标为未开始、可移除；slot 保留当前 OCCUPIED/DEPARTING 的优先显示并提供未来区间。旧归档不因显示修复而被重写。

## 终端区外等待的模型定义

上游队列表示尚未进入终端区的外部请求，不添加物理飞机，不生成入口悬停轨迹，也不把队列放入终端区成对间隔或能量积分。请求的 `initial_soc` 是其实际跨入边界时的 SOC；等待不重新抽样 SOC、gate、服务类别或车辆配置，也不虚构模型外飞行/充电能耗。等待计入系统延迟。

同 gate 请求遵循 FIFO，在事件触发及 5 s retry 时机重新检查。只有能够给出容量不足依据时才 reroute：37 个外环槽位均有有效永久占用，且没有已知释放。存在有限未来释放不能直接判作满；空位存在而连续间隔、运动、能量或候选搜索暂时不通过时保留上游等待。`NOT_FOUND` 仅表示本次候选搜索耗尽，不能充当全局无解证明。

新 snapshot/root/metrics 以 `policy_id` 区分口径。snapshot 的 `upstream` 只含已 revealed 且未解决的请求，每项保留 `id`、`release_time`、`gate_id`、`initial_soc`、`service_class`、`vehicle_profile_id`、`status`（`WAITING` 或 `CENSORED_UPSTREAM`）、`wait_seconds`、`reason`、`reason_categories`、`last_attempt_time`。实际 Aircraft 才有 `admission_time` / `upstream_wait_seconds`；回放仅将上游请求数、等待和原因显示为外部需求，不把它们映射为物理 UAV 或新 UID。

新守恒口径为：

```text
offered = admitted + rerouted + upstream_waiting
```

`upstream_waiting` 表示当前未解决数；物理失败后仍纳入上述守恒，并另外记录 `censored_upstream`。未释放需求单列，不混入 offered。`upstream_wait_time` 累计已解决及未解决请求的等待，`cost_upstream_wait` 同值并加入延迟成本；仍有未决队列或失败截尾时，累计成本保留 lower-bound 含义。不能把边界 SOC 不变误写成已经计算过模型外等待能耗。

上游等待不设强制超时，不因等待较久自动 reroute。到达窗口加 drain 结束时仍未解决的请求继续保留为 waiting；发生物理失败时另标 `CENSORED_UPSTREAM`。两者都不能计作已入场、已落地或已清空，未决等待及对应成本仍保留下界含义。

## pad 到达、承诺与应急保护

普通和应急进近均可在服务结束前出发，但其 pad 保护区间从下降开始，到接地后的服务结束。下降开始必须不早于资源释放，不能仅保证 touchdown 在释放后而允许两机在保护区内重叠。连续 80 m 中心间隔、运动/功率/能量约束、SOC 与累计峰值预算仍须通过。

应急只可撤销尚未实际开始且与其保护区间重叠的 soft 计划；未重叠的未来预约不因同一 pad 就一并清空。实际已开始的正常或应急 flight 及其 committed 依赖受到保护，不为应急反事实重新改写轨迹。旧“必须等 pad 先空闲才能出发”的 deadline 不可行证明不再适用于新到达口径；取消沿用该证明，不能以 `NOT_FOUND` 补作全局不可行结论。

## 新口径的有界反事实证据

[counterfactual.json](../../artifacts/rotating_slots/user_observations_20261018/counterfactual.json)明确记录 `formal_evidence=false`，对比旧 measured SHA `110be32211d37e82772d806d86ae599d0a58a32e01af3465be836a58de902750` 与本次查询实现 SHA `1b43e1437b3bf4b11a740e442fc97a3fedddb8d58b56b5912b62081db11fd8e2`，新 policy 为上述 Oct18 标识。它从精确旧状态独立查询各 pad，不是 checkpoint 恢复或完整新策略 episode。7 个已核原归档成员的 bytes/hash 保持不变；旧规则精确复现仍在独立的 probe.json 中保存。

| 查询 | 实际结果与适用范围 |
|---|---|
| 16.25545875046598 s / req_000001 | 新规则返回 `WAITING`；仅证明该次查询保留等待，未证明以后一定入场。 |
| 102.49614261797981 s / req_000008 | 新规则返回 `WAITING`；候选原因保留，未证明以后一定入场。 |
| 540 s 决策前 / UID9 / P5 | 仅回退同刻 UID26 的正常启动以重建应急优先的决策前状态，独立查询为 `ACCEPTED`；start 540、descent 575、touchdown 588.5 s。预测接地 SOC `0.272295717182477`，累计 peak `119.38814771473415 s < 120 s`；记录的 continuous checks 均 `SAFE`。这是该候选可接受的证据，不是整集救援成功。 |
| 540 s 决策前 / UID10 / P5 | `PEAK_BUDGET`，累计 peak `128.3330148717713 s`；其他 pad 仍受峰值、资源或 SOC 条件约束。没有 UID10 已救回的结论。 |
| 550 s / UID9 / P5 | `RESOURCE_CONFLICT`；UID26 已开始的 committed `[580,653.5)` 预约保留。没有撤销已开始的正常航段。 |
| 550 s / UID10 / P5 | 仍为 `PEAK_BUDGET`，累计 peak `124.13695799916502 s`。 |

这些独立 pad 查询不是可同时执行的联合方案，没有新 episode 的总体 landing、失败率、延迟改善或吞吐结论。下节补充定向执行与受控小回放证据；仍不能从旧 traffic 结局或上述单点 ACCEPTED 外推新高密度 traffic 的结果。

## 最终有效准则执行与受控小回放

[最终八项结果](../../artifacts/rotating_slots/upstream_policy_20261018/qualification_changed_amended/results.json)实际执行 C03、C04、C05、C12、C13、F08、F09、G04，记录 **8 PASS、86 NOT_RUN**；所选 8 项 coverage 的 missing、duplicates、unexpected、failures、pending 均为空。[核对记录](../../artifacts/rotating_slots/upstream_policy_20261018/qualification_amended_checks.json)确认 10 个真实 ReplayBundle 均验证通过。统一 policy 为 `upstream_capacity_only_pad_arrival_v1_20261018`，源码集合 SHA 为 `a29add1b8daf01a24438887c0a0a320b7de78015fed9f43c2a3c25d3b39d8eaa`；`code_commit=POLICY_AMENDMENT_WORKTREE_20261018` 是工作树运行标签，不冒充 Git 提交。`stage1_status=NOT_ACCEPTED`，其余 86 项没有在本次执行。先前 `qualification_changed` 的 8 PASS 是旧 expected 元数据下的 preliminary 输出，不作为本次有效准则的最终证据，也不覆写原结果。

[交付回放索引](../../artifacts/rotating_slots/upstream_policy_20261018/delivery_replay_inventory.json)进一步区分物理规模与时长：最终同 SHA 的 F08 为 6 架初始地面服务机＋1 架 rescue，0 s 出发、0.35 s 原服务全部释放、130 s 下降、143.5 s 接地，记录推进约 0→250 s；其 peak airborne 为 1。C03/C04 虽分别记录 37/38 架物理 airborne，但都是 0 s 单帧 fixture，不能称为已运行的动态复杂交通。这些包与下面最多 4 架 airborne 的小组合示范各自保留用途和边界。

| 定向检查 | 实际记录与范围 |
|---|---|
| 当前 G09 + 旧 replay + 新 reservation/HUD | [统一日志](../../artifacts/rotating_slots/user_observations_20261018/pytest_g09_viewer_final.log)：62 PASS、112.20 s；[meta](../../artifacts/rotating_slots/user_observations_20261018/pytest_g09_viewer_final.meta.json)为 exit 0、elapsed 112.898 s、timed_out=false。 |
| world / audit / upstream 主套件 | [日志](../../artifacts/rotating_slots/upstream_policy_20261018/pytest_core_final.log)：74 PASS、182.09 s；之后仍有少量返回值、实际准入 source 时间及 metadata 修改，不能称此 74 项在最终版本全部重跑。 |
| 后续五个边界分支 | [pytest_metadata.log](../../artifacts/rotating_slots/upstream_policy_20261018/pytest_metadata.log)：5 PASS、19.96 s，覆盖随后修改的相关分支。 |
| 有效准则 metadata | [pytest_criteria.log](../../artifacts/rotating_slots/upstream_policy_20261018/pytest_criteria.log)：1 PASS、1.60 s。 |

这些是各自范围的真实执行记录，不将不同轮次相加成一套最终全量 PASS。

[小组合示范汇总](../../artifacts/rotating_slots/upstream_policy_20261018/small_multi_aircraft_summary.json)、[实际日志](../../artifacts/rotating_slots/upstream_policy_20261018/small_multi_aircraft.log)与[回放审计](../../artifacts/rotating_slots/upstream_policy_20261018/small_multi_aircraft_audit.json)已真实生成。示范初始放置 3 个 fixture（P0 地面服务、inner rescue、middle hold），再加入 2 个同 gate 请求；仅运行 admission callback，抑制普通逐环转移，以隔离展示上游等待与服务结束前应急出发。最多 4 架同时 airborne，仿真推进到 `143.7 s`，记录 wall `728.7209350001067 s`。此前约 10 min 是返回检查阈值，任务随后正常达到该示范终态并自然退出（exit 0），未取消。

第二请求 `entry_waiting` 在 `0.1 s` 释放、`5.0 s` 实际入场，累计上游等待 `4.9 s`，rerouted 为 0；等待期间没有模型外物理飞机或能耗轨迹。`inner_rescue` 在 `0 s` 出发，`130 s` 开始下降、`143.5 s` 接地，接地 SOC 为 `0.12093874485489294`，低于正常储备单独记录。P0 初始服务仍完整执行至 `10 s`，没有缩短服务。审计为 PASS、0 issues、连续间隔 `CERTIFIED`，记录 16 个连续 pair runs、3 个守恒等式均 true。物理 episode 为 `IN_PROGRESS`，其余 3 架仍在空中，不把审计 PASS 写作全部任务完成。

此次实际运行的是输出目录内 ignored helper `make_small_replay.py`，不是后来保留的[版本化复现入口](../../scripts/diagnostics/build_policy_amendment_replay.py)已执行。[运行中来源说明](../../artifacts/rotating_slots/upstream_policy_20261018/inflight_small_fixture_context.json)明确启动时未捕获 source hashes，运行期间发生了四项小修改；不能把归档末端取得的源码 hash 当作启动版本证明。该包是具名受控轨迹示范；最终固定实现的执行证据来自上述有效准则 8 cases 及受影响边界测试。它不是新高密度全 traffic、正式 benchmark、完整冻结矩阵或吞吐测量。

在既有 PyVista 平台打开[小回放目录](../../artifacts/rotating_slots/upstream_policy_20261018/small_multi_aircraft_replay)，可依次观察 `0.1 / 5 / 10 / 130 / 143.5 s`：外部请求等待、实际入场、原服务释放、下降与接地。使用下面命令只打开观察窗口；本文没有记录该新包的负责人最终认可：

```powershell
& 'D:\7_codespace\terminal\venv\Scripts\python.exe' -m terminal_marl.scripts.open_stage1_replay 'D:\7_codespace\terminal\artifacts\rotating_slots\upstream_policy_20261018\small_multi_aircraft_replay' --time 0.1
```

主对话已实际激活新窗口 `36704010`（Terminal MARL 3D Viewer）并查看 `t=0.100 s`：AMENDED 标识可见，physical display aircraft=4（3 架 airborne＋1 架 ground）；P0 同时显示 SERVICE UID4、end 10 / remaining 9.9，以及 COMMITTED UID2、flight start 0 / descent 130 / touchdown 143.5 / service end 263.5。上游 unresolved=1，`entry_waiting` 为 WAITING、reason `NO_PLAN_FOUND`，latest diagnostic 为 `HARD_SEPARATION`；等待请求没有被画成物理飞机。这是 0.1 s 视图的实际观察，未据此声明已看过后续关键时刻或完成 V12。上方 HUD 仍遮挡部分大图，保留布局限制。使用 `p` / `n` / `d` 等字母快捷键前切换英文输入法；本次 `n` 曾被中文输入法 composition 拦截，已用 Escape 清除，未为此修改源码。

## G09、V06/V12 与后台旧任务的证据边界

2026-10-18 的[独立 G09 结果](../../../training_outputs/terminal/rotating_slots_g09_f67b5c8_20261018/results.json)实际验证旧 40 个 benchmark，8 组各 5 次；G09 唯一执行行 `PASS`、退出码 0、0 新仿真，原 catalogue fixture 未执行。被测 f67 身份与验证器当前环境分别记录，`current_source_matches_measured_source=false`。这项工作已完成，不等待第 61 个 job；计时汇总和来源见[诊断报告](diagnostics_20261016.md#2026-10-18-独立-g09-既有-benchmark-验证)。无绝对吞吐通过门槛，也不认证新 policy、稳态、实时性能或新吞吐。

旧 [92 PASS partial](../../artifacts/rotating_slots/qualification_f67b5c8_partial_20261016/results.json)原件保持不变。旧 V06 自动 PASS 未覆盖本次未来 committed 预约显示，不能保留为新规则通过的依据。V12 是跨 V01–V11 的负责人实际观察，不等同于单一 traffic 场景；复杂 traffic 是补充观察。用户已经提出具体问题，V12 仍为 `HUMAN_PENDING`，不由自动测试、counterfactual、截图或 G09 代签。

旧目录中部分 expected 文本与新规则相反，尤其 F08/F09 的旧准入/应急资源口径。最终 qualification 执行路径对 C05、C13、F08、F09 使用明确的 Oct18 有效准则，结果同时保留 `legacy_catalogue_title`、`legacy_catalogue_fixture_requirement`、`legacy_catalogue_expected`、新 `policy_id` 与 `effective_criteria_source`；冻结 catalogue 文件及原 `catalogue()` 读取保持不变。这些 fixture 的最终 PASS 表示执行了标明来源的有效新准则，不能写成旧 expected 文本逐字通过。独立 G09 仍只验证旧测量来源。

主对话已在窗口 `48040676` 实际查看更新 viewer 的旧 traffic 550 s：time、OLD MODEL、current minimum，以及 P5 RESERVED UID26、flight start 540、descent 580、touchdown 593.5、service end 653.5 均可见，顶部文字未再裁切。默认文字仍遮住部分上方 Gate/outer 图形；这次观察确认核心信息可以读取，不表示布局已无问题，也不构成跨 V01–V11 的 V12 最终认可。原第一布局窗口 `18875348` 未关闭；上述观察针对更新窗口，不能据此把旧布局或新小回放的人工观察标为通过。

旧 f67 `traffic_rate180_seed71` 的 20-inner-worker 后台任务仅补旧矩阵。03:32 UTC 的[直接快照](../../artifacts/rotating_slots/user_observations_20261018/live_frozen_pre_edit/snapshot.stdout)记录父进程 PID6275 与 20 个 worker 存续，progress 395 s、较晚 states 419.8 s，未有完整结果。旧进程及这些 worker 在修改前已载入 f67 模块；当前磁盘已修改，不能再声称本轮磁盘 f67 parity。以后若 PID 或 worker 变化，需要核查实际载入来源；不把旧模块继续运行写成新 policy 已开始正式测量。
