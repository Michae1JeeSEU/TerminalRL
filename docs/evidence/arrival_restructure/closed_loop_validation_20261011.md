# Arrival closed-loop validation：2026-10-11 协议与终态证据（V1）

状态：`COMPLETED`；本轮 M4 measured-value decision 已记录。本文保留第二批 smoke 与 suite 的
冻结范围、边界和输出语义；smoke 仍是 `2 ticks` 的 `CENSORED/tick_cap` replay，不能写成
full-cycle completion。第一批结果见
[first_batch_20261011.md](first_batch_20261011.md)，当前阶段见
[CURRENT_STATUS.md](../../CURRENT_STATUS.md)。

## 版本与范围

第一批已完成。其实现 source commit 为
`59e3f24652c7480087b8110a832105bbd010bea1`，文档提交为
`84a2f07e399626cf3214e334556aa0a4caca2ad5`。第二批 source/protocol commit 为
`3e61fb5611fa11a5673363ad08253fb6b7e20238`，已推送并完成 remote verification；本记录的
registration 在该提交中冻结，实际运行的 `source_commit` 使用该冻结值。

本批保留三个既有场景族 `flow6_mixed`、`dense24_fixed`、`dense24_mixed`，但使用 fresh
demand/SOC realizations。geometry、physics、safety、K3 和 service 不变。所有 controller
都在同一 `external_dispatch_v1` authority 下执行：`downstream_soc_external`（new rule）、
`energy_slack` 和 `rolling_reorder`。它们共用 prestep boundary；同 tick 新 admitted UID
由既有 interface 追加。legacy 在 admission 后计算，因此本批不作 legacy stepwise
equivalence assertion。

`downstream_soc_external` 的 prestep active-UAV 顺序键为
`(ring_rank, SOC, -dynamic_priority, -total_wait, UID)`，其中 `ring_rank` 为
`inner=0, middle=1, outer=2, other=3`。

该 new rule 不使用 energy budget 或 rollout，`nominal_requests` 保持 K3。rolling search
冻结为 `lookahead=4`、`max_orders=4`、`replan_interval=10`、`planning_wall_seconds=60`；
60 s 是 cooperative planning 上限，不是 hard realtime 保证。

## 预注册 arms、seed 与边界

Smoke 只有三个 new-rule arms，每 arm 固定 `2 ticks`，保留 `CENSORED` 与 `stop_reason=tick_cap`，
并要求 `3` 次 independent replay 均为 `MATCH`、`matched_steps=2` 且 `integrity_ok=true`：

| 场景 | controller | seed |
|---|---|---:|
| `flow6_mixed` | `downstream_soc_external` | `896191` |
| `dense24_fixed` | `downstream_soc_external` | `896192` |
| `dense24_mixed` | `downstream_soc_external` | `896193` |

Suite 共 `18` arms，分为 3 个 bridge 与 15 个 fresh arms：

| 部分 | 场景/控制器 | seed 或 cohort | 解释 |
|---|---|---|---|
| bridge | 三场景 / `downstream_soc_external` | `896101`, `896102`, `896103` | prior seeds；只作 descriptive old-legacy comparison，不作 independent evidence |
| fresh | `flow6_mixed` / 三 controllers | `896201` | 一个 common-seed paired cohort |
| fresh | `dense24_fixed` / 三 controllers | `896211`, `896212` | 两个 common-seed paired cohorts |
| fresh | `dense24_mixed` / 三 controllers | `896221`, `896222` | 两个 common-seed paired cohorts |

每 arm 的停止边界为 `2400 ticks / 7200 s wall` 与 unchanged native cap 三者先到者；未完整
闭合时保留 `CENSORED/unknown`。Suite 最多 `18` workers，禁止 nested pool，BLAS threads 为
`1`。Smoke 使用 `3` workers。所有 seeds 都是 development，永远不进入 Q；
`PPO_ALLOWED=false`，本批不做 RL、IL 或 geometry experiment。

预先禁止增加 seed、替换 arm、事后 tuning 或根据结果重新登记；失败和 `CENSORED` 保留在
最终报告中，不改写为完成。

## 控制、桥接与评估字段

三个 external controller 共享同一 prestep boundary、external authority、nominal K3、
native scheduler slot、committed flight、reservation、trajectory、安全和 emergency 语义。
新 admitted UID 通过现有接口追加。legacy 在 admission 后计算，故 bridge 只提供旧 legacy
的 descriptive comparison，不提出 stepwise equivalence assertion。

每个 arm 逐项记录：

- normal、emergency、fail、divert、known_unresolved（`U`）与 arrival integrity；
- `not_yet_requested` 单独记录，不并入 `U` 或 unrequested 汇总；
- collisions、NMAC、hard-separation violations、checked/executed trajectory hash mismatch；
- overall-request mean、p90、max wait，makespan，energy，min landing SOC；
- planning、native、wall timing；
- requested/selected/approved/actual order 与 order differences。

五个 fresh paired cohorts 使用各 controller 的 common seed，只报告 per-seed ranges/directions；
不做 statistical significance，也不把 aircraft 数量当作 independent-n。`COMPLETE` 或 replay
`MATCH` 不认证 global optimality、generic feasibility 或 realtime performance。

## CLI 与输出

服务器执行入口为 `python -m terminal_marl.scripts.evaluate_arrival_closed_loop_validation`，
`--source-commit` 需要实际 source 的完整 40 位十六进制 SHA：

```bash
python -m terminal_marl.scripts.evaluate_arrival_closed_loop_validation smoke --workers 3 --source-commit "$SOURCE_COMMIT" --output "$HOME/training_outputs/terminal/arrival_closed_loop_validation_20261011_v1/smoke"
python -m terminal_marl.scripts.evaluate_arrival_closed_loop_validation suite --workers 18 --source-commit "$SOURCE_COMMIT" --output "$HOME/training_outputs/terminal/arrival_closed_loop_validation_20261011_v1/suite"
```

服务器输出为 `/home/user/training_outputs/terminal/arrival_closed_loop_validation_20261011_v1/{smoke,suite}`；
本地同步输出为 `D:/7_codespace/training_outputs/terminal/arrival_closed_loop_validation_20261011_v1/{smoke,suite}`。
旧入口 `evaluate_arrival_restructure` 与其已有 protocol 保持不变。

## 执行记录（终态）

Smoke 为 `SMOKE_PASS`：`error=null`、`exit=0`、elapsed `49.75270622968674 s`；3 arms 均为
`CENSORED`、`stop_reason=tick_cap`、`ticks=2`、`integrity_ok=true`，3 次 independent replay
均为 `MATCH`、`matched_steps=2`、`whole_cycle=false`，safety/hash mismatches 均为 `0`，全部
进程已退出。

Suite 于 `2026-10-11T04:19:03.048981+00:00` 启动，`2026-10-11T05:24:49.811080+00:00` 结束，
elapsed `3946.7621019110084 s`；wrapper PID `3234325`、child PID `3234326`，使用 `18`
workers、`18` specs、BLAS threads=`1`。18/18 arms 均为自然终止的 `COMPLETED`，无 `ERROR` 或
`CENSORED`，`full_valid=true` 与 `integrity_ok=true` 全部成立；native collisions、NMAC、
hard-separation violations 与 hash mismatches 全为 `0`。heartbeat 已删除，当前无 evaluator
进程残留。完整逐 arm 数值见 [closed-loop metrics JSON](closed_loop_validation_metrics.json)。

唯一 fresh-arm failure 为 `dense24_mixed/896221/downstream_soc_external` 的 `F=1`：
arrival `2` 为 `failed_energy`；`full_valid` 不断言 joint completion。其余 suite arms 的
`F/D/U=0`，`expected=actual` 且 `not_yet_requested=0`。3 个 bridge 与 old legacy scalar/
per-arrival metrics（timing 除外）相等，仍只作 descriptive comparison，不作 trajectory 或
stepwise equivalence assertion。五个 fresh cohorts 只报告 per-seed 描述，不作 statistical
significance 或 aircraft-independent-n 声明；`COMPLETE`/`MATCH` 不认证 global optimality、
generic feasibility 或 realtime performance。

本轮 measured-value decision 为继续 traditional route：`energy_slack` 是 provisional
traditional comparison baseline，`rolling_reorder` 保持 opt-in research/ablation。该结论不
启动 qualification、RL/IL 或 M5 geometry；M5+ 需要独立 protocol。全部 seeds 为 development，
永不作为 Q，`PPO_ALLOWED=false`。

## Suite outcome and timing record

以下数值为可读四舍五入；逐 arm 完整精度、诊断计数、integrity 和安全计数见 metrics JSON。
`U` 为 known unresolved，`not_yet_requested` 单列并均为 `0`。

| arm | scenario/controller/seed | N/E/F/D/U | mean/p90/max wait | makespan | energy | min SOC | planning/native/wall (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| bridge dense24 fixed | downstream_soc_external/896102 | 20/4/0/0/0 | 150.753/291.415/385.948 | 1033 | 11479.979 | 0.265 | 0.093/1658.238/1683.615 |
| bridge dense24 mixed | downstream_soc_external/896103 | 17/7/0/0/0 | 134.477/357.123/526.477 | 1096 | 10544.693 | 0.195 | 0.087/1287.147/1312.542 |
| bridge flow6 mixed | downstream_soc_external/896101 | 6/0/0/0/0 | 5.517/13.493/25.101 | 620 | 1602.603 | 0.486 | 0.023/30.020/33.395 |
| fixed/896211 | downstream_soc_external | 21/3/0/0/0 | 170.334/335.932/476.913 | 953 | 11229.045 | 0.273 | 0.102/1787.550/1813.828 |
| fixed/896211 | energy_slack | 21/3/0/0/0 | 97.909/235.886/332.491 | 1036 | 9771.980 | 0.310 | 408.079/931.178/1363.748 |
| fixed/896211 | rolling_reorder | 23/1/0/0/0 | 110.068/270.845/362.989 | 907 | 9476.194 | 0.331 | 1540.166/710.459/2268.674 |
| fixed/896212 | downstream_soc_external | 19/5/0/0/0 | 192.523/400.095/443.691 | 977 | 12060.342 | 0.220 | 0.093/1482.325/1505.712 |
| fixed/896212 | energy_slack | 24/0/0/0/0 | 112.182/235.817/353.700 | 893 | 9646.281 | 0.326 | 303.848/685.820/1011.148 |
| fixed/896212 | rolling_reorder | 23/1/0/0/0 | 114.748/277.329/365.925 | 926 | 9499.823 | 0.328 | 1350.080/598.966/1969.707 |
| mixed/896221 | downstream_soc_external | 15/8/1/0/0 | 208.899/433.266/670.351 | 1031 | 12036.600 | 0.123 | 0.088/1734.162/1756.425 |
| mixed/896221 | energy_slack | 15/9/0/0/0 | 181.937/436.516/538.810 | 984 | 11515.312 | 0.186 | 793.613/1603.874/2414.067 |
| mixed/896221 | rolling_reorder | 15/9/0/0/0 | 181.937/436.516/538.810 | 984 | 11515.312 | 0.186 | 2481.236/1450.168/3946.622 |
| mixed/896222 | downstream_soc_external | 15/9/0/0/0 | 137.632/281.674/406.170 | 995 | 10834.116 | 0.148 | 0.087/2088.979/2111.407 |
| mixed/896222 | energy_slack | 17/7/0/0/0 | 91.882/234.661/291.945 | 1103 | 10176.626 | 0.153 | 660.513/1286.604/1972.122 |
| mixed/896222 | rolling_reorder | 17/7/0/0/0 | 91.882/234.661/291.945 | 1103 | 10176.626 | 0.153 | 1952.632/984.827/2957.757 |
| flow/896201 | downstream_soc_external | 5/1/0/0/0 | 1.278/1.996/2.106 | 798 | 1902.271 | 0.351 | 0.043/110.105/117.234 |
| flow/896201 | energy_slack | 5/1/0/0/0 | 1.278/1.996/2.106 | 798 | 1902.271 | 0.351 | 15.268/53.177/73.136 |
| flow/896201 | rolling_reorder | 5/1/0/0/0 | 1.278/1.996/2.106 | 798 | 1902.271 | 0.351 | 44.101/67.840/116.704 |

The fresh pairing comparison records initial physical equality and the first release/physical
difference where it occurred:

| cohort | initial physical equal | first release / physical tick | membership differs | E→N | N→E |
|---|---|---:|---|---|---|
| fixed/896211 | true | 1 / 1 | true | 18, 19, 21 | 15 |
| fixed/896212 | true | 1 / 1 | true | — | 18 |
| mixed/896221 | true | — / — | — | — | — |
| mixed/896222 | true | — / — | — | — | — |
| flow/896201 | true | — / — | — | — | — |

For fresh mixed/flow cohorts, equal observed outcome metrics and no recorded active-physical or
approved-list divergence over common ticks are descriptive; native physical records omit SOC.
Rolling selected order always matched energy order in both fresh mixed cohorts; mixed/896221 had
fallback `0`, `candidate_evaluated=241`, `timeout=4`, while mixed/896222 had fallback `1` with
`no_valid_candidate_within_budget`, `candidate_evaluated=252`, `timeout=4`. Across the five fresh
cohorts, rolling planning was approximately `2.89–4.44×` energy planning under concurrent timing;
the earlier mixed rolling advantage was not reproduced in these two fresh mixed cohorts.

## 本轮 M4 结论记录

在同一 `external_dispatch_v1` authority 下，`energy_slack` 相对同权限
`downstream_soc_external` 的 mean wait 在四个 fresh dense cohort 分别改善约 `42.5%`
（fixed/896211）、`41.7%`（fixed/896212）、`12.9%`（mixed/896221）和 `33.2%`
（mixed/896222）。`mixed/896221` 同时记录了 SOC baseline 的 `F=1` 到 energy 的 `F=0`，
但 p90 wait 从 `433.2656202168702` 增至 `436.5164484257149`，故该结果保留为带有尾部代价
的对照。

`rolling_reorder` 在 fixed/896211 为 `N+2`，但 mean/p90/max wait 均高于 energy；在
fixed/896212 为 `N−1`，等待指标仍更差。两个 fresh mixed cohort 与 flow/896201 没有观察到
rolling 相对 energy 的 outcome 或记录 order 收益，同时 planning 计算增加；native physical
records 不含 SOC，这些比较限于已记录字段。

基于上述注册范围内的测量，本轮 M4 决定继续 traditional route：暂以 `energy_slack` 作为
provisional traditional comparison baseline，保留 `rolling_reorder` 为 opt-in
research/ablation。该 bounded decision 不改变 default/global controller，也不授权
qualification、RL/IL 或 M5 geometry；后续 M5+ 需另行 protocol。

## 既有 trace extraction（descriptive temporal evidence）

以下不是第二批 validation 结果。输入是 preexisting nine-arm artifact，提取为 newly run，
输出为 `tmp_restructure_checks/dispatch_analysis_v1/{summary.json,report.md}`。提取边界为：
single-seed scenario traces，不做 independent-sample pooling 或 counterfactual causality；
concurrent planning wall time 只作 descriptive timing。

- flow 与 fixed 的 energy/rolling 在 common ticks 未观察到 recorded-active-physical 或
  approved-order difference。
- mixed 的 selected order 在 tick `355` 首次不同，同时出现 first approved order 和
  recorded physical divergence；energy approved 为 `[1]`，rolling 为 `[128,1]`。
- 在原始 tick `355`，`terminal_marl.core.arrival_dispatch._evaluate_order` 的 selected score 为 `[1,0,0,12010,-1593.3111673475812]`，base score 为 `[1,0,0,11909,-1597.3111673475812]`；score 按 `(valid,-badterminal,normalcompleted,phaseprogress,-wait)` lexicographic 比较。
- 该选择直接反映四步 progress，并未把后续四个 E→N 作为 terminal count 进行预测；这是 code interpretation，不是 counterfactual causality。
- UID `128` 是 arrival `22`、interface `m2i`：energy 的三个 K3 均为
  `safety_reject/predicted_separation`，rolling 的首个 K3 被接受。UID `1` 是 arrival `1`、
  interface `i2f`。
- 最终 E→N arrivals 为 `14,15,17,19`，无 N→E。该事件是 consequential ordering 的
  temporal evidence，不证明 entire eventual delta 仅由此事件造成，也不证明 hidden-state
  equivalence。recorded active-UAV 字段缺少 SOC；相等性只限已记录字段。
- fixed/mixed/flow 的 evaluated/timeout 分别为 `224/3`、`270/3`、`97/0`，fallback 为 `0`。
  既有 mixed rolling planning 约为 energy 的 `3.45×`，max wait 增加约 `13.48%`；没有
  universal superiority 结论。

## 预运行检查

主代理已核验定向 preflight：

```text
venv/Scripts/python.exe -m pytest tests/test_arrival_dispatch.py tests/test_external_dispatch_authority.py tests/test_arrival_restructure_evaluation.py tests/test_arrival_dispatch_analysis.py tests/test_arrival_closed_loop_validation.py -q
77 passed / 10.51 s
```

该检查不是第二批 smoke 或 suite 的运行结果；suite 终态结论已记录在上方执行记录，且
18/18 arms 的逐项数值见 metrics JSON。
