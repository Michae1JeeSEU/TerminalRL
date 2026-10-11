# Arrival closed-loop validation：2026-10-11 预注册协议（V1）

状态：`PREREGISTERED / NOT_STARTED`。本记录登记第二批 smoke 与 suite 的固定范围、边界和
输出语义；未把进程启动、目录存在或 smoke 结果写成 full-cycle completion。第一批结果见
[first_batch_20261011.md](first_batch_20261011.md)，当前阶段见
[CURRENT_STATUS.md](../../CURRENT_STATUS.md)。

## 版本与范围

第一批已完成。其实现 source commit 为
`59e3f24652c7480087b8110a832105bbd010bea1`，文档提交为
`84a2f07e399626cf3214e334556aa0a4caca2ad5`。第二批的实际 `source_commit` 必须在本次预注册
提交后由 launcher/header 写入完整 40 位 SHA；本记录不预填 self-SHA。

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

- normal、emergency、fail、divert、unrequested 与 arrival integrity；
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

## 既有 trace extraction（descriptive temporal evidence）

以下不是第二批 validation 结果。输入是 preexisting nine-arm artifact，提取为 newly run，
输出为 `tmp_restructure_checks/dispatch_analysis_v1/{summary.json,report.md}`。提取边界为：
single-seed scenario traces，不做 independent-sample pooling 或 counterfactual causality；
concurrent planning wall time 只作 descriptive timing。

- flow 与 fixed 的 energy/rolling 在 common ticks 未观察到 recorded-active-physical 或
  approved-order difference。
- mixed 的 selected order 在 tick `355` 首次不同，同时出现 first approved order 和
  recorded physical divergence；energy approved 为 `[1]`，rolling 为 `[128,1]`。
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

该检查不是第二批 smoke 或 suite 的运行结果；validation full suite 在本记录建立时仍为
`NOT_STARTED`。
