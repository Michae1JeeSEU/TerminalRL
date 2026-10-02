# V3 R2 ring-budget exception repair (pending application)

> 2026-10-03 状态更新：用户随后授权停止无必要的旧诊断并统一修复。R2 已主动停止；此单行异常补丁由[统一实现修复](p3_integrated_repair_20261003.md)取代，旧 patch 不再直接应用。新实现同时恢复合法高速边界的可减速性，原测试中“合法高速必须拒绝”的断言不再适用；历史 patch/日志保留供追溯。以下 pending/application order 是当时记录，不是当前执行指令。

## Scope and observed failure

The user requested preparing this repair while the frozen R2 matrix continues, then analysing the complete comparison. Primary wrote the patch, tests and diagnostic runner. Workers only executed specified tests and collected observations.

R2 C1 `legacy88` ended with `RuntimeError: ring_hold_tangential_budget_unavailable` after `1017` successful ticks; result SHA=`4f1bf7d52d448572fc0f524bc2513be61cfc0ff689d01a41b903d89709854199`. Its partial ledger contains `24` requests, `6` normal, `2` failed and `16` unresolved; hard safety/invariant/uncertified counters are zero. The matrix continued to C1 `repaired88`. This arm error remains a failed result, regardless of ledger-only validity.

The result records the exception type/message but no failing PVA, speed, ring role or traceback. Consequently the exact state at tick `1017` has not been replayed. The new tests below reproduce the same rejection branch through real generator and energy-query calls; they are constructed boundary fixtures, not a reconstruction of that missing state.

## Diagnosis and exact change

`TrajectoryGenerator._v3_ring_transition_parameters` computes a bounded smooth-ramp budget using `.97 * min(horizontal_accel, horizontal_decel, lateral_accel)` and the peak radial load. A nonzero speed change with no remaining tangential budget is an unavailable candidate under this generator. It is currently raised as an ordinary `RuntimeError`. The two energy queries deliberately catch only `TrajectoryInfeasibleError`, so this expected rejection escapes the conservative pad scan and aborts the query.

The independently constructed inner-ring fixtures use `radius=200 m`, speeds `17.2 m/s` and `sqrt(1.5*200)=17.320508075688775 m/s`, target `10 m/s`, and exact circular acceleration. These live states obey the unchanged physical cap. At `17.2 m/s`, radial acceleration is `1.4792 m/s²`, which exceeds the generator's reserved budget `1.455 m/s²`. This establishes a conservative generator rejection, not proof that every physically valid braking curve is impossible.

The production change is exactly one exception class:

```diff
-            raise RuntimeError("ring_hold_tangential_budget_unavailable")
+            raise TrajectoryInfeasibleError("ring_hold_tangential_budget_unavailable")
```

No trajectory, acceleration reserve, speed limit, energy/SOC gate, SafetyFilter, action authority or reward rule changes. A queried pad with this unavailable route is skipped; if all compatible pads are unavailable, the existing required-energy `inf` and slack `-inf` reject the candidate. Invalid live PVA and unrelated RuntimeErrors still propagate. This repairs exception handling; it does not qualify throughput or solve high-load braking reachability.

## Newly run verification

- Original source: `4 failed / 4 passed in .30 s`, exit `1`. Both high-load braking fixtures and both real energy-query calls expose the same ordinary RuntimeError. Baseline log SHA=`59823f03bf56d51e0856ba8e42f94f19449947b2e2d87cc84b92d28b4746a70c`.
- Isolated candidate: `8 passed in .18 s`, exit `0`; candidate log SHA=`c513a336404899b0e4e59540faac60f730d81e38e236b1b942334605c26575ed`. Tests cover rejection type, real candidate/immediate energy queries, no holder/pad mutation, invalid-PVA error propagation, identical feasible V3 braking PVA/time and identical V2 curves.
- The first expanded runner exited before tests because direct `.tmp` script execution lacked the workspace import path. Primary corrected only that runner; failure log SHA=`09b31ce569ec902402e2c4913e3e760048a624c3cd9b8deb4d9e00f3705283c5` is retained.
- Expanded retry: `50 passed in 3.23 s`, exit `0`, no skip; log SHA=`61553db5d7141b5b7d8e6e2ecdb6594744a82514decfae05ae9e1cc4948eb838`. Scope: V3 generator, V3 energy failures, energy reachability, emergency energy, warm-start energy and the canonical 8-case pending test. The isolated process installed only the candidate method on the canonical generator class, reported its actual definition file, and confirmed all 15 R2 source hashes before/after. PID `2323277` remained alive. The byte-preserving candidate and canonical test hashes below were verified locally and on the server.
- `git apply --check` passes for the durable patch. Its source context preserves the current mixed line endings; the initial normalized-context patch failed the check and was replaced before application. Production source bytes remain unchanged.

Logs: `/home/user/training_outputs/terminal/trajectory_v3_repair_20261002/r2_ring_budget_pending_repair/` (synced to the matching local training-output directory). No full matrix was launched by these checks; no training, held-out, optimizer, checkpoint write or promotion occurred. `PPO_ALLOWED=false`.

## Durable patch and application order

The reviewable patch [v3_r2_ring_budget_20261002.patch](patches/v3_r2_ring_budget_20261002.patch) contains the one-line core fix plus `tests/test_v3_ring_budget_infeasibility.py` (8 cases). It is **prepared and isolated, not applied to R2 sources**.

| Binding | SHA256 |
|---|---|
| Frozen `terminal_marl/core/trajectory_generator.py` | `489d0306297089b8fd7bd437e1a4f846ef1bf8ddac9bd22bab4448cd7b1f1919` |
| Exact byte-preserving candidate | `1a20545b78bb7441e3c555e5cb9800257df79826cff0be56057ad7ec857f92ec` |
| Durable core + test patch | `0254a970abf509ff86ce70792003487ab392df620dbdf04efbf5f6ca1c6b0ae7` |
| Canonical regression test content | `765d31c77ac143b91c6664918de3d630b12493f0b6e646abb66d9ed43e4bee8d` |

Working copies and primary-written runners live in `.tmp/p3_v3_r2_budget_repair_20261002/`; the durable patch includes everything required for the production change and tests. The initial candidate tests used the same one-line semantic change with normalized newlines; the expanded retry binds the byte-preserving candidate above.

Until PID `2323277` and its matrix are terminal, preserve all 15 frozen sources, execution commit `592d9758469130062540793c699964c198249eca` and old artifacts. At terminal, primary first records the complete R2 comparison and source binding, deletes the heartbeat, then applies this approved narrow repair locally after checking source SHA and patch applicability. Workers may run tests/Git commands but must not apply or edit code. Run the new canonical tests against the actual core, make a small commit and retain user dirty files. Do not launch another matrix or claim performance improvement from this repair alone. Analyse complete error/censor/safety/request-closure and both cells' zero-init 88/90 comparisons before selecting further work; seek user review for major changes.

Current production application and repair-code commit: pending R2 terminal. No Git remote is configured; no push has occurred.
