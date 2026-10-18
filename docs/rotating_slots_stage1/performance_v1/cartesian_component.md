# 单阶 Cartesian 包络

本轮仅把 `Segment.bounds(a,b,k)` 从 `_cartesian(*polar)[k]` 改为私有
`_cartesian_component(*polar,k)`。原 `_cartesian` 一次求 0—3 阶四组结果，
而 separation 路径分别请求位置和速度包络；新 helper 只求所请求阶数的
radial/tangential 表达式和旋转，避免计算未使用的三组结果。

原 `_cartesian`、`Segment.state`、`Segment.polar_bounds` 保持原实现。
新 helper 逐项保留原 rr/tt 算术表达式及运算顺序、Interval 外扩舍入、
cos/sin 计算与旋转顺序；0 阶仍执行原来的 `0*sin`、`0*cos`，不作代数消去。
每次 bounds 仍构建一次原 polar jet，不缓存、不修改输入。普通整数、
-1 到 -4、bool 和 `__index__` 的索引按原 tuple 规则选择；越界或非法类型拒绝，
不 clamp 到某个合法导数。

本轮不改多项式系数、trajectory payload/hash、state 路径、验证器、递归深度、
阈值、候选顺序、缓存或物理参数。没有 fastmath、float32、Horner 重排或新近似。
合同 §9.2 与 §14 的冻结要求保持。本实现只删除未使用结果的计算；不能由此
预测真实窗口的速度。若后续对照出现不一致，原 `_cartesian(*polar)[k]` 仍为
未修改的参照路径；回退/发布决定由主对话处理，本 helper 不自选近似或改阈值。

## 选择依据：此前 P2 单窗口诊断

已完成的服务器先导诊断位于
`/home/user/training_outputs/terminal/rotating_slots_perf_v1/window_diagnostic_fe66ee4_20261018a`。
三项均从真实 W71-A 的 300 继续到 310，退出码均为 0；完整 prefix 各有 1613 帧，
时间范围 `[0,310]`，六个 ReplayBundle 数据成员逐字节相同。CPU 亲和性为 CPU0，
nice 为 10，native 数值库线程数为 1。

| 项目 | rollout wall / s | rollout parent CPU / s |
| --- | ---: | ---: |
| R0 benchmark | 10.6046597480017 | 10.593785477 |
| OPT benchmark | 10.3894541539994 | 10.377966995 |
| OPT profile | 73.2988349650041 | 73.224996648 |

这只是先前 P2 的单次诊断，benchmark 比值约 1.0207，未隔离旧 recovery 成本，
没有正式三次配对重复。profile wall 是 benchmark 的约 7.055 倍，额外增加
约 6.055 倍，不是加速证据。
profile 中 `_cartesian` 为 118516 calls、累计约 15.133 s，`Segment.bounds`
为 50104 calls；这些累计调用时间有嵌套，不能相加，`time.process_time` profile
数字也不能充作 benchmark wall。上述调用结构是本轮范围的选择依据。

独立回放再验证位于
`/home/user/training_outputs/terminal/rotating_slots_perf_v1/window_diagnostic_replay_recheck_fe66ee4_20261018b/recheck.json`，
主对话已实际读取三个 `VALID` 结果及正确的 R0/OPT package path。此前两次验证
入口脚本错误证据保留于同一输出基准下的
`window_diagnostic_terminal_inspection_fe66ee4_20261018a` 和
`window_diagnostic_replay_recheck_fe66ee4_20261018a`：首次 flat import、第二次遗漏
`sys.path.insert`，两次均未执行验证，不作为物理结果。

## 有界 exact 验证

新增测试 `tests/test_rotating_slots_performance_cartesian_component.py` 使用独立
固定 R0 export 经 `run_reference` 产生 expected；OPT 在另一 `python -I -B`
进程中执行，导入前设置 `sys.dont_write_bytecode=True`。浮点输出统一保存为
`float.hex`，比较 0—3 阶完整 Cartesian 包络、原 state、payload/hash、候选、
separation 的 SAFE/UNSAFE/UNKNOWN 和原函数源码摘要。

输入覆盖全部 83 个稳定槽位、六个地面 pad、六 pad 的全部 18 个 descent
segments、实际 transfer/final、phase wrap、高阶 final-height、短 segment、
零 radius 和 seed `20261018` 的 32 个有限随机多项式 segment。每个 segment
包括起终点、全区间、非网格子区间、零宽查询和 `nextafter` 端点邻域。
额外比较同一个未改动 polar jet 上新 helper 与原 `_cartesian` 的所有阶数，
记录 Interval 运算/构造调用数量，验证 cos/sin 路径保持而未用计算减少。
这些计数只说明代码重复计算的删除，不作为正式 wall/CPU 性能测量。

局部验证范围限定为新增测试、已有 P2 测试和
`tests/test_rotating_slots_physics.py`；不宣称全 70 项或完整 traffic 验收。
当前 `NOT_QUALIFIED / HUMAN_PENDING`。

2026-10-18 本地实际命令：

```powershell
& './venv/Scripts/python.exe' -I -B -m pytest tests/test_rotating_slots_performance_cartesian_component.py tests/test_rotating_slots_performance_p2.py tests/test_rotating_slots_physics.py -q -p no:cacheprovider --basetemp=artifacts/rotating_slots/cartesian_component_tests_20261018a
```

实际退出码 0，`67 passed in 11.52s`。完整日志位于
`D:/7_codespace/terminal/artifacts/rotating_slots/cartesian_component_tests_20261018a.log`。
新增 exact 对照位于 basetemp 的 `cartesian_component_equivalence0`：149 个显式
segment、1043 组区间（每组比较 0—3 阶）、745 个 state 查询，
`comparison_provenance.json` 的 `exact_match=true`。独立 R0 与 OPT 结果文件
`r0_results.json`、`opt_results.json` 字节 SHA256 同为
`530fb8ec94dfc48d5c6824226444760cd9c4df45e472b461299afd6ae559683a`。
原 `_cartesian`、polar/state 路径、payload/hash 函数源码摘要也与独立 R0 精确
相同。已有 P2 对照在 `equivalence0/comparison_provenance.json` 中同样记录
`exact_match=true`，只按其既有协议排除 `archive.cpu_seconds` 墙钟诊断字段。

同一个已建 polar jet 的实际调用计数保存在
`test_interval_work_removed_wit0/interval_operation_counts.json`：

| 路径 | Interval 构造 | `__mul__` | `__rmul__` | cos | sin |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原 `_cartesian`，全部四阶 | 70 | 29 | 7 | 1 | 2 |
| 新 helper，0 阶 | 13 | 2 | 2 | 1 | 2 |
| 新 helper，1 阶 | 12 | 5 | 0 | 1 | 2 |
| 新 helper，2 阶 | 20 | 8 | 1 | 1 | 2 |
| 新 helper，3 阶 | 37 | 14 | 4 | 1 | 2 |

sin 计数包含原 `Interval.cos` 内部移相后的 sin 调用；表中各方法调用可能
嵌套，不将它们相加为独占成本。输入 jet 的每个端点 `float.hex` 保持不变，
四阶输出均与原全量 `_cartesian` 对应结果精确相同。上述计数不含 polar jet
构建，只量当前有界实现删除的重复工作；尚未测量新实现的真实窗口速度。
