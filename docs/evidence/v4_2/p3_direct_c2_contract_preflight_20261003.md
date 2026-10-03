# P3 direct C²：三状态无梯度契约预检

## 授权、证据边界与执行前协议

用户在A/B/C环境闭环完成后授权“按你说的继续下一步”。本轮执行此前登记的3个snapshot接口预检；primary负责代码、文档、判断，worker仅执行指定SSH、测试及Git。没有新的训练准入，PPO_ALLOWED=false，source仍NOT_QUALIFIED。

前置证据为 [直接C²完整周期结果](p3_direct_c2_repair_validation_20261003.md)：A两臂12N、B两臂24N、C规则19N5E/source20N4E；实际安全、耗尽与技术异常未出现，但这些结果不包含参数更新，无法证明训练后不会再次退化。此次快照是已看过的开发数据，不能进入fresh训练/untouched资格集。

输入根 `/home/user/training_outputs/terminal/p3_direct_c2_validation_20261003`；每个输入为对应`X_source/snapshot_400.pkl`。本地对应 `D:/7_codespace/training_outputs/terminal`。执行前worker已核验本地/服务器大小及SHA完全相同，无旧评估进程，新输出根尚不存在。

| case | bytes | snapshot SHA256 | 新诊断policy RNG seed |
|---|---:|---|---:|
| A | 481119 | `4d4a6961d8be636158a8db25c9681d4e086523c97d6730595cacdfeddbc1be9f` | 891701 |
| B | 2401799 | `9caa7b67cd358c52b127d4261601eca65909c57ec7a07101151c4efe32a34427` | 891702 |
| C | 2816488 | `236adbd8bac2b48834d185e74915c805a5bf17e9d6d20e2cf935ee97162392d3` | 891703 |

source仍固定 `/home/user/training_outputs/terminal/p3_normal_advantage_gate_repair_20260928/fit_smoke/terminal_gate_repair.pt`，518035 bytes，SHA256 `3fe7dd1e01d032227521f6fc5c8464fe8d00d98fb0e9f0dbd13411123a27d4ab`。仅在哈希验证后加载可信pickle/checkpoint。新输出根为 `/home/user/training_outputs/terminal/p3_direct_c2_contract_preflight_20261003`，case独立子目录；存在即拒绝覆盖。原快照未含policy RNG tape，因此这里不是重建原第400tick随机动作，而是在相同物理状态上生成一次新固定请求。

固定上限：3个状态，每状态构造一次source请求与K3；每状态两个独立快照副本各step一次。即**3个逻辑决策tick、6次env.step调用**，不延长轨迹、不换seed补覆盖。先合成单测，后A单状态pilot，primary核验完整JSON及字段后才运行B/C。不重跑已成功case；任何契约失配立即停止扩展，保留错误与已写产物，不改阈值或状态凑通过。

检查内容：

- 所有实际UID使用运行时既定字符串排序，90D观测逐行绑定；forced-WAIT从阶段、commit、future reservation、SOC、retry和接口冻结状态独立复算。
- WAIT proposal=-1/条件概率缺省，forced WAIT gate/joint logprob=0；RELEASE概率必须为原始sampled requested proposal的条件概率，不能用fallback动作替换。K3第一项必须是采样请求，余项按条件logits稳定降序取两项。
- 88→90迁移张量前缀逐位相同、新增列为0；同一90D观测前88维的gate/proposal logits与集中式critic输出使用既有迁移测试容差atol=1e-6、rtol=0。这只检验初始化，不证明新MDP资格。
- 请求写JSON再读回后驱动两副本；下一观测、reward、done、info、ledger、实际K3 rank、状态P/V/A、能量、checked/executed轨迹hash及安全统计必须相同。沿用SafetyTraceRecorder去除不影响语义的运行时UUID拼写，保留trajectory content hash、时序与数值。
- 碰撞、NMAC、硬间隔、operational buffer、fail-closed、runtime takeover、checked/executed hash mismatch必须为0，缺字段不能当0。检查模型张量与checkpoint文件前后不变。

执行CLI：`PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/user/envs/terminal_env/bin/python -u -m terminal_marl.scripts.diagnose_p3_direct_contract --case X --input-root /home/user/training_outputs/terminal/p3_direct_c2_validation_20261003 --checkpoint /home/user/training_outputs/terminal/p3_normal_advantage_gate_repair_20260928/fit_smoke/terminal_gate_repair.pt --output /home/user/training_outputs/terminal/p3_direct_c2_contract_preflight_20261003/X`，只替换已登记的X。预期几秒至一分钟/case，前台有界执行；如实际转为长运行再创建监控，不复建已经结束的A/B/C heartbeat。

## 新发现与修复：加载器曾创建空优化器

primary审阅发现`initialize_legacy_source`调用旧proposal-head evaluator的`_load_policy`。后者实例化`TemperatureCalibratedGateResidualAdapterStudentPPOV42`时会创建Adam，随后`load_optimizer=False`只是不恢复优化器状态。90D推理wrapper本身确实无optimizer，但先前A/B/C全过程“无优化器”的表述不准确。

影响边界：A/B/C没有backward/step或训练更新，返回给runner的是冻结90D推理wrapper，空Adam已随临时88D容器离开作用域。因此这是不必要的对象创建及报告口径问题，**没有证据将它归因为P3训练退化，也不据此废弃环境闭环结果**。先前migration字段`optimizer_created=false`只能描述迁移步骤，不能证明上游loader未创建optimizer。

本轮primary在`terminal_marl/configs/p3_integrated_repair.py`改用同一具体88D calibrated模型类与只含model/dimension/device的容器，仍通过既有checkpoint schema校验、strict state加载及固定temperature=2.5验证，再冻结全部参数。不修改生产轨迹、动作映射、奖励、模型张量和安全阈值。合成回归会令任何Optimizer构造直接抛错，同时验证精确权重、输出、RNG保存及错误schema/缺tensor/错误温度拒绝。真实预检再次检查固定checkpoint SHA与张量保持。

## 执行与结果

合成及相关回归（newly_run）：`tests/test_p3_direct_contract.py`、`tests/test_service_context_runtime_v4_2.py`、`tests/test_service_context_migration_v4_2.py`、`tests/test_p3_integrated_repair.py`，服务器`59 passed in 39.62s`，exit0。日志`/home/user/training_outputs/terminal/p3_direct_c2_contract_tests_20261003.log`。包括JSON往返与两独立合成env step，脚本输出路径已先验证。源码本地/服务器SHA匹配；primary据此只启动A真实pilot，未提前启动B/C。

实际运行（newly_run）按A→primary审阅JSON→B→C顺序完成，各case只运行一次，exit均0、error=null。每臂`result.json`的PASS表示**已观察到的请求与重放检查通过**，不表示RELEASE分支覆盖或qualification通过。

| case | 实际UID数 | forced WAIT | actor WAIT | RELEASE / K3 | 两副本相等 | 单副本checked/executed采样增量 | 运行s |
|---|---:|---:|---:|---:|---|---:|---:|
| A | 3 | 3 | 0 | 0 / 0 | 是 | 11 | .0412 |
| B | 13 | 13 | 0 | 0 / 0 | 是 | 45 | .4511 |
| C | 14 | 12 | 2 | 0 / 0 | 是 | 55 | .6830 |

所有状态的gate logits、proposal logits、critic在88D前缀与零扩展90D输入间最大绝对差均**0**；复制tensor逐位相同、新增列全0，推理前后张量和固定checkpoint均不变。本次加载到执行全过程无optimizer构造、无训练。共3次固定请求、3个逻辑tick、6次env.step；三种状态共30行，其中28行forced WAIT、2行actor WAIT，没有真实RELEASE/K3或非null实际rank。checked/executed增量111是取每对的一个副本，另一个副本完全重复，不能当222个独立样本。

两份JSON replay在每case逐字节同SHA；下一观测、reward/components、done、info、请求账本、执行状态、能量与轨迹content hash一致。所列8项安全counter的原值与增量均0，没有新终态失败或mask/action错位。单步账本保留尚在飞行/服务的未决请求，这是非终态快照预检，不应把这些U误报为完整周期失败。A/B/C完整周期终态仍引用前一轮结果，本轮不重复计数。

primary另外直接读取三份哈希绑定可信pickle，只检查时刻400的mask原因，没有step或额外policy请求：

- A三架都是已commit的final/transfer阶段，因此全部forced WAIT合理；不是字符串UID排序把行打乱。
- B九架已commit；另外四架12/15/16/32分别retry到408/408/405/407秒。当前400秒，全部forced WAIT符合预动作限制。
- C十一架已commit，26号retry到402秒；1号内环与22号中环是两个真实actor。两者采样WAIT，gate/joint logprob分别`-1.9073486328125e-6`、`-.002051830291748047`，proposal缺省而非伪造0号proposal。相应RELEASE概率约`1.81e-6`和`.00205`，说明这两个时刻的冻结source输出很偏WAIT；其条件proposal均极度集中于10号。不能据两个状态证明全局策略塌缩，也不能证明该时刻RELEASE可行：C当时已有三个final承诺，内环立即放行至少受final资源限制。

本轮的实质性缺口是**真实RELEASE/requested conditional likelihood/K3 rank选择没有覆盖**。合成测试验证了解码、稳定排序、K3执行重放和错误绑定拒绝，但不能代替真实轨迹场景中的该分支。原来选择统一tick400的抽样方案没有保证actor/RELEASE覆盖；这属于本轮预检设计的局限，不能包装成“整个接口已完整通过”。不改变原seed、不延长原3态预算，也不把新反事实动作混入原结果。

产物SHA256（每case目录内保留完整request_tape、两份replay和result，父目录保留log/exit）：

| case | result.json | replay_1.json = replay_2.json |
|---|---|---|
| A | `78944e333e85c8a9eee3bbc3c4a9d0a430a343cf6523beefcbafc1b94d48e969` | `26927c1f587920d0b69afec961cef83b169e44fb6074dd4f490e8c34ce156741` |
| B | `95891e49cc762579beff76bba7326ccbf5f4ead02481ca9f070baf0ec7b626dd` | `dd5f9ac33fae2237dec45af7e741030dc70fda77449747ffc4e960ab423f85a1` |
| C | `8edab73482436359c30e0b773c757c895206b926563f63561bd36fcb7b276bae` | `f0f733227c7ccd357d5d7ab8d4cde64287e0f1e9e8be1685819a4450f9472039` |

已执行脚本SHA=`44a3f7d3bc4e7053f99dce021e5127a692ff25534bb1ee74821990c2a6188e4a`；initializer SHA=`eee7eb0a2419e2990334edeed73ffb28f94699b4157d54c7815002ec792e2f23`；测试SHA=`097ade0148208a8e6ebbd6a8313bfa1aba32aea72b427fe55601bf9aff1e7338`。没有改动a7096fb轨迹/动力学核心、proposal物理映射、安全或reward。

终态核对：12个result/tape/replay文件大小及SHA全部本地/服务器一致，A.exit/B.exit/C.exit均0，固定checkpoint仍3fe7。无本轮或旧integrated-repair诊断Python进程；本轮每case不足1秒，不需要长运行heartbeat，前轮heartbeat已删除。所有历史产物与无关dirty保留。当前仓库无Git remote，本轮只本地提交，不声称推送。

## 资格与后续边界

即使3态预检通过，也只得到已有开发状态的接口一致性证据。不能覆盖完整状态空间、优化器更新后的likelihood/ratio、fresh student imitation，不能据此准入PPO。本轮不创建优化器、不反传、不更新权重、不覆盖checkpoint、不运行held-out或硬件。

**本轮结论：已观察契约PASS；真实RELEASE覆盖未建立；P3训练退化仍未验证解决。** 已修复的环境退出阻塞没有在这三个tick重新出现，但本轮没有学习前后比较。当前没有支持调PPO超参数或强制解冻旧source proposal的证据。

下一步应先登记一个独立、按决策事件采样的无训练RELEASE分支预检，再进入fresh数据流程。采用预先固定的开发cohort和单个完整周期上限，保留所有观察到的actor WAIT/RELEASE及失败，按时间顺序检查首次RELEASE，不按结果换seed、不清除retry/commit掩码、不人为提高source RELEASE概率。需同时记录requested probability、rank1、实际K3执行rank和双副本next-state；如果直到完整周期仍无RELEASE，如实以覆盖失败结束并诊断，不能无限跑到通过。该独立预检应复用本轮校验器、既有完整周期runner与snapshot机制，不另造训练框架；它尚未运行，也不追溯升级本轮的覆盖结论。

后续仍需按研究规范制定V3/90D fresh I0/I1/I2/I3协议并冻结数据、seeds、样本/训练上限和正式资格集；新schema用fresh actor，3fe7只作冻结对照。数据覆盖高速内环、busy LZ、低SOC、retry与拥堵恢复；本轮三个开发快照不得充当新训练数据。student-only资格完成后，才评审有界PPO及更新前后同场景完整周期配对。本轮不会自动沿用旧V2/88D准入启动训练。
