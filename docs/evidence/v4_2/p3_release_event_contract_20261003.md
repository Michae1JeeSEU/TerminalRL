# P3真实RELEASE/K3事件预检（2026-10-03）

## 问题与执行前方案

用户已授权继续上一步。本轮沿用academic-experiments的最小无训练验证及research-pragmatist的有界机制检查：只补[上一轮三快照](p3_direct_c2_contract_preflight_20261003.md)未覆盖的真实RELEASE/K3分支。此前30行全为WAIT，已观察的迁移/重放一致性通过，不能将其称为RELEASE概率与执行资格通过。生产轨迹核心保持a7096fb，模型纯推理加载及已有validator保持1b7a792；primary亲自实现、诊断及写文档，worker仅执行明确测试/SSH/Git。

研究问题：冻结3fe7 source在修复后的V3/90D环境自然提出RELEASE时，requested gate/proposal概率是否仍绑定原请求，K3是否把请求放在rank1，实际被接受的rank及轨迹是否与序列化请求重放一致。这个问题不需要训练、性能对照矩阵或重新测A/B/C完整完成率。

固定一个新开发cohort：12请求、warm6、SOC .90–1.0、env seed891901、policy seed891902；均在运行前登记，`rg`未在现有docs/config/tests中找到这些seed。使用`build_config`中的direct_c2、nominal_until_release、90D、pre-action forced-WAIT、K3、teacher关闭、static geometry和原安全/能量/奖励配置。固定checkpoint `/home/user/training_outputs/terminal/p3_normal_advantage_gate_repair_20260928/fit_smoke/terminal_gate_repair.pt`，SHA256 `3fe7dd1e01d032227521f6fc5c8464fe8d00d98fb0e9f0dbd13411123a27d4ab`。此前A/B/C原进程和输出不重启、不覆盖。

事件规则在运行前固定：

1. 按时间自然推进，每tick仅调用一次原source请求函数；不修改gate温度、proposal、掩码、retry或安全层，不强制RELEASE。
2. 保存所有tick的UID、actor WAIT/forced WAIT/RELEASE、requested gate/proposal/logprob、K3、接口、执行trace及拒绝原因；所有被拒请求保留。
3. 对o2m、m2i、i2f三个接口，各取**首次自然请求**和**首次实际K3批准commit**；同tick合并。最多6个事件tick，不以rank2/3出现为追加运行条件，不把未出现fallback写成已验证。
4. 每个事件在执行前保存可信env pickle与policy RNG前后状态，动作先JSON序列化再执行。在主环境与独立快照副本上执行完全相同的动作/shortlist；每事件仅增加1次env.step，不重新采样policy。比较原validator的next observation、reward、done、info、ledger、P/V/A、能量、content hash、安全、实际rank。新commit必须有对应checked hash与正确轨迹mode，i2f为final_approach，其余ring_transfer。
5. 三接口请求与执行均覆盖且重放一致即结束；否则最多1800个主tick或1800墙钟秒，并在自然完整终态停止。最坏1800主step+6 replay step。无RELEASE/未覆盖接口返回COVERAGE_MISSING，契约/技术/实际安全/耗尽异常返回ERROR并保存失败状态。不更换seed、追加cohort或调整阈值凑PASS。

这里“执行”指K3批准、轨迹commit且本tick按该checked plan推进；若proposal含延时，该tick可以仍在计划内的等待段，不能误称已经完成降落。达到覆盖即停止的结果是非终态prefix，未决请求不用于计算完整周期成功率；环境完整周期结论仍引用上一轮A/B/C。

代码范围仅新`terminal_marl/scripts/diagnose_p3_release_event_contract.py`、`tests/test_p3_release_event_contract.py`及本轮文档；复用已有check_requests、execute_tape、load_legacy_source、env.step。生产轨迹、动力学、安全、奖励与参数不改。新增synthetic测试先覆盖首事件规则、真实合成env RELEASE/commit/UUID重放、零RELEASE封顶返回、重放不一致留证停止、错误checkpoint拒绝及输出JSON。

## 执行安排与产物

- 执行环境：`ssh lab-server`，cwd `/home/user/projects/terminal`，Python `/home/user/envs/terminal_env/bin/python`；`PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`。代码/docs仅本地修改，通过Syncthing核对后执行，禁止源码上传或server Git。
- 新输出根 `/home/user/training_outputs/terminal/p3_release_event_contract_20261003`。前置只读检查确认目录不存在、无三种P3诊断Python进程、checkpoint仍518035 bytes/3fe7、前轮validator与initializer本地/服务器一致。Git HEAD=1b7a792、index空、无remote；无关dirty保留。
- 先运行新旧validator的两文件synthetic回归；后运行独立`smoke`：3请求/warm3、高SOC、env891911/policy891912、固定2主ticks、120s上限。smoke只是新脚本输出校验，不能并入正式事件覆盖。
- primary检查smoke的result、decisions.jsonl、事件tape/live/replay与log/exit后，仅启动一次`run`。预计通常数分钟内；若启动检查后仍在运行，则创建5分钟heartbeat并按实际耗时调整。最多一个主评估进程，重放在同进程执行。

正式CLI（smoke仅在同命令增加`--smoke`且output改为`.../smoke`）：

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/user/envs/terminal_env/bin/python -u -m terminal_marl.scripts.diagnose_p3_release_event_contract --checkpoint /home/user/training_outputs/terminal/p3_normal_advantage_gate_repair_20260928/fit_smoke/terminal_gate_repair.pt --output /home/user/training_outputs/terminal/p3_release_event_contract_20261003/run
```

每个输出目录保留result.json、progress.json、全部decisions.jsonl；所选事件目录保存before.pkl、request_tape.json、policy_rng.json、live.json、replay.json。异常额外保存failure_request.json、failure_state.pkl；旁路log/exit不得覆盖。没有训练、optimizer、反传、checkpoint写入或held-out/硬件操作。

## 执行与结果

相关回归（newly_run）：新`test_p3_release_event_contract.py`与前轮`test_p3_direct_contract.py`共**18 passed in 16.86s**，exit0。日志`/home/user/training_outputs/terminal/p3_release_event_contract_tests_20261003.log`。新runner/test的执行SHA分别`7dfbec8f55ca7b5271db4610ecc64f5b4d945ba718421de8356caa8a41a23c18`、`9069634e41b51ea34b3dd16d5f225da3f019580ee42fed996e76c93b15ddc41d`；两者及原validator/initializer本地服务器一致。

独立smoke（newly_run）：`SMOKE_OK`、exit0、error=null，固定891911/891912，2主ticks+1事件重放=3次env.step；2次policy请求，共3 RELEASE+3 forced WAIT，无fallback/拒绝。第1tick三接口都首次请求并批准，实际rank均1；第2tick三架均已commit而forced WAIT。88→90三项输出差均0，权重不变，所有所列安全统计及hash mismatch为0，checked/executed plan matches=3/samples=30。stop=tick_limit、terminal=false、censored=true，这是正常的两步输出试运行，不计入正式run资格或完整周期完成率。

primary亲自审阅smoke的三个请求：o2m为proposal13、K3=[13,6,12]、gate/proposal/joint logprob=-.0098350048/0/-.0098350048；m2i为proposal13、K3=[13,12,6]、-.0663838387/-.0052127838/-.0715966225；i2f为proposal12、K3=[12,6,14]、0/0/0（float32归一化饱和值）。实际selected均rank1，相应两份轨迹为ring_transfer、一份为final_approach，checked hash与content hash相等。event_0001 live/replay JSON逐字节SHA相同：`78c45c32465605a8fea630718586dce618fe442f2a48b749d737d56bab5905b3`。这证明真实RELEASE存在且新输出路径可用，不能从先前400tick偏WAIT的两行推断source整体不放行。

smoke完整产物在新根`smoke/`，相邻`smoke.log`198bytes、`smoke.exit`为0，primary本地确认。primary据此准许仅一次既定`run`，不重复smoke或旧A/B/C。

### 注册开发run（newly_run）

固定891901/891902开发场景在**第1个主tick**已覆盖三接口的首次请求和首次批准，按预定停止规则正常结束：status=PASS、error=null、stop_reason=coverage_reached，1次policy请求、1主step+1副本step，事件数1，脚本collect墙钟14.4848s。六架warm-start活跃飞机中4 RELEASE、2 actor WAIT，3个请求批准、1个未批准。不是因runtime被截断或另换seed结束；没有推进余下完整周期，terminal=false/censored=true是这个事件前缀的预期语义，不能据此报12/12降落完成。

| UID / 接口 | 请求proposal / K3 | requested gate / proposal / joint logprob | 执行 |
|---|---|---|---|
| 0 / o2m | 13 / [13,6,12] | -.0002508163 / 0 / -.0002508163 | 三候选全部拒绝，canonical WAIT |
| 2 / o2m | 12 / [12,11,6] | -.0091729164 / 0 / -.0091729164 | rank1批准，ring_transfer |
| 4 / m2i | 12 / [12,20,11] | -2.6905918121 / 0 / -2.6905918121 | rank1批准，ring_transfer |
| 5 / i2f | 12 / [12,6,14] | 0 / 0 / 0 | rank1批准，final_approach |

UID1/3自主WAIT的gate/joint logprob分别-.011477232/-1.682053089，proposal缺省。特别是UID4自然采到了一个较低概率的RELEASE，说明本轮不是把gate改成argmax或人为强制放行。proposal logprob=0是当前float32分布的饱和输出，仍保存真实原值；不伪造小负值或以执行动作重算请求概率。

primary亲自核对UID0的拒绝链：主blocker=HEADWAY，三个候选均为`merge_slot_retime_conflict`，K3检查数3、actual rank=null。请求trace仍为`[1,0,.5]` / RELEASE proposal13；执行trace为`[0,0,0]` / WAIT，teacher/shield_used均false，requested joint index仍14。**被拒后的canonical WAIT没有覆盖原RELEASE或其likelihood**。这是合法的放行间隔拒绝，不是实际安全违规，也不是“本次仍然卡住”的证据。保留原请求和拒绝记录，不删掉该行计算全通过。

三条新commit的checked hash与trajectory content hash全部相同；两条ring_transfer及一条direct_c2 final_approach均按相同物理状态重放。live/replay完整JSON逐字节SHA相同：`14a969234cceef5d1f68b4c62445436354f75065f7f7d34f2bda508959c4649f`。next observation/reward/ledger/执行rank/能量/PVA/轨迹hash一致，reward=.24；88→90迁移三个输出差均0，模型权重及3fe7 checkpoint不变。

实际碰撞、NMAC、硬间隔、operational buffer、shield/joint fail-closed、runtime takeover、hash mismatch均0；min actual/swept separation=202.34294/202.23748m，checked plan matches=3/samples=15，trajectory_infeasible=0、emergency attempt=0。K3累计检查6候选，3批准、1次shortlist耗尽；**成功选中rank2/3未出现**，不能称该稀有分支已被本次真实场景覆盖。它不是本协议的追加运行条件，不为了制造fallback改变源策略/候选或继续堆积场景。

run/result.json SHA=`baa2d65b572ad3b0041eaaeed98e5abfc690c94957823238bd7b5226c62f26d1`。上述检查由primary直接读取同步JSON与事件trace完成，不以worker的PASS措辞代替诊断。

终态：实际Python PID2466391、wrapper PID2466390均已退出，run.exit=0；run.log=200bytes，decisions.jsonl=1行/15882bytes，result=13408bytes。运行在启动后30秒核对窗口内已结束，因此没有创建新的长运行heartbeat，也没有停止其他任务。progress.json保留running/tick1只是最后中间快照，终态以result/exit/进程三者确认。

最终只读核对：smoke/run各自result与live/replay共6文件本地/服务器大小及SHA全部一致，两个exit均0，固定checkpoint仍3fe7、PID2466391 absent且无本模块进程。smoke/result SHA=`fdc3fc07d77e6dd03ad0b7e68ec1bbeb22adab921c340f47f8a58a3932d277d9`。本轮按4文件提交：新runner、新测试、本报告和CURRENT_STATUS最新块；其他dirty保留。仓库无remote，不推送。

## 结论边界与后续

证据单位为一个开发cohort中的前缀和最多6个事件，不是多seed泛化评估。筛选首次请求/首次批准是为了覆盖接口机制，所有未获准请求仍留存；不据所选事件计算成功率。没有独立测试集、性能baseline或方差估计；这些不是接口一致性结论的缺项补跑理由，但禁止将结果外推为吞吐提升、策略最优或训练退化解决。

如果覆盖通过，环境及冻结source请求链的已知实现检查可收束，下一阶段按低层策略规范准备V3/90D fresh I0专家数据与I1/I2/I3 student-only资格。新schema须fresh actor；历史3fe7保留对照，旧快照和本轮开发数据不混入fresh训练/untouched资格集。当前PPO_ALLOWED=false，不启动PPO/full training。若出现具体契约阻塞，由primary用所保留同状态tape/pickle完成最小诊断，不先改变学习超参数。

**本轮判断：计划要求的真实RELEASE、K3请求排序、rank1批准执行及拒绝请求保真/重放检查均通过，没有发现新的接口失配。** 结合此前完整A/B/C，这已足以结束针对已修复内环退出机制及冻结source运行接口的重复开发诊断。未观察到rank2/3成功fallback、忙LZ/低SOC下全部释放分支或优化器更新后的行为，属于证据边界，不另设无限补测门槛。此前三快照覆盖不足保持原结论，不追溯改成全分支PASS。

接下来应把工作转向**新MDP下的数据与student学习能力**，而不是继续使用旧88D source直接PPO：

1. 复用现有I0 collector/teacher，明确90D观测、direct_c2动作物理语义、nominal holder与同一安全/能量配置，先检查teacher标签确实基于新generator。既有88D label/schema或旧physics默认不能直接复用。
2. 在collection前登记fresh train/internal-audit及未消费资格集分区、固定seed/episode预算/完整停止规则；覆盖高速内环、busy LZ、retry、低SOC和拥堵恢复。首个小样本只验证collector输入、标签、保留规则与JSON输出，不能替代数据覆盖或student闭环。
3. I0通过后训练fresh 90D actor做BC，再Safe DAgger和student-only完整周期资格。I3通过前不安排PPO；之后才固定有界更新预算及同场景更新前后配对，从而真正回答“训练后是否退化”。

这三项是下一阶段顺序，**本轮没有执行新数据采集或任何学习**。环境开发闭环及接口预检通过，不等于fresh student资格，更不等于P3训练退化已解决。
