# 持续课程推进（2026-09-17）

用户授权持续推进直到获得满意低层策略。目标为docs/execution_plan.md P2–P5覆盖的终端区静态低层策略：自然入流、多UAV等待/切环/最终进近/降落、任务及资源异质性、20/40/60总请求和持续随机流、合法静态几何变化。高层动态控制/硬件飞控不混入此次低层训练。

“满意”以独立场景中的安全、可执行轨迹、非过载完成/排空、等待延误/吞吐、shield负担及rule对比为依据，不以训练回报或一次演示判定。沿用已有安全硬门槛；负荷/属性/几何逐项增加，每阶段先确定可达性、实际参数、seed及训练预算，预注册后执行，失败保留并修复，不将训练与最终评估seed混用。最终覆盖未实现前目标保持未完成。

## 当前第一步：checkpoint24独立评估（执行前冻结）

- 候选：block2/train/checkpoint_24.pt，SHA6420df9a8562135756819459c7682ec897d47acd64e841b75a21c697298ea14f。
- 配对基线：原checkpoint12；两者均在修复后同一生成器下评估，避免混淆几何修复和策略变化。
- 保留seeds710001–710006，仅评估一次；warm3/flow3/flow6，前三deterministic、后三stochastic，900s上限，无训练，无Teacher。
- 接受沿用I3逐episode gate与修复几何gate；报告配对时间变化，若效果弱先诊断，不据此直接声称满意。
- smoke seed740000、warm3 32步验证输出闭环，不作qualification。
- 历史manifest只读，新结果独立保存；正式评估后才设计P2 N8/12的rule可达性pilot及训练参数。

已知差异：requirements.md仍存在旧高度/速度数字，当前已验证场景为config与CURRENT_STATUS的100/70/40m、22/20/16m/s；此次沿用已验证参数，P5单独审查合法变化，不静默切换物理配置。

## checkpoint24评估结果与P2 pilot预注册

独立评估 `block2_eval_formal_20260917` 已完成，candidate和baseline各6回合全部gate PASS，均24/24架完成、安全及逆环向全0。候选相对baseline完成时间差为0/0/+12/0/+9/-2秒，未证明性能改善；不能据此认定满意。710001–710006已消费，不再作为untouched评估。

下一步仅做可达性pilot，不训练：
- seeds748001/748004 N6锚点，748002/748005 N8，748003/748006 N12。
- 每seed配对rule（标准nominal RELEASE意图+同一安全链，无Teacher）与checkpoint24 deterministic student。
- 自然流首到达10s、间隔60s；P1 SOC=.8、priority固定1（emergency5），相同随机mission/gate；物理约束全部保留。
- pilot horizon预设为末次到达+900s，即1210/1330/1570秒，属于扩大请求数后的可达性诊断，不覆盖/更改P1 900秒gate。正式课程截止时间将在pilot结果后另行冻结。
- 保存完成/未完成、active峰值/队列峰值、平均等待和请求至touchdown时间、shield、安全、geometry、exact replay；不把总请求数称为并发数。
- smoke seed748000 N6、32步，先验证双方输入与序列化；正式pilot结果决定是否准入N8/N12训练，不自动跳P3。

## 下一训练块的局部奖励修正（执行前）

源码检查发现旧progress奖励使用 `_advance_plan + _advance_ring_hold` 的累计距离，等待外环22m/s带来每秒约+0.022奖励，大于时间惩罚-0.01。其偏好与减少等待冲突；仅此证据不能证明它是全部性能差异的原因。P2新配置将 `low_level_progress_reward_per_meter=0`，不改旧配置/模型/manifest，不改环境推进；其余time/touchdown/release/shield/failure奖励保持原值。单测验证配置仅该奖励项不同、真实warm-ring WAIT总reward为负。

P2计划（仅pilot PASS后执行）：恢复checkpoint24模型及optimizer，固定N6/N8/N12自然流、60秒间隔、相同P1属性，horizon=末次到达+900秒。六资格seeds750001–750006（前三deterministic后三stochastic）、两入口seeds753001–753002（flow6 160步、flow8 192步），全部gate通过后36回合seeds754001–754036，N6/N8/N12循环，无warm-start。超参数保持block2（lr3e-5、2epochs、batch32、gamma.99、lambda.95、clip.1、entropy.001）。smoke752000 flow6 160步。新独立评估预留755001–755006，与训练/资格隔离。训练每回合沿用safety/geometry/replay/initial ratio/finite/KL/restore及complete gates；任一失败终止并定位，不修改已消费结果。不以训练通过代替延误效果评估。

P2 pilot终态PASS：12/12配对rollout，rule和student均完成52/52架，全部安全/几何/replay gate通过。两组N12 student完成1251/1278s，rule1187/1029s；student平均等待65.66/73.91s，rule19.07/26.88s。证据支持该负荷可达，但不支持student优于rule。pilot manifest SHA edc2772860c705cc08dd52f7bf3574bde964ea14c15c5bf44d848ac51b6662d4；block2独立评估manifest SHA 1fce2a4889b8b5d4b53c01b71698d03cb92e65b020671136c72f99ce9fd6ff24。

P2新协议与参数文件：`terminal_marl/configs/low_v4_2/ppo_p2_natural_flow.yaml`；runner `run_v4_2_ppo_p2`。资格、入口、36回合训练分开，仍按阶段gate推进。后续评估必须报告rule差距而非只看completion。

## P2训练后独立评估（训练进行中预注册）

固定评估最终checkpoint_36，不按验证集选中间checkpoint。一次性保留seeds755001–755006，N6/N8/N12循环两次，全部自然入流、deterministic部署动作；配对本阶段初始checkpoint24与nominal rule，均在P2相同物理/场景/奖励配置下运行，禁止更新。候选沿用安全/完成/geometry/replay/likelihood/shield gate。报告按完成UAV加权的平均等待、request-to-touchdown及每场景完成时间、active峰值、队列峰值和rule差距。硬gate通过但效果未改善时不能宣称训练收敛，先分析原因再决定后续参数；不自动扩大压力。独立评估不提供统计显著性结论。smoke756000 N6 160步只验证输出路径，可用checkpoint24自配对，无资格作用。

P2 V1在第24回合seed754024停止，前23次更新全部通过；该回合12/12完成、1226s、安全/geometry/replay/requested_logprob均通过，唯一失败为initial_ratio检查误差1.1444091796875e-5略超原1e-5阈值，未执行更新。保留原FAIL与checkpoint23，不评估/覆盖原预留最终模型。需只读诊断同一action tape：原采样使用完整world batch，initial_ratio验证用actor子集batch；核验是否由float32矩阵计算batch shape导致差异。该旧seed只用于定位，不作为fresh qualification，不放宽阈值。

诊断已完成1226步原action tape只读复放：full_world_error=0.0，actor_subset_error=1.1444091796875e-5，最大差出现在step379（world5架、actor2架）。修正仅initial_ratio验证器保留采样时world批次后再mask；不改变PPO优化目标、动作/Teacher语义、物理或1e-5阈值。新增回归测试验证检查器保持batch，并仍能拒绝真实模型变化。V1历史FAIL不改。

恢复V2预注册：从最后合格checkpoint23（SHA b9c66290a00896d760070f9c2e9e5077afd986a802d55f38114adb4977fa068d）恢复模型及optimizer；核对V1 23个成功更新、最后回合未更新且唯一ratio gate失败。fresh qualification758001–758006，entry759001/759002，smoke760000，正式训练757001–757013共13回合，补足总计36次合格更新，其他参数/场景/gate不变。配置`ppo_p2_resume_v2.yaml`。原训练剩余预注册seed不复用。最终评估计划755001–755006仍untouched，因V1未生成checkpoint36，明确改为V2最终checkpoint13（不按效果选checkpoint）；不会评估中间权重择优。

## 2026-09-18：V2终态及连续折线边界修复

V2第6回合757006停止，前5次更新成功。失败回合12/12完成、1277s、安全/shield/逆环向0、replay/likelihood通过，initial ratio误差0；唯一失败为内侧偏差0.0592194m>0.05m，未更新。V2 FAIL保持不变。

只读复放1277步exact：UID39，t1037s，legacy_cubic节点最小半径200m，但线段连续最小半径199.9275204m，最近点位于该段55.885%处。原修复只检查节点半径（测试也只检查中点）漏掉弦内切。现在使用每段投影最近点精确检查，沿用0.05m容限，仅超限旧曲线进入现有forward-annular fallback；不改物理/安全阈值，不改合格旧曲线。固定见证测试和原几何/动力学/预约测试验证。

V3执行前预注册：从V2最后合格checkpoint5恢复（SHA8a8466f61119f7f5b5f780249d26f218ee9b8ae3e4aededc5c7f5fe2e51b201b），此前23+5=28次合格更新，剩8次。新资格762001–762006、入口763001/763002、smoke764000、训练761001–761008；其余场景/奖励/PPO参数/门槛不变。V3资格绑定新生成器，正式训练前小样本验证与入口检查。最终预留评估755001–755006仍未消费，候选明确为V3最终checkpoint8，三个对比控制器都使用同一最新生成器。未依验证成绩挑checkpoint。

## 2026-09-18：V3 入口终态与 KL 只读诊断

V3 资格阶段 `p2_resume_v3_20260918/qualify` 通过 6/6；六个 seed 的完成数量、safety、shield、geometry、replay、initial requested-logprob、finite 与 restore gate 均通过。入口阶段只执行了 seed763001（flow6，160步），因 `approx_kl=0.11257042959332467` 超过冻结上限 `0.05` 而 FAIL；该回合 `actor_decisions=8`、`updates=10`、`clip_fraction=0.16666666865348817`、`grad_norm=48.13680276870728`，其余 entry gate 通过。未执行 formal train，没有 `checkpoint_8.pt`；V3 FAIL 和旧 V1/V2 FAIL 均保留。entry manifest SHA256 为 `2098af1c3109c17fc489b7c9f20a9db11eb67aa62b77b5a03beb2de66909cd7d`。

为区分 Adam 恢复、短 rollout/actor 样本和实际策略漂移，新增独立诊断协议
`terminal_marl/configs/low_v4_2/ppo_p2_v3_kl_diagnostic_v1.yaml`（SHA256
`846c15454e20ba02e330011ec2675df2fec62f007a7274dfc0d130e7db6c4d9c`）及 runner
`terminal_marl/training/diagnose_v4_2_p2_v3_kl.py`（SHA256
`09e454b2d3fdf158b2425ddeafbd90e5c40f9a029f3584f75b92775b16768b47`）。它固定 V2
checkpoint5，分别对同一次 stochastic rollout 使用恢复 Adam 和 fresh-zero Adam，
复用真实 `StudentPPOV42.update`，固定同一 minibatch permutation，并记录 actor GAE、
每批 actor/world 数量、每批及 full-rollout requested-action KL、梯度范数、参数变化、
safety/replay。诊断 seed765001/765002 为 fresh，heldout 755001–755006 未使用，
并避开已注册/消费的 761xxx–764xxx seed 段；smoke seed765000 只验证输出闭环。

| seed / 场景 | actor decisions | raw actor GAE mean/std | restored Adam approx/full KL | fresh-zero Adam approx/full KL |
|---|---:|---:|---:|---:|
| 765001 / flow6 / 160 | 3 | -2.92631555 / 1.27700114 | 0 / 0 | 0 / 0 |
| 765002 / flow8 / 192 | 11 | -2.74803329 / 0.98559004 | 0.00201741193 / 0.00530911097 | 0.0342860984 / 0.0970129967 |

两条件均通过 safety、geometry、replay、requested-logprob exact 和 initial ratio；每个
seed 的两条件 batch-plan SHA 相同。seed765002 的 fresh-zero trace 中 batch10
（2个 actor decision）的 pre-step KL 为 `0.3883524537`，对应 restored 为
`0.0217540264`。因此这两个 fresh seed 未复现原始 `0.11257`，且不支持把 Adam
清零作为修复；fresh-zero 在 flow8 反而产生更大的实际漂移，但单个新 seed 不能排除
原 seed 与原 Adam 状态的交互。诊断 manifest SHA256 为
`41017e10f45105a6125213ac51a630bdcfda9c35bac196608e2affc891955f69`。

原 PPO kernel 的 `approx_kl` 是各 minibatch **更新前** actor KL 的平均值，而 full-rollout
KL 是最后一次更新后的整段 action tape 漂移；诊断中二者可明显不同（例如
765002 fresh 的 `0.0342861` 对 `0.0970130`）。在任何 fresh formal rerun 前，最小候选
修复是预注册显式 target-KL 的逐步检查，并在超限时对该 optimizer step 做事务回滚；
不扩大阈值、不用新 seed 覆盖原 FAIL、不消费 heldout 评估 seed。当前 P2 仍未准入正式
训练，独立评估与 P3–P5 保持阻塞，直到 root 完成审查并另行预注册。

## 2026-09-18：V4 target-KL 训练与 heldout 效果审查

target-KL kernel 与独立 V4 runner 已完成审查、定向测试、小样本输出验证并提交为
`c9d5b38`。新协议从V2最后合格checkpoint5恢复，资格seeds766001–766006为6/6 PASS，
入口767001/767002为2/2 PASS并得到`PPO_ALLOWED=true`；正式训练769001–769008为8/8
PASS，66/66架完成，468个optimizer step全部接受、0拒绝，最大accepted full-rollout
KL为0.006534<0.01，safety/shield/逆环向均0。最终checkpoint8 SHA256为
`5e6811b8349d875fb007943476aec62d93550a74ffaf4ae46614c5c188ab004f`，manifest SHA256为
`73502ed0f956e517c47dd14ea6bfcc411cb927eba43c4a409fdee119fa7c82f0`。V3原FAIL不变。

V4 heldout入口提交为`c030858`。一次性seeds755001–755006在candidate/baseline/rule三个
控制器下共18/18 records全gate PASS，各52/52架完成，safety与shield全0。candidate
加权平均等待55.1096s、request-to-touchdown 355.8077s、平均完成时间957.1667s；baseline
分别65.0183s、361.3269s、962.1667s；rule分别16.1842s、319.3077s、939.5s。candidate
相对baseline有改善，但相对rule仍有38.9254s等待和36.5s request-to-touchdown差距，
所以P2效果审查BLOCKED，不进入P3。heldout manifest SHA256为
`167d22cb29d8ef6d08275cb0ff4cbe270c0d9005611a176b85d81fd64730f104`；755001–755006已消费。

既有tape只读分解显示candidate WAIT占2,506/2,851=87.90%，baseline为
3,019/3,370=89.58%；candidate首次release尝试后的lifecycle平均审批延迟6.73s，baseline
8.21s，rule 4.43s。按三interface折算，candidate-rule审批后差约6.9s/机，明显小于
38.9254s/机的平均等待差。因此当前优先诊断pre-attempt WAIT，而不是放宽安全或把全部
差距归因于release后的重试。完整终态见
[P2 V4 训练与评估终态](ppo_p2_v4_training_evaluation_summary.md)。

## 2026-09-18：fresh WAIT/credit诊断与checkpoint归因

为定位candidate与rule之间的pre-attempt WAIT差距，预注册fresh stochastic
seeds771001–771006，candidate/checkpoint24各6回合，Teacher仅label-only审计、不替换行为，
不训练。12/12 records均生成，但candidate完成50/52、baseline48/52，三个回合未在冻结
horizon排空，另一个小样本回合缺Teacher双类覆盖；所以`DIAGNOSTIC_COMPLETE=false`。
所有safety和shield计数仍为0。manifest SHA256为
`b290e2d5f69749e1109bff2c5f348458732a22fbe8c0ce34c15ce98880eefc93`，seeds已消费，不重跑。

candidate聚合false-WAIT为1,601/1,924=83.21%，baseline为80.76%；candidate三个interface
分别为o2m84.86%、m2i75.19%、i2f90.81%。false-WAIT状态release-eligible、retry为0，
RELEASE概率中位数6.59e-7。candidate requested WAIT raw team GAE均值+2.1719，RELEASE
-1.0538；false-WAIT均值+2.6025且61.46%为正。六个candidate seed中WAIT均值都高于
RELEASE，支持team-step credit与gate目标失配。

同一批9,597个冻结actor observations离线评分显示：I3/checkpoint24/V4的combined
false-WAIT分别82.85%/83.11%/80.37%，Brier为0.4636/0.4703/0.4826；I3并未系统优于
PPO checkpoint，因此不支持“单纯PPO遗忘”，更支持P2自然流状态support/calibration缺口。
离线结果SHA256为`64c34bfbbeece10c6308c7b9cffc43b58bdb6000ec64357c82bb51943facabb3`。
下一步转为fresh自然流label support → seed-disjoint gate repair → student-only stochastic
qualification；不继续盲加原team-GAE PPO。完整结果见
[WAIT/credit诊断终态](ppo_p2_v4_wait_credit_diagnostic_summary.md)。

## 2026-09-18：fresh natural-flow gate support预注册

在任何新reset前冻结只收集协议：train support seeds772001–772009、dev support
seeds773001–773006、未来student-only qualification seeds774001–774006，三组互斥；场景为
flow6/flow8/flow12循环，行为保持V4 candidate stochastic student authoritative，Teacher仅
label-only。旧8,552-row D1与失败WAIT diagnostic只做hash绑定，不混入新support。

本阶段要求每回合安全/shield为0、replay与requested logprob精确，并对train/dev分别冻结
双类、三个interface、88D finite schema、event唯一性和数量下限。即使全部PASS也只授权
offline learnability audit；`GATE_REPAIR_ALLOWED=false`、`PPO_ALLOWED=false`。runner/配置
提交为`4166249`，协议见
[P2 gate support collection V1](ppo_p2_gate_support_collection_protocol.md)。

V1 formal随后产生15/15 records，train/dev分别4,342/3,274 rows，完成78/78与52/52请求，
Safety、replay、logprob、双类和interface覆盖均通过；但train seed772009在959个actor decisions
中有3次shield，违反预注册的`shield_zero`，所以V1保持FAIL且rows不得训练。完整证据见
[V1终态](ppo_p2_gate_support_collection_v1_summary.md)。

fresh V2在任何新reset前冻结：train775001–775009、dev776001–776006，未来资格
774001–774006不动。唯一语义变化是继承历史I3已存在的每回合shield actor fraction≤5%；
其余Safety、replay、schema、class/interface和数量gate不变。V2 runner提交为`3a60c9f`，见
[V2协议](ppo_p2_gate_support_collection_v2_protocol.md)。
