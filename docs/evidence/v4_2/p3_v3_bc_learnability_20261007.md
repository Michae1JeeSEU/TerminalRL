# V3/90D监督可学习性小试（2026-10-07）

## 问题与授权范围

用户在完整mixed采集及“准备90D监督可学习性小试”的建议后要求继续下一步。本轮先完成0reset/0query/0optimizer的数据准备和支持范围审查，冻结下面的具体设置；通过定向测试与独立合成smoke后，仅执行这一次有界、训练集内的BC诊断。它回答“当前90D模型和原专家目标能否在这些已知状态上学出WAIT/RELEASE及条件proposal”，不作为正式I1、闭环性能、泛化、student-only资格或PPO准入结果。没有新增环境采集、audit/Q读取或旧checkpoint迁移。

继续由primary本人负责研究文档、核心实现、诊断及最终判断，worker只执行已指定的测试/SSH/Git命令。使用research-pragmatist与academic-experiments的最小实验流程，不扩大seed矩阵或增加独立审计系统。

## 环境、I0支持范围与student判断（任何真实数据学习之前）

已有采集证据（preexisting_artifact）覆盖静态direct-C²/V3/90D/task-neutral规则场景，三个完整回合合计60请求52N/8E、0F/D/U，无实际安全/hash/耗尽异常；最新mixed 8/24应急，不是全正常降落。原inner耗尽阻塞在这些场景未复现；更广场景和新student访问分布未验证。

| 唯一允许的train case（相对training_outputs/terminal） | 原seed | 完整gate标签 | RELEASE/WAIT |
|---|---:|---:|---:|
| p3_v3_i0_20261004/train_01 | 893001 | 59 | 37/22 |
| p3_v3_i0_v2_20261004/dense_high_launch2 | 893101 | 613 | 90/523 |
| p3_v3_i0_v3_20261007/dense_mixed | 893201 | 374 | 102/272 |

三个case保留各自原collection protocol身份；源文件只读，不回溯合并为一个成功的原I0 cycle。新的consumer必须先确认COLLECTION_OK、完整cycle/全部actor OK标签、无overflow/F/D/U/安全/hash阻塞，再调用已验证90D loader；loader与collector计数必须一致。失败893002/893102、所有development/pilot/smoke/diagnostic均不允许载入。环境seed不等于下文模型初始化seed。

完整输入为1046条相关的actor行，229 RELEASE/817 WAIT，21类proposal，有效低SOC188、retry恢复513；只来自3个独立episode。该范围支持**训练集内的可拟合性诊断**，不足以完成原24train/6audit或证明独立泛化。因此`I0_QUALIFIED=false`、`STUDENT_QUALIFIED=false`、`PPO_ALLOWED=false`继续保留，不借本小试给正式I1或PPO放行。当前尚无新的90D student表现证据。

teacher目标是原pre-state单机first-feasible，不是联合最优/逐机批准。原rule请求、应急接管与执行trace不参与loss或改标；特别是`execution_applied_action`不等于正常release批准。forced、null和未标注不进入gate loss，WAIT的proposal为-1且无proposal loss。保留正常批准/应急事件的解释要求，物理和日志生产源码本轮不改。

## 冻结计算设置

- 协议ID：`p3_v3_90d_bc_in_sample_20261007`；新入口`terminal_marl.training.p3_v3_bc_learnability`，旧88D正式BC入口保持原样。
- 模型：直接实例化现有`HierarchicalActorCentralCriticV42(90, hidden=96)`，actor encoder为现有3个Linear/两次Tanh，gate 2类、proposal 25类。fresh初始化、无adapter/temperature迁移。critic全部requires_grad=false，optimizer仅包含actor encoder及两个head，结束须逐参数确认critic未变。不创建JointPPO容器或加载任何旧checkpoint。
- 唯一模型seed：**91701、91702、91703**，按此顺序各一次；每seed恰好**1000次**更新。它们是初始化/小批次随机性重复，不是3组独立环境样本。
- CPU、torch单线程；Adam lr=3e-4、默认betas(.9,.999)/eps1e-8、weight_decay=0；梯度范数裁剪0.5；无scheduler、entropy项、label smoothing、类别加权、过采样、数据复制。
- batch size128，每轮torch独立generator按seed打乱1046行、不放回，最后不足128的batch保留，耗尽后再打乱。固定1000更新可能在一轮中间结束，不根据loss/指标提前选模。
- loss=原`hierarchical_bc_loss`：gate CE在batch全部有效actor平均，proposal CE只在该batch原teacher RELEASE平均；若batch无RELEASE，proposal项为0。共享encoder正常接受两个head梯度。
- 每seed墙钟上限300s，整个三seed序列最多900s计算预算；非有限loss/梯度/参数、墙钟或输入错误保留ERROR并停止后续，不延时/换lr/补seed/复跑。
- 只记录初始与固定final的完整指标；每100步progress的batch loss仅用于故障观察，不用于选择checkpoint。每个seed只保存一个全新的`final_diagnostic.pt`，带90D/schema/seed/diagnostic_only/qualified=false信息，不覆盖任何已有checkpoint；critic未训练的事实保留。

## 指标、比较与解释

所有读数是**in-sample/resubstitution**，没有把相关行随机拆分后冒充独立验证。整体、原episode、o2m/m2i/i2f、低SOC/高速inner/busyLZ/retry恢复分别报告样本数、gate混淆矩阵、WAIT/RELEASE召回、balanced accuracy、预测类别数量；proposal在teacher RELEASE子集报告top1/top3及逐类支持/召回，另外报告gate与proposal都正确的joint accuracy。空子集或缺失类别的指标为null，不能填0后当失败。

baseline为全WAIT（现有清单78.11% accuracy、50% balanced accuracy）及训练标签的多数proposal；同时保留每seed随机初始化读数。只有总体accuracy提高不代表消除WAIT塌缩，conditional proposal正确也不等于模型能先输出RELEASE。比较固定final相对初始的loss、召回与proposal指标，报告全部3seed，不挑最好一个；不为本诊断再发明全局资格阈值。即使训练集拟合良好，也只支持下一步另冻独立audit/有界student-only闭环；不声称性能或安全收益。

## 验证与命令

新consumer的完整输入拒绝、目标独立于behavior、空子集/全WAIT指标、WAIT proposal零梯度、critic冻结、错误保留与输出不覆盖由合成tests检查，复用33项loader回归。首个本地pytest因测试fixture导入写成顶层模块而在collection报错、0tests执行；primary修正为tests包导入，旧日志保留，不归因生产模型。

执行次序：定向tests → `prepare`真实1046行（无model/optimizer）→ 独立synthetic `smoke`（16个合成90D行、seed17、2更新；不读真实数据）→ primary核对准备与smoke输出 → 本地提交冻结代码/docs → 唯一`fit`。新脚本先小样本输出检查；代码只本地经Syncthing，无上传/serverGit。

```bash
cd /home/user/projects/terminal
# 三条分别在新的prepare、smoke、fit子目录执行，不能复用已有输出。
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/user/envs/terminal_env/bin/python -u -m terminal_marl.training.p3_v3_bc_learnability \
  --root /home/user/training_outputs/terminal \
  --output /home/user/training_outputs/terminal/p3_v3_bc_learnability_20261007/prepare --mode prepare
# 后续两条保持同root，分别改为.../smoke --mode smoke及.../fit --mode fit。
```

预计小网络3×1000更新为数十秒至几分钟，无环境/teacher查询，不预先创建长任务monitor；如实际转为长任务，按用户偏好60分钟heartbeat，正常静默。任何启动先核对Syncthing/no重复输出/真实cwd，启动后确认进程/日志/至少初产物。结果与结论更新本报告和CURRENT_STATUS，仅本地显式提交，保留他人dirty，无remote不推送。

## 实际记录

以上设置在真实数据optimizer创建前冻结；结果只能追加实际执行事实。

### 准备与小样本通过，接受执行本次有界诊断

- 定向tests：本地44 passed / 19.72s，服务器44 passed / 1.56s，均exit0。首次导入失败日志`.tmp/p3_v3_bc_tests_20261007.log`保留；修正后日志为`..._retry1.log`，服务器为输出根`tests.log`。没有因训练结果调参。
- 服务器实际prepare为PREPARED/exit0：仅893001/893101/893201，分别59/613/374个有效actor标签，合计1046、817W/229R；1046个decision_id唯一、原协议身份保持，raw共24360行，23314 forced均未进入训练数组。`trained=false/optimizer_created=false`，没有.pt。primary亲自核对inputs/result，baseline全WAIT accuracy=.7810707457、balanced=.5，多数proposal12占106/229。
- 独立synthetic smoke为SMOKE_OK/exit0：只16个合成90D行（8W/8R）、seed17、2更新，无真实输入episode，actor_changed=true/critic_unchanged=true；loss3.9542682→3.8698366，内部0.4484s。primary核对schema、实际更新预算、原/末指标和新诊断checkpoint路径通过，不把这个合成loss变化当成真实学习结果。
- 原服务器环境Python3.12.3、NumPy2.5.1、Torch2.13.0+cu130；实际使用CPU/单线程。runner SHA `260797096285cdfbba5eb463cfa74b0a58497ebdf24ee56fb6e27c937339dfe4`、test `a2839afa7291ec6f4cb2f6fcbd9c553004758728b3d28219550f1be685bbb010`；原model `abc2f3678b68de7808328ffdeccc5fa7f18d6c21527a38627fdcfc38fd173490`、loader `109df36b42576d87a2137c96b71febfd6e11876191d1be60ec44eaf3b7122d6f`两端一致，后两者本轮不改。
- primary接受按上面的91701/2/3×1000固定设置执行一次in-sample BC诊断；先显式本地提交，再执行新`fit`目录。本次支持范围判断只准许这项诊断，`I0_QUALIFIED/STUDENT_QUALIFIED/PPO_ALLOWED`仍false，不启动其他采集或闭环，不访问audit/Q。上述准备/smoke为newly_run，原3回合为preexisting_artifact。

### 固定三seed终态：学到部分信号，但RELEASE拟合仍弱

代码和协议先提交为`79e114b9db699958ef263f0b8e7f9fb1290ecbab`，随后唯一执行新输出`/home/user/training_outputs/terminal/p3_v3_bc_learnability_20261007/fit`。这是本轮newly_run学习证据，不能与先前无更新的环境验证混为一谈。根result为FIT_COMPLETE、相邻fit.exit=0；Python2774158/wrapper2774156已退出。3个seed均恰好1000更新，actor_changed=true、critic_unchanged=true，分别耗时1.310/0.838/0.835s；每个只生成一个349597字节的final_diagnostic.pt，无中途选模、补seed或延长训练。短任务未创建heartbeat。

primary核对根与每seed result一致、1046个输入ID及各类计数、实际预算、有限指标和诊断checkpoint存在；完整数值在原result中，本地汇总为`.tmp/p3_v3_bc_analysis_20261007.json`。全部指标仅在这三个训练episode的原1046行上计算：

| 初始化seed | loss 初始→固定末尾 | gate accuracy | balanced accuracy | WAIT召回 | RELEASE召回 | proposal top1 / top3 | gate与proposal联合正确 |
|---|---|---:|---:|---:|---:|---:|---:|
| 91701 | 3.9475→1.8339 | 82.79% | 68.40% | 94.00% | 42.79% | 55.02% / 77.29% | 79.06% |
| 91702 | 3.8728→1.8196 | 81.93% | 66.12% | 94.25% | 37.99% | 55.90% / 77.29% | 78.97% |
| 91703 | 3.9506→1.8379 | 82.41% | 68.94% | 92.90% | 44.98% | 56.33% / 75.55% | 78.68% |
| 三seed均值 | 末尾1.8304 | 82.38% | 67.82% | 93.72% | 41.92% | 55.75% / 76.71% | 78.90% |

混淆矩阵按行真值WAIT/RELEASE、列预测WAIT/RELEASE分别为`[[768,49],[131,98]]`、`[[770,47],[142,87]]`、`[[759,58],[126,103]]`。没有塌缩为全WAIT，但229个原teacher RELEASE中仍漏掉126–142个。全WAIT本身就有78.11% accuracy；当前gate accuracy仅比其高3.82–4.68个百分点，联合正确率只高0.57–0.96个百分点。不能只报告约82%的accuracy便称修复成功。

proposal top1是在**真值RELEASE的229行**上计算，不要求预测gate为RELEASE；它高于多数proposal12的46.29%基线，但不代表可以执行这些动作。21个有支持的proposal中，11类在全部3seed的top1召回都为0；这些类合计38个标签。数据完全没有0/6/21/23类，不对这些缺失类的学习能力作结论。安全过滤与K3执行也尚未在新模型闭环中测试。

| context（重叠子集，不可相加） | actor行 / 真值RELEASE | 三seed RELEASE召回范围 |
|---|---:|---:|
| o2m | 181 / 61 | 45.90%–49.18% |
| m2i | 239 / 102 | 42.16%–50.00% |
| i2f | 626 / 66 | 24.24%–33.33% |
| low SOC（当前SOC<.55） | 188 / 50 | 20.00%–28.00% |
| fast inner（i2f且速度≥15） | 602 / 56 | 23.21%–32.14% |
| busy LZ | 1011 / 198 | 28.28%–36.36% |
| retry后首个eligible状态 | 513 / 140 | 23.57%–30.71% |

低SOC、inner和恢复状态虽已采集完整，当前模型仍未充分学出这些状态上的RELEASE。这个结果将后续问题收窄到**现有监督目标、样本分布与模型学习的组合**；不能再把它简单归为采集超时缺行，也没有证据据此回改轨迹或削弱安全过滤。反过来，本试验没有独立闭环，所以也不能唯一归因原P3退化，更不能声称原PPO credit问题已经排除。

三个seed的总loss、gate loss和proposal loss均下降，balanced accuracy相对各自随机初始均提高。个别随机初始化最初RELEASE召回较高，伴随大量WAIT误放行；不能把末尾RELEASE召回低于这种初始值单独称为训练退化。更准确的结论是：**更新生效且存在可学习信号，但固定预算后的训练集拟合仍偏WAIT，尚未形成可用于资格评估的充分依据。**

### 原因边界与下一项有界方案

primary补做一次精确输入冲突检查：按loader实际float32的完整90D值分组，1046行对应1046个不同输入，重复组、gate冲突组及RELEASE proposal冲突组均为0。合成用例先核对signed-zero相等、gate冲突和proposal冲突的统计；真实结果为`.tmp/p3_v3_exact_inputs_20261007.json`，EXACT_INPUT_CHECK_COMPLETE。第一次调用漏PYTHONPATH，在项目模块导入时exit1、未读数据；补充项目PYTHONPATH后唯一实际检查exit0，0model/0optimizer/0reset/0query。这个结果仅排除**当前样本中完全相同输入的自相矛盾监督**，不能证明90D充分可观测，不能排除近邻状态复杂边界或未编码上下文。

817/1046（78.11%）的标签为WAIT，未加权CE的总体指标容易受多数类主导；低SOC及inner子集也有明显类别不平衡。这是可直接测量的事实，与当前高WAIT召回/低RELEASE召回相符，但**尚未证明类别不平衡是唯一原因**。共享encoder的两个loss、有限更新次数、观测压缩或teacher决策边界都仍可能影响拟合，本轮没有通过额外学习来挑选解释。

下一项优先建议为一次**只改gate类别权重的配对诊断**，不直接扩大采集或恢复PPO：保持这1046行、原teacher、模型结构、proposal loss、91701/2/3初始化、batch顺序、1000更新及其余设置，复用本次未加权结果作对照；gate逐行CE乘固定`w_WAIT=1046/(2×817)`、`w_RELEASE=1046/(2×229)`后按batch行数平均，避免同时增加过采样、改阈值或延长预算。先用合成测试核对加权公式和WAIT proposal零梯度，再运行新目录；原结果不得替换。该方案尚未实施。

对照须同时报告RELEASE召回变化、WAIT误放行数、balanced accuracy、条件proposal和上述业务context，不能只追求RELEASE更高。若召回上升但WAIT误放行同步增加、balanced无改善，只能说明代价偏好改变；若分类整体改善，也仍仅是训练集内诊断。观察结果后再判断是否有必要做共享head冲突或观测缺失诊断，不预先扩成多因素矩阵。独立episode泛化、student-only闭环以及联合执行语义的验证仍是后续工作。

本轮技术执行完成，但不授予正式I0/student/PPO资格，`I0_QUALIFIED=false/STUDENT_QUALIFIED=false/PPO_ALLOWED=false`。未消费audit/Q、未启动环境或teacher、full training/硬件、未覆盖旧checkpoint；没有遗留进程或heartbeat。代码与文档仅本地修改和提交，仓库无remote，无法推送。

## 用户继续后的gate类别权重对照（2026-10-07，运行前冻结）

用户在上述结论和单因素建议后明确要求继续下一步，本次授权实施该对照。继续使用research-pragmatist/academic-experiments的最小实验方式：primary亲自实现、读取和分析，worker只测试/SSH/Git。环境与数据支持范围沿用本报告前述审查；1046行只来自三个规则访问episode，前轮拟合不足，因此本次仅允许进一步解释训练集内偏WAIT现象，I0/student/PPO仍未qualified。

- 唯一新因子：gate CE固定权重`[1046/(2×817),1046/(2×229)]`，先逐行相乘再按batch**行数**平均；不使用PyTorch加权CE默认的权重和分母。所有batch（包括只有WAIT或RELEASE的batch）保持该固定权重，不根据局部类别频率重算。全数据两类的总权重均为523，平均行权重为1。
- 原proposal CE在原teacher RELEASE上平均，不加权；共享encoder、Adam3e-4、batch128、clip.5、CPU单线程、91701/2/3初始化、随机排列算法及各1000更新不变。等价训练代码默认路径仍为未加权，新增`--gate-balance`显式启用；没有修改生产PPO/model/loader或teacher。
- 原未加权fit原样复用，不重训；新路径`p3_v3_bc_gate_balance_20261007/fit`。新协议`p3_v3_90d_bc_gate_balance_20261007`。全部参数fresh初始化，不加载旧checkpoint，每seed只存一个新诊断checkpoint。
- 每seed optimizer创建前将完整初始指标与原同seed比较，要求精确相同；同时核对输入ID及原episode摘要。固定seed及不变的独立随机generator保持同一batch顺序，初始前向统计不消耗随机数。若不一致保留ERROR并停止，不掩盖配对失败。
- 各seed仍1000更新/300s上限，不增加第二组超参数、不补seed或延长预算。比较指标仍为**原未加权定义**的loss/混淆矩阵/balanced/条件proposal及joint准确率，另存initial/final_training_objective以说明实际优化目标；加权目标与原未加权loss不得直接当作同一量相减。
- 所有3seed及原分组完整报告：RELEASE召回、WAIT误判为RELEASE的数量与比例、balanced、proposal和joint；同时看低SOC/inner/retry等重叠子集。此处“误放行”仅指预测与原teacher WAIT不一致，**不是实际安全违规**；本试验没有执行环境动作。原teacher仍是pre-state单机可行标签，不变为联合批准标签。
- 测试先核对权重公式/分母、gate梯度缩放、proposal梯度不变、WAIT无proposal梯度、相同初始化和输出、配对错误在optimizer之前停止。然后新prepare与独立16行synthetic smoke各一次，primary核查后本地提交，才执行真实加权fit。准备/合成结果不作为真实训练效果。
- 执行命令沿用上文module/root，新增`--gate-balance`，output改为`/home/user/training_outputs/terminal/p3_v3_bc_gate_balance_20261007/{prepare,smoke,fit}`，显式cwd及原服务器Python/三个环境变量。先查Syncthing SHA及无重复进程/输出；现有未加权fit的inputs/protocol/result两端SHA已匹配，原进程已退出。预计秒级计算，无长任务heartbeat；若异常转为长任务按用户60min偏好管理。

本次对照用于识别类别权重的作用，不凭结果直接部署、恢复PPO或宣称泛化。若balanced/重要子集改善且proposal没有明显损失，支持在后续训练候选中保留此设置；若仅提高RELEASE召回但整体区分能力没有改善，则记录代价偏好转移。任何后续独立episode/闭环范围另据本轮结果确定，不自动消费audit/Q。

### 加权对照准备结果

本地48tests通过12.83s、服务器相同48tests通过1.55s，exit0；比前轮44项增加4项权重/配对相关测试，不能相加成独立92项。新prepare为PREPARED/exit0、1046行817W229R、无optimizer；inputs SHA与原未加权准备和fit完全相同。新synthetic smoke为SMOKE_OK/exit0，16合成行、seed17、2更新，实际权重`[0.6401468788249693,2.2838427947598254]`，实际加权目标4.2807636→4.1809330、actor变/critic不变，新checkpoint存在。primary亲自读取并核对协议/结果，两端关键文件SHA一致；以上是newly_run验证，不是实际数据效果。运行前冻结runner SHA`3f9f83bcf6463354cc31dd4e531095b79538300b4bc39c115396eec6d589522a`、tests SHA`27efa650c700a59def5d8ed5f65bdf09f22a7912f4fb2e58301496c7e220ddc1`，原model/loader SHA保持不变。primary接受按上面唯一设置在提交后执行三seed加权fit。

### 初始比较的技术失败与修正（未发生真实训练更新）

首版代码/协议commit`6ac68ee18be01211392be95fa17da4096ef56c1d`后，`.../p3_v3_bc_gate_balance_20261007/fit`唯一启动即ERROR/exit1：`paired initialization metrics differ`，seeds=[]、没有per-seed result/checkpoint；Python2777583/wrapper2777581均退出。错误发生在`measure`之后、Adam创建之前，参数更新为0。该失败目录完整保留，不能记作一个学习失败seed或覆盖成成功。

primary定位为比较实现缺陷：运行时`Counter`的proposal target_counts/prediction_counts键为int，持久化JSON再读回后为str，直接比较字典必然失败。独立只读seed91701证明`initial_check.json`（newly_run，exit0）为raw_equal=false、json_form_equal=true；全部初始指标转为JSON表示后精确一致，包括loss3.9474971294、gate .6963557601、proposal3.2511413097。该证明0optimizer/0更新/0reset/0teacher query，没有读取checkpoint或重新训练。

修正仅把新测初始记录转换为同一JSON表示再作**精确比较**，没有放宽浮点容差、跳过比较、变更样本/模型/权重/更新预算。新增含Counter整数键的序列化正向回归，原不一致先于optimizer拒绝的测试保留；本地49tests通过12.68s、服务器49tests通过1.56s，均exit0。新runner SHA`933b5696cbb0b818eabb789e0473df48a4e199b90ece7a4f1d698e8c1804e37d`、test SHA`a0d650f8bdcd68f30fdc230bd2ae6849f14f03ca67d9a5ebc27badd8ff94a6a7`已同步核对。

用户本次授权的真实加权学习尚未执行任何更新；primary接受修复起始检查后在**新目录`fit_serialization_fix`**执行原定3seed×1000更新，旧fit和prepare/smoke/proof均不重跑。本轮唯一因子仍为gate类别权重，全部结果必须记录，包括上述技术失败；不是根据拟合结果调参、选择seed或延长训练。先提交这项修正，再启动新目录。

### 加权结果后的最小只读解释检查（执行前记录）

修正版已完成三seed×1000更新，初读总体RELEASE召回提高但WAIT误判增加、联合正确率下降。为区分“gate排序区分能力提高”和“判定边界移动”，primary补充一次只读评分检查：只加载这3个原未加权及3个加权诊断末尾checkpoint，重算原1046行的末尾指标并要求与保存结果精确相同；不创建optimizer或更新参数。计算gate logit差的pairwise ROC AUC（相等分数计半分），并只在原未加权模型使用**唯一解析阈值`229/1046≈0.21893`**报告分类结果。该值由已冻结权重的`w_WAIT/(w_WAIT+w_RELEASE)`推导，不扫描/挑选阈值，不保存新策略，不把训练集阈值对照称为部署建议。理想的加权CE可以移动logit截距；有限网络/有限更新下不假定完全等价，故同时读取AUC。

范围为overall及已存在的o2m/m2i/i2f/低SOC/高速inner/busyLZ/retry子集，无新环境、teacher或audit/Q。先用可分/全tie两种合成scores检验AUC，再运行同服务器前向；各checkpoint读取前后SHA不变。脚本输出新`score_probe.json`及相邻log/exit，失败保留，不追加学习。此检查由本轮已观察的取舍直接触发，不增加训练臂或超参数试验。

### 加权对照实际终态与分组判断

修正commit`08befbd6dd169064d943700e84645fb1f114d1bf`后，`fit_serialization_fix`为FIT_COMPLETE/exit0，Python2778464/wrapper2778462已退出。91701/2/3的完整初始指标与原同seed精确相同，均完成1000更新、actor_changed=true、critic_unchanged=true；耗时1.339/0.879/0.881s，未扩预算。inputs/protocol/result三文件两端SHA一致；primary亲自读取终态与完整逐组统计，并核对原输入、三seed预算及资格标志。完整比较保存在`.tmp/p3_gate_balance_analysis_20261007.json`（PAIRED_RESULT_CHECK_COMPLETE）。本节加权结果为newly_run，原未加权对照为preexisting_artifact；以下均是原1046行的训练集内测量。

| seed | 加权gate accuracy | balanced | RELEASE召回 | WAIT误判RELEASE（/817） | proposal top1/top3 | joint正确率 |
|---|---:|---:|---:|---:|---:|---:|
| 91701 | 75.43% | 77.83% | 82.10% | 216 | 54.59% / 77.29% | 67.40% |
| 91702 | 75.81% | 77.92% | 81.66% | 211 | 54.15% / 76.86% | 67.69% |
| 91703 | 74.57% | 77.91% | 83.84% | 229 | 55.02% / 74.67% | 66.25% |

加权混淆矩阵依次为`[[601,216],[41,188]]`、`[[606,211],[42,187]]`、`[[588,229],[37,192]]`。实际优化的加权目标分别3.94199→1.95203、3.87708→1.92846、3.94576→1.94560；末尾原未加权定义loss分别1.96929/1.93818/1.98146。二者定义不同，不拿加权目标的绝对值与原未加权目标宣称优化优劣。

| 三初始化seed均值 | 原未加权 | 加权 |
|---|---:|---:|
| gate accuracy | 82.38% | 75.27% |
| balanced accuracy | 67.82% | 77.88% |
| RELEASE召回 | 41.92% | 82.53% |
| WAIT误判为RELEASE的比例 | 6.28% | 26.76% |
| RELEASE precision | 65.19% | 46.38% |
| 条件proposal top1 / top3 | 55.75% / 76.71% | 54.59% / 76.27% |
| gate与proposal联合正确 | 78.90% | 67.11% |

全局balanced提高10.06个百分点，但不能据此说各接口都改善。原样本中inner占626/1046行且560行为WAIT，类别分布随接口明显变化：o2m真值RELEASE61/181、m2i102/239、i2f66/626。一个全局权重对这三个条件分布施加同样的相对放大，具体表现为：

| 重叠context；三seed均值 | RELEASE召回 原→加权 | WAIT误判率 原→加权 | balanced 原→加权 |
|---|---:|---:|---:|
| o2m | 48.09%→76.50% | 13.61%→57.78% | 67.24%→59.36% |
| m2i | 46.08%→92.16% | 20.92%→80.78% | 62.58%→55.69% |
| i2f | 29.80%→73.23% | 1.13%→6.90% | 64.33%→83.16% |
| 低SOC | 25.33%→75.33% | 11.84%→42.03% | 56.75%→66.65% |
| 高速inner | 28.57%→70.83% | 0.79%→6.65% | 63.89%→82.09% |
| busy LZ | 32.83%→79.80% | 5.90%→26.40% | 63.46%→76.70% |
| retry恢复 | 27.62%→76.90% | 11.44%→49.96% | 58.09%→63.47% |

分episode的balanced均值也非一致改善：893001为72.03%→62.47%，893101为65.29%→80.25%，893201为66.48%→71.04%。这些仍是同三回合内的相关行；不能把三个初始化重复当独立场景统计显著性。

**本轮判断：全局类别加权有效改变了决策取舍，却不是完整修复。** 模型更容易输出RELEASE，但o2m/m2i对原teacher WAIT的识别明显变差，联合动作正确率下降；proposal识别没有改善，稀有proposal问题仍在。WAIT误判增多不等于发生实际安全违规，本次无闭环；其对安全层拒绝、等待能耗、正常/应急结局的影响仍未知。不把这些checkpoint晋级为student或恢复PPO，也不通过放松安全过滤抵消误判。

### 只读评分检查：大部分召回变化可由移动判定边界复现

`score_probe.json`为SCORE_PROBE_COMPLETE/exit0，六个诊断checkpoint重新前向均与保存的末尾完整指标精确相同，读取前后SHA不变。合成AUC自测通过；0optimizer/0参数更新/0reset/0teacher/0checkpoint写入。两端JSON SHA相同（`32e7216d4d51ed321a2d2fc5d1d47943c4212aea7ef081fd80204be15e87d31b`），无遗留fit/probe进程；primary读取全部分组结果后判断如下。它是newly_run只读测量，没有增加训练臂。

| 训练集内三seed均值 | 原模型默认判定 | 原模型解析阈值0.218929 | 加权模型默认判定 |
|---|---:|---:|---:|
| RELEASE召回 | 41.92% | 79.62% | 82.53% |
| WAIT误判率 | 6.28% | 26.23% | 26.76% |
| balanced accuracy | 67.82% | 76.69% | 77.88% |
| joint正确率 | 78.90% | 67.62% | 67.11% |
| gate ROC AUC | 0.86066 | 同一模型、不随阈值改变 | 0.86873 |

这表明**大部分召回变化可以在原模型上仅移动判定边界复现**。加权模型整体AUC有约0.0081的额外提高，不能说它只改变了截距；但也不能把约40个百分点的RELEASE召回提升全算作学到了更好的可行性表示。解析阈值只有这一个事先推导值，未扫描或选择最优阈值，也未写回任何策略配置。

分组AUC揭示剩余问题的位置（原→加权）：o2m `0.71330→0.72272`，m2i `0.67676→0.67332`，i2f `0.93136→0.94401`，低SOC `0.70150→0.71420`，retry恢复 `0.68666→0.70004`。m2i在三个加权seed的AUC都比原同seed略低；因此不能只靠改变全局RELEASE偏好解决切环状态的区分问题。AUC仍是训练集内相关状态的排序指标，不能当独立泛化或闭环安全证明。

### 本次收束及下一项优先方案

当前有三个明确事实：环境退出与采集成本修复在已测范围成立；fresh模型更新正常且能学习一部分标签；单一全局类别加权会使切环接口更容易误判WAIT，不能作为整体修复直接启用。所有加权checkpoint仅保留为诊断产物，生产默认损失没有切换；本轮未改安全层/轨迹/动力学/teacher标签，I0/student/PPO资格仍false。

**下一项优先检验共享encoder中的proposal监督是否妨碍gate学习。** 原gate在m2i等状态的训练集AUC偏低，加权几乎未改善其排序；这支持把诊断重点从全局类别比例转到表示学习，但尚不能归因“梯度冲突”或“90D信息缺失”。建议仅做一个有界对照：回到原未加权gate，保持同1046行、91701/2/3、1000更新/300s和全部其他设置，只阻断proposal loss回传到共享encoder的梯度；proposal head本身仍按原RELEASE CE学习，critic冻结。与已有未加权基线比较gate逐接口AUC/混淆矩阵及proposal top1/top3，新的输出全部保留，不调阈值或再改类别权重。

该单因素设置若在三个seed中一致改善gate拟合，将支持继续处理共享表示的任务竞争，同时须保留proposal性能代价；若未改善，则不能据此证明观测不足，后续应针对m2i的固定误判状态审查可观测量与teacher依赖，并区分有限优化/容量与缺失信息。上述对照**尚未实现或执行**，本轮不自动再训练。不得把按接口再调多组权重、增加epoch、换网络同时展开，以免无法解释变化。独立场景和student-only闭环验证仍未完成，不消费audit/Q或恢复PPO。

本轮实际修改为离线诊断入口、对应tests、本报告与CURRENT_STATUS最新块。最终49项定向tests在local/server均通过；首次起始比较失败及其0更新证明完整保留，修正后3个固定seed实际训练正常完成。所有任务结束，无需heartbeat。Git仅本地，未配置remote，未推送；其他任务dirty保留。
