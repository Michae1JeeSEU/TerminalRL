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
