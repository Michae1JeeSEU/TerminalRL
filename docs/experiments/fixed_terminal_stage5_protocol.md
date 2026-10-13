# 阶段5：当前状态联合候选筛选验证协议

状态：**四类评分器、训练接口及针对性小fixture测试已实现；正式训练与本协议评价矩阵尚未运行。** 本文按主对话确定的方向A冻结范围，落实[用户阶段5计划](../fixed_terminal_stage5_learning_validation_plan.md)，保留[阶段4结果](../evidence/fixed_terminal_stage4/results.md)及原始数据，不按后续结果调整模型、种子、表示或预算。

## 研究对象与既有事实

研究当前可见状态下，整个联合候选批次的排序/价值筛选能否以较低完整成本保留有限池闭合参考的结果。目标仅为已到达机群在固定FCFS continuation下至episode 240 s的价值；不训练长期Q、期望未来Q或teacher动作，不改变[物理合同](../fixed_terminal_problem_definition.md)、独立安全、能源、容量、pad与应急语义。保留固定93维特征，阶段4已完成的train/val用于本轮四类模型训练，不重新选择特征。

阶段4的108个离线context中，rank与beam加入anchors后各103次保留最佳key；各自5次miss均在delay/exposure/energy后位，全部108次与full比较的N/E/F/C相同。rank独有的3次命中相对beam仅节省能耗0.002846/0.008996/0.008996 SOC；beam独有的3次命中中，rank各多104.719755 s暴露与正常延迟总和。pair正确率改善没有在这些固定behavior快照改变N/E/F，不能据此解释另一自适应在线轨迹中的额外失败。

value加入anchors后98次保优，逐context raw J regret求和为`[0, 2, 1, 1, 701.549669, 580.940798, 0.53941459]`。这是分量诊断差额，不能压成未经定义的标量regret，也不能假定每个分量非负。阶段4存在3对同context、同93维表征却不同J的候选，这是信息压缩别名；存储标签由确定性FCFS生成，不能将MAE直接归因随机标签噪声。本轮保留表示限制做受控比较。

## 冻结数据域与信息边界

|用途|seed|范围|
|---|---|---|
|train|4101、4102、4103|复用阶段4：27场景、108context，t0/5/15/90|
|validation|4201|复用阶段4：9场景、36context，t0/5/15/90|
|新test|5301、5302、5303|27个标准场景及3个初相变体，共30场景|
|pilot|5001|只指定小样本验证流程，不并入正式数据|
|模型初始化|5401|每类模型仅一次固定种子训练|

train/val按完整场景seed分组，所有同状态候选留在同集合；scaler只拟合train。正式训练严格读取阶段4完整108/36context，不遍历任何test目录。四模型冻结后才首次生成530x test；旧430x test仅用于明确标注的development诊断，不能充当本轮独立test或据此调参。

标准场景为3seed×3load×3SOC。每个seed另有一个medium/medium的`initial_rotated`变体：每环初始角度额外加`(ring_index+1)*pi/7`并取mod 2pi；其余请求、SOC、到达路径和物理配置与对应标准场景完全配对，并沿用初态安全检查。SOC/load test是同profile family的新随机样本；初相是另列的留出变体，不混称外分布泛化。

部署输入只含当前已到达状态、既有预约与可见候选特征；禁止未到达身份/SOC/time、seed/load/SOC档标签、方法来源、ID数值或真实未来结局。闭合标签是累计`J=(-F,N,-E,-C,-T,-D,-energy)`及相对HOLD的ΔJ；计数整数精确，T/D round8、SOC round10比较，raw值保留。词典序胜出不等于Pareto改善。

## 四类固定评分器

|名称|模型与训练目标|文件|
|---|---|---|
|linear_rank|93→1线性层；context等权的全部非tie pair logistic loss|linear_rank.pt|
|mlp_rank|93→64→64→1，ReLU；同一rank loss|mlp_rank.pt|
|gradient_boost|GradientBoostingRegressor，100树、depth3、learning_rate0.05、random_state5401、squared_error|gradient_boost.joblib|
|value|93→64→64→7，ReLU；标准化ΔJ的逐候选MSE|value.pt|

两个神经rank沿用阶段4的pair构造与context等权：每个有非tie pairs的context总权重相同，真正同key的pair跳过。三个神经模型均Adam、lr0.001、weight_decay0.0001、batch256、300epoch、CPU threads1。value的输入/输出scaler均train-only，损失保持candidate加权；预测反标准化后前四计数用Python round最近整数（.5 ties-to-even），余三维保留raw值，再词典序排序。rank按标量降序，平分用固定action_key。

gradient_boost目标为context内真实词典序key的不同等级，按升序映射至0..1；同key同分，只有一等级时为0。每个context全部候选sample_weight合计1；validation loss为context等权ordinal MSE。一次拟合100树，通过staged_predict检查第1至100棵的前缀，选择相应树数，保存该前缀，不另行重训。

四类checkpoint统一以**validation informative best-key-retention@TOTAL B5**最大者优先，再取该模型validation loss最小者，再取最早epoch/tree。informative为同池存在不同真实key；没有informative context时召回记null，选择仅按loss及最早迭代，不能声称有效验证召回。模型loss量纲不同，不横向比较loss数值。单一模型seed不支持训练稳定性结论。

TOTAL预算先按HOLD、现场FCFS anchors去重纳入，再按候选排序补至B或池大小；anchors占用预算，不是topB之后额外添加。validation固定B5；B3/B5/B8仅用于冻结后的评价。hybrid使用mlp_rank，不训练第五个模型。

## 冻结评价矩阵

统一标准池上限14，full_reference评价全部根，仅是该池、该闭合需求假设下的有限参考。其他筛选保持TOTAL B口径。所有方法最终提交仍由独立可行性层复查，并调用共同应急逻辑；完整预约不撤销。

### 完整episode

30场景×11arms=330episode，每场240 s、5 s决策周期，共48次决策。

|arm|方法|TOTAL预算/窗口|
|---|---|---|
|beam_b3、beam_b5、beam_b8|beam|3、5、8|
|full_reference|full|池内全部根|
|rolling20_b5、rolling60_b5|rolling|5；20/60 s|
|linear_rank_b5、mlp_rank_b5、gradient_boost_b5、value_b5|对应冻结评分器|5|
|hybrid_b5|hybrid，使用mlp_rank|5|

rolling两个窗口共用外层root pool及TOTAL B5，但保留旧joint continuation与commitment surrogate。记录后续内部搜索与simulation steps，不能把根数相同冒称工作量相同。完整报告全请求N/E/F/C、应急比例、暴露、正常条件延迟、能耗、间隔违反/未认证、着陆最低SOC和分项在线时间；不以推理时间代替完整成本，不以间隔无违反代替零能源失败。

### 固定状态离线质量

30场景的joint_window共同behavior prefix，在t0/15/90决策前取90个snapshot，生成同状态全池closed标签。beam、linear_rank、mlp_rank、gradient_boost、value、hybrid分别在B3/5/8评价质量预算曲线，保留退化与informative分母。标签查表只用于质量比较，不能将查表时间称为在线求解时间。

### 真实同状态计时

固定24states：标准low/high、medium/medium、high/low×3seed×t0/15为18个，加3个初相变体×t0/15为6个。每个state的11arms以预定确定性轮换顺序执行，逐个deepcopy，每arm实际重新生成池并进行所需rollout，无跨arm缓存。记录完整生成、特征、推理、rollout、最终复查/应急与total，模型加载另列。该面板用于同状态计时，和各自episode所到状态的累计时间分开解释。

### 有限池敏感性与continuation诊断

largepool上限32，仅在上述3个标准组合×3seed的t15共9context作离线质量预算曲线，不扩展线上矩阵。

阶段4continuation诊断固定9个profile的seed4303、t15：重建原joint behavior并对照snapshot；取既有beam/rank/value/full-best/HOLD根的并集，去重≤5，在完整原需求路径上用FCFS continuation作realized_future重放，与closed值对照。这仅是ex-post敏感性，不是长期训练标签、未来输入或本轮独立test。由不同整策略轨迹观察到的差异不能替代同状态干预证据。

## 实现接口、持久化与运行状态

配置为`terminal_marl/configs/fixed_terminal_stage5.py`，含MODEL_NAMES/MODEL_FILES、BUDGETS、两个POOL_LIMIT、SNAPSHOT_TIMES、ONLINE_ARMS及scenario_jobs。`ValidationScorer.load(path)`、`.kind`（value/rank）、`.predict(feature_rows)`返回逐候选预测向量。训练实现为`core/fixed_terminal/validation_models.py`；TOTAL-budget辅助`total_budget_indices`与validation指标按上述口径执行。

```bash
python -m terminal_marl.scripts.train_fixed_terminal_stage5 --stage4-root STAGE4_RESULTS_ROOT --output NEW_STAGE5_MODELS
python -m terminal_marl.scripts.train_fixed_terminal_stage5 --stage4-root PILOT_FIXTURE_ROOT --output NEW_SMOKE_MODELS --smoke
```

STAGE4_RESULTS_ROOT含`data/train`、`data/val`；CLI不提供正式超参数覆盖。smoke专用输入两目录的metadata均须明确pilot5001，允许复用同一小fixture，只跑3epoch/trees，所有产物标smoke_only；拒绝4001或任何正式seed，不伪写seed绕过阶段4检查，也不作validation泛化结论。已有输出目录一律拒绝覆盖。

训练保存protocol.json、training.json、每模型curve.jsonl/training.json及四模型。最终manifest.json schema为`fixed-terminal-stage5-model-manifest-v1`，记录producer、status、frozen、smoke_only、test_read、版本、seed、预算、文件SHA256及selected_iteration；完成保存重载预测一致核对后标`roundtrip_predictions_equal=true`。正式评价须统一检查完成、冻结、非smoke、未读test及文件hash，不接受流程smoke作为正式模型。Torch用weights_only安全加载；joblib仅加载本地已知来源的自产模型，先核对manifest及hash，该检查不使外来pickle变安全。

实现版本由本地核定后通过`TERMINAL_SOURCE_COMMIT`环境变量传入；训练代码不运行服务器Git。CLI在导入数值库前设OMP/MKL/OPENBLAS线程1，训练与预测显式torch threads1；dispatcher继续保证线程限制。

依赖已准备：本地sklearn1.9.1、SciPy1.18.1、NumPy2.4.4、torch2.5.1+cpu；服务器sklearn1.9.1、SciPy1.18.1、NumPy2.5.1、torch2.13.0+cu130，正式仅CPU。`requirements-stage5.txt`引用阶段4依赖并固定scikit-learn1.9.1；实际版本写入protocol。正式训练与评价统一使用服务器runtime，本地仅单测/smoke。服务器已有完整阶段4train108/val36数据，路径`~/projects/terminal/.worktrees/arrival-restructure/tmp_stage4_results`。

服务器28 logical CPU，拟20workers、每进程threads1；具体并发须由主对话依据小pilot内存/时长核对后确定，当前不把拟定值写成已运行事实。服务器正式输出写入`$HOME/training_outputs/terminal`，按既有同步规则取回，不在服务器提交Git。

当前针对性模型单测8项通过（最终4.68 s，exit0），实际小fixture训练/保存加载四模型，覆盖TOTAL B5去重、ordinal ties、checkpoint排序、test隔离和覆盖拒绝。该检查未运行正式训练或评价矩阵，所有模型仅pilot流程产物。

## 固定任务分发接口

`terminal_marl/scripts/run_fixed_terminal_stage5.py`默认只列任务，不建输出目录；`--execute`才启动独立子进程。test为30snapshot+330episode，共360任务，采用固定Random(5401)打乱执行顺序，仅改变调度次序，不改变场景内容或随机流。continuation为9个旧4303诊断，可独立于新模型训练执行，不加载或校验stage5模型。

```bash
python -m terminal_marl.scripts.run_fixed_terminal_stage5 --phase test --models FROZEN_MODELS --output NEW_OUTPUT_ROOT --workers 20
python -m terminal_marl.scripts.run_fixed_terminal_stage5 --phase test --models FROZEN_MODELS --output NEW_OUTPUT_ROOT --workers 20 --execute
python -m terminal_marl.scripts.run_fixed_terminal_stage5 --phase continuation --stage4-root STAGE4_RESULTS_ROOT --output OUTPUT_ROOT --workers 20 --execute
```

上例并发20为拟定默认值，正式执行由小pilot核对后主对话指定；不以文档命令表示已经授权或启动。test开始和结束均验证formal manifest及所有模型文件hash一致，任何失败最终exit1；continuation只使用旧labels/selections和重建snapshot对照，不设置新模型前置条件。训练与子进程继承`TERMINAL_SOURCE_COMMIT`；OMP/MKL/OPENBLAS线程均1，不执行服务器Git或源码传输。

ROOT下每job独立输出至`episodes/arm-load-soc-seed-variant`、`snapshots/load-soc-seed-variant`或`continuation/load-soc-4303`。`logs/`保存每job stdout/stderr；`execution_<phase>.json`原子更新PID、命令、起止、退出码、耗时与计数，`progress_<phase>.jsonl`写完成事件并flush。总日志只打印简短启动/完成事件与phase摘要。已存在job目录、日志或phase状态一律拒绝重复execute，无自动恢复、重训或重跑；失败处理由主对话决定。

分发器5项小测试通过（0.82 s、exit0），使用真实短子进程和冻结检查fixture验证360/9覆盖、并发、code0/非零、覆盖拒绝、哈希变化与continuation独立性。测试未启动交通仿真或正式模型训练。

## 解释边界

本轮仅比较冻结表示、目标、候选池与预算中的方法；新场景seed数和模型seed数有限。闭合标签并不代表未来需求期望，有限池参考不是全动态最优，计算成本不代表算法固有下限；离线交通时钟不计CPU等待。结果可以支持或不支持本范围中的学习收益，不能宣布普遍不需要学习或期望长期价值优越。后续科学结论由完整证据给出，当前没有正式效果结论。
