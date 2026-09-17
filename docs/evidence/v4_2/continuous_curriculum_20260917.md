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
