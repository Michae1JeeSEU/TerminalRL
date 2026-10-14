# 阶段7结果：当前Learning to Search配置停止扩展

更新日期：2026-10-14。主对话已审查正式strict summary、全部质量/计算分组与冻结比较精度，完成研究验收：**B未成立，按A停止扩展当前Learning to Search配置；C未观察到新选中值超过有限full参考。** 保留传统联合优化及beam_adaptive作为strong工程参考，保留full/B3/B5/B8的质量与成本tradeoff和本阶段实验资产。不自动扩大Q/value/模型、新训练或下一阶段；这一判断不表示所有learning方法普遍无效。

数据36jobs、108train/36val context、唯一GB fit、264episodes、15panel generation/30states/240rows及严格汇总均完成，四模型文件hash不变。研究运行源码统一为`c892d82a77ec4647e0c9f77fe3f73786ae57bab2`。最终文档提交SHA及结果包尚待实际版本交付，不提前记录ZIP或推送完成。运行命令、PID与失败历史见[run_status.md](run_status.md)。

## 在线质量与完整请求分母

每方法33场、1,066请求，保留normal/emergency/failed/censored全部结局。每方法仍有2次能源失败、合计16次；spacing违反与未认证均0，最终执行checks全部feasible，全局最低落地SOC=.06162345028264405。0jobfailed只表示任务正常退出，不能写成0能源失败。SOC是既定合成研究能耗单位。

| 方法 | N | E | F | C | 正常条件delay秒 | 正常条件energy SOC |
|---|---:|---:|---:|---:|---:|---:|
| beam_b3 | 562 | 27 | 2 | 475 | 77.810676 | 0.096632023 |
| beam_b5 | 563 | 26 | 2 | 475 | 77.938977 | 0.096765601 |
| beam_b8 | 559 | 26 | 2 | 479 | 77.657769 | 0.096414503 |
| beam_adaptive | 567 | 24 | 2 | 473 | 77.729331 | 0.096642449 |
| full_reference | 570 | 23 | 2 | 471 | 76.843453 | 0.095671248 |
| learned_pruning | 568 | 23 | 2 | 473 | 77.387630 | 0.096251537 |
| learned_beam_width | 567 | 24 | 2 | 473 | 77.490891 | 0.096319458 |
| hybrid | 568 | 24 | 2 | 472 | 77.497071 | 0.096339089 |

正常条件delay=正常delay总和/N，正常条件energy=正常落地请求energy总和/N，按请求加权，不取场景mean的平均。下表是全部七个J分量的累计量：`J=(-F,N,-E,-C,-exposure,-normal_delay,-all_request_energy)`，后三项保留全部请求/正常请求的原定分母。

| 方法 | J1 | J2 | J3 | J4 | J5 | J6 | J7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| beam_b3 | -2 | 562 | -27 | -475 | -124163.486287105 | -43729.600015018 | -104.447341413 |
| beam_b5 | -2 | 563 | -26 | -475 | -124286.288520828 | -43879.644097015 | -104.515560115 |
| beam_b8 | -2 | 559 | -26 | -479 | -124511.281080624 | -43410.692812595 | -104.613188109 |
| beam_adaptive | -2 | 567 | -24 | -473 | -124026.588027908 | -44072.530441723 | -104.299533433 |
| full_reference | -2 | 570 | -23 | -471 | -123335.732551454 | -43800.768106194 | -103.761621420 |
| learned_pruning | -2 | 568 | -23 | -473 | -123813.251255875 | -43956.173950650 | -104.121014446 |
| learned_beam_width | -2 | 567 | -24 | -473 | -123807.704581717 | -43937.334961860 | -104.116020474 |
| hybrid | -2 | 568 | -24 | -472 | -123747.650949774 | -44018.336262962 | -104.081396894 |

完整160组、加权分母及原精度数值见[episode_groups.csv](episode_groups.csv)与[compact_evidence.json](compact_evidence.json)。三学习相对beam_adaptive全局质量条件均通过：每场配对F不增、aggregate E不增、aggregate N比例不降（epsilon0）及确定性安全条件成立；相对full三者均未通过，N分别少2/3/2，pruning E相同，width/hybrid E多1。相对B3/B5/B8质量条件通过，但计算明显更高。

质量门槛通过不表示全J每场不退化。冻结比较`value_comparison_key`先四int，exposure/delay round8，energy round10。对adaptive的win/tie/loss：pruning6/25/2、width3/30/0、hybrid5/26/2；对full为8/19/6、3/26/4、7/20/6。Pruning与hybrid的两loss均在7303 low/high和low/medium：F/N/E/C及exposure/delay在冻结精度相同，实际all-request SOC energy各多.00473215917222114。原始scratch tuple曾让2.27e-13的exposure差遮蔽后续能耗亏损，不能采用scratch胜场；本记录以strict summary为准。

## 共同同状态完整成本与接受条件

Panel为独立同state、单worker、threads1、旋转方法顺序，无跨方法缓存。所有首次生成、partial checks、features/inference、累计unique rollouts、恢复到128、最终验证、应急及bookkeeping都计入完整wall；加载、IO、环境step和事后label另列。共同full-defined分层为24非退化与6单根状态，不以方法自身单根删掉困难状态。

| 方法（24非退化states） | candidate checks | expanded nodes | unique rollouts | 完整wall秒 |
|---|---:|---:|---:|---:|
| beam_b3 | 55967 | 458 | 109 | 23.377302 |
| beam_b5 | 75769 | 680 | 154 | 33.482054 |
| beam_b8 | 105149 | 997 | 223 | 48.684085 |
| beam_adaptive | 269956 | 6261 | 243 | 88.533887 |
| full_reference | 458753 | 10840 | 314 | 116.752668 |
| learned_pruning | 303771 | 5133 | 294 | 93.604029 |
| learned_beam_width | 259604 | 5448 | 233 | 83.919485 |
| hybrid | 271472 | 5133 | 235 | 83.181296 |

Primary要求同一reference的actual candidate checks与完整wall均≤90%；旧rollout+wall双阈值仅辅助。对adaptive，pruning checks+12.53%、wall+5.73%、roots+20.99%；width checks−3.83%、wall−5.21%；hybrid checks+0.56%、wall−6.05%，均未达到主双10%门槛，辅助亦未通过。855个compute分组中，对adaptive没有任何子组primary通过。

| 学习方法 | 传统reference | 在线质量 | checks比 | wall比 | primary计算 |
|---|---|---|---:|---:|---|
| learned_pruning | beam_b3 | 通过 | 5.427681 | 4.004056 | 未通过 |
| learned_pruning | beam_b5 | 通过 | 4.009173 | 2.795648 | 未通过 |
| learned_pruning | beam_b8 | 通过 | 2.888958 | 1.922682 | 未通过 |
| learned_pruning | full_reference | 未通过 | 0.662167 | 0.801729 | 通过 |
| learned_pruning | beam_adaptive | 通过 | 1.125261 | 1.057268 | 未通过 |
| learned_beam_width | beam_b3 | 通过 | 4.638519 | 3.589785 | 未通过 |
| learned_beam_width | beam_b5 | 通过 | 3.426256 | 2.506402 | 未通过 |
| learned_beam_width | beam_b8 | 通过 | 2.468916 | 1.723756 | 未通过 |
| learned_beam_width | full_reference | 未通过 | 0.565891 | 0.718780 | 通过 |
| learned_beam_width | beam_adaptive | 通过 | 0.961653 | 0.947880 | 未通过 |
| hybrid | beam_b3 | 通过 | 4.850573 | 3.558208 | 未通过 |
| hybrid | beam_b5 | 通过 | 3.582890 | 2.484355 | 未通过 |
| hybrid | beam_b8 | 通过 | 2.581784 | 1.708593 | 未通过 |
| hybrid | full_reference | 未通过 | 0.591761 | 0.712457 | 通过 |
| hybrid | beam_adaptive | 通过 | 1.005616 | 0.939542 | 未通过 |

相对full，三学习计算主条件通过，但在线质量不通过，不能给B；相对较窄beam，质量通过但计算不通过。15个全局比较没有同时满足质量与primary计算者。[acceptance.csv](acceptance.csv)保留全部300质量分组及可匹配同状态计算，缺失的panel组留空；[generalization.csv](generalization.csv)保留160 episode组、300质量条件与855计算组，不挑选有利子组。分组轴/范围重叠，不能相加。

六个共同单根状态每方法各执行6个unique rollouts，成本单列，不误用阶段6零rollout fast path：

| 方法（6单根states） | candidate checks | unique rollouts | 完整wall秒 |
|---|---:|---:|---:|
| beam_b3 | 375 | 6 | 0.827573 |
| beam_b5 | 375 | 6 | 0.829510 |
| beam_b8 | 375 | 6 | 0.805028 |
| beam_adaptive | 375 | 6 | 0.806492 |
| full_reference | 375 | 6 | 0.804301 |
| learned_pruning | 375 | 6 | 0.806421 |
| learned_beam_width | 375 | 6 | 0.881607 |
| hybrid | 375 | 6 | 0.807036 |

## 成本与恢复的实际记录

| 方法（24非退化states） | generation秒 | exact rollout秒 | features+inference秒 |
|---|---:|---:|---:|
| beam_adaptive | 28.421806 | 59.390151 | 0.000000 |
| learned_pruning | 25.423935 | 67.328716 | 0.127215 |
| learned_beam_width | 28.216571 | 54.901194 | 0.076946 |
| hybrid | 25.478058 | 56.917733 | 0.066393 |

Pruning generation低于adaptive，但exact成本及unique294对243增加，抵消生成阶段节省；width/hybrid的wall减少仍不足10%。三学习features+inference各低于.13 s，不能称网络开销是主要原因。Pruning/hybrid partial candidate checks为144,107，adaptive149,495，只减少约3.6%；width同为149,495。展开节点较少不表示全部检查或rollout都减少。上述只解释已记录成本，不推断其他模型或场景的普遍机制。

Panel adaptive及三学习非退化恢复到full128比例均62.5%。全30状态fallback count为adaptive51、学习各17，但前者是传统stage transitions，后者是learning fallback events，单位不同，不能用51/17声称节省。EP nondeg恢复率分别为adaptive73.203%（1127决策）、pruning72.597%（1113）、width73.381%（1127）、hybrid72.523%（1110）；各方法非退化分母不同，不能直接作因果对照。全部320 fallback分组保存在紧凑JSON。

动态width panel实际为141个width3层加75个恢复128层；EP width模型层数3/5/8为3890/29/13，hybrid为3890/17/8，恢复128另列。只报告实际分布，不根据test更改映射。Step7.1 positive约1.46%、val recall43.9%的标签/分类记录维持其有限范围，不追加模型普适无必要假设。

## C、泛化与结论边界

240行的chosen全部在full有限pool内，没有new selected value超过finite full best。保留闭合最佳的state数为full30/30，B3/B5/B8为25/26/27，adaptive及三学习均27/30；本次未观察C，不扩大长期Q/value路线。有限fullpool仍不是全域oracle。

结论限定单GradientBoostingClassifier seed7401、固定catalogue8holders/cap8/scan48、partial widths3/5/8/128、109维、test seeds7301–7303的同profile family及6个初态占用变体，目标是已到达机群FCFS closed有限参考。泛化只覆盖冻结负荷/SOC组与initial_sparse/dense，不泛称所有学习无效，不证明未来到达或长期价值。完整分组保留于CSV/JSON，模型hash和实验协议未随结果改动。

严格summary为completed、allow_incomplete=false，无missing formal paths，144context SHA及一次fit冻结核验通过。完整20,348,580-byte summary及全部node/leaf raw留在结果目录，未写入Git；其SHA256为`ec2bfd0a09c6abad466c6580900f1733f2480ffe263414020874750576f6fdae`。紧凑结果只投影可核验分组，没有改变科学计数或精度。首次SOC source错误、33成功/3失败及936files归档、pilot流程全部保留。研究验收已完成，最终版本提交与打包待后续实际交付。
