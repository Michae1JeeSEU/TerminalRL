# P3数值优化：恢复连接后的原服务器验证

## 范围与运行前条件

用户2026-10-07要求“查看服务器是否恢复，然后继续”。本轮继续[已记录的数值优化方案](p3_numeric_bounds_optimization_20261005.md)，primary本人负责证据判断、核心与文档；worker仅执行指定SSH/测试/Git命令。

2026-10-07 15:47（Asia/Shanghai，服务器UTC07:47）只读检查成功：SSH可登录，未找到本轮既有collector/profile/validator进程。服务器生产polynomial与独立baseline的SHA均为`832a88f432f083010cbc34a9dbc2c14b8b9fe6cbe016292bc43c0e12695d0a9f`，独立候选为`603a2997df35bac03771bc9d69d99f1fed48c899cce0ff1e93acf75908fea1e6`，验证脚本为`d507cbee0c7d76859f5c65d4aee46f273a010a2dbac472fe19c32f8f1dec0d9b`，与本地一致。运行环境Python3.12.3、NumPy2.5.1、Torch2.13.0+cu130；此前本地版本不同，此事实尚不足以把旧本地hash差异唯一归因到某一依赖。

生产代码暂保持原版，只在诊断子进程中装载冻结baseline/candidate。新输出根为`/home/user/training_outputs/terminal/p3_numeric_bounds_validation_20261007`，不覆盖2026-10-05本地失败或任何I0数据。

## 固定比较计划

既有mixed slot0/19/299、high slot177，原/候选各一次；共最多8query/200个RELEASE意图。mixed299与high177各用原/候选执行原tape一step，共4step，0reset。每child120s，smoke与full实际耗时合计20min；所有输出为diagnostic_never_train。2026-10-05本地失败已消耗的1次原版query单独保留，不混成此次8次预算内成功。

先以`--phase smoke`在新`smoke`目录执行slot0原/候选，primary核对结果/原证书等价与序列化后，才能在新`full`目录用`--phase full --smoke-output .../smoke`继续其余三个状态。复用smoke，禁止重复查询。使用原服务器venv、显式cwd `/home/user/projects/terminal`，`PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`；不与其他本轮耗时任务同时运行，以免干扰计时。

原版先须复现原label/audit/选中证书；候选须与原版及原产物相同。tape检查覆盖原post-state hash、action trace、release telemetry和实际安全检查，原/候选完整step输出也比较。任何失败保留并停止，不能放宽hash/容差、补样或提前启用优化。全部通过且有实际查询收益后才应用候选并运行相关回归；本轮不恢复失败I0、不启动学习/held-out/硬件、不覆盖checkpoint。I0_QUALIFIED=false、PPO_ALLOWED=false。

## 实际结果

### 首个状态smoke（newly_run）

原服务器固定mixed slot0的原版/候选均QUERY_OK，根SMOKE_OK/exit0，0reset/0step。primary亲自读取同步result与query：原动作、完整audit、90D、上下文和选中证书相同，原版与候选均恢复原服务器证书`648386e1de3dc61f79a1622188aee48522680e2f860516153751f8372c188f8f`。该结果表明可在原服务器继续既定比较，不将之前本地不同hash的失败改成成功或据此唯一归因某一依赖。

原版query 0.9925285904s、候选0.6707358425s；smoke合计3.2902641790s。两查询各1个RELEASE意图/1plan，原输入文件不变。local/server根result SHA一致为`7f0a4d071f70ecf5488dc9d5c7bb939f374c3db2cd08763be345ccface301f83`。primary接受输出闭环，继续新`full`目录的其余三个既有状态；不重跑smoke，不在计时期间并行测试或修改核心。

### 完整固定状态结果与应用决定（newly_run）

full父PID2757361自然完成COMPARISON_OK/exit0，UTC07:56检查无项目Python遗留进程。smoke复用后的总运行281.8852s，低于20min。primary亲自核对全部query、各child exit和4个step输出，并与原label逐字段比较；结果见`.tmp/p3_numeric_server_analysis_20261007.json`。full result的local/server SHA相同：`32faf7b9a9f56dcf72b5da93fb7b354348395f7f188b58f257db6c9e46bc4900`。protocol中的RUNNING是创建时静态字段，终态以result为准，未将其误当继续运行。

| 固定状态 | 原teacher | 原版query(s) | 候选query(s) | 耗时减少 |
|---|---|---:|---:|---:|
| mixed0 | RELEASE proposal12 | 0.9925 | 0.6707 | 32.42% |
| mixed19 | WAIT | 33.6063 | 23.5343 | 29.97% |
| mixed299 | RELEASE proposal22 | 28.5222 | 19.9323 | 30.12% |
| high177 | WAIT | 83.3446 | 56.7119 | 31.95% |
| 四状态合计 | 两R/两W | **146.4656** | **100.8492** | **31.14%** |

总共8query、136个RELEASE意图/136plan、4step、0reset；完整audit、选中证书、原tape post-state hash/action trace/release telemetry与安全检查均通过，原/候选4个step输出相同，输入文件不变。不能把这几个固定开发状态当成总体置信区间、完整采集成本保证或新student资格；之前本地原版失败1query另计。该比较没有删掉穷举WAIT候选，也未以减少可行域换速度。

primary接受此候选为有实际收益的等价数值实现，在确认进程退出后亲自在本地将冻结候选应用到生产`polynomial_trajectory.py`，文件SHA恰为603a299所对应完整值。候选副本/补丁、原版副本和全部失败证据保留。接下来在Syncthing核对后对实际生产导入路径运行相关轨迹/集成回归，不再注入测试模块。完成后另冻一个fresh密集混合SOC采集协议，验证完整标注能否在原7200s内结束；不恢复旧mixed、不把局部31%收益直接当完整I0通过。

### 生产路径回归与本轮结论（newly_run）

服务器已同步603a299生产文件。原8文件的数值/轨迹回归加`test_p3_integrated_repair.py`，实际生产导入路径为**90 passed in 29.85s/exit0**；日志`.../p3_numeric_bounds_validation_20261007/applied_tests.log`，SHA `49c7530a54426e07ac7729f7785971d68e4e71e08c4b8d751bf44f86d406e1ea`，相邻.exit为0。前后无相关活动任务。测试是定向实现回归，不与历史64项累计，也不计入8query的固定状态实验。

数值优化现已应用并通过所需回归，原服务器访问阻塞解除。完整查询在4个固定状态中加速约1.45倍，安全/轨迹及teacher输出保持一致；**完整混合SOC采集能否在7200s内标完仍待fresh回合验证，P3学习退化仍未验证解决**。下一步只启动另冻协议后的一个fresh mixed case，保留全部actor和原有停止条件；若再次超时则保留INCOMPLETE并据实际成本决定资源/采样方案，不继续盲目铺开种子。仍无训练、held-out或硬件。本轮核心/报告/CURRENT_STATUS显式本地提交，保留其他dirty；无remote不推送。
