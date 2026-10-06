# P3专家查询：多项式边界优化候选与验证限制

## 当前结论

用户2026-10-05继续授权后，primary实现了保持既有搜索与数值判据的优化候选，完成本地64项回归和数值基准。**候选尚未应用到生产核心，也没有完成原服务器上的专家查询等价/耗时验证。** 本地原版快照查询已经出现与服务器记录不同的轨迹hash；候选查询随即被阻止。服务器SSH连续三次在建连时关闭，因此本轮保持生产源码冻结，将可审阅的候选保存为[待应用补丁](p3_numeric_bounds_candidate_20261005.patch)。

完整I0仍不成立，P3训练退化尚未验证解决。原V2 mixed的INCOMPLETE、300/482标签和182条缺失保持不变，不补标、不恢复采集、不将失败改成成功。`I0_QUALIFIED=false`、`PPO_ALLOWED=false`。

## 依据与实施范围

已有证据（preexisting_artifact）见[V2结果及成本剖析](p3_v3_i0_v2_budget_20261004.md)：mixed已完成查询耗时6937.736s，切环WAIT穷举占主要成本；固定slot19的profile中轨迹生成约占90.56%，边界计算包含大量小数组/标量调用。原跨对象bounds缓存试验收益很小，已撤回，不重复该方案。

primary本次亲自编写候选、测试和诊断脚本；worker仅执行指定读取、测试、SSH与Git命令。候选限定为`terminal_marl/core/polynomial_trajectory.py`以下实现细节：

1. 对已有NumPy数组直接裁剪，避免先逐元素转为Python list。最后保留的系数已超过原容差时，省去一次全数组非零检查；零/非有限值规则保留。
2. 将单个浮点数的`np.clip`改为相同端点的标量裁剪，保留负零、端点容差与按“上一个保留点”去重的规则。
3. 内部多项式点积直接调用原`polymul`使用的`np.convolve`。保留输入尾部精确零裁剪、卷积顺序、轴累加顺序与最终相对容差裁剪，预分配结果以减少通用序列转换和反复pad。
4. 将逐segment求一、二、三阶导数改为同一批数组操作，仍按原顺序逐阶相乘。

候选不改系数求解、求根方法、根筛选阈值、bounds项目、轨迹搜索集合、25个意图的顺序、first-feasible选择、容量/安全margin、mask或teacher语义。不用采样近似代替解析边界，不减少困难WAIT候选。

本地生产文件SHA仍为`832a88f432f083010cbc34a9dbc2c14b8b9fe6cbe016292bc43c0e12695d0a9f`；独立候选`.tmp/p3_polynomial_candidate_20261005.py`的SHA为`603a2997df35bac03771bc9d69d99f1fed48c899cce0ff1e93acf75908fea1e6`；补丁SHA为`240a14e8f89056f834e7cc2ae9ece44317596dc933f77cf8743ee5d5b4cf45e5`。补丁使用零行上下文格式，检查命令为`git apply --check --unidiff-zero docs/evidence/v4_2/p3_numeric_bounds_candidate_20261005.patch`，当前不执行实际apply。原版副本为`.tmp/p3_polynomial_baseline_20261005.py`。这些文件由primary在本地准备，未手动上传源码，也未进行server Git。

## 本地验证（newly_run）

新增`tests/test_polynomial_trajectory_numeric_equivalence.py`覆盖原公式精确点积、消去/尾零、相对容差上下边界、生成器输入与无别名、负零/端点去重、近零段、batch bounds和根接收端点。它与7个既有轨迹测试文件一起，通过进程内加载器测试独立候选，生产文件保持原版。

| 检查 | 实际结果与边界 |
|---|---|
| 首次bundled Python测试 | 缺pytest，导入失败，0项执行；未安装依赖 |
| 本地已有venv首次候选套件 | 57 passed/7 failed；加载器先import顶层包，包提前缓存原版函数，再换模块后混用了类身份 |
| 原版7个失败项对照 | 7 passed/4.79s |
| 修正加载器后同一套件 | **64 passed/7.76s**；先装载候选再import包，并断言parameterizer实际引用候选函数 |
| 纯bounds基准 | 512个合成段、4轮交替顺序、每批64段，所有原/新bounds字典精确相同；无reset/teacher query |

正确套件包含`test_polynomial_trajectory_numeric_equivalence.py`、`test_polynomial_trajectory.py`、`test_polynomial_trajectory_batch.py`、`test_trajectory_generator_v3.py`、`test_trajectory_time_parameterization_v3.py`、`test_trajectory_plan_v3.py`、`test_trajectory_execution_v3.py`、`test_trajectory_safety_v3.py`。17个新增测试和47个既有测试共64项，不与此前失败或历史测试累计。加载器错误及日志均保留，不能把第一次失败藏掉或说成核心修复后的物理失败。

原版bounds四轮为`.149324/.128168/.133994/.129015s`，候选为`.068840/.071890/.073964/.072038s`；中位数比为**1.82736倍**。这是纯数值组件的合成基准，不能外推为完整查询或采集的加速。数据在`.tmp/p3_numeric_bounds_benchmark_20261005.json`，SHA `f1ac1b73250cb95effa1a957e730d343ba89b51113354f56b340c5ca610d1139`。通过日志`.tmp/p3_numeric_candidate_tests_fixed_harness_20261005.log`，SHA `316f4a2955f2151651e02fea7ef091558000385a77162cd5fdbfb70a5785ff2a`。

本地venv为Python3.12.14、NumPy2.3.5、Torch2.14.1+cpu，Gymnasium可用；测试未使用服务器环境。没有把本地版本信息冒充服务器版本。

## 固定快照smoke的失败与停止判断（newly_run）

原计划沿用预登记4个既有状态：mixed slot0/19/299和high slot177；原/候选各一次，最多8query/200RELEASE意图，两个状态各一原/候选tape step，共4step，query子进程120s、总运行20min封顶。首次简单状态先验输出检查与后续检查分开，smoke成功才复用其结果继续，不重复查询。新脚本`.tmp/validate_p3_numeric_saved_states.py`只在私有进程加载原版/候选，核对既有selected、快照和tape哈希，原始产物只读；不reset正式seed、不训练。

服务器SSH三次均返回`Connection closed by 192.168.30.12 port 22`、exit1，连`pwd`都未执行。使用了已有SSH配置和必要的授权权限，不是把别名展开失败误判成服务器程序故障。当前无法核对实际服务器进程或新源码同步；没有启动、停止或重新运行任何服务器任务。

因此尝试用本地现有venv执行固定slot0的小样本。**仅原版完成1个query、1个RELEASE意图、1次candidate plan，0reset/0step；候选与其余状态均未执行。** 输出`.tmp/p3_numeric_saved_states_20261005_smoke`保留FAILED/exit1，运行约3.938s。子query自身返回OK，但父流程的原始证书等价断言失败；根`cases=[]`表示未接受任何完整子结果，不能误解成没有消耗query。

primary逐字段比较原label与新query：动作、90D、完整teacher audit、上下文、plan次数等全部相同，唯一不同字段为`selected_certificate`中的轨迹hash及对应checked hash：

| 项目 | trajectory_hash与checked_hash（两者各自相同） |
|---|---|
| 服务器原产物 | `648386e1de3dc61f79a1622188aee48522680e2f860516153751f8372c188f8f` |
| 本地原版查询 | `21334b56bdbd6866ef15653e67224730087791055549b801becc64ebecccc33d` |

这是**跨环境原版重放不满足完全相同证书条件**，发生在候选运行之前，不能归因于本次数值优化。现有证据也不足以把差异唯一归因于操作系统/BLAS或某一浮点操作，未进一步作猜测。没有实际checked/executed不一致证据：本地证书自己的checked与trajectory hash仍相等；此查询也没有执行环境step。为维持原验证标准，停止本地后续查询，不放宽hash要求、不拿标签一致替代轨迹一致。

## 后续最小步骤与交付状态

1. 服务器连接恢复后，先核对真实进程、冻结源码与Syncthing；已有I0高/混合SOC臂不重启。当前生产核心不变，补丁暂不自动应用。
2. 在原服务器环境先完成原版slot0→候选slot0的输出/证书smoke，再复用它继续其余3个状态和预定4step，仍最多8query/200意图、120s/child、总20min。**这将是另记的服务器验证，必须把本轮已发生的1次本地原版失败查询单独计入历史总成本**，不能声称全历史仅8次或覆写本地失败。
3. 原版首先必须复现原产物；候选须完整audit、选中证书、tape/hash及安全一致，使用同机未剖析计时。任何不一致先诊断，不应用补丁、不减少候选或修改容差。通过后才把候选应用生产文件、再核对相应测试与源码；如有活动任务，待其结束后再变更。
4. 有真实查询收益后才决定fresh采集预算与新seed；不能从合成bounds基准推断mixed完整标注一定落在7200s内。监督pre-state与服务推进/顺序批准的语义限制仍须在学习前处理。无BC/PPO/full training、held-out、硬件、optimizer或checkpoint写入。

本轮交付为待应用补丁、17个数值等价测试和本记录/CURRENT_STATUS最新块；生产优化未启用。新测试在未改动的原版上也为17 passed/0.39s，证明提交后不会要求提前启用候选。本轮没有启动后台长任务或heartbeat。补丁可应用性/文档diff检查后显式本地提交，保留用户/并行dirty；当前仓库无remote，不能推送。

## 2026-10-06连接恢复检查（newly_run）

用户要求检查服务器是否恢复并继续。一次普通`ssh -o ConnectTimeout=15 lab-server 'pwd'`仍在执行远程命令前关闭；随后一次限定的verbose连接诊断记录了`Connection established`、本地SSH版本发送和`kex_exchange_identification: Connection closed by remote host`，进程exit255。没有收到可用远端SSH会话，不能读取实际进程、源码SHA或运行环境版本。该证据说明TCP22可连接、SSH标识交换阶段连接被对端关闭；不足以确定是sshd、连接限制或其他服务器/网络配置，也不等同于服务器整机不可达。

日志保留在`.tmp/p3_ssh_recovery_20261006.log`，未把包含客户端配置细节的完整verbose日志纳入Git。没有启动或重跑查询、采集、训练，也未应用候选或改动生产核心。原64tests与bounds基准仅沿用既有结果，本次没有新增性能证据。下一步仍是恢复SSH后核对真实进程/Syncthing，再执行上方既定原服务器smoke及有界对照；当前无法通过SSH执行服务器侧恢复操作。仅更新本文和CURRENT_STATUS最新块，作文档diff检查后本地提交；无remote不推送。
