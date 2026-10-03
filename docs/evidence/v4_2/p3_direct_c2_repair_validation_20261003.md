# P3直接连续离环修复与重跑（2026-10-03）

## 决策与适用边界

用户要求比较“直接进入降落轨迹、途中平滑调速”和“先向内收缩再减速”，由primary选择并实现后重跑。primary读取研究文档、分析、修改核心代码和本记录；worker只执行明确的测试、SSH、监控和Git命令。

选择 **直接C²连续进近至停机坪上方，再接原垂直下降段**。离环前减速不是本项目动力学的必要条件；必要的是位置、速度、加速度连续，整条轨迹满足物理性能与交通间隔。旧Bezier构造因曲率限制先降低入口速度，导致占用内环22–29s减速，后机维持16m/s而追近。这个先决条件属于原轨迹构造，不应被当成所有可执行降落轨迹的物理要求。

“不必先向内收缩”指不再强制飞完一个独立200→100m收缩段，仍需沿进近曲线从环飞向中心停机坪。此前prototype在60s内保持角速度0.08rad/s、半径200→100m，所以切向速度实际已由16→8m/s；“先收缩再减速”只是简写，不能理解为收缩过程中完全不降速。直接曲线同时决定空间运动和速度变化，也不保证任意时刻速度单调下降。

选择根据当前固定状态与研究接口，未声称这是所有流量/几何下的全局最优轨迹；原SafetyFilter及停机坪可用性继续决定是否接纳。若直接曲线仍存在冲突，不切换未记录的隐藏规划器，不放松间隔，也不增加电池来掩盖失败。

## 比较证据（newly_run，同一个已记录状态）

原快照：`p3_integrated_validation_20261003/A_blocker_replay/snapshot_600.pkl`，可信本项目pickle，7个未提交内环holder，速度16m/s；无环境推进、训练、权重更新。原A完整周期及根因报告见 [退出阻塞](p3_inner_exit_blocker_findings_20261003.md)。

| 构造 | 检查范围 | 全世界安全通过 | 时长与说明 |
|---|---|---:|---|
| 原环上制动+Bezier | 7机×6pad×3speed=126 | 0/126 | 全部单机动力学和能量通过，但真实邻机冲突 |
| 固定半径收缩prototype | uid6、pad1一个候选 | 1/1 | 105.823s，最小间隔136.966m；不能按1/1与其他全量比率直接排名 |
| 直接quintic至地面探索 | uid6×6pad | 3/6 | 约38.5–49.7s；未选用，最终方案保留已有垂直降落段 |
| 直接quintic至pad上方+垂直段 | 7机×6pad、目标速度参考10m/s=42 | 11/42 | 全部动力学通过，完整路线52.5166–70.4846s |

最后一行分机通过数：uid0=0、uid6=2、uid7=5、uid8=4、uid9/10/11=0（每机6个pad）。证明该状态有可用退出路线，尚不证明所有飞机能依次完成。原K3对照为126候选，新几何探索仅42候选，不把这两者当成同一候选分布上的成功率增益。

比较输出保存在服务器旧输出根的`direct_quintic_compare.json`和`direct_staging_compare.json`；运行脚本分别为本地`.tmp/p3_direct_quintic_compare_20261003.py`、`.tmp/p3_direct_staging_compare_20261003.py`，全部保留。本轮重跑用下述正式生产实现，不以临时prototype代替执行。

## 实现

1. `SimConfig.v3_landing_path_mode`新增显式`direct_c2`选择；历史默认仍为`brake_then_curve`。仅V3、continuous landing geometry允许直接模式。统一开发profile升为`p3_integrated_repair_direct_c2_v2_20261003`并启用新模式；旧88D/V2冻结source加载时显式还原旧模式，然后按既有规则迁移为90D，不重新训练或自动qualified。
2. 正常与应急降落共用新入口。以实际p/v/a作为五次多项式初边界，pad上方的staging位置、零速度/加速度作为末边界，再连接原垂直参数化下降段。内环保持当前高度；外/中环应急沿用原高空staging高度语义。整个轨迹保持C²连续，并经原解析多项式动力学与Safety检查后才能执行。
3. 有预约延迟时，先按当前名义环速飞行至实际离环时刻，不为入口曲率额外环上制动。正常speed proposal仍由`10+residual`映射到轨迹时长参考；入口真实16m/s不会瞬间重置为目标8/10/12m/s。时长以距离、当前速度与水平/垂直物理限制计算下界参考，再按目标速度缩放；最多20次、每次×1.1寻找物理可行时长。只检查单机动力学，不根据交通安全结果暗中搜索路线。
4. metadata记录mode、requested/actual entry duration和是否物理限幅，保留requested/accepted departure、真实entry speed、required braking=0等可审查字段。Safety、资源预约、能量和checked/executed hash逻辑不变。
5. 修复独立数值错误：`ramp_end`与总时长在1e-9s内相等时，在两种浮点偏差方向上统一为同一末端节点，内部ease/ramp节点也排除端点邻域。消除原约1e-13s段；不放宽动力学边界误差阈值。复现原6个transfer候选，并用正/负时间舍入误差回归。

## 有界验证顺序（运行前确定）

新输出根：`/home/user/training_outputs/terminal/p3_direct_c2_validation_20261003`。保留原A与所有失败，不覆盖输出、checkpoint。测试/评估前核对Syncthing相关文件SHA；只本地改代码，运行时核心冻结。最多两个本轮评估进程。

先运行direct landing、integrated repair、V3 generator/execution/safety/dynamics相关测试；再对规则/source各做2tick smoke，检查输出、error和迁移参数。通过后执行原A完整配对：

```text
python = /home/user/envs/terminal_env/bin/python
module = terminal_marl.scripts.diagnose_p3_integrated_repair
args = --run --requests 12 --warm-start 6 --high-soc
       --seed 890501 --policy-seed 890601 --max-steps 1800
       --trace-blockers --snapshot-ticks 400 600 850
controllers = rule / source
output = <新输出根>/A_rule / <新输出根>/A_source
checkpoint = /home/user/training_outputs/terminal/p3_normal_advantage_gate_repair_20260928/fit_smoke/terminal_gate_repair.pt
checkpoint SHA256 = 3fe7dd1e01d032227521f6fc5c8464fe8d00d98fb0e9f0dbd13411123a27d4ab
```

检查完整N/E/F/D/U、error/censor、耗尽和应急原因、实际安全计数、checked/executed一致性及能量消耗。A有技术/安全/F/U阻塞即不升负载、不训练；primary基于trace与可信快照选择最小诊断。A通过才继续已预定B（24请求/warm12、高SOC、seed890502/890602）及C（同B仅SOC.55–.90）。这一轮无训练对照首先验证环境可达性和冻结source；即使A/B/C改善，也不单独证明PPO训练退化已解决。任何学习前仍需student-only资格判断与有界更新方案，`PPO_ALLOWED=false`；无full training、held-out或硬件。

## 执行记录

实施完成；Syncthing四个改动源码/测试SHA一致，六个定向测试文件 **66 passed / 48.86s / exit0**，日志`targeted_tests.log`。正在执行规则/source两步smoke。完整周期尚未启动、尚无新A结果。执行结果、提交、进程及监控信息在到达后追加。


两臂smoke完成：各2ticks、预定CENSORED、error=null、exit0；各2行blockers与snapshot_2存在，config确认direct_c2。rule/source checked samples=60/25、hash mismatch=0/0，碰撞/NMAC/硬间隔/joint fail-closed均0；原6个transfer数值失配未再出现。source迁移记录checkpoint_weights_unchanged=true、source_policy_weights_unchanged=true、optimizer_created=false、NOT_QUALIFIED，固定checkpoint SHA与上述一致。两步不能证明完整周期，primary据此仅放行预定A完整对照。


### A完整周期结果（newly_run）

核心提交`a7096fb66efc50401db3db267fa1f22ce55f6137`；无Git remote，未推送。两个实际Python PID为rule2445800/source2445799，argv与协议一致，初始progress与trace确认后创建heartbeat `p3`。本轮运行速度显著快于原失败A（新约2–3分钟墙钟），监控设为5分钟，并在当前任务继续后续阶段。

| 臂 | N/E/F/D/U | ticks / 墙钟s | min swept m | checked samples / hash mismatch | 最低终态SOC |
|---|---|---:|---:|---:|---:|
| A_rule | 12/0/0/0/0 | 660 / 128.290 | 81.4774 | 15506 / 0 | .662702 |
| A_source | 12/0/0/0/0 | 611 / 169.842 | 82.3455 | 13475 / 0 | .691173 |

两臂natural terminal、error=null、censored=false，generated=12，active/queue/service=0；碰撞、NMAC、硬间隔、operational buffer、joint fail-closed、trajectory_infeasible均0。应急attempts/committed与energy negative budgets均0，外部队列累计各20 UAV-s。规则/source全过程已检查plan matches各38；最大同时final=2/3，无overcapacity。正常release仍有36/49次拒绝，不能将需要等待的安全拒绝本身认定为bug。

原同场景5N7F / 4N8F的持续内环耗尽，在新A完整配对中消失；支持本次组合修复对已定位环境阻塞有效。这里同时修复直接进近和transfer近零段，完整周期改善不能被分摊成两项各自的独立因果效应；同状态几何探针与数值复现提供各自的机制证据。此结果仍不是PPO更新前后比较或普遍liveness证明。

primary据完整业务/安全/能量结果批准进入预定B：保持生产commit、checkpoint、max1800、trace和400/600/850快照，改为24请求/warm12、高SOC、env890502/policy890602，独立输出`B_rule`/`B_source`。新脚本已在两步smoke与A验证输出，不重复构造新验证矩阵；运行前仍核对实际进程及Syncthing，最多两臂。B结果返回后才决定C。


A终态核对：两exit=0、两个原PID均退出且无重复Python诊断进程；相邻log各108bytes，仅终态摘要。result SHA本地/服务器一致：rule=`9b5a390bfe971e84cb3d97f4071175dc493f88dd845a83e678e4f8d56054c289`，source=`f72e18c20c504d2baaeb76e59a833c57e75c8ab8853bdb81a9738db2d3802ac0`。400/600快照存在，850因提前自然终态未生成，符合输出条件。progress保留running标签，终态以result+exit+实际进程为准。

规则/source总能耗=3236.017/3056.491（模型能量单位），每请求均值=269.668/254.708；只作当前两臂描述，不把单个seed差异称为学习优势。


B已实际启动，Python PID rule2446805/source2446806，primary已核对完整argv与预定B一致。约运行3分52秒时，两臂progress分别150/175ticks、正常完成2/3、活跃18/18、队列0/1，尚未终态；不能以中间状态作通过结论。heartbeat `p3`已经切换到B及其后续条件C，每5分钟，正常推进静默；预计B比A墙钟更长，维持当前间隔。A证据文档提交`964e6a4`，无remote、未推送。运行期间核心保持`a7096fb`版本。
