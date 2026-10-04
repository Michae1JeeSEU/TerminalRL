# V3/90D专家数据接入验证（2026-10-04）

## 范围与已有证据

用户在两臂稀疏pilot结束后授权“继续下一步”。本轮执行上一轮明确记录的独立90D loader与定向检查：0个环境reset、0个teacher query、0个optimizer step、0个checkpoint写入。primary本人负责研究文档、判断、实现和测试编写；worker只执行精确SSH/测试/Git命令。继续采用research-pragmatist与academic-experiments的最小验证，不扩展仿真矩阵。

已有证据（preexisting_artifact）：[稀疏pilot](p3_v3_sparse_pilot_20261004.md)的high892402与mixed892403共24N、0E/F/D/U，6587raw、121 actor、13有效label；行为是固定rule K3，不是训练中的student。旧88D入口`run_v4_2_v2_1_i1_bc._arrays`固定`OBSERVATION_DIM=88`且直接读取行内teacher字段；本轮新数据是90D raw与独立label文件，不能直接使用旧入口。不能以这些开发状态训练后再当作独立资格测试。

## 实现与数据语义

新增`terminal_marl/training/p3_v3_expert_dataset.py`，保留既有collector、88D BC和物理/安全核心。仅依赖NumPy和已有动作编码，不创建学习对象。

1. 读取case protocol和episode config，核对预期partition、显式允许seed、90D observation schema、V4.2 action schema与V3 trajectory schema；逐行检查dimension/有限float32、数值UID、整数tick/seed、唯一decision_id和raw teacher_label=null。
2. annotation_result的逐项status为标注是否完成的依据，只读取OK所引用的label。TIMEOUT/ERROR/NOT_ATTEMPTED即使残留OK文件，也不能升级为监督；未选中的raw行没有目标，不填WAIT。
3. 原teacher标签按decision_id关联，检查label中的tick/UID/window、观测、pre-state、actor_mask及joint/gate/proposal/request一致。RELEASE必须带相同checked/content轨迹hash，WAIT必须无proposal/RELEASE证书。此检查复用已有证书，不重新查询teacher或证明联合批次可执行。
4. gate mask为`status OK && actor_mask`，proposal mask再要求teacher RELEASE。返回训练batch时只保留gate mask为真的行；WAIT的proposal在内存中记为-1。forced行即使带有效标签，也计入ignored_forced_labels而不参与actor监督。实际行为/拒绝/执行不用于改写teacher target。
5. 重复raw/annotation ID、孤立label、88D/错误schema、非有限值、错位观测或目标均报错，不静默修补。训练接口拒绝development、audit、Q及不在显式train种子集中的episode。

公开接口：

```python
from terminal_marl.training.p3_v3_expert_dataset import (
    inspect_sparse_episode, load_sparse_training_episode,
)

# 开发数据只返回统计；不给训练入口返回开发数组。
report = inspect_sparse_episode(episode, expected_seed=892402)

# 仅将来正式train数据可用；本轮只用合成文件测试此接口。
# batch = load_sparse_training_episode(train_episode, allowed_train_seeds=frozen_train_seeds)
```

原开发schema仍为`p3_v3_90d_sparse_expert_development_v1`；新增非开发字段契约为`p3_v3_90d_sparse_expert_v1`，不改变pilot文件、不把development改名成train。该契约没有定义I0准入或正式采样密度。后续正式collector采用它时，仍须先冻结正式场景/选样/预算/seed协议。

## 验证方式

本地Python3.12.14有NumPy但缺pytest/torch；不安装依赖，在服务器既有环境Python3.12.3、NumPy2.5.1、pytest9.1.1测试。新代码通过Syncthing核对两端SHA后执行，不上传源码、不server Git。

定向测试`tests/test_p3_v3_expert_dataset.py`只构造合成JSON，seed101/102只是文件字段、不reset，不消费预留研究种子。测试涉及mask/未知标签、分区、seed/schema、全部26种动作的joint→proposal偏移、错误join/重复ID/数值/索引、只读CLI JSON输出。真实pilot仅运行只读counts入口，与上一轮primary逐行统计比较，不调用环境、teacher或训练接口导出数据。

### 本轮结果（newly_run）

服务器执行：

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/user/envs/terminal_env/bin/python -B -m pytest \
  tests/test_p3_v3_expert_dataset.py -q -p no:cacheprovider
```

33 passed in 0.22s、exit0，日志`/home/user/training_outputs/terminal/p3_v3_loader_20261004_tests.log`。loader两端SHA为`109df36b42576d87a2137c96b71febfd6e11876191d1be60ec44eaf3b7122d6f`；test两端SHA为`854f39cf11771076eaa4ea2d2c240f29f48372007dfee4c7bd022a3c83bbe98c`。本轮没有重跑涉及环境reset的旧collector tests，其原17tests是前轮证据。

真实文件检查的工作目录为`/home/user/projects/terminal`，同一Python与环境变量，运行模块`terminal_marl.training.p3_v3_expert_dataset`，两次`--episode`分别为`/home/user/training_outputs/terminal/p3_v3_sparse_pilot_20261004/high/seed_892402`、`.../mixed/seed_892403`，对应`--seed 892402/892403`，默认development分区。两次CLI均exit0，stderr为空；primary亲自读回同步JSON并与上一轮逐行分析比较：

| 原开发episode | raw | actor eligible | forced | 有效gate标签 | RELEASE proposal标签 | 未标注 |
|---|---:|---:|---:|---:|---:|---:|
| high892402 | 3204 | 64 | 3140 | 6（2 RELEASE/4 WAIT） | 2，均proposal12 | 3198 |
| mixed892403 | 3383 | 57 | 3326 | 7（5 RELEASE/2 WAIT） | 5，proposal12/13/20为3/1/1 | 3376 |

全部计数与旧分析一致。6574条未标注行没有变成WAIT目标；6466条forced行不计入actor监督。真实pilot统计仍只是13条可解释标签，检查没有导出训练数组或消费正式数据。

输出目录`/home/user/training_outputs/terminal/p3_v3_loader_validation_20261004`中的`high.json`、`mixed.json`保存CLI统计，`train_partition_guards.json`保存训练入口拒绝结果。把两个真实seed显式放进allowed_train_seeds也不能绕过development分区，两臂均`rejected=true`、`protocol partition mismatch`。guard首次命令把新输出目录误作为输入目录，产生FileNotFoundError，未写guard文件；之后使用原source episode路径检查成功。这是命令输入路径错误，没有改数据、运行仿真或新增采样，保留此失败记录，不计为有效guard检查。

primary接受数据接口验证：90D原始观测、独立专家标签、mask与分区能正确进入指定接口。该结论只针对数据读入，不建立模型的学习能力或泛化证据。CLI检查是对preexisting_artifact的新读取，不把旧24N结果重新标成新仿真。

## 后续边界

此步只解决数据接入正确性，不能证明监督规模足够或P3训练退化已经解决。正式I0尚未采集，reserved train893001–893024/audit894001–894006/Q895001–895012保持未消费。下一项是根据既有121个eligible行的时序分布冻结更充分、与标签无关的正式选样及query预算，再用协议中第一个固定三场景cycle检查数据与成本；不直接沿用pilot的75tick一条密度，不按正式标签挑seed或补成功样本，不自动进入BC/PPO。

本轮不启动长任务，无需新heartbeat。I0/student-only仍未qualified，PPO_ALLOWED=false；没有full training、held-out或硬件操作。

修改文件限定为新loader、新tests、本报告、technical_spec新增接口段及CURRENT_STATUS最新块。代码与文档由primary完成；检查后本地显式提交，保留其他dirty。仓库未配置remote，不推送。
