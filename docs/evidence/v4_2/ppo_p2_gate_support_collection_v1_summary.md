# P2 natural-flow gate support collection V1 summary

> 日期：2026-09-18
>
> 终态：`SUPPORT_COLLECTION=FAIL`；不得训练、PPO或课程升级

## 1. 产物与完整性

formal按预注册协议运行15个fresh stochastic student-authoritative回合，无训练、无Teacher
动作替换，进程exit code为0，15/15 records与两份support文件均生成：

- manifest SHA256：`3f376b68209dbd7a68db058b8344204138dae3ea107cd2678d2f4966312fd979`；
- train support：4,342 rows，SHA256
  `643ec5f31a489f7530fb038f2386890f48e062a2007ce2d77580a91eb08b98a9`；
- dev support：3,274 rows，SHA256
  `f3a4b3eabd2c9c51c56ed7eaf872105f5b1e40c4df8875e77c1e36299112d1a6`；
- consumed reservation SHA256：
  `dba7d1d1e3ad875b6cc92991e3cf1045ad195e2aafcb8aba146f4c32917c911d`。

train/dev分别完成78/78与52/52请求；全部安全计数为0，ActionTape replay、requested logprob、
Teacher label-only、student request provenance与88D schema均通过。双类、三个interface、数量、
event唯一性和split独立性也全部达到冻结阈值。

| split | rows | Teacher WAIT/RELEASE | o2m/m2i/i2f |
|---|---:|---:|---:|
| train | 4,342 | 1,591 / 2,751 | 1,814 / 1,251 / 1,277 |
| dev | 3,274 | 1,366 / 1,908 | 1,291 / 942 / 1,041 |

## 2. 唯一失败 gate

train seed772009（flow12）在959个actor decisions中记录3次SafetyFilter shield，
`shield_actor_fraction=0.003128`；其余14回合shield为0。15回合总计7,616个actor decisions，
总shield fraction为`0.000394`。三次均属于同一m2i lifecycle，SafetyFilter把proposal修正后
执行；对应release intent在`predicted_separation`约束下经历
`safety_reject → safety_wait → safety_wait → approved`。该回合仍12/12完成，碰撞、NMAC、
hard separation、uncertified execution与SafetyInvariantError均为0，replay精确。

V1预注册要求每回合`shield_zero=true`，因此该单一失败不能事后忽略或升级：

```text
SUPPORT_COLLECTION=FAIL
SUPPORT_FROZEN=false
LEARNABILITY_AUDIT_ALLOWED=false
GATE_REPAIR_ALLOWED=false
PPO_ALLOWED=false
```

V1的7,616行只能作为失败诊断证据，不得进入训练或正式learnability audit。

## 3. Fresh V2边界

V1暴露的是数据收集gate与系统既有SafetyFilter边界不一致：历史I3 student-only qualification
已预注册并使用每回合`shield_actor_fraction_max=0.05`，而不是零干预。若继续收集，必须保留
V1 FAIL，使用全新train/dev seeds，并把V2的唯一语义变化预注册为继承该0.05 gate；不得使用
V1 rows补足V2，不得改变安全全0、replay、class/interface或数量阈值。
