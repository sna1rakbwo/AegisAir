# 架构决策：LLM = Mission-level Constraint Recovery and Replanning（2026-08-15）

## 决策

LLM 的职责不是「实时避碰」，也不是窄化为「解决 repeated CBF conflict」，而是：

> **当原本的任务计划因为安全约束、环境变化或资源变化变得不再合适时，LLM
> 负责重新解释任务并生成新的高层任务方案。**

一句话核心：

```text
CBF preserves safety; the LLM preserves mission intent under changing constraints.
```

职责边界：

```text
Runtime Assurance / CBF
    instantaneous feasibility and safety（同步）

Semantic Mission Manager（LLM）
    longer-horizon mission consistency（异步）
```

LLM 不抢方向盘，LLM 改路线图。

## 触发：Mission Validity Monitor

LLM 不依赖「CBF 是否反复推」。触发来源分三类：

```text
E_replan = E_safety OR E_mission OR E_coord
```

- `E_safety`：CBF 已保证安全，但原任务被安全干预改变得不再合理。
  例如 route deviation（偏离原航线/走廊过大）。
- `E_mission`：任务优先级、目标、禁飞区、临时障碍、通信状态等发生变化。
- `E_coord`：继续按原计划走不划算，例如等待过久、绕路太长、某 UAV
  任务负载过高。

在轻量 2D 环境里的可计算 proxy：

```text
E_safety:  cross-track deviation from [start, goal] segment > d_th
E_coord:   no goal progress for stall_window_s
E_mission: 外部注入的 mission/priority/goal 变化
```

## LLM 输出：只允许高层动作

> 注（2026-08-16 更新）：针对「安全约束造成的对称 coordination deadlock」，
> 高层动作从「只改几何（REROUTE）」扩展到「改 coordination responsibility」。
> 见文末「Coordination Deadlock Resolution」。

```json
{"action": "REASSIGN", "agent": "uav_3", "task": "inspection_B"}
{"action": "REROUTE", "agent": "uav_2", "via": ["corridor_C"]}
{"action": "CHANGE_PRIORITY", "high_priority": "uav_1", "yield": "uav_4"}
```

禁止输出 velocity / acceleration / turn 等低层动作。输出仍经过
schema + action whitelist + validator，并由 Runtime Assurance 保留最终否决权。

## 异步语义

1. LLM 后台生成，不阻塞 CBF / MARL 控制回路。
2. candidate 就绪后 validate + commit。
3. 延迟预算不再是「0.4-0.6s 实时避碰」，而是允许数秒的异步 replanning。

## 指标

评价 LLM 的是「能否恢复 mission-level consistency」，不是「能否实时反应」：

- repeated / persistent CBF intervention 是否下降；
- route deviation 是否下降；
- mission completion time 是否改善；
- path efficiency 是否改善；
- recurrent conflict / invalidation rate 是否下降。

## 论文实验设计（优先三类 LLM 场景）

1. 任务优先级冲突；
2. 航路 / 区域突然不可用；
3. 某 UAV 失效后任务重分配。

这三类比「被 CBF 推来推去」更能证明语义级重规划的必要性。

## Coordination Deadlock Resolution（2026-08-16 更新）

4 机对称交叉暴露了一个新问题：barrier 会给出最保守的「大家都不动」，此时
`REROUTE`/`YIELD` 只是改变 nominal geometry，最终仍会被 sampled-data barrier
投影掉，所以 LLM「想去哪」不等于「谁先走」。

因此 LLM 的死锁恢复职责从「位置修改」升级为「通行权 / coordination
structure」：

```json
{
  "mode": "SEQUENTIAL_PASS",
  "priority_order": ["uav_1", "uav_3", "uav_2", "uav_4"],
  "reason": "uav_1 carries an urgent medical mission"
}
```

deterministic executor 只取结构化 `priority_order`，翻译成：

```text
right-of-way UAV -> GO
others           -> HOLD
等 right-of-way 离开冲突区 -> 释放下一个
```

barrier 全程在线，不放松任何安全约束。LLM 不直接输出连续控制量，也不决定
「怎么走」，只决定「谁先走」。

分层职责：

```text
Barrier:  谁都不能撞（safety）
LLM:      谁更应该先走（mission semantics）
Executor: 按顺序 GO/HOLD 执行（deterministic）
```

这为 LLM 提供一个比「固定 ID priority」更有说服力的实验场景：紧急任务位置
随机变化时，只有 mission-aware / LLM priority 能优化紧急任务完成时间，而
固定 ID priority 做不到。
