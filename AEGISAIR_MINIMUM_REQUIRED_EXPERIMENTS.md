# AegisAir 最小必做实验清单

**版本：** Minimum Publishable Version  
**日期：** 2026-08-20  
**目标：** 先完成一套最小但完整的论文证据，后续再增加规模、消融和统计强度。  
**证据范围：** 轻量仿真 + PX4 SITL/Gazebo；不包含真实硬件。

---

## 1. 最小论文主线

只验证三件事：

1. 多源风险包络能否改善安全边界判断；
2. exact-ZOH sampled-data RA能否在PX4执行滞后下过滤危险动作；
3. fail-closed任务恢复能否在不绕过硬安全的情况下恢复任务。

---

## 2. 必须冻结的内容

正式实验前冻结：

- Git commit和配置hash；
- MAPPO checkpoint；
- LLM模型和推理参数；
- 场景参数；
- collision、violation、completion定义；
- seed manifest；
- primary metrics；
- 统计方法；
- fallback和QP infeasible定义。

正式test开始后不修改阈值、不删异常trial、不追加seed救结果。

---

## 3. 最小方法矩阵

### 3.1 Safety baseline

至少运行以下四组：

| 编号 | 方法 |
|---|---|
| B0 | Nominal only，无安全过滤 |
| B1 | Static-distance CBF-QP |
| B2 | HOCBF或传统sampled-data baseline，二选一即可，但必须说明选择理由 |
| B3 | AegisAir risk-adaptive exact-ZOH RA |

如果ZOCBF-style baseline能够公平适配，再加入；无法公平适配时，不伪称完整复现。

### 3.2 Recovery baseline

在B3基础上运行：

| 编号 | 方法 |
|---|---|
| R0 | RA only，无任务恢复 |
| R1 | RA + deterministic RuleMissionPlanner |
| R2 | RA + LLM + validator + fallback |

---

## 4. 轻量仿真必做实验

### 4.1 2机机制实验

场景只保留：

- head-on；
- perpendicular crossing；
- high closing speed；
- telemetry delay；
- dropout；
- execution lag mismatch。

比较B0–B3。

用途：验证公式、margin、CBF过滤和执行滞后，不做大规模统计。

### 4.2 4机主实验

场景只保留：

- randomized start/goal；
- dense intersection；
- blocked corridor；
- single-drone failure；
- priority conflict。

安全对比：B0、B1、B2、B3。  
任务对比：R0、R1、R2。

这是论文的主要统计实验。

### 4.3 8机最低扩展实验

只做两个场景：

- randomized_8；
- dense/randomized congestion。

只比较：B0、B1、B3。

8机只需要轻量仿真，不必进入Gazebo主矩阵。

---

## 5. C1：多源安全包络最小消融

在4机主实验中至少比较：

1. 固定安全距离；
2. 完整risk-adaptive envelope；
3. 去掉perception margin；
4. 去掉AoI/communication margin。

故障映射：

| 消融 | 必须对应的场景 |
|---|---|
| 去掉perception margin | covariance增大、measurement noise、dropout |
| 去掉AoI margin | telemetry delay、stale state |
| 固定安全距离 | mixed speed、dense crossing |

必须回答：

- 完整包络是否降低boundary violation；
- 是否提高minimum `rho`；
- 是否只是通过停住来减少碰撞；
- stale propagation和AoI margin是否重复补偿。

---

## 6. C2：执行一致性与intersample最小验证

### 6.1 执行模型

保留已有PX4 SITL/Gazebo速度阶跃识别结果，并至少完成一次held-out验证：

- 不同速度命令；
- RUN和STOP响应；
- 拟合得到的`tau`区间；
- held-out响应是否落入经验区间；
- one-step位置/速度预测误差。

经验`tau`区间只能称为empirical coverage，不是全局deterministic bound。

### 6.2 执行模型对比

在2机high-closing-speed场景比较：

- instantaneous/ideal execution；
- 早期启发式执行近似；
- AegisAir exact-ZOH execution model。

至少报告：

- minimum `rho`；
- geometric minimum distance；
- boundary violation；
- intervention magnitude。

### 6.3 Intersample dense audit

在2机和4机Gazebo关键episode中：

- 使用精确ZOH轨迹重建控制周期内部状态；
- 每个控制周期至少采样100个内部时间点；
- 重新计算距离、`d_safe`、projected barrier、squared barrier和`rho`；
- 记录endpoint与intersample最小值。

如果没有解析证明，只能写：

> 在测试范围和审计分辨率下未观察到intersample violation。

不能写成连续时间安全定理。

### 6.4 QP和fallback

至少包含：

- 一个feasible-QP case；
- 一个near-boundary case；
- 一个QP-infeasible case；
- 一个hard-brake fallback case。

分别报告QP可行、QP不可行和fallback，不能把fallback混入feasible-QP保证。

---

## 7. C3：任务恢复最小实验

### 7.1 三个必做任务场景

只保留：

1. drone failure → task reassignment；
2. blocked corridor → yielding/reroute；
3. priority conflict → priority-aware recovery。

### 7.2 三种恢复条件

比较：

- R0：RA only；
- R1：RA + rule recovery；
- R2：RA + LLM recovery。

必须报告：

- critical-task completion；
- overall completion；
- recovery time；
- path/time overhead；
- waiting/deadlock duration；
- repeated CBF intervention duration。

只有LLM相对rule有稳定增益时，才声称LLM带来额外任务价值。否则只声称fail-closed semantic recovery architecture。

### 7.3 最小fail-closed注入

至少测试：

- LLM timeout；
- malformed JSON/schema error；
- schema-valid but semantic-invalid output；
- stale/expired RecoveryPlan。

每种故障至少30次注入，并记录：

- validator是否拒绝；
- 是否触发fallback；
- 是否到达PX4 adapter；
- 是否被RA veto；
- 是否发生safety bypass。

核心gate：

> safety bypass = 0。

---

## 8. PX4 SITL/Gazebo最小矩阵

### 8.1 2机Gazebo

必须做：

- head-on；
- crossing；
- high closing speed；
- delay/dropout；
- execution lag；
- endpoint/intersample audit。

每个关键条件至少10次重复。

### 8.2 4机Gazebo

必须做：

- randomized 4-UAV；
- dense crossing；
- stale/dropout；
- drone failure；
- blocked corridor；
- LLM timeout/invalid output。

每个核心cell至少20个paired seeds；计算资源不足时，优先保证randomized、故障和恢复三类，不扩展场景数量。

### 8.3 8机Gazebo

不是最小必做项。

8机只做轻量仿真即可。后续如果要增强，再加入8机Gazebo压力测试。

---

## 9. 最小统计协议

### 9.1 Seed

- MAPPO training：至少5个training seeds；
- calibration、validation、final test分离；
- final test使用固定paired seeds；
- nominal和所有安全/recovery方法使用相同扰动流。

### 9.2 最低episode数

轻量仿真：

- 4机核心对比：每个cell至少50个paired seeds；
- 4机recovery：每个场景每个条件至少30个paired seeds；
- 8机扩展：每个核心条件至少30个paired seeds。

Gazebo：

- 2机关键场景：每cell至少10次；
- 4机核心场景：每cell至少20个paired seeds；
- fail-closed故障：每类至少30次。

这是最小版本，适合先形成可投稿稿件；后续扩展到100/200 seeds用于增强统计可信度。

### 9.3 最小统计输出

必须报告：

- collision rate；
- boundary violation rate；
- minimum `rho`；
- minimum geometric distance；
- completion rate；
- completion time；
- intervention rate；
- QP infeasible；
- fallback；
- path/time overhead；
- paired difference；
- 95% bootstrap confidence interval。

安全和任务二元指标使用配对检验；连续指标使用paired bootstrap或配对置换检验。不要只报告平均值。

---

## 10. 最小完成判据

### C1通过

- 完整risk-adaptive envelope相对固定距离和关键消融改善对应安全指标；
- 结果不是单纯停住；
- perception/AoI margin作用方向可解释。

### C2通过

- exact-ZOH在执行滞后场景中优于ideal/启发式执行模型；
- endpoint/intersample审计结果可复现；
- QP和fallback被分开报告；
- 没有把endpoint结果夸大为连续时间保证。

### C3通过

- recovery相对RA-only改善至少一个预注册任务指标；
- 安全指标没有明显恶化；
- timeout/invalid/stale均fail-closed；
- safety bypass=0。

### 论文最低可写

至少C1、C2、C3中两个主线通过；第三条如结果不足，必须降级为系统设计或限制性结果，不通过新增模块挽救。

---

## 11. 以后再补的实验

以下全部从最小必做清单中移出，后续有时间再加：

- 8机Gazebo完整矩阵；
- 16/32机轻量仿真；
- GCBF+完整复现；
- ZOCBF完整理论复现；
- 全量C1组合消融；
- full-factorial delay × dropout × noise；
- adversarial scenario search；
- failure envelope全参数扫描；
- CV/CPA与神经预测器比较；
- Qwen3-8B、Phi-4、QLoRA；
- 100/200+ final seeds；
- 真实硬件飞行；
- robust-`tau` episode-level新控制逻辑；
- 16/32机Gazebo；
- 三维高度和移动障碍物完整实验。

这些实验不能在最小版本结果不理想时临时加入来“救”假设；只能作为预先说明的后续增强。

---

## 12. 最小执行顺序

1. 冻结配置、seed、阈值和日志；
2. 完成MAPPO 5 training seeds；
3. 完成2机轻量机制测试；
4. 完成4机轻量主对比；
5. 完成C1最小消融；
6. 完成PX4执行模型held-out验证；
7. 完成2机Gazebo执行和intersample审计；
8. 完成4机Gazebo主闭环；
9. 完成三类任务恢复；
10. 完成四类fail-closed故障注入；
11. 完成8机轻量扩展；
12. 运行统一统计脚本；
13. 按C1/C2/C3完成判据决定论文claim；
14. 写最小版本论文；
15. 后续再增加大样本、8机Gazebo、OOD、failure envelope和硬件证据。

---

## 13. 最终边界

本最小版本可以支持：

- risk-adaptive multi-source safety envelope；
- execution-consistent sampled-data RA；
- fail-closed semantic mission recovery；
- PX4 SITL/Gazebo closed-loop evidence。

本最小版本不能支持：

- 任意状态和任意故障下绝对安全；
- intersample continuous-time theorem；
- 全局PX4执行时间常数保证；
- LLM实时避碰；
- 真实硬件飞行安全；
- 大规模分布式多机保证；
- robust-`tau` episode-level benefit。
     精简回答 节约token