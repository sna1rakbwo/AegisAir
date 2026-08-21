# AegisAir 正式实验增强版协议（扩规模，草案）

状态：**草案，运行前冻结。** 目标：把已通过的最小版本结论增强到面向二区的统计强度与规模。
原则：扩规模是**预注册的增强**，不是“追加 seed 把结果做显著”；结果无论过/不过都原样报告。

## 1. 目的

- C1 / C2 / C3 的最小版本（30–50 seed）已通过各自判据。本协议把主矩阵扩到 200 seed，
  收紧置信区间、证明结论不是小样本巧合。
- 证据等级不变：轻量仿真 + PX4 SITL/Gazebo，不含真机、不含连续时间定理。

## 2. 运行前冻结对象

- seed manifest（与最小版本 disjoint，见 §3）
- 场景、阈值、指标、统计方法（沿用最小协议，不放松）
- MAPPO checkpoint、LLM 模型/参数、RA 参数（沿用已冻结值）
- 原始产物哈希 + 外部盘路径（不覆盖最小版本产物）

## 3. 扩规模矩阵

| 支柱 | 场景 × 条件 | seed 数 | 新 seed 范围 | 相对最小版本 |
|---|---|---:|---:|---|
| C1 | 4 场景 × 4 消融 | 200 | 251–450 | 50 → 200（800 → 3200 episode） |
| C2 | 2 机 high-closing × 3 执行模型 | 200 | 121–320 | 20 → 200 |
| C2 | 4 机 multi_uav E2 dense audit | 30 | 211–240 | 10 → 30 |
| C3 R0/R1 | 3 场景 × 2 条件 | 200 | 31–230 | 30 → 200（180 → 1200 episode） |
| C3 R2 | 3 场景 × 1 条件（真 LLM） | 30（保持） | 1–30 | 不变 |
| C3 | 五类故障注入（priority_conflict） | 30（保持） | 1–30 | 不变 |

> C3 的 R2（真 4B LLM）保持 30 seed：它已是负结果（无稳定任务收益），且真实模型推理慢、
> 扩规模性价比低；30 seed 足以作为负结果证据。只把确定性的 R0/R1（正向架构结论）扩到 200。

## 4. PX4/Gazebo 最小闭环矩阵（硬件在环 / 部署边界）

补上最小协议 §8 承诺但未跑的 Gazebo 闭环，使论文「PX4 SITL/Gazebo」证据名实相符。
**主统计仍在轻量仿真（§3，200 seed）**；Gazebo 是小样本的部署边界证据，只做方向性对比，
不做强 CI 结论。基础设施已具备：本机 PX4 SITL v1.16 + Gazebo Sim 8.14 + ROS 2 Humble +
`px4_adapter`（Docker）。

| 支柱 | 规模 | 场景 | 条件 | seed/条件 |
|---|---|---|---|---:|
| C1 | 2 机 | head-on / crossing | B0 / B1 / B3 | 10 |
| C2 | 2 机 | high-closing | E0 / E1 / E2（执行模型） | 10 |
| C1 | 4 机 | randomized / dense | B0 / B1 / B3 | 20 |
| C3 | 4 机 | drone failure / blocked corridor | R0 / R1 | 20 |

判据（方向性，不要求达到轻量仿真的精确数字）：

- 安全：B3 相对 B0 降低 collision / boundary violation，min rho 方向一致。
- 执行一致性（C2）：E2（exact-ZOH）相对 E1（历史启发式）在真实 PX4 执行滞后下保持更高
  安全裕度 / 更大物理分离，方向与轻量仿真一致。
- 恢复：R1 相对 R0 恢复对应预注册任务指标。
- 全程 `safety bypass = 0`。

边界与统计：

- 小样本（10–20 seed），只报告描述性/方向性对比，**不报 bootstrap CI 强结论**。
- C2 的 execution-lag：Phase 7 empirical tau 是执行模型的**输入**；E0/E1/E2 对比必须在
  Gazebo 真实 PX4 执行回路里跑（而不是只在轻量 exact-ZOH plant 里自证）。intersample
  100 点**解析重建只在轻量仿真可行**，Gazebo 改用高频遥测日志检查控制周期内部距离/rho，
  方法不同需单独说明，不能混写成同一种证据。
- 8 机 Gazebo 不是最小，继续推迟。

## 5. 8 机轻量扩展（推迟）

原计划的 8 机轻量扩展（`randomized_8` / dense congestion × B0/B1/B3）**推迟到投稿返修按需
补做**，不进入本轮冻结。它是增强而非主线必需。

## 6. 统计

- 所有安全/任务二元指标：配对检验（nominal 与 RA/recovery 用相同扰动流）。
- 连续指标：paired bootstrap（10,000 draws），报告 95% CI。
- 报告：collision rate、boundary violation、min rho、min distance、completion、
  intervention、QP infeasible、fallback、配对差 + 95% CI。

## 7. 停止规则 / 判据（预注册，不救失败）

沿用最小版本判据，不因“想发二区”放宽：

- C1 通过：完整包络相对固定距离与关键消融改善对应安全指标，且非单纯停住。
- C2 通过：exact-ZOH 在执行滞后场景优于历史启发式；endpoint/intersample 可复现；QP 与
  fallback 分开报告。
- C3 通过：恢复相对 RA-only 改善至少一个预注册任务指标；安全不恶化；五类故障
  safety bypass = 0。

若 200 seed 下某支柱不再满足判据 → 冻结为“样本增大后结论不再成立”的降级/负结果，
不追加 seed、不改阈值、不加模块挽救。

## 8. claim 边界

- 可写：在 200-seed 主矩阵范围内，C1/C2/C3 的对应结论成立。
- 可写：在最小 Gazebo 闭环矩阵内，RA/recovery 在真实 PX4 执行回路下方向性成立（小样本、
  描述性）。
- 不可写：真机安全、连续时间定理、全局 PX4 时间常数、LLM 实时避碰或任务收益、
  大规模分布式保证。

## 9. 执行顺序（least-cost）

1. 冻结本协议（锁定 seed/指标/判据）+ 提交；
2. 先跑 C1 200 seed（最核心、决定论文主线）；
3. 再跑 C2、C3（R0/R1）；
4. 跑最小 Gazebo 闭环矩阵（2 机 → 4 机）；
5. 统一统计脚本输出配对差 + CI；
6. 按 §7 逐支柱决定 Go/No-Go，写中文决策文档。
