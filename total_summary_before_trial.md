# AegisAir 正式实验前全系统总结

> 更新日期：2026-08-17
>
> 状态：Phase 0--7 实验前工程与方法验证结束；robust-τ 支线 CLOSED；下一阶段只做冻结协议下的正式实验与论文写作。
>
> 仓库：`/Users/lijiajun/Documents/drone/AegisAir`
>
> 数据原则：代码和文档留在仓库；原始数据、checkpoint 与批量结果保存在移动硬盘。

## 1. 研究问题与总体思路

AegisAir 研究：当多无人机系统中的 MARL、预测器、共享状态估计和本地 LLM
都可能出错时，如何让独立 Runtime Assurance（RA）保留最终安全控制权，同时
通过任务级语义恢复减少纯安全过滤造成的停滞和任务损失。

```text
MARL / LLM / Predictor / Estimator 都不是安全证明本身
                         ↓
Risk-Adaptive Runtime Assurance 独立审查每个 nominal action
                         ↓
短时危险由 sampled-data barrier 立即处理
                         ↓
持续冲突、失效和任务变化由异步 Semantic Mission Manager 重规划
```

论文主线冻结为：

```text
Risk-adaptive safety envelope
        +
sampled/execution-aware runtime assurance
        +
semantic mission recovery
```

一句话概括：`Protect now, understand later, replan the future.`

## 2. 冻结后的系统架构

```text
Mission / User
   ↓ goals, priorities, constraints
Semantic Mission Manager
   ├── compact LLM decision
   ├── deterministic schema/semantic validation
   └── deterministic fallback and plan expansion
   ↓ mission waypoint / reassignment
MARL Nominal Pilot（MAPPO checkpoint 或 rule pilot）
   ↓ u_nom
SharedStateEstimator
   ├── delayed/dropout-prone telemetry
   ├── state estimate and covariance
   └── age of information (AoI)
   ↓ estimated shared state
Risk-Adaptive Runtime Assurance
   ├── dynamic safety envelope
   ├── normalized safety margin and degradation
   ├── CV/CPA predictive pre-alert
   └── sampled/execution-aware barrier QP
   ↓ u_safe
PX4 Offboard → PX4 SITL / Gazebo
   ↑
   └── RiskEvent / mission-validity event → asynchronous replanning
```

安全与任务职责严格分离：RA/CBF 是同步安全层，不能被 LLM 绕过；Predictor
只提供 pre-alert；LLM 不做实时避碰，只做异步任务推理；validator、fallback
和 PX4 adapter fail-closed 均为确定性边界。

## 3. Risk-adaptive safety envelope

### 3.1 相对状态与接近速度

对任意 UAV pair `(i,j)`：

```text
r_ij = p_i - p_j
v_ij = v_i - v_j
n_ij = r_ij / ||r_ij||
v_cl = max(0, -r_ij^T v_ij / ||r_ij||)
```

`v_cl=0` 表示两机未沿连线方向接近；正值越大，动态安全距离越大。

### 3.2 动态安全边界

```text
d_safe = d0 + M_dyn + M_perc + M_comm
```

动力学裕度：

```text
M_dyn = v_cl(τ_r + τ_ctrl) + v_cl²/(2a_eff)
```

它包含反应/控制延迟内的继续接近距离，以及有限有效减速度下的制动距离。

感知裕度：

```text
σ_i = sqrt(n_ij^T P_i n_ij)
σ_j = sqrt(n_ij^T P_j n_ij)
M_perc = β_conf sqrt(σ_i² + σ_j²)
```

`P_i,P_j` 是共享状态估计器输出的位置协方差；无协方差时回退到固定
`perception_sigma`。`β_conf` 是工程置信系数，不描述成已校准的概率保证。

通信/AoI 裕度：

```text
M_comm = v_max A + 0.5 a_max A²
```

其中 `A` 是 pair telemetry age。它对 stale state 提供额外 reachable-distance
buffer。

### 3.3 安全裕度、barrier 与退化速度

```text
ρ = (d-d_safe)/d_safe
h = ||r_ij||²-d_safe²
g_k = (ρ_{k-1}-ρ_k)/Δt
g_bar,k = λg_bar,k-1+(1-λ)g_k
```

- `ρ>0`：位于动态安全边界之外；
- `ρ=0`：边界；
- `ρ<0`：违反当前模型下的安全包络，但不等同于几何碰撞；
- `g_bar>0`：安全裕度持续恶化。

## 4. 状态估计与预测监视器

### 4.1 SharedStateEstimator

对 stale telemetry，先传播到当前控制时刻：

```text
p_hat(t_now) = p(t_stamp)+v(t_stamp)(t_now-t_stamp)
```

协方差随未接收新测量的时间增长：

```text
P = clip(P_measurement+Q_process Δt_rx, 0, P_max)
```

- delay/AoI：状态前向传播，同时进入 `M_comm`；
- dropout：hold 最近状态并增长 covariance；
- covariance：沿 pair 连线投影后进入 `M_perc`。

这仍是 centralized shared-state architecture，不是完整分布式 ego-only 感知。

### 4.2 CV/CPA predictive pre-alert

```text
t_CPA = clip(-r^T v/||v||², 0, H)
d_CPA = ||r+vt_CPA||
ρ_hat(t) = (d_hat(t)-d_safe(t))/d_safe(t)
ρ_hat_min = min_{t∈[0,H]} ρ_hat(t)
TTSB = inf{t:ρ_hat(t)≤0}
```

未来 uncertainty：

```text
σ²(t)=σ²(0)+q_pred t
AoI(t)=AoI(0)+t
```

Phase 3 已冻结 Predictor 为 pre-alert：`Prediction suggests; runtime evidence
confirms.` 其误差 heavy-tailed，不能把 reliability score 写成严格概率。

## 5. Runtime Assurance 控制公式

### 5.1 velocity CBF

在单积分近似 `p_dot=u` 下：

```text
2r^T u_i ≥ 2r^T v_j-α_cbf h
u_safe = argmin_u ||u-u_nom||²
         s.t. pairwise CBF constraints and velocity bounds
```

### 5.2 acceleration HOCBF

双积分模型：

```text
p_dot=v, v_dot=a
ψ2=h_ddot+(k1+k2)h_dot+k1k2h≥0
```

冻结单个 QP step 内的安全边界导数后：

```text
2r^T(a_i-a_j) ≥
  -2||v||²-2(k1+k2)r^T v-k1k2(||r||²-d_safe²)
```

HOCBF 是重要基线；PX4 velocity execution 下最终使用下面的精确 sampled-data
模型。

### 5.3 PX4 一阶速度执行模型：精确 ZOH 离散化

```text
v_dot=(u-v)/τ
α(τ)=1-exp(-Δt/τ)
β(τ)=Δt-τα(τ)

v_{k+1}=(1-α)v_k+αu_k
p_{k+1}=p_k+v_kΔt+β(u_k-v_k)
```

以虚拟加速度参数化速度命令：

```text
u_k=v_k+a_kΔt
v_{k+1}=v_k+αa_kΔt
p_{k+1}=p_k+v_kΔt+βa_kΔt
```

`τ→0` 时 `α→1,β→Δt`。该公式替换早期启发式 `0.5αaΔt²`，是
sampled/execution-aware RA 的正式实现。

### 5.4 sampled-data projected barrier

```text
n=r_k/||r_k||
h_under,k=n^T r_k-D_k
r_{k+1}=r_k+v_ij,kΔt+Δt(β_i a_i-β_j a_j)
```

由于 `n^T r≤||r||`，`h_under` 是保守投影 barrier。离散条件：

```text
n^T r_{k+1}-D_{k+1} ≥ (1-γ)h_under,k
```

QP：

```text
min_{a_1,...,a_N} Σ_i ||a_i-a_nom,i||²
```

subject to pairwise sampled-data barriers、`||a_i||∞≤a_max` 和下一拍速度界。
若不可行，执行确定性 hard-brake：

```text
a_safe=clip(-v,-a_max,a_max)
```

fallback 是 fail-safe 行为，但不属于 feasible-QP theorem guarantee。

## 6. identified execution uncertainty 与 robust-τ 分析

### 6.1 经验执行包络

PX4 SITL/Gazebo 阶跃协议：`vx∈{0.5,1.0,1.5} m/s`，每速度 3 次；r1/r2
identification、r3 held-out；RUN/STOP 前 3 s 拟合一阶响应，异常振荡不删除。

```text
T_empirical=[τ_min,τ_max]=[0.53,1.76] s
τ_hat=0.7 s
```

held-out r3 的 RUN/STOP 拟合均在区间内。它是本轮样本支持的 empirical coverage，
不是所有 PX4 配置、方向和工况下的 deterministic bound。

### 6.2 robust sampled-data constraint

```text
inf_{τ_i∈T_i,τ_j∈T_j}
 [n^T r_{k+1}(τ_i,τ_j)-D_{k+1}^rob]
 ≥ (1-γ)h_under,k
```

下一拍安全边界也 robustify：

```text
D_{k+1}^rob=max_{(τ_i,τ_j)∈vertices}
             d_safe(v_cl,k+1(τ_i,τ_j))
```

投影约束对 `(β_i,β_j)` affine，因此 rectangle 四个 vertices 足以覆盖 projected
formulation。每 pair 4 条；4 UAV、6 pairs 共 24 条。

原始 squared-distance barrier：

```text
h_{k+1}(τ_i,τ_j)=||r_{k+1}(τ_i,τ_j)||²-D_{k+1}²
```

最坏点不先验保证在端点，因此 Phase 7 用 `50×50` dense τ grid 单独审计。

### 6.3 one-step activation

```text
G(τ_i,τ_j;x_k,u_k)
 =h_projected,k+1(τ_i,τ_j)-(1-γ)h_projected,k
```

搜索目标：

```text
G(τ_hat,τ_hat)≥0
min_{τ_i,τ_j∈T}G(τ_i,τ_j)<0
```

即 nominal execution model 接受动作，但经验允许的执行响应中至少一种使 robust
condition 不成立，因此 robust controller 必须改变动作。

### 6.4 Phase 7 结果与边界

找到 225 个 `G_nominal≥0,G_robust<0,Δu>0` 的 case，其中：

```text
robust-QP feasible          44
robust-QP infeasible       181
projected dense pass       44/44 feasible
exact squared-h dense pass 44/44 feasible
```

代表性 feasible case：

```text
δ=0.45536 m
v_cl=0.90 m/s
yielding actual speed=0.10 m/s
G_nominal=+1.0e-6
G_robust=-1.6267e-4
Δu=2.1745 m/s²
```

支持的结论：identified execution uncertainty 改变 feasible one-step safety
decision，且 robust action 在 feasible subset 上通过 projected 与 exact squared-h
dense audit。

不支持：infeasible fallback theorem、intersample continuous safety，或
episode-level mission benefit。

### 6.5 episode-level null result 与关闭决定

后续测试 1D head-on、2D near-crossing、finite-state guard、1--3 s rollback、二维
recovery corridor 和 `SEPARATE→CROSS→REJOIN`。最终 handoff 试验：

```text
violation=0
QP infeasible=0
completion=0
nominal result=robust result
```

one-step correctness 未转化为可测量 episode-level efficacy。继续添加 guard 或
planning state 已转向 recursive feasibility/viability/mission planning。

```text
robust-τ branch: CLOSED
```

论文定位：`Execution-uncertainty-aware one-step robustness analysis.`

建议 limitation：

> Robustifying identified execution uncertainty changes one-step safety
> decisions under model uncertainty, but did not produce measurable
> episode-level gains in the tested recovery corridors.

## 7. Semantic Mission Recovery

### 7.1 分层触发

```text
Level 1: Predictive pre-alert（预热，不改任务）
Level 2: Runtime confirmation（ρ低且持续恶化/CBF持续，允许任务修改）
Level 3: Hard RA/CBF（独立即时安全兜底）
```

```text
E_replan=E_safety(route deviation)
         OR E_mission(task/priority/environment change)
         OR E_coord(waiting/repeated conflict)
```

单次 imminent collision 不调用 LLM；由 RA 直接处理。

### 7.2 LLM 与确定性边界

```text
compact MissionDecision JSON
 → syntax validation
 → schema validation
 → semantic validation
 → deterministic RecoveryPlan expansion
 → execution acceptance
 → RA final veto
```

任一层失败回退 `RuleMissionPlanner`。timeout、invalid action、schema error 均不能
绕过 validator 或 PX4 adapter。

### 7.3 任务恢复几何

```text
c=mean(p_j,j≠i)
p_hold=p_i+normalize(p_i-c)·offset
```

safe holding 修复 yielding UAV 原地停车、反被 moving UAV 追撞的危险语义。

```text
e=normalize(goal-p)
n_perp=perpendicular(e)
p_waypoint=p+sign(agent_id)·2n_perp
```

deadlock/corridor waypoint 均裁剪到 arena。

## 8. Phase 0--7 证据链

### Phase 0：接口冻结

冻结 Telemetry、Observation、MarlAction、RiskEvent、RecoveryPlan、
SafetyDecision、LogEvent 等 schema；额外字段不能静默进入安全路径。

### Phase 1：MARL nominal pilot

完成多机环境、vectorized rollout 和 MAPPO。dense randomized 场景中纯 MARL
存在明显 collision，证明 nominal autonomy 不能承担硬安全。

### Phase 2：Risk-Adaptive Runtime Assurance

动态边界、`ρ`、`g`、velocity-CBF/HOCBF/sampled-data QP 完成。历史轻量评测：
randomized_4 collision 18%→0%、completion 78%→96%；randomized_8 collision
59%→1%、completion 38%→85%。正式实验仍按最终冻结实现重跑。

### Phase 3：预测监视器

冻结 Filtered CA/CPA、`H=1.5 s`、`ρ_threshold=0`。untouched seeds 15--24：
precision 49.5%、recall 54.9%、FPR 1.0%、TTSB MAE 0.18 s。负结果是 prediction
error heavy-tailed，因此 Predictor 降级为 pre-alert。

### Phase 4：异步语义任务恢复

完成 `AsyncMissionReplanner`、strict validator、deterministic fallback。本地
Qwen3-4B-4bit compact decision P50 约 2.8 s。三场景 10 seeds、30 次真实 LLM
调用校验全过；drone failure 关键任务完成率 0→1.0，blocked corridor zone
crossing 1.0→0.0，priority conflict 有温和改善。只 claim 异步 mission recovery。

### Phase 5：PX4/Gazebo 闭环

telemetry→estimator→nominal pilot→RA→velocity setpoint→PX4 Offboard 闭环完成。
safe holding、AoI propagation 和 execution-aware barrier 修复后，代表性 4-UAV
live run 从 `min_rho=-0.508` 改善到 `+0.564`，`min_distance=1.523 m`。

### Phase 6：故障注入与 fail-closed

- adapter local gate：90 episodes，`bypass=0`；
- mission fault gate：60 episodes，stale/latency/LLM timeout/invalid stop rules 全过；
- estimator gate：50 episodes，delay 增 age、dropout 增 covariance；
- live single-UAV：1100 ms expired command 100% fail-closed；
- live 4-UAV：baseline/stale/dropout/LLM timeout/invalid 均无几何碰撞，LLM 故障
  均触发 deterministic fallback。

### Phase 7：execution identification 与 robust-τ closure

- 9 条 PX4 step-response records；`[0.53,1.76] s`，held-out r3 covered；
- 225 个 activation cases；44 feasible cases 的 projected/exact dense audit 全过；
- episode-level nominal-vs-robust 无稳定收益；
- robust-τ branch 按 stop condition 关闭。

## 9. 论文 claim 边界

### 可以支持

- risk-adaptive multi-source safety envelope 的实现与故障响应；
- sampled/execution-aware RA 在给定模型与可行条件下修改不安全 nominal action；
- adapter、validator 和 recovery fallback 的 deterministic fail-closed 路径；
- semantic mission recovery 对失效重分配、blocked corridor 等长时程变化的作用；
- identified τ uncertainty 会改变部分 feasible one-step safety decisions。

### 不可以支持

- 对任意 MARL、初始状态或故障的绝对安全；
- empirical τ interval 是 PX4 global deterministic bound；
- QP infeasible/hard-brake 状态满足同一 theorem；
- sampled endpoint safety 自动推出 intersample safety；
- Predictor reliability 是严格概率；
- LLM 是实时 collision-avoidance controller；
- robust-τ 已证明 episode-level benefit；
- SITL/Gazebo 等同于真实硬件飞行。

## 10. 正式实验冻结原则

1. 不为改善结果继续添加 guard、planning state 或新 controller；
2. 不放松 `ρ`、collision、completion 或 feasibility 阈值；
3. 不追加 seed、删除异常 trial 或改变场景定义救失败假设；
4. calibration/validation/final test seed 保持分离；
5. nominal 与 RA/recovery 条件使用 paired seeds 和相同扰动；
6. 报告 collision、boundary violation、min `ρ`、completion、intervention、QP
   infeasible、fallback、path/time overhead 及 uncertainty interval；
7. 原样保留 Predictor 限制和 robust-τ episode null；
8. 区分 lightweight simulation、PX4 SITL/Gazebo 与真实硬件证据。

正式主对比：

```text
nominal autonomy
vs nominal + risk-adaptive RA
vs nominal + RA + semantic mission recovery
```

## 11. 关键文件与结果

- `FORMULAS.md`：公式索引；
- `swarm/ra/margins.py`、`margin.py`、`hocbf.py`、`runtime_assurance.py`：RA；
- `swarm/estimation.py`：SharedStateEstimator；
- `swarm/ra/predictor.py`：predictive pre-alert；
- `swarm/recovery/`：语义恢复、validator、fallback；
- `marllib/phase5_runner.py`：sim/live 主闭环；
- `marllib/robust_activation_search.py`：one-step activation 与 dense audit；
- `docs/decisions/phase4_summary.md`；
- `docs/decisions/phase6_fault_injection.md`；
- `docs/decisions/phase7_step_response_identification.md`；
- `docs/decisions/phase7_robust_activation.md`。

Phase 7 外部结果：

```text
/Volumes/Expansion/aegisair_phase7_step_response_20260817/
```

## 12. Git 状态（生成本总结时）

```text
branch: main
HEAD: 9e7a1f3
origin/main...HEAD: 0 ahead / 0 behind
```

没有“已 commit 但未 push”的提交。但 Phase 7 实现、测试、决策记录和本总结仍在
working tree，尚未 commit，因此也尚未 push。提交前需在项目环境运行完整测试，
并审查全部 Phase 7 closure 文件。

## 13. 最终阶段结论

```text
Phase 0--7 pre-experiment development: COMPLETE
robust-τ branch: CLOSED
main research line: FROZEN FOR FORMAL EXPERIMENTS
next: run frozen experiments, analyze statistics, write the paper
```

AegisAir 最重要的贡献不是证明某个 τ-robust controller 在所有 episode 中更好，
而是建立可审计 evidence chain：多源风险进入动态安全包络，执行模型进入
sampled-data RA，AI failure 被 deterministic safety/fallback 隔离，长时程任务
损失再由 semantic mission recovery 处理。
