# SafeDrones 综合研究与开发计划

> Version: 1.0\
> Date: 2026-08-15\
> 目标：将当前 SafeDrones 原型发展为可复现、可系统验证、具备 SCI
> 投稿潜力的多无人机研究系统。\
> 核心方向：**Risk-Adaptive Runtime Assurance for LLM--MARL Multi-UAV
> Autonomy with Predictive Safety Monitoring and Semantic Recovery**

------------------------------------------------------------------------

# 1. 核心研究问题

不把原始 **Safety Gate** 或 **Safety → LLM feedback**
总体构想作为论文唯一创新。

论文集中回答：

> **当 LLM 高层规划器和 MARL
> 学习控制器都可能出错时，能否通过独立的风险自适应 Runtime Assurance
> 保持多无人机硬安全，并利用安全裕度及其预测/退化信息触发有效的任务级语义恢复？**

核心假设：

\[ `\boxed{\text{LLM = untrusted},\qquad \text{MARL = untrusted}}`{=tex}
\]

而确定性的 Runtime Assurance 保留最终安全控制权。

LLM 永远不能降低或绕过硬安全约束。

------------------------------------------------------------------------

# 2. 最终系统架构

``` text
Mission / User
      ↓
Local LLM Commander
      ↓
High-level goals / tasks
      ↓
MARL Nominal Pilot
      ↓
Risk-Adaptive Runtime Assurance
      ├── Dynamic Safety Boundary
      ├── Normalized Safety Margin
      ├── Safety-Margin Degradation
      ├── CV / CPA Predictive Monitor
      └── Time-Varying CBF / QP
      ↓
PX4 Offboard
      ↓
PX4 SITL / Gazebo

Runtime Assurance
      ↓
Structured RiskEvent
      ↓
Local LLM Semantic Recovery
      ↓
New mission plan
```

总体逻辑：

\[ `\boxed{
Risk\rightarrow Margin\rightarrow Prediction\rightarrow
Runtime\ Assurance\rightarrow Semantic\ Recovery
}`{=tex} \]

------------------------------------------------------------------------

# 3. LLM 层

## 主模型

优先：

-   Qwen3-4B；
-   本地部署；
-   4-bit quantization；
-   non-thinking mode；
-   constrained structured output。

模型消融可加入：

-   Qwen3-8B；
-   Phi-4-mini-instruct。

第一版不微调，采用：

``` text
Prompt
+
JSON Schema
+
Action Whitelist
+
Validator
```

允许高层动作：

``` text
HOLD
YIELD
REROUTE
REASSIGN
CHANGE_PRIORITY
ABORT
RETURN
```

LLM 禁止修改：

-   (d_0)；
-   CBF constraint；
-   uncertainty confidence factor；
-   Safety Gate threshold；
-   actuator safety limit。

只有在数百至 1000+ recovery cases 中表现不足时，再按：

``` text
Zero-shot
↓
Few-shot
↓
QLoRA
```

逐步升级。

微调只提升 semantic recovery，不承担硬安全保证。

------------------------------------------------------------------------

# 4. MARL Nominal Pilot

推荐算法：

\[ `\boxed{\text{MAPPO}}`{=tex} \]

采用 CTDE：

-   centralized training；
-   decentralized execution。

MARL 负责：

\[ `\boxed{o_i\rightarrow \pi_\theta\rightarrow u_i^{nom}}`{=tex} \]

不负责最终硬安全。

## Observation

优先使用相对量：

\[ o_i=\[ v_i,, p_i\^{goal}-p_i,, p_j-p_i,, v_j-v_i,`\ldots`{=tex}\] \]

## Action

第一版固定高度二维：

\[ `\boxed{a_i=[v_x^{cmd},v_y^{cmd}]}`{=tex} \]

成功后升级：

\[ a_i=\[v_x^{cmd},v_y^{cmd},v_z\^{cmd}\] \]

MARL 不直接输出 thrust、attitude、torque 或 motor command。

------------------------------------------------------------------------

# 5. MARL 训练环境

不要直接在 PX4/Gazebo 里进行百万级训练。

建立轻量 Gymnasium/PettingZoo 风格环境。

基础动力学：

\[ p\_{t+1}=p_t+v_t`\Delta `{=tex}t \]

或加入加速度限制：

\[ v\_{t+1} = v_t+ `\operatorname{clip}`{=tex}
(v_t\^{cmd}-v_t,-a\_{max}`\Delta `{=tex}t,a\_{max}`\Delta `{=tex}t) \]

\[ p\_{t+1}=p_t+v\_{t+1}`\Delta `{=tex}t \]

推荐：

\[ `\Delta `{=tex}t`\approx`{=tex}0.1s \]

轻量环境用于训练，PX4/Gazebo 用于最终系统验证。

------------------------------------------------------------------------

# 6. MARL Reward

初始奖励：

\[ `\boxed{
r_i=
r_{goal}
+r_{progress}
+r_{collision}
+r_{near}
+r_{smooth}
+r_{time}
}`{=tex} \]

目标奖励：

\[ r\_{goal}=
```{=tex}
\begin{cases}
+R_g,&d_{goal}<\epsilon\\
0,&otherwise
\end{cases}
```
\]

进度奖励：

\[ `\boxed{
r_{progress}
=
k_p(d_{goal}^{t-1}-d_{goal}^{t})
}`{=tex} \]

碰撞：

\[ r\_{collision}=-R_c \]

near-collision：

\[ r\_{near} = -k_n`\max`{=tex}(0,d\_{near}-d\_{ij})\^2 \]

平滑性：

\[ r\_{smooth} = -k_s\|a_t-a\_{t-1}\|\^2 \]

另加小幅 timestep penalty。

不要把完整 Adaptive Safety Gate 复制进 MARL reward。

------------------------------------------------------------------------

# 7. MARL Curriculum

按顺序：

1.  单 UAV start → goal；
2.  2 UAV head-on；
3.  2 UAV perpendicular / diagonal crossing；
4.  randomized starts/goals；
5.  4 UAV；
6.  8 UAV dense traffic；
7.  可选 16/32 UAV scalability。

训练初期不要加入所有故障。

------------------------------------------------------------------------

# 8. Domain Randomization

训练时随机：

-   start / goal；
-   speed limit；
-   acceleration limit；
-   timestep；
-   position noise；
-   velocity noise；
-   moderate command delay。

目的：

\[ Lightweight Simulator `\rightarrow`{=tex} PX4/Gazebo \]

主 MARL 训练默认不开完整 Safety Gate。

最终不同 safety variant 使用相同 MARL checkpoint。

------------------------------------------------------------------------

# 9. MARL 训练协议

初始预算：

\[ 2`\times10`{=tex}^6`\rightarrow10`{=tex}^7 \]

environment steps，根据 learning curve 调整。

周期保存：

``` text
π20%
π50%
π80%
π100%
```

正式训练至少：

\[ `\boxed{5\ training\ seeds}`{=tex} \]

记录：

-   return；
-   mission success；
-   collision rate；
-   path efficiency；
-   smoothness。

中间 checkpoint 用于验证 Runtime Assurance 是否能保护 weak/medium/strong
policy。

------------------------------------------------------------------------

# 10. Dynamic Safety Boundary

从固定阈值升级为：

\[ `\boxed{
d_{ij}^{safe}
=
d_0+
M_{ij}^{dyn}
+
M_{ij}^{perc}
+
M_{ij}^{comm}
}`{=tex} \]

------------------------------------------------------------------------

# 11. Dynamics Margin

定义：

\[ n\_{ij} = `\frac{p_i-p_j}{\|p_i-p_j\|}`{=tex} \]

closing speed：

\[ `\boxed{
v_{ij}^{cl}
=
\max\left(
0,
-\frac{(p_i-p_j)^T(v_i-v_j)}
{\|p_i-p_j\|}
\right)
}`{=tex} \]

动态裕度：

\[ `\boxed{
M_{ij}^{dyn}
=
v_{ij}^{cl}\tau_r
+
\frac{(v_{ij}^{cl})^2}{2a_{eff}}
}`{=tex} \]

------------------------------------------------------------------------

# 12. Perception Margin

如果有 covariance：

\[ `\boxed{
M_{ij}^{perc}
=
\beta
\sqrt{
n_{ij}^{T}
(\Sigma_i+\Sigma_j)
n_{ij}
}
}`{=tex} \]

感知实验逐步：

``` text
GT
↓
Noisy GT
↓
Ego
↓
Stale Ego
↓
Missing / degraded perception
```

后续可选升级：让 (`\beta`{=tex}) 从目标 collision-risk / chance
constraint 中校准，而不是纯经验设定。

------------------------------------------------------------------------

# 13. Communication / AoI Margin

\[ A\_{ij}(t)=t-t\_{ij}\^{last} \]

\[ `\boxed{
M_{ij}^{comm}
=
v_{j,max}A_{ij}
+
\frac12a_{j,max}A_{ij}^2
}`{=tex} \]

统一表示：

-   communication delay；
-   packet loss；
-   stale telemetry。

------------------------------------------------------------------------

# 14. 综合动态安全边界

\[ `\boxed{
\begin{aligned}
d_{ij}^{safe}
=&\ d_0
+v_{ij}^{cl}\tau_r
+\frac{(v_{ij}^{cl})^2}{2a_{eff}}\\
&+\beta
\sqrt{
n_{ij}^{T}(\Sigma_i+\Sigma_j)n_{ij}
}\\
&+v_{j,max}A_{ij}
+\frac12a_{j,max}A_{ij}^2
\end{aligned}
}`{=tex} \]

对应：

\[ Physical+Dynamics+Perception+Communication \]

------------------------------------------------------------------------

# 15. Normalized Safety Margin

\[ `\boxed{
\rho_{ij}(t)
=
\frac{
d_{ij}(t)-d_{ij}^{safe}(t)
}{
d_{ij}^{safe}(t)
}
}`{=tex} \]

解释：

-   (`\rho`{=tex}\>0)：safe；
-   (`\rho=0`{=tex})：dynamic safety boundary；
-   (`\rho`{=tex}\<0)：unsafe region。

------------------------------------------------------------------------

# 16. Safety-Margin Degradation

\[ `\boxed{
g_{ij}(t)
=
\frac{
\rho_{ij}(t-\Delta T)-\rho_{ij}(t)
}{
\Delta T
}
}`{=tex} \]

(g\>0) 表示 margin 正在恶化。

EMA：

\[ `\boxed{
\bar g_t=
\lambda\bar g_{t-1}
+(1-\lambda)g_t
}`{=tex} \]

降低 sensor jitter 引起的误触发。

------------------------------------------------------------------------

# 17. Predictive Safety Margin

预测只用于 early warning，不取代硬安全 CBF。

## Constant Velocity

\[ `\hat `{=tex}p_i(t+`\tau`{=tex})=p_i(t)+v_i(t)`\tau`{=tex} \]

\[ `\hat `{=tex}p_j(t+`\tau`{=tex})=p_j(t)+v_j(t)`\tau`{=tex} \]

\[ `\hat `{=tex}d\_{ij}(t+`\tau`{=tex}) =
\|`\hat `{=tex}p_i(t+`\tau`{=tex})-`\hat `{=tex}p_j(t+`\tau`{=tex})\| \]

\[ `\boxed{
\hat\rho_{ij}(t+\tau)
=
\frac{
\hat d_{ij}(t+\tau)-\hat d_{ij}^{safe}(t+\tau)
}{
\hat d_{ij}^{safe}(t+\tau)
}
}`{=tex} \]

预测窗口：

\[ H=1`\sim`{=tex}3s \]

最坏预测 margin：

\[ `\boxed{
\rho_{ij}^{pred}(t)
=
\min_{\tau\in[0,H]}
\hat\rho_{ij}(t+\tau)
}`{=tex} \]

------------------------------------------------------------------------

# 18. CPA Predictor

定义：

\[ r=p_i-p_j,`\qquad `{=tex}v=v_i-v_j \]

\[ `\boxed{
t_{CPA}
=
-\frac{r^Tv}{v^Tv}
}`{=tex} \]

限制：

\[ t\_{CPA}`\in[0,H]`{=tex}\]

\[ d\_{CPA} = \|r+vt\_{CPA}\| \]

\[ `\boxed{
\rho_{CPA}
=
\frac{
d_{CPA}-d_{safe}(t+t_{CPA})
}{
d_{safe}(t+t_{CPA})
}
}`{=tex} \]

输出：

-   predicted minimum margin；
-   predicted minimum separation；
-   time to minimum margin。

可选 CA predictor 做小消融，但第一篇不需要 neural trajectory predictor。

------------------------------------------------------------------------

# 19. Time-Varying CBF

\[ `\boxed{
h_{ij}(x,t)
=
\|p_i-p_j\|^2-[d_{ij}^{safe}(t)]^2
}`{=tex} \]

因为安全边界变化：

\[ `\boxed{
\dot h_{ij}
=
2p_{ij}^{T}v_{ij}
-
2d_{ij}^{safe}\dot d_{ij}^{safe}
}`{=tex} \]

CBF 条件：

\[ `\boxed{
\dot h_{ij}+\alpha(h_{ij})\ge0
}`{=tex} \]

Safety Gate 求：

\[ `\boxed{
u^*
=
\arg\min_u
\|u-u_{MARL}\|^2
}`{=tex} \]

subject to：

-   CBF；
-   velocity limit；
-   acceleration limit；
-   PX4/control constraints。

目标：**minimally invasive safety filtering**。

------------------------------------------------------------------------

# 20. Proactive + Reactive Recovery

## Proactive

\[ `\boxed{
\rho_{ij}^{pred}<\rho_{pred}
}`{=tex} \]

预测未来冲突时提前调用 LLM。

## Reactive

\[ C_1:`\rho`{=tex}*{ij}\<`\rho`{=tex}*{warn} \]

\[ C_2:`\bar `{=tex}g\_{ij}\>g\_{th} \]

\[ C_3:N\_{int}(T)\>N\_{th} \]

触发：

\[ `\boxed{
C_1\land(C_2\lor C_3)
}`{=tex} \]

## Unified Trigger

\[ `\boxed{
\mathcal T_{ij}
=
\mathbf1
\left[
\rho_{ij}^{pred}<\rho_{pred}
\lor
\left(
\rho_{ij}<\rho_{warn}
\land
(
\bar g_{ij}>g_{th}
\lor
N_{int}>N_{th}
)
\right)
\right]
}`{=tex} \]

形成：

\[ `\boxed{
Predict\rightarrow Prevent\rightarrow Protect\rightarrow Recover
}`{=tex} \]

------------------------------------------------------------------------

# 21. Structured RiskEvent

Proactive：

``` json
{
  "event": "PREDICTED_CONFLICT",
  "agent_i": "uav_2",
  "agent_j": "uav_4",
  "current_margin": 0.72,
  "predicted_min_margin": -0.18,
  "time_to_min_margin": 1.43,
  "cause": "TRAJECTORY_CONFLICT",
  "severity": "HIGH"
}
```

Reactive：

``` json
{
  "event": "SAFETY_MARGIN_DEGRADATION",
  "agent_i": "uav_2",
  "agent_j": "uav_4",
  "margin": 0.21,
  "margin_degradation": 0.31,
  "intervention_count": 7,
  "cause": "COMMUNICATION_STALE",
  "severity": "HIGH"
}
```

------------------------------------------------------------------------

# 22. Cause-Aware Recovery

\[ M\_{total}=M\_{dyn}+M\_{perc}+M\_{comm} \]

\[ C_k=`\frac{M_k}{M_{total}}`{=tex} \]

用贡献比例判断风险主要来源。

例如：

``` text
Dynamics      18%
Perception    12%
Communication 70%
```

策略示例：

``` text
communication → HOLD / increase separation
trajectory conflict → REROUTE / YIELD
perception uncertainty → SLOW / WAIT
mission priority → REASSIGN / CHANGE_PRIORITY
```

------------------------------------------------------------------------

# 23. LLM Failure Injection

明确测试 LLM 不可靠：

-   malformed JSON；
-   hallucinated waypoint；
-   illegal action；
-   contradictory task；
-   unsafe priority；
-   repeated replanning；
-   oscillating decisions；
-   timeout。

所有输出经过：

``` text
LLM
↓
Schema Validator
↓
Action Whitelist
↓
Mission Planner
↓
MARL
↓
Runtime Assurance
```

即使 LLM recovery 错误，Safety Gate 仍保持最终否决权。

------------------------------------------------------------------------

# 24. 主消融矩阵

  Variant   Planner   Pilot   Safety   Margin     Prediction   Recovery
  --------- --------- ------- -------- ---------- ------------ ----------------------
  C1        Fixed     MARL    No       None       No           No
  C2        LLM       MARL    CBF      Fixed      No           No
  C3        LLM       MARL    CBF      Adaptive   No           No
  C4        LLM       MARL    CBF      Adaptive   No           Reactive
  C5        LLM       MARL    CBF      Adaptive   CV/CPA       Proactive + Reactive

不要把所有变量一次性做成巨大 full-factorial；重点问题单独做 focused
ablation。

------------------------------------------------------------------------

# 25. LLM Baseline

回答 reviewer 最可能的问题：

> Why LLM?

比较：

-   no recovery；
-   rule-based recovery；
-   deterministic/optimization planner；
-   local LLM recovery。

加入有 mission semantics 的任务，例如：

``` text
UAV1 = urgent medical delivery
UAV2 = routine inspection
```

检验 LLM 是否能基于任务优先级恢复，而不只是几何最短路。

------------------------------------------------------------------------

# 26. Policy-Quality Robustness

用：

\[
`\pi`{=tex}*{20%},`\pi`{=tex}*{50%},`\pi`{=tex}*{80%},`\pi`{=tex}*{100%}
\]

分别比较：

``` text
MARL only
vs
MARL + Runtime Assurance
```

核心问题：

> Runtime Assurance 是否降低系统对 nominal-policy quality 的依赖？

------------------------------------------------------------------------

# 27. 实验场景

## Geometry

-   head-on；
-   perpendicular crossing；
-   diagonal crossing；
-   dense intersection；
-   asymmetric velocity；
-   multi-UAV congestion。

## Scale

\[ N=2,4,8 \]

后续：

\[ N=16,32 \]

## Perception

-   GT；
-   noisy GT；
-   Ego；
-   stale Ego；
-   degraded/missing perception。

## Fault

-   packet loss；
-   telemetry stale；
-   command latency；
-   GCS loss；
-   offboard loss；
-   LLM timeout。

## OOD

训练只覆盖部分场景，测试：

-   unseen diagonal；
-   different speeds；
-   denser traffic；
-   communication degradation；
-   moving obstacle。

------------------------------------------------------------------------

# 28. Adversarial Scenario Search（可选高价值升级）

自动寻找最危险初始条件：

\[ `\boxed{
x^*
=
\arg\min_x\min_t\rho(x,t)
}`{=tex} \]

搜索：

-   start；
-   goal；
-   velocity；
-   latency；
-   packet loss；
-   perception noise。

方法可从 random/evolutionary search 开始，不需要复杂神经网络。

------------------------------------------------------------------------

# 29. Failure Envelope

构建：

\[ Packet Loss`\times `{=tex}Latency \]

以及：

\[ Perception Noise`\times `{=tex}AoI \]

等二维 robustness map。

目标不是只报告平均成功率，而是确定：

> 系统在什么条件边界以内仍保持安全和有效。

------------------------------------------------------------------------

# 30. Evaluation Metrics

## Safety

-   collision rate；
-   near-miss rate；
-   minimum pairwise separation；
-   safety violation duration；
-   minimum TTC。

## Runtime Assurance

-   intervention count/rate/duration；
-   (`\Delta `{=tex}u=\|u\_{safe}-u\_{nominal}\|)；
-   (`\rho`{=tex})；
-   (g)；
-   (d\_{safe})。

## Prediction

-   predicted minimum margin error；
-   CPA error；
-   conflict precision/recall；
-   false-positive rate；
-   early-warning lead time；
-   avoided CBF interventions。

## Mission

-   success；
-   completion time；
-   path length；
-   path efficiency；
-   energy proxy；
-   tracking error。

## LLM

-   TTFT；
-   total latency；
-   end-to-end recovery latency；
-   valid JSON rate；
-   legal action rate；
-   recovery success；
-   timeout；
-   replan count。

## Scalability

-   CBF/QP solve time；
-   prediction latency；
-   CPU/GPU utilization；
-   communication overhead。

------------------------------------------------------------------------

# 31. Statistics

MARL：

\[ `\boxed{5\ training\ seeds}`{=tex} \]

最终 evaluation：

-   固定 evaluation seed set；
-   重要条件尽量 20--50 episodes/condition；
-   报 mean + uncertainty interval；
-   training seed 与 evaluation seed 分开；
-   不挑最好看的 seed。

------------------------------------------------------------------------

# 32. 算力计划

基本不需要 GPU：

-   dynamic margin；
-   normalized margin；
-   degradation；
-   CV/CPA；
-   CBF/QP；
-   fault injection。

GPU 有帮助：

-   MAPPO training；
-   local Qwen inference；
-   Gazebo rendering。

工作流：

``` text
本地开发
↓
轻量 MARL 调试
↓
benchmark steps/sec 和 GPU utilization
↓
确认瓶颈后再决定是否短租 GPU
↓
保存 checkpoints
↓
PX4/Gazebo inference
↓
论文末期必要时租多核 CPU + GPU 批量实验
```

MARL 很可能同时受 CPU simulation throughput 限制，不应只追求昂贵 GPU。

------------------------------------------------------------------------

# 33. 关键论文图

1.  **Architecture Figure**：LLM → MARL → Runtime Assurance → PX4 +
    RiskEvent feedback。
2.  **Dynamic Margin Figure**：(d_0,M\_{dyn},M\_{perc},M\_{comm})。
3.  **Core
    Timeline**：实际距离、(d\_{safe})、(`\rho`{=tex})、(`\rho`{=tex}\^{pred})、AoI、CBF
    intervention、LLM replan。
4.  **C1--C5 Safety--Performance Tradeoff**。
5.  **Failure Envelope Heatmap**。
6.  **Weak→Strong MARL × Runtime Assurance**。

------------------------------------------------------------------------

# 34. Research Questions

**RQ1**：Adaptive safety margin 相比 fixed-margin CBF 是否改善
safety--efficiency tradeoff？

**RQ2**：Runtime Assurance 在 perception uncertainty 和 communication
staleness 下的鲁棒性如何？

**RQ3**：Normalized safety-margin degradation 能否作为 deterministic
low-level safety 与 semantic recovery 之间有效的跨层接口？

**RQ4**：Predictive safety margin 能否在 CBF hard intervention
前提供有效 proactive recovery？

**RQ5**：LLM recovery 相比 no recovery、rule-based recovery 和
deterministic planner 是否改善任务表现？

**RQ6**：系统硬安全对 MARL policy quality 和 LLM quality
的依赖程度是多少？

**RQ7**：随着 swarm density、通信故障和感知不确定性增加，系统 safe
operating envelope 在哪里？

------------------------------------------------------------------------

# 35. 预期论文贡献

## C1 --- Risk-Adaptive Runtime Assurance

基于 relative dynamics、perception uncertainty 和 AoI
构造动态多无人机安全边界。

## C2 --- Cross-Layer Safety-Margin Interface

利用 normalized margin 和 degradation 将连续低层安全状态转换成 semantic
risk event。

## C3 --- Predictive + Reactive Semantic Recovery

结合 CV/CPA short-horizon prediction 与
degradation/intervention-triggered recovery。

## C4 --- Untrusted-Autonomy Architecture

明确把 LLM 和 MARL 都视为 fallible component，由独立 deterministic
Runtime Assurance 保留最终安全权。

## C5 --- Systematic Validation

在 PX4/Gazebo 中系统测试 policy quality、perception
degradation、communication faults、LLM failures、OOD 与 swarm scale。

------------------------------------------------------------------------

# 36. 开发阶段

## Phase 0 --- Freeze Interfaces

先确定：

-   Observation schema；
-   MARL action schema；
-   RiskEvent schema；
-   RecoveryPlan schema；
-   Safety Gate API；
-   telemetry representation；
-   logging format。

------------------------------------------------------------------------

## Phase 1 --- Lightweight MARL

1.  2D fixed-altitude simulator；
2.  MAPPO；
3.  single UAV；
4.  head-on；
5.  crossing；
6.  randomized start/goal；
7.  4 UAV；
8.  intermediate checkpoints；
9.  multiple training seeds。

**完成标准：** 无 Safety Gate 时能稳定导航并具备基本避碰能力。

------------------------------------------------------------------------

## Phase 2 --- Risk-Adaptive Runtime Assurance

1.  closing speed；
2.  (M\_{dyn})；
3.  covariance margin；
4.  AoI；
5.  (M\_{comm})；
6.  (d\_{safe})；
7.  normalized (`\rho`{=tex})；
8.  degradation (g)；
9.  time-varying CBF/QP；
10. intervention logging。

**完成标准：** 能保护故意选择的 imperfect MARL policy。

------------------------------------------------------------------------

## Phase 3 --- Predictive Monitor

1.  CV；
2.  CPA；
3.  (`\rho`{=tex}\^{pred})；
4.  prediction error；
5.  horizon tuning；
6.  proactive trigger；
7.  可选 CA comparison。

**完成标准：** 有有效提前量且 false-positive 可接受。

------------------------------------------------------------------------

## Phase 4 --- Local LLM Recovery

1.  Qwen3-4B local；
2.  non-thinking；
3.  JSON schema；
4.  whitelist；
5.  timeout/fallback；
6.  proactive RiskEvent；
7.  reactive RiskEvent；
8.  recovery execution；
9.  unsafe/malformed output test。

**完成标准：** LLM 能完成 semantic recovery，且错误输出无法突破 Safety
Gate。

------------------------------------------------------------------------

## Phase 5 --- PX4/Gazebo

1.  telemetry → observation；
2.  MARL inference；
3.  velocity setpoint；
4.  Runtime Assurance；
5.  PX4 Offboard；
6.  head-on；
7.  crossing；
8.  multi-UAV。

**完成标准：** 完整 closed loop 工作。

------------------------------------------------------------------------

## Phase 6 --- Fault Injection

加入：

-   packet loss；
-   stale telemetry；
-   latency；
-   perception noise；
-   GCS/offboard loss；
-   LLM timeout；
-   invalid LLM command。

------------------------------------------------------------------------

## Phase 7 --- Main Experiments

1.  C1--C5；
2.  policy-quality robustness；
3.  LLM baseline；
4.  OOD；
5.  swarm scaling；
6.  failure envelope；
7.  statistics。

------------------------------------------------------------------------

## Phase 8 --- Optional Upgrades

核心论文完成后再考虑：

1.  chance-constrained / probability-calibrated safety boundary；
2.  adversarial scenario search；
3.  16--32 UAV；
4.  QLoRA；
5.  neural trajectory prediction（仅当 CV/CPA 明显不足）；
6.  hardware flight。

------------------------------------------------------------------------

# 37. Minimum Publishable Version

最低完整版本：

``` text
MAPPO nominal pilot
+
PX4/Gazebo multi-UAV
+
Dynamic safety boundary
+
Normalized safety margin
+
Safety-margin degradation
+
Time-varying CBF
+
CV/CPA predictive monitor
+
Proactive + Reactive recovery
+
Local LLM
+
Fault injection
+
C1–C5 ablation
+
Multiple seeds / statistics
```

------------------------------------------------------------------------

# 38. Stronger Version

进一步加入：

``` text
Cause-aware recovery
+
Rule/planner vs LLM
+
Weak/strong MARL robustness
+
LLM failure injection
+
OOD
+
Failure envelope
+
8–16 UAV scalability
+
Adversarial scenario search
```

------------------------------------------------------------------------

# 39. Scope Control

第一篇论文原则上不要继续堆：

-   VLM；
-   large multimodal model；
-   自创新 MARL；
-   neural trajectory predictor；
-   federated learning；
-   multi-agent LLM society；
-   与核心 RQ 无关的 digital twin 功能。

优先把现有主线做深，而不是做宽。

------------------------------------------------------------------------

# 40. 最终原则

项目不是试图让每一个 AI 模块永远正确，而是：

\[ `\boxed{\text{Assume AI can fail}}`{=tex} \]

并验证：

\[ `\boxed{
AI\ failure
\not\Rightarrow
Safety\ failure
}`{=tex} \]

最终科学目标是证明：

> **多无人机系统的 autonomy quality 与 hard runtime safety
> 可以在一定程度上解耦；同时通过预测安全裕度和语义恢复机制，在保持硬安全的前提下恢复任务性能。**
