# AegisAir 综合正式实验计划

**版本：** v2.0  
**日期：** 2026-08-20  
**来源：** `/Users/lijiajun/Documents/drone/AegisAir/plan.md` 与 `AEGISAIR_COMPLETE_EXPERIMENT_PLAN.md` 逐项合并  
**执行对象：** 2、4、8 架无人机；轻量仿真 + PX4 SITL/Gazebo  
**真实硬件：** 不在本版论文证据范围内

---

## 1. 综合后的最终立意

论文只保留三个主创新，原 `plan.md` 中的 C4/C5 改为支撑性系统属性和验证维度。

### C1：Risk-adaptive multi-source safety envelope

把以下风险统一映射为在线 pairwise separation requirement：

- relative closing dynamics；
- directional state-estimation uncertainty；
- communication staleness/AoI。

目标问题：

> 当前需要多大的安全间隔？

### C2：Execution-consistent sampled-data Runtime Assurance

将识别得到的 PX4 一阶速度响应通过 exact ZOH 离散化，直接进入下一采样时刻的 pairwise safety constraint。

目标问题：

> 在真实执行滞后存在时，下一控制周期的动作怎样满足安全约束？

### C3：Fail-closed semantic mission recovery separated from hard safety

将即时安全控制与异步任务级恢复完全分离：

```text
imminent risk → RA/CBF
persistent conflict/task change → RiskEvent → rule/LLM planner → validator → fallback → RA veto
```

目标问题：

> 安全过滤避免碰撞后，怎样恢复任务而不把 safety authority 交给 LLM？

统一叙事：

> **Protect now, understand later, replan the future.**

---

## 2. 与原 `plan.md` 的合并决策

| 原 `plan.md` 内容 | 综合后处理 | 原因 |
|---|---|---|
| Dynamic safety boundary | 保留为 C1 | 是核心安全表示 |
| Normalized margin `rho` | 保留 | 连接安全状态与事件 |
| Degradation `g` | 保留为跨层接口 | 触发持续冲突/任务事件，不直接替代RA |
| CV/CPA predictor | 保留为 pre-alert | 只提示，不拥有 safety authority |
| Acceleration-aware HOCBF | 降为基线 | 不是最终PX4执行模型 |
| `a_nom = k_v(v_nom-v_actual)` 启发式转换 | 不再作为正式方法 | 被 exact-ZOH execution model 替代 |
| Time-varying CBF/QP | 保留，但实现为 sampled-data projected barrier | 与PX4控制周期一致 |
| Proactive recovery | 保留，但必须经过 runtime confirmation | 单次预测风险不直接调用LLM |
| Reactive recovery | 保留 | 用于持续冲突、任务失效和死锁 |
| LLM/rule/deterministic planner baseline | 全部保留 | 回答“为什么需要LLM” |
| Policy-quality robustness | 加入正式实验 | 验证RA是否降低对MARL质量的依赖 |
| OOD scenarios | 加入正式实验 | 验证泛化边界 |
| Failure envelope | 加入正式实验 | 报告安全运行边界，不只报平均值 |
| Adversarial scenario search | 作为增强实验 | 不改变控制器，只寻找最危险状态 |
| 16–32 UAV | 不列主论文必做 | 无实机、工程成本高；最多作为附录/后续工作 |
| 5 training seeds | 保留为MAPPO训练最低要求 | training seed与final evaluation seed分离 |
| 20–50 episodes/condition | 升级为最低正式统计要求 | 覆盖原方案的最小统计强度 |
| 轻量仿真训练 | 保留 | 不在Gazebo做百万级训练 |
| PX4/Gazebo最终验证 | 保留 | 用于执行链和故障闭环证据 |

---

## 3. 最终系统架构

```text
Mission / User
      ↓
Semantic Mission Manager
      ↓ goals / priorities / constraints
MARL Nominal Pilot
      ↓ u_nom
SharedStateEstimator
      ↓ estimated state / covariance / AoI
Risk-Adaptive Runtime Assurance
      ├── dynamic safety envelope
      ├── normalized margin rho
      ├── degradation g
      ├── CV/CPA pre-alert
      └── exact-ZOH sampled-data projected-barrier QP
      ↓ u_safe
PX4 Offboard → PX4 SITL/Gazebo
```

异步任务链：

```text
RiskEvent / MissionValidityEvent
      ↓
RuleMissionPlanner or local LLM
      ↓
syntax validation
      ↓
schema validation
      ↓
semantic validation
      ↓
deterministic RecoveryPlan expansion
      ↓
PX4 adapter acceptance
      ↓
RA final veto
```

硬边界：

- LLM不能修改 `d0`、CBF约束、confidence factor、Safety threshold或actuator limit；
- 单次 imminent collision不调用LLM；
- LLM不进入实时安全回路；
- prediction只做pre-alert；
- validator、fallback、adapter和RA采用fail-closed；
- MARL和LLM均视为untrusted components。

---

## 4. 证据层级

### 4.1 轻量仿真

用于：

- MAPPO训练；
- 2/4/8机大样本统计；
- 全部baseline；
- 全部C1/C2消融；
- policy-quality robustness；
- OOD；
- failure envelope；
- recovery和LLM对比；
- adversarial scenario search。

### 4.2 PX4 SITL/Gazebo

用于：

- 完整Telemetry→Estimator→MARL→RA→Offboard闭环；
- PX4一阶执行模型识别与held-out验证；
- 2机机制和intersample验证；
- 4机主要高保真论文证据；
- 8机少量扩展压力确认；
- command expiration、dropout、delay、LLM failure和adapter gate。

### 4.3 真实硬件

当前不做。所有论文用语必须明确为：

> PX4 SITL/Gazebo closed-loop evaluation

不得写成real flight、hardware validation或physical flight evidence。

---

## 5. 训练与模型协议

### 5.1 MAPPO nominal pilot

保留原 `plan.md` 的CTDE训练方式：

- centralized training；
- decentralized execution；
- 相对状态优先；
- 二维固定高度优先；
- action为 `[vx_cmd, vy_cmd]`；
- 不直接输出thrust、attitude、torque或motor command；
- 轻量仿真训练；
- PX4/Gazebo只做最终验证。

训练至少使用5个独立training seeds。不得把training seed当final-test seed。

### 5.2 MARL质量分层

构造政策质量分层：

- `pi*20%`；
- `pi*50%`；
- `pi*80%`；
- `pi*100%`。

每个质量层比较：

- MARL only；
- MARL + Runtime Assurance。

目标是验证：

> Runtime Assurance是否降低系统对nominal-policy quality的依赖。

不得通过final test重新选择“最好看的政策质量层”。

### 5.3 LLM

主模型冻结为本地Qwen3-4B-4bit、non-thinking、structured output。

至少比较：

- no recovery；
- deterministic rule recovery；
- deterministic/optimization planner；
- local LLM recovery。

Qwen3-8B、Phi-4-mini和QLoRA不进入主实验，除非预先冻结为模型消融；不得因为主模型结果不理想临时增加模型。

---

## 6. 统一实验条件矩阵

### 6.1 Safety条件

- S0：No safety filter；
- S1：Static-distance velocity CBF-QP；
- S2：Acceleration HOCBF-QP；
- S3：Sampled-data/ZOCBF-style adapted baseline；
- S4：Risk-adaptive envelope + ideal instantaneous execution；
- S5：Risk-adaptive envelope + 早期启发式执行近似；
- S6：完整exact-ZOH execution-consistent RA；
- S7：S6 + frozen robust-`tau` one-step analysis；
- S8：GCBF+或其他强learned-CBF baseline，在能公平适配的场景使用。

HOCBF是baseline，不是最终PX4正式执行器。S6是AegisAir正式方法。

### 6.2 Recovery条件

- R0：No recovery；
- R1：RuleMissionPlanner；
- R2：LLM + validator + deterministic fallback；
- R3：LLM timeout；
- R4：malformed JSON/schema error；
- R5：schema-valid but semantic-invalid；
- R6：stale/expired RecoveryPlan；
- R7：unsafe but schema-valid mission proposal；
- R8：forced deterministic fallback；
- R9：RA final veto。

### 6.3 主因果矩阵

```text
M0: N1 + S0 + R0                 nominal autonomy
M1: N1 + S1 + R0                 static CBF
M2: N1 + S2 + R0                 HOCBF
M3: N1 + S3 + R0                 sampled-data reference
M4: N1 + S6 + R0                 AegisAir RA
M5: N1 + S6 + R1                 RA + rule recovery
M6: N1 + S6 + R2                 RA + LLM recovery
```

N0 rule pilot、N2强MARL和S8 GCBF+用于外部稳健性，不替换N1主因果矩阵。

---

## 7. 场景总表

### 7.1 几何场景

- head-on；
- perpendicular crossing；
- diagonal crossing；
- near-crossing；
- overtaking；
- converging merge；
- dense intersection；
- position swap；
- asymmetric velocity；
- mixed-speed congestion；
- narrow/bidirectional corridor；
- formation crossing；
- moving obstacle（仅在所有baseline能公平支持时）。

### 7.2 规模

- 2机：机制、公式和最危险几何；
- 4机：主论文高保真证据；
- 8机：密度和计算压力；
- 16/32机：仅轻量仿真增强，不列无硬件版本主论文必做。

### 7.3 感知与通信

- ground truth；
- noisy ground truth；
- covariance increase；
- stale telemetry；
- burst dropout；
- packet loss；
- communication delay；
- command latency；
- GCS/offboard loss；
- degraded/missing perception；
- perception noise × AoI；
- packet loss × latency。

### 7.4 任务和故障

- single-drone failure；
- blocked corridor；
- priority conflict；
- repeated conflict/deadlock；
- prolonged waiting；
- route deviation；
- task cancellation/addition；
- priority change；
- stale/expired plan；
- LLM timeout；
- malformed/invalid/semantic-invalid output；
- invalid agent ID、越界waypoint、不安全高度；
- planner/validator process crash。

### 7.5 OOD

- unseen diagonal geometry；
- different speed range；
- denser traffic；
- unseen delay/dropout combination；
- unseen covariance level；
- unseen task priority arrangement；
- arena boundary stress；
- moving obstacle（条件允许时）。

---

## 8. C1实验：多源风险自适应安全包络

### 8.1 消融

- fixed `d0`；
- `d0 + M_dyn`；
- `d0 + M_perc`；
- `d0 + M_comm`；
- dynamics + perception；
- dynamics + communication；
- perception + communication；
- full three-source envelope；
- stale propagation only；
- AoI margin only；
- stale propagation + AoI margin。

### 8.2 扫描

- closing speed；
- reaction/control delay；
- effective deceleration；
- covariance magnitude；
- covariance anisotropy；
- telemetry age；
- dropout burst length；
- maximum velocity/acceleration；
- agent density；
- approach angle。

### 8.3 必须回答的问题

1. 动态边界是否优于固定距离；
2. 每个margin是否在对应故障中产生正确作用；
3. stale传播与AoI是否重复补偿；
4. 安全收益是否只是因为停住；
5. 感知与通信不确定性是否只是工程buffer而非概率保证。

### 8.4 Failure envelope

至少构造：

- packet loss × latency；
- perception noise × AoI；
- closing speed × effective deceleration；
- density × communication degradation。

输出安全运行区间、边界违规区间和任务失效区间，不只报告均值。

---

## 9. C2实验：execution-consistent sampled-data RA

### 9.1 PX4执行响应识别

至少完成：

- 0.5、1.0、1.5 m/s阶跃；
- x/y正负方向；
- 如果论文保留三维，再加入z轴上升/下降；
- RUN/STOP分别拟合；
- 每个方向和幅值至少3次；
- identification与held-out validation分离；
- 异常振荡不删除；
- 报告拟合误差、残差、`tau`分布和held-out coverage；
- 比较one-step与multi-step预测误差。

经验区间只能称为empirical coverage，不得称为global deterministic bound。

### 9.2 执行模型消融

- instantaneous execution；
- 早期启发式位移近似；
- nominal `tau_hat` exact ZOH；
- empirical `tau` vertex robustification；
- dense `tau` grid audit；
- 故意过快模型错配；
- 故意过慢模型错配。

### 9.3 Endpoint/intersample审计

每个控制周期内部至少100个时间点，使用精确一阶ZOH解重建：

- `p_i(t)`、`v_i(t)`；
- `r_ij(t)`；
- closing speed；
- `d_safe(t)`；
- projected barrier；
- squared-distance barrier；
- `rho(t)`。

必须分别报告：

- endpoint violation；
- intersample violation；
- endpoint minimum margin；
- intersample minimum margin；
- 仅在周期内部发生的违规；
- 最危险时刻、pair、状态和动作。

如果只完成dense audit而没有解析证明，论文只能声称：

> 在测试范围和审计分辨率下未观察到intersample violation。

### 9.4 QP和fallback

扫描：

- initial margin；
- closing speed；
- density；
- velocity/acceleration limit；
- execution `tau`；
- communication age。

记录：

- feasible/infeasible；
- solve time；
- active constraints；
- hard-brake触发；
- fallback后安全和任务结果。

feasible-QP theorem claim与fallback结果分开。

### 9.5 robust-`tau`关闭规则

保留：

- nominal-safe/robust-unsafe activation search；
- 4 vertices per pair；
- dense `tau` grid；
- projected和exact squared barrier audit；
- feasible/infeasible分组；
- `Delta u`统计。

不再添加guard、rollback、planning state或recovery corridor来制造episode-level收益。episode-level null result按原样报告。

---

## 10. Predictor与跨层触发实验

### 10.1 Predictor

冻结CV/CPA方法、horizon和threshold后，使用独立test seeds测量：

- precision；
- recall；
- FPR；
- AUROC/AUPRC；
- CPA error；
- TTSB MAE、median和tail quantiles；
- early-warning lead time；
- heavy-tail error。

Predictor只能pre-alert，不直接修改hard-safety action。

### 10.2 Proactive与reactive触发

比较：

- no prediction；
- prediction pre-alert only；
- runtime confirmation后触发recovery；
- degradation/intervention-triggered reactive recovery；
- proactive + reactive。

触发必须满足：

- 单次imminent collision仍由RA处理；
- 持续低`rho`、持续正`g`、重复CBF干预或任务变化才产生RiskEvent；
- RiskEvent进入rule/LLM任务层前必须经过事件去重和cooldown。

### 10.3 RQ3/RQ4

验证：

- `rho`和`g`是否能作为低层安全到任务层的跨层接口；
- prediction是否减少无效replan；
- proactive+reactive是否优于reactive only；
- prediction误报是否增加任务抖动。

---

## 11. C3实验：任务恢复与LLM安全边界

### 11.1 Recovery对比

在同一RA下比较：

- no recovery；
- rule-based recovery；
- deterministic/optimization planner；
- local LLM recovery。

任务中加入mission semantics，例如：

```text
UAV1 = urgent medical delivery
UAV2 = routine inspection
```

验证planner是否理解优先级，而不是只做几何最短路。

### 11.2 Cause-aware recovery

按RiskEvent原因分别测试：

- route deviation → reroute；
- drone failure → reassign；
- blocked corridor → yield/waypoint；
- repeated conflict → coordination change；
- priority conflict → change priority；
- mission invalidity → abort/return。

不允许用一个无原因的统一replan掩盖不同故障。

### 11.3 LLM failure injection

全部测试：

- timeout；
- empty output；
- malformed JSON；
- schema type error；
- unknown action；
- missing field；
- out-of-arena waypoint；
- unsafe altitude；
- invalid agent ID；
- stale plan；
- contradictory priorities；
- semantic-invalid plan；
- unsafe schema-valid proposal；
- duplicate plan；
- planner crash；
- validator exception；
- delayed plan arriving after state change；
- RA final veto。

### 11.4 C3主要指标

- overall completion；
- critical-task completion；
- completion time；
- path length/efficiency；
- waiting/deadlock duration；
- repeated CBF intervention duration；
- replan count；
- valid JSON rate；
- legal action rate；
- validator rejection category；
- fallback rate；
- LLM latency P50/P95/P99；
- safety bypass count。

只有LLM显著优于rule时，才声称LLM额外有效；否则保留fail-closed architecture贡献。

---

## 12. 2/4/8机与Gazebo完整安排

### 12.1 2机

轻量仿真：全部几何、通信、执行和intersample扫描。  
Gazebo：head-on、crossing、near-crossing、high closing speed、delay、dropout、execution mismatch、QP boundary、fallback，每个关键条件至少10次。

### 12.2 4机

轻量仿真：全部主矩阵、消融、OOD、failure envelope、recovery和统计。  
Gazebo：center crossing、position swap、randomized_4、dense randomized_4、stale、dropout、mixed delay、drone failure、blocked corridor、priority conflict、LLM timeout、invalid output、expired command和multi-fault，每个核心cell至少20个paired seeds。

### 12.3 8机

轻量仿真：全部scale、density、OOD和policy-quality robustness。  
Gazebo：randomized_8、dense randomized_8、mixed delay、deadline/computation stress，核心cell至少10个paired seeds。

8机Gazebo不需要复制全部消融和全部LLM故障排列。若无法稳定运行，保留失败日志，并把8机结论限制为轻量仿真。

### 12.4 16/32机

只作为高价值增强项：

- 轻量仿真scale curve；
- QP constraint/latency curve；
- failure envelope；
- 不列入无实机版本的主论文必要条件；
- 不外推为GCBF+级别的大规模分布式保证。

---

## 13. Seed、样本和统计

### 13.1 Seed划分

- MAPPO training：至少5个training seeds；
- calibration：独立，不进入最终统计；
- validation：独立，用于代码和协议检查；
- final test：独立且只运行一次；
- 所有方法使用paired final seeds和相同扰动流。

### 13.2 最低样本量

轻量仿真：

- smoke：每cell 5；
- calibration：每cell 20；
- validation：每cell 30；
- final core：每cell 100 paired seeds；
- rare-event stress：每cell 200 paired seeds；
- recovery：每场景每主要条件至少50 paired seeds。

Gazebo：

- 2机典型场景：每关键条件10次；
- 4机核心cell：20 paired seeds；
- 4机故障cell：20 paired seeds；
- 8机压力cell：10 paired seeds；
- deterministic adapter/validator gate：每类故障至少30次注入。

### 13.3 统计

- collision/completion/violation：paired difference、bootstrap 95% CI、McNemar exact或合适配对检验；
- 连续指标：median、IQR、mean、paired bootstrap CI、effect size；
- 尾部：worst 10%/20%或CVaR20；
- 主比较使用Holm correction；
- exploratory结果明确标记；
- 不以单个p值替代工程意义。

---

## 14. 统一指标

### Safety

- collision rate；
- near-miss rate；
- minimum pairwise separation；
- boundary violation rate/duration；
- minimum `rho`；
- minimum TTC/CPA；
- endpoint/intersample violation。

### Runtime Assurance

- intervention count/rate/duration；
- `Delta u`；
- `rho`；
- `g`；
- `d_safe`；
- QP feasible/infeasible；
- fallback；
- active constraints。

### Mission

- overall success；
- critical-task completion；
- completion time；
- path length/efficiency；
- energy proxy；
- tracking error；
- deadlock/waiting duration；
- repeated-intervention duration。

### Prediction

- precision/recall/FPR；
- AUPRC；
- CPA/TTSB error；
- early-warning lead time；
- error tail。

### LLM/recovery

- valid JSON；
- legal action；
- replan count；
- recovery latency；
- fallback；
- rejection categories；
- bypass；
- RA veto；
- LLM versus rule effect。

### Scalability

- pair count；
- QP constraints；
- QP latency P50/P95/P99/max；
- prediction latency；
- CPU/GPU/memory；
- communication overhead；
- deadline miss。

---

## 15. 日志和复现

每个episode保存：

- protocol/version/commit/config/checkpoint hash；
- seed、场景、条件、规模；
- raw telemetry source/receive timestamps；
- AoI、covariance；
- nominal/safe action；
- QP status、solve time、active constraints；
- pairwise distance、`d_safe`、`rho`、barrier；
- intersample reconstruction data；
- intervention、fallback；
- RiskEvent；
- planner input/output；
- validator/final veto；
- PX4 adapter acceptance/rejection；
- collision/completion；
- process/infrastructure failure。

必须提供：

- frozen config；
- seed manifest；
- checkpoint hash；
- raw event logs；
- summary scripts；
- failure episode list；
- environment versions；
- one-command reproduction instructions。

---

## 16. Go/No-Go规则

### Gate A：接口与审计

通过：schema一致、日志完整、paired replay可复现、adapter bypass=0。  
不通过：停止正式实验并发布新协议版本。

### Gate B：C1

通过：完整三源包络在对应故障场景中相对固定边界和关键消融改善主要安全指标，且不是单纯停住。  
不通过：C1降级为工程表示，不声称性能优势。

### Gate C：C2

通过：exact-ZOH RA在执行滞后/错配下优于理想或传统执行基线，且QP满足控制周期。  
不通过：C2降级为实现细节，保留负结果。

### Gate D：intersample

通过：dense audit与claim一致。  
不通过：禁止连续时间安全表述，限制为endpoint或测试范围。

### Gate E：C3

通过：recovery相对RA-only改善任务指标、无safety bypass、故障输出全部fail-closed。  
不通过：C3只保留架构和安全边界，不声称任务收益。

### Gate F：投稿资格

至少两个主贡献通过对应gate，所有负结果、失败案例和SITL限制原样保留；否则收紧论文范围，不通过增加模块救结果。

---

## 17. 执行顺序

1. 逐公式核对ZOCBF、Robust sampled-data CBF、Predictive CBF和GCBF+；
2. 冻结接口、场景、seed、指标和阈值；
3. 完成全部baseline接口；
4. 运行smoke test；
5. MAPPO多seed训练和policy-quality分层；
6. C1消融validation；
7. PX4执行模型识别和held-out validation；
8. endpoint/intersample validation；
9. Predictor与RiskEvent validation；
10. recovery、rule、LLM和fail-closed validation；
11. 冻结final protocol；
12. 轻量仿真final core；
13. 轻量仿真stress、OOD、failure envelope和adversarial search；
14. 2机Gazebo机制验证；
15. 4机Gazebo主矩阵；
16. 8机Gazebo扩展压力；
17. 统一统计脚本；
18. 失败、缺失和日志审计；
19. 生成主文图、表和补充材料；
20. 按Go/No-Go规则冻结最终claims；
21. 写作和投稿，不再追加方法模块。

---

## 18. 最终论文结构

### 主文主线

1. Problem and threat model；
2. Risk-adaptive multi-source envelope；
3. Exact-ZOH sampled-data RA；
4. Predictor/pre-alert与RiskEvent；
5. Fail-closed semantic recovery；
6. Baselines and experimental protocol；
7. 2/4/8机轻量统计；
8. 2/4机PX4/Gazebo闭环；
9. C1/C2/C3消融；
10. OOD/failure envelope/policy robustness；
11. Limitations and claim boundary。

### 原 `plan.md` 的RQ保留方式

- RQ1：adaptive margin vs fixed margin；
- RQ2：perception uncertainty和communication staleness；
- RQ3：`rho/g`是否是有效跨层接口；
- RQ4：prediction是否提供有效pre-alert；
- RQ5：LLM/rule/no recovery的任务差异；
- RQ6：RA对MARL policy quality的依赖；
- RQ7：安全运行边界在哪里。

这些是研究问题，不是额外主创新。C4 Untrusted-Autonomy Architecture和C5 Systematic Validation作为系统属性与验证方法写入，不单独占贡献编号。

---

## 19. 最终原则

项目不是假设每个AI组件永远正确，而是验证：

```text
AI failure does not automatically imply safety failure.
```

但实验结论必须严格限制在：

- 给定模型与参数范围；
- 给定2/4/8机规模；
- 给定轻量仿真和PX4 SITL/Gazebo环境；
- 给定seed、扰动、阈值和QP可行条件；
- 给定endpoint/intersample审计范围。

如果结果不支持某项主张，就收紧主张，不增加新模块、不放松阈值、不删除异常trial、不用robust-`tau`或LLM结果挽救失败假设。
