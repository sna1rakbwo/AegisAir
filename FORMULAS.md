# AegisAir 重要公式总结

> 与代码一一对应；参数默认值见末尾 §8。

## 1. 安全边界与裕度（swarm/ra/margins.py, margin.py）

### 1.1 接近速度 closing speed

```text
v_cl = max(0, - (p_i - p_j)·(v_i - v_j) / ||p_i - p_j||)
```

### 1.2 动态安全边界 d_safe

```text
d_safe = d0 + M_dyn + M_perc + M_comm
```

三个分量：

```text
M_dyn  = v_cl·(τ_r + τ_ctrl) + v_cl² / (2·a_eff)     （动力学）
M_perc = β·sqrt(σ_i² + σ_j²)                          （感知）
M_comm = v_max·AoI + 0.5·a_max·AoI²                   （通信）
```

协方差感知：`σ_i = sqrt(n^T P_i n)`，`n` 为两机连线方向，`P_i` 为 estimator 的
2×2 协方差；无协方差时回退到固定 `perception_sigma`。

### 1.3 归一化安全裕度 ρ

```text
ρ = (d - d_safe) / d_safe
```

### 1.4 安全裕度退化 g

```text
g_k = (ρ_{k-1} - ρ_k) / Δt
g = EMA(g_k) = λ·g + (1-λ)·g_k
```

## 2. 安全滤波（swarm/ra/cbf.py, hocbf.py）

### 2.1 velocity CBF（单积分，h = ||r||² - d_safe²）

```text
h = ||p_i - p_j||² - d_safe²
2(p_i - p_j)·u_i  >=  2(p_i - p_j)·v_j  -  α·h
```

`u_i` 为被安全化的速度命令，`u_safe = argmin ||u - u_nom||` s.t. 所有 pair 约束。

### 2.2 HOCBF（加速度，双积分 ṗ=v, v̇=a）

```text
ψ₂ = ḧ + (k1 + k2)·ḣ + k1·k2·h  >=  0
```

在同一 QP 步内取 `ṡ = s̈ = 0`，化为相对加速度线性约束：

```text
2 r^T (a_i - a_j)  >=
    -2||v||² - 2(k1+k2)·r^T v - k1·k2·(||r||² - s²)
```

其中 `r = p_i - p_j`、`v = v_i - v_j`、`s = d_safe`。

### 2.3 robust sampled-data projected barrier（最终冻结）

使用**投影 barrier**（对 acceleration 精确线性、对 τ 仿射）：

```text
n = (p_i - p_j) / ||p_i - p_j||            （固定 separation 方向）
h_underline = n^T r - D_safe                （保守：n^T r ≤ ||r||）

r_next = r + v·Δt + Δt·(β_i a_i - β_j a_j)   （精确离散，见 §3.1）

constraint:
  inf_{τ_i∈T_i, τ_j∈T_j} [ n^T r_next - D_next ]  >=  (1 - γ)·h_underline
```

因为对 `a` 线性、对 `β` affine，`(β_i, β_j)` rectangle 的 **4 个顶点**全部进
QP。`D_next` 同时取四个 τ 顶点对应的 `d_safe(v_next)` 最大值，避免只对位置
系数 robustify、却仍用 nominal execution 计算下一拍 closing-speed boundary。
因此当前实现对 projected barrier 使用每 pair 4 条约束，4 机 6 pair = 24 条。
QP 输出 `a_safe`；不可行时硬刹车 `a_safe = clip(-v, -a_max, a_max)`。
速度命令：`v_safe = clip(v + a_safe·Δt, -v_max, v_max)`。

## 3. PX4 执行动力学建模（Phase 6 修复）

### 3.1 PX4 一阶速度跟踪（精确离散，ZOH）

```text
α = 1 - exp(-Δt / τ)
β = Δt - τ·α

v_next = (1 - α)·v + α·u
p_next = p + v·Δt + β·(u - v)

u = v + a·Δt   ⟹   p_next = p + v·Δt + β·a·Δt
```

`τ = 0` 时 `α=1, β=Δt`，退化为离散 velocity-command 模型
`p_next = p + v·Δt + a·Δt²`（不再是启发式 `0.5·α·a·Δt²`）。

### 3.2 AoI 状态传播（dead-reckoning）

```text
p̂(t_now) = p(t_stamp) + v(t_stamp)·(t_now - t_stamp)
```

传播后的状态进 barrier；`M_comm` 仍保留 AoI 的 uncertainty margin。

## 4. SharedStateEstimator（swarm/estimation.py）

### 4.1 协方差增长

```text
P = min( measurement_cov + process_noise·Δt_rx,  max_cov )
Δt_rx = (t_now - t_last_receive)
```

- 正常更新：`Δt_rx ≈ 0`，`P ≈ measurement_cov`；
- dropout：hold 上次估计，`Δt_rx` 增长，`P` 增大；
- delay：输出更旧状态，`AoI` 增大（进 `M_comm`）。

## 5. CV/CPA 预测监视器（swarm/ra/predictor.py）

### 5.1 CPA

```text
t_cpa = - r·v / ||v||²    （clamp 到 [0, horizon]）
d_cpa = ||r + v·t_cpa||
```

### 5.2 预测最小裕度

```text
ρ̂(τ) = (d̂(τ) - d_safe(τ)) / d_safe(τ)
ρ̂_min = min_{τ∈[0,H]} ρ̂(τ),   τ* = argmin
```

未来协方差与 AoI：

```text
σ²(τ) = σ² + q_pred·τ
AoI(τ) = AoI + τ
```

`TTSB` = 首次 `ρ̂(τ) ≤ 0` 的时刻。

## 6. MAPPO nominal pilot（marllib/policies/mappo.py）

### 6.1 共享高斯 actor

```text
mean = speed_limit · tanh(net(obs))
std  = exp(log_std) ∈ [1e-3, 1.0]
a ~ N(mean, std),  clip 到 [-speed_limit, speed_limit]
```

### 6.2 PPO clipped surrogate（训练）

```text
ratio = π_new(a|s) / π_old(a|s)
L_policy = -E[ min(ratio·A, clip(ratio, 1-ε, 1+ε)·A) ]
```

### 6.3 观测（相对、共享）

```text
obs_i = [v_i, goal_i - p_i, { (p_j - p_i, v_j - v_i, 1) }_j≠i 按距离排序, 0-pad]
dim = 4 + 5·max_neighbors
```

## 7. 恢复 / 协调层几何

### 7.1 safe holding point

```text
centroid = mean(p_j | j≠i)
holding  = p_i + normalize(p_i - centroid)·offset     （clip 到 arena）
```

### 7.2 deadlock 横向让路点

```text
u = normalize(goal - p)
n̂ = perpendicular(u)
waypoint = p + sign(drone_id)·2.0·n̂     （clip 到 arena）
```

### 7.3 corridor 绕行点

```text
corner = 距 goal 最近的障碍区角点
waypoint = corner + sign·2.0·(corner - center)        （clip 到 arena）
```

## 8. 默认参数（RuntimeAssuranceParams）

```text
d0=0.5, tau_r=0.3, tau_ctrl=0.2(live), a_eff=2.0,
beta=3.0, v_max=2.0, a_max=3.0, alpha=1.0,
ema_lambda=0.8, degradation_dt=0.1, prediction_horizon=1.5,
rho_pred_threshold=0.0, q_pred=0.0, estimated_recovery_latency=0.5
```

Phase 5/6 冻结控制：`sampled-data, gamma=0.1`；`tau_px4 ∈ [tau_min, tau_max]`
（nominal `tau_px4=0.2`，robust QP 用区间 `[tau_px4_min, tau_px4_max]`）。
