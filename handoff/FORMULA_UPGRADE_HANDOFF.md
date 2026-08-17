# 公式升级（C2 exact model + robust τ QP）交接文档

> 日期：2026-08-17
> 范围：把 C2 从“PX4-lag 启发式修正”升级为“execution-consistent robust QP”。
> 依据：用户粘贴的 formal-property 方案（attachments/pasted-text.txt 要点）。

## 1. 一句话状态

**Exact PX4 离散模型 + τ 区间 robust QP 已实现并通过单测；Step A（execution
identification）只跑了一次粗略 step response，还没做多 seed/多速度辨识与
held-out validation。** C1 covariance 与 Proposition 暂缓（用户明确要求）。

## 2. 已完成（本阶段）

1. **Exact PX4 离散模型**（替换启发式 `0.5·α·a·Δt²`）
   - `swarm/ra/hocbf.py:beta_of_tau(dt, tau)`
   - `v_{k+1} = (1-α)v_k + α·u_k`，`p_{k+1} = p_k + v_k·Δt + β·(u_k-v_k)`
   - `u = v + a·Δt ⟹ p_{k+1} = p_k + v_k·Δt + β·a_k·Δt`
   - `α = 1-exp(-Δt/τ)`，`β = Δt - τ·α`

2. **Robust τ 区间 QP**（`swarm/ra/hocbf.py:solve_robust_sampled_data_qp`）
   - 投影 barrier：`h_underline = n^T r - D`（对 `a` 精确线性、对 `β` affine）
   - 每 pair 把 `(β_i, β_j)` 的 4 个 rectangle 顶点全部进 QP
   - 约束：`inf_{τ∈T} [n^T r_{k+1} - D_{k+1}] ≥ (1-γ)·h_underline`
   - 不可行时硬刹车回退 `a_safe = clip(-v, -a_max, a_max)`

3. **RA/CLI 接线**
   - `RuntimeAssurance` 新增 `tau_px4_min/tau_px4_max`
   - `phase5_runner.py` 新增 `--tau-px4-min` / `--tau-px4-max`
   - `_filter_sampled_data` 已切到 robust QP（`s_now/s_next` 仍用 nominal
     `alpha` 预测 velocity 算 `d_safe`）

4. 测试：119 全绿（`beta_of_tau`、robust QP feasible/硬刹车 测试）。

5. `FORMULAS.md` §2.3 / §3.1 已更新为精确公式。

## 3. Step A 已做的一小步 + 发现

### 3.1 已采集数据

- 一次单机 PX4 速度阶跃（`vx=1.5`）：
  `/Volumes/Expansion/aegisair_phase6_20260816/step_response/step_v1.jsonl`
- 入口：`marllib/phase5_step_response.py --drone 2 --vx 1.5 ...`

### 3.2 初步读数（重要，但不是最终 τ）

- 上升阶段 10%→90% 用时约 **1.0 s**（目标 1.5 m/s）。
- 按一阶近似 `τ ≈ rise / ln9 ≈ 0.45 s`。
- 但响应有 **overshoot / 振荡**（run 阶段末 vx 又掉到 -0.7，stop 阶段
  vx 最低 -3.8 m/s），**不是干净的一阶系统**。
- 线性化最小二乘拟合给出 `τ≈1.56 s`，与 rise-time 估计不一致，说明单 τ
  一阶模型对真实 PX4 只是近似。

**结论**：Step A 不能只用一个 τ；需要多速度、多 seed 的阶跃，得到一个
empirical execution envelope（并额外加 margin），再用 held-out trials 验证
coverage。是否升为二阶/带 overshoot 模型，由更多数据决定。

## 4. 未完成（下一步）

### 4.1 Step A 完整 identification

1. 跑多组 `phase5_step_response.py`：
   - `--vx`：0.5 / 1.0 / 1.5（覆盖 v_max 附近）
   - 每组多 seed（3–5 次）
2. 从 RUN（上升）与 STOP（减速）分别拟合 τ，得到
   `T = [τ_min, τ_max]`（建议：观测区间再加 margin，不是直接用 [P5,P95] 当
   deterministic bound）。
3. Held-out validation：留几组不参与拟合，验证真实 step response 是否被
   `T` 覆盖。

### 4.2 用 T 验证 robust QP

1. 先在 lightweight sim（`phase6_residual_check.py`）用
   `--tau-px4-min/max` 跑 MAPPO + robust QP，看 `min_rho`；
2. 再 4 机 live 复测（`--tau-px4-min/max`），对比 nominal vs robust；
3. 最后故意用 `τ > τ_max` 做 out-of-envelope，观察 guarantee breakdown
   （这组实验直接验证 theorem assumption）。

## 5. 已明确暂缓（用户要求，不要现在做）

- C1 perception covariance formalization（`β_δ = sqrt(χ²_{d,1-δ})`）
- C1 AoI reachable bound（`M_stale = 0.5·a_max·A²`，去掉 `v_max·A` 防 double
  count）
- 拆分 `τ_r = τ_sense + τ_compute + τ_network`（把 `τ_PX4` 从 `M_dyn` 里拆走）
- Proposition 1（robust sampled-step safety + proof）
- intersample reachable bound（ZOH interval 内连续安全）

## 6. 关键文件

- `swarm/ra/hocbf.py`：`beta_of_tau` + `solve_robust_sampled_data_qp`
- `swarm/ra/runtime_assurance.py`：`tau_px4_min/max` + `_filter_sampled_data`
- `marllib/phase5_runner.py`：CLI `--tau-px4-min/max`
- `marllib/phase5_step_response.py`：Step A 数据采集
- `tests/test_hocbf.py`：robust QP 测试
- `FORMULAS.md`：精确公式
- `docs/decisions/phase7_robustness_plan.md`：robustness 实验设计

## 7. 重要约定（不变）

- 不放松阈值、不加 seed 救失败假设；
- `τ ∈ T` 是**假设**，实验负责证明它在 PX4 上合理，不要把 empirical 区间
  说成 deterministic bound；
- novelty 边界：不写“首次 robust sampled-data CBF”，落点在
  **identified PX4 execution uncertainty + multi-source separation envelope +
  multi-UAV RA + semantic recovery** 的组合与 evidence chain。
