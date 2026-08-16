# Phase 7 —— RA 鲁棒性实验设计（含 MAPPO 越界诊断）

> 日期：2026-08-16
> 状态：诊断先行；不重训 MAPPO 去“背固定场景”

## 1. 诊断结论：先定位问题在哪

live 4 机 + `randomized_4` MAPPO 复测 `min_rho < 0`。用新增的 barrier 诊断
字段（`a_nom/a_safe/feasible/accel_saturated/vel_saturated/d_safe`）看到：

- `feasible=True`：QP 每步都有解；
- `accel_saturated=False` / `vel_saturated=False`：无控制饱和；
- `intervened=False`：ρ 第一次转负的那一刻，RA 认为不需要修正；
- tracking error ≈ 0.01 m/s：PX4 忠实执行了 `v_safe`。

因此排除：

- **A（执行没跟上）**——tracking error 太小；
- **B（QP infeasible / saturation）**——feasible 且无饱和。

结论是 **C：sampled-data barrier 的离散预测模型不准**。

## 2. 数值证据（worst pair 3–5）

| step | dist | d_safe | h = dist² - d_safe² |
| --- | --- | --- | --- |
| 133 | 0.9794 | 0.9618 | 0.03423 |
| 134 | 0.9771 | 0.9677 | 0.01827 |

实际 h 衰减比 = 0.01827 / 0.03423 = **0.53**，而 barrier 只允许
`(1-γ) = 0.90`。即实际 **47%/step** 的衰减，远超 10% 预算。

把 Δh 分解：

- `Δ(dist²) = -0.00450`（距离变近）；
- `Δ(d_safe²) = +0.01138`（安全边界变大，**贡献约 2.5 倍于距离项**）。

所以击穿主要来自 **`d_safe` 随 closing speed 增长**，而不是单纯“距离没保住”。

## 3. 根因假设（待 sim 级残差确认）

`d_safe = d0 + M_dyn(v_cl) + M_perc + M_comm` 是**状态依赖**的。QP 的一步
预测用 `s_next = d_safe(v_pred)`，其中 `v_pred = v + a_nom*dt`。当 MAPPO 的
`a_nom` 没有足够避让意图时，实际 closing speed 没有按预测下降，`d_safe` 实际
增长高于预测，导致 `h` 被 `d_safe²` 项击穿。

一句话：

> 时变安全集 + γ-relaxed 单步离散 barrier，在 aggressive nominal 下预测失准。

这不是“算法坏了”，也不是“MAPPO 训练得差”。它恰好把论文核心 claim 逼了出来：

> RA 能降低对 nominal policy 质量的依赖，但一旦 nominal 把系统带出
> 可控/可精确建模的安全包络，安全就可能失守。

## 4. 对齐 Phase 7：policy-quality robustness experiment

不要只测一个 checkpoint。做成 **nominal policy × RA** 的因子实验。

### 独立变量

- nominal policy：
  - rule（`go_to_goal`）
  - MAPPO weak / medium / strong（先按“无 RA 时”的任务完成率与 aggressiveness 分档）
- RA：off vs on（sampled-data，冻结 `gamma=0.1`、`tau_ctrl=0.2`）

### 因变量（每 episode）

- collision rate；
- boundary violation rate（`ρ<0` 的 step 占比）；
- `min_rho`；
- mission completion；
- intervention magnitude `|u_safe - u_nom|`；
- QP feasible rate；
- accel / velocity saturation rate；
- constraint residual（predicted `h` vs actual `h` 的残差）；
- tracking error。

### 统计（冻结）

- 固定 evaluation seed set，与 training seed 分开；
- 20–50 episodes / condition；
- 报 mean + uncertainty interval，不挑最好看的 seed。

### claim boundary（关键，先写死）

> RA reduces dependence on nominal policy quality, but safety can fail once
> nominal behavior drives the system outside its feasible / well-modeled
> safety envelope.

并把三类 failure mode 作为正式观测变量：**infeasibility、saturation、
model-mismatch**。

## 5. 下一步（先不训练 MAPPO）

1. 把 `s_next`、`v_cl`、`h_now / h_next_pred / h_next_actual` 也加入诊断日志
   （现在只记了 `s_now`）。
2. 先在 lightweight sim 复现这个 model-mismatch（sim 动力学可控、残差可精确
   计算），确认根因，再回 live。
3. 确认后候选修法（按证据决定，不预设）：
   - feasibility-aware early intervention；
   - 对 `d_safe` 的预测用更保守的 closing-speed 上界；
   - 调小 `γ` 或加鲁棒余量；
   - 仅在“确实进入 viability 边界外”时，把 claim 收紧为 feasible-envelope
     内的 safety guarantee。
