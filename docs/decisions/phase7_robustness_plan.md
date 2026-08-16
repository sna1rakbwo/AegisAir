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

### 3.1 sim 级复现（已做，`marllib/phase6_residual_check.py`）

在 `MultiUAVEnv` 精确动力学下，用相同 `randomized_4` checkpoint + sampled-data
barrier（`tau_ctrl=0.2`, `gamma=0.1`, `aoi=0`）复跑，**没有出现 `ρ<0`**：

```text
min-rho point: rho=0.3305, residual = h_next_actual - (1-gamma)*h_now = +0.172
```

残差为**正**，说明在精确 sim 里 barrier 的离散预测是**保守**的（实际 h 高于
预算），不是乐观的。因此：

> live 的 `ρ<0` 不是 sampled-data barrier 算法本身的问题，而是 live 执行/遥测
> 与 barrier 假设的动力学之间的差距。

候选 live 专属因素（下一步逐一隔离）：

1. PX4 velocity-tracking 动力学 ≠ sim 的离散 velocity-command 模型；
2. 遥测 age（`aoi`）使 live 的 `d_safe` 被 `M_comm` 抬高（实测 mean 0.011s、
   max 0.058s，`M_comm` 约 0.02–0.12m）；
3. live 用了 `--sequential-pass`，而当前 sim 复现未启用（需补上再对照）。

### 3.2 隔离结果（`phase6_residual_check.py` 已补 `--sequential-pass` / `--aoi-ms`）

同样 MAPPO + sampled-data（`tau_ctrl=0.2`, `gamma=0.1`）在精确 sim 里：

| seq-pass | aoi | 结果 |
| --- | --- | --- |
| off | 0 | 不越界，min `rho=0.33` |
| on | 0 | 不越界，min `rho=0.19` |
| on | 11ms | **越界**，step 301 `rho=-0.007`（很晚、很浅） |
| on | 58ms | 不越界，min `rho=0.14` |

结论：

- `sequential-pass` 本身不直接造成越界，只把 min `rho` 压到更低；
- `aoi` 是**影响因素**：11ms 能让 sim 出现浅越界，但比 live（t≈7s 就
  `rho=-0.5`）晚且浅得多；
- 因此 live 的强越界仍主要来自 **PX4 velocity-tracking 动力学**与 sim 的
  精确 velocity-command 模型的差距，`aoi` 只是叠加项。

下一步：给 sim 加一个更贴近 PX4 的 velocity-tracking 模型（例如一阶速度
响应 `dv/dt = (v_cmd - v)/tau_px4` 或带加速度上限的跟踪滞后），看能否把
live 的强越界复现出来；确认后就能决定是修 barrier 的动力学假设，还是把
claim 收紧为 feasible/well-modeled envelope。

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
