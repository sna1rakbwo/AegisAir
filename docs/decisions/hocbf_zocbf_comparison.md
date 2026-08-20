# HOCBF vs ZOCBF 最小受控对比

> 日期：2026-08-17
>
> 目的：回答 AegisAir 历史 HOCBF baseline 与 Tan et al. ZOCBF 在同一受限
> 双积分 head-on 场景中的数值差异。
>
> 边界：这是简化受控比较，不是 ZOCBF 论文完整复现，也不改变已经冻结的
> sampled-data RA 主线。

## 1. 两种约束

### AegisAir HOCBF

对 `h=||r||²-D²`、`p_dot=v, v_dot=a`：

```text
psi2 = h_ddot+(k1+k2)h_dot+k1k2h >= 0

2r^T(a_i-a_j) >=
  -2||v||²-2(k1+k2)r^T v-k1k2(||r||²-D²)
```

本实验使用历史冻结值 `k1=k2=3`。

### ZOCBF

论文定义的一步约束为：

```text
h(phi(T;x_k,u_k),u_k)-h(x_k,u_{k-1})
  >= -gamma(h(x_k,u_{k-1}))+delta
```

本实验的 barrier 只依赖状态，并取线性 class-K function：

```text
h_{k+1} >= (1-gamma_c)h_k+delta
```

对固定顺序 `p0<p1` 的 1D head-on 双积分：

```text
r_next = r+v_rel*T+0.5(a0-a1)T²
h_next = r_next²-D²
```

保持 `r_next<0` 后可把 exact squared-distance ZOCBF 化为一个关于 `(a0,a1)`
的线性 half-space，因此不需要引入 SLSQP/Scipy。

## 2. 公平设置

```text
initial pair distance = {4,5,6} m
initial speed per UAV = {0.5,1.0,1.5} m/s
cases = 3x3 = 9
D = 1.5 m
T = 0.05 s
horizon = 120 steps = 6 s
a_max = 2.0 m/s²
a_nom = 0（若不干预则持续 head-on）
HOCBF: k1=k2=3
ZOCBF: gamma_c=0.1, delta=0.01
```

两者使用相同状态、nominal acceleration、box constraint 和精确双积分仿真。
每个控制周期再用 20 个子采样点审计 intersample minimum `h`。

## 3. 结果

| controller | sample violations | intersample violations | cases with infeasible step | worst min h | mean intervention rate | mean control effort |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HOCBF | 0/9 | 0/9 | 0/9 | `+7.78e-6` | 0.773 | 1.998 |
| ZOCBF | 6/9 | 6/9 | 6/9 | `-1.959` | 0.684 | 1.989 |

结果文件：

```text
/Volumes/Expansion/aegisair_phase7_step_response_20260817/hocbf_zocbf_comparison.json
```

## 4. 解读

在这个 bounded-acceleration head-on stress test 中，HOCBF 的 `h_dot`/closing-speed
项会较早产生制动约束，9 个 case 均保持 feasible 和 `h>=0`。当前 ZOCBF 参数
允许 `h` 每一步下降一定比例；高 closing speed 下，控制影响位置的系数只有
`0.5T²`，到接近边界时所需 acceleration 已超过 box constraint，随后 ZOCBF
constraint 不可行。

这说明：

> 对当前 head-on braking envelope，显式使用相对速度的 HOCBF 比这一组
> one-step ZOCBF 参数更能提前保留 recursive feasibility。

它不说明：

> HOCBF 普遍优于 ZOCBF，或 ZOCBF theorem 被反例推翻。

ZOCBF 的安全定理以每个 sampling instant 都存在满足条件的输入为前提；本实验
恰好显示该前提在 6/9 个受限 case 中丢失。

## 5. 重要限制

1. 论文的 ZOCBF 适用于一般高 relative-degree 和 state-input constraints；本实验
   只覆盖 1D state-only collision barrier。
2. `delta=0.01` 沿用论文双积分数值例子的量级，但没有为 AegisAir 当前状态域
   正式证明 `delta >= h_bar_x M T`，所以不能据此 claim 论文的完整 intersample
   theorem。
3. 论文双积分示例使用 `T=0.1 s`、`gamma_c=1`、`u in [-10,10]`，而本实验使用
   AegisAir 的 `T=0.05 s`、`a_max=2`。这是系统匹配比较，不是 paper-parameter
   reproduction。
4. ZOCBF 可通过更小 `gamma_c`、更大的可控输入、viability-aware tuning 或更早
   activation 改善；继续调这些参数会成为新的研究支线，不属于当前正式实验主线。
5. AegisAir 最终控制器是 exact PX4 sampled-data barrier，不是本历史 HOCBF。

## 6. 决定

保留该实验为简短相关方法对比或 appendix diagnostic。它可以支持“当前 HOCBF
baseline 在有限加速度 head-on 场景更早介入”的局部观察，但不将 ZOCBF 纳入
正式主实验，也不据此改变已冻结系统。
