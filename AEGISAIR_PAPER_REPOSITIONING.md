# AegisAir 论文重定位（写作骨架）

目标：把论文从“组件展示 / LLM 好不好用”重定位为“**安全集成不可信 AI 的运行时架构 +
可审计证据链**”，以支撑 SCI 二区投稿。所有措辞以冻结证据为准，不越界。

## 1. 一句话命题

> 多无人机系统里的 AI 组件（MARL、LLM、预测器、状态估计）都不可信；真正的安全来自把
> 它们隔离在确定性 Runtime Assurance 与 fail-closed 恢复之后，并给出可审计的证据链。

英文：*Assume AI can fail: a deterministic runtime-assurance + fail-closed-recovery
boundary keeps untrusted autonomy components from endangering multi-UAV safety.*

## 2. 三支柱重命名与排序

| 编号 | 论文术语 | 冻结证据 | 定性 |
|---|---|---|---|
| C1 | risk-adaptive multi-source safety envelope | `minimum_c1_v2_observation_consistent_decision.md` | 正向，核心 |
| C2 | execution-consistent sampled-data Runtime Assurance | `minimum_c2_v2_exact_zoh_result.md` | 正向，核心 |
| C3 | fail-closed semantic mission recovery | `minimum_c3_v1_recovery_result.md` | 架构正向，LLM 作 limitation |

## 3. 负结果的写法（写成支持论点的证据，不是硬伤）

- 本地 4B LLM 因推理延迟（P50 2.8 s）与 1 s 计划有效期不匹配，未带来稳定任务收益 →
  “正因为 AI 不可信，validator + 规则回退 + RA 才是安全边界”。
- 预测器误差重尾、robust-τ 无 episode 级收益 → “预测只能作 pre-alert，不能作概率保证”。
- 这些负结果不是失败，而是**对核心命题的实证**：确定性安全网必要且有效。

## 4. headline 结果（摘要里必须出现的硬数字/结论）

- C1：完整风险包络相对固定距离与消融降低碰撞/边界违规、提高最小几何间距（引用具体表）。
- C2：exact-ZOH 相对历史启发式在控制周期内部把负裕度恢复为正（E1 `-0.0012` → E2 `+0.383`）。
- C3：确定性规则恢复相对 RA-only 恢复预注册任务指标；五类故障注入各 30 次 **safety bypass = 0**。

## 5. 摘要骨架（中文，可直接改写）

> 多无人机系统中，MARL 策略、LLM 语义推理、预测器与共享状态估计均可能出错，直接信任这些
> AI 组件会带来不可控的安全风险。本文提出 AegisAir——一套把不可信 AI 隔离在确定性安全
> 边界之后的运行时架构：多源风险（动力学、感知不确定、通信/AoI）进入动态安全包络；PX4
> 执行滞后由精确 ZOH 离散化的 sampled-data 控制屏障函数处理；任务级恢复采用 fail-closed
> 语义层（LLM 只产生高层意图，经确定性校验与规则回退，最终仍受 Runtime Assurance 否决）。
>
> 在轻量仿真与 PX4 SITL/Gazebo 闭环上，本文（1）证明完整风险包络相对固定距离及各消融降低
> 碰撞率与边界违规、提高最小几何间距；（2）证明执行一致性的 exact-ZOH 控制相对历史启发式
> 在控制周期内部保持正安全裕度；（3）证明确定性规则恢复相对 RA-only 恢复预注册任务指标，
> 且五类故障注入下 safety bypass = 0。同时诚实报告 AI 组件边界：本地 4B LLM 因延迟与有效期
> 不匹配未带来稳定任务收益，预测器误差重尾、robust-τ 无 episode 级收益。这些负结果共同支持
> 本文核心命题——AI 不可信，确定性安全网才是多无人机安全的可靠基础。

## 6. 标题候选

1. Safe Integration of Untrusted AI in Multi-UAV Systems via Runtime Assurance and
   Fail-Closed Semantic Recovery
2. Assume AI Can Fail: Runtime Assurance for AI-Enabled Multi-UAV Safety
3. AegisAir: Risk-Adaptive Runtime Assurance and Fail-Closed Recovery for AI-Enabled
   Multi-UAV Systems

## 7. 红线（措辞禁区）

- 不写“LLM 提升恢复/避碰”，不写“LLM 优于规则”。
- 不写“连续时间安全定理/任意状态保证”，只写“在测试范围与审计分辨率内未观察到违规”。
- 不把 SITL/Gazebo 写成真机证据，不写“真实硬件飞行安全”。
- 不把 empirical τ 区间写成 PX4 全局 deterministic bound。
