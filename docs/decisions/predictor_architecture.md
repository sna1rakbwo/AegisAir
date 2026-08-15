# 架构决策：Predictor 降级为 Pre-alert（2026-08-15）

## 决策

Predictor **不再作为 LLM 任务级重规划的触发器**，降级为「预警/预热器
（Pre-alert / Speculative Planning）」。真正允许改变 UAV 任务的触发条件来自
Runtime Confirmation（当前裕度 + 裕度退化 + CBF 介入），而不是预测。

核心原则：

> Prediction suggests; Runtime evidence confirms.

## 三层最终架构

```text
Level 1 - Predictive Pre-alert（预测器，best-effort）
    rho_min_pred < 0（H=1.5s, K=1，不加 CPA）
    预热语义恢复上下文 / 生成候选恢复计划
    不修改任务

Level 2 - Runtime Confirmation（确认后才允许改任务）
    rho < rho_warn 且 (g_bar > g_th 或 N_CBF(T) > N_th)
    LLM 确认恢复计划 -> 验证 -> 执行

Level 3 - Hard Safety（CBF，独立）
    TTSB <= T_hard 或任何真实危险
    立即介入，不等待 LLM
```

流程图：

```text
MARL -> Runtime Assurance (Margin + CBF) -> PX4

Predictor -> Possible Future Risk -> Speculative LLM Planning
    -> Candidate Recovery -> WAIT
    -> Runtime Risk Confirmed?
        No  -> discard
        Yes -> validate -> execute
```

## 为什么这样改（实验依据）

Phase 3 在 untouched test seeds（15-24，H=1.5s）上的结果：

- Precision 49.5%，Recall 54.9%，FPR 1.0%，FNR 45.1%。
- Persistence/CPA 只能把 precision 从 37.5% 提到 44.2%，同时 recall 从 31% 掉到 19%。
- Event-level aggregation 不能解决误报，因为误报往往成串出现。

结论：short-horizon CV/CA predictor 的可分辨能力已接近上限，继续调参无意义。
因此把「预测直接触发任务变更」改为「预测只做预热，任务变更由 runtime 证据确认」。

## 负面消融（保留进论文）

> Persistence filtering increased precision from 37.5% to 43.2%, but reduced
> recall from 31.0% to 19.1%. CPA confirmation provided only marginal
> additional improvement. Event-level aggregation did not resolve false alarms
> because false predictions tended to occur in persistent bursts.

这证明：simple temporal/geometric filtering cannot fundamentally overcome
short-horizon model mismatch，从而支撑「Prediction should not be trusted as
a safety-critical trigger」的论点。

## 论文叙事（更新）

原：Predictive Safety Margin enables proactive semantic recovery.

改为：

> Short-horizon prediction provides an advisory pre-alert for anticipatory
> semantic preparation, while mission-changing recovery is committed only after
> runtime risk confirmation.

## 冻结的 Predictor 配置

```text
Motion model:   Filtered CA
Uncertainty:    Q = 0
Threshold:      rho_th = 0
Horizon:        H = 1.5 s
Persistence:    K = 1（pre-alert 便宜，不追求 precision）
Confidence:     prediction_reliability_score（heuristic，非 probability）
```

## Phase 4 实验设计（R0/R1/R2）

- R0：无预测器，Runtime conflict -> LLM starts -> Recovery。
- R1：预测器直接触发 LLM 执行（预期较多无意义重规划）。
- R2：预测器 speculative planning + runtime confirmation（本决策架构）。

比较指标：recovery latency、unnecessary mission changes、CBF intervention
duration、mission completion time。

## AegisAir 哲学扩展

至此，系统对三类智能模块都持「不可信」假设：

- LLM untrusted；
- MARL untrusted；
- Predictor untrusted。

只有 deterministic Runtime Assurance（margin + time-varying CBF）负责硬安全。
