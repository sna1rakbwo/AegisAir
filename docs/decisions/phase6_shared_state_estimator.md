# 架构决策：Shared-State Estimator（Phase 6，2026-08-16）

## 决策

保留 centralized shared-state 架构，但不再把 PX4 `vehicle_local_position`
当“完美共享真值”。改为引入 `SharedStateEstimator`，向 Runtime Assurance 输出
**带协方差、延迟、dropout 的共享状态估计**。

核心句：

> Retain centralized shared-state assumption; replace perfect shared ground
> truth with delayed, dropout-prone, covariance-bearing shared state
> estimates.

## 现状

```text
PX4 vehicle_local_position
-> MQTT telemetry
-> DroneSnapshot(position, velocity)
-> RA 直接算 pairwise d、v_rel
```

其中：

- 位置/速度当真值；
- `perception_sigma=0.1` 固定标量；
- delay 只通过 telemetry age 进 `M_comm`；
- 无 dropout 的估计维持 / 不确定度膨胀。

## 目标

```text
raw telemetry
-> SharedStateEstimator（per-drone：position / velocity / P / t_last / dropped）
-> EstimatedState
-> RA（covariance + AoI -> d_safe）
```

三点映射：

- covariance -> `perception_margin`，`sigma_i` 由协方差在连线方向的投影替代；
- delay -> 估计器输出延迟后的状态，`M_comm` 继续用 age 放大；
- dropout -> hold 上次估计 + 协方差增长，或 stale fail-closed。

## 边界

- 这仍是 **centralized shared-state**，不是 ego 感知。
- 它不改变 HOCBF / sampled-data barrier 的结构，只改变进入 RA 的状态质量。
- `Assume AI can fail` 不变：估计器也只是输入，Runtime Assurance 仍是最终
  硬安全。

## 验证顺序

1. 轻量 sim：synthetic delay/dropout/covariance，看 `min_rho`、fail-closed。
2. live 单机：packet loss / latency / stale telemetry。
3. live 4 机：sampled-data + SEQUENTIAL_PASS + fault injection。
