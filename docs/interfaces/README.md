# SafeDrones 冻结接口规范（Phase 0）

> 状态：已冻结 v1
> 权威定义：`swarm/interfaces.py`（pydantic v2 模型）
> 机器可读 schema：`schemas/interfaces/*.schema.json`
> 示例：`examples/interfaces/*.json`

本文档是 SafeDrones 各层之间的数据契约。所有跨层消息（遥测、观测、动作、
风险事件、恢复计划、安全决策、日志）都以这里的 schema 为准。

## 0. 版本规则

- 每个消息都带 `schema_version`。
- **破坏性变更**（删除/改名字段、改变类型或枚举、改变数组长度语义）必须
  `schema_version + 1`。
- **只增不改**：新增可选字段不算破坏性变更，不升级版本号。
- 所有模型启用 `extra="forbid"`，未知字段直接拒绝，保证线上格式与声明一致。

## 1. 遥测 Telemetry

对应 MQTT topic：`swarm/drone/{id}/telemetry`。统一 FLU 世界系
（`x_forward, y_left, z_up`）。

字段：

- `drone`：UAV 实例 id；
- `position` / `velocity`：FLU 三维位置与速度；
- `yaw_deg`：FLU 航向角（度）；
- `status`：`disarmed | armed | failsafe | gcs_connection_lost`；
- `armed` / `nav_state` / `failsafe` / `connection_lost`：PX4 状态；
- `source_frame`：固定 `FLU`；
- `source_timestamp_us` / `timestamp_ms`：源时间戳与采集墙钟。

该形状与 `px4_adapter/mqtt_codec.py` 的 `encode_safedrones_telemetry` 对齐。

## 2. 观测 Observation

MARL 单智能体观测，遵循计划中的相对量约定：

```text
o_i = [v_i, p_goal - p_i, p_j - p_i, v_j - v_i, ...]
```

字段：

- `agent_id`：本机 id；
- `position_flu` / `velocity_flu`：本机 FLU 位置与速度（`v_i` 绝对量）；
- `goal_relative` / `goal_distance`：目标相对向量与距离；
- `neighbors[]`：每架邻机的 `relative_position`、`relative_velocity`、
  `distance`、`closing_speed`；
- `role`：可选语义角色编码；
- `perception`：可选感知质量（噪声标准差、staleness）。

策略网络的输入由该结构扁平化得到；扁平化顺序属于 Phase 1 的实现约定，不改变
本 schema。

## 3. MARL 动作 MarlAction

字段：

- `agent_id`；
- `velocity_cmd`：FLU 二维速度指令 `[vx, vy]`（v1 定高）。

升级到三维 `[vx, vy, vz]` 属于破坏性变更，需 `schema_version + 1`。

## 4. 风险事件 RiskEvent

Runtime Assurance 发给语义层的结构化事件。

`event` 取值：

- `PREDICTED_CONFLICT`（主动预测冲突）；
- `SAFETY_MARGIN_DEGRADATION`（安全裕度退化）；
- `HARD_INTERVENTION`（硬安全介入）。

公共字段：`agent_i` / `agent_j`、`current_margin`（当前 ρ）、`cause`、
`severity`、`timestamp_ms`。

条件字段：

- 主动事件：`predicted_min_margin`、`time_to_min_margin_s`；
- 退化事件：`margin_degradation`、`intervention_count`。

`cause` 取值：`DYNAMICS | PERCEPTION_UNCERTAINTY | COMMUNICATION_STALE |
TRAJECTORY_CONFLICT | UNKNOWN`。

## 5. 恢复计划 RecoveryPlan

本地 LLM 语义恢复输出，动作白名单：

```text
HOLD | YIELD | REROUTE | REASSIGN | CHANGE_PRIORITY | ABORT | RETURN
```

字段：

- `intent_text`：自然语言意图（仅用于审计，不作为低层输入）；
- `commands[]`：`drone`、`action`、`waypoint`（REROUTE 必填）、`priority`、
  `ttl_sec`、`command_id`；
- `constraints`：任务级约束（如最小间距、避让中心区），**不能降低硬安全**；
- `rationale`：决策理由。

LLM 输出必须经过 JSON Schema + 动作白名单 + 校验器；非法输出走超时/回退，且
无法突破 Safety Gate。

## 6. 安全决策 SafetyDecision

Runtime Assurance 的单机输出，即最小侵入安全过滤结果：

- `mode`：`normal | warning | override`；
- `nominal_action`：MARL 名义动作 `u_nom`；
- `safe_action`：过滤后安全动作 `u_safe`；
- `safety_margin`：归一化安全裕度 ρ；
- `predicted_margin`：可选预测裕度；
- `reason`：介入原因。

## 7. 执行保证决策 ExecutionAssuranceDecision

C2′ 对执行模型的运行时有效性判断：

- `active`：确定性备份控制是否已接管；
- `reasons`：速度/位置残差、QP 不可行、求解超时、遥测陈旧、可恢复裕度耗尽或迟滞保持；
- `residuals`：各机一步速度与位置残差；
- `recoverability`：各配对的闭合速度、保守制动所需距离和裕度；
- `backup_drones`：在前瞻式 gate 下确定性让行并执行备份控制的无人机；
- `qp_feasible`、`solve_elapsed_s`、`telemetry_ages_s`：交叉层触发证据；
- `consecutive_bad/clean`：锁存与释放的可审计状态。

该消息只记录监测和切换，不允许修改 RA 安全阈值。

## 8. C3 冲突准入决策 CoordinationAdmissionDecision

C3-Admission 每个控制周期只记录任务层通行结构：状态、双门触发证据、准入机、
等待机、一次性冻结等待点与进入步、冲突区、RA 否决及规则/回退来源。该消息不含
执行器命令，C3 生成的意图仍须经过 RA。

## 9. C3 终点占用感知预约 CoordinationReservationDecision

C3-Reservation 复用既有安全等待点，并显式记录 staging、通行、目标服务锁存、
终点清空、下一机释放、最终归位及 RA veto 撤销。该消息仍不含执行器命令。

## 10. C3 空时预约决策 CoordinationSpaceTimeReservationDecision

C3-STR 在原终点占用生命周期上增加可审计的服务/归位时间窗、当前开放槽、短时承诺
航点、计划修订号和累计延迟。时间窗只约束任务层准入；RA 否决时计划只能后移，不能
强制执行或绕过安全过滤。兼容的最终归位可以共享一个槽，不兼容航路仍保持互斥。

该消息不允许携带速度、加速度或执行器命令。

## 11. 日志事件 LogEvent

append-only JSONL，每行一条：

```text
schema_version + event_type + timestamp_ms + seed + episode_id + payload
```

`event_type` 取值：

```text
episode_meta | telemetry | observation | marl_action | safety_decision
| execution_assurance | coordination_admission | coordination_reservation
| coordination_space_time_reservation
| risk_event | recovery_plan
| fault_injection
```

`payload` 是 `event_type` 对应的子 schema 载荷。

## 校验与回归

```bash
# 校验示例与拒绝规则
python -m unittest tests.test_interfaces -v

# 重新生成 JSON Schema 文件
python scripts/dump_interfaces.py
```

任何模块接入新消息时，必须先在此冻结 schema 并加测试，禁止临时拼 JSON。
