"""Frozen cross-layer interfaces for the SafeDrones runtime.

These pydantic models are the single source of truth for Phase 0.  Any breaking
change MUST bump ``schema_version``; additive optional fields are allowed
without a bump.  ``extra="forbid"`` rejects unknown fields so the wire format is
exactly what is declared here.

JSON Schema artifacts and example payloads can be regenerated with
``scripts/dump_interfaces.py``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]


class _Frozen(BaseModel):
    """Common strict-model settings for every interface schema."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------------------
# 1. Telemetry representation
# ---------------------------------------------------------------------------


class Telemetry(_Frozen):
    """One UAV state snapshot in the shared FLU world frame.

    This is the canonical shape published on ``swarm/drone/{id}/telemetry`` and
    is aligned with ``px4_adapter/mqtt_codec.encode_safedrones_telemetry``.
    """

    schema_version: Literal[1] = 1
    drone: int = Field(ge=0, description="UAV instance id")
    position: Vec3 = Field(description="FLU position [x_forward, y_left, z_up] in meters")
    velocity: Vec3 = Field(description="FLU velocity [vx, vy, vz] in m/s")
    yaw_deg: float = Field(default=0.0, description="FLU heading in degrees")
    status: Literal["disarmed", "armed", "failsafe", "gcs_connection_lost"]
    armed: bool
    nav_state: int
    failsafe: bool
    connection_lost: bool
    source_frame: Literal["FLU"] = "FLU"
    source_timestamp_us: int = Field(ge=0, description="Source PX4 timestamp in microseconds")
    timestamp_ms: int = Field(ge=0, description="Wall-clock capture time in milliseconds")


# ---------------------------------------------------------------------------
# 2. MARL observation
# ---------------------------------------------------------------------------


class NeighborObservation(_Frozen):
    """Relative state of one other UAV."""

    agent_id: int = Field(ge=0)
    relative_position: Vec3 = Field(description="p_j - p_i in FLU")
    relative_velocity: Vec3 = Field(description="v_j - v_i in FLU")
    distance: float = Field(ge=0.0)
    closing_speed: float = Field(ge=0.0, description="Non-negative closing speed")


class PerceptionMeta(_Frozen):
    """Optional perception-quality metadata carried alongside an observation."""

    noise_std_m: float = Field(ge=0.0, description="Perceived position noise std in meters")
    staleness_ms: int = Field(ge=0, description="Age of the perceived state in milliseconds")


class Observation(_Frozen):
    """MARL observation for one agent.

    Self velocity is absolute; goal and neighbors are relative quantities, as
    specified by the plan's ``o_i = [v_i, p_goal-p_i, p_j-p_i, v_j-v_i, ...]``.
    """

    schema_version: Literal[1] = 1
    agent_id: int = Field(ge=0)
    position_flu: Vec3 = Field(description="Self FLU position (for logging/debug)")
    velocity_flu: Vec3 = Field(description="Self FLU velocity v_i")
    goal_relative: Vec3 = Field(description="p_goal - p_i in FLU")
    goal_distance: float = Field(ge=0.0)
    neighbors: list[NeighborObservation] = Field(default_factory=list)
    role: str | None = Field(default=None, description="Optional semantic role encoding")
    perception: PerceptionMeta | None = Field(default=None)
    timestamp_ms: int = Field(ge=0)


# ---------------------------------------------------------------------------
# 3. MARL action
# ---------------------------------------------------------------------------


class MarlAction(_Frozen):
    """Nominal MARL command for one agent.

    v1 is 2D fixed-altitude velocity control ``[vx_cmd, vy_cmd]``.  Moving to
    3D velocity command is a schema_version bump, not a silent field change.
    """

    schema_version: Literal[1] = 1
    agent_id: int = Field(ge=0)
    velocity_cmd: Vec2 = Field(description="Commanded FLU velocity [vx, vy] in m/s")
    timestamp_ms: int = Field(ge=0)


# ---------------------------------------------------------------------------
# 4. Structured RiskEvent
# ---------------------------------------------------------------------------


RiskCause = Literal[
    "DYNAMICS",
    "PERCEPTION_UNCERTAINTY",
    "COMMUNICATION_STALE",
    "TRAJECTORY_CONFLICT",
    "UNKNOWN",
]
RiskSeverity = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]


class RiskEvent(_Frozen):
    """Structured event emitted by Runtime Assurance to the semantic layer."""

    schema_version: Literal[1] = 1
    event: Literal["PREDICTED_CONFLICT", "SAFETY_MARGIN_DEGRADATION", "HARD_INTERVENTION"]
    agent_i: int = Field(ge=0)
    agent_j: int = Field(ge=0)
    current_margin: float = Field(description="Normalized safety margin rho at capture time")
    predicted_min_margin: float | None = Field(
        default=None, description="Worst predicted rho over the horizon (proactive only)"
    )
    time_to_min_margin_s: float | None = Field(
        default=None, ge=0.0, description="Seconds until predicted minimum margin"
    )
    margin_degradation: float | None = Field(
        default=None, description="Safety-margin degradation rate g (reactive only)"
    )
    intervention_count: int | None = Field(default=None, ge=0)
    cause: RiskCause
    severity: RiskSeverity
    timestamp_ms: int = Field(ge=0)


# ---------------------------------------------------------------------------
# 5. RecoveryPlan (LLM semantic recovery)
# ---------------------------------------------------------------------------


RecoveryAction = Literal[
    "HOLD",
    "YIELD",
    "REROUTE",
    "REASSIGN",
    "CHANGE_PRIORITY",
    "ABORT",
    "RETURN",
]


class RecoveryCommand(_Frozen):
    """One high-level recovery command in the LLM action whitelist."""

    drone: int = Field(ge=0)
    action: RecoveryAction
    waypoint: Vec3 | None = Field(default=None, description="Required for REROUTE")
    priority: Literal["normal", "safety"] = "normal"
    ttl_sec: float = Field(gt=0.0)
    command_id: str = Field(min_length=1)


class RecoveryConstraints(_Frozen):
    """Task-level constraints the LLM may express but never bypass hard safety."""

    keep_min_distance_m: float | None = Field(default=None, ge=0.0)
    avoid_center_zone: bool | None = None


class RecoveryPlan(_Frozen):
    """Structured semantic-recovery plan returned by the local LLM."""

    schema_version: Literal[1] = 1
    intent_text: str = Field(min_length=1)
    commands: list[RecoveryCommand] = Field(min_length=1)
    constraints: RecoveryConstraints | None = None
    rationale: str = ""
    timestamp_ms: int = Field(ge=0)


# ---------------------------------------------------------------------------
# 6. Safety Decision (Runtime Assurance output)
# ---------------------------------------------------------------------------


class SafetyDecision(_Frozen):
    """Minimally-invasive safety-filter result for one agent."""

    schema_version: Literal[1] = 1
    drone: int = Field(ge=0)
    mode: Literal["normal", "warning", "override"]
    nominal_action: Vec2 = Field(description="MARL nominal action u_nom")
    safe_action: Vec2 = Field(description="Filtered safe action u_safe")
    safety_margin: float = Field(description="Normalized safety margin rho")
    predicted_margin: float | None = Field(default=None)
    reason: str | None = None
    timestamp_ms: int = Field(ge=0)


# ---------------------------------------------------------------------------
# 7. Append-only log event
# ---------------------------------------------------------------------------


LogEventType = Literal[
    "episode_meta",
    "telemetry",
    "observation",
    "marl_action",
    "safety_decision",
    "risk_event",
    "recovery_plan",
    "fault_injection",
]


class LogEvent(_Frozen):
    """One line of the append-only JSONL episode log."""

    schema_version: Literal[1] = 1
    event_type: LogEventType
    timestamp_ms: int = Field(ge=0)
    seed: int
    episode_id: str = Field(min_length=1)
    payload: dict[str, Any] = Field(
        description="Payload matching the schema named by ``event_type``"
    )
