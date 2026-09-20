from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

from swarm.geometry import Vector3


def safe_holding_point(
    drone_id: int,
    positions: dict[int, Vector3],
    *,
    offset: float = 1.5,
    arena: tuple[float, float, float, float] = (-6.0, 6.0, -6.0, 6.0),
) -> Vector3:
    """Retreat point away from the swarm centroid, for YIELD/HOLD semantics."""
    pos = positions[drone_id]
    others = [positions[d] for d in positions if d != drone_id]
    if others:
        cx = sum(p[0] for p in others) / len(others)
        cy = sum(p[1] for p in others) / len(others)
    else:
        cx, cy = 0.0, 0.0
    dx = pos[0] - cx
    dy = pos[1] - cy
    length = math.hypot(dx, dy)
    if length < 1e-6:
        dx, dy = 1.0, 0.0
        length = 1.0
    ux, uy = dx / length, dy / length
    x = min(max(pos[0] + ux * offset, arena[0]), arena[1])
    y = min(max(pos[1] + uy * offset, arena[2]), arena[3])
    return (x, y, 0.0)


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _sub(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(a: Vector3, value: float) -> Vector3:
    return (a[0] * value, a[1] * value, a[2] * value)


def _dot(a: Vector3, b: Vector3) -> float:
    return sum(a_i * b_i for a_i, b_i in zip(a, b))


def _norm(a: Vector3) -> float:
    return math.sqrt(_dot(a, a))


def _normalize(a: Vector3) -> Vector3:
    length = _norm(a)
    if length == 0:
        return (0.0, 0.0, 0.0)
    return (a[0] / length, a[1] / length, a[2] / length)


def _as_vector3(value: Any, field_name: str) -> Vector3:
    if not isinstance(value, list | tuple) or len(value) != 3:
        raise ValueError(f"{field_name} must be a list of three numbers")
    return (float(value[0]), float(value[1]), float(value[2]))


@dataclass(frozen=True)
class SafetyConfig:
    safe_distance_m: float = 1.6
    ttc_safe_sec: float = 3.0
    max_speed_mps: float = 2.0
    escape_distance_m: float = 1.2
    low_threshold: float = 0.32
    high_threshold: float = 0.70
    release_threshold: float = 0.25
    hold_sec: float = 1.0


@dataclass(frozen=True)
class DroneSnapshot:
    drone_id: int
    position: Vector3
    target: Vector3 | None = None
    velocity: Vector3 | None = None
    speed_mps: float = 0.0
    status: str = "unknown"
    last_command_id: str | None = None
    timestamp_ms: int = 0
    source_timestamp_us: int = 0

    @classmethod
    def from_telemetry(cls, payload: dict[str, Any]) -> DroneSnapshot:
        drone_id = int(payload["drone"])
        if "position" in payload:
            position = _as_vector3(payload["position"], "position")
        else:
            position = (float(payload["x"]), float(payload["y"]), float(payload["z"]))

        target = _as_vector3(payload["target"], "target") if payload.get("target") is not None else None
        velocity = _as_vector3(payload["velocity"], "velocity") if payload.get("velocity") is not None else None
        speed = float(payload.get("speed_mps") or 0.0)
        return cls(
            drone_id=drone_id,
            position=position,
            target=target,
            velocity=velocity,
            speed_mps=speed,
            status=str(payload.get("status") or "unknown"),
            last_command_id=payload.get("last_command_id"),
            timestamp_ms=int(payload.get("timestamp_ms") or 0),
            source_timestamp_us=int(payload.get("source_timestamp_us") or 0),
        )


@dataclass(frozen=True)
class PairRisk:
    drone: int
    target_drone: int
    distance_m: float
    approach_mps: float
    ttc_sec: float | None
    risk: float
    target_position: Vector3 | None = None


@dataclass(frozen=True)
class SafetyDecision:
    drone: int
    mode: str
    risk_level: float
    target_drone: int | None
    reason: str | None
    command_id: str | None
    position: Vector3
    diverted_from: Vector3 | None
    safety_waypoint: Vector3 | None
    estimated_recovery: float
    timestamp_ms: int


def velocity_toward_target(snapshot: DroneSnapshot) -> Vector3:
    if snapshot.velocity is not None:
        return snapshot.velocity

    if snapshot.target is None or snapshot.speed_mps <= 0:
        return (0.0, 0.0, 0.0)

    delta = _sub(snapshot.target, snapshot.position)
    length = _norm(delta)
    if length == 0:
        return (0.0, 0.0, 0.0)

    scale = snapshot.speed_mps / length
    return (delta[0] * scale, delta[1] * scale, delta[2] * scale)


def pair_collision_risk(a: DroneSnapshot, b: DroneSnapshot, config: SafetyConfig) -> PairRisk:
    relative_position = _sub(b.position, a.position)
    separation = _norm(relative_position)
    if separation == 0:
        return PairRisk(a.drone_id, b.drone_id, 0.0, config.max_speed_mps, 0.0, 1.0, b.position)

    relative_velocity = _sub(velocity_toward_target(a), velocity_toward_target(b))
    approach_speed = _clip(_dot(relative_velocity, relative_position) / separation, 0.0, config.max_speed_mps)
    ttc = separation / approach_speed if approach_speed > 0 else None

    distance_risk = _clip(1.0 - separation / config.safe_distance_m)
    velocity_risk = _clip(approach_speed / config.max_speed_mps)
    ttc_risk = _clip(1.0 - (ttc or config.ttc_safe_sec) / config.ttc_safe_sec) if ttc is not None else 0.0
    risk = _clip(max(distance_risk, distance_risk * velocity_risk, ttc_risk))
    return PairRisk(a.drone_id, b.drone_id, separation, approach_speed, ttc, risk, b.position)


def safety_diversion_waypoint(snapshot: DroneSnapshot, pair: PairRisk, config: SafetyConfig) -> Vector3:
    if pair.target_position is None:
        return snapshot.position

    away = _sub(snapshot.position, pair.target_position)
    horizontal_away = (away[0], away[1], 0.0)
    if _norm(horizontal_away) == 0:
        horizontal_away = (0.0, 1.0 if snapshot.drone_id <= pair.target_drone else -1.0, 0.0)

    away_dir = _normalize(horizontal_away)
    lateral_dir = _normalize((-away_dir[1], away_dir[0], 0.0))
    safety_dir = _normalize(_add(_scale(away_dir, 0.45), _scale(lateral_dir, 0.90)))
    if _norm(safety_dir) == 0:
        safety_dir = away_dir

    escape_distance = max(config.escape_distance_m, config.safe_distance_m * 0.75)
    return _add(snapshot.position, _scale(safety_dir, escape_distance))


class SafetyGate:
    def __init__(self, config: SafetyConfig | None = None):
        self.config = config or SafetyConfig()
        self._override_until: dict[int, float] = {}

    def evaluate(self, snapshots: list[DroneSnapshot], now: float | None = None) -> list[SafetyDecision]:
        now = time.monotonic() if now is None else now
        decisions: list[SafetyDecision] = []

        for snapshot in snapshots:
            worst = self._worst_pair(snapshot, snapshots)
            risk = worst.risk if worst else 0.0
            mode = self._mode(snapshot.drone_id, risk, now)
            reason = "collision_risk_exceeded" if mode == "override" else None
            safety_waypoint = (
                safety_diversion_waypoint(snapshot, worst, self.config)
                if mode == "override" and worst is not None
                else None
            )
            decisions.append(
                SafetyDecision(
                    drone=snapshot.drone_id,
                    mode=mode,
                    risk_level=risk,
                    target_drone=worst.target_drone if worst else None,
                    reason=reason,
                    command_id=snapshot.last_command_id,
                    position=snapshot.position,
                    diverted_from=snapshot.target,
                    safety_waypoint=safety_waypoint,
                    estimated_recovery=self.config.hold_sec if mode == "override" else 0.0,
                    timestamp_ms=int(time.time_ns() // 1_000_000),
                )
            )

        return decisions

    def _worst_pair(self, snapshot: DroneSnapshot, snapshots: list[DroneSnapshot]) -> PairRisk | None:
        pairs = [
            pair_collision_risk(snapshot, other, self.config)
            for other in snapshots
            if other.drone_id != snapshot.drone_id
        ]
        return max(pairs, key=lambda pair: pair.risk, default=None)

    def _mode(self, drone_id: int, risk: float, now: float) -> str:
        held_until = self._override_until.get(drone_id, 0.0)
        if risk >= self.config.high_threshold:
            self._override_until[drone_id] = now + self.config.hold_sec
            return "override"

        if now < held_until and risk >= self.config.release_threshold:
            return "override"

        if risk >= self.config.low_threshold:
            return "warning"

        return "normal"


def build_hover_command(decision: SafetyDecision) -> dict[str, Any]:
    return {
        "drone": decision.drone,
        "action": "hover",
        "priority": "safety",
        "confidence": 1.0,
        "command_id": f"safety-hover-{decision.timestamp_ms}-{decision.drone}",
        "reason": decision.reason or "safety_gate",
        "timestamp_ms": decision.timestamp_ms,
    }


def build_safety_command(decision: SafetyDecision) -> dict[str, Any]:
    if decision.safety_waypoint is None:
        return build_hover_command(decision)

    return {
        "drone": decision.drone,
        "action": "move_to",
        "target": list(decision.safety_waypoint),
        "waypoint": list(decision.safety_waypoint),
        "priority": "safety",
        "confidence": 1.0,
        "ttl_sec": max(0.5, decision.estimated_recovery),
        "command_id": f"safety-divert-{decision.timestamp_ms}-{decision.drone}",
        "reason": decision.reason or "safety_gate",
        "rationale": "minimum-intervention safety diversion",
        "timestamp_ms": decision.timestamp_ms,
    }


def build_override_event(decision: SafetyDecision) -> dict[str, Any]:
    return {
        "drone": decision.drone,
        "event": "safety_override",
        "event_name": "safety_override",
        "reason": decision.reason or "collision_risk_exceeded",
        "risk_level": round(decision.risk_level, 4),
        "target_drone": decision.target_drone,
        "diverted_from": list(decision.diverted_from) if decision.diverted_from is not None else None,
        "estimated_recovery": decision.estimated_recovery,
        "command_id": decision.command_id,
        "timestamp_ms": decision.timestamp_ms,
    }


def build_status_event(decision: SafetyDecision) -> dict[str, Any]:
    return {
        "drone": decision.drone,
        "event": "safety_status",
        "event_name": "safety_status",
        "mode": decision.mode,
        "risk_level": round(decision.risk_level, 4),
        "target_drone": decision.target_drone,
        "status": "safety_override" if decision.mode == "override" else decision.mode,
        "anomalies": ["collision_risk"] if decision.mode == "override" else [],
        "timestamp_ms": decision.timestamp_ms,
    }
