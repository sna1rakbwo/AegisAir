from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

from swarm.safety import DroneSnapshot
from swarm.geometry import Vector3


@dataclass(frozen=True)
class EgoPosition:
    drone_id: int
    position: Vector3
    estimated_depth: float | None
    timestamp_ms: int
    received_at: float


def parse_ego_position(payload: dict[str, Any], received_at: float | None = None) -> EgoPosition:
    drone_id = int(payload["drone"])
    value = payload["position"]
    if not isinstance(value, list | tuple) or len(value) != 3:
        raise ValueError("position must be a list of three numbers")
    position = (float(value[0]), float(value[1]), float(value[2]))
    depth = payload.get("estimated_depth")
    return EgoPosition(
        drone_id=drone_id,
        position=position,
        estimated_depth=float(depth) if depth is not None else None,
        timestamp_ms=int(payload.get("timestamp_ms") or 0),
        received_at=time.monotonic() if received_at is None else received_at,
    )


class EgoStateStore:
    def __init__(self) -> None:
        self._positions: dict[int, EgoPosition] = {}

    def update(self, payload: dict[str, Any], received_at: float | None = None) -> None:
        estimate = parse_ego_position(payload, received_at=received_at)
        self._positions[estimate.drone_id] = estimate

    def position_for(self, drone_id: int, ttl_sec: float, now: float | None = None) -> Vector3 | None:
        now = time.monotonic() if now is None else now
        estimate = self._positions.get(drone_id)
        if estimate is None or now - estimate.received_at > ttl_sec:
            return None
        return estimate.position

    def snapshot_for_pilot(
        self,
        self_snapshot: DroneSnapshot,
        snapshots: list[DroneSnapshot],
        ttl_sec: float,
        now: float | None = None,
    ) -> list[DroneSnapshot]:
        now = time.monotonic() if now is None else now
        observed = [self_snapshot]
        for snapshot in snapshots:
            if snapshot.drone_id == self_snapshot.drone_id:
                continue
            position = self.position_for(snapshot.drone_id, ttl_sec, now)
            if position is None:
                continue
            observed.append(
                DroneSnapshot(
                    drone_id=snapshot.drone_id,
                    position=position,
                    target=snapshot.target,
                    velocity=snapshot.velocity,
                    speed_mps=snapshot.speed_mps,
                    status=snapshot.status,
                    last_command_id=snapshot.last_command_id,
                    timestamp_ms=snapshot.timestamp_ms,
                )
            )
        return observed

    def snapshots_for_gate(
        self,
        snapshots: list[DroneSnapshot],
        ttl_sec: float,
        now: float | None = None,
    ) -> list[DroneSnapshot]:
        now = time.monotonic() if now is None else now
        observed: list[DroneSnapshot] = []
        for snapshot in snapshots:
            position = self.position_for(snapshot.drone_id, ttl_sec, now)
            if position is None:
                continue
            observed.append(
                DroneSnapshot(
                    drone_id=snapshot.drone_id,
                    position=position,
                    target=snapshot.target,
                    velocity=snapshot.velocity,
                    speed_mps=snapshot.speed_mps,
                    status=snapshot.status,
                    last_command_id=snapshot.last_command_id,
                    timestamp_ms=snapshot.timestamp_ms,
                )
            )
        return observed
