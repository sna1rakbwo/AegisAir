"""PX4 message and coordinate codecs without ROS 2 runtime imports."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


# PX4 VehicleCommand values needed by the adapter.  The numeric constants are
# kept local so the codec remains testable on a host without px4_msgs.
VEHICLE_CMD_NAV_RETURN_TO_LAUNCH = 20
VEHICLE_CMD_NAV_LAND = 21
VEHICLE_CMD_NAV_TAKEOFF = 22
VEHICLE_CMD_DO_SET_MODE = 176
VEHICLE_CMD_COMPONENT_ARM_DISARM = 400

PX4_ARMING_STATE_ARMED = 2
PX4_NAV_STATE_OFFBOARD = 14


def px4_ned_to_flu(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Convert PX4 NED (forward/right/down) to the plan's FLU convention.

    The plan's unified internal convention is ``[x_forward, y_left, z_up]``.
    PX4 NED is ``[x_forward, y_right, z_down]``, so ``y`` and ``z`` flip sign.
    """
    return (float(x), -float(y), -float(z))


def flu_to_px4_ned(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Inverse of :func:`px4_ned_to_flu`."""
    return (float(x), -float(y), -float(z))


def flu_to_local_ned(
    x: float,
    y: float,
    z: float,
    origin_offset_ned: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[float, float, float]:
    """Convert a shared-frame FLU position to a PX4 local-NED position.

    Telemetry is emitted in the shared frame (``local_ned + origin_offset``);
    PX4 position setpoints are local-NED.  This applies the FLU->NED flip and
    then removes the per-instance origin offset.
    """
    nx, ny, nz = flu_to_px4_ned(x, y, z)
    return (
        nx - origin_offset_ned[0],
        ny - origin_offset_ned[1],
        nz - origin_offset_ned[2],
    )


def _finite(value: Any, field_name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    return value


@dataclass(frozen=True)
class OffboardControlSpec:
    position: bool = True
    velocity: bool = False
    acceleration: bool = False
    attitude: bool = False
    body_rate: bool = False


@dataclass(frozen=True)
class TrajectorySetpointSpec:
    position: tuple[float, float, float] | None = None
    velocity: tuple[float, float, float] | None = None
    acceleration: tuple[float, float, float] | None = None
    yaw: float = 0.0
    yawspeed: float = 0.0


@dataclass(frozen=True)
class VehicleCommandSpec:
    command: int
    param1: float = 0.0
    param2: float = 0.0
    param3: float = 0.0
    param4: float = 0.0
    param5: float = 0.0
    param6: float = 0.0
    param7: float = 0.0
    target_system: int = 3
    target_component: int = 1
    source_system: int = 1
    source_component: int = 1
    from_external: bool = True


@dataclass(frozen=True)
class ControlPlan:
    """Pure-codec control output; node.py converts it to ROS 2 messages."""

    offboard: OffboardControlSpec
    trajectory: TrajectorySetpointSpec
    commands: tuple[VehicleCommandSpec, ...]


def _command(
    command: int,
    *,
    param1: float = 0.0,
    param2: float = 0.0,
    target_system: int,
    source_system: int,
    source_component: int,
    target_component: int,
) -> VehicleCommandSpec:
    return VehicleCommandSpec(
        command=command,
        param1=param1,
        param2=param2,
        target_system=target_system,
        target_component=target_component,
        source_system=source_system,
        source_component=source_component,
    )


def build_control_plan(
    action: str,
    *,
    target: tuple[float, float, float] | None = None,
    velocity: tuple[float, float, float] | None = None,
    yaw: float = 0.0,
    takeoff_altitude_m: float = 2.5,
    current_position: tuple[float, float, float] | None = None,
    target_system: int = 3,
    target_component: int = 1,
    source_system: int = 1,
    source_component: int = 1,
) -> ControlPlan:
    """Map a validated high-level action to pure PX4 control specs.

    All positions are expected in PX4 NED.  The caller is responsible for
    converting an upstream source frame before calling this function.
    """
    offboard = OffboardControlSpec(position=True)
    trajectory = TrajectorySetpointSpec(position=current_position, yaw=yaw)
    commands: list[VehicleCommandSpec] = []

    def cmd(command: int, param1: float = 0.0, param2: float = 0.0) -> VehicleCommandSpec:
        return _command(
            command,
            param1=param1,
            param2=param2,
            target_system=target_system,
            target_component=target_component,
            source_system=source_system,
            source_component=source_component,
        )

    if action == "arm":
        commands.extend(
            [
                cmd(VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0),
                cmd(VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0),
            ]
        )
    elif action == "takeoff":
        altitude = abs(takeoff_altitude_m)
        if target is not None:
            altitude = abs(target[2])
        trajectory = TrajectorySetpointSpec(
            position=(0.0, 0.0, -altitude),
            yaw=yaw,
        )
        commands.extend(
            [
                cmd(VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0),
                cmd(VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0),
            ]
        )
    elif action == "move_to":
        if target is None:
            raise ValueError("move_to requires a target position")
        trajectory = TrajectorySetpointSpec(position=target, velocity=velocity, yaw=yaw)
        commands.append(cmd(VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0))
    elif action == "velocity":
        if velocity is None:
            raise ValueError("velocity requires a velocity vector")
        offboard = OffboardControlSpec(position=False, velocity=True)
        trajectory = TrajectorySetpointSpec(velocity=velocity, yaw=yaw)
        commands.append(cmd(VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0))
    elif action == "hold":
        if current_position is None:
            raise ValueError("hold requires a current position")
        trajectory = TrajectorySetpointSpec(position=current_position, yaw=yaw)
    elif action == "land":
        commands.append(cmd(VEHICLE_CMD_NAV_LAND))
    elif action == "rtl":
        commands.append(cmd(VEHICLE_CMD_NAV_RETURN_TO_LAUNCH))
    else:
        raise ValueError(f"unsupported action: {action!r}")

    return ControlPlan(offboard=offboard, trajectory=trajectory, commands=tuple(commands))
