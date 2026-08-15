"""Compact LLM decision -> full RecoveryPlan expansion."""

from __future__ import annotations

import math
from typing import Any

from pydantic import ValidationError

from swarm.interfaces import MissionDecision, RecoveryCommand, RecoveryPlan
from swarm.recovery.llm import RecoveryContext, _distance_2d


def parse_decision(raw: Any) -> tuple[MissionDecision | None, list[str]]:
    """Validate a compact LLM decision object."""
    if not isinstance(raw, dict):
        return None, ["decision must be a JSON object"]
    try:
        decision = MissionDecision.model_validate(raw)
    except ValidationError as exc:
        errors = [
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}"
            for e in exc.errors()
        ]
        return None, errors
    return decision, []


def semantic_errors(decision: MissionDecision, context: RecoveryContext) -> list[str]:
    """Semantic validity: every referenced drone must be an active agent."""
    active = set(context.snapshots)
    errors: list[str] = []
    for field in ("agent", "high", "low"):
        value = getattr(decision, field)
        if value is not None and value not in active:
            errors.append(f"{field} {value} is not an active agent")

    change = context.mission_change or {}
    if change.get("kind") == "fail_drone":
        failed = int(change["drone"])
        if decision.agent == failed:
            errors.append("cannot reassign the failed drone to itself")
    return errors


def expand_decision(
    decision: MissionDecision,
    context: RecoveryContext,
    arena: tuple[float, float, float, float] = (-6.0, 6.0, -6.0, 6.0),
) -> RecoveryPlan:
    """Deterministically expand a compact decision into a RecoveryPlan."""
    ts = context.timestamp_ms
    change = context.mission_change or {}
    kind = change.get("kind")

    if kind == "fail_drone" and decision.action == "REASSIGN":
        failed = int(change["drone"])
        healthy = (
            decision.agent
            if decision.agent is not None
            else _nearest_healthy(context, failed)
        )
        target = context.base_goals[failed]
        commands = [
            _cmd(failed, "ABORT", None, ts, "abort"),
            _cmd(healthy, "REASSIGN", target, ts, "reassign"),
        ]
        intent = f"reassign task of failed drone {failed} to drone {healthy}"

    elif kind == "block_corridor" and decision.action == "REROUTE":
        drone = decision.agent if decision.agent is not None else int(change["drone"])
        waypoint = _corridor_waypoint(change["zone"], context.base_goals[drone], arena)
        commands = [_cmd(drone, "REROUTE", waypoint, ts, "reroute", ttl_sec=5.0)]
        intent = f"reroute drone {drone} around blocked corridor"

    elif kind == "priority_change" and decision.action == "CHANGE_PRIORITY":
        high = decision.high if decision.high is not None else int(change["high"])
        low = decision.low if decision.low is not None else int(change["low"])
        commands = [
            _cmd(high, "CHANGE_PRIORITY", None, ts, "prio", priority="safety"),
            _cmd(low, "YIELD", None, ts, "yield", ttl_sec=2.0),
        ]
        intent = f"prioritize drone {high} over drone {low}"

    else:
        if decision.agent is None:
            raise ValueError(f"agent required for action {decision.action}")
        drone = decision.agent
        if decision.action in {"HOLD", "ABORT", "RETURN"}:
            commands = [_cmd(drone, decision.action, None, ts, decision.action.lower())]
            intent = f"{decision.action} for drone {drone}"
        elif decision.action == "YIELD":
            commands = [_cmd(drone, "YIELD", None, ts, "yield", ttl_sec=2.0)]
            intent = f"yield drone {drone}"
        else:
            raise ValueError(
                f"action {decision.action} requires a mission_change context"
            )

    return RecoveryPlan(
        schema_version=1,
        intent_text=intent,
        commands=commands,
        constraints=None,
        rationale="expanded from compact LLM decision",
        timestamp_ms=ts,
    )


def _cmd(
    drone: int,
    action: str,
    waypoint: tuple[float, float, float] | None,
    ts: int,
    tag: str,
    *,
    priority: str = "normal",
    ttl_sec: float = 1.0,
) -> RecoveryCommand:
    return RecoveryCommand(
        drone=drone,
        action=action,  # type: ignore[arg-type]
        waypoint=waypoint,
        priority=priority,  # type: ignore[arg-type]
        ttl_sec=ttl_sec,
        command_id=f"{tag}-{ts}-{drone}",
    )


def _nearest_healthy(context: RecoveryContext, failed: int) -> int:
    target = context.base_goals[failed]
    return min(
        (d for d in context.snapshots if d != failed),
        key=lambda d: _distance_2d(context.snapshots[d].position, target),
    )


def _corridor_waypoint(
    zone: list[float],
    goal: tuple[float, ...],
    arena: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    x0, x1, y0, y1 = zone
    corners = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    corner = min(corners, key=lambda p: _distance_2d(p, goal))
    wx = corner[0] + math.copysign(2.0, corner[0] - cx)
    wy = corner[1] + math.copysign(2.0, corner[1] - cy)
    ax0, ax1, ay0, ay1 = arena
    return (
        float(min(max(wx, ax0), ax1)),
        float(min(max(wy, ay0), ay1)),
        0.0,
    )
