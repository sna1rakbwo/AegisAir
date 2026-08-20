"""Schema + whitelist + safety invariant validation for LLM recovery plans."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from swarm.interfaces import RecoveryAction, RecoveryPlan


# Fields the LLM must never be able to touch.  ``RecoveryPlan`` uses
# ``extra="forbid"`` so these are already rejected by the schema; this set is a
# defense-in-depth check that fails loudly for legacy/payload-carrying keys.
FORBIDDEN_FIELDS = {
    "d0",
    "cbf",
    "cbf_constraint",
    "safety_gate_threshold",
    "safety_threshold",
    "actuator_safety_limit",
    "uncertainty_confidence_factor",
    "rho_warn",
    "rho_pred_threshold",
}

WAYPOINT_REQUIRED_ACTIONS = {"REROUTE", "REASSIGN"}


@dataclass
class ValidationResult:
    plan: RecoveryPlan | None
    valid: bool
    errors: list[str] = field(default_factory=list)


class RecoveryValidator:
    """Validate raw LLM output against the frozen RecoveryPlan interface."""

    def __init__(
        self,
        *,
        known_drones: set[int] | None = None,
        arena: tuple[float, float, float, float] = (-6.0, 6.0, -6.0, 6.0),
        max_ttl_sec: float = 10.0,
        min_distance_m: float = 0.5,
        now_ms: int = 0,
    ) -> None:
        self.known_drones = known_drones
        self.arena = arena
        self.max_ttl_sec = max_ttl_sec
        self.min_distance_m = min_distance_m
        self.now_ms = now_ms

    def validate(self, payload: Any, *, now_ms: int | None = None) -> ValidationResult:
        errors: list[str] = []
        if not isinstance(payload, dict):
            return ValidationResult(plan=None, valid=False, errors=["payload must be a JSON object"])

        for key in FORBIDDEN_FIELDS:
            if key in payload:
                errors.append(f"forbidden field '{key}' present")

        if "timestamp_ms" not in payload:
            payload = dict(payload)
            payload["timestamp_ms"] = self.now_ms

        try:
            plan = RecoveryPlan.model_validate(payload)
        except ValidationError as exc:
            errors.extend(_flatten_validation_errors(exc))
            return ValidationResult(plan=None, valid=False, errors=errors)

        effective_now_ms = self.now_ms if now_ms is None else now_ms

        for command in plan.commands:
            if command.action not in RecoveryAction.__args__:  # type: ignore[attr-defined]
                errors.append(f"illegal action '{command.action}'")

            if self.known_drones is not None and command.drone not in self.known_drones:
                errors.append(f"unknown drone {command.drone}")

            if command.action in WAYPOINT_REQUIRED_ACTIONS and command.waypoint is None:
                errors.append(f"{command.action} requires a waypoint")

            if command.waypoint is not None:
                if not _waypoint_in_arena(command.waypoint, self.arena):
                    errors.append(f"waypoint {command.waypoint} outside arena")

            if command.ttl_sec <= 0 or command.ttl_sec > self.max_ttl_sec:
                errors.append(
                    f"ttl_sec {command.ttl_sec} outside (0, {self.max_ttl_sec}]"
                )
            if effective_now_ms > plan.timestamp_ms + int(command.ttl_sec * 1000):
                errors.append(f"expired recovery command '{command.command_id}'")

        if plan.constraints is not None:
            if (
                plan.constraints.keep_min_distance_m is not None
                and plan.constraints.keep_min_distance_m < self.min_distance_m
            ):
                errors.append(
                    "keep_min_distance_m below hard safety floor "
                    f"{self.min_distance_m}"
                )

        if errors:
            return ValidationResult(plan=None, valid=False, errors=errors)
        return ValidationResult(plan=plan, valid=True, errors=[])


def _flatten_validation_errors(exc: ValidationError) -> list[str]:
    errors: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err["loc"])
        errors.append(f"{loc}: {err['msg']}")
    return errors


def _waypoint_in_arena(
    waypoint: tuple[float, float, float],
    arena: tuple[float, float, float, float],
) -> bool:
    x0, x1, y0, y1 = arena
    return x0 <= waypoint[0] <= x1 and y0 <= waypoint[1] <= y1
