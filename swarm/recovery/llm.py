"""Local semantic-recovery LLM backends.

The client is a swappable backend so the whole Phase 4 pipeline can be verified
without a large model present.  :class:`DeterministicRecoveryClient` is the
default safe backend and is also the fallback when a real local model times
out.  :class:`TransformersQwenClient` is the real local-model backend.

4-bit quantisation on this Apple-silicon conda environment is intentionally a
hard failure rather than a silent fallback: ``bitsandbytes`` (CUDA) and
``mlx-lm`` (Metal) are not installed, so we refuse to pretend a model is 4-bit.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any

from swarm.geometry import Vector3
from swarm.interfaces import RecoveryCommand, RecoveryPlan
from swarm.safety import DroneSnapshot


@dataclass(frozen=True)
class RecoveryContext:
    """Everything the recovery client needs to produce a structured plan."""

    event: str
    agent_i: int
    agent_j: int
    current_margin: float
    predicted_min_margin: float | None
    margin_degradation: float | None
    intervention_count: int
    cause: str
    severity: str
    snapshots: dict[int, DroneSnapshot]
    current_goals: dict[int, Vector3]
    base_goals: dict[int, Vector3]
    priorities: dict[int, str]
    timestamp_ms: int
    mission_change: dict[str, Any] | None = None


@dataclass
class LLMRecoveryResult:
    """Normalised result from any recovery backend."""

    plan: RecoveryPlan | None
    raw: dict[str, Any] | None
    latency_s: float
    timeout: bool
    valid: bool
    errors: list[str]
    backend: str


class RecoveryLLMError(RuntimeError):
    """Raised when a local LLM backend cannot be constructed or used."""


RECOVERY_SYSTEM_PROMPT = (
    "You are the Semantic Mission Manager for a multi-UAV mission. "
    "When the current mission plan becomes invalid under safety, environment, or "
    "coordination changes, replan at the mission level. "
    "Return only one compact JSON object with fields 'action' and optional "
    "'agent', 'high', 'low'. Do not output waypoints, ttl, command_id, or "
    "schema fields; those are filled deterministically. "
    "Whitelisted actions: HOLD, YIELD, REROUTE, REASSIGN, "
    "CHANGE_PRIORITY, ABORT, RETURN. "
    "For HOLD/YIELD/REROUTE/ABORT/RETURN set 'agent' to the affected drone id. "
    "For REASSIGN when a drone failed, set 'agent' to the healthy drone that "
    "takes over the failed drone's task. HARD CONSTRAINT: never reassign a "
    "drone whose priority is 'critical' or 'safety' — such a drone must keep "
    "its own mission; choose a normal-priority healthy drone instead. "
    "For CHANGE_PRIORITY set 'high' and 'low' to drone ids. "
    "If mission_change.kind is 'fail_drone', respond with action REASSIGN. "
    "If mission_change.kind is 'block_corridor', respond with action REROUTE. "
    "If mission_change.kind is 'priority_change', respond with action CHANGE_PRIORITY. "
    "If cause is 'COORDINATION_DEGRADATION' (a symmetric deadlock with no "
    "goal progress), set the optional 'priority_order' to the right-of-way "
    "order according to mission semantics (urgent missions first), and choose "
    "action REROUTE or YIELD for the affected drones; never emit velocity or "
    "acceleration commands. "
    "Never modify d0, CBF constraints, safety thresholds, or actuator limits. "
    "Never output velocity, acceleration, or turn commands. "
    "Do not include reasoning or markdown; output JSON only."
)


def _context_payload(context: RecoveryContext) -> dict[str, Any]:
    return {
        "event": context.event,
        "agent_i": context.agent_i,
        "agent_j": context.agent_j,
        "current_margin": context.current_margin,
        "predicted_min_margin": context.predicted_min_margin,
        "margin_degradation": context.margin_degradation,
        "intervention_count": context.intervention_count,
        "cause": context.cause,
        "severity": context.severity,
        "priorities": context.priorities,
        "current_goals": context.current_goals,
        "positions": {
            i: list(snap.position) for i, snap in context.snapshots.items()
        },
        "mission_change": context.mission_change,
    }


class LLMRecoveryClient:
    """Protocol implemented by every semantic-recovery backend."""

    name: str = "base"

    def generate(self, context: RecoveryContext) -> LLMRecoveryResult:
        raise NotImplementedError


def _lateral_waypoint(
    actor_pos: Vector3,
    other_pos: Vector3,
    actor_goal: Vector3,
    offset_m: float = 2.0,
    arena: tuple[float, float, float, float] = (-6.0, 6.0, -6.0, 6.0),
) -> Vector3:
    """A short detour perpendicular to the actor->other direction."""
    dx = other_pos[0] - actor_pos[0]
    dy = other_pos[1] - actor_pos[1]
    length = math.hypot(dx, dy)
    if length == 0:
        perp_x, perp_y = 0.0, 1.0
    else:
        perp_x, perp_y = -dy / length, dx / length
    # Prefer the side that keeps the detour roughly toward the goal.
    toward_goal_x = actor_goal[0] - actor_pos[0]
    toward_goal_y = actor_goal[1] - actor_pos[1]
    if perp_x * toward_goal_x + perp_y * toward_goal_y < 0:
        perp_x, perp_y = -perp_x, -perp_y
    x0, x1, y0, y1 = arena
    waypoint = (
        float(min(max(actor_pos[0] + perp_x * offset_m, x0), x1)),
        float(min(max(actor_pos[1] + perp_y * offset_m, y0), y1)),
        0.0,
    )
    return waypoint


class DeterministicRecoveryClient(LLMRecoveryClient):
    """Deterministic, rule-based stand-in for the local LLM.

    It emits the same whitelisted ``REROUTE`` mission change the local model
    would be asked to produce, so R0/R1/R2 differ only in the trigger policy
    rather than the backend.  ``plan_latency_s`` is optional and lets a smoke
    test exercise the timeout/fallback path without loading a model.
    """

    name = "deterministic"

    def __init__(
        self,
        *,
        arena: tuple[float, float, float, float] = (-6.0, 6.0, -6.0, 6.0),
        plan_latency_s: float = 0.0,
    ) -> None:
        self.arena = arena
        self.plan_latency_s = plan_latency_s

    def _choose_actor(self, context: RecoveryContext) -> int:
        """Lower-priority drone yields; ties break toward the higher id."""
        pair = (context.agent_i, context.agent_j)
        order = sorted(
            pair,
            key=lambda drone: (
                0 if context.priorities.get(drone, "normal") == "normal" else 1,
                drone,
            ),
        )
        return order[0]

    def _build_plan(self, context: RecoveryContext) -> RecoveryPlan:
        if context.agent_i == context.agent_j:
            actor = context.agent_i
            command = RecoveryCommand(
                drone=actor,
                action="RETURN",
                waypoint=None,
                priority="normal",
                ttl_sec=1.0,
                command_id=f"det-{context.timestamp_ms}-{actor}",
            )
            return RecoveryPlan(
                schema_version=1,
                intent_text=f"rejoin original mission for drone {actor}",
                commands=[command],
                constraints=None,
                rationale="deterministic return to base goal after mission invalidation",
                timestamp_ms=context.timestamp_ms,
            )

        actor = self._choose_actor(context)
        other = context.agent_i if actor == context.agent_j else context.agent_j
        waypoint = _lateral_waypoint(
            context.snapshots[actor].position,
            context.snapshots[other].position,
            context.base_goals[actor],
            arena=self.arena,
        )
        command_id = f"det-{context.timestamp_ms}-{actor}"
        command = RecoveryCommand(
            drone=actor,
            action="REROUTE",
            waypoint=waypoint,
            priority="normal",
            ttl_sec=1.0,
            command_id=command_id,
        )
        return RecoveryPlan(
            schema_version=1,
            intent_text=f"recover pair {actor}-{other} after {context.event}",
            commands=[command],
            constraints=None,
            rationale=f"deterministic lateral detour for lower-priority drone {actor}",
            timestamp_ms=context.timestamp_ms,
        )

    def generate(self, context: RecoveryContext) -> LLMRecoveryResult:
        started = time.perf_counter()
        if self.plan_latency_s:
            time.sleep(self.plan_latency_s)
        plan = self._build_plan(context)
        return LLMRecoveryResult(
            plan=plan,
            raw=plan.model_dump(mode="json"),
            latency_s=time.perf_counter() - started,
            timeout=False,
            valid=True,
            errors=[],
            backend=self.name,
        )


class TransformersQwenClient(LLMRecoveryClient):
    """Local Qwen3-4B (or Qwen-family) backend via ``transformers``.

    This is best-effort constrained generation: a non-thinking prompt plus
    greedy decoding, followed by strict schema validation.  Grammar-constrained
    decoding (outlines/vLLM) is intentionally not used, matching plan.md's
    "Prompt + JSON Schema + Whitelist + Validator" first version.
    """

    name = "transformers-qwen"

    def __init__(
        self,
        *,
        model_id: str,
        device: str = "auto",
        load_in_4bit: bool = False,
        max_new_tokens: int = 256,
        load: bool = True,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None
        if load:
            self.load()

    def load(self) -> None:
        if self.load_in_4bit:
            try:
                import bitsandbytes  # noqa: F401
            except ImportError as exc:
                raise RecoveryLLMError(
                    "4-bit quantization in TransformersQwenClient requires "
                    "bitsandbytes (CUDA), which is not available in this env. "
                    "On Apple Silicon use MlxLmClient for the 4-bit path."
                ) from exc

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RecoveryLLMError("transformers/torch are required") from exc

        resolve_device = self.device
        if resolve_device == "auto":
            resolve_device = "mps" if torch.backends.mps.is_available() else "cpu"

        dtype = torch.float16 if resolve_device == "mps" else torch.float32
        kwargs: dict[str, Any] = {"torch_dtype": dtype}
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        else:
            kwargs["device_map"] = resolve_device

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        self._model.eval()

    def _build_prompt(self, context: RecoveryContext) -> str:
        return (
            RECOVERY_SYSTEM_PROMPT
            + "\nContext:\n"
            + json.dumps(_context_payload(context), default=str)
            + "\nReturn JSON:\n"
        )

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        if not text:
            return None
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match is None:
            return None
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None

    def generate(self, context: RecoveryContext) -> LLMRecoveryResult:
        if self._model is None or self._tokenizer is None:
            self.load()
        started = time.perf_counter()
        prompt = self._build_prompt(context)
        inputs = self._tokenizer(prompt, return_tensors="pt")
        if self.device == "mps" or self.device == "auto":
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        with __import__("torch").no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        text = self._tokenizer.decode(output[0], skip_special_tokens=True)
        raw = self._extract_json(text)
        return LLMRecoveryResult(
            plan=None,
            raw=raw,
            latency_s=time.perf_counter() - started,
            timeout=False,
            valid=raw is not None,
            errors=[] if raw is not None else ["no JSON object in model output"],
            backend=self.name,
        )


class MlxLmClient(TransformersQwenClient):
    """Local Qwen backend via MLX (the Apple-silicon 4-bit path).

    Point ``model_id`` at a pre-quantised MLX repository (for example
    ``mlx-community/Qwen3-4B-4bit``) or a local directory produced by
    ``mlx_lm.convert --quantize``.  This reuses the same non-thinking prompt
    and JSON extraction as :class:`TransformersQwenClient`.
    """

    name = "mlx-lm-qwen"

    def __init__(
        self,
        *,
        model_id: str,
        max_tokens: int = 256,
        load: bool = True,
        load_options: dict[str, Any] | None = None,
    ) -> None:
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.load_options = load_options or {}
        self._model = None
        self._tokenizer = None
        if load:
            self.load()

    def load(self) -> None:
        try:
            from mlx_lm import load as mlx_load
        except ImportError as exc:
            raise RecoveryLLMError("mlx-lm is required for the MLX backend") from exc
        self._model, self._tokenizer = mlx_load(self.model_id, **self.load_options)

    def generate(self, context: RecoveryContext) -> LLMRecoveryResult:
        if self._model is None or self._tokenizer is None:
            self.load()
        started = time.perf_counter()
        messages = [
            {"role": "system", "content": RECOVERY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(_context_payload(context), default=str),
            },
        ]
        prompt_tokens = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        try:
            from mlx_lm import stream_generate
        except ImportError as exc:
            raise RecoveryLLMError("mlx-lm is required for the MLX backend") from exc
        text = ""
        for response in stream_generate(
            self._model,
            self._tokenizer,
            prompt=prompt_tokens,
            max_tokens=self.max_tokens,
        ):
            text += response.text
        raw = self._extract_json(text)
        return LLMRecoveryResult(
            plan=None,
            raw=raw,
            latency_s=time.perf_counter() - started,
            timeout=False,
            valid=raw is not None,
            errors=[] if raw is not None else ["no JSON object in model output"],
            backend=self.name,
        )


class RuleMissionPlanner(DeterministicRecoveryClient):
    """Deterministic semantic mission planner for the three Phase-7 scenarios.

    This is a stand-in for the local LLM: it turns a structured mission change
    into the correct whitelisted high-level actions (REROUTE / REASSIGN /
    CHANGE_PRIORITY / ABORT) without ever emitting velocity commands.
    """

    name = "rule-mission-planner"

    def _build_plan(self, context: RecoveryContext) -> RecoveryPlan:
        change = context.mission_change
        if change and change.get("kind") == "fail_drone":
            return self._fail_drone_plan(context, change)
        if change and change.get("kind") == "block_corridor":
            return self._block_corridor_plan(context, change)
        if change and change.get("kind") == "priority_change":
            return self._priority_change_plan(context, change)
        if context.cause == "COORDINATION_DEGRADATION":
            return self._coordination_degradation_plan(context)
        return super()._build_plan(context)

    def _coordination_degradation_plan(
        self, context: RecoveryContext
    ) -> RecoveryPlan:
        """Deterministic deadlock breaker for a symmetric stand-off.

        The stalled drone yields briefly and reroutes to a lateral waypoint
        whose sign depends on the drone id.  This breaks the symmetric
        "everyone waits, nobody moves" state while HOCBF keeps the maneuver
        collision-free.
        """
        drone = context.agent_i
        snap = context.snapshots[drone]
        goal = context.base_goals[drone]
        delta_x = goal[0] - snap.position[0]
        delta_y = goal[1] - snap.position[1]
        length = math.hypot(delta_x, delta_y)
        if length < 1e-6:
            perp_x, perp_y = 0.0, 1.0
        else:
            ux, uy = delta_x / length, delta_y / length
            perp_x, perp_y = -uy, ux
        sign = 1.0 if drone % 2 == 0 else -1.0
        waypoint = (
            snap.position[0] + perp_x * sign * 2.0,
            snap.position[1] + perp_y * sign * 2.0,
            0.0,
        )
        commands = [
            RecoveryCommand(
                drone=drone,
                action="YIELD",
                waypoint=None,
                priority="normal",
                ttl_sec=2.0,
                command_id=f"yield-{context.timestamp_ms}-{drone}",
            ),
            RecoveryCommand(
                drone=drone,
                action="REROUTE",
                waypoint=waypoint,
                priority="normal",
                ttl_sec=4.0,
                command_id=f"reroute-{context.timestamp_ms}-{drone}",
            ),
        ]
        return RecoveryPlan(
            schema_version=1,
            intent_text=f"break symmetric deadlock for drone {drone}",
            commands=commands,
            constraints=None,
            rationale="deterministic lateral yield for coordination degradation",
            timestamp_ms=context.timestamp_ms,
        )

    def _fail_drone_plan(
        self, context: RecoveryContext, change: dict[str, Any]
    ) -> RecoveryPlan:
        failed = int(change["drone"])
        target = context.base_goals[failed]
        healthy = min(
            (d for d in context.snapshots if d != failed),
            key=lambda d: _distance_2d(context.snapshots[d].position, target),
        )
        commands = [
            RecoveryCommand(
                drone=failed,
                action="ABORT",
                waypoint=None,
                priority="normal",
                ttl_sec=1.0,
                command_id=f"abort-{context.timestamp_ms}-{failed}",
            ),
            RecoveryCommand(
                drone=healthy,
                action="REASSIGN",
                waypoint=target,
                priority="normal",
                ttl_sec=1.0,
                command_id=f"reassign-{context.timestamp_ms}-{healthy}",
            ),
        ]
        return RecoveryPlan(
            schema_version=1,
            intent_text=f"reassign task of failed drone {failed} to drone {healthy}",
            commands=commands,
            constraints=None,
            rationale=f"drone {failed} failed; nearest healthy drone {healthy} takes over",
            timestamp_ms=context.timestamp_ms,
        )

    def _block_corridor_plan(
        self, context: RecoveryContext, change: dict[str, Any]
    ) -> RecoveryPlan:
        drone = int(change["drone"])
        x0, x1, y0, y1 = change["zone"]
        goal = context.base_goals[drone]
        corners = [
            (x0, y0),
            (x1, y0),
            (x0, y1),
            (x1, y1),
        ]
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        corner = min(corners, key=lambda p: _distance_2d(p, goal))
        waypoint = (
            corner[0] + math.copysign(2.0, corner[0] - cx),
            corner[1] + math.copysign(2.0, corner[1] - cy),
            0.0,
        )
        command = RecoveryCommand(
            drone=drone,
            action="REROUTE",
            waypoint=waypoint,
            priority="normal",
            ttl_sec=5.0,
            command_id=f"reroute-{context.timestamp_ms}-{drone}",
        )
        return RecoveryPlan(
            schema_version=1,
            intent_text=f"reroute drone {drone} around blocked corridor",
            commands=[command],
            constraints=None,
            rationale=f"corridor {change['zone']} became unavailable",
            timestamp_ms=context.timestamp_ms,
        )

    def _priority_change_plan(
        self, context: RecoveryContext, change: dict[str, Any]
    ) -> RecoveryPlan:
        high = int(change["high"])
        low = int(change["low"])
        commands = [
            RecoveryCommand(
                drone=high,
                action="CHANGE_PRIORITY",
                waypoint=None,
                priority="safety",
                ttl_sec=1.0,
                command_id=f"prio-{context.timestamp_ms}-{high}",
            ),
            RecoveryCommand(
                drone=low,
                action="YIELD",
                waypoint=None,
                priority="normal",
                ttl_sec=2.0,
                command_id=f"yield-{context.timestamp_ms}-{low}",
            ),
        ]
        return RecoveryPlan(
            schema_version=1,
            intent_text=f"prioritize drone {high} over drone {low}",
            commands=commands,
            constraints=None,
            rationale=f"mission priority change: {high} high, {low} normal",
            timestamp_ms=context.timestamp_ms,
        )


def _distance_2d(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
