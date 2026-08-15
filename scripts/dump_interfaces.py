#!/usr/bin/env python3
"""Regenerate JSON Schema artifacts from the frozen pydantic interfaces."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from swarm.interfaces import (
    LogEvent,
    MarlAction,
    Observation,
    RecoveryPlan,
    RiskEvent,
    SafetyDecision,
    Telemetry,
)


SCHEMA_DIR = ROOT / "schemas" / "interfaces"

MODELS = {
    "telemetry": Telemetry,
    "observation": Observation,
    "marl_action": MarlAction,
    "risk_event": RiskEvent,
    "recovery_plan": RecoveryPlan,
    "safety_decision": SafetyDecision,
    "log_event": LogEvent,
}


def main() -> int:
    SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
    for name, model in MODELS.items():
        schema = model.model_json_schema()
        out = SCHEMA_DIR / f"{name}.schema.json"
        out.write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
