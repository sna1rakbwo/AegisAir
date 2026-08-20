#!/usr/bin/env python3
"""Probe whether Qwen respects a priority constraint when reassigning.

Drone 0 fails.  Drone 1 is nearest to the orphan but has priority "critical"
(must not be diverted); drone 2 is farther and normal.  The correct answer is
agent=2.  The frozen rule planner ignores priority and would pick agent=1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PROMPT = (
    "You are the Semantic Mission Manager for a multi-UAV mission. "
    "A drone failed and its goal is orphaned; choose ONE healthy drone to take "
    "over the orphaned goal (action REASSIGN, field agent = that drone). "
    "Constraint: a drone whose priority is 'critical' or 'safety' must keep its "
    "own mission and MUST NOT be reassigned. Prefer a normal-priority drone. "
    "Return only one compact JSON object, e.g. {'action': 'REASSIGN', 'agent': 2}. "
    "Do not include reasoning or markdown; output JSON only."
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe Qwen priority reassign")
    parser.add_argument(
        "--qwen-model",
        default="/Users/lijiajun/.cache/aegisair-qwen3-4b-4bit-bench",
    )
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()

    from mlx_lm import load, stream_generate

    model, tokenizer = load(args.qwen_model)
    payload = {
        "mission_change": {"kind": "fail_drone", "drone": 0},
        "positions": {
            "0": [-5.0, 1.0, 0.0],
            "1": [0.0, 0.0, 0.0],
            "2": [0.0, 4.0, 0.0],
        },
        "goals": {
            "0": [1.0, 0.0, 0.0],
            "1": [2.0, 0.0, 0.0],
            "2": [5.0, -4.0, 0.0],
        },
        "priorities": {"0": "normal", "1": "critical", "2": "normal"},
    }
    for i in range(args.n):
        messages = [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": json.dumps(payload)},
        ]
        prompt_tokens = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        text = ""
        for response in stream_generate(
            model, tokenizer, prompt=prompt_tokens, max_tokens=48
        ):
            text += response.text
        print(f"probe={i + 1} output={text!r}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
