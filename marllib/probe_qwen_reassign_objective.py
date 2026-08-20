#!/usr/bin/env python3
"""Probe whether Qwen picks the optimal reassignment when told the objective.

Same context as ``probe_qwen_reassign``, but the system prompt explicitly asks
to minimize total remaining path length across healthy drones.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


OBJECTIVE_PROMPT = (
    "You are the Semantic Mission Manager for a multi-UAV mission. "
    "A drone failed; its goal is now orphaned and must be covered by exactly "
    "one healthy drone. Choose which healthy drone takes over so that the "
    "TOTAL REMAINING PATH LENGTH across all healthy drones is minimized. "
    "Account for each healthy drone's own goal distance: a drone whose own "
    "goal is far is cheaper to reassign; a drone whose own goal is near should "
    "stay on its own goal. "
    "Return only one compact JSON object: {'action': 'REASSIGN', 'agent': <id>}. "
    "Do not include reasoning or markdown; output JSON only."
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe Qwen reassign with objective")
    parser.add_argument(
        "--qwen-model",
        default="/Users/lijiajun/.cache/aegisair-qwen3-4b-4bit-bench",
    )
    parser.add_argument("--n", type=int, default=3)
    args = parser.parse_args()

    from mlx_lm import load, stream_generate

    model, tokenizer = load(args.qwen_model)
    user_payload = {
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
        "failed_drone": 0,
    }
    for i in range(args.n):
        messages = [
            {"role": "system", "content": OBJECTIVE_PROMPT},
            {"role": "user", "content": json.dumps(user_payload)},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        text = ""
        for response in stream_generate(
            model, tokenizer, prompt=prompt, max_tokens=48
        ):
            text += response.text
        print(f"probe={i + 1} output={text!r}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
