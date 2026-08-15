"""Benchmark the local MLX Qwen recovery decision path.

Measures, on a real RiskEvent -> short JSON workload:

    - TTFT (time to first token)
    - generation time (TTFT -> last token)
    - total latency
    - valid-JSON rate

The model is loaded once, warmed, then measured over many consecutive requests
with ``enable_thinking=False`` so the reported numbers reflect steady-state,
non-thinking inference rather than a single cold request.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

WHITELIST = {
    "HOLD",
    "YIELD",
    "REROUTE",
    "REASSIGN",
    "CHANGE_PRIORITY",
    "ABORT",
    "RETURN",
}

SYSTEM_PROMPT = (
    "You are the semantic-recovery commander for a multi-UAV mission. "
    "Given a structured risk event, return only a JSON object with keys "
    '"action" and "agent". "action" must be exactly one of: '
    "HOLD, YIELD, REROUTE, REASSIGN, CHANGE_PRIORITY, ABORT, RETURN. "
    '"agent" must be the UAV id string that should take the action. '
    "Output JSON only. No markdown, no explanation, no reasoning."
)


def _build_messages() -> list[dict[str, str]]:
    event = {
        "event": "CONFIRMED_RUNTIME_RISK",
        "agent_i": "uav_2",
        "agent_j": "uav_4",
        "mission_i": "medical_delivery",
        "mission_j": "inspection",
        "risk_source": "trajectory_conflict",
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(event)},
    ]


def _parse(text: str) -> dict | None:
    try:
        start = text.index("{")
        end = text.rindex("}")
        value = json.loads(text[start : end + 1])
    except (ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _is_valid(text: str) -> bool:
    value = _parse(text)
    if value is None:
        return False
    return (
        value.get("action") in WHITELIST
        and value.get("agent") in {"uav_2", "uav_4"}
    )


def _percentile(samples: list[float], q: float) -> float:
    if not samples:
        return float("nan")
    ordered = sorted(samples)
    idx = min(len(ordered) - 1, int(round(q / 100.0 * (len(ordered) - 1))))
    return ordered[idx]


def _summarize(samples: list[float]) -> dict[str, float]:
    return {
        "p50": _percentile(samples, 50),
        "p90": _percentile(samples, 90),
        "p95": _percentile(samples, 95),
        "mean": statistics.fmean(samples) if samples else float("nan"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/Users/lijiajun/.cache/aegisair-qwen3-4b-4bit-bench",
    )
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=24)
    args = parser.parse_args()

    from mlx_lm import load, stream_generate

    model, tokenizer = load(args.model)
    messages = _build_messages()
    prompt_tokens = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    def run_once() -> tuple[float, float, float, str]:
        t0 = time.perf_counter()
        text = ""
        first_token_t: float | None = None
        last_token_t = t0
        for response in stream_generate(
            model,
            tokenizer,
            prompt=prompt_tokens,
            max_tokens=args.max_tokens,
        ):
            if first_token_t is None:
                first_token_t = time.perf_counter()
            last_token_t = time.perf_counter()
            text += response.text
        total = last_token_t - t0
        ttft = (first_token_t - t0) if first_token_t is not None else total
        generation = total - ttft
        return ttft, generation, total, text

    # Warm up (Metal kernel compilation and lazy weight paging happen here).
    for _ in range(args.warmup):
        run_once()

    ttfts: list[float] = []
    gens: list[float] = []
    totals: list[float] = []
    valid = 0
    samples: list[str] = []
    for _ in range(args.n):
        ttft, gen, total, text = run_once()
        ttfts.append(ttft)
        gens.append(gen)
        totals.append(total)
        samples.append(text)
        valid += int(_is_valid(text))

    result = {
        "model": args.model,
        "warmup": args.warmup,
        "measured_requests": args.n,
        "max_tokens": args.max_tokens,
        "ttft_s": _summarize(ttfts),
        "generation_s": _summarize(gens),
        "total_s": _summarize(totals),
        "valid_json_rate": valid / args.n,
        "sample_outputs": samples[:5],
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
