#!/usr/bin/env python3
"""N independent long prefixes: cold prefill, then a warmed concurrent decode.

This separates prefill admission/queueing from N-active decode. It writes a
timestamped JSON report under evidence/.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from openai_probe import (  # noqa: E402
    build_long_prompt,
    chat,
    message_text,
    usage,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = "http://127.0.0.1:8013"
DEFAULT_MODEL = "dsv41-flash-1017"


def metric_values(base: str) -> dict:
    with urllib.request.urlopen(f"{base}/metrics", timeout=5) as resp:
        text = resp.read().decode("utf-8", "replace")
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line.startswith("sglang:") or line.startswith("#"):
            continue
        name = line.split("{", 1)[0]
        try:
            value = float(line.rsplit(" ", 1)[1])
        except (IndexError, ValueError):
            continue
        out[name] = max(out.get(name, float("-inf")), value)
    return out


def cold_request(base: str, model: str, prompt: str) -> dict:
    resp, elapsed = chat(
        base,
        model,
        [{"role": "user", "content": prompt}],
        max_tokens=64,
        temperature=0.0,
        reasoning_effort="none",
        timeout=3600.0,
    )
    content, _, _ = message_text(resp)
    return {
        "elapsed_s": round(elapsed, 3),
        "prompt_tokens": usage(resp).get("prompt_tokens"),
        "content": content,
    }


def warm_request(base: str, model: str, prompt: str, output_tokens: int) -> dict:
    resp, elapsed = chat(
        base,
        model,
        [{"role": "user", "content": prompt}],
        max_tokens=output_tokens,
        temperature=0.0,
        reasoning_effort="none",
        extra={"ignore_eos": True},
        timeout=3600.0,
    )
    content, _, _ = message_text(resp)
    u = usage(resp)
    completion = u.get("completion_tokens") or 0
    return {
        "elapsed_s": round(elapsed, 3),
        "prompt_tokens": u.get("prompt_tokens"),
        "completion_tokens": completion,
        "tok_s": round(completion / elapsed, 3) if elapsed else None,
        "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "content_head": content[:160],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=DEFAULT_BASE)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--target-tokens", type=int, default=400000)
    ap.add_argument("--output-tokens", type=int, default=512)
    args = ap.parse_args()

    prepared = [
        build_long_prompt(args.base_url, args.target_tokens)
        for _ in range(args.concurrency)
    ]
    report: dict = {
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "target_tokens": args.target_tokens,
        "concurrency": args.concurrency,
        "output_tokens": args.output_tokens,
        "prompts": [
            {"codes": codes, "chars": len(prompt)} for prompt, codes in prepared
        ],
    }

    # Cold pass: establish N independent prefixes.
    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        cold = list(
            pool.map(
                lambda p: cold_request(args.base_url, args.model, p),
                [p for p, _ in prepared],
            )
        )
    report["cold"] = {
        "batch_wall_s": round(time.monotonic() - start, 3),
        "results": cold,
        "all_prompt_tokens": [r["prompt_tokens"] for r in cold],
        "all_recalled": all(
            all(
                c in r["content"]
                for c in (codes["start_code"], codes["middle_code"], codes["end_code"])
            )
            for r, (_, codes) in zip(cold, prepared)
        ),
    }

    # Warm pass: same prompts, forced output, sample live gauges.
    samples: list[dict] = []
    stop = threading.Event()

    def sampler() -> None:
        while not stop.is_set():
            try:
                samples.append(
                    {"t": time.monotonic(), **metric_values(args.base_url)}
                )
            except Exception as exc:  # keep the load result even if metrics fail
                samples.append({"t": time.monotonic(), "error": str(exc)})
            time.sleep(0.5)

    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()
    start = time.monotonic()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            warm = list(
                pool.map(
                    lambda p: warm_request(
                        args.base_url, args.model, p, args.output_tokens
                    ),
                    [p for p, _ in prepared],
                )
            )
    finally:
        stop.set()
        thread.join(timeout=5)
    wall = time.monotonic() - start

    rates = [r["tok_s"] for r in warm if r["tok_s"]]
    report["warm"] = {
        "batch_wall_s": round(wall, 3),
        "results": warm,
        "aggregate_tok_s": round(sum(rates), 3),
        "median_tok_s": round(statistics.median(rates), 3) if rates else None,
    }

    def peak(name: str) -> float | None:
        vals = [s[name] for s in samples if name in s]
        return max(vals) if vals else None

    report["gauges"] = {
        "samples": len(samples),
        "peak_running": peak("sglang:num_running_reqs"),
        "peak_queue": peak("sglang:num_queue_reqs"),
        "peak_retracted": peak("sglang:num_retracted_reqs"),
        "peak_token_usage": peak("sglang:token_usage"),
        "peak_hicache_host_used_tokens": peak("sglang:hicache_host_used_tokens"),
        "hicache_host_total_tokens": peak("sglang:hicache_host_total_tokens"),
    }

    out = ROOT / "evidence" / (
        f"long-{args.target_tokens}-c{args.concurrency}-warm-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
