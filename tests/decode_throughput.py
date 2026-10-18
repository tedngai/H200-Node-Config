#!/usr/bin/env python3
"""Client-observed decode throughput on natural coding prompts.

Unlike the generic prose `throughput` probe, this uses a fixed set of coding
prompts, natural sampling and no forced EOS. It is intended for A/B comparison
of decode settings (for example a context-length cap) on the same prompts.

Reports real `completion_tokens` per request, wall time, per-request and
aggregate rates, plus live running/queue/retraction gauges. Writes a
timestamped JSON report under evidence/.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from openai_probe import chat, message_text, usage  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = "http://127.0.0.1:8013"
DEFAULT_MODEL = "dsv41-flash-1017"

TOPICS = [
    "an LRU cache with TTL in Python",
    "a token-bucket rate limiter in Go",
    "a trie with prefix search in Rust",
    "a REST client with retries in TypeScript",
    "a thread pool in C++17",
    "a JSON schema validator in Python",
    "a Redis-backed job queue in Node.js",
    "a B-tree insert routine in C",
    "an async web crawler in Python",
    "a Kotlin coroutine-based downloader",
    "a SQL query builder in TypeScript",
    "a Dijkstra router in Java",
    "a ring buffer in Rust",
    "a markdown tokenizer in Go",
    "a CSV streaming parser in Python",
    "an event sourcing aggregate in C#",
]


def prompt_for(index: int) -> str:
    topic = TOPICS[index % len(TOPICS)]
    return (
        f"Write production-quality code for {topic} (variant {index}). "
        "Include type hints or types, docstrings/comments, error handling and "
        "unit tests."
    )


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


def start_sampler(base: str) -> tuple[list[dict], threading.Event, threading.Thread]:
    samples: list[dict] = []
    stop = threading.Event()

    def sampler() -> None:
        start = time.monotonic()
        while not stop.is_set():
            try:
                samples.append(
                    {"t": round(time.monotonic() - start, 3), **metric_values(base)}
                )
            except Exception as exc:  # keep the load running if metrics blip
                samples.append(
                    {"t": round(time.monotonic() - start, 3), "error": str(exc)}
                )
            time.sleep(0.25)

    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()
    return samples, stop, thread


def peak(samples: list[dict], name: str) -> float | None:
    values = [s[name] for s in samples if name in s]
    return max(values) if values else None


def one_request(
    base: str,
    model: str,
    slot: int,
    prompt_index: int,
    output_tokens: int,
    batch_start: float,
    results: list,
) -> None:
    started = time.monotonic()
    try:
        resp, elapsed = chat(
            base,
            model,
            [{"role": "user", "content": prompt_for(prompt_index)}],
            max_tokens=output_tokens,
            temperature=0.0,
            reasoning_effort="none",
            timeout=3600.0,
        )
        content, reasoning, _ = message_text(resp)
        u = usage(resp)
        completion = u.get("completion_tokens") or 0
        results[slot] = {
            "slot": slot,
            "prompt_index": prompt_index,
            "start_offset_s": round(started - batch_start, 3),
            "finish_offset_s": round(time.monotonic() - batch_start, 3),
            "elapsed_s": round(elapsed, 3),
            "prompt_tokens": u.get("prompt_tokens"),
            "completion_tokens": completion,
            "finish_reason": resp["choices"][0].get("finish_reason"),
            "tok_s": round(completion / elapsed, 3) if elapsed else None,
            "content_chars": len(content),
            "reasoning_chars": len(reasoning),
        }
    except Exception as exc:
        results[slot] = {"slot": slot, "prompt_index": prompt_index, "error": str(exc)}


def run_concurrency(
    base: str,
    model: str,
    concurrency: int,
    output_tokens: int,
    round_index: int,
) -> dict:
    results: list = [None] * concurrency
    samples, stop, sampler = start_sampler(base)
    batch_start = time.monotonic()
    threads = [
        threading.Thread(
            target=one_request,
            args=(
                base,
                model,
                i,
                round_index * concurrency + i,
                output_tokens,
                batch_start,
                results,
            ),
        )
        for i in range(concurrency)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall = time.monotonic() - batch_start
    time.sleep(0.3)
    stop.set()
    sampler.join(timeout=2)

    ok = [r for r in results if r and "error" not in r]
    tokens = sum(r["completion_tokens"] for r in ok)
    rates = [r["tok_s"] for r in ok if r["tok_s"] is not None]
    return {
        "concurrency": concurrency,
        "round": round_index + 1,
        "batch_wall_s": round(wall, 3),
        "requests_ok": len(ok),
        "requests_failed": concurrency - len(ok),
        "total_completion_tokens": tokens,
        "aggregate_tok_s": round(tokens / wall, 3) if wall else None,
        "median_request_tok_s": round(statistics.median(rates), 3) if rates else None,
        "min_request_tok_s": round(min(rates), 3) if rates else None,
        "max_request_tok_s": round(max(rates), 3) if rates else None,
        "all_hit_output_cap": all(
            r["completion_tokens"] >= output_tokens for r in ok
        )
        if ok
        else None,
        "peak_running": peak(samples, "sglang:num_running_reqs"),
        "peak_queue": peak(samples, "sglang:num_queue_reqs"),
        "peak_retracted": peak(samples, "sglang:num_retracted_reqs"),
        "results": results,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument(
        "--concurrency",
        default="1,4,16",
        help="Comma-separated concurrency levels, e.g. 1,4,16.",
    )
    ap.add_argument("--output-tokens", type=int, default=1024)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--label", default="decode-throughput")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    levels = [int(x) for x in args.concurrency.split(",") if x.strip()]
    report = {
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "base_url": args.base,
        "model": args.model,
        "mode": args.label,
        "output_tokens": args.output_tokens,
        "temperature": 0.0,
        "reasoning_effort": "none",
        "ignore_eos": False,
        "prompts": [prompt_for(i) for i in range(max(levels) * args.rounds)],
        "runs": [],
    }
    for round_index in range(args.rounds):
        for level in levels:
            run = run_concurrency(
                args.base, args.model, level, args.output_tokens, round_index
            )
            report["runs"].append(run)
            print(
                f"c{level} round {round_index + 1}: "
                f"wall {run['batch_wall_s']:.2f}s, "
                f"aggregate {run['aggregate_tok_s']}, "
                f"median {run['median_request_tok_s']} tok/s, "
                f"running {run['peak_running']}, queue {run['peak_queue']}, "
                f"failed {run['requests_failed']}"
            )

    path = Path(args.report) if args.report else None
    if path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = ROOT / "evidence" / f"{args.label}-{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"report: {path}")
    return 0 if all(r["requests_failed"] == 0 for r in report["runs"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
