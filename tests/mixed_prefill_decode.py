#!/usr/bin/env python3
"""Measure warmed decode while fresh long prefills are admitted.

The baseline pass decodes W warmed prefixes with no prefill traffic. The mixed
pass starts the same W warmed decodes, waits until they are running, then
submits C fresh long prompts. The report compares warmed decode rates and
records live running/queue/retraction gauges. It writes a timestamped JSON
report under evidence/.
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


def wait_for_running(base: str, target: int, timeout: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if metric_values(base).get("sglang:num_running_reqs", 0) >= target:
                return True
        except Exception:
            pass
        time.sleep(0.2)
    return False


def request(
    base: str,
    model: str,
    prompt: str,
    *,
    max_tokens: int,
    ignore_eos: bool,
) -> dict:
    extra = {"ignore_eos": True} if ignore_eos else None
    resp, elapsed = chat(
        base,
        model,
        [{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
        reasoning_effort="none",
        extra=extra,
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
        "content": content,
    }


def start_sampler(base: str) -> tuple[list[dict], threading.Event, threading.Thread]:
    samples: list[dict] = []
    stop = threading.Event()

    def sampler() -> None:
        start = time.monotonic()
        while not stop.is_set():
            try:
                samples.append(
                    {
                        "t": round(time.monotonic() - start, 3),
                        **metric_values(base),
                    }
                )
            except Exception as exc:
                samples.append(
                    {"t": round(time.monotonic() - start, 3), "error": str(exc)}
                )
            time.sleep(0.25)

    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()
    return samples, stop, thread


def gauges(samples: list[dict]) -> dict:
    def peak(name: str) -> float | None:
        vals = [s[name] for s in samples if name in s]
        return max(vals) if vals else None

    return {
        "samples": len(samples),
        "peak_running": peak("sglang:num_running_reqs"),
        "peak_queue": peak("sglang:num_queue_reqs"),
        "peak_retracted": peak("sglang:num_retracted_reqs"),
        "peak_token_usage": peak("sglang:token_usage"),
        "peak_full_token_usage": peak("sglang:full_token_usage"),
        "peak_swa_token_usage": peak("sglang:swa_token_usage"),
        "peak_hicache_host_used_tokens": peak("sglang:hicache_host_used_tokens"),
    }


def summarize(results: list[dict]) -> dict:
    rates = [r["tok_s"] for r in results if r["tok_s"]]
    return {
        "aggregate_tok_s": round(sum(rates), 3),
        "median_tok_s": round(statistics.median(rates), 3) if rates else None,
        "results": results,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=DEFAULT_BASE)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--warm-concurrency", type=int, default=4)
    ap.add_argument("--cold-concurrency", type=int, default=4)
    ap.add_argument("--target-tokens", type=int, default=400000)
    ap.add_argument("--output-tokens", type=int, default=2048)
    ap.add_argument("--warmup-output-tokens", type=int, default=1)
    ap.add_argument("--min-cached-tokens", type=int, default=300000)
    args = ap.parse_args()

    warm_prepared = [
        build_long_prompt(args.base_url, args.target_tokens)
        for _ in range(args.warm_concurrency)
    ]
    cold_prepared = [
        build_long_prompt(args.base_url, args.target_tokens)
        for _ in range(args.cold_concurrency)
    ]

    report: dict = {
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "target_tokens": args.target_tokens,
        "warm_concurrency": args.warm_concurrency,
        "cold_concurrency": args.cold_concurrency,
        "output_tokens": args.output_tokens,
        "warm_prompts": [
            {"codes": codes, "chars": len(prompt)} for prompt, codes in warm_prepared
        ],
        "cold_prompts": [
            {"codes": codes, "chars": len(prompt)} for prompt, codes in cold_prepared
        ],
    }

    # Warm the prefixes first so the baseline and mixed passes are GPU-cached.
    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.warm_concurrency
    ) as pool:
        warmup = list(
            pool.map(
                lambda p: request(
                    args.base_url,
                    args.model,
                    p,
                    max_tokens=args.warmup_output_tokens,
                    ignore_eos=True,
                ),
                [p for p, _ in warm_prepared],
            )
        )
    warmup_wall = time.monotonic() - start
    warmup_cached = [r.get("cached_tokens") or 0 for r in warmup]
    report["warmup"] = {
        "batch_wall_s": round(warmup_wall, 3),
        "cached_tokens": warmup_cached,
        "results": [
            {k: v for k, v in r.items() if k != "content"} for r in warmup
        ],
    }

    # The warmup request itself is cold; probe again to prove the prefix is now
    # cached before running the baseline and mixed passes.
    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.warm_concurrency
    ) as pool:
        probes = list(
            pool.map(
                lambda p: request(
                    args.base_url,
                    args.model,
                    p,
                    max_tokens=1,
                    ignore_eos=True,
                ),
                [p for p, _ in warm_prepared],
            )
        )
    probe_cached = [r.get("cached_tokens") or 0 for r in probes]
    report["cache_probe"] = {
        "batch_wall_s": round(time.monotonic() - start, 3),
        "cached_tokens": probe_cached,
        "all_cached": all(c >= args.min_cached_tokens for c in probe_cached),
        "min_cached_tokens": args.min_cached_tokens,
        "results": [
            {k: v for k, v in r.items() if k != "content"} for r in probes
        ],
    }
    if not report["cache_probe"]["all_cached"]:
        print(json.dumps(report, indent=2))
        print("aborting: warm prefixes were not cached", file=sys.stderr)
        return 2

    # Baseline: warmed decode with no fresh prefill.
    samples, stop, thread = start_sampler(args.base_url)
    start = time.monotonic()
    try:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.warm_concurrency
        ) as pool:
            baseline = list(
                pool.map(
                    lambda p: request(
                        args.base_url,
                        args.model,
                        p,
                        max_tokens=args.output_tokens,
                        ignore_eos=True,
                    ),
                    [p for p, _ in warm_prepared],
                )
            )
    finally:
        stop.set()
        thread.join(timeout=5)
    baseline_wall = time.monotonic() - start
    report["baseline"] = {
        "batch_wall_s": round(baseline_wall, 3),
        **summarize(baseline),
        "gauges": gauges(samples),
        "samples": samples,
    }

    # Mixed: same warmed decode, then fresh long prefills while it is running.
    samples, stop, thread = start_sampler(args.base_url)
    start = time.monotonic()
    warm_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=args.warm_concurrency
    )
    cold_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=args.cold_concurrency
    )
    try:
        warm_futures = [
            warm_pool.submit(
                request,
                args.base_url,
                args.model,
                prompt,
                max_tokens=args.output_tokens,
                ignore_eos=True,
            )
            for prompt, _ in warm_prepared
        ]
        running_seen = wait_for_running(
            args.base_url, args.warm_concurrency, timeout=10.0
        )
        cold_start = time.monotonic()
        cold_futures = [
            cold_pool.submit(
                request,
                args.base_url,
                args.model,
                prompt,
                max_tokens=64,
                ignore_eos=False,
            )
            for prompt, _ in cold_prepared
        ]
        cold = [f.result() for f in cold_futures]
        cold_wall = time.monotonic() - cold_start
        warm = [f.result() for f in warm_futures]
    finally:
        warm_pool.shutdown(wait=True)
        cold_pool.shutdown(wait=True)
        stop.set()
        thread.join(timeout=5)
    mixed_wall = time.monotonic() - start

    cold_recall = []
    for result, (_, codes) in zip(cold, cold_prepared):
        cold_recall.append(
            all(
                code in result.get("content", "")
                for code in (
                    codes["start_code"],
                    codes["middle_code"],
                    codes["end_code"],
                )
            )
        )
    report["mixed"] = {
        "batch_wall_s": round(mixed_wall, 3),
        "cold_batch_wall_s": round(cold_wall, 3),
        "warm_running_seen_before_cold": running_seen,
        **summarize(warm),
        "cold_recall": cold_recall,
        "all_cold_recalled": all(cold_recall),
        "cold_results": [
            {k: v for k, v in r.items() if k != "content"} for r in cold
        ],
        "gauges": gauges(samples),
        "samples": samples,
    }
    report["comparison"] = {
        "baseline_aggregate_tok_s": report["baseline"]["aggregate_tok_s"],
        "mixed_aggregate_tok_s": report["mixed"]["aggregate_tok_s"],
        "aggregate_ratio": (
            round(
                report["mixed"]["aggregate_tok_s"]
                / report["baseline"]["aggregate_tok_s"],
                3,
            )
            if report["baseline"]["aggregate_tok_s"]
            else None
        ),
        "baseline_median_tok_s": report["baseline"]["median_tok_s"],
        "mixed_median_tok_s": report["mixed"]["median_tok_s"],
        "median_ratio": (
            round(
                report["mixed"]["median_tok_s"] / report["baseline"]["median_tok_s"],
                3,
            )
            if report["baseline"]["median_tok_s"]
            else None
        ),
    }

    out = ROOT / "evidence" / (
        f"mixed-prefill-decode-w{args.warm_concurrency}-c{args.cold_concurrency}-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    out.write_text(json.dumps(report, indent=2) + "\n")
    # Keep stdout readable: omit the raw metric series and full content.
    printable = dict(report)
    for key in ("baseline", "mixed"):
        printable[key] = {
            k: v
            for k, v in report[key].items()
            if k not in ("samples", "results", "cold_results")
        }
    print(json.dumps(printable, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
