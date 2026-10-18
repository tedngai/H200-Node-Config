#!/usr/bin/env python3
"""Replay the exact 16 prompts from the c16 cold test, warm-only.

The prompt body is deterministic given the salt; the saved report contains each
salt in its marker codes. A single low-token precheck verifies the prefix is
still cached before sending all 16 warm requests.
"""

from __future__ import annotations

import concurrent.futures
import glob
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
    _build_long_prompt,
    chat,
    message_text,
    usage,
)

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8013"
MODEL = "dsv41-flash-1017"
TARGET = 400000
OUTPUT = 512


def metrics() -> dict:
    with urllib.request.urlopen(f"{BASE}/metrics", timeout=5) as r:
        text = r.read().decode("utf-8", "replace")
    out = {}
    for line in text.splitlines():
        if not line.startswith("sglang:") or line.startswith("#"):
            continue
        name = line.split("{", 1)[0]
        try:
            v = float(line.rsplit(" ", 1)[1])
        except (IndexError, ValueError):
            continue
        out[name] = max(out.get(name, float("-inf")), v)
    return out


def warm(prompt: str) -> dict:
    resp, elapsed = chat(
        BASE,
        MODEL,
        [{"role": "user", "content": prompt}],
        max_tokens=OUTPUT,
        temperature=0.0,
        reasoning_effort="none",
        extra={"ignore_eos": True},
        timeout=3600.0,
    )
    u = usage(resp)
    c = u.get("completion_tokens") or 0
    return {
        "elapsed_s": round(elapsed, 3),
        "prompt_tokens": u.get("prompt_tokens"),
        "completion_tokens": c,
        "tok_s": round(c / elapsed, 3) if elapsed else None,
        "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
    }


def main() -> int:
    report_path = sorted(glob.glob(str(ROOT / "evidence" / "long-400000-c16-warm-*.json")))[-1]
    source = json.loads(Path(report_path).read_text())
    def build_for_chars(target_chars: int, salt: str):
        lo, hi = 100000, 1000000
        for _ in range(30):
            mid = (lo + hi) // 2
            prompt, _ = _build_long_prompt(mid, salt=salt)
            if len(prompt) < target_chars:
                lo = mid + 1
            else:
                hi = mid
        prompt, rebuilt = _build_long_prompt(lo, salt=salt)
        if len(prompt) != target_chars:
            raise SystemExit(f"prompt length mismatch: {len(prompt)} != {target_chars}")
        return prompt, rebuilt

    prepared = []
    for item in source["prompts"]:
        codes = item["codes"]
        salt = codes["start_code"].split("-")[1]
        prompt, rebuilt = build_for_chars(item["chars"], salt)
        prepared.append((prompt, rebuilt))

    # Precheck: one request, low output, must hit the cached prefix.
    pre, pre_elapsed = chat(
        BASE,
        MODEL,
        [{"role": "user", "content": prepared[0][0]}],
        max_tokens=8,
        temperature=0.0,
        reasoning_effort="none",
        timeout=3600.0,
    )
    pre_usage = usage(pre)
    pre_cached = (pre_usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    if pre_cached < 300000:
        print(
            json.dumps(
                {
                    "aborted": True,
                    "reason": "prefix not cached",
                    "precheck_prompt_tokens": pre_usage.get("prompt_tokens"),
                    "precheck_cached_tokens": pre_cached,
                    "precheck_elapsed_s": round(pre_elapsed, 3),
                    "source_report": report_path,
                },
                indent=2,
            )
        )
        return 2

    samples = []
    stop = threading.Event()

    def sampler() -> None:
        while not stop.is_set():
            try:
                samples.append(metrics())
            except Exception:
                pass
            time.sleep(0.25)

    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()
    start = time.monotonic()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(prepared)) as pool:
            results = list(pool.map(lambda p: warm(p[0]), prepared))
    finally:
        stop.set()
        thread.join(timeout=5)
    wall = time.monotonic() - start

    rates = [r["tok_s"] for r in results if r["tok_s"]]
    def peak(name):
        vals = [s[name] for s in samples if name in s]
        return max(vals) if vals else None

    out = {
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_report": report_path,
        "precheck": {
            "prompt_tokens": pre_usage.get("prompt_tokens"),
            "cached_tokens": pre_cached,
            "elapsed_s": round(pre_elapsed, 3),
        },
        "concurrency": len(prepared),
        "output_tokens": OUTPUT,
        "batch_wall_s": round(wall, 3),
        "aggregate_tok_s": round(sum(rates), 3),
        "median_tok_s": round(statistics.median(rates), 3) if rates else None,
        "results": results,
        "gauges": {
            "samples": len(samples),
            "peak_running": peak("sglang:num_running_reqs"),
            "peak_queue": peak("sglang:num_queue_reqs"),
            "peak_retracted": peak("sglang:num_retracted_reqs"),
            "peak_token_usage": peak("sglang:token_usage"),
            "peak_full_token_usage": peak("sglang:full_token_usage"),
            "peak_swa_token_usage": peak("sglang:swa_token_usage"),
            "peak_hicache_host_used_tokens": peak("sglang:hicache_host_used_tokens"),
        },
    }
    path = ROOT / "evidence" / (
        f"replay-c16-warm-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
