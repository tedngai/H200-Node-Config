#!/usr/bin/env python3
"""Streaming cold-prompt TTFT probe.

Builds N distinct natural synthetic 400k prompts, sends them concurrently with
stream=True, and records time to first generated token, finish time, and cold
recall. Use the same script before and after a config change to enforce a
cold-TTFT ceiling.
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
from openai_probe import build_long_prompt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = "http://127.0.0.1:8013"
DEFAULT_MODEL = "dsv41-flash-1017"


def stream_request(
    base: str,
    model: str,
    prompt: str,
    codes: dict,
    max_tokens: int,
    batch_start: float,
    slot: int,
    results: list,
) -> None:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "reasoning_effort": "none",
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    started = time.monotonic()
    first_token: float | None = None
    finish_reason = None
    content: list[str] = []
    usage = None
    error = None
    try:
        with urllib.request.urlopen(req, timeout=3600.0) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                if obj.get("usage"):
                    usage = obj["usage"]
                for choice in obj.get("choices") or []:
                    delta = choice.get("delta") or {}
                    text = delta.get("content") or delta.get("reasoning_content") or ""
                    if text and first_token is None:
                        first_token = time.monotonic()
                    if text:
                        content.append(text)
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
    except Exception as exc:  # keep the other requests running
        error = str(exc)
    finished = time.monotonic()
    text = "".join(content)
    results[slot] = {
        "slot": slot,
        "start_offset_s": round(started - batch_start, 3),
        "ttft_s": round(first_token - started, 3) if first_token else None,
        "finish_offset_s": round(finished - batch_start, 3),
        "elapsed_s": round(finished - started, 3),
        "finish_reason": finish_reason,
        "prompt_tokens": codes.get("calibrated_tokens"),
        "completion_tokens": (usage or {}).get("completion_tokens"),
        "content_chars": len(text),
        "recalled": all(
            codes[k] in text for k in ("start_code", "middle_code", "end_code")
        ),
        "error": error,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=DEFAULT_BASE)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--target-tokens", type=int, default=400000)
    ap.add_argument("--output-tokens", type=int, default=512)
    ap.add_argument("--label", default="cold-ttft")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    prepared = [
        build_long_prompt(args.base_url, args.target_tokens)
        for _ in range(args.concurrency)
    ]
    results: list = [None] * args.concurrency
    batch_start = time.monotonic()
    threads = [
        threading.Thread(
            target=stream_request,
            args=(
                args.base_url,
                args.model,
                prompt,
                codes,
                args.output_tokens,
                batch_start,
                slot,
                results,
            ),
        )
        for slot, (prompt, codes) in enumerate(prepared)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall = time.monotonic() - batch_start

    ttfts = [r["ttft_s"] for r in results if r and r.get("ttft_s") is not None]
    report = {
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "base_url": args.base_url,
        "model": args.model,
        "label": args.label,
        "target_tokens": args.target_tokens,
        "output_tokens": args.output_tokens,
        "concurrency": args.concurrency,
        "batch_wall_s": round(wall, 3),
        "ttft_s": ttfts,
        "ttft_median_s": round(statistics.median(ttfts), 3) if ttfts else None,
        "ttft_max_s": round(max(ttfts), 3) if ttfts else None,
        "all_recalled": all(r and r.get("recalled") for r in results),
        "prompts": [
            {"codes": codes, "chars": len(prompt)} for prompt, codes in prepared
        ],
        "results": results,
    }
    print(json.dumps(report, indent=2))
    path = Path(args.report) if args.report else None
    if path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = ROOT / "evidence" / f"{args.label}-{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"report: {path}")
    return 0 if report["all_recalled"] and not any(
        r and r.get("error") for r in results
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
