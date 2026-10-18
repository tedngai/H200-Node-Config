#!/usr/bin/env python3
"""Scoped OpenAI-API probes for the DeepSeek-V4.1-Flash-1017 test server.

No third-party Python packages are required. The script writes a timestamped
JSON report and never changes the service.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def post_json(url: str, payload: dict, timeout: float = 1800.0) -> tuple[dict, float]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    elapsed = time.monotonic() - start
    return json.loads(body), elapsed


def get_json(url: str, timeout: float = 30.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read())


def chat(
    base: str,
    model: str,
    messages: list[dict],
    *,
    max_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 0.95,
    reasoning_effort: str | None = None,
    tools: list[dict] | None = None,
    extra: dict | None = None,
    timeout: float = 1800.0,
) -> tuple[dict, float]:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
    if tools is not None:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    if extra:
        payload.update(extra)
    return post_json(f"{base}/v1/chat/completions", payload, timeout=timeout)


def message_text(resp: dict) -> tuple[str, str, list]:
    choice = resp["choices"][0]
    msg = choice.get("message", {})
    return (
        msg.get("content") or "",
        msg.get("reasoning_content") or "",
        msg.get("tool_calls") or [],
    )


def usage(resp: dict) -> dict:
    return resp.get("usage") or {}


def smoke(base: str, model: str) -> dict:
    report: dict = {"models": None, "plain": None, "reasoning": None, "tools": None}
    report["models"] = get_json(f"{base}/v1/models")

    plain, elapsed = chat(
        base,
        model,
        [{"role": "user", "content": "Reply with exactly: SMOKE-OK"}],
        max_tokens=32,
        temperature=0.0,
        reasoning_effort="none",
    )
    content, reasoning, calls = message_text(plain)
    report["plain"] = {
        "elapsed_s": round(elapsed, 3),
        "content": content,
        "reasoning_content": reasoning,
        "tool_calls": calls,
        "usage": usage(plain),
        "ok": "SMOKE-OK" in content,
    }

    reason, elapsed = chat(
        base,
        model,
        [{"role": "user", "content": "A train travels 240 km in 1.5 hours. What is its average speed in km/h? Think carefully."}],
        max_tokens=512,
        temperature=0.0,
        reasoning_effort="high",
    )
    content, reasoning, calls = message_text(reason)
    report["reasoning"] = {
        "elapsed_s": round(elapsed, 3),
        "content": content,
        "reasoning_content": reasoning,
        "tool_calls": calls,
        "usage": usage(reason),
        "ok": "160" in content and bool(reasoning.strip()),
    }

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }
    ]
    tool_resp, elapsed = chat(
        base,
        model,
        [{"role": "user", "content": "What is the weather in Beijing? Use the get_weather tool."}],
        max_tokens=512,
        temperature=0.0,
        reasoning_effort="high",
        tools=tools,
    )
    content, reasoning, calls = message_text(tool_resp)
    report["tools"] = {
        "elapsed_s": round(elapsed, 3),
        "content": content,
        "reasoning_content": reasoning,
        "tool_calls": calls,
        "usage": usage(tool_resp),
        "finish_reason": tool_resp["choices"][0].get("finish_reason"),
        "ok": bool(calls) and not ("<｜DSML｜" in content),
    }
    return report


def make_tools() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }
    ]


def tool_second_turn(base: str, model: str) -> dict:
    tools = make_tools()
    messages = [{"role": "user", "content": "What is the weather in Beijing? Use the get_weather tool."}]
    first, elapsed1 = chat(
        base,
        model,
        messages,
        max_tokens=512,
        temperature=0.0,
        reasoning_effort="high",
        tools=tools,
    )
    content, reasoning, calls = message_text(first)
    if not calls:
        return {
            "first_elapsed_s": round(elapsed1, 3),
            "first_content": content,
            "first_reasoning": reasoning,
            "ok": False,
            "error": "no tool_calls returned",
        }
    call = calls[0]
    messages.append(
        {
            "role": "assistant",
            "content": content or None,
            "reasoning_content": reasoning or None,
            "tool_calls": calls,
        }
    )
    messages.append(
        {
            "role": "tool",
            "tool_call_id": call.get("id", "call_0"),
            "content": json.dumps({"temperature_c": 21, "conditions": "clear"}),
        }
    )
    second, elapsed2 = chat(
        base,
        model,
        messages,
        max_tokens=512,
        temperature=0.0,
        reasoning_effort="high",
        tools=tools,
    )
    content2, reasoning2, calls2 = message_text(second)
    return {
        "first_elapsed_s": round(elapsed1, 3),
        "first_tool_calls": calls,
        "second_elapsed_s": round(elapsed2, 3),
        "second_content": content2,
        "second_reasoning": reasoning2,
        "second_tool_calls": calls2,
        "usage": usage(second),
        "ok": bool(content2.strip()) and not calls2,
    }


def count_tokens(base: str, text: str) -> int:
    resp, _ = post_json(f"{base}/tokenize", {"prompt": text}, timeout=120.0)
    return int(resp["count"])


def _build_long_prompt(target_tokens: int, salt: str | None = None) -> tuple[str, dict]:
    salt = (salt or uuid.uuid4().hex[:6]).upper()
    start_code = f"ALPHA-{salt}-91QZ"
    middle_code = f"BRAVO-{salt}-44VX"
    end_code = f"CHARLIE-{salt}-73NW"
    paragraphs: list[str] = []
    i = 0
    # Natural synthetic ops text, roughly 30-45 tokens per record.
    while sum(len(p) for p in paragraphs) < target_tokens * 3.8:
        i += 1
        paragraphs.append(
            f"Record {i:06d}: the deployment check inspected service health, "
            f"queue depth, cache reuse, GPU memory headroom and request latency. "
            f"The operator recorded a normal result and continued the runbook."
        )
    mid = len(paragraphs) // 2
    paragraphs.insert(0, f"BEGIN-MARKER {start_code}. Remember this code.")
    paragraphs.insert(mid, f"MIDDLE-MARKER {middle_code}. Remember this code.")
    paragraphs.append(f"END-MARKER {end_code}. Remember this code.")
    prompt = (
        "You are given a long synthetic operations log. Read it and answer the "
        "question at the end.\n\n"
        + "\n".join(paragraphs)
        + "\n\nQuestion: Return the three marker codes exactly, separated by commas. "
        "Do not explain."
    )
    return prompt, {
        "start_code": start_code,
        "middle_code": middle_code,
        "end_code": end_code,
        "approx_chars": len(prompt),
    }


def build_long_prompt(base: str, target_tokens: int) -> tuple[str, dict]:
    """Build a prompt and calibrate its length with the server tokenizer."""
    prompt, codes = _build_long_prompt(target_tokens)
    actual = count_tokens(base, prompt)
    for _ in range(2):
        if 0.97 * target_tokens <= actual <= 1.03 * target_tokens:
            break
        prompt, codes = _build_long_prompt(max(1, int(target_tokens * target_tokens / actual)))
        actual = count_tokens(base, prompt)
    codes["calibrated_tokens"] = actual
    codes["approx_chars"] = len(prompt)
    return prompt, codes


def long_request(base: str, model: str, prompt: str, codes: dict) -> dict:
    resp, elapsed = chat(
        base,
        model,
        [{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.0,
        reasoning_effort="none",
        timeout=3600.0,
    )
    content, reasoning, calls = message_text(resp)
    u = usage(resp)
    return {
        "elapsed_s": round(elapsed, 3),
        "prompt_tokens": u.get("prompt_tokens"),
        "completion_tokens": u.get("completion_tokens"),
        "content": content,
        "reasoning_content": reasoning,
        "tool_calls": calls,
        "usage": u,
        "codes": codes,
        "recalled": all(
            c in content
            for c in (codes["start_code"], codes["middle_code"], codes["end_code"])
        ),
    }


def long_context(base: str, model: str, target_tokens: int) -> dict:
    prompt, codes = build_long_prompt(base, target_tokens)
    result = long_request(base, model, prompt, codes)
    result["target_tokens"] = target_tokens
    result["approx_chars"] = codes["approx_chars"]
    return result


def long_context_batch(base: str, model: str, target_tokens: int, concurrency: int) -> dict:
    prepared = [build_long_prompt(base, target_tokens) for _ in range(concurrency)]
    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(long_request, base, model, prompt, codes)
            for prompt, codes in prepared
        ]
        results = [f.result() for f in futures]
    wall = time.monotonic() - start
    return {
        "target_tokens": target_tokens,
        "concurrency": concurrency,
        "batch_wall_s": round(wall, 3),
        "results": results,
        "all_recalled": all(r["recalled"] for r in results),
    }


def throughput_round(base: str, model: str, prompt: str, output_tokens: int) -> dict:
    resp, elapsed = chat(
        base,
        model,
        [{"role": "user", "content": prompt}],
        max_tokens=output_tokens,
        temperature=0.0,
        reasoning_effort="none",
        extra={"ignore_eos": True},
        timeout=1800.0,
    )
    u = usage(resp)
    completion = u.get("completion_tokens") or 0
    return {
        "elapsed_s": round(elapsed, 3),
        "prompt_tokens": u.get("prompt_tokens"),
        "completion_tokens": completion,
        "tok_s": round(completion / elapsed, 3) if elapsed > 0 else None,
        "content": message_text(resp)[0][:200],
    }


def throughput(base: str, model: str, concurrency: int, output_tokens: int, rounds: int) -> dict:
    prompts = [
        f"Unique prefix {uuid.uuid4().hex}. Write a concise paragraph about reliable distributed systems."
        for _ in range(concurrency)
    ]
    all_rounds = []
    for r in range(rounds):
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [
                pool.submit(throughput_round, base, model, p, output_tokens)
                for p in prompts
            ]
            results = [f.result() for f in futures]
        rates = [x["tok_s"] for x in results if x["tok_s"] is not None]
        all_rounds.append(
            {
                "round": r + 1,
                "results": results,
                "aggregate_tok_s": round(sum(rates), 3),
                "median_tok_s": round(statistics.median(rates), 3) if rates else None,
            }
        )
    return {"concurrency": concurrency, "output_tokens": output_tokens, "rounds": all_rounds}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8013")
    ap.add_argument("--model", default="dsv41-flash-1017")
    ap.add_argument(
        "--mode",
        choices=["smoke", "tools", "long", "throughput"],
        default="smoke",
    )
    ap.add_argument("--prompt-tokens", type=int, default=100000)
    ap.add_argument("--output-tokens", type=int, default=256)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    report: dict = {
        "utc": utc_now(),
        "base_url": args.base_url,
        "model": args.model,
        "mode": args.mode,
    }
    try:
        if args.mode == "smoke":
            report["result"] = smoke(args.base_url, args.model)
        elif args.mode == "tools":
            report["result"] = tool_second_turn(args.base_url, args.model)
        elif args.mode == "long":
            if args.concurrency > 1:
                report["result"] = long_context_batch(
                    args.base_url, args.model, args.prompt_tokens, args.concurrency
                )
            else:
                report["result"] = long_context(args.base_url, args.model, args.prompt_tokens)
        elif args.mode == "throughput":
            report["result"] = throughput(
                args.base_url, args.model, args.concurrency, args.output_tokens, args.rounds
            )
        else:  # pragma: no cover
            raise ValueError(args.mode)
        report["ok"] = True
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        report["ok"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, urllib.error.HTTPError):
            try:
                report["error_body"] = exc.read().decode("utf-8", "replace")[:4000]
            except Exception:
                pass

    report_path = Path(args.report) if args.report else (
        ROOT / "evidence" / f"{args.mode}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
