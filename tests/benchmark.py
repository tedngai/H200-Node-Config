#!/usr/bin/env python3
"""Small repeatable decode benchmark; no elevated privileges required."""

import argparse
import concurrent.futures
import json
import pathlib
import statistics
import time
import urllib.request


def request(base_url, endpoint, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        base_url + endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as response:
        return json.load(response)


def generate(base_url, output_tokens, index):
    payload = {
        "text": (
            f"Benchmark sample {index}. "
            "Explain how continuous batching and a key-value cache work in "
            "a language-model inference server. Give a detailed technical explanation."
        ),
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": output_tokens,
            "ignore_eos": True,
        },
        "stream": False,
    }
    start = time.monotonic()
    response = request(base_url, "/generate", payload)
    elapsed = time.monotonic() - start
    meta = response.get("meta_info", {})
    return {
        "elapsed_seconds": elapsed,
        "end_to_end_tokens_per_second": meta.get("completion_tokens", 0) / elapsed,
        "meta_info": meta,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8012")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output-tokens", type=int, default=256)
    args = parser.parse_args()
    if min(args.concurrency, args.rounds, args.output_tokens) < 1:
        parser.error("concurrency, rounds and output-tokens must be positive")
    if not args.label or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.label):
        parser.error("label must contain only letters, numbers, hyphens and underscores")

    info = request(args.base_url, "/server_info")
    config_keys = [
        "version", "status", "model_path", "context_length", "tp_size",
        "pp_size", "dp_size", "kv_cache_dtype", "mem_fraction_static",
        "max_running_requests", "max_queued_requests", "max_total_num_tokens",
        "speculative_algorithm", "dsa_prefill_backend", "dsa_decode_backend",
    ]
    report = {
        "label": args.label,
        "config": {key: info.get(key) for key in config_keys},
        "memory_usage": [state.get("memory_usage") for state in info.get("internal_states", [])],
        "concurrency": args.concurrency,
        "output_tokens": args.output_tokens,
        "warmup": generate(args.base_url, args.output_tokens, -1),
        "rounds": [],
    }
    for round_index in range(args.rounds):
        start = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [
                pool.submit(generate, args.base_url, args.output_tokens, round_index * args.concurrency + index)
                for index in range(args.concurrency)
            ]
            samples = [future.result() for future in futures]
        elapsed = time.monotonic() - start
        report["rounds"].append({
            "elapsed_seconds": elapsed,
            "aggregate_tokens_per_second": sum(sample["meta_info"].get("completion_tokens", 0) for sample in samples) / elapsed,
            "samples": samples,
        })

    samples = [sample for row in report["rounds"] for sample in row["samples"]]
    report["summary"] = {
        "median_end_to_end_tokens_per_second_per_request": statistics.median(sample["end_to_end_tokens_per_second"] for sample in samples),
        "median_aggregate_tokens_per_second": statistics.median(row["aggregate_tokens_per_second"] for row in report["rounds"]),
        "total_retractions": sum(sample["meta_info"].get("num_retractions", 0) for sample in samples),
        "completion_tokens": [sample["meta_info"].get("completion_tokens") for sample in samples],
    }
    if any(sample["meta_info"].get("completion_tokens") != args.output_tokens for sample in samples):
        raise RuntimeError("A forced-length sample did not generate the requested token count")
    destination = pathlib.Path(__file__).parent / f"{args.label}-c{args.concurrency}.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(destination), "config": report["config"], "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
