#!/usr/bin/env python3
"""Verify actual active concurrency with independently cached long prompts."""

import argparse
import concurrent.futures
import json
import pathlib
import re
import statistics
import subprocess
import threading
import time
import urllib.request

from benchmark import request
from smoke_test import make_prompt


def build_prompts(token_count, concurrency, namespace='default'):
    base = make_prompt(token_count)
    code = r'''
import json
import sys
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('zai-org/GLM-5.3', local_files_only=True)
print(json.dumps({
    'user_marker': t.convert_tokens_to_ids('<|user|>'),
    'labels': [t.encode(f'Independent test conversation {sys.argv[2]}/{i}.\n', add_special_tokens=False) for i in range(int(sys.argv[1]))],
}))
'''
    data = json.loads(subprocess.check_output(
        ['docker', 'exec', 'sglang-glm53', 'python3', '-B', '-c', code, str(concurrency), namespace],
        text=True,
        timeout=60,
    ))
    position = base.index(data['user_marker']) + 1
    prompts = []
    for label in data['labels']:
        # Insert the unique label immediately after the user role marker. Trim
        # only synthetic filler, leaving the codes and question intact. Since
        # every prefix diverges here, the long KV prefixes cannot be shared.
        trim_position = min(1000, token_count // 4)
        prompt = base[:position] + label + base[position:trim_position] + base[trim_position + len(label):]
        assert len(prompt) == token_count
        prompts.append(prompt)
    assert len({tuple(prompt[:100]) for prompt in prompts}) == concurrency
    return prompts


def monitor_metrics(base_url, stop, samples, errors):
    pattern = re.compile(r'^sglang:(num_running_reqs|num_queue_reqs|num_retracted_reqs|token_usage)\{[^\n]*\}\s+([-+\d.eE]+)$', re.MULTILINE)
    while not stop.is_set():
        try:
            with urllib.request.urlopen(base_url + '/metrics', timeout=5) as response:
                text = response.read().decode()
            values = {}
            for name, value in pattern.findall(text):
                values[name] = max(values.get(name, 0), float(value))
            samples.append({'time': time.monotonic(), **values})
        except Exception as error:
            errors.append(str(error))
        stop.wait(0.2)


def generate(base_url, prompt, output_tokens, forced):
    start = time.monotonic()
    response = request(base_url, '/generate', {
        'input_ids': prompt,
        'sampling_params': {
            'temperature': 0,
            'max_new_tokens': output_tokens,
            'ignore_eos': forced,
        },
        'stream': False,
    })
    meta = response.get('meta_info', {})
    answer = response.get('text', '').rsplit('</think>', 1)[-1]
    return {
        'elapsed_seconds': time.monotonic() - start,
        'meta_info': meta,
        'retrieval_passed': all(code in answer for code in ('MAPLE-4821', 'RIVER-7306', 'CLOUD-9158')),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:8012')
    parser.add_argument('--concurrency', type=int, default=6)
    parser.add_argument('--prompt-tokens', type=int, default=100000)
    parser.add_argument('--output-tokens', type=int, default=512)
    args = parser.parse_args()
    if not args.label or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in args.label):
        parser.error('Invalid report label')
    if min(args.concurrency, args.prompt_tokens, args.output_tokens) < 1:
        parser.error('Numeric arguments must be positive')
    info = request(args.base_url, '/server_info')
    if args.concurrency > info['max_running_requests']:
        parser.error('Concurrency exceeds the configured running-request cap')
    if args.prompt_tokens + max(args.output_tokens, 1024) > info['context_length']:
        parser.error('Request exceeds configured context')
    if args.concurrency * (args.prompt_tokens + args.output_tokens) > info['max_total_num_tokens'] * 0.9:
        parser.error('Test would consume over 90% of the token pool; reduce the prompt length')

    prompts = build_prompts(args.prompt_tokens, args.concurrency, namespace=args.label)
    warmup = []
    for index, prompt in enumerate(prompts):
        result = generate(args.base_url, prompt, 1024, False)
        warmup.append(result)
        print(json.dumps({'warmup_conversation': index + 1, 'prompt_tokens': result['meta_info'].get('prompt_tokens'), 'retrieval_passed': result['retrieval_passed'], 'elapsed_seconds': result['elapsed_seconds']}), flush=True)
        if not result['retrieval_passed']:
            raise RuntimeError('Long-prompt warmup failed retrieval')

    samples, errors = [], []
    stop = threading.Event()
    monitor = threading.Thread(target=monitor_metrics, args=(args.base_url, stop, samples, errors), daemon=True)
    monitor.start()
    barrier = threading.Barrier(args.concurrency)

    def worker(prompt):
        barrier.wait(timeout=30)
        return generate(args.base_url, prompt, args.output_tokens, True)

    start = time.monotonic()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(worker, prompt) for prompt in prompts]
            results = [future.result() for future in futures]
        elapsed = time.monotonic() - start
    finally:
        stop.set()
        monitor.join(timeout=6)

    peak_running = max((sample.get('num_running_reqs', 0) for sample in samples), default=0)
    peak_queue = max((sample.get('num_queue_reqs', 0) for sample in samples), default=0)
    token_total = sum(result['meta_info'].get('completion_tokens', 0) for result in results)
    retractions = sum(result['meta_info'].get('num_retractions', 0) for result in warmup + results)
    passed = (
        peak_running >= args.concurrency
        and retractions == 0
        and all(result['meta_info'].get('prompt_tokens') == args.prompt_tokens for result in results)
        and all(result['meta_info'].get('completion_tokens') == args.output_tokens for result in results)
    )
    summary = {
        'passed': passed,
        'peak_running_requests': peak_running,
        'peak_queued_requests': peak_queue,
        'peak_active_token_usage': max((sample.get('token_usage', 0) for sample in samples), default=0),
        'total_retractions': retractions,
        'aggregate_tokens_per_second': token_total / elapsed,
        'median_tokens_per_second_per_request': statistics.median(result['meta_info']['completion_tokens'] / result['elapsed_seconds'] for result in results),
        'elapsed_seconds': elapsed,
        'cached_prompt_tokens': [result['meta_info'].get('cached_tokens') for result in results],
        'metric_sampling_errors': errors,
    }
    report = {
        'label': args.label,
        'concurrency': args.concurrency,
        'prompt_tokens_per_request': args.prompt_tokens,
        'output_tokens_per_request': args.output_tokens,
        'config': {key: info.get(key) for key in ('max_running_requests', 'max_total_num_tokens', 'context_length', 'mem_fraction_static', 'kv_cache_dtype', 'speculative_algorithm')},
        'summary': summary,
        'warmup': warmup,
        'results': results,
        'metrics': samples,
    }
    path = pathlib.Path(__file__).parent / f'{args.label}-long-c{args.concurrency}-{args.prompt_tokens}.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'report': str(path), 'summary': summary}, indent=2), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
