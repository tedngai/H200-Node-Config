#!/usr/bin/env python3
"""Rank-pinned DP4 benchmarks and HiCache restoration, without service changes."""

import argparse
import concurrent.futures
import datetime
import json
import math
from pathlib import Path
import re
import statistics
import threading
import time
import urllib.request

from benchmark import request
from concurrency_test import build_prompts


METRIC = re.compile(
    r'^sglang:(num_running_reqs|num_queue_reqs|num_retracted_reqs|token_usage)'
    r'\{([^\n]*)\}\s+([-+\d.eE]+)$', re.MULTILINE)
CONFIG_KEYS = (
    'status', 'version', 'model_path', 'served_model_name', 'context_length',
    'tp_size', 'dp_size', 'ep_size', 'enable_dp_attention', 'kv_cache_dtype',
    'mem_fraction_static', 'max_running_requests', 'max_queued_requests',
    'max_total_num_tokens', 'chunked_prefill_size', 'speculative_algorithm',
    'speculative_num_steps', 'speculative_eagle_topk',
    'speculative_num_draft_tokens', 'enable_hierarchical_cache', 'hicache_size',
    'hicache_write_policy', 'hicache_io_backend', 'hicache_mem_layout',
    'enable_cache_report', 'reasoning_parser', 'tool_call_parser',
    'allow_auto_truncate', 'dsa_prefill_backend', 'dsa_decode_backend')
CODES = ('MAPLE-4821', 'RIVER-7306', 'CLOUD-9158')


def parse_metrics(text):
    """Deduplicate TP replicas within a DP rank, never sum replicated gauges."""
    values = {}
    for name, labels, value in METRIC.findall(text):
        rank = re.search(r'dp_rank="(\d+)"', labels)
        if not rank:
            raise ValueError(f'Missing dp_rank on {name}')
        if 'model_name="pratt-high"' not in labels:
            continue
        per_rank = values.setdefault(name, {})
        key = rank.group(1)
        per_rank[key] = max(per_rank.get(key, 0), float(value))
    return values


def fetch_metrics(base_url):
    with urllib.request.urlopen(base_url + '/metrics', timeout=5) as response:
        return parse_metrics(response.read().decode())


def require_idle(base_url):
    values = fetch_metrics(base_url)
    expected = {'0', '1', '2', '3'}
    for name in ('num_running_reqs', 'num_queue_reqs'):
        if set(values.get(name, {})) != expected:
            raise RuntimeError(f'Expected four exported DP worker gauges: {name}')
        if any(values[name].values()):
            raise RuntimeError('Server has active/queued requests; refusing to start a test')
    return values


def validate_config(info, concurrency, prompt_tokens, output_tokens):
    if info.get('status') != 'ready' or info.get('dp_size') != 4 or not info.get('enable_dp_attention'):
        raise ValueError('Expected ready DP4-attention server')
    if not 1 <= concurrency <= 4:
        raise ValueError('Use one request per worker, concurrency 1-4')
    states = info.get('internal_states', [])
    if len(states) != 4 or any(s.get('effective_max_running_requests_per_dp', 0) < 1 for s in states):
        raise ValueError('Missing per-worker admission data')
    capacities = [s.get('memory_usage', {}).get('token_capacity', 0) for s in states]
    if prompt_tokens + output_tokens > info['context_length']:
        raise ValueError('Prompt plus output exceeds context; truncation is not allowed')
    # There is one test request in each private pool, not concurrency requests
    # in one shared pool. Leave at least 10% headroom in every selected pool.
    if any(prompt_tokens + output_tokens > capacity * 0.9 for capacity in capacities):
        raise ValueError('A request would consume over 90% of a private GPU pool')
    if info.get('allow_auto_truncate'):
        raise ValueError('Auto-truncation must be disabled')


def generate(base_url, prompt, rank, output_tokens, forced, retrieval=False,
             expected_prompt_tokens=None):
    payload = {
        'input_ids' if isinstance(prompt, list) else 'text': prompt,
        'routed_dp_rank': rank,
        'sampling_params': {'temperature': 0, 'max_new_tokens': output_tokens,
                            'ignore_eos': forced},
        'stream': False,
    }
    started = time.monotonic()
    response = request(base_url, '/generate', payload)
    elapsed = time.monotonic() - started
    meta = response.get('meta_info', {})
    text = response.get('text', '')
    answer = text.rsplit('</think>', 1)[-1].strip()
    details = meta.get('cached_tokens_details') or {}
    expected_prompt_tokens = len(prompt) if isinstance(prompt, list) else expected_prompt_tokens
    prefill = None
    if 'prefill_finished_time' in meta and 'forward_entry_time' in meta:
        prefill = meta['prefill_finished_time'] - meta['forward_entry_time']
    return {
        'requested_dp_rank': rank, 'elapsed_seconds': elapsed,
        'end_to_end_tokens_per_second': meta.get('completion_tokens', 0) / elapsed,
        'prefill_or_restore_seconds': prefill,
        'device_cached_tokens': details.get('device', 0),
        'host_cached_tokens': details.get('host', 0),
        'retrieval_passed': all(code in answer for code in CODES) if retrieval else None,
        'answer_text': answer if retrieval else None,
        'forced_output': forced, 'meta_info': meta,
        'valid': (meta.get('dp_rank') == rank
                  and (expected_prompt_tokens is None or meta.get('prompt_tokens') == expected_prompt_tokens)
                  and (not forced or meta.get('completion_tokens') == output_tokens)
                  and meta.get('num_retractions', 0) == 0
                  and (not retrieval or all(code in answer for code in CODES))),
    }


def timed_batch(base_url, calls):
    samples, errors = [], []
    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            try:
                samples.append({'time': time.monotonic(), **fetch_metrics(base_url)})
            except Exception as error:
                errors.append(str(error))
            stop.wait(0.1)

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    barrier = threading.Barrier(len(calls))

    def worker(call):
        barrier.wait(timeout=30)
        return generate(base_url, **call)

    started = time.monotonic()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(calls)) as pool:
            results = list(pool.map(worker, calls))
        elapsed = time.monotonic() - started
    finally:
        stop.set()
        monitor_thread.join(timeout=6)
    total = sum(result['meta_info'].get('completion_tokens', 0) for result in results)
    peak_running = max((sum(s.get('num_running_reqs', {}).values()) for s in samples), default=0)
    peak_queue = max((sum(s.get('num_queue_reqs', {}).values()) for s in samples), default=0)
    rank_set = {str(call['rank']) for call in calls}
    simultaneous_selected = max((sum(s.get('num_running_reqs', {}).get(rank, 0) for rank in rank_set)
                                 for s in samples), default=0)
    drafts = sum(r['meta_info'].get('spec_num_proposed_drafts', 0) for r in results)
    accepted = sum(r['meta_info'].get('spec_num_correct_drafts', 0) for r in results)
    verified = sum(r['meta_info'].get('spec_verify_ct', 0) for r in results)
    summary = {
        'passed': (all(result['valid'] for result in results) and not errors
                   and peak_queue == 0 and simultaneous_selected >= len(calls)),
        'elapsed_seconds': elapsed, 'actual_output_tokens': total,
        'aggregate_tokens_per_second': total / elapsed,
        'median_tokens_per_second_per_request': statistics.median(r['end_to_end_tokens_per_second'] for r in results),
        'peak_running_requests': peak_running, 'peak_queued_requests': peak_queue,
        'peak_simultaneous_selected_workers': simultaneous_selected,
        'peak_running_by_rank': {rank: max((s.get('num_running_reqs', {}).get(rank, 0) for s in samples), default=0)
                                 for rank in sorted(rank_set)},
        'peak_active_token_usage_by_rank': {rank: max((s.get('token_usage', {}).get(rank, 0) for s in samples), default=0)
                                           for rank in sorted(rank_set)},
        'total_retractions': sum(r['meta_info'].get('num_retractions', 0) for r in results),
        'cached_prompt_tokens': [r['meta_info'].get('cached_tokens') for r in results],
        'device_cached_tokens': [r['device_cached_tokens'] for r in results],
        'host_cached_tokens': [r['host_cached_tokens'] for r in results],
        'weighted_mtp_draft_acceptance': accepted / drafts if drafts else None,
        'weighted_mtp_tokens_per_verification': total / verified if verified else None,
        'metric_sampling_errors': errors,
    }
    return {'summary': summary, 'results': results, 'metrics': samples}


def historical_prompt(index):
    # Exactly the old benchmark text. This raw completion is a speed-only
    # compatibility test, NOT an instruction/tool/coding-quality gate.
    return (f'Benchmark sample {index}. '
            'Explain how continuous batching and a key-value cache work in '
            'a language-model inference server. Give a detailed technical explanation.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('short', 'long', 'host'), required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:8012')
    parser.add_argument('--concurrency', type=int, default=4)
    parser.add_argument('--prompt-tokens', type=int, default=100000)
    parser.add_argument('--output-tokens', type=int, default=512)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--rank', type=int, choices=range(4), default=0)
    args = parser.parse_args()
    if min(args.output_tokens, args.rounds) < 1 or args.prompt_tokens < 2000:
        parser.error('Positive output/rounds and at least 2000 prompt tokens required')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    namespace = f'dp4-mtp-hicache128-{args.mode}-{stamp}'
    suffix = f'c{args.concurrency}' if args.mode != 'host' else f'rank{args.rank}'
    path = Path(__file__).resolve().parents[1] / 'evidence' / f'{namespace}-{suffix}-{args.prompt_tokens if args.mode != "short" else args.output_tokens}.json'
    report = {'label': namespace, 'started_at_utc': stamp, 'mode': args.mode,
              'concurrency': args.concurrency if args.mode != 'host' else 1,
              'rounds_requested': args.rounds,
              'prompt_tokens_per_request': args.prompt_tokens if args.mode != 'short' else 28,
              'output_tokens_per_request': args.output_tokens if args.mode != 'host' else 1024,
              'limitations': ['No service settings changed; not an isolated HiCache on/off comparison.',
                              'Synthetic limited-duration tests, not sustained production or broad coding quality.',
                              'Native decode_throughput is ignored; rates use client time and actual output.'],
              'stages': [], 'rounds': []}

    def save():
        path.write_text(json.dumps(report, indent=2) + '\n')

    def stage(name, batch):
        report['stages'].append({'stage': name, **batch})
        save()
        print(json.dumps({'stage': name, **batch['summary']}), flush=True)
        if not batch['summary']['passed']:
            raise RuntimeError(f'{name} failed rank/length/retrieval/concurrency/queue/error checks')

    try:
        with path.open('x') as stream:
            stream.write(json.dumps(report, indent=2) + '\n')
    except FileExistsError:
        parser.error('Report already exists; refusing to overwrite')
    try:
        info = request(args.base_url, '/server_info')
        report['config'] = {key: info.get(key) for key in CONFIG_KEYS}
        report['workers'] = [{'index': i, 'memory_usage': s.get('memory_usage'),
                              'effective_max_running_requests_per_dp': s.get('effective_max_running_requests_per_dp')}
                             for i, s in enumerate(info.get('internal_states', []))]
        validate_config(info, args.concurrency if args.mode != 'host' else 1,
                        args.prompt_tokens if args.mode != 'short' else 100,
                        max(args.output_tokens, 1024))
        report['idle_metrics_before'] = require_idle(args.base_url)
        if args.mode == 'short':
            report['prompt_format'] = 'historical raw 28-token completion; speed only'
            report['warmup'] = []
            for rank in range(args.concurrency):
                result = generate(args.base_url, historical_prompt(-1), rank, args.output_tokens, True,
                                  expected_prompt_tokens=28)
                report['warmup'].append(result)
                save()
                if not result['valid']:
                    raise RuntimeError('Short warmup failed')
            prompts = [historical_prompt(index) for index in range(args.concurrency)]
        elif args.mode == 'long':
            report['prompt_format'] = 'model chat template; natural synthetic records; distinct early prefixes'
            prompts = build_prompts(args.prompt_tokens, args.concurrency, namespace)
            stage('initial_independent_retrieval', timed_batch(args.base_url, [
                {'prompt': prompt, 'rank': rank, 'output_tokens': 1024,
                 'forced': False, 'retrieval': True} for rank, prompt in enumerate(prompts)]))
        else:
            if not info.get('enable_hierarchical_cache') or not info.get('enable_cache_report'):
                raise RuntimeError('Host restore requires HiCache and cache reporting')
            report['prompt_format'] = 'model chat template; natural synthetic records; distinct early prefixes'
            pressure = math.ceil(info['max_total_num_tokens'] / args.prompt_tokens)
            report['pressure_conversations'] = pressure
            report['routed_dp_rank'] = args.rank
            prompts = build_prompts(args.prompt_tokens, pressure + 1, namespace)

            def host_stage(name, prompt):
                batch = timed_batch(args.base_url, [
                    {'prompt': prompt, 'rank': args.rank, 'output_tokens': 1024,
                     'forced': False, 'retrieval': True}])
                stage(name, batch)
                return batch['results'][0]

            cold = host_stage('cold_target', prompts[0])
            warm = host_stage('gpu_warm_target', prompts[0])
            for index, prompt in enumerate(prompts[1:]):
                host_stage(f'pressure_{index + 1}', prompt)
            restored = host_stage('restored_target', prompts[0])
            report['summary'] = {
                'passed': (cold['device_cached_tokens'] + cold['host_cached_tokens'] <= 128
                           and warm['device_cached_tokens'] >= args.prompt_tokens - 128
                           and restored['host_cached_tokens'] >= args.prompt_tokens - 128
                           and restored['device_cached_tokens'] <= 128),
                'cold_elapsed_seconds': cold['elapsed_seconds'],
                'gpu_warm_elapsed_seconds': warm['elapsed_seconds'],
                'restored_elapsed_seconds': restored['elapsed_seconds'],
                'cold_prefill_seconds': cold['prefill_or_restore_seconds'],
                'gpu_warm_prefill_seconds': warm['prefill_or_restore_seconds'],
                'restored_prefill_seconds': restored['prefill_or_restore_seconds'],
                'host_cached_tokens_on_restore': restored['host_cached_tokens'],
                'device_cached_tokens_on_restore': restored['device_cached_tokens'],
                'cold_vs_restore_end_to_end_speedup': cold['elapsed_seconds'] / restored['elapsed_seconds'],
                'all_retrievals_passed': True, 'total_retractions': 0,
                'note': 'Natural answer lengths may differ; prefill/restore intervals isolate the cache benefit better.',
            }

        if args.mode != 'host':
            for index in range(args.rounds):
                if args.mode == 'short':
                    prompts = [historical_prompt(index * args.concurrency + rank) for rank in range(args.concurrency)]
                batch = timed_batch(args.base_url, [
                    {'prompt': prompt, 'rank': rank, 'output_tokens': args.output_tokens,
                     'forced': True, 'expected_prompt_tokens': 28 if args.mode == 'short' else None}
                    for rank, prompt in enumerate(prompts)])
                report['rounds'].append(batch)
                save()
                print(json.dumps({'round': index + 1, **batch['summary']}), flush=True)
                if not batch['summary']['passed']:
                    raise RuntimeError('Timed batch failed validation')
            rows = [row['summary'] for row in report['rounds']]
            report['summary'] = {
                'passed': True,
                'median_aggregate_tokens_per_second': statistics.median(row['aggregate_tokens_per_second'] for row in rows),
                'median_end_to_end_tokens_per_second_per_request': statistics.median(
                    r['end_to_end_tokens_per_second'] for row in report['rounds'] for r in row['results']),
                'peak_running_requests': max(row['peak_running_requests'] for row in rows),
                'peak_queued_requests': max(row['peak_queued_requests'] for row in rows),
                'total_retractions': sum(row['total_retractions'] for row in rows),
                'actual_output_tokens': sum(row['actual_output_tokens'] for row in rows),
                'median_weighted_mtp_draft_acceptance': statistics.median(
                    row['weighted_mtp_draft_acceptance'] for row in rows if row['weighted_mtp_draft_acceptance'] is not None)
                    if any(row['weighted_mtp_draft_acceptance'] is not None for row in rows) else None,
            }
        after = request(args.base_url, '/server_info')
        if any(info.get(key) != after.get(key) for key in CONFIG_KEYS):
            raise RuntimeError('Server config changed during test')
        report['idle_metrics_after'] = require_idle(args.base_url)
        report['finished_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()
        print(json.dumps({'report': str(path), 'summary': report['summary']}, indent=2), flush=True)
        if not report['summary']['passed']:
            raise RuntimeError('Cache tier/eviction proof failed')
    except Exception as error:
        report['error'] = repr(error)
        report.setdefault('summary', {})['passed'] = False
        save()
        raise


if __name__ == '__main__':
    main()
