#!/usr/bin/env python3
"""Prove host-prefix restoration after eviction without flushing live caches."""

import argparse
import json
import math
import pathlib
import time

from benchmark import request
from concurrency_test import build_prompts


def generate(base_url, prompt):
    start = time.monotonic()
    response = request(base_url, '/generate', {
        'input_ids': prompt,
        'sampling_params': {'temperature': 0, 'max_new_tokens': 1024},
        'stream': False,
    })
    meta = response.get('meta_info', {})
    answer = response.get('text', '').rsplit('</think>', 1)[-1]
    details = meta.get('cached_tokens_details') or {}
    prefill_seconds = None
    if 'prefill_finished_time' in meta and 'forward_entry_time' in meta:
        prefill_seconds = meta['prefill_finished_time'] - meta['forward_entry_time']
    return {
        'elapsed_seconds': time.monotonic() - start,
        'prefill_or_restore_seconds': prefill_seconds,
        'retrieval_passed': all(code in answer for code in ('MAPLE-4821', 'RIVER-7306', 'CLOUD-9158')),
        'device_cached_tokens': details.get('device', 0),
        'host_cached_tokens': details.get('host', 0),
        'answer_text': answer,
        'meta_info': meta,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:8012')
    parser.add_argument('--prompt-tokens', type=int, default=260000)
    args = parser.parse_args()
    if not args.label or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in args.label):
        parser.error('Invalid report label')
    info = request(args.base_url, '/server_info')
    if not info.get('enable_hierarchical_cache') or not info.get('enable_cache_report'):
        parser.error('HiCache and cache reporting must be enabled')
    if args.prompt_tokens < 2000 or args.prompt_tokens + 1024 > info['context_length']:
        parser.error('Invalid prompt size or context limit exceeded')
    pressure_count = math.ceil(info['max_total_num_tokens'] / args.prompt_tokens)
    prompts = build_prompts(args.prompt_tokens, pressure_count + 1, namespace=args.label)
    path = pathlib.Path(__file__).parent / f'{args.label}-host-restore-{args.prompt_tokens}.json'
    report = {
        'label': args.label,
        'prompt_tokens': args.prompt_tokens,
        'pressure_conversations': pressure_count,
        'config': {key: info.get(key) for key in (
            'max_total_num_tokens', 'context_length', 'mem_fraction_static',
            'max_running_requests', 'kv_cache_dtype', 'speculative_algorithm',
            'enable_hierarchical_cache', 'hicache_size', 'hicache_write_policy',
            'hicache_io_backend', 'hicache_mem_layout',
        )},
        'pressure': [],
    }

    def record(stage, result):
        print(json.dumps({
            'stage': stage,
            'elapsed_seconds': result['elapsed_seconds'],
            'prefill_or_restore_seconds': result['prefill_or_restore_seconds'],
            'retrieval_passed': result['retrieval_passed'],
            'device_cached_tokens': result['device_cached_tokens'],
            'host_cached_tokens': result['host_cached_tokens'],
            'num_retractions': result['meta_info'].get('num_retractions'),
        }), flush=True)
        path.write_text(json.dumps(report, indent=2) + '\n')
        if not result['retrieval_passed'] or result['meta_info'].get('prompt_tokens') != args.prompt_tokens:
            raise RuntimeError(f'{stage} failed retrieval or exact prompt-length validation')

    try:
        report['cold'] = generate(args.base_url, prompts[0])
        record('cold_target', report['cold'])
        report['gpu_warm'] = generate(args.base_url, prompts[0])
        record('gpu_warm_target', report['gpu_warm'])
        for index, prompt in enumerate(prompts[1:]):
            result = generate(args.base_url, prompt)
            report['pressure'].append(result)
            record(f'pressure_{index + 1}', result)
        report['restored'] = generate(args.base_url, prompts[0])
        record('restored_target', report['restored'])

        tool = request(args.base_url, '/v1/chat/completions', {
            'model': 'pratt-high',
            'messages': [{'role': 'user', 'content': 'Call lookup_room for room B204. Do not answer in prose.'}],
            'tools': [{'type': 'function', 'function': {
                'name': 'lookup_room',
                'description': 'Look up the location of a room by identifier.',
                'parameters': {'type': 'object', 'properties': {'room_id': {'type': 'string'}}, 'required': ['room_id'], 'additionalProperties': False},
            }}],
            'tool_choice': {'type': 'function', 'function': {'name': 'lookup_room'}},
            'temperature': 0,
            'max_tokens': 512,
            'stream': False,
        })
        calls = tool['choices'][0]['message'].get('tool_calls', [])
        tool_passed = False
        for call in calls:
            function = call.get('function', {})
            try:
                arguments = json.loads(function.get('arguments', '{}'))
            except (TypeError, json.JSONDecodeError):
                continue
            tool_passed |= function.get('name') == 'lookup_room' and arguments.get('room_id') == 'B204'
        report['tool_call'] = {'passed': tool_passed, 'tool_calls': calls, 'usage': tool.get('usage')}
        phases = [report['cold'], report['gpu_warm'], *report['pressure'], report['restored']]
        retractions = sum(phase['meta_info'].get('num_retractions', 0) for phase in phases)
        restored = report['restored']
        passed = restored['host_cached_tokens'] > 0 and restored['retrieval_passed'] and tool_passed and retractions == 0
        report['summary'] = {
            'passed': passed,
            'host_cached_tokens_on_restore': restored['host_cached_tokens'],
            'device_cached_tokens_on_restore': restored['device_cached_tokens'],
            'cold_elapsed_seconds': report['cold']['elapsed_seconds'],
            'gpu_warm_elapsed_seconds': report['gpu_warm']['elapsed_seconds'],
            'restored_elapsed_seconds': restored['elapsed_seconds'],
            'cold_vs_restore_end_to_end_speedup': report['cold']['elapsed_seconds'] / restored['elapsed_seconds'],
            'cold_prefill_seconds': report['cold']['prefill_or_restore_seconds'],
            'restored_prefill_seconds': restored['prefill_or_restore_seconds'],
            'total_retractions': retractions,
            'tool_call_passed': tool_passed,
        }
        path.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'report': str(path), 'summary': report['summary']}, indent=2), flush=True)
        if not passed:
            raise SystemExit(1)
    except Exception as error:
        report['error'] = str(error)
        path.write_text(json.dumps(report, indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()
