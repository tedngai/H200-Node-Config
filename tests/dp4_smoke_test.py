#!/usr/bin/env python3
"""Small DP4 rank-routing/retrieval/tool smoke; not long-context qualification."""

import argparse
import concurrent.futures
import datetime
import json
from pathlib import Path
import re
import subprocess
import threading
import time
import urllib.request

from benchmark import request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8012')
    args = parser.parse_args()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = Path(__file__).resolve().parents[1] / 'evidence'
    output.mkdir(exist_ok=True)
    destination = output / f'dp4-startup-smoke-{stamp}.json'
    info = request(args.base_url, '/server_info')
    keys = ('status', 'version', 'model_path', 'served_model_name', 'context_length',
            'tp_size', 'dp_size', 'ep_size', 'kv_cache_dtype', 'mem_fraction_static',
            'max_running_requests', 'max_queued_requests', 'max_total_num_tokens',
            'chunked_prefill_size', 'speculative_algorithm', 'speculative_num_steps',
            'speculative_eagle_topk', 'speculative_num_draft_tokens',
            'enable_hierarchical_cache', 'hicache_size', 'allow_auto_truncate',
            'reasoning_parser', 'tool_call_parser')
    report = {'checked_at': stamp, 'scope': 'small_rank_routing_retrieval_and_tool_smoke',
              'config': {key: info.get(key) for key in keys}, 'probes': []}
    report['workers'] = [
        {'internal_state_index': i, 'memory_usage': state.get('memory_usage'),
         'effective_max_running_requests_per_dp': state.get('effective_max_running_requests_per_dp')}
        for i, state in enumerate(info.get('internal_states', []))
    ]
    def save():
        destination.write_text(json.dumps(report, indent=2) + '\n')

    save()
    if info.get('status') != 'ready' or info.get('dp_size') != 4 or not info.get('enable_dp_attention'):
        raise RuntimeError('Expected a ready DP4-attention server')
    snapshot = output / f'dp4-server-info-{stamp}.json'
    snapshot.write_text(json.dumps(info, indent=2) + '\n')
    report['server_info_snapshot'] = snapshot.name

    codes = [f'RANK{rank}-CHECK-4821' for rank in range(4)]
    tokenizer_code = """
import json,sys
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('zai-org/GLM-5.3',local_files_only=True)
formatted=[t.apply_chat_template([{'role':'user','content':
    f'Independent startup check {i}. The access code is {code}. '
    'Return the access code exactly and nothing else.'}],
    tokenize=False,add_generation_prompt=True) for i,code in enumerate(json.load(sys.stdin))]
prompts=[t.encode(text,add_special_tokens=False) for text in formatted]
print(json.dumps(prompts))
"""
    prompts = json.loads(subprocess.check_output(
        ['docker', 'exec', '-i', 'sglang-glm53', 'python3', '-B', '-c', tokenizer_code],
        input=json.dumps(codes), text=True, timeout=60))
    metrics, errors = [], []
    stop = threading.Event()
    pattern = re.compile(r'^sglang:(num_running_reqs|num_queue_reqs|num_retracted_reqs)\{([^\n]*)\}\s+([-+\d.eE]+)$', re.MULTILINE)
    def monitor():
        while not stop.is_set():
            try:
                with urllib.request.urlopen(args.base_url + '/metrics', timeout=5) as response:
                    text = response.read().decode()
                values = {}
                for name, labels, value in pattern.findall(text):
                    rank = re.search(r'dp_rank="(\d+)"', labels)
                    if rank:
                        per_rank = values.setdefault(name, {})
                        key = rank.group(1)
                        per_rank[key] = max(per_rank.get(key, 0), float(value))
                metrics.append({'time': time.monotonic(), **values})
            except Exception as error:
                errors.append(str(error))
            stop.wait(0.1)

    def probe(rank):
        start = time.monotonic()
        response = request(args.base_url, '/generate', {
            'input_ids': prompts[rank], 'routed_dp_rank': rank,
            'sampling_params': {'temperature': 0, 'max_new_tokens': 512}, 'stream': False})
        meta = response.get('meta_info', {})
        answer = response.get('text', '').rsplit('</think>', 1)[-1].strip()
        passed = (meta.get('dp_rank') == rank and codes[rank] in answer
                  and meta.get('prompt_tokens') == len(prompts[rank])
                  and meta.get('num_retractions', 0) == 0)
        return {'requested_dp_rank': rank, 'passed': passed, 'answer': answer,
                'elapsed_seconds': time.monotonic() - start, 'meta_info': meta}

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            report['probes'] = list(pool.map(probe, range(4)))
        tool = request(args.base_url, '/v1/chat/completions', {
            'model': 'pratt-high',
            'messages': [{'role': 'user', 'content': 'Call lookup_room for room B204. Do not answer in prose.'}],
            'tools': [{'type': 'function', 'function': {
                'name': 'lookup_room', 'description': 'Look up a room by identifier.',
                'parameters': {'type': 'object', 'properties': {'room_id': {'type': 'string'}},
                               'required': ['room_id'], 'additionalProperties': False}}}],
            'tool_choice': {'type': 'function', 'function': {'name': 'lookup_room'}},
            'temperature': 0, 'max_tokens': 512, 'stream': False})
        choice = tool['choices'][0]
        calls = choice['message'].get('tool_calls', [])
        tool_passed = False
        for call in calls:
            function = call.get('function', {})
            try:
                arguments = json.loads(function.get('arguments', '{}'))
            except (ValueError, TypeError):
                continue
            tool_passed |= function.get('name') == 'lookup_room' and arguments.get('room_id') == 'B204'
        tool_passed &= choice.get('finish_reason') == 'tool_calls'
        report['tool_call'] = {'passed': tool_passed, 'finish_reason': choice.get('finish_reason'),
                               'tool_calls': calls, 'usage': tool.get('usage'),
                               'reasoning_content_present': bool(choice['message'].get('reasoning_content'))}
        report['passed'] = all(probe['passed'] for probe in report['probes']) and tool_passed
    except Exception as error:
        report['error'] = str(error)
        report['passed'] = False
        raise
    finally:
        stop.set()
        monitor_thread.join(timeout=6)
        report['metrics'] = metrics
        report['metric_sampling_errors'] = errors
        report['peak_running_by_rank'] = {
            str(rank): max((sample.get('num_running_reqs', {}).get(str(rank), 0) for sample in metrics), default=0)
            for rank in range(4)}
        report['peak_total_running'] = max((sum(sample.get('num_running_reqs', {}).values()) for sample in metrics), default=0)
        report['peak_total_queued'] = max((sum(sample.get('num_queue_reqs', {}).values()) for sample in metrics), default=0)
        save()
    print(json.dumps({'report': str(destination), 'passed': report['passed'],
                      'probes': report['probes'], 'tool_call': report['tool_call'],
                      'peak_total_running': report['peak_total_running'],
                      'peak_total_queued': report['peak_total_queued']}, indent=2))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
