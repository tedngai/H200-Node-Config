#!/usr/bin/env python3
"""Synthetic near-limit retrieval and OpenAI tool-call smoke tests."""

import argparse
import json
import pathlib
import subprocess
import time

from benchmark import request


def make_prompt(token_count):
    tokenizer_code = r'''
import json
import random
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('zai-org/GLM-5.3', local_files_only=True)
encode = lambda text: t.encode(text, add_special_tokens=False)
document = (
    'Read this synthetic document and report the three codes associated with ALPHA, BETA, and GAMMA. '
    'The routine archive records are unrelated to those codes.\nALPHA code is MAPLE-4821.\n'
    '__LEFT_FILLER__'
    '\nBETA code is RIVER-7306.\n'
    '__RIGHT_FILLER__'
    '\nGAMMA code is CLOUD-9158.\n'
    'Question: What are the codes for ALPHA, BETA, and GAMMA? Return the three codes exactly.'
)
formatted = t.apply_chat_template([{'role': 'user', 'content': document}], tokenize=False, add_generation_prompt=True)
beginning, remainder = formatted.split('__LEFT_FILLER__')
middle, end = remainder.split('__RIGHT_FILLER__')
rng = random.Random(20261013)
teams = ['archival', 'facilities', 'library', 'conservation', 'inventory', 'maintenance']
objects = ['wooden crates', 'paper folders', 'spare lamps', 'canvas covers', 'empty cabinets', 'metal shelves']
places = ['north wing', 'south annex', 'east courtyard', 'west storeroom', 'central hall', 'basement office']
records = [
    f'Record {i}: In the fictional campus archive, the {rng.choice(teams)} team catalogued '
    f'{rng.choice(objects)} from the {rng.choice(places)}. Routine notes describe maintenance '
    'schedules, weather conditions and the movement of ordinary supplies.\n'
    for i in range(1, 8193)
]
print(json.dumps({
    'filler': encode(''.join(records)),
    'beginning': encode(beginning),
    'middle': encode(middle),
    'end': encode(end),
}))
'''
    result = subprocess.check_output(
        ['docker', 'exec', 'sglang-glm53', 'python3', '-B', '-c', tokenizer_code],
        text=True,
        timeout=60,
    )
    parts = json.loads(result)
    filler = parts['filler']
    if not filler:
        raise RuntimeError('Filler tokenization is empty')
    remaining = token_count - sum(len(parts[key]) for key in ('beginning', 'middle', 'end'))
    if remaining < 0:
        raise ValueError('Requested context is too small')
    filler = (filler * ((remaining + len(filler) - 1) // len(filler)))[:remaining]
    left = remaining // 2
    tokens = parts['beginning'] + filler[:left] + parts['middle'] + filler[left:] + parts['end']
    assert len(tokens) == token_count
    return tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:8012')
    parser.add_argument('--context-tokens', type=int, default=260000)
    args = parser.parse_args()
    if not args.label or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in args.label):
        parser.error('Invalid report label')
    info = request(args.base_url, '/server_info')
    if args.context_tokens < 1 or args.context_tokens + 1024 > info['context_length']:
        parser.error('Prompt plus output must fit the configured context')
    prompt = make_prompt(args.context_tokens)
    start = time.monotonic()
    generation = request(args.base_url, '/generate', {
        'input_ids': prompt,
        'sampling_params': {'temperature': 0, 'max_new_tokens': 1024},
        'stream': False,
    })
    text = generation.get('text', '')
    answer = text.rsplit('</think>', 1)[-1]
    meta = generation.get('meta_info', {})
    expected = ['MAPLE-4821', 'RIVER-7306', 'CLOUD-9158']
    retrieval = {
        'prompt_tokens_requested': args.context_tokens,
        'elapsed_seconds': time.monotonic() - start,
        'passed': all(code in answer for code in expected) and meta.get('prompt_tokens') == args.context_tokens,
        'found_codes': {code: code in answer for code in expected},
        'response_text': text,
        'answer_text': answer,
        'meta_info': meta,
    }

    start = time.monotonic()
    tool_response = request(args.base_url, '/v1/chat/completions', {
        'model': 'pratt-high',
        'messages': [{'role': 'user', 'content': 'Call lookup_room for room B204. Do not answer in prose.'}],
        'tools': [{
            'type': 'function',
            'function': {
                'name': 'lookup_room',
                'description': 'Look up the location of a room by its identifier.',
                'parameters': {
                    'type': 'object',
                    'properties': {'room_id': {'type': 'string'}},
                    'required': ['room_id'],
                    'additionalProperties': False,
                },
            },
        }],
        'tool_choice': {'type': 'function', 'function': {'name': 'lookup_room'}},
        'temperature': 0,
        'max_tokens': 512,
        'stream': False,
    })
    choice = tool_response['choices'][0]
    calls = choice['message'].get('tool_calls', [])
    tool_passed = False
    for call in calls:
        function = call.get('function', {})
        try:
            arguments = json.loads(function.get('arguments', '{}'))
        except (json.JSONDecodeError, TypeError):
            continue
        tool_passed |= function.get('name') == 'lookup_room' and arguments.get('room_id') == 'B204'
    tool_test = {
        'passed': tool_passed,
        'elapsed_seconds': time.monotonic() - start,
        'finish_reason': choice.get('finish_reason'),
        'tool_calls': calls,
        'usage': tool_response.get('usage'),
    }
    report = {
        'label': args.label,
        'prompt_format': 'model chat template with synthetic archive records',
        'config': {key: info.get(key) for key in ('max_total_num_tokens', 'mem_fraction_static', 'context_length', 'kv_cache_dtype', 'speculative_algorithm')},
        'retrieval': retrieval,
        'tool_call': tool_test,
    }
    path = pathlib.Path(__file__).parent / f'{args.label}-smoke-{args.context_tokens}.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({
        'report': str(path),
        'config': report['config'],
        'retrieval': {key: value for key, value in retrieval.items() if key not in ('response_text', 'answer_text')},
        'tool_call': tool_test,
    }, indent=2))
    if not retrieval['passed'] or not tool_test['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
