"""Offline checks of DP gauge accounting and per-worker admission guards."""

import unittest
from unittest.mock import patch

from dp4_profile_test import generate, historical_prompt, parse_metrics, validate_config


class ProfileTests(unittest.TestCase):
    def test_deduplicates_tp_replicas(self):
        text = '\n'.join(
            f'sglang:num_running_reqs{{model_name="pratt-high",dp_rank="{dp}",tp_rank="{tp}"}} 1.0'
            for dp in range(4) for tp in (dp * 2, dp * 2 + 1))
        parsed = parse_metrics(text)
        self.assertEqual(sum(parsed['num_running_reqs'].values()), 4)

    def test_missing_rank_is_not_activity_proof(self):
        with self.assertRaises(ValueError):
            parse_metrics('sglang:num_running_reqs{model_name="pratt-high",tp_rank="0"} 4.0')

    def test_other_model_ignored(self):
        self.assertEqual(parse_metrics('sglang:num_running_reqs{model_name="other",dp_rank="0"} 5.0'), {})

    def info(self):
        return {'status': 'ready', 'dp_size': 4, 'enable_dp_attention': True,
                'context_length': 262144, 'allow_auto_truncate': False,
                'internal_states': [{'effective_max_running_requests_per_dp': 1,
                                     'memory_usage': {'token_capacity': 472768}} for _ in range(4)]}

    def test_four_near_limit_requests_use_four_private_pools(self):
        validate_config(self.info(), 4, 260000, 1024)

    def test_context_guard(self):
        with self.assertRaises(ValueError):
            validate_config(self.info(), 4, 262000, 1024)

    def test_smallest_worker_pool_guard(self):
        info = self.info()
        info['internal_states'][2]['memory_usage']['token_capacity'] = 280000
        with self.assertRaises(ValueError):
            validate_config(info, 4, 260000, 1024)

    def test_more_than_one_request_per_worker_rejected(self):
        with self.assertRaises(ValueError):
            validate_config(self.info(), 6, 100000, 512)

    def test_historical_prompt_is_unchanged(self):
        self.assertEqual(historical_prompt(0),
                         'Benchmark sample 0. Explain how continuous batching and a key-value cache work in '
                         'a language-model inference server. Give a detailed technical explanation.')

    def test_raw_benchmark_requires_exact_prompt_count(self):
        response = {'meta_info': {'dp_rank': 0, 'prompt_tokens': 27, 'completion_tokens': 256,
                                 'num_retractions': 0}}
        with patch('dp4_profile_test.request', return_value=response):
            result = generate('http://unused', historical_prompt(0), 0, 256, True,
                              expected_prompt_tokens=28)
        self.assertFalse(result['valid'])


if __name__ == '__main__':
    unittest.main()
