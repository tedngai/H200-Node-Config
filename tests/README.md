# Tests

All probes use the OpenAI-compatible endpoint and require no third-party Python
packages. Reports are written to `../evidence/` unless `--report` is supplied.

```bash
python3 tests/openai_probe.py --mode smoke
python3 tests/openai_probe.py --mode tools
python3 tests/openai_probe.py --mode throughput --concurrency 1 --output-tokens 256 --rounds 3
python3 tests/openai_probe.py --mode throughput --concurrency 4 --output-tokens 256 --rounds 3
python3 tests/openai_probe.py --mode long --prompt-tokens 100000
python3 tests/decode_throughput.py --concurrency 1,4,16 --output-tokens 1024 --rounds 2
python3 tests/cold_ttft.py --concurrency 1 --target-tokens 400000 --output-tokens 512
```

Modes:

- `smoke` — `/v1/models`, plain chat, separated reasoning, one weather tool call.
- `tools` — tool call plus a second turn carrying the tool result.
- `throughput` — forced-output rounds at the requested concurrency; reports
  client wall time and actual `completion_tokens`.
- `long` — one natural synthetic retrieval prompt with unique beginning,
  middle and end codes; reports the server's own `prompt_tokens`.
- `decode_throughput.py` — fixed coding prompts at c1/c4/c16 with natural
  sampling and live gauges, for A/B comparison of decode settings.
- `cold_ttft.py` — streaming time-to-first-token for distinct cold 400k
  prompts, with recall and batch-wall reporting.

Do not run two modes concurrently on the same engine. The long-context probe is
intentionally single-request; a c4 long test needs a separate admission and
memory review.
