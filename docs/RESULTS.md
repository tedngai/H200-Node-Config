# Measured results — 2026-10-18

Scoped validation of the cookbook H200 / Flash-1017 / FP4 / low-latency recipe
with the user-requested DSPARK, engram host-table and HiCache additions.

**Verdict:** the recipe starts and serves correctly on this 8xH200 host. Chat,
separated reasoning, OpenAI tool calls, 100k/400k retrieval, four independent
400k prefixes and four-active warmed decode all passed. A scoped HiCache host
hit was observed. A full 400k host-restoration test was **not** run; see the
HiCache section for why.

## Identity

| Item | Value |
| --- | --- |
| Model | `deepseek-ai/DeepSeek-V4.1-Flash-1017` |
| Checkpoint snapshot | `1bae0e1088993b9b82de08f59de5be5a34e42170` |
| Checkpoint bytes | 510,286,023,000 tensor bytes; 48 shards; verified |
| Image | `lmsysorg/sglang@sha256:4a5d132a06a77c8331e15845f2e925adc788b00105097ad55409afa3f4fa4860` |
| SGLang build | `0.0.0.dev1+gda64c5cbb`, CUDA 13.0.3 |
| API | `http://127.0.0.1:8013`, served name `dsv41-flash-1017` |
| Startup | container start 14:42:34 UTC, ready 14:52:39 UTC; **585 s** |

## Effective engine state

From `/server_info`, the startup log and `/v1/models`:

| Setting | Effective value |
| --- | --- |
| Context length | **1,048,576** (checkpoint default; `/v1/models.max_model_len`) |
| TP / EP | 8 / 8, DP1 |
| Max running requests | **4** (explicit; DSpark would otherwise default to 48) |
| Max total tokens | **45,420,032** |
| Max prefill tokens / chunked prefill | 16,384 / 8,192 |
| Page size | 256 |
| Static memory fraction | 0.80 |
| Attention backend | `dsv4` (`DeepseekV4AttnBackend`) |
| MoE runner | `flashinfer_mxfp4`; FP4 experts auto-detected |
| FP8 GEMM runner | `auto` |
| DSpark | enabled, gamma **5**, verify draft tokens **6** |
| SWA bounded replay | enabled; prefill CUDA graph disabled |
| Reasoning parser | `deepseek-v41` |
| Tool parser | `deepseekv41` |
| HiCache | enabled, ratio 2, `direct`, `page_first_direct`, `write_back` |
| Engram host table | enabled, layout `private` |

`/server_info` reports `context_length: null` and `max_total_tokens: null`
because those are derived values; the log and `/v1/models` provide the effective
values above.

## Engram host-table allocation

The two checkpoint tables are ~101.5 GB each (~203.1 GB total). The pinned
image's `private` layout does **not** produce a single 8-way shard per table:

- Each table reports **24,171 MiB resident per rank**, 100% in huge pages, pinned.
- There are **16 mappings** (2 tables × 8 ranks).
- Total host cost: **386,736 MiB = 377.7 GiB ≈ 405.5 GB decimal**.

The measured per-rank size is exactly one quarter of a 101.5 GB table
(24,170 MiB), so the runtime is holding **two replicated 4-way shards** across
the TP8 group, not one 8-way shard. Capacity planning must use the measured
**~386.7 GiB**, not the naive 203 GB checkpoint-byte estimate.

Startup log examples:

```text
engram host table layer 1: layout=private, 24171 MiB resident, 24168 MiB in huge pages (100%), pinned
engram host table layer 14: layout=private, 24171 MiB resident, 24168 MiB in huge pages (100%), pinned
```

No engram gather error, page-backing warning or host OOM was found.

## Memory after startup and load

| Measurement | Value |
| --- | ---: |
| HBM used per GPU at startup | ~120.8 GB |
| HBM free per GPU at startup | ~22.3 GB |
| HBM free per GPU after long-context load | **~7.3 GB** |
| Container RSS | **1.42 TiB** |
| Host available after load | **~665 GB** |
| Swap used | ~112 MB |
| HiCache host pool | **90,840,064 tokens** |

The KV pool/workspace is allocated on first use; free HBM fell from 22.3 GB to
7.3 GB after the first long request. No OOM occurred.

## Interfaces

| Test | Result |
| --- | --- |
| `/v1/models` | `dsv41-flash-1017`, max model len 1,048,576 |
| Plain chat | `SMOKE-OK`, 0.195 s, 6 completion tokens |
| Reasoning `high` | separated `reasoning_content` (31 tokens), correct answer `160 km/h`, 0.591 s |
| Tool call | `finish_reason="tool_calls"`, structured `get_weather({"location":"Beijing"})`, 1.137 s |
| Tool second turn | tool result consumed, final answer returned, 0.364 s, cached 256 tokens |
| Post-load smoke | plain, reasoning and tool call all passed again |

## Short-request throughput

Client-observed, 256 forced output tokens, three rounds, no concurrent load:

| Concurrency | Aggregate tok/s (rounds) | Median per-request tok/s |
| --- | --- | --- |
| c1 | 77.03 / 77.19 / 77.27 | 77.03 / 77.19 / 77.27 |
| c4 | 132.89 / 141.76 / 145.82 | 33.10 / 35.06 / 35.50 |

This is the **low-latency** recipe, not the high-throughput recipe. The c4
numbers are not a claim that the model is throughput-optimal at four users.

## Long-context retrieval

All prompts used natural synthetic records with unique beginning/middle/end
codes and the model's chat encoding. Times are client-observed cold requests.

| Actual prompt tokens | Response time | Recall |
| ---: | ---: | --- |
| 74,921 (superseded probe) | 6.008 s | pass |
| 100,001 | 8.502 s | pass |
| 399,979 | 66.667 s | pass |

The 74,921-token probe was an early length-calibration result; it is retained
but is not the 100k gate. The calibrated 100k and 400k probes passed.

### Four independent 400k prompts — cold

- Prompt tokens: 399,973 / 399,973 / 399,976 / 399,973.
- Batch wall: **246.5 s**.
- All four recalled their three codes.
- Server queue durations reached **181.2 s**, and forward durations ranged from
  **61.6 s to 186.3 s**. This was a staggered prefill admission pipeline, **not**
  four simultaneous cold prefills. Do not cite the 246.5 s batch as four-way
  prefill scaling.

### Same four prefixes — warmed, four-active decode

- Cached prompt tokens: **399,872 device hits each**, zero host hits.
- Output: 512 forced tokens each.
- Batch wall: **11.629 s**.
- Aggregate: **190.986 tok/s**; median per-request: **47.63 tok/s**.
- Live gauges: peak **4 running**, **0 queued**, **0 retracted**.
- Peak token usage: 0.11 of the pool.
- This is the four-active decode result; it used the same four 400k prefixes.

## HiCache

HiCache initialized with a **90,840,064-token** host pool (ratio 2). During the
tests it backed up and evicted **45,056 tokens**, with **0 dropped tokens**.
Post-load host usage was 5,632 tokens.

A genuine but small host hit was observed after GPU eviction:

```text
cached_tokens_details: {"device": 0, "host": 256}
```

That was the post-load tool second turn; the 256-token prefix had been evicted
from the GPU pool and was restored from host DRAM. This demonstrates the host
tier is functional, but it is **not** a 400k host-restoration test.

A large host-restore test was not run because the GPU pool is 45.4M tokens.
Forcing eviction of a 400k prefix would require roughly 45M distinct prompt
tokens of pressure (many rounds of four 400k or 1M prompts). That is a separate
experiment, not implied by this pass.

## Stability observations

- No OOM, Traceback, RuntimeError, CUDA error, watchdog event or retraction was
  found in the container log. The only grep match for `watchdog` is the
  `watchdog_timeout=3600.0` configuration line.
- Startup printed non-fatal warnings:
  - `User lacks permission to set NUMA affinity` (container lacks
    `--cap-add SYS_NICE`).
  - `FABRIC handle export failed ... falling back to POSIX fd transport` during
    CUDA graph capture.
- Post-load state was ready and idle: 0 running, 0 queued, 0 retracted.
- HBM free after load was ~7.3 GB per GPU; host available ~665 GB.

## Evidence

- `evidence/preflight-20261018T132818Z/` — pre-change GLM snapshot, image/flag
  checks.
- `evidence/checkpoint-verify-20261018T133054Z.json` — 48-shard integrity.
- `evidence/startup-20261018T145302Z/` — `/server_info`, metrics, startup log,
  engram/HiCache highlights.
- `evidence/smoke-20261018T145403Z.json`
- `evidence/tools-20261018T145411Z.json`
- `evidence/throughput-c1-20261018T145420Z.json`
- `evidence/throughput-c4-20261018T145434Z.json`
- `evidence/long-100k-20261018T145525Z.json` — 74,921-token calibration probe.
- `evidence/long-100k-20261018T145556Z.json` — calibrated 100,001-token probe.
- `evidence/long-400k-20261018T145610Z.json`
- `evidence/long-400k-c4-20261018T145739Z.json`
- `evidence/long-400k-c4-warm-20261018T150723Z.json`
- `evidence/postload-smoke-20261018T150748Z.json`
- `evidence/postload-20261018T150754Z/` — post-load metrics, GPU/RAM, logs.

## What this does not prove

- Sustained real coding/agent load or broad model-quality equivalence.
- Four simultaneous cold 400k prefills (the cold test was staggered).
- A 400k HiCache host restoration (only a 256-token host hit was observed).
- Exact 1,048,576-token occupancy.
- Production readiness behind the `pratt-high` gateway or on port 8012.
