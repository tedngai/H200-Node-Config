# Measured results — 2026-10-18

Scoped validation of the cookbook H200 / Flash-1017 / FP4 / low-latency recipe
with the user-requested DSPARK, engram host-table and HiCache additions.

**Verdict:** the recipe starts and serves correctly on this 8xH200 host. Chat,
separated reasoning, OpenAI tool calls, 100k/400k retrieval, four independent
400k prefixes and four-active warmed decode all passed. A scoped HiCache host
hit was observed. The original mr16 16-request pass completed all requests but
peaked at **14 running + 1–2 queued**. Phase 1 (`--min-free-slots-delay 0`)
then proved **16 simultaneous active warmed 400k decodes** in a 2048-token
replay (peak running 16, queue 0, retracted 0). The 512-token warm pass still
peaked at 15 because server-side tokenization/ingress staggered admission.
Sixteen simultaneous **cold** 400k prefills remain unproven. A full 400k
host-restoration test was **not** run; see the HiCache section for why.

Prepared next step: **phase 2a**, a single `--context-length 409600` change.
It is not installed. See `docs/DECODE_PROFILE.md` and
`docs/CONTEXT_409600_TEST.md`.

## Identity

| Item | Value |
| --- | --- |
| Model | `deepseek-ai/DeepSeek-V4.1-Flash-1017` |
| Checkpoint snapshot | `1bae0e1088993b9b82de08f59de5be5a34e42170` |
| Checkpoint bytes | 510,286,023,000 tensor bytes; 48 shards; verified |
| Image | `lmsysorg/sglang@sha256:4a5d132a06a77c8331e15845f2e925adc788b00105097ad55409afa3f4fa4860` |
| SGLang build | `0.0.0.dev1+gda64c5cbb`, CUDA 13.0.3 |
| API | `http://127.0.0.1:8013`, served name `dsv41-flash-1017` |
| Startup | first start 14:42:34→14:52:39 UTC, **585 s**; phase 1 restart 20:04:52→20:14:30 UTC, **578 s** |

## Effective engine state

From `/server_info`, the startup log and `/v1/models`:

| Setting | Effective value |
| --- | --- |
| Context length | **1,048,576** (checkpoint default; `/v1/models.max_model_len`) |
| TP / EP | 8 / 8, DP1 |
| Max running requests | **4** in the baseline; **16** in mr16 and phase-1 nodelay |
| Min free slots delay | unset/default in mr16; **0** in phase 1 |
| Max total tokens | **45,420,032** (4-request start); **44,957,952** (mr16 and phase-1 nodelay) |
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

## 16-request test at 400k

The service was restarted with only `--max-running-requests` changed from 4 to
**16**. All other settings were identical.

### Cold pass

- 16 distinct prompts, 399,970–399,979 prompt tokens each.
- All 16 returned their three marker codes.
- Batch wall: **995.5 s** (16.6 min).
- Prefill admission was heavily staggered; server queue durations reached
  **918 s**. This was **not** 16 simultaneous cold prefills.

### Warm pass

The same 16 prefixes were replayed. All had **399,872 cached device tokens**
each.

| Measurement | First warm pass | Replay warm pass |
| --- | ---: | ---: |
| Batch wall | 35.003 s | 34.268 s |
| Aggregate tok/s | 290.762 | 317.527 |
| Median per-request tok/s | 17.05 | 16.95 |
| Peak running | **14** | **14** |
| Peak queued | **2** | **1** |
| Peak retracted | 0 | 0 |
| Peak `token_usage` | 0.18 | 0.16 |
| Peak `full_token_usage` | — | 0.125 |
| Peak `swa_token_usage` | 0.18 | 0.158 |

**Important:** 16 requests were submitted and all completed, but the engine did
**not** demonstrate 16 simultaneous active decode. The peak observed was
**14 running + 1–2 queued**. The likely cause is the DSpark/DFlash
min-free-slots admission policy, which delays fresh admissions until several
slots free; it is not the `max_running_requests=16` cap itself. A guaranteed
16-active test would require disabling that delayer
(`--min-free-slots-delay 0`) and another restart.

Scaling is also clear: going from 4 active to 14 active raised aggregate
throughput from ~191 to ~291–318 tok/s, but per-request throughput fell from
~47.6 to ~17 tok/s. Sixteen concurrent 400k sessions are not a low-latency
coding profile.

After the 16-request warm pass, HBM free was **~5.0 GB/GPU** and host available
was **~613 GB**. There were no OOMs or retractions.

## Phase 1 — `--min-free-slots-delay 0`

Installed candidate:
`configs/systemd/candidates/sglang-dsv41-8013-mr16-nodelay.service`. The only
change from mr16 is `--min-free-slots-delay 0`. The service restarted at
20:04:52 UTC and was ready at 20:14:30 UTC (**578 s**). `/server_info` confirmed
`max_running_requests=16`, `min_free_slots_delay=0`,
`prefill_decode_interval=0`, `chunked_prefill_size=8192`,
`swa_prefix_tails=null`, with HiCache and DSpark unchanged.

### 16 × 400k cold pass

- 16 distinct prompts, **399,970–399,979** prompt tokens each.
- Batch wall: **991.414 s**; all 16 recalled their three marker codes.
- Engine-reported queue durations: **0.01–913.1 s**; forward durations:
  **63.2–990.1 s**. Client-observed completions were ~976–990 s.
- This is still a staggered admission/prefill pipeline, **not** 16
  simultaneous cold 400k prefills. Do not cite the 991 s wall as 16-way cold
  prefill scaling.

### 512-token warm pass

- Same 16 prefixes, all **399,872 device-cached tokens** each; scheduler queue
  was 0.01 s per request.
- Batch wall: **35.173 s**; aggregate **290.367 tok/s**; median **16.377
  tok/s** per request.
- Peak sampled running: **15**; peak queue **0**; peak retracted **0**;
  peak `token_usage` 0.18.
- Client threads started all 16 requests within 0.09 s, but server-side
  tokenization/ingress staggered scheduler entry. The engine decode log ramped
  **2 → 7 → 15**, and the first 512-token request finished at 14.7 s before the
  last request was admitted. This pass therefore does **not** prove 16 active.

### 2048-token warm replay

The same 16 prefixes were replayed with **2048** forced output tokens.

| Measurement | Result |
| --- | ---: |
| Batch wall | **96.254 s** |
| Aggregate | **372.979 tok/s** |
| Median per request | **22.878 tok/s** |
| Per-request range | 21.35–27.71 tok/s |
| Peak running | **16** |
| Peak queued | **0** |
| Peak retracted | **0** |
| Peak `token_usage` | 0.22 |
| Cached tokens each | **399,872** device |
| Completion tokens each | **2048** |

Sixteen requests were simultaneously active from about **20.3 s** to
**76.6 s** into the pass, a **~56 s** plateau. This is the direct
16-active-decode proof for the 400k prefixes. It is still a synthetic warmed
replay, not sustained real coding load.

### Mixed prefill/decode baseline

The same warmed prefixes were decoded while four fresh 400k prompts were
admitted. Output was 2048 forced tokens in both passes.

| Pass | Batch wall | Aggregate | Median | Peak running | Peak queue | Retracted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Warmed decode only | 31.483 s | **268.669** | **67.203** | 4 | 0 | 0 |
| Warmed decode + 4 cold 400k | 279.123 s | **29.556** | **7.388** | 7 | 3 | 0 |

The mixed/baseline ratios were **0.11** for both aggregate and median
throughput. The four cold prompts all recalled their codes; their batch wall
was **247.601 s**, with all four completing at ~247.4–247.6 s. This is severe
prefill/decode interference and is the phase 1 comparison point for the phase 2
overlap candidate. It is not evidence that overlap settings will fix it.

### Post-load state

- HBM free: **5,161–5,545 MiB/GPU** (about 5.0–5.4 GiB).
- Host available: **617 GiB**; container RSS **1.341 TiB**.
- Post-ready runtime error scan: **0 matches**.
- Post-load `smoke` and `tools` probes passed, including separated reasoning,
  OpenAI tool call and second tool turn.
- Known non-fatal messages: the previous container's shutdown traceback
  (sigterm watchdog, `SystemExit: 0`) at 20:04:50 UTC, before the new container
  started; and the FABRIC→POSIX fd fallback on all ranks at 20:13:18 UTC.

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

- Post-ready runtime scan found no OOM, Traceback, RuntimeError, CUDA error,
  watchdog event or retraction; see
  `evidence/postload-phase1-20261018T210803Z/errors.txt`.
- Known non-fatal messages: the previous container's shutdown traceback
  (sigterm watchdog, `SystemExit: 0`) at 20:04:50 UTC, before the phase 1
  container started; and the FABRIC→POSIX fd fallback at 20:13:18 UTC.
- Startup also printed `User lacks permission to set NUMA affinity` (container
  lacks `--cap-add SYS_NICE`).
- Post-load state was ready and idle: 0 running, 0 queued, 0 retracted.
- After phase 1, HBM free was **5.0–5.4 GiB/GPU**, host available **617 GiB**,
  container RSS **1.341 TiB**. The mr16 post-load values were ~5.0 GiB free
  HBM/GPU and ~613 GiB host available.

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
- `evidence/long-400000-c16-warm-20261018T160533Z.json` — mr16 16-request cold+warm.
- `evidence/replay-c16-warm-20261018T161736Z.json` — mr16 16-request warm replay.
- `evidence/postload-mr16-20261018T161850Z/` — post-mr16 metrics, GPU/RAM, errors.
- `evidence/long-400000-c16-warm-20261018T204521Z.json` — phase 1 16×400k cold + 512 warm.
- `evidence/replay-c16-warm-20261018T205717Z.json` — phase 1 2048-token 16-active replay.
- `evidence/mixed-prefill-decode-w4-c4-20261018T210728Z.json` — phase 1 mixed baseline.
- `evidence/postload-phase1-20261018T210803Z/` — post-phase-1 metrics, GPU/RAM, errors.
- `evidence/postload-smoke-phase1-20261018T210748Z.json`
- `evidence/postload-tools-phase1-20261018T210748Z.json`
- `evidence/phase1-nodelay-summary-20261018T210803Z.json` — compact phase 1 summary.
- `evidence/postload-smoke-20261018T150748Z.json`
- `evidence/postload-20261018T150754Z/` — post-load metrics, GPU/RAM, logs.

## What this does not prove

- Sustained real coding/agent load or broad model-quality equivalence.
- **16 simultaneous cold 400k prefills**; cold admission remained staggered.
- Sustained 16-active coding sessions; the 16-active result is a ~56 s synthetic
  warmed decode replay.
- A 400k HiCache host restoration (only a 256-token host hit was observed).
- Exact 1,048,576-token occupancy.
- Production readiness behind the `pratt-high` gateway or on port 8012.
- That the phase 2 overlap settings improve the 0.11 mixed/baseline ratio.
