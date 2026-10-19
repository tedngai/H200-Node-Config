# Measured results — 2026-10-19

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

**Phase 2a was installed and measured** (2026-10-18 22:02 UTC, ready in 574 s):
a single `--context-length 409600` cap. C16 decode rose to **660.5 tok/s**
(1.963× phase 1), the 16-active 2048-token replay reached **704.5 tok/s** with
peak 16 running / 0 queued / 0 retracted, 16 × 400k cold again recalled all
codes, and a 409k near-limit prompt recalled correctly. The mixed/baseline
ratio fell to **0.072** because the warmed baseline improved more than the
mixed case; absolute mixed decode improved about 6%. See the phase 2a section
and `docs/DECODE_PROFILE.md`.

**Phase 2b is installed** on **8012 / pratt-high** (started 2026-10-18
23:31 UTC, ready 23:40 UTC, 562 s). It adds `--prefill-decode-interval 16` and
`--chunked-prefill-size 4096` to phase 2a. The cold-TTFT gate passed
(74.7 s single; four-cold max 268.3 s / wall 268.5 s), and mixed prefill/decode
rose from **31.3 to 292.4 tok/s** (ratio **0.649** vs 0.072). Costs: c16 decode
about 2–3% lower, 16 × 400k cold wall **+9.2%** with serialized completions, and
the 16-active replay wall **81.6 s vs 55.6 s** because one request straggled.
This is a scoped result; it is not production-qualified.

## Identity

| Item | Value |
| --- | --- |
| Model | `deepseek-ai/DeepSeek-V4.1-Flash-1017` |
| Checkpoint snapshot | `1bae0e1088993b9b82de08f59de5be5a34e42170` |
| Checkpoint bytes | 510,286,023,000 tensor bytes; 48 shards; verified |
| Image | `lmsysorg/sglang@sha256:4a5d132a06a77c8331e15845f2e925adc788b00105097ad55409afa3f4fa4860` |
| SGLang build | `0.0.0.dev1+gda64c5cbb`, CUDA 13.0.3 |
| API | `http://127.0.0.1:8012`, served name **pratt-high** (phase 2b); phase 2a test endpoint was 8013/dsv41-flash-1017 |
| Installed context cap | **409,600** (phase 2a, retained by phase 2b); phase 1 used 1,048,576 |
| Startup | first start 585 s; phase 1 578 s; phase 2a 574 s; **phase 2b 562 s** |

## Effective engine state — phase 1 (historical)

The table below records phase 1. Phase 2a changes only `context_length` to
**409600**; every other flag and pool size below is unchanged. Current phase 2a
values are in the phase 2a section.

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

## Phase 2a — context cap 409600 (historical, installed 2026-10-18)

Candidate:
`configs/systemd/candidates/sglang-dsv41-8013-mr16-nodelay-ctx409600.service`.
It is the phase 1 unit plus exactly one engine flag:

```text
--context-length 409600
```

Started at 22:02:18 UTC and was ready at 22:11:52 UTC (**574 s**).
`/v1/models.max_model_len` and `/server_info.context_length` both report
**409600**. The phase 1 settings are otherwise unchanged:
`max_running_requests=16`, `min_free_slots_delay=0`, `chunked_prefill_size=8192`,
`prefill_decode_interval=0`, default 64 SWA prefix tails, DSpark block size 5,
HiCache ratio 2.

Startup memory:

- `DSV4 SWA sizing`: **59904 tokens**, `prefix_tails=64` — unchanged.
- `DSV4 memory calculation`: **44,957,952 full tokens** — unchanged.
- HBM used at idle: **121,334–121,430 MiB/GPU**, versus about 138,000 MiB in
  phase 1. The cap reduces indexer/CUDA-graph scratch outside the KV pool.
- HBM after all long-context load: **136,116–136,596 MiB/GPU**, leaving
  **7,175–7,655 MiB (7.0–7.5 GiB) free** versus 5.0–5.4 GiB in phase 1.
- Host available **616 GB**; container RSS **1.337 TiB**.

### Decode A/B

Same `tests/decode_throughput.py` prompts, temperature 0, 1024 output tokens,
two rounds. Values are aggregate tok/s for rounds 1/2.

| Load | Phase 1 | Phase 2a | Ratio of medians | Phase 2a per-request median |
| --- | ---: | ---: | ---: | ---: |
| c1 | 132.884 / 146.106 | **167.846 / 180.506** | 1.249× | 167.9 / 180.5 |
| c4 | 259.018 / 253.966 | **398.087 / 390.674** | 1.538× | 103.8 / 105.8 |
| c16 | 335.794 / 337.024 | **646.416 / 674.551** | **1.963×** | 42.7 / 44.5 |

All runs reached the requested concurrency with 0 queue and 0 retracted. A
post-load c16 check after all long-context tests measured **673.243 aggregate /
43.758 median tok/s**, still peak 16. This clears the 1.5× c16 gate.

### 16 × 400k cold and warm

- Cold batch wall **978.719 s** (phase 1: 991.414 s); all 16 prompts
  (399,970–399,979 tokens) recalled their three codes.
- Prefill logs show **one new sequence per batch** (777 prefill batches), so
  cold prefill was interleaved/one-at-a-time, **not** a 16-sequence batch. Do
  not claim 16 simultaneous cold prefills.
- 512-token warm pass: **23.131 s**, aggregate **474.665 tok/s**, median
  **28.105 tok/s**, peak running **10**, 0 queue, 0 retracted. Faster decode
  plus staggered ingress let early requests finish before all 16 were admitted;
  this pass does not prove 16 active.
- 2048-token replay: **55.576 s**, aggregate **704.503 tok/s**, median
  **41.426 tok/s**, peak **16 running**, 0 queue, 0 retracted. Sixteen were
  concurrently active from about 14.6 s to 30.3 s. Phase 1 was 96.254 s /
  372.979 / 22.878, so the replay is about **1.73× faster**.
- Cached prompts were 399,872 device tokens each, zero host hits.

### Cold TTFT baseline (streaming)

New probe `tests/cold_ttft.py` measures time from request submission to the
first generated content token. Distinct cold 400k prompts:

| Load | TTFT values | Median | Max | Batch wall | Recall |
| --- | --- | ---: | ---: | ---: | --- |
| c1 | 61.841 s | 61.841 s | 61.841 s | 62.020 s | pass |
| c4 | 62.316 / 123.088 / 183.866 / 243.633 s | 153.477 s | 243.633 s | 243.983 s | pass |

This confirms serialized cold prefill: four requests receive their first token
at roughly 62, 123, 184 and 244 s. The 2b gate is **90 s single / 300 s
worst-of-four / 360 s four-cold batch wall**, with all codes recalled.

### Mixed prefill/decode

Four warmed 2048-token decodes plus four fresh 400k prefills:

| Pass | Batch wall | Aggregate | Median | Peak running | Peak queue | Retracted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Warmed only | 21.732 s | **434.154** | **106.543** | 4 | 0 | 0 |
| Warm + 4 cold 400k | 264.691 s | **31.342** | **7.848** | 7 | 3 | 0 |

The four cold prompts all recalled their codes; cold batch wall was **244.04 s**
(phase 1: 247.601 s). The mixed/baseline ratio fell to **0.072 aggregate /
0.074 median** from phase 1's 0.11. This is not a regression in absolute mixed
decode: mixed aggregate improved about **6.0%** (29.556 → 31.342) and median
about **6.2%** (7.388 → 7.848). The ratio fell because the warmed baseline
improved much more. Prefill/decode interference remains severe and is still the
target of the overlap experiment.

### Near-limit context and rejection

- A 408,985-token prompt returned 40 output tokens (total **409,025**) and
  recalled all three codes in **63.977 s**. It fits under the 409,600 cap.
- A request with `max_completion_tokens=409700` was rejected with HTTP **400**
  (`max_completion_tokens is too large`), not truncated.

### Post-load

- 0 runtime errors after excluding request/response logs and the server_args
  line; idle 0 running / 0 queued / 0 retracted.
- 16 non-fatal one-time compile warnings: `_hc_mix_stats_reduce_kernel` took
  about 2.84–3.43 s to compile after serving started on the first long
  requests. This can add a one-time stall to the first request; it was not a
  runtime failure.
- Post-load smoke, separated reasoning, tool call and second tool turn passed.
- HiCache host use was 20,480 tokens. No large 400k host restoration was run.

## Phase 2b — overlap on 8012/pratt-high (installed 2026-10-19)

Candidate:
`configs/systemd/candidates/sglang-dsv41-8012-pratt-high-mr16-nodelay-ctx409600-overlap2b.service`.
It adds `--prefill-decode-interval 16` and `--chunked-prefill-size 4096` to
phase 2a, serves **8012 / pratt-high**, and keeps default 64 SWA tails.
Started 2026-10-18 23:31:16 UTC; ready 23:40:38 UTC (**562 s**).

| Setting | Value |
| --- | --- |
| `context_length` / `max_model_len` | **409600** |
| `prefill_decode_interval` | **16** |
| `chunked_prefill_size` | **4096** |
| `max_running_requests` / `min_free_slots_delay` | 16 / 0 |
| `swa_prefix_tails` | default 64 |
| SWA pool | **51,712 tokens** (reduced from 59,904 by the smaller chunk) |
| Full KV pool | **45,075,200 tokens** (up from 44,957,952) |
| HiCache host pool | **90,150,400 tokens** |

### Cold TTFT gate

Streaming distinct cold 400k prompts:

| Load | TTFT | Max | Batch wall | Gate | Recall |
| --- | ---: | ---: | ---: | --- | --- |
| c1 | 74.721 s | 74.721 s | 74.915 s | ≤ 90 s | pass |
| c4 | 68.378 / 135.406 / 201.636 / 268.290 s | 268.290 s | 268.498 s | max ≤300, wall ≤360 | pass |

The gate passes, but c4 is ~10% slower than phase 2a (243.633 / 243.983 s).

### Short decode

| Load | Phase 2a | Phase 2b | Ratio |
| --- | ---: | ---: | ---: |
| c1 | 174.176 | **181.144** | 1.040× |
| c4 | 394.381 | **413.995** | 1.050× |
| c16 | 660.484 | **643.358** | 0.974× |

C16 still reaches 16 running with 0 retracted, but the sampler records a
**transient queue of 15** at simultaneous submission. C1/c4 queue gauges also
show a transient 3; it drains. Decode is within a few percent of phase 2a.

### 16 × 400k cold and warm

- Cold batch wall **1068.931 s** (phase 2a: 978.719 s, **+9.2%**); all 16
  prompts recalled.
- Completions are now **serialized** rather than finishing together:
  68, 135, 202, 268, 335, 402, 468, 535, 602, 668, 735, 802, 869, 935, 1002,
  and 1069 s. Early requests get their first token much sooner; the last is
  about 10% slower than phase 2a's all-at-once completion.
- 512-token warm pass: 24.226 s, **452.91 aggregate / 26.548 median tok/s**,
  peak 11 running, 0 queue, 0 retracted.
- 2048-token replay: **81.579 s**, **715.822 aggregate / 42.864 median tok/s**,
  peak **16 running**, 0 queue, 0 retracted. One request straggled to 81.4 s;
  phase 2a wall was 55.576 s, so aggregate improved ~1.6% but wall regressed
  ~47%. This is the main 2b trade-off.

### Mixed prefill/decode

| Pass | Batch wall | Aggregate | Median | Peak running | Peak queue | Retracted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Warmed only | 19.847 s | **450.436** | **112.789** | 4 | 0 | 0 |
| Warm + 4 cold 400k | 288.763 s | **292.388** | **68.797** | 4 | 3 | 0 |

All four cold prompts recalled; cold batch wall **283.96 s** (phase 2a:
244.04 s). The mixed/baseline ratio is **0.649 aggregate / 0.610 median**,
versus phase 2a's **0.072**. Absolute mixed decode is **9.33×** phase 2a. This
is the clear win: the interval keeps the four warmed decodes running while
cold prefills proceed.

### Post-load

- 0 runtime errors after excluding request logs and server_args; idle
  0 running / 0 queued / 0 retracted.
- HBM used **130,068–130,548 MiB/GPU**, free **13,223–13,703 MiB
  (12.9–13.4 GiB)**; host available 617 GB; RSS 1.336 TiB. The cache state
  differs from the phase 2a post-load snapshot, so this is not a direct memory
  A/B.
- 16 one-time `_hc_mix_stats_reduce_kernel` compile warnings remain.
- Post-load smoke, separated reasoning, tool call and second turn pass through
  `pratt-high`.
- A 408,976-token prompt returned 31 output tokens (total **409,007**) and
  recalled all codes in **69.436 s**, confirming the near-limit boundary
  through the `pratt-high` alias.

**Verdict:** TTFT gate and mixed-load goal pass. Keep 2b installed for real
coding evaluation. Do not describe it as production-qualified: cold 16-request
wall is ~9% slower, the replay has a straggler, and c16 decode is ~2–3% lower.

## HiCache

HiCache initialized with a **90,840,064-token** host pool in the 4-request
start (ratio 2); mr16 through phase 2a report **89,915,904**, and phase 2b
reports **90,150,400**. During the tests it backed up and evicted **45,056
tokens**, with **0 dropped tokens**. Post-load host usage was 5,632 tokens in
phase 1, 20,480 in phase 2a and **24,320** in phase 2b.

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
experiment, not implied by this pass. Phase 2a post-load host use was
**20,480 tokens** and phase 2b **24,320**; no large restore was attempted.

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
- Phase 2a post-load: HBM free **7,175–7,655 MiB/GPU (7.0–7.5 GiB)**, host available
  **616 GB**, container RSS **1.337 TiB**, 0 running / 0 queued / 0 retracted.
- Phase 2a runtime error scan: **0 matches**. One non-fatal one-time message
  class remains: `_hc_mix_stats_reduce_kernel` compiled after serving started
  (2.84–3.43 s per rank) on the first long requests.
- Phase 2b post-load: HBM free **13,223–13,703 MiB/GPU (12.9–13.4 GiB)**, host
  available **617 GB**, RSS **1.336 TiB**, 0 running / 0 queued / 0 retracted;
  runtime error scan **0 matches**. The cache state differs from phase 2a, so
  the larger free HBM is not a direct memory A/B.
- Phase 2b still emits the one-time `_hc_mix_stats_reduce_kernel` compile
  warnings (16 lines across ranks).

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
- `evidence/decode-profile-20261018T213400Z/` — kernel profile, traces and scripts.
- `evidence/preflight-ctx409600-20261018T215639Z/` — phase 2a idle preflight.
- `evidence/decode-throughput-baseline-20261018T215614Z.json` — scripted phase 1 decode baseline.
- `evidence/phase2a-prepared-20261018T215639Z.json` — phase 2a preparation summary.
- `evidence/startup-ctx409600-20261018T221345Z/` — phase 2a startup and effective flags.
- `evidence/smoke-20261018T221347Z.json` — phase 2a startup smoke.
- `evidence/tools-20261018T221347Z.json` — phase 2a startup tools.
- `evidence/decode-throughput-ctx409600-20261018T221334Z.json` — phase 2a decode A/B.
- `evidence/cold-ttft-phase2a-c1-20261018T231500Z.json` — single cold 400k TTFT baseline.
- `evidence/cold-ttft-phase2a-c4-20261018T231500Z.json` — four-cold 400k TTFT baseline.
- `evidence/phase2a-ttft-baseline-20261018T231500Z.json` — compact TTFT baseline and 2b gate.
- `evidence/long-400000-c16-warm-20261018T223131Z.json` — phase 2a 16×400k cold + 512 warm.
- `evidence/replay-c16-warm-20261018T224045Z.json` — phase 2a 2048-token 16-active replay.
- `evidence/mixed-prefill-decode-w4-c4-20261018T225014Z.json` — phase 2a mixed result.
- `evidence/near-limit-409k-ctx409600-20261018T225343Z.json` — 409k retrieval.
- `evidence/long-20261018T225126Z.json` — earlier 409k retrieval probe.
- `evidence/over-cap-rejection-ctx409600-20261018T225343Z.json` — HTTP 400 boundary check.
- `evidence/postload-ctx409600-20261018T225142Z/` — post-load metrics, GPU/RAM, errors.
- `evidence/postload-smoke-ctx409600-20261018T225142Z.json`
- `evidence/postload-tools-ctx409600-20261018T225143Z.json`
- `evidence/startup-overlap2b-20261019T011800Z/` — phase 2b startup and flags.
- `evidence/cold-ttft-overlap2b-c1-20261019T012000Z.json` — 2b single cold TTFT.
- `evidence/cold-ttft-overlap2b-c4-20261019T012000Z.json` — 2b four-cold TTFT.
- `evidence/decode-throughput-overlap2b-20261019T012525Z.json` — 2b c16 decode.
- `evidence/decode-throughput-overlap2b-c1c4-20261019T020557Z.json` — 2b c1/c4 decode.
- `evidence/long-400000-c16-warm-20261019T014443Z.json` — 2b 16×400k cold + 512 warm.
- `evidence/replay-c16-warm-20261019T015435Z.json` — 2b 2048-token replay.
- `evidence/mixed-prefill-decode-w4-c4-20261019T020449Z.json` — 2b mixed result.
- `evidence/near-limit-409k-overlap2b-20261019T021000Z.json` — 2b 409k retrieval.
- `evidence/postload-overlap2b-20261019T020500Z/` — 2b post-load metrics/GPU/errors.
- `evidence/postload-smoke-overlap2b-20261019T020500Z.json`
- `evidence/postload-tools-overlap2b-20261019T020500Z.json`
- `evidence/phase2b-summary-20261019T020600Z.json` — compact 2b summary.
- `evidence/decode-throughput-ctx409600-postload-20261018T225237Z.json`
- `evidence/phase2a-summary-20261018T225300Z.json` — compact phase 2a summary.
- `evidence/postload-smoke-20261018T150748Z.json`
- `evidence/postload-20261018T150754Z/` — post-load metrics, GPU/RAM, logs.

## What this does not prove

- Sustained real coding/agent load or broad model-quality equivalence.
- **16 simultaneous cold 400k prefills**; phase 2a and 2b both ran one new
  sequence per batch, and 2b serialized completions further.
- Sustained 16-active coding sessions; the 2b replay peak was 16 running for
  about 20 s, with one request straggling to 81 s.
- A 400k HiCache host restoration (only small host hits were observed; 2b
  post-load host use was 24,320 tokens).
- Exact 409,600-token occupancy; the near-limit probe reached 409,025 total.
- Production qualification behind the gateway, even though the engine now
  serves 8012/pratt-high.
- That 2b's mixed gain transfers to real coding prompts; it is a synthetic
  4-warm/4-cold result (292.4 vs 31.3 tok/s).
- That the one-time `_hc_mix_stats_reduce_kernel` compile stall is eliminated;
  it remains a first-request latency risk until precompiled.
- That the phase 2c SWA-tails candidate helps; it is not installed or tested.
