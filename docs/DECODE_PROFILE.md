# Decode profile and optimization assessment — 2026-10-18

Scoped diagnostic on the phase 1 unit (`mr16-nodelay`), engine idle before and
after. Workload: synthetic short coding prompts, natural sampling, no
`ignore_eos`. Evidence: `evidence/decode-profile-20261018T213400Z/`.
The phase 2a context cap was subsequently installed and measured; see the
phase 2a section below and `docs/RESULTS.md`.

## Finding: decode cost scales with batch size, not context

| Load | Est. step time | Median per-request | Aggregate |
| --- | ---: | ---: | ---: |
| c1, short coding | ~30–33 ms | 141–160 tok/s | — |
| c4, short coding | 65.7 ms | 62.0 tok/s | 240.9 tok/s |
| c16, short coding | 201.6 ms | 20.1–21.8 tok/s | 309–340 tok/s |
| c4, 400k warmed (phase 1) | 65.2 ms | 67.2 tok/s | 268.7 tok/s |
| c16, 400k warmed (phase 1) | 201.9 ms | 22.9 tok/s | 373.0 tok/s |

Step time = `running × accept_len / gen_tps` from the decode log. Short and
400k contexts give the same step time; each extra request adds ~11–12 ms per
verify step. During c16 the GPUs were 100% busy but only 13–18% memory-bandwidth
utilized at ~400 W of 700 W, i.e. latency/compute-bound kernels.

### Scripted phase 1 baseline for A/B

`tests/decode_throughput.py` was added to repeat the same coding prompts,
temperature 0, 1024 output tokens and live gauges. Phase 1 baseline
(`evidence/decode-throughput-baseline-20261018T215614Z.json`, two rounds):

| Load | Aggregate tok/s (rounds 1/2) | Median per request | Peak running |
| --- | ---: | ---: | ---: |
| c1 | 132.884 / 146.106 | 132.894 / 146.113 | 1 |
| c4 | 259.018 / 253.966 | 65.819 / 64.559 | 4 |
| c16 | 335.794 / 337.024 | 21.727 / 21.968 | **16** (0 queue, 0 retracted) |

## Root cause: the H200 indexer scores the full configured context width

Torch profile, TP0, 6 decode steps:

| Kernel group | c1 ms/step | c16 ms/step |
| --- | ---: | ---: |
| `_fp4_index_logits_kernel` (DSA indexer) | 8.60 | **136.39 (64%)** |
| Triton block-FP8 dense GEMM | **11.29 (34%)** | 11.51 |
| MoE (CUTLASS + routing) | 4.56 | 11.14 |
| torch top-k + index gather | 1.64 | 14.05 |
| all-reduce | 1.77 | 6.05 |
| sparse MLA attention itself | ~0.9 | ~1.2 |
| engram host-table gather | 0.01 | 0.02 |
| **Kernel-busy total** | **33.2** | **212.2** |

Indexer kernel grids show the scored width:

| Layers/step | Width `L` | c1 rows | c16 rows | c16 ms/call |
| --- | ---: | ---: | ---: | ---: |
| 5 (ratio 1) | 1,048,896 | 6 | 96 | 20.98 |
| 3 (ratio 2) | 524,480 | 6 | 96 | 10.50 |

On SM90 DeepGEMM's FP4 MQA-logits kernel is unavailable, so SGLang uses a
Triton fallback (`sm90_fp4_indexer.py`) that launches over every position up
to `L = req_to_token.shape[1] // ratio`. Under CUDA-graph replay that width is
the **configured context length** (1,048,576 + extra), not the real length.
Each request contributes 6 rows (DSpark verify tokens), and with TP8 every rank
scores every row. Candidate-consuming layers also score the full width before
masking to their 16k candidate window.

## Opportunities (ranked)

1. **Cap `--context-length` — installed as phase 2a.** Measured on 2026-10-18:
   409,600 raised c16 aggregate decode from 336.4 to **660.5 tok/s (1.96×)**,
   c4 from 256.5 to 394.4 (1.54×), and c1 from 139.5 to 174.2 (1.25×).
   `max_model_len` is 409600 and requests above it are rejected, not truncated;
   client context/compaction limits must be at or below the cap. The SWA pool
   (`swa_tokens=59904`) and full KV pool (44,957,952) are unchanged.
   A 262,144 cap is untested; the phase 1 projection was about 2.1× at c16.
   See `docs/CONTEXT_409600_TEST.md` and `docs/RESULTS.md`.
2. **Tune Triton block-FP8 GEMM configs.** All ten `block_shape=[32, 32]`
   dense shapes use default configs on H200 (startup "Config file not
   found"). They were 34% of the phase 1 c1 step and remain about 11.3 ms, now
   a larger share of the reduced c1 step. Tuned JSON configs can be mounted
   read-only without changing the image. Tuning needs GPU time while the engine
   is stopped. Gain is unmeasured.
3. **Raise concurrency after phase 2a.** The GPU pool is still 44.96M tokens
   versus 6.55M for 16 × 409,600, so memory is not the limit; per-request step
   cost is. Re-measure c16/c24/c32 with a per-user tok/s floor after the cap.
   C16 already runs at about 42–45 tok/s per request with 0 queue.
4. **DP attention (larger change, later).** It would remove the 8× redundant
   per-rank indexer work, and the cookbook's high-concurrency Blackwell cells
   use it. In this build it is incompatible with
   `--enable-decoder-swa-bounded-replay`, so it trades cold-prefill speed and
   needs full DSpark/HiCache/engram revalidation.
5. **Engine-side indexer fix (needs an approved patched image or upstream fix).**
   Early-exit beyond each row's visible length and score only candidate blocks;
   cost would then follow actual context instead of the cap.

Not worth pursuing now: engram back to HBM (gather is 0.02 ms/step); MoE
backend changes (5% at c16). V4.1 rejects two-batch overlap.

## Phase 2a measured (historical; phase 2b now installed)

Phase 2a is installed at `context_length=409600`; startup was 574 s. Measured
on the same scripted prompts:

| Load | Phase 1 aggregate | Phase 2a aggregate | Ratio |
| --- | ---: | ---: | ---: |
| c1 | 139.495 (median of rounds) | **174.176** | 1.249× |
| c4 | 256.492 | **394.381** | 1.538× |
| c16 | 336.409 | **660.484** | **1.963×** |

The 2048-token 16-active replay improved from 96.254 s / 372.979 tok/s to
**55.576 s / 704.503 tok/s** (1.73× wall, 1.89× aggregate), with peak 16
running, 0 queue and 0 retracted. A post-load c16 run measured 673.243 tok/s.

From the measured rate and the observed accept length (~4.4), the c16 step is
now roughly **105–110 ms**, versus 212 ms profiled in phase 1. This is derived
from throughput, not a fresh kernel profile. The remaining c16 cost is expected
to be spread across dense FP8 GEMM, MoE, top-k/index gather and all-reduce;
a post-cap profile was not taken.

Cold 16 × 400k completed in 978.719 s with all codes recalled, but prefill logs
still show one new sequence per batch, so it was interleaved rather than a
16-sequence cold batch. The 512-token warm pass reached 474.665 tok/s but only
peak 10 running because faster decode plus ingress stagger finished early
requests first. The replay is the 16-active proof.

Mixed prefill/decode remains the weak point: the warmed baseline rose to
434.154 tok/s, the mixed case to 31.342 tok/s, and the ratio fell to **0.072**
(phase 1: 0.11). Absolute mixed decode improved about 6%; the ratio fell because
the baseline improved more. Prefill interference is still severe.

Memory improved: idle HBM was about 121.4 GB/GPU instead of ~138 GB, and
post-load free HBM was **7.0–7.5 GiB/GPU (minimum 7.0)** instead of 5.0–5.4 GiB, with the
same 44,957,952-token KV pool and `swa_tokens=59904`. The cap reduced
indexer/CUDA-graph scratch, not KV capacity.

## Phase 2b measured (installed)

Phase 2b is installed on **8012 / pratt-high**. It adds
`--prefill-decode-interval 16` and `--chunked-prefill-size 4096`, keeps
`context_length=409600` and default tails, and changes the pools to
`swa_tokens=51,712` / `max_total_num_tokens=45,075,200`.

- Cold TTFT gate passed: single **74.721 s** (≤90); four-cold
  **68.4/135.4/201.6/268.3 s**, max 268.3 (≤300), wall 268.5 (≤360).
- Short decode vs phase 2a: c1 **181.1** (1.04×), c4 **414.0** (1.05×),
  c16 **643.4** (0.974×). C16 has a transient submission queue, no retractions.
- 16 × 400k cold: **1068.9 s**, all recalled, completions serialized ~67 s apart;
  worst request ~10% slower than phase 2a, early requests much faster.
- 2048 replay: **715.8 aggregate / 42.9 median tok/s**, peak 16, queue 0, but
  wall **81.6 s** because one request straggled to 81.4 s (phase 2a 55.6 s).
- Mixed 4-warm/4-cold: **292.4 aggregate / 68.8 median**, ratio **0.649**,
  vs phase 2a **31.3 / 0.072**. Absolute mixed decode is **9.33×** phase 2a;
  cold wall 284.0 s, all recalled, no retractions.

Verdict: the overlap setting fixes most of the mixed prefill/decode collapse,
at the cost of serialized cold completions and a replay straggler. It is the
right candidate for real coding evaluation, not a production qualification.

## Effect on the next experiments

Phase 2b is installed. Do not add more scheduler flags before real coding
traffic is measured on `pratt-high`; the synthetic mixed gain is large but the
cold/replay trade-offs are real.

Recommended order:

1. Run real client coding/agent tasks through 8012/pratt-high and measure
   end-to-end task time, TTFT, and any queue growth.
2. Phase 2c: test `--swa-prefix-tails 1024` separately as a multi-session
   prefix-reuse experiment (~8 GB/GPU; 16 active requests do not inherently
   need 1024 tails).
3. Tune the Triton block-FP8 GEMM configs and measure c24/c32 if more
   concurrency is needed. Neither changes model weights or serving shape.

## Acceptance note

Realistic coding output accepted 3.1–5.7 tokens per verify step (c16 median
3.91), close to the forced-output 400k tests. The earlier 77 tok/s c1 result
used generic prose. Phase 1 coding c1 measured 132.9–146.1 tok/s; phase 2a c1
measured **167.8–180.5 tok/s**. Phase 2a c16 per-request medians were
**42.7–44.5 tok/s** with 0 queue and 0 retracted. Phase 2b c1/c4 medians were
**172.7–189.6 / 105.5–107.6 tok/s**, and c16 **41.5–43.9 tok/s** with a
transient submission queue.

## Minor host observations

- `kcompactd1` (NUMA node 1) was at 100% CPU with ~6.7 h cumulative time; host
  free memory is ~9 GB beside ~1.17 TB shmem. `compact_stall=0`, so no direct
  allocation stalls were observed. Low priority.
- The container lacks `SYS_NICE` (`User lacks permission to set NUMA affinity`).
