# Decode profile and optimization assessment — 2026-10-18

Scoped diagnostic on the installed phase 1 unit (`mr16-nodelay`), engine idle
before and after. Workload: synthetic short coding prompts, natural sampling,
no `ignore_eos`. Evidence: `evidence/decode-profile-20261018T213400Z/`.
Nothing was installed or restarted.

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

1. **Cap `--context-length` (config only, single flag).** `L` scales with
   `context_len`. Projected from the profile (not measured):

   | Cap | `L` factor | c16 step | c1 step |
   | --- | ---: | ---: | ---: |
   | 1,048,576 (now) | 1.00 | 212 ms | 33 ms |
   | 409,600 | 0.39 | ~113–121 ms (~1.8×) | ~27 ms (~1.2×) |
   | 262,144 | 0.25 | ~93–100 ms (~2.1×) | ~25.5 ms (~1.3×) |

   409,600 matches the 400k approved target. Requests above the cap are
   rejected, not truncated. `/v1/models` will report the new limit; client
   context/compaction limits must be at or below it.
2. **Tune Triton block-FP8 GEMM configs.** All ten `block_shape=[32, 32]`
   dense shapes use default configs on H200 (startup "Config file not
   found"). They are 34% of single-user step time and latency-bound. Tuned
   JSON configs can be mounted read-only without changing the image. Tuning
   needs GPU time while the engine is stopped. Gain is unmeasured.
3. **Raise concurrency only after (1).** The GPU pool is 44.96M tokens versus
   6.55M for 16 × 409,600, so memory is not the limit; per-request step cost
   is. Re-measure c16/c24/c32 with a per-user tok/s floor after the cap.
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

## Effect on the phase 2 plan

Phase 2 (`--prefill-decode-interval 16`, `--chunked-prefill-size 4096`,
`--swa-prefix-tails 1024`) targets prefill/decode interference, not the decode
bottleneck. The interval multiplies decode step time: at today's ~200 ms c16
step, 16 rounds is ≥3.2 s between prefill chunks, and a cold 400k prompt has
~98 chunks at 4096. Cold time-to-first-token could grow by minutes under load.
Recommended order:

1. Phase 2a: phase 1 + `--context-length 409600` only; re-run c1/c4/c16 short,
   16 × 400k + 2048 replay, mixed test, tools/reasoning.
2. Phase 2b: add interval + chunk size, with an explicit cold-TTFT ceiling.
3. Phase 2c: `--swa-prefix-tails 1024` separately, as a multi-session
   prefix-reuse test (~8 GB/GPU, affordable given pool headroom).

## Acceptance note

Realistic coding output accepted 3.1–5.7 tokens per verify step (c16 median
3.91), close to the forced-output 400k tests. The earlier 77 tok/s c1 result
used generic prose; coding c1 measured 141–160 tok/s.

## Minor host observations

- `kcompactd1` (NUMA node 1) was at 100% CPU with ~6.7 h cumulative time; host
  free memory is ~9 GB beside ~1.17 TB shmem. `compact_stall=0`, so no direct
  allocation stalls were observed. Low priority.
- The container lacks `SYS_NICE` (`User lacks permission to set NUMA affinity`).
