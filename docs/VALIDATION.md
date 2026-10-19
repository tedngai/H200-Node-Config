# Validation plan and evidence standard

A startup success is not a sustained-load qualification. Each stage below is
reported separately, with the exact effective configuration and whether prefixes
were cold or warm.

## Result summary (2026-10-18)

| Stage | Result |
| --- | --- |
| 0 — Preflight | Passed |
| 1 — Startup/resources | Passed; ready in 585 s first start, 578 s phase 1 restart; 45.4M-token GPU pool |
| 2 — OpenAI interfaces | Passed: chat, separated reasoning, tool call + second turn |
| 3 — Short performance | Passed as a scoped measurement; c1 ~77 tok/s, c4 ~133-146 aggregate |
| 4 — Long context | Passed: 100,001 and 399,979 tokens; four 400k cold prompts recalled |
| 4b — 16-request 400k | **Passed with phase 1**: mr16 peaked at 14 running + 1–2 queued; `--min-free-slots-delay 0` plus a 2048-token replay reached 16 running, 0 queued, 0 retracted |
| 5 — HiCache restoration | **Partial**: 256-token host hit after eviction; no 400k host restore |
| 6 — Engram host table | Passed; 386.7 GiB, 100% huge pages, pinned |
| 7 — Stability/rollback | No post-ready runtime errors; rollback not yet exercised |

Full measurements and evidence are in `docs/RESULTS.md`. The remaining gap is a
large-prefix HiCache host restoration; the 45.4M-token GPU pool makes forced
eviction expensive.

## Stage 0 — Preflight (no GPU change)

- [x] Live GLM service is ready and idle: zero running, zero queued, zero
      retracted requests.
- [x] Record the installed GLM systemd unit, `/server_info`, image digest,
      container inspect, GPU state and host memory before any change.
- [x] Verify `/raid` free space for the 510.3 GB checkpoint.
- [x] Pull `lmsysorg/sglang:dev-dsv41`, record its immutable digest, verify the
      image contains the expected `sglang` CLI.
- [x] Verify the image's `sglang serve --help` accepts every candidate flag.
- [x] Download/verify the checkpoint in the shared HF cache.
- [x] Confirm port 8013 is free and the candidate unit parses.
- [ ] User runs the privileged stop/install/start hand-off.

## Stage 1 — Startup and resources

Record from the startup log and `/server_info`:

- [ ] SGLang version/build and pinned image digest.
- [ ] Model path, served name **pratt-high**, **context cap 409600**, TP8, EP8,
      static fraction 0.80; `/v1/models.max_model_len=409600`.
- [ ] Phase 2b: `prefill_decode_interval=16`, `chunked_prefill_size=4096`,
      `swa_tokens≈51,712`, full pool ≈45,075,200; or the expected values for
      the installed candidate.
- [ ] Resolved attention backend (`dsv4`), MoE runner (`flashinfer_mxfp4`),
      FP8 GEMM backend, DSpark enabled and block size 5.
- [ ] HiCache enabled, ratio 2, direct I/O, `page_first_direct`, `write_back`.
- [ ] Engram host table enabled and `per_rank`; per-rank table allocation.
- [ ] GPU KV token pool per rank and total; HiCache host token slots.
- [ ] Startup time, host RAM/pinned memory, swap, free HBM per GPU.
- [ ] No OOM, no JIT-transfer fallback, no watchdog or CUDA-graph failure that
      affects serving.

## Stage 2 — OpenAI interfaces

- [ ] `/v1/models` lists `dsv41-flash-1017`.
- [ ] Plain chat returns coherent content.
- [ ] `reasoning_effort="high"` returns a separated `reasoning_content` block
      and an answer; the parser is `deepseek-v41`.
- [ ] A weather-style tool definition returns `finish_reason="tool_calls"` with
      a structured `get_weather` call, not raw DSML in `content`.
- [ ] A second turn can feed the tool result back and complete the answer.
- [ ] If time permits, one interleaved image request confirms the vision path;
      this is not a quality benchmark.

## Stage 3 — Short-request performance

Use the model's chat template and natural text. Record client-observed latency
and actual usage tokens, not only native `decode_throughput`.

- [ ] Three cold/warmed c1 rounds, 256 forced output tokens: median and range.
- [ ] c4 if the effective admission limit is at least 4; otherwise record that
      the recipe default queues and do not claim four-user capacity.
- [ ] Metrics prove the number of active requests; HTTP submission alone is not
      proof.
- [ ] DSpark accept length/rate from `meta_info` or metrics.

## Stage 4 — Long-context correctness

- [ ] One cold ~32k-token natural retrieval prompt: recall codes placed at the
      beginning, middle and end.
- [ ] One cold ~100k-token retrieval prompt if the measured KV pool has enough
      headroom; do not silently truncate.
- [ ] Report exact prompt tokens from usage, not character-count guesses.
- [ ] Report cold response time, server prefill time and output tokens.
- [ ] No queueing, retraction or hidden truncation.

Do not claim full 409,600-token occupancy from a 100k test. The engine caps
the model at 409,600 positions in phase 2a; the actual pool and memory headroom
still determine what is servable. A 409,025-token total passed; an over-cap
request must be rejected, not truncated.

## Stage 5 — HiCache restoration

HiCache only counts as validated if a prefix is genuinely evicted from the GPU
pool and later restored from host memory.

- [ ] Warm a long prefix and verify GPU device hits on a repeat.
- [ ] Create enough distinct pressure to evict that prefix from the GPU pool.
- [ ] Repeat the original prefix and require host hits > 0 with device hits 0
      (or the build's equivalent host/device counters).
- [ ] Verify correct retrieval after restoration, not just a fast response.
- [ ] Record cold response, GPU-warm response and host-restored response
      separately.
- [ ] Record peak host RAM and swap; confirm no OOM.

If the test cannot force eviction, report "HiCache initialized but restoration
not demonstrated", not "HiCache works".

## Stage 6 — Engram host-table verification

- [ ] Startup log explicitly reports host-resident engram tables and per-rank
      layout, or the source/allocator confirms it.
- [ ] Host RSS/pinned memory increases by approximately the expected ~203 GB
      total (~25.4 GB/rank).
- [ ] GPU memory does not contain the full engram tables (compare free HBM and
      KV pool with the expected non-engram weight size).
- [ ] A prompt that exercises both engram layers (natural text, not one repeated
      filler token) returns coherent output.
- [ ] No engram gather error, page-backing warning or OOM.

## Stage 7 — Stability and rollback

- [ ] Serve for a scoped period after the load tests; check logs for OOM,
      watchdog, NCCL, CUDA graph and parser errors.
- [ ] Record post-test `/server_info`, metrics, free HBM and host RAM.
- [ ] Stop the test unit cleanly and restart the validated GLM service.
- [ ] Verify GLM `/server_info` is back to the 400k DP4+MTP+HiCache profile and
      the alias/port are unchanged.

## Evidence format

Each run writes a timestamped JSON/Markdown report under `evidence/`. Reports
must include:

- UTC timestamp, image digest, model revision, candidate unit hash;
- exact prompt token counts and output token counts from the server;
- cold/warm/host-restored classification;
- active/queued/retracted request gauges;
- client-observed timings and server timings;
- any error lines, with the failing command retained.

Failed or invalid probes are kept. Do not delete a failed report or overwrite it
with a later pass.
