# GLM-5.3 FP8 KV comparison

## Measurements collected so far

| Setting / measurement | Original BF16 KV + MTP | FP8 KV + MTP | FP8 KV without MTP |
| --- | ---: | ---: | ---: |
| TP size | 8 | 8 | 8 |
| Static memory fraction | 0.90 | 0.90 | 0.90 |
| Context limit | 262,144 | 262,144 | 262,144 |
| Shared token capacity | 384,640 | 660,032 | 705,024 |
| Startup available GPU memory reported by engine | 10.055 GB | 9.816 GB | 9.506 GB |
| Running request cap | 4 | 4 | 4 |
| Queued request cap | 8 | 8 | 8 |
| Short-request end-to-end rate, median | 190.95 tok/s | 165.35 tok/s | 96.89 tok/s |
| Four-request aggregate end-to-end rate, median | Not measured | 449.74 tok/s | 328.14 tok/s |
| Four-request per-request end-to-end rate, median | Not measured | 113.41 tok/s | 82.06 tok/s |
| Retracted requests in short benchmark | 0 | 0 | 0 |

The FP8 pool is 1.716 times the original pool (71.6% more tokens). Capacity
matches the allocation estimate. The speed sample is one warmup followed by
three 28-token-prompt requests with 256 forced output tokens at concurrency
one. It is not a production throughput or quality evaluation. Native engine
`decode_throughput` sometimes reports implausible values in this build, so
comparisons use the client's elapsed time and actual output token counts.

The service became ready at 2026-10-13 13:45:58 UTC after a 5.5-minute reload.
Startup logs contain soft-watchdog stack-dump attempts during startup; the
py-spy diagnostics failed, but startup subsequently completed. No startup
OOM was found. Runtime stability and long-context checks are separate gates.

Removing MTP increased FP8 capacity by 44,992 tokens (6.8%). The no-MTP pool
is 83.3% larger than the original BF16 pool. Single-request short benchmark
speed fell 41.4% relative to FP8+MTP; four-request aggregate speed fell 27.0%.
The no-MTP profile became ready at 2026-10-13 13:58:01 UTC. Its near-limit
chat-formatted retrieval and tool-call probes passed.

## Initial smoke checks

- Tool-call formatting passed: `lookup_room` with `room_id=B204`.
- A raw native-completion request with 260,000 input tokens completed in
  33.28 seconds, with no retractions or runtime OOM. It failed retrieval and
  emitted repetitive filler. This first probe lacked the chat template and
  used a single repeated filler token, so it is not accepted as a valid
  instruction-following quality gate.
- The corrected chat-template probe with natural synthetic archive records
  passed: all three codes, placed at the beginning, middle and end, appeared
  correctly in the final answer after `</think>`. The engine processed all
  260,000 prompt tokens without truncation; response time was 29.67 seconds,
  including 28.39 seconds of cold prefill. No retractions occurred.
- The tool-call check passed again after the long request.
- Four concurrent short requests, repeated for three rounds, completed with
  all 12 requested 256-token outputs and no retractions. Aggregate median
  throughput was 449.74 tok/s; per-request median was 113.41 tok/s.
- No runtime OOM, CUDA error, traceback or watchdog warning was found in the
  journal after 13:46:00 UTC during these checks.

This is a smoke-test pass, not a full accuracy evaluation or sustained-load
qualification. The original failed raw-completion probe is retained for
transparency.

## No-MTP smoke checks

- The corrected 260,000-token chat-formatted retrieval probe passed, with all
  three codes present in the final answer. Response time was 31.07 seconds,
  including 29.55 seconds of cold prefill. The full prompt was processed
  without truncation and with zero retractions.
- The OpenAI tool-call probe passed: `lookup_room` with `room_id=B204`.
- Single-request and four-request short benchmarks generated every requested
  token, with zero retractions. All 12 four-request samples completed.
- No runtime OOM, CUDA error, traceback or watchdog warning was found in the
  journal after the no-MTP server became ready.

## Six-request no-MTP profile

The six-request candidate is installed and became ready at 2026-10-13
14:10:08 UTC. Its effective running-request cap is six, queued cap is eight,
and shared token pool is still 705,024. FP8 KV, MTP-off, TP8, 262,144 context,
and static memory fraction 0.90 are unchanged. Startup GPU headroom was
9.477 GB per rank.

Six concurrent short requests, repeated for three rounds, completed with all
18 requested 256-token outputs and zero retractions. Median aggregate
throughput was 427.51 tok/s; median per-request rate was 71.28 tok/s.

A separate long-conversation concurrency probe passed:

- Six distinct 100,000-token chat-formatted prompts were independently warmed;
  every conversation passed three-code retrieval. Cold requests took
  9.20-11.67 seconds each when submitted sequentially.
- Each prefix diverged immediately after the user role marker, so the large
  cached prefixes could not be shared across the six conversations.
- Six simultaneous warmed generations each produced 512 tokens. All six had
  99,968 cached prompt tokens, representing independent returning sessions.
- Engine metrics confirmed six genuinely active requests at once, with zero
  queued requests. Peak active token usage was 0.86.
- Aggregate throughput was 412.40 tok/s; median per-request rate was
  68.88 tok/s. The concurrent generation interval lasted 7.45 seconds.
- No requests were retracted. Metric sampling had no errors.
- Post-test checks found the server ready, container not OOM-killed, running
  and queued gauges at zero, and no runtime OOM, CUDA error, traceback or
  watchdog warning since this profile became ready.

This confirms six active independently cached ~100k-token sessions on this
profile. It does not measure a simultaneous six-request cold-prefill storm,
sustained real-user load, or six full 262k contexts.

## Outcome

The two FP8 profiles and the six-request no-MTP profile passed the scoped
smoke checks. The six-request no-HiCache profile is the validated rollback.
No eight-request profile has been installed or tested.

At this historical stage, the live service had MTP disabled and a six-request cap.
No result so far establishes six/eight simultaneous full 262k contexts.

## HiCache 128 GB per rank: installed and scoped smoke checks passed

The user requested direct testing of 128 decimal GB per TP rank, skipping the
64 GB stage. The candidate is `fp8-no-mtp-6-hicache128.service`; it preserves
the validated TP8/FP8/no-MTP/six-running/eight-queued/262k/0.90 profile and adds
HiCache, write-through, kernel I/O, page-first layout and cache reporting.
Docker memlock limits are set to unlimited for pinned host-memory allocation.

Expected host allocation is roughly 1.23 TB decimal (about 1.12 TiB), including
the separate DSA indexer pools. Expected unique retained host prefix capacity
is roughly 2.5M tokens. These are code-derived estimates, not measurements.
Host buffers are replicated across TP ranks; HiCache is not extra active GPU
token capacity. Startup may change the GPU budget and take longer.

Static unit verification, exact protected-setting comparisons, XML validation
and host-memory preflight passed. About 2.105 TB decimal were available before
the experiment; the estimated cache would leave about 0.875 TB decimal before
additional overhead. Details are in `hicache128-preflight.json`.

The user installed the candidate; it became ready at 14:58:45 UTC. Its unit
and serving flags match. The exact prior service is saved in
`pre-hicache128.service`; the preferred practical rollback is
`fp8-no-mtp-6.service`. The next gates are startup/footprint checks, near-limit
retrieval/tool checks, six-user comparisons and a proven host-cache restore
after GPU eviction.

Startup logs confirm 2,501,568 host slots, 128.00 GB main KV and 25.76 GB DSA
indexer buffers per rank (about 1.23 TB decimal total). Memlock is unlimited.
Host `Shmem` is about 1.231 TB, with about 852 GB decimal still available;
see `hicache128-allocation.json`.

The GPU pool dropped 12.24% to 618,752 tokens. Source explicitly disables
DSA indexer-cache elision under HiCache. The optional JIT transfer path warns
about the 656-byte layout and falls back to generic kernels.

Short benchmarks passed: c1 98.18 tok/s; c6 410.57 aggregate / 68.48 median
per-request tok/s, all forced outputs completed and no retractions.

The forced-eviction gate passed using a 260k-token target and three distinct
260k-token pressure conversations. The restored target reported zero device
hits and 259,968 host-hit tokens, with all three codes correctly recalled in
the final answer. Tool calling passed and there were no retractions.

| Target state | Client end-to-end response | Server prefill/restore interval |
| --- | ---: | ---: |
| Cold | 32.18 s | 30.63 s |
| GPU-warm | 1.47 s | 0.060 s |
| RAM-restored after GPU eviction | 3.38 s | 1.18 s |

The RAM-restored response was 9.52x faster than cold. These intervals include
different generated answer lengths and are smoke measurements, not a full
latency distribution or a client streaming TTFT benchmark. The direct host-
hit counters establish that a real L2 restore occurred.

The six-session long test passed with 90k tokens per session: six genuinely
active requests, zero queued, six correct warmup retrievals, 89,984 cached
tokens and 512 outputs per timed request, zero retractions, and peak active
token usage 0.88. Throughput was 409.98 aggregate / 68.47 median per-request
tok/s over 7.49 seconds. Warmups could reuse partial prefixes from prior
probes, so they are not treated as uniformly cold requests.

This is not a matched comparison to the earlier 100k/session test. Post-test
checks found the service ready, no OOM kill, no runtime errors/watchdogs and
no queued/active requests. HiCache stays installed after these scoped checks;
sustained real-user load and broader accuracy are not yet qualified.
MTP remains off and is being reconsidered for 10-15-minute coding tasks.

## Approved next profile: DP4 + lightweight MTP + HiCache

The user approved DP4 attention with MTP and HiCache retained. Candidate:
`dp4-mtp-hicache128.service`, using TP8/DP4, no DeepEP (EP default one), EAGLE
1/1/2, FP8 KV, HiCache 128, static memory 0.90, initial context 262,144 and
four active requests (expected one per DP worker; verify after launch).

Static systemd and the installed SGLang argument-constructor checks passed;
`dp4-mtp-hicache128-preflight.json` records the result. No candidate engine or
weights were launched. Constructor normalization changes the 8192 launch
prefill chunk to 2048 per worker. Live service was TP-only and MTP-off at preparation.

At preparation no DP4 capacity, throughput or 400k success was measured. Do not
assume the DP8 journal's ~193k/~207k pool doubles under DP4. Rank-pinned
requests and DP-aware activity/restoration helpers are required. After the
user compacts, resume from `/home/advtech/NEXT_EXPERIMENT.md` and `PROGRESS.xml`.

## DP4 installed: startup and small rank/tool smoke passed

Verified 2026-10-13. Container started at 15:50:15 UTC, ready log at 15:59:43
UTC (~9.5 min). The unit matches `dp4-mtp-hicache128.service`, image identity
is unchanged, and effective settings preserve context 262,144, memory 0.90,
FP8 KV, HiCache 128 per TP rank, alias/port/parsers, with EAGLE MTP 1/1/2.

- Four independent attention-TP2 pools each have **472,768 GPU token slots**:
  **1,891,072 total unique slots**, replicated within each pair, not a shared
  pool. Effective running cap is one per worker; prefill chunk is 2,048.
- Engine startup available memory is 9.801-9.846 GB. Host pools have 2,469,952
  slots per worker, packing target plus MTP (79 layers): 128 GB main plus
  25.76 GB indexer per TP rank, still ~1.23 TB decimal node footprint.
- Four simultaneous rank-pinned, chat-formatted 41-token probes all recalled
  the correct code and matched `dp_rank`/prompt length. Metrics confirmed four
  active requests, zero queued, zero retractions. OpenAI tool calling returned
  correct `lookup_room(B204)` with `finish_reason=tool_calls`; separated
  `reasoning_content` was present.
- Startup had 72 soft-watchdog warnings, 144 failed py-spy diagnostics and
  the known optional 656-byte JIT fallback warning. No post-ready runtime
  OOM/CUDA/traceback/watchdog/retraction issue or container OOM kill was found.
  Host available memory was ~857 GB decimal, swap used ~13.6 MB.

Evidence: `dp4-startup-validation-summary.json`,
`dp4-startup-smoke-20261013T160727Z.json`, and corresponding server snapshot.
Primary new evidence/code also resides under `glm53-inference/`.
An initial local tokenizer serialization failure occurred before sending any
inference requests; its invalid probe record is retained in project evidence.

This is startup plus small smoke, **not a speed benchmark, near-limit load,
DP4+MTP host-restoration qualification or 400k validation**. One short request
again emitted implausible native decode throughput; no performance comparison
is based on it. Context remains 262,144. Next: rank-pinned independent long
retrieval/concurrency and same-rank host restoration before increasing context.
