# GLM-5.3 FP8 KV / MTP comparison

**Approved next:** 400,000-token context, four active requests, DP4/MTP/HiCache
retained. Context-only candidate and exact 262k rollback are prepared in
`glm53-inference/configs/systemd/candidates/`; user install/restart is pending.
Live context is still 262,144. See `/home/advtech/NEXT_EXPERIMENT.md` for commands
and queue details: eight waiting per worker (up to 32), 120-second timeout.

**Current experiment:** TP8 + DP4 attention, lightweight MTP and HiCache
128 GB/rank is installed and ready. Each worker has 472,768 GPU slots, one
active request allowed, and a 2,048-token effective prefill chunk. Four
simultaneous 260k retrievals passed, with warmed generation ~352 aggregate /
89.6 per-request tok/s. DP0 HiCache restored 259,968 host tokens with zero
device hits (3.29 s versus 53.80 s cold); post-load tools/reasoning passed.
Context stays 262,144. Sustained coding load, six/eight users, DP1-3 restore
and 400k are unqualified. New helper: `glm53-inference/tests/dp4_profile_test.py`.
See the runtime comparison section in `results.md`; resume from
`/home/advtech/NEXT_EXPERIMENT.md`. New code/evidence is primarily in
`/home/advtech/glm53-inference/`; older instructions below describe historical
TP-only experiments and must not be used unchanged to qualify DP4.

The original configuration is saved as `original.service`. That historical
baseline used BF16 KV, EAGLE MTP 1/1/2, TP8, a 262,144-token context,
`mem-fraction-static=0.90`, and four running requests.

The last validated baseline is FP8 KV, MTP off, TP8, six running and eight
queued requests, 262,144 context and memory fraction 0.90. HiCache 128 is now
installed and passed scoped validation; `results.md` and `/home/advtech/PROGRESS.xml`
record the current state.

Both initial candidates pin the exact Docker image used by the baseline.
They preserved the context, memory fraction, prefill chunk size, graph batch
limit, four running requests, eight queued requests, timeouts, metrics and API
identity. The six-request profile was added and tested after those comparisons.

## First candidate: FP8 KV with MTP

Run these commands with sudo when requests have drained:

```bash
sudo install -o root -g root -m 0644 /home/advtech/sglang-glm53-fp8-tests/fp8-mtp.service /etc/systemd/system/sglang-glm53.service
sudo systemctl daemon-reload
sudo systemctl restart sglang-glm53.service
```

After the engine is ready, the assistant will verify the effective settings,
record capacity, run smoke/long-context tests, and compare a small benchmark.

## Second candidate: FP8 KV without MTP

Only install this after collecting the first candidate's measurements:

```bash
sudo install -o root -g root -m 0644 /home/advtech/sglang-glm53-fp8-tests/fp8-no-mtp.service /etc/systemd/system/sglang-glm53.service
sudo systemctl daemon-reload
sudo systemctl restart sglang-glm53.service
```

## Rollback

```bash
sudo install -o root -g root -m 0644 /home/advtech/sglang-glm53-fp8-tests/original.service /etc/systemd/system/sglang-glm53.service
sudo systemctl daemon-reload
sudo systemctl restart sglang-glm53.service
```

The original service references `lmsysorg/sglang:latest`. Its image identity at
backup time was `lmsysorg/sglang@sha256:9e148f5ac788e856a06166bd6347a831831eb9fcfab4d1770874823a7c29a1a1`.
Do not pull or retag `latest` during this comparison.

## Optional six-request admission test

`fp8-no-mtp-6.service` differs from the four-request no-MTP candidate only in
its description and `--max-running-requests 6`. Install it only after the
four-request no-MTP capacity/correctness checks pass. It keeps the queue at
eight, context at 262,144, and static memory fraction at 0.90. Six admitted
requests share the same token pool; this does not guarantee six independent
262k contexts fit at once.

```bash
sudo install -o root -g root -m 0644 /home/advtech/sglang-glm53-fp8-tests/fp8-no-mtp-6.service /etc/systemd/system/sglang-glm53.service
sudo systemctl daemon-reload
sudo systemctl restart sglang-glm53.service
```

This six-request candidate is now installed. A short six-request benchmark
passed. See `results.md` for the separate long-conversation concurrency gate.

The long-conversation probe warms independent prompts before concurrent
generation, modeling returning coding-agent sessions. It deliberately gives
each conversation a distinct early prefix to prevent shared-prefix caching
from understating memory use. It also samples the running-request gauge:

```bash
python3 -B /home/advtech/sglang-glm53-fp8-tests/concurrency_test.py --label fp8-no-mtp-6 --concurrency 6 --prompt-tokens 100000 --output-tokens 512
```

## Small decode benchmark

```bash
python3 -B /home/advtech/sglang-glm53-fp8-tests/benchmark.py --label fp8-mtp
```

The benchmark sends one warmup and three measured short-prompt requests,
each with 256 forced output tokens. It is a microbenchmark, not a full
quality evaluation or proof of long-context concurrency. Reports contain
server configuration and request metadata, not generated response text.

Speed comparisons use generated token count divided by the client's measured
elapsed time. The engine-provided `decode_throughput` field remains in raw
metadata, but is not used in the summary: this build occasionally reports
implausible values for that field on short requests.

## HiCache 128 GB per TP rank

The user approved this candidate directly, without first testing 64 GB:

```bash
sudo install -o root -g root -m 0644 /home/advtech/sglang-glm53-fp8-tests/fp8-no-mtp-6-hicache128.service /etc/systemd/system/sglang-glm53.service
sudo systemctl daemon-reload
sudo systemctl restart sglang-glm53.service
```

Run after active and queued requests drain. This introduces HiCache,
write-through, kernel I/O, page-first host layout and cache reporting while
preserving the current model, immutable image, TP8, FP8 KV, MTP-off, six
running requests, eight queued requests, context 262,144 and memory fraction
0.90. Docker memlock is unlimited for pinned-memory allocation. The user has
installed it and live verification confirmed the unit and flags.

128 is decimal GB **per rank** for the main host KV pool. DSA indexer buffers
are extra. The estimated node footprint is about 1.23 TB decimal; actual
allocation, restore performance and GPU token capacity must be measured.
This does not make six full-262k active contexts fit in the GPU token pool.

The GPU pool is now 618,752 tokens (12.24% below baseline), because indexer-
cache elision is disabled under HiCache. The optional 656-byte JIT transfer
path falls back. Generic transfers passed the targeted host-restore gate:

```bash
python3 -B /home/advtech/sglang-glm53-fp8-tests/hicache_test.py --label hicache128-no-mtp --prompt-tokens 260000
```

The test never flushes caches: it checks cold and GPU-warm target responses,
evicts GPU pages with distinct conversations, then requires host hits and
correct recall. Restoration returned 259,968 host-hit tokens and took 3.38 s
versus 32.18 s cold. Use 90k prompts for the long six-user gate on this pool;
100k each exceeds the helper's prudent headroom guard.

The six-user 90k gate passed at 409.98 aggregate / 68.47 median per-request
tok/s, with six actual active requests, no queueing and no retractions. This
is not a matched comparison to the previous 100k/user test. MTP remains off.

Rollback to the selected six-request configuration without HiCache:

```bash
sudo install -o root -g root -m 0644 /home/advtech/sglang-glm53-fp8-tests/fp8-no-mtp-6.service /etc/systemd/system/sglang-glm53.service
sudo systemctl daemon-reload
sudo systemctl restart sglang-glm53.service
```

An exact prechange copy is also saved as `pre-hicache128.service`. The broader
role and experiment context are documented in `/home/advtech/AGENTS.md`,
`/home/advtech/PROGRESS.xml` and `/home/advtech/INFERENCE_CONTEXT.md`.
