# Inference optimization context

Updated: 2026-10-13. Read `PROGRESS.xml` for structured state and `AGENTS.md`
for the optimization role and operating rules. Always verify the live service:
this document is a record, not proof that a pending candidate is installed.

## Goal and preferences

- Private coding/agent inference with the best useful model/configuration
  combination, balancing quality, concurrency, latency and reliability.
- Six to eight active users, preserving 262,144 context tokens **including
  output**. Do not silently truncate or reduce context to claim success.
- SGLang first; exhaust its reasonable options before migrating. The user has
  had poor experiences with vLLM.
- Static GPU memory fraction stays 0.90. The user runs 10-15-minute coding
  tasks and is reconsidering MTP speed versus its 7% token-pool cost. Harness
  compaction is acceptable, but its overhead and quality impact matter. MTP
  is enabled in the installed DP4 + HiCache experiment. Startup and small
  rank/tool, four near-limit concurrent sessions and same-worker host restoration
  passed; sustained real coding load and broad quality remain pending.
- Preserve `pratt-high`, port 8012 and reasoning/tool-call functionality.

## Hardware and paths

After conversation compaction, start with `/home/advtech/NEXT_EXPERIMENT.md`.
It records the approved DP4+MTP+HiCache plan, installation/rollback and DP-aware
validation requirements. DP4 startup/small-smoke results are recorded below.

- Eight NVIDIA H200 GPUs; `nvidia-smi` reports 143,771 MiB per GPU.
- About 2.0 TiB host RAM; about 1.9 TiB available before HiCache. Roughly
  1.1 TiB is reclaimable buffer/page cache, not permanently unavailable RAM.
- Hugging Face cache: `/raid/advtech/cache/huggingface`, mounted at
  `/root/.cache/huggingface` in the container.
- Authoritative service: `/etc/systemd/system/sglang-glm53.service`.
- Container: `sglang-glm53`; direct API: `http://127.0.0.1:8012`.
- Experiment artifacts: `/home/advtech/sglang-glm53-fp8-tests/` (mode 0700).
- Copies such as `/home/advtech/sglang-glm53.service` and older GLM-5.2 notes
  are historical and may not match the effective service.

## Last validated deployment, before HiCache

Full `zai-org/GLM-5.3`, not GLM-5.3-Flash. Native FP8 model weights, 78 layers,
512-dimensional MLA latent cache and 64 rotary dimensions; indexer top-k is
shared at frequency four. SGLang reports version 0.5.18.

Pinned image:

```text
lmsysorg/sglang@sha256:9e148f5ac788e856a06166bd6347a831831eb9fcfab4d1770874823a7c29a1a1
```

Image ID:

```text
sha256:cb940197908678dacb041556b64798cc0493d547816a8e8a6731fa37dddfc4fb
```

Key settings:

```text
--tp 8
--kv-cache-dtype fp8_e4m3
--dsa-prefill-backend flashmla_sparse_q8
--dsa-decode-backend flashmla_kv
--mem-fraction-static 0.90
--cuda-graph-max-bs 32
--context-length 262144
--chunked-prefill-size 8192
--max-running-requests 6
--max-queued-requests 8
--watchdog-timeout 180
--soft-watchdog-timeout 60
--reasoning-parser glm45
--tool-call-parser glm47
```

MTP is off: no speculative flags. Waiting timeout is 120 seconds via
`SGLANG_REQ_WAITING_TIMEOUT`. Metrics and request-time statistics are enabled.
In that baseline, HiCache/cache reporting were off and the GPU pool was
705,024 tokens, with engine-reported startup headroom 9.477 GB per rank.

HiCache 128 is now installed. The unit and effective flags match the candidate;
the server became ready at 14:58:45 UTC. The GPU pool is 618,752 tokens.
Host restoration, near-limit/tool smoke checks and the six-session warmed
90k-token gate passed. Sustained real-user load and broad accuracy are not
qualified. The no-HiCache profile remains the rollback.
Static unit syntax, protected-setting comparisons, XML parsing and a host-RAM
margin preflight passed; see `sglang-glm53-fp8-tests/hicache128-preflight.json`.

## Measured progression

| Profile | GPU token pool | Single-request short test | Concurrent short test |
| --- | ---: | ---: | ---: |
| Original BF16 KV + MTP, cap 4 | 384,640 | 190.95 tok/s | Not measured |
| FP8 KV + MTP, cap 4 | 660,032 | 165.35 tok/s | 449.74 aggregate tok/s at c4 |
| FP8 KV, no MTP, cap 4 | 705,024 | 96.89 tok/s | 328.14 aggregate tok/s at c4 |
| FP8 KV, no MTP, cap 6 | 705,024 | Not repeated | 427.51 aggregate tok/s at c6 |

Short tests used three measured rounds, 28-token prompts and 256 forced output
tokens per request. Rates are client-observed, not production throughput.
Raw native `decode_throughput` sometimes has implausible values in this build.

Correctly chat-formatted 260k-token retrieval and OpenAI tool-call checks passed
with FP8 both with and without MTP. No-MTP near-limit response took 31.07 seconds,
including 29.55 seconds of cold prefill.

The selected six-request profile also passed a long-session test: six
independent 100k-token prefixes were warmed and checked for correct retrieval,
then generated 512 tokens simultaneously. Metrics confirmed six active and
zero queued requests; aggregate rate was 412.40 tok/s, median per-request rate
68.88 tok/s, peak active token usage 0.86, with zero retractions. Each prompt
had a distinct early prefix, preventing accidental sharing of the large KV
footprints. Sequential cold requests took 9.20-11.67 seconds each.

These are smoke/limited performance tests, not sustained-load or full accuracy
qualification. Six full 262k contexts need about 1.57M tokens; eight need about
2.10M. They cannot all fit in the earlier 705k or current 619k GPU pool. A 262k per-request
limit does not guarantee every active user can occupy it simultaneously.

## HiCache-128 experiment approved by the user

Keep the selected profile and add:

```text
--enable-hierarchical-cache
--hicache-size 128
--hicache-write-policy write_through
--hicache-io-backend kernel
--hicache-mem-layout page_first
--enable-cache-report
```

The candidate also adds Docker `--ulimit memlock=-1:-1`; the prior container
had a 65,536-byte soft/hard memory-lock limit. New limits were verified as
unlimited. This removes a potential pinned-
memory limit without changing the bounded cache allocation. All other serving
settings and timeouts are unchanged.

No L3 storage backend initially. Local installed code has a DSA HiCache stack
with a main MLA host pool plus separate DSA-indexer host buffers. The flags
and allocation behavior were inspected and startup verified. Runtime checks
passed the scoped gates below. `hicache-size` overrides ratio-based sizing.

**Units and footprint:** this build multiplies `hicache-size` by 1e9. Thus 128
means 128 decimal GB for the main KV pool **per TP rank**, not 128 GiB and not
128 GB for the node. Additional indexer buffers are allocated separately.
With TP8 and this model, the expected node allocation is about 1.23 trillion
bytes (roughly 1.12 TiB), plus metadata/temporary overhead. The retained host
prefix capacity is roughly 2.5M unique tokens, not eight times that: TP-only
MLA/DSA buffers are replicated across ranks. Measure actual startup logs and
host memory before accepting these estimates.

At preflight, about 2.105 trillion bytes of host memory were available. The
estimated 1.230-trillion-byte cache would leave roughly 875 billion bytes
available before additional allocation overhead. This is an estimate, not a
runtime allocation result.

### Observed HiCache tradeoff and restoration

Logs confirm 2,501,568 host slots, 128.00 GB main KV and 25.76 GB indexer
buffers per rank: about 1.23 TB decimal total. Host `Shmem` is about 1.231 TB
and `MemAvailable` about 852 GB decimal after allocation. `free` classifies
much of this pool as shared/cache, not ordinary process "used" memory. Swap
usage was about 12 MB; this is not evidence that the KV tier is actively
swapped. See `hicache128-allocation.json`.

GPU capacity fell 12.24%, from 705,024 to 618,752. Local source
`kv_cache_configurator._should_elide_dsa_index_k` disables indexer-buffer
elision under HiCache, increasing per-token GPU storage. Do not force elision
on without validating the host/indexer transfer integration.

The optional JIT copy path does not support this 656-byte FP8 MLA layout;
generic fallback kernels are used. They passed the forced-eviction probe:
after three distinct 260k-token pressure conversations, the target was
restored with zero device hits and 259,968 host-hit tokens. All codes in the
final answer and the tool-call check were correct, with no retractions.
End-to-end response was 32.18 seconds cold, 1.47 seconds GPU-warm, and 3.38
seconds RAM-restored (9.52x faster than cold). Server-measured prefill/restore
time was 30.63 versus 1.18 seconds; this is not a client streaming TTFT metric.

Short benchmarks passed: c1 98.18 tok/s, c6 410.57 aggregate / 68.48 median
per-request tok/s. The six-session long gate uses 90k per request; six 100k
requests with 512 output tokens would consume over 97% of this smaller pool.
This changed workload is not a matched comparison to the earlier 100k test.

The long gate passed: six genuinely active requests, zero queued, six correct
warmup retrievals, 89,984 cached tokens per timed request, 512 generated tokens
each, and no retractions. Aggregate rate was 409.98 tok/s, median per-request
68.47 tok/s, peak active token usage 0.88. Some warmups reused partial prefixes
from earlier probes and are not cold-prefill measurements. Post-test checks
found the service ready, no OOM kill, no runtime errors/watchdogs, and no active
or queued requests. See `hicache128-validation-summary.json`.

Given the new GPU-capacity cost, any approved MTP comparison should use the
same six-user context sizes on both profiles and leave output headroom; an
80k/session matched starting point is safer than assuming six 100k sessions
remain comfortable. The next approved candidate is now DP4+MTP+HiCache with
four active requests, not a TP-only MTP A/B. It is installed; see the next section.

## Current plan: DP4 attention + MTP + HiCache

**Verified live on 2026-10-13:** the user installed the candidate. Container
started at 15:50:15 UTC and the ready log appeared at 15:59:43 UTC (~9.5 min).
The authoritative unit matches the candidate and uses the same pinned image.
`/server_info` is ready with TP8/DP4, EAGLE 1/1/2, HiCache 128, FP8 KV,
memory 0.90, context 262,144, `pratt-high` and parsers `glm45`/`glm47`.

- All four worker pools contain **472,768 GPU token slots each**, about
  **1,891,072 unique slots total**. These are separate pools, not one shared
  pool; caches are replicated within each GPU pair.
- Effective admission is **one running request per worker**. Prefill chunks
  are normalized to **2,048**. Engine startup headroom is 9.801-9.846 GB;
  `nvidia-smi` showed about 8.3-8.8 GiB physical free memory per GPU at idle.
- HiCache allocates **2,469,952 host slots per worker**, with MTP packed into
  79 layers. Each TP rank retains 128 GB main plus 25.76 GB indexer buffers;
  node footprint is still about 1.23 TB decimal. Available RAM was ~857 GB
  decimal and swap used ~13.6 MB after smoke.
- Four simultaneous 41-token chat-formatted native probes pinned to ranks
  0-3 returned the correct code, matching `dp_rank` and full prompt lengths.
  Per-DP metrics confirmed four active requests, zero queued and zero
  retractions. OpenAI tool calling and separated reasoning passed.
- Startup emitted 72 soft-watchdog warnings and 144 failed py-spy diagnostic
  attempts; the known 656-byte HiCache JIT fallback warning remains. No
  post-readiness runtime OOM/CUDA/traceback/watchdog/retraction issue was found.
- Subsequent scoped runtime tests passed four simultaneous 260k prompts and
  same-worker DP0 host restoration, as described below. This is not sustained
  production or broad coding-quality qualification. Context remains 262,144;
  400k is not enabled or validated.

Evidence: `glm53-inference/evidence/dp4-startup-validation-summary.json` and
`dp4-startup-smoke-20261013T160727Z.json`. The project now includes
`glm53-inference/tests/dp4_smoke_test.py`; the older long-context helpers remain
TP-only, while `dp4_profile_test.py` now handles DP-aware profiling. One short
sample had implausible native throughput,
so no speed comparison is inferred from it.

### Current-config scoped runtime comparison (17:45-17:52 UTC)

The user requested tests on the current config and a comparison. No service,
model, context, image, MTP or HiCache settings were changed. The new profile
helper pins ranks and deduplicates TP replicas within each DP worker; nine
offline guard/accounting tests passed. Idle/queue checks ran before each test.

| Current DP4 + MTP + HiCache workload | Aggregate tok/s | Median per-request tok/s |
| --- | ---: | ---: |
| Historical short prompt, c1, 256 forced output | 93.51 | 93.52 |
| Historical short prompt, c4, 256 forced output each | 336.61 | 86.08 |
| Four independently GPU-warmed 100k prompts, 512 forced output each | 360.63 | 91.92 |
| Four independently GPU-warmed 260k prompts, 512 forced output each | 352.00 | 89.60 |

Each timed workload used three rounds and client-observed time/actual output.
Short prompts matched the earlier raw 28-token speed-only test, not a chat
quality gate. Current short c1/c4 rates are 43.4% / 25.2% below historical
TP-only FP8+MTP (165 / 450), but near the TP-only no-MTP rates (97 / 328).
**This does not isolate HiCache overhead**: DP attention and normalized
prefill chunks differ too. No engine change was made to create an A/B.

Both long tests began with four simultaneous natural, chat-formatted retrieval
requests with distinct early prefixes and zero prefix-cache hits. All four
recalled beginning/middle/end codes and processed exactly 100k or 260k input
tokens. Cold batch wall times were 19.52 / 54.96 s, including natural output.
Metrics proved four active workers and no queueing. Warmed timed requests
reported 99,968 / 259,968 **device** hits each, zero host hits, and every
512-token output. MTP draft acceptance was about 93.0% / 92.2%. Four 260k
prompts represent 1.04M independent active prompt tokens, about 55% of each
private GPU pool. Six/eight active users remain untested under this profile.

For descriptive context, the older TP-only no-MTP six-user 100k gate ran at
412 aggregate / 68.9 per-request tok/s, and HiCache six-user 90k at 410 / 68.5.
Those are **not matched comparisons** to four-user DP4 results.

Same-worker HiCache restoration was proved on DP0 using a fresh 260k target,
GPU-warm reuse, then two distinct 260k pressure conversations on DP0. The
returning target had **259,968 host hits and zero device hits**, with correct
recall in every stage and no retractions:

- Cold response: **53.80 s**, server prefill **51.24 s**.
- GPU-warm response: **1.49 s**, server prefill **0.086 s**.
- RAM-restored response: **3.29 s**, server restore/prefill **1.29 s**.

Natural answer lengths differed (202 / 120 / 176 output tokens), so the
prefill/restore intervals better isolate cache benefit. This is not streaming
TTFT. The host benefit is now demonstrated for DP4+MTP on DP0, not yet DP1-3.

Post-load rank probes, OpenAI tool parsing and separated reasoning passed.
The service remained ready/idle, with zero cumulative retractions and no
runtime errors/watchdog warnings since test start. Available RAM was ~857 GB
decimal, swap use ~13.6 MB, and minimum physical GPU free memory ~5.35 GiB
(5480 MiB), lower than startup but without OOM. Unit hash/image identity and
effective settings were verified unchanged.

**Decision:** retain current profile and HiCache while measuring real coding
workloads. Capacity and recovery improved, but DP4 is not an automatic speed
upgrade. An isolated HiCache-on/off comparison requires approval, a restart
and matched tests. No 400k context or sustained-production claim is made.
Evidence: `glm53-inference/evidence/dp4-runtime-comparison-20261013T175226Z.json`,
which links all per-test reports and historical baselines.

The user requests retaining HiCache, enabling lightweight MTP and testing DP4.
Candidate: `sglang-glm53-fp8-tests/dp4-mtp-hicache128.service`. It retains the
model, pinned image, FP8 KV/backends, 0.90 memory, 262,144 initial context,
HiCache 128 and API identity. It adds `--dp-size 4 --enable-dp-attention`,
explicit no-DeepEP `--moe-a2a-backend none`, EAGLE 1/1/2, and an initial
four-running-request cap. EP stays at its default one; this is not a copy of
the published DP8/EP8 recipe. The queue launch setting stays eight.

At preparation, installed-build argument construction and static systemd
verification passed without loading weights. The launch prefill chunk of 8192
is automatically normalized to 2048 per worker. Static grouping expects
attention TP2. An early DSA warning prints the global TP value; verify actual
groups after startup rather than treating that warning as proof. The exact
prechange unit is `pre-dp4-mtp-hicache128.service`.

Four ~400k contexts are an aspiration, not a measurement. MLA KV is replicated
within each pair, so a DP8 pool does not simply double. Profile first at 262k,
then decide whether a larger-context candidate has enough output headroom.
No 400k limit or runtime success is currently claimed.

Pin requests to `routed_dp_rank` 0-3 and check `dp_rank` metadata. Existing
helpers need DP-aware routing/activity accounting: a max across metric series
cannot prove four worker activity, and an unpinned eviction probe would rotate
requests across workers. Production session-affinity routing is not installed.
Do not use DP8-only `SGLANG_DP_USE_GATHERV` for attention TP2.

Read `/home/advtech/NEXT_EXPERIMENT.md` for rollback commands and the remaining
ordered test gates. Scoped four-near-limit concurrency and DP0 host restoration
are complete; sustained coding load, DP1-3 restoration, six/eight users and
larger-context validation remain pending.

HiCache protects reusable prefixes after GPU eviction. It does not directly
increase active GPU decoding capacity, and host-to-GPU restoration needs GPU
slots. It may modestly change the startup GPU budget. RAM contents are lost
on restart. Prefix reuse needs matching token sequences; changing early system
prompts, tools or history can invalidate a match.

Pinned allocation and RAM reclaim can extend startup and teardown. Do not
blindly copy NUMA settings from other hosts. Check local CPU/GPU topology and
transfer bandwidth if restoration is slow. Stop if allocation fails, runtime
errors appear or host memory pressure becomes unsafe.

The essential test is **evict and restore**, not just a warmed request:
cache a near-limit chat-formatted document, process distinct conversations
until its GPU prefix is evicted, then confirm `cached_tokens_details.host` or
equivalent host-hit metrics are nonzero and the final answer remains correct.
Compare restoration latency against cold prefill and repeat the six-user test.

## Permissions, deployment and rollback

The account cannot run noninteractive sudo. The user elected to run privileged
commands personally; do not use Docker host-root access to bypass that choice.
Before restart, check the metrics and drain active/queued requests. A full-model
parallel canary cannot be assumed to fit on these same eight GPUs.

The prior validated TP-only HiCache profile (preferred rollback from DP4) is:

```text
/home/advtech/sglang-glm53-fp8-tests/fp8-no-mtp-6-hicache128.service
```

Installation commands used by the user (do not rerun on a busy server):

```bash
sudo install -o root -g root -m 0644 /home/advtech/sglang-glm53-fp8-tests/fp8-no-mtp-6-hicache128.service /etc/systemd/system/sglang-glm53.service
sudo systemctl daemon-reload
sudo systemctl restart sglang-glm53.service
```

Rollback to the last validated six-request profile **without HiCache**:

```bash
sudo install -o root -g root -m 0644 /home/advtech/sglang-glm53-fp8-tests/fp8-no-mtp-6.service /etc/systemd/system/sglang-glm53.service
sudo systemctl daemon-reload
sudo systemctl restart sglang-glm53.service
```

`original.service` is the older BF16+MTP baseline, not the preferred rollback
for the HiCache experiment. It references mutable `latest`; do not pull or
retag that image during these comparisons.

An exact read-only prechange snapshot is also retained as
`/home/advtech/sglang-glm53-fp8-tests/pre-hicache128.service`, together with
`pre-hicache128-server-info.json` for comparing the next startup.

## Existing test commands

```bash
python3 -B /home/advtech/sglang-glm53-fp8-tests/benchmark.py --label PROFILE --concurrency 6
python3 -B /home/advtech/sglang-glm53-fp8-tests/smoke_test.py --label PROFILE --context-tokens 260000
python3 -B /home/advtech/sglang-glm53-fp8-tests/concurrency_test.py --label PROFILE --concurrency 6 --prompt-tokens 100000 --output-tokens 512
```

Use unique labels to preserve reports. The raw repetitive failed probe was
retained but is not used as a valid chat-quality gate. A dedicated forced-
eviction/host-restore check has passed:

```bash
python3 -B /home/advtech/sglang-glm53-fp8-tests/hicache_test.py --label PROFILE --prompt-tokens 260000
```

## Sources

- https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.3
- https://docs.sglang.io/docs/advanced_features/hicache_best_practices
- https://docs.sglang.io/docs/advanced_features/hicache_design
- `/home/advtech/sglang-glm53-fp8-tests/results.md` and its JSON evidence.
