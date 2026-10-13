# Resume after conversation compaction: approved 400k DP4 + MTP + HiCache

Updated 2026-10-13. DP4 is installed and passed scoped four-near-limit
concurrency, tool/reasoning and DP0 host-restoration tests. Sustained real
coding load, six/eight users and 400k remain unqualified. The user approved
400k/four active requests; that new candidate is prepared, not installed.
Read `AGENTS.md`, `PROGRESS.xml`, and `INFERENCE_CONTEXT.md` first.

## Latest user instruction

"400k context limit, and change to 4 concurrent users. now what happens to
the queued users?"

This approves **400,000 tokens including output and four active requests**,
not four fixed accounts or exclusive whole-agent-task reservations. Four
active requests (one per worker) are already configured. Retain DP4, lightweight
MTP, HiCache 128 per TP rank, full GLM-5.3 and static memory 0.90. Only the
context limit changes. No queue/routing/timeout settings were authorized to change.

## Next candidate: 400k prepared, pending user install

- Candidate: `glm53-inference/configs/systemd/candidates/dp4-mtp-hicache128-400k.service`.
- Exact 262k DP4 rollback:
  `glm53-inference/configs/systemd/candidates/pre-400k-dp4-mtp-hicache128.service`.
- Preflight:
  `glm53-inference/evidence/dp4-mtp-hicache128-400k-preflight-20261013T181759Z.json`.
- Authoritative unit/image and baseline `/server_info` were preserved; static
  systemd and installed-build argument construction passed. The live unit is
  unchanged, and context remains **262,144** until the user installs/restarts.
- Candidate differs only in context **262144 -> 400000** and description.
  Prefill chunk resolves to 2048, max-running remains four, queues remain eight
  per worker, waiting timeout 120 seconds. No engine/weights were launched.
- Cached model configs allow 1,048,576 position indexes; this is not evidence
  of accuracy at 400k. Estimated occupancy is 84.6% of the current pool, with
  72,768 slots spare; actual pool/input limits/headroom must be reprofiled.
- Restart the full model on all GPUs, with roughly 9.5-minute prior startup
  time and loss of GPU/RAM prefixes. Do not launch a second full-model engine.

After user restart: verify context/settings and each worker's capacity,
then small rank/tools/reasoning, one **398,000-token** chat retrieval, then
four independent 398k requests and warmed forced-output rounds. Allow at
least 1024 output tokens within 400k; never truncate. Follow with same-worker
398k host restoration and bounded queue/load checks. Existing profile-helper
guards must pass on the **new measured** pool; a CLI limit alone is not success.

Run these **only after** the new service is ready and its pools pass inspection;
do not run benchmarks in parallel or assume the helper starts an engine:

```bash
cd /home/advtech/glm53-inference
python3 -B tests/dp4_smoke_test.py
python3 -B tests/dp4_profile_test.py --mode long --concurrency 1 --prompt-tokens 398000
python3 -B tests/dp4_profile_test.py --mode long --concurrency 4 --prompt-tokens 398000
python3 -B tests/dp4_profile_test.py --mode host --rank 0 --prompt-tokens 398000
```

These use 1024 max output tokens for natural retrieval and 512 forced tokens
for timed rounds, within 400k including output. Broader coding quality and
queue overload behavior are separate gates; no 400k test has run yet.

## Queued requests: installed-source review

`--max-queued-requests 8` is passed unchanged to each scheduler: **eight per
worker, up to 32 aggregate**, not eight globally. One generation runs per
worker; others wait FCFS in that worker's queue. New queued prompts do not
become additional active GPU contexts or decode in RAM. A previously cached
prefix may be restored from HiCache when its request is admitted.

`SGLANG_REQ_WAITING_TIMEOUT=120`: a request still waiting past 120 seconds
is aborted. Full worker queues reject incoming requests. Both use 503 errors
for non-streaming calls; an already-open stream can carry an abort/error.
Client/proxy deadlines can end the wait earlier. Current routing is round-robin,
not session-affine or work-stealing: a request can wait behind one worker
while another becomes idle. There is no global FIFO or per-account fairness.
These details are source-reviewed, not overload-runtime qualification.
Leave settings unchanged unless separately approved; long outputs/400k cold
prefill could make the 120-second queue limit important.

## Last verified live state: DP4 ready

The user installed `dp4-mtp-hicache128.service`. Verified ready with TP8/DP4,
EAGLE 1/1/2, FP8 KV, HiCache 128 per TP rank, static memory 0.90 and context
262,144. Four independent pools: **472,768 GPU slots per worker**,
**1,891,072 total**, one running request per worker, prefill chunk 2,048.
HiCache has 2,469,952 host slots per worker (packed target+MTP layers), still
about 1.23 TB node RAM. Engine startup headroom is 9.80-9.85 GB.

Container start 15:50:15 UTC; ready log 15:59:43 UTC (~9.5 min). Startup had
soft-watchdog/failed py-spy diagnostics and the known JIT fallback warning,
but no post-readiness runtime issue was found. Four rank-pinned 41-token
probes passed, metrics showed four active/no queued/no retracted requests,
and OpenAI tools/separated reasoning passed. Evidence:
`glm53-inference/evidence/dp4-startup-validation-summary.json`.

The new `glm53-inference/tests/dp4_profile_test.py` now provides DP-aware
short/long/concurrency and same-worker host-restoration tests. The old
`concurrency_test.py`/`hicache_test.py` remain TP-only. Runtime results from
17:45-17:52 UTC, with no service changes:

- Three-round historical short tests: c1 **93.52 tok/s**; c4 **336.61
  aggregate / 86.08 per request**. Slower than historical TP-only FP8+MTP
  (165 / 450), not an isolated HiCache A/B; this raw test is speed-only.
- Four independent chat-formatted **100k** and **260k** prompts passed
  simultaneous cold retrieval with zero prefix-cache hits. Cold batch wall
  times were 19.52 / 54.96 s. Three warmed 512-output rounds ran at
  **360.63 / 352.00 aggregate**, **91.92 / 89.60 per-request tok/s**.
  Metrics showed four active, no queued/retracted requests, exact prompt/output
  lengths and device hits of 99,968 / 259,968 each. MTP acceptance ~93% / 92%.
- Fresh 260k target plus two pressure conversations pinned to **DP0** proved
  eviction and restoration: **259,968 host hits, zero device hits**, correct
  recall. Response times: **53.80 s cold**, **1.49 s GPU-warm**, **3.29 s
  restored**. Server prefill/restore: **51.24 / 0.086 / 1.29 s**. Natural
  output lengths differed; this is not streaming TTFT.
- Post-load tools/separated reasoning passed; no runtime errors/watchdogs,
  OOM or retractions. RAM available ~857 GB decimal; physical GPU free memory
  minimum 5480 MiB (~5.35 GiB), below startup but still no OOM.
- Comparison and linked reports:
  `glm53-inference/evidence/dp4-runtime-comparison-20261013T175226Z.json`.

**400k is not enabled or validated**, and four 260k prompts plus output are
not proof of exact-full-context occupancy, six/eight-user capacity, sustained
production load or broad coding quality. Host restoration was tested on DP0,
not DP1-3. Keep current profile/HiCache while measuring real coding task time;
an isolated on/off test needs an approved restart, not a historical comparison.

Remaining priorities: install and qualify the approved 400k candidate, real
task/load measurement, production session affinity and DP1-3 host restore.
Six/eight active requests would require a separately approved admission
profile; the current approved target is four, one per worker.

## Prior validated TP-only baseline (rollback)

- `/etc/systemd/system/sglang-glm53.service`, container `sglang-glm53`.
- `zai-org/GLM-5.3`, alias `pratt-high`, port 8012.
- Pinned SGLang 0.5.18 image; digest is in the candidate and context document.
- TP8, DP1, MTP **off**, FP8 KV, HiCache 128 GB per TP rank.
- Six running / eight queued, context 262,144, static fraction 0.90.
- GPU token pool 618,752. Host tier 2,501,568 token slots, about 1.23 TB RAM.
- HiCache restored 259,968 tokens after GPU eviction with correct recall:
  32.18 s cold versus 3.38 s restored; six warmed 90k sessions passed at
  about 410 aggregate / 68.5 per-request tok/s, with no queueing/retractions.

This is the historical MTP-off baseline, not the live DP4 profile. Always
verify `/server_info` before resuming; do not assume these notes are current.

## Historical installed 262k DP4 candidate

`/home/advtech/sglang-glm53-fp8-tests/dp4-mtp-hicache128.service`

Changes from the current HiCache service:

```text
--tp 8                     # unchanged global FFN/model TP
--dp-size 4
--enable-dp-attention
--moe-a2a-backend none      # no DeepEP initially; EP remains its default 1
--speculative-algorithm EAGLE
--speculative-num-steps 1
--speculative-eagle-topk 1
--speculative-num-draft-tokens 2
--max-running-requests 4   # initial four-worker test, expected one per worker
```

HiCache 128, FP8 KV/backends, memory 0.90, context 262,144, chunk size 8192,
graph max BS 32, queue launch setting 8, timeouts, API identity and memlock
remain unchanged. This is a combined DP4+MTP profile, not an isolated A/B of
DP alone. EP8/DeepEP and DP8-specific tuned flags are not being copied.

Static systemd verification and this installed SGLang's actual argument
constructor/parser preflight passed, without launching an engine or loading
weights. Evidence: `sglang-glm53-fp8-tests/dp4-mtp-hicache128-preflight.json`.
The constructor **divides the launch prefill chunk 8192 by DP4 to 2048 per
worker**; record the resolved value after startup. An early DSA warning prints
`attn_tp_size=8` using the global TP field, but the DP grouping helper and model
attention code derive the actual attention group from the parallel state;
verify its expected size two at runtime rather than trusting that warning.

TP8+DP4 attention resolves to four DP workers, each with attention TP2. MLA
cache is replicated inside each two-GPU group, not combined into a doubled
pool. **Do not promise 400k per worker.** The linked experiment only measured
DP8; it used different settings (including 0.85 memory and SGLang 0.5.20).
`SGLANG_DP_USE_GATHERV` in this installed code applies only to attention TP1 /
TP=DP, so do not enable it for DP4. Do not copy the other host's NUMA remap.

## Installation and rollback

This account has no passwordless sudo. The user elected to run privileged
commands personally. Do not use a Docker root helper to bypass that decision.
Check request gauges/drain and explain downtime before restarting.

```bash
sudo install -o root -g root -m 0644 /home/advtech/glm53-inference/configs/systemd/candidates/dp4-mtp-hicache128-400k.service /etc/systemd/system/sglang-glm53.service
sudo systemctl daemon-reload
sudo systemctl restart sglang-glm53.service
```

Preferred narrow rollback from 400k retains the scoped-tested 262k DP4
profile (same image, MTP, HiCache and admission):

```bash
sudo install -o root -g root -m 0644 /home/advtech/glm53-inference/configs/systemd/candidates/pre-400k-dp4-mtp-hicache128.service /etc/systemd/system/sglang-glm53.service
sudo systemctl daemon-reload
sudo systemctl restart sglang-glm53.service
```

An exact pre-DP4 unit/server-info snapshot is saved separately. Do not pull
or retag images or load a second full-model engine on the same GPUs.
The older TP-only fallback remains
`sglang-glm53-fp8-tests/fp8-no-mtp-6-hicache128.service`, but changes DP and
MTP too and is not the narrow rollback for this context-only experiment.

## Ordered validation

1. Wait for readiness; inspect errors and verify TP8/DP4, MTP 1/1/2, HiCache
   128, FP8, context 262,144 and memory 0.90. Verify the effective admission
   limit is one per DP worker; do not infer it solely from the launch flag.
2. Record **each DP worker's** GPU token pool, input limit, headroom, host pool
   and rank. Do not multiply per-worker capacity by eight: there are four
   independent pools, replicated within pairs. Verify RAM and swap remain safe.
3. Send small native `/generate` probes pinned with `routed_dp_rank` 0-3 and
   require response metadata `dp_rank` to match. Check OpenAI tools/reasoning.
4. Warm four distinct correctly chat-formatted prefixes, one per worker, then
   benchmark four simultaneously. Start with lengths comfortably below the
   smallest pool and context limit. Measure MTP acceptance and client timings.
5. Verify one active request on each worker from metadata/metrics and no
   retractions, hidden queueing or truncation. The existing TP-only metric
   sampler takes a max across series, **not** the sum of four DP workers;
   adapt it with per-rank deduplication before claiming total concurrency.
   If only one worker's gauges are exported, prove overlapping generation
   intervals across the four returned `dp_rank` values or deliberately enable
   per-worker exports; never sum duplicate TP-rank gauges.
6. Validate near-262k retrieval per worker only if the profiled limits allow
   adequate output/headroom. Pin the target and pressure requests to the same
   worker for a host-restore test. The existing `hicache_test.py` is not yet
   DP-aware; round-robin requests would invalidate its eviction/reuse proof.
7. Evaluate larger context **after** profiling. The user discussed four users
   near 400k, but 400k capacity is unproven. Current CLI caps at 262,144; a
   larger-context candidate needs an explicit limit change, a restart and
   fresh long-prefill/correctness/load tests. Do not auto-truncate or claim
   400k from merely changing the advertised limit.
8. Record results and decision in the progress/context/results files. Broader
   real coding-task time, compaction frequency and quality still matter more
   than a synthetic tok/s result. Eight shorter users could later share four
   workers if the memory budget and effective admission caps permit.

Production session-affinity routing is not installed. Direct rank pinning is
for validation; keeping each real session on its warm worker needs a separate
routing/client design after the profile is proven. Do not claim adding DP4
alone reproduces the Ramjet experiment's session-cache performance.

## References and evidence

- `/home/advtech/sglang-glm53-fp8-tests/results.md`
- `/home/advtech/sglang-glm53-fp8-tests/hicache128-validation-summary.json`
- https://github.com/helixml/ramjet/blob/main/EXPERIMENTS.md (2026-09-29 section)
- Local `srt/layers/dp_attention.py` resolves attention TP as TP/DP.

Startup, tools, four-near-limit concurrency and same-worker DP0 host restore
passed in scoped synthetic tests. DP1-3 restoration, sustained real workload,
six/eight-user admission and larger context remain pending. The original
ordered gates above are retained as the plan, not as evidence of completion.

## Reproduce current-profile tests (idle server only)

```bash
cd /home/advtech/glm53-inference
python3 -B -m unittest discover -s tests -p 'test_dp4_profile.py' -v
python3 -B tests/dp4_profile_test.py --mode short --concurrency 1 --output-tokens 256
python3 -B tests/dp4_profile_test.py --mode short --concurrency 4 --output-tokens 256
python3 -B tests/dp4_profile_test.py --mode long --concurrency 4 --prompt-tokens 100000
python3 -B tests/dp4_profile_test.py --mode long --concurrency 4 --prompt-tokens 260000
python3 -B tests/dp4_profile_test.py --mode host --rank 0 --prompt-tokens 260000
python3 -B tests/dp4_smoke_test.py
```

Run one test at a time; do not run concurrent benchmarks on the same engine.
The helper checks idle gauges, exact context and private-pool headroom, pins
ranks, deduplicates TP metric replicas and writes timestamped `evidence/`
reports, retaining failures. It never flushes caches or changes the service.
