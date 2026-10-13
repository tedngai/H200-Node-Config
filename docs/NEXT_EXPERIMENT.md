# Resume after conversation compaction: DP4 + MTP + HiCache

Updated 2026-10-13. DP4 is now installed and passed startup/small smoke;
long-context and sustained-load qualification remain pending.
Read `AGENTS.md`, `PROGRESS.xml`, and `INFERENCE_CONTEXT.md` first.

## Latest user instruction

"Update context with the current plan, keep MTP and hicache, and test DP4."
The user will compact the conversation. This authorizes **enabling lightweight
MTP in the DP4 candidate** and retaining HiCache, not removing either to gain
capacity. Prefer SGLang; keep full GLM-5.3 and static memory fraction 0.90.

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

The new `glm53-inference/tests/dp4_smoke_test.py` is a small smoke helper only.
The old long-context helpers still require DP-aware routing/metrics changes.
**400k is not enabled or validated**, even though the allocated per-worker
pool is promising. Next: independent near-262k rank-pinned retrieval and
concurrency, MTP acceptance/client latency, and a same-rank host restore.

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

## Candidate

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
sudo install -o root -g root -m 0644 /home/advtech/sglang-glm53-fp8-tests/dp4-mtp-hicache128.service /etc/systemd/system/sglang-glm53.service
sudo systemctl daemon-reload
sudo systemctl restart sglang-glm53.service
```

Rollback retains the already validated HiCache profile:

```bash
sudo install -o root -g root -m 0644 /home/advtech/sglang-glm53-fp8-tests/fp8-no-mtp-6-hicache128.service /etc/systemd/system/sglang-glm53.service
sudo systemctl daemon-reload
sudo systemctl restart sglang-glm53.service
```

An exact pre-DP4 unit/server-info snapshot is saved separately. Do not pull
or retag images or load a second full-model engine on the same GPUs.

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

Startup and small rank/tool gates passed; full long-context/load and DP4+MTP
host-restoration gates remain pending. Static checks and small smoke are not
sustained-load or broad quality qualification.
