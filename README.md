# H200 Inference Optimization for a Small Development Team

An evidence-driven SGLang deployment and measurement project for serving the
full GLM-5.3 model on a multi-GPU NVIDIA H200 host. It explores how to balance
long-context coding and agent workloads, useful concurrency, response time,
and reliability for a small team—not just maximize headline tokens per second.

This repository captures one tested environment. It is a reference for
experimentation, not a turnkey distribution or a claim that every workload or
deployment path has been production-qualified.

## Reference profile

The latest recorded profile was validated on 2026-10-13. Its configuration and
test reports are included in this repository.

| Component | Recorded setting |
| --- | --- |
| Hardware | Eight NVIDIA H200 GPUs |
| Model and engine | Full `zai-org/GLM-5.3`; SGLang 0.5.18 |
| Parallelism | Tensor parallelism across eight GPUs with four data-parallel attention workers |
| KV cache and decoding | FP8 KV cache and lightweight EAGLE speculative decoding |
| GPU memory budget | `--mem-fraction-static 0.90` |
| Context limit | 400,000 tokens, including output |
| Admission | Four active generation requests, one per worker |
| Prefix cache | HiCache configured for 128 decimal GB per TP rank |

HiCache uses host memory for reusable prefixes; it is not additional GPU memory
for active decoding. In this setup its host-memory footprint was about 1.23 TB
across the eight TP ranks, including indexer buffers. See the linked evidence
for effective settings and environment-specific details.

## Selected measured results

These are scoped synthetic tests on the reference system, not production
service-level objectives or broad coding-quality evaluations.

| Test | Result |
| --- | --- |
| Four independent near-limit prompts | Four separate 398k-token prompts passed retrieval checks concurrently, with no queued or retracted requests. This is near-limit validation, not an exact-full-400k test. |
| Warmed concurrent generation | 325.3 aggregate tokens/s, or 85.5 tokens/s per request, across four requests generating 512 tokens each. Prefixes were warmed before timing. |
| Host-cache restoration | After GPU eviction, the DP0 worker restored 397,952 tokens from host cache with zero device-cache hits. Server-side restore took 1.90 s versus 88.18 s for cold prefill; client-observed times were 4.13 s and 90.84 s respectively. Output lengths differed, so the end-to-end comparison is illustrative rather than a matched latency benchmark. |
| Interface checks | OpenAI-compatible tool calling and separated reasoning checks passed before and after the load test. |

The reports record test inputs, actual output lengths, worker metrics, cache
hits, and limitations. In particular, the near-limit results do not establish
sustained real-user performance, comprehensive model quality, or successful
restoration on every worker.

## Optimization principles

- **Optimize for useful work.** Evaluate coding-task completion time,
  correctness, responsiveness, and reliable concurrency alongside throughput.
- **Measure real admission.** DP workers have private GPU KV pools. Aggregate
  capacity is not one shared pool; confirm active requests and per-worker
  queueing with server metrics.
- **Treat cache options as trade-offs.** FP8 KV, speculative decoding, and
  host-backed prefix caching affect capacity, latency, memory use, and runtime
  behavior differently. Change one major setting at a time and validate it on
  the target model and software build.
- **Prove cache reuse.** A host-cache benefit requires a real host hit after GPU
  eviction. HiCache can speed up repeated prefixes, but does not turn host RAM
  into active GPU decode capacity.
- **Benchmark client-observed behavior.** Record wall time and actual output
  tokens; separate cold prefill, GPU-cache hits, host restores, and decode.
  Short forced-output benchmarks are useful for speed comparisons but do not
  measure coding quality.
- **Test correctness at realistic inputs.** Use chat-formatted prompts and
  natural test data; include long-context retrieval, tool calls, reasoning
  format, and checks that the full prompt was processed without truncation.

## Repository contents

- `configs/systemd/` — reference service unit and historical or experimental
  alternatives.
- `tests/` — short benchmarks, DP-aware concurrency and routing checks,
  long-context retrieval tests, and HiCache restoration tests.
- `evidence/` — captured configuration snapshots and experiment reports.
- `docs/results.md` and `docs/test-history.md` — detailed results and test
  history.
- `docs/INFERENCE_CONTEXT.md` and `docs/NEXT_EXPERIMENT.md` — technical
  background and experiment notes.

## Validation

The host-side test scripts use Python's standard library. Start with syntax and
unit checks:

```bash
python3 -m py_compile tests/*.py
python3 -B -m unittest discover -s tests -p 'test_dp4_profile.py' -v
```

Live checks require a compatible, ready SGLang DP4 server. The smoke test also
expects Docker access to a container named `sglang-glm53` with the model
tokenizer available locally:

```bash
python3 -B tests/dp4_smoke_test.py --base-url http://127.0.0.1:8012
python3 -B tests/dp4_profile_test.py --mode short --concurrency 4 --output-tokens 256
```

Long-context and host-cache tests can create substantial GPU and host-memory
load. Run them only against a server you control, when it is idle, and follow
the admission and setup guidance in `docs/NEXT_EXPERIMENT.md`. The scripts
exercise a live server and may write reports under `evidence/`; they do not
install or modify the service.

## Adapting the configurations safely

The service units are specific to one host and model. Before using them, review
the container image, model and cache paths, GPU topology, memory limits, port,
and served-model name. Do not install a candidate based solely on its filename
or on results from a different profile.

The reference unit uses host networking, listens on all interfaces, and enables
request logging. Do not expose it to an untrusted network without an
appropriately reviewed authentication, TLS, firewall, and logging setup. Model
weights are not included; obtain them under the model provider's access terms.
Captured reports and service files may contain environment-specific metadata
and should be reviewed before being reused or redistributed.
