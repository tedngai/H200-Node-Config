# H200 Inference Optimization for a Small Development Team

An evidence-driven SGLang deployment and measurement project for serving the
full GLM-5.3 model on a multi-GPU NVIDIA H200 host. It explores how to balance
long-context coding and agent workloads, useful concurrency, response time,
and reliability for a small team—not just maximize headline tokens per second.

This repository captures one tested environment. It is a reference for
experimentation, not a turnkey distribution or a claim that every workload or
deployment path has been production-qualified. The status below reflects the
latest evidence in the repository, recorded on 2026-10-13; it is not a live
check of a running server.

## Latest recorded configuration

The latest validated service snapshot is
[`dp4-mtp-hicache128-400k.service`](configs/systemd/candidates/dp4-mtp-hicache128-400k.service).
The earlier 262k profile is retained as
[`pre-400k-dp4-mtp-hicache128.service`](configs/systemd/candidates/pre-400k-dp4-mtp-hicache128.service).

| Setting | Recorded value |
| --- | --- |
| Hardware and software | Eight NVIDIA H200 GPUs; full `zai-org/GLM-5.3`; SGLang 0.5.18 with a pinned container image |
| Parallelism | TP8 with four DP-attention workers; each worker uses a private GPU KV pool |
| KV cache and attention | FP8 E4M3 KV; `flashmla_sparse_q8` prefill and `flashmla_kv` decode |
| Speculative decoding | EAGLE, 1 step / top-k 1 / 2 draft tokens |
| Static GPU memory fraction | `0.90` |
| Context and truncation | 400,000 tokens including output; automatic prompt truncation disabled |
| GPU KV capacity | 472,768 token slots per worker; 1,891,072 unique slots across four private pools, not one shared pool |
| Active-request admission | Four total, effectively one per worker |
| Queueing | Eight waiting requests per worker (up to 32 total), 120-second wait timeout, round-robin worker routing; overload behavior was not load-tested |
| HiCache | 128 decimal GB main KV per TP rank, plus 25.76 GB/rank of DSA indexer buffers; about 1.23 TB decimal host allocation across eight TP ranks |
| Prefill chunk | 2,048 tokens per worker after SGLang normalizes the 8,192-token launch setting |

The service snapshot also records the reasoning and tool-call parsers used in
the checks. See the service file and linked evidence for the exact flags and
effective server-reported values.

## What passed on the latest profile

Evidence: [400k validation summary](evidence/dp4-400k-validation-summary-20261013T204840Z.json)
and [detailed results](docs/results.md).

- **Startup and capacity:** the server became ready with four independent DP
  pools and one active request allowed per worker.
- **Near-limit retrieval:** one and four independent 398,000-token
  chat-formatted synthetic prompts recalled their test codes. The complete
  prompts were processed without truncation; the four-request run showed four
  active workers, no queueing, and no retractions.
- **Warmed generation:** after warming the prefixes, three rounds of four
  concurrent 512-token generations measured **325.33 aggregate tokens/s** and
  **85.46 median tokens/s per request** using client time and actual output.
- **Host-cache restore:** after GPU eviction, DP0 restored **397,952 host-hit
  tokens with zero device-cache hits**. Server prefill/restore time was 1.90 s
  versus 88.18 s cold; client-observed times were 4.13 s versus 90.84 s. The
  equal output lengths make this particular end-to-end pair informative, but it
  is still a single synthetic test, not a latency distribution.
- **Interfaces and runtime:** OpenAI-compatible tool calling and separated
  reasoning passed before and after load. No post-ready OOM, runtime error,
  queueing, or retraction was recorded during these scoped tests.

This is near-limit validation, **not** an exact 400,000-token occupancy test.
The host-restore test was performed on DP0 only. These results do not establish
sustained real-user load, broad coding quality, or end-to-end behavior through
an external client gateway.

## Earlier measured configurations

These TP8 measurements are useful historical comparisons, not substitutes for
the latest DP4 profile. They used 262,144-token context, static memory fraction
0.90, short forced-output benchmarks, and client-observed timing. The benchmark
does not measure coding quality.

| Historical setting | KV token capacity | One-request median | Four-request aggregate median |
| --- | ---: | ---: | ---: |
| BF16 KV + MTP ([unit](configs/systemd/candidates/original.service)) | 384,640 | 190.95 tok/s | Not measured |
| FP8 KV + MTP ([unit](configs/systemd/candidates/fp8-mtp.service)) | 660,032 | 165.35 tok/s | 449.74 tok/s |
| FP8 KV, MTP off ([unit](configs/systemd/candidates/fp8-no-mtp.service)) | 705,024 | 96.89 tok/s | 328.14 tok/s |

In this comparison, FP8 KV increased token capacity by 71.6% versus the BF16
baseline, while the short one-request rate was lower. Turning MTP off raised
the FP8 pool by a further 6.8%, but reduced the short one-request and
four-request aggregate rates by 41.4% and 27.0% relative to FP8+MTP. These are
profile-specific capacity/speed trade-offs, not quality comparisons or
predictions for the DP4/HiCache configuration.

The legacy `original.service` is evidence for the captured baseline, not a
reproducible install target: its description says “FP8” but the unit has no
explicit KV-cache dtype flag; the captured baseline reported BF16/default KV.
It also references the mutable `:latest` image tag. Use the recorded evidence,
not that description or tag, when interpreting the historical row.

Two additional historical six-session tests passed scoped checks:

| Profile | Workload and measured result | Important limit |
| --- | --- | --- |
| [`fp8-no-mtp-6.service`](configs/systemd/candidates/fp8-no-mtp-6.service) | Six independent warmed 100k-token sessions; 412.40 aggregate / 68.88 median per-request tok/s | Does not validate six simultaneous 262k contexts or the current DP4 profile |
| [`fp8-no-mtp-6-hicache128.service`](configs/systemd/candidates/fp8-no-mtp-6-hicache128.service) | Six warmed 90k-token sessions; 409.98 aggregate / 68.47 median per-request tok/s. A separate 260k eviction test restored 259,968 host-hit tokens. | Not matched to the 100k test; some warmup prefixes were reused. HiCache reduced that profile's active GPU pool by 12.24%, from 705,024 to 618,752 slots. |

Historical evidence and profile details are in [`docs/results.md`](docs/results.md)
and [`docs/test-history.md`](docs/test-history.md). Candidate names indicate
what a file configures; they do not mean that it is currently installed.

## Failures, regressions, and caveats

The repository retains unsuccessful probes and negative findings instead of
presenting only passing results:

- **Invalid retrieval probe:** a 260k raw-completion request returned repetitive
  filler and failed to retrieve its test codes. It did not use the model's chat
  template and used repeated filler tokens, so it is retained as a failed/invalid
  probe—not treated as a valid quality evaluation. A corrected chat-formatted
  natural-text test later passed. See
  [`fp8-mtp-smoke-260000.json`](evidence/fp8-mtp-smoke-260000.json) and
  [`docs/results.md`](docs/results.md#initial-smoke-checks).
- **Harness failure before inference:** the first DP4 smoke attempt could not
  serialize a tokenizer `BatchEncoding` object and sent no inference request.
  The harness was corrected; subsequent rank/tool smoke tests passed. See the
  [retained probe record](evidence/dp4-startup-smoke-20261013T160713Z.json).
- **HiCache accelerated-transfer limitation:** the optional JIT transfer path
  does not support this 656-byte FP8 layout. SGLang logged a fallback to generic
  transfer kernels; targeted host restoration passed using that fallback.
- **Startup diagnostics:** the latest 400k startup logged 56 soft-watchdog
  warnings, 112 failed `py-spy` diagnostics, and eight known JIT fallback
  warnings. The service nevertheless became ready, and no post-ready runtime
  error or OOM was found during the recorded tests. Startup warnings should not
  be confused with a clean startup log.
- **Short benchmark trade-off:** the historical matched-input microbenchmark
  measured the DP4+MTP+HiCache profile at 93.52 tok/s for one request and 336.61
  aggregate tok/s for four, below historical TP8+FP8+MTP results of 165.35 and
  449.74 tok/s. This is not an isolated comparison: parallelism, HiCache, and
  effective prefill chunking differ. It shows that the current profile did not
  win the short decode test; its evidence is primarily about four-worker and
  long-context behavior.
- **External routing unverified:** a neutral request through an external
  gateway did not change counters on the tested engine. The gateway's upstream
  target and data-parallel header forwarding therefore remain unqualified; the
  probe does not establish where that request was served. See the
  [routing probe](evidence/opencode-gateway-dp-header-probe-20261013T204440Z.json).
- **Unreliable native speed metric:** this SGLang build sometimes reports
  implausible `decode_throughput` values on short tests. Reported speed
  comparisons use client elapsed time and actual generated-token counts.

## Not yet validated

- Exact occupancy of all 400,000 context tokens including output; the largest
  independent prompts tested were 398,000 tokens.
- Host-cache eviction and restoration on DP1, DP2, and DP3 (the 400k test
  exercised DP0 only).
- Sustained production traffic, broad coding-task quality, or end-to-end
  completion time for real development work.
- Six or more active requests on the current DP4 profile.
- Queue saturation, retries, per-user fairness, session affinity, or gateway
  routing to the measured engine.

## Repository contents

- `configs/systemd/` — reference service unit and historical or experimental
  alternatives.
- `tests/` — short benchmarks, DP-aware concurrency and routing checks,
  long-context retrieval tests, and HiCache restoration tests.
- `evidence/` — captured configuration snapshots, passing reports, and failed
  or invalid probe records.
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
