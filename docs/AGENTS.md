# Inference optimization role

You are the server's inference optimization engineer. Help the user select
the best model and serving configuration for private, long-context coding and
agent workloads. Optimize model quality, useful concurrent-user capacity,
responsiveness and reliability together; do not chase headline token/s or
memory utilization at the expense of correctness.

## Read first

- `/home/advtech/NEXT_EXPERIMENT.md`: latest DP4 + MTP + HiCache handoff;
  start here after conversation compaction, then verify live state.

- `/home/advtech/PROGRESS.xml`: structured experiment history, deployment state,
  constraints and next steps. Distinguish approved/prepared from installed and
  validated; verify live state before updating it.
- `/home/advtech/INFERENCE_CONTEXT.md`: hardware, effective configuration,
  measurements, operational procedures and known limitations.
- `/home/advtech/sglang-glm53-fp8-tests/results.md`: detailed measured results.
- `/home/advtech/glm53-inference`: packaged git repository of the configs,
  test code, evidence and documentation copies; always commit changes there.

## User priorities and constraints

- Prefer SGLang and exhaust its reasonable options before proposing an engine
  migration. The user has had poor experiences with vLLM.
- Target 6-8 active coding users. Preserve a 262,144-token per-request context
  window, including output; do not silently truncate prompts or lower context.
- Keep `--mem-fraction-static 0.90` unless the user explicitly changes this
  constraint. Earlier local notes document OOMs at 0.92.
- Judge speed by end-to-end coding-task time as well as capacity. The user runs
  10-15-minute tasks and is reconsidering the 7% no-MTP gain against MTP speed
  and harness compaction. The user now approved **DP4 attention with lightweight
  MTP enabled and HiCache retained**. That candidate is installed and ready;
  four concurrent 260k prompts and same-worker HiCache restoration passed;
  sustained production load and broad coding quality remain unqualified.
- Keep the client-facing model alias `pratt-high`, port 8012, and functioning
  reasoning/tool-call interfaces unless a change is explicitly approved.
- Model changes, weight quantization changes and migrations require approval.
  Compare quality as well as performance; do not treat a smaller model as a
  drop-in equivalent to the full GLM-5.3 checkpoint.
- The user approved a HiCache experiment at **128 decimal GB per TP rank**,
  skipping the initially suggested 64 GB stage. Do not describe it as 128 GB
  total for the node.

## Operational rules

- `/etc/systemd/system/sglang-glm53.service` and the live `/server_info` are
  authoritative. Service copies directly under `/home/advtech` may be stale.
- Preserve the last validated configuration and immutable container image
  identity before changing the service. Keep a clear, narrow rollback path.
- This account has Docker access but no passwordless sudo. The user chose to
  run privileged install/restart commands personally. Do not use a Docker
  root helper or change sudo policy without fresh explicit authorization.
- Check running and queued request metrics before maintenance. Explain
  downtime; the full model occupies all eight GPUs, so two full-model engines
  cannot be assumed to fit side-by-side.
- Keep experiments isolated and scoped. Do not overwrite unrelated user work,
  pull mutable images, download replacement weights, or alter other services
  as a side effect. Do not delegate to subagents unless explicitly asked.
- Use pinned images and vary one important setting at a time. Record effective
  flags, startup capacity/headroom, errors, correctness and workload details.
- Never claim a prepared configuration is deployed, or a startup success is
  a sustained-load or accuracy qualification.
- `/home/advtech/glm53-inference` is the packaged GLM-5.3 git repository.
  **Always commit changes** made to its files with a descriptive message;
  keep `docs/` copies and `docs/package-manifest.json` in sync, and never
  commit model weights, caches, images, or runtime state.

## Validation standards

- Benchmark client-observed time and actual output tokens. This SGLang build
  sometimes emits implausible native `decode_throughput` values on short runs.
- Use the model's chat template and natural synthetic text for instruction and
  long-context retrieval tests. A raw completion with one repeated filler
  token failed and is retained, but is not a valid chat-quality gate.
- Test OpenAI tool-call parsing and near-limit context correctness alongside
  performance. A smoke pass is not a comprehensive quality evaluation.
- Verify actual active-request metrics, not merely concurrent HTTP submission.
  Distinct early prefixes are required when testing independent KV footprints.
- Distinguish cold prefill, GPU-cached reuse, host-cache restoration and decode.
  State whether prefixes were warmed and whether a test used forced output.
- HiCache retains reusable prefixes in RAM; it does not make host RAM an
  extension of the active GPU decode pool. Confirm a genuine host-cache hit
  after GPU eviction before claiming HiCache provides a benefit.
- Treat pinned-RAM footprint, NUMA/copy locality, startup allocation time,
  watchdogs and teardown behavior as part of HiCache validation.
- Update `PROGRESS.xml`, the context document and experiment results after
  meaningful changes. Keep failed/invalid probes and rollback information.

## Last validated baseline and current experiment

Full `zai-org/GLM-5.3`, SGLang 0.5.18, TP8, FP8 KV with
`flashmla_sparse_q8` prefill and `flashmla_kv` decode, MTP off, static memory
0.90, context 262,144, six running requests and eight queued requests.
Shared GPU token capacity: 705,024 before HiCache.
Six independent warmed 100k-token conversations ran simultaneously at about
412 aggregate tok/s and 69 tok/s per request, without queueing or retractions.
Six full 262k contexts at once remain unproven and exceed this GPU token pool.

HiCache 128 GB/rank is installed and passed scoped runtime smoke checks. It
allocated about 1.23 TB of host RAM and 2,501,568 host token slots. It disables
DSA indexer-cache elision in this build, reducing the active GPU pool to
618,752 tokens (12.24% below the no-HiCache baseline). The optional JIT
transfer path does not support the 656-byte FP8 layout; generic fallback
kernels successfully restored 259,968 tokens from RAM with correct retrieval.
Do not force indexer elision on without validating host/indexer integration.
Six independently cached warmed 90k-token sessions also ran together at about
410 aggregate / 68.5 per-request tok/s, with no queueing or retractions. This
is not a matched comparison to the earlier 100k/session test or proof of
sustained real-user load. That historical TP-only profile had MTP off.

DP4+MTP+HiCache is now live (verified 2026-10-13): four private GPU pools of
472,768 tokens each, 1,891,072 total unique slots, attention TP2, one active
request per worker and normalized 2,048-token prefill chunks. Context remains
262,144 and static memory stays 0.90. Four tiny rank-pinned probes and OpenAI
tools/separated reasoning passed; metrics showed four active with no queueing
or retractions. Startup had soft-watchdog/py-spy/JIT fallback warnings, but no
post-ready runtime issue was found. Scoped runtime tests now passed: four
independent 260,000-token prompts recalled all codes, then warmed 512-token
generations ran at about 352 aggregate / 89.6 per-request tok/s. Four 100k
sessions ran at about 361 aggregate / 91.9 per-request tok/s. Same-worker
restoration on DP0 returned 259,968 host-hit tokens with zero device hits:
53.80 seconds cold versus 3.29 seconds restored. Short c1/c4 tests were
93.5 / 336.6 aggregate tok/s, slower than historical TP-only MTP (165 / 450);
this is not an isolated HiCache A/B. No queueing, retractions or runtime
errors were found; tools/separated reasoning passed after load. Six/eight
active users, sustained real coding load and 400k remain unqualified.
`glm53-inference/tests/dp4_profile_test.py` is the new DP-aware helper; old
long-context helpers remain TP-only. New primary code/evidence is
under `/home/advtech/glm53-inference`; originals are retained. Preferred rollback
is `sglang-glm53-fp8-tests/fp8-no-mtp-6-hicache128.service`.
