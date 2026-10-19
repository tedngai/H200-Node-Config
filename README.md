# DeepSeek-V4.1-Flash-1017 on 8 × H200

Run `deepseek-ai/DeepSeek-V4.1-Flash-1017` on one Linux machine with **8 × H200
141 GB GPUs, about 2 TB of host RAM, and local NVMe**. SGLang serves the model
with TP8/EP8, DSpark, native Engram host tables and HiCache. The reference
profile uses a **409,600-token context** and **16 running requests**.

This repository contains the systemd launcher, test harness and measured
inference results. It is an experimental recipe: successful loading, measured
speed, populated context capacity and feature correctness are separate
milestones. It is not a turnkey distribution or a production SLA.

**Last measured:** 2026-10-19 on the reference 8×H200 host.

## 1. Current status

| Item | Verified result or current limit |
| --- | --- |
| Model precision | Published mixed precision: FP4 routed experts, FP8/BF16 components, FP8 Engram. No EXL3 conversion or pruning. |
| Local inference | Running on **8012 / `pratt-high`** and completing chat, reasoning and tool requests. |
| Context | **409,600 tokens**; a 409,007-token total near-limit request recalled correctly. Exact full 409,600-token occupancy is not tested. |
| GPU KV capacity | **45,075,200 full tokens + 51,712 SWA tokens** in the phase 2b profile. |
| Host KV tier | **90,150,400 HiCache token slots**, ratio 2. A 256-token host hit was observed; a 400k host restoration was not run. |
| Running requests | **16** configured. A 16-active warmed 400k replay reached peak 16 with 0 queue and 0 retractions. |
| Cold 400k concurrency | 16 × 400k cold requests completed in **1068.9 s**, but prefill was serialized. **16 simultaneous cold 400k prefills are not proven.** |
| Decode | c1 **181.1 tok/s**, c4 **414.0 tok/s**, c16 **643.4 tok/s** in the scripted short test. |
| Mixed load | Four warmed 400k decodes plus four cold 400k prefills: **292.4 aggregate / 68.8 median tok/s**, ratio **0.649**. |
| Cold TTFT | 400k c1 **74.7 s**; four-cold max **268.3 s**, batch wall **268.5 s**. |
| Text, reasoning, tools | Chat, separated reasoning, tool call and second tool turn passed. |
| Vision / video / audio | Not tested on this host. Do not treat these as supported. |
| Reliability | No post-ready OOM, runtime error, retraction or watchdog event in the scoped tests. No sustained production soak. |

## 2. Requirements

- Linux x86-64 with a CUDA 13-compatible NVIDIA driver.
- **8 × NVIDIA H200 141 GB** GPUs (143,771 MiB reported by `nvidia-smi`).
- About **2 TB host RAM**. The Engram host tables alone measure **386,736 MiB**.
- Docker and the NVIDIA Container Toolkit.
- About **510 GB** for the checkpoint plus container, cache and kernel headroom.
- Local NVMe for the Hugging Face cache. The reference unit mounts
  `/raid/advtech/cache/huggingface`; adjust that path for your host.
- Access to the `deepseek-ai/DeepSeek-V4.1-Flash-1017` checkpoint. Model weights
  are not included in this repository and remain under the model provider's
  terms.

Pinned engine image:

```text
lmsysorg/sglang@sha256:4a5d132a06a77c8331e15845f2e925adc788b00105097ad55409afa3f4fa4860
```

The image reports SGLang `0.0.0.dev1+gda64c5cbb` and CUDA 13.0.3.

## 3. Start the service

The reference unit is:

```text
configs/systemd/candidates/sglang-dsv41-8012-pratt-high-mr16-nodelay-ctx409600-overlap2b.service
```

Review the unit first. It is specific to this host: eight GPUs, the model cache
mount, port 8012 and the served name `pratt-high`. Change those values for your
environment.

```bash
git clone https://github.com/tedngai/H200-Node-Config.git
cd H200-Node-Config

# Stop any other full-model engine first. Both need all eight GPUs.
sudo systemctl stop sglang-glm53.service sglang-glm53-flash.service

sudo install -o root -g root -m 0644 \
  configs/systemd/candidates/sglang-dsv41-8012-pratt-high-mr16-nodelay-ctx409600-overlap2b.service \
  /etc/systemd/system/sglang-dsv41.service
sudo systemctl daemon-reload
sudo systemctl start sglang-dsv41.service

# Startup takes about 9–10 minutes with a warm checkpoint cache.
docker logs -f sglang-dsv41
```

The unit above is the exact measured test unit (`Restart=no`). For a boot-enabled
service with bounded crash restart, use the `...-production.service` candidate in
the same directory; it preserves every phase 2b flag and adds
`Restart=on-failure` plus a model-cache mount dependency.

Verify readiness and the effective configuration:

```bash
curl -s http://127.0.0.1:8012/v1/models | python3 -m json.tool
curl -s http://127.0.0.1:8012/server_info | python3 -c \
  'import json,sys; d=json.load(sys.stdin); print({k:d.get(k) for k in ("model_path","max_running_requests","context_length","prefill_decode_interval","chunked_prefill_size","enable_hierarchical_cache")})'
```

A minimal request:

```bash
curl http://127.0.0.1:8012/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"pratt-high","messages":[{"role":"user","content":"What is 19 + 23?"}],"chat_template_kwargs":{"thinking":false}}'
```

The reference unit has **no engine-level API key** and listens on all
interfaces. Put it behind an authenticated, TLS-terminated gateway or firewall.
Do not expose it directly to an untrusted network.

## 4. Configuration

| Setting | Reference profile | Why |
| --- | --- | --- |
| Model | `deepseek-ai/DeepSeek-V4.1-Flash-1017` | Checkpoint revision `1bae0e1088993b9b82de08f59de5be5a34e42170` |
| Parallelism | `--tp 8 --ep-size 8` | One eight-GPU tensor/expert group |
| Context | `--context-length 409600` | Includes output; no automatic truncation |
| Running requests | `--max-running-requests 16` | Effective admission cap |
| Admission delay | `--min-free-slots-delay 0` | Keeps all 16 slots usable; phase 2b |
| Prefill chunk | `--chunked-prefill-size 4096` | Finer prefill/decode interleaving |
| Scheduler overlap | `--prefill-decode-interval 16` | Protects warmed decode during cold prefill |
| Static memory | `--mem-fraction-static 0.8` | Leaves about 13 GiB/GPU free after load |
| CUDA graph batch | `--cuda-graph-max-bs-decode 64` | Decode graph capture limit; admission remains 16 |
| DSpark | `--speculative-algorithm DSPARK --speculative-dspark-block-size 5` | Checkpoint-native draft path |
| Attention / MoE | `--attention-backend dsv4 --moe-runner-backend flashinfer_mxfp4` | Cookbook H200 Flash-1017 cell |
| SWA replay | `--enable-decoder-swa-bounded-replay` | Required for this checkpoint's SWA path |
| HiCache | `--enable-hierarchical-cache --hicache-io-backend direct --hicache-mem-layout page_first_direct --hicache-ratio 2 --hicache-write-policy write_back` | Host KV prefix tier |
| Engram host table | `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1`; `SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=private` | Frees the two FP8 tables from HBM |
| Parsers | `--reasoning-parser auto --tool-call-parser auto` | OpenAI reasoning/tool compatibility |
| Watchdog | `--watchdog-timeout 3600` | The 510 GB load plus host tables is slow |

The pinned image accepts the layout spelling `private`; newer SGLang sources
call the same layout `per_rank`. This is a naming difference, not a different
memory layout.

Unlike the 4 × RTX PRO 6000 recipe, this H200 profile keeps the two Engram tables
in pinned host RAM rather than an NVMe row cache. NVMe is used for checkpoint and
cache storage; no runtime Engram reads are expected after loading.

### Measured allocation

| Item | Value |
| --- | ---: |
| Startup to ready | **562 s** (phase 2b, warm checkpoint cache) |
| Engram host tables | **386,736 MiB** measured; 16 mappings × 24,171 MiB; 100% huge pages, pinned |
| GPU full-KV pool | **45,075,200 tokens** |
| GPU SWA pool | **51,712 tokens** |
| HiCache host pool | **90,150,400 tokens** |
| HBM free after load | **12.9–13.4 GiB per GPU** |
| Host available after load | **617 GB** |
| Container RSS | **1.336 TiB** |

The Engram host cost is **386,736 MiB (about 378 GiB)**, not the naive ~203 GB
of checkpoint table bytes. With TP8, the runtime keeps two replicated 4-way
shards, one per table per rank. Plan host memory from the measured value.

## 5. Measured speed

All numbers are client-observed and use actual generated-token counts. They are
performance results, not quality scores. The short tests use 1,024 forced output
tokens; the long-context tests use 512 or 2,048 tokens as noted.

### Decode

| Load | Aggregate tok/s | Median per-request tok/s | Notes |
| --- | ---: | ---: | --- |
| c1, short | **181.1** | 181.1 | 1,024 output tokens |
| c4, short | **414.0** | — | 1,024 output tokens each; median not recorded |
| c16, short | **643.4** | — | 1,024 output tokens each; median not recorded |
| 4 × 400k warmed | **450.4** | **112.8** | 2,048 output tokens each |
| 16 × 400k warmed replay | **715.8** | **42.9** | Peak 16 running, 0 queue, 0 retracted; one straggler made the wall 81.6 s |
| 4 warmed + 4 cold 400k | **292.4** | **68.8** | Mixed/baseline ratio 0.649 |

The 16-active replay is a warmed, synthetic 2,048-token generation. It is not a
sustained 16-user coding workload.

### Long-context prefill and capacity

| Workload | Result |
| --- | --- |
| Single cold 400k TTFT | **74.7 s** |
| Four cold 400k TTFT | 68.4 / 135.4 / 201.6 / **268.3 s**; batch wall 268.5 s |
| 16 × 400k cold | **1068.9 s**, all codes recalled; completions serialized about 67 s apart |
| 16 × 400k warm 512-token pass | 452.9 aggregate / 26.5 median tok/s; peak 11 running |
| 16 × 400k warm 2,048-token replay | 715.8 aggregate / 42.9 median tok/s; peak 16 running |
| Near-limit 409,007-token total | Recalled correctly in **69.4 s** |
| Over-cap request | HTTP 400 rejection, not truncation |

Cold 400k prefill is serialized. Do not cite the 16-request cold wall as 16-way
prefill scaling.

### Phase history

| Profile | Change | c1 / c4 / c16 tok/s | Mixed 4+4 tok/s |
| --- | --- | --- | ---: |
| Phase 1 | 16 slots, no context cap | 139.5 / 256.5 / 336.4 | 29.6 |
| Phase 2a | `--context-length 409600` | 174.2 / 394.4 / 660.5 | 31.3 |
| Phase 2b | `--prefill-decode-interval 16`, chunk 4096 | **181.1 / 414.0 / 643.4** | **292.4** |

Phase 2b trades a small c16 decode loss and slower cold completion for a large
mixed prefill/decode gain. It is the current reference profile, not a claim that
every workload improves.

## 6. Feature acceptance

| Test | Result |
| --- | --- |
| OpenAI chat, separated reasoning | Passed |
| Tool call and second tool turn | Passed |
| 100k and 400k retrieval | Passed; natural synthetic records, chat template, unique early prefixes |
| Four independent 400k prefixes | Passed cold and warmed |
| 16-active warmed 400k replay | Passed; peak 16 running, 0 queue, 0 retracted |
| Near-limit 409,007-token total | Passed |
| HiCache host restoration | Only a 256-token host hit observed; no 400k restoration |
| Vision / video / audio | Not tested on this host |

## 7. Reproduce the measurements

Run one probe at a time against an idle engine. The scripts use only the Python
standard library and write timestamped reports under `evidence/`.

```bash
python3 -m py_compile tests/*.py

python3 tests/openai_probe.py --base-url http://127.0.0.1:8012 \
  --model pratt-high --mode smoke
python3 tests/openai_probe.py --base-url http://127.0.0.1:8012 \
  --model pratt-high --mode tools

python3 tests/decode_throughput.py --base http://127.0.0.1:8012 \
  --model pratt-high --concurrency 1,4,16 --output-tokens 1024 --rounds 2

python3 tests/cold_ttft.py --base-url http://127.0.0.1:8012 \
  --model pratt-high --concurrency 1 --target-tokens 400000 --output-tokens 512
python3 tests/cold_ttft.py --base-url http://127.0.0.1:8012 \
  --model pratt-high --concurrency 4 --target-tokens 400000 --output-tokens 512

python3 tests/long_c4_warm.py --base-url http://127.0.0.1:8012 \
  --model pratt-high --concurrency 16 --target-tokens 400000 --output-tokens 512
python3 tests/replay_c16_warm.py --base-url http://127.0.0.1:8012 \
  --model pratt-high --output-tokens 2048
python3 tests/mixed_prefill_decode.py --base-url http://127.0.0.1:8012 \
  --model pratt-high --warm-concurrency 4 --cold-concurrency 4 \
  --target-tokens 400000 --output-tokens 2048
```

Long-context tests create substantial GPU and host-memory load. Check running
and queued requests first, and do not run concurrent benchmarks from multiple
shells. The tests do not change the service configuration.

## 8. Provenance

| Component | Pinned source / value |
| --- | --- |
| Model | [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), revision `1bae0e1088993b9b82de08f59de5be5a34e42170` |
| Checkpoint | 510,286,023,000 tensor bytes; 48 shards |
| Engine | `lmsysorg/sglang@sha256:4a5d132a06a77c8331e15845f2e925adc788b00105097ad55409afa3f4fa4860` |
| Cookbook | [SGLang DeepSeek-V4.1 cookbook](https://lmsysorg.mintlify.app/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1), H200 / Flash-1017 / FP4 / low-latency / single node |
| Parallelism | TP8 / EP8, single node |
| Draft path | DSpark block 5; this checkpoint has no EAGLE/MTP path |
| Host Engram layout | `private` in the pinned image; equivalent to the cookbook's `per_rank` |
| HiCache | Cookbook H200 donor values: direct I/O, `page_first_direct`, ratio 2, `write_back` |

The cookbook marks the H200 Flash-1017 cell as `verified: false`. This project is
a measured preview validation, not a previously qualified recipe.

Related work: the 4 × RTX PRO 6000 Blackwell version of this model at
[`sybil-solutions/deepseek-v4.1-flash-4x-rtx-pro-6000`](https://github.com/sybil-solutions/deepseek-v4.1-flash-4x-rtx-pro-6000).
That recipe uses a different memory architecture: NVMe-backed Engram rows and
GPU-only KV. Do not copy its flags across without revalidating.

## 9. Repository contents

- `configs/systemd/candidates/` — reference and historical service units.
- `tests/` — chat, tools, decode, cold-TTFT, long-context and mixed-load probes.
- `evidence/` — timestamped reports, including failed or invalid probes.
- `scripts/` — checkpoint verification, startup capture and readiness helpers.
- `docs/RESULTS.md` — detailed measured results.
- `docs/DECODE_PROFILE.md` — decode kernel profile and batch-scaling finding.
- `docs/SCHEDULER_OVERLAP_TEST.md` — phase 2b overlap results and phase 2c plan.
- `docs/CONTEXT_409600_TEST.md` — context-cap test and rollback.
- `docs/RECIPE.md` — cookbook provenance and exact launch flags.
- `docs/OPERATIONS.md` — install, monitoring, shutdown and rollback.
- `docs/VALIDATION.md` — staged checks and evidence requirements.
- `docs/HANDOFF.md` — restart-safe operational handoff, kept separate from this recipe.

## 10. Non-claims

- No 16-way simultaneous cold 400k prefill result.
- No 400k HiCache host restoration; only a small host hit was observed.
- No vision, video or audio qualification.
- No sustained real coding/agent load, broad model-quality evaluation, or
  multi-user SLA.
- No gateway, TLS, authentication or client-routing qualification.
- No claim that the exact 409,600-token boundary was occupied.
- No claim that these results transfer to 4 × RTX PRO 6000, H100 or other
  GPU counts without a new validation.
