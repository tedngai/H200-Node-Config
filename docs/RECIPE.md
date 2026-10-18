# Recipe provenance and exact launch settings

## Source

Cookbook page:

```
https://lmsysorg.mintlify.app/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1
```

User-selected UI state:

```
hw=h200
variant=flash-1017
quant=fp4
strategy=low-latency
nodes=single
```

The pasted URL also contained `ax_hw=h100` and `ax_cfg=agg-tp8-ep8` in the
query string while the fragment said `hw=h200`. The `agg-tp8-ep8` cell on the
page is the H100 cell, but the H200 Flash-1017 low-latency cell independently
resolves to the same **TP8 / EP8** shape. This host is 8xH200, so the H200 cell
is authoritative for this project.

The cookbook marks the `flash-1017` H200 cell as **`verified: false`**. The
checkpoint was published/updated on 2026-10-17, after the cookbook snapshot.
This experiment is therefore a preview validation, not a previously qualified
recipe.

## Cookbook H200 Flash-1017 low-latency cell

Raw cell flags from the page:

```text
--trust-remote-code
--model-path {{MODEL_NAME}}
--tp 8
--ep-size 8
--mem-fraction-static 0.8
--attention-backend dsv4
--moe-runner-backend flashinfer_mxfp4
--enable-decoder-swa-bounded-replay
--cuda-graph-max-bs-decode 64
--reasoning-parser auto
--tool-call-parser auto
--host {{HOST_IP}}
--port {{PORT}}
```

`{{MODEL_NAME}}` resolves to `deepseek-ai/DeepSeek-V4.1-Flash-1017`.

The cookbook HiCache donor settings for this hardware/checkpoint are:

```text
--enable-hierarchical-cache
--hicache-io-backend direct
--hicache-mem-layout page_first_direct
--hicache-ratio 2
--hicache-write-policy write_back
```

The cookbook agentic cells additionally use:

```text
SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank
SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1
```

## Final candidate command

See
`configs/systemd/candidates/sglang-dsv41-8013-mr16-nodelay.service` for the
current phase 1 authoritative candidate. The original 4-request candidate is
`sglang-dsv41-8013.service`. The rendered engine command is:

```text
sglang serve \
  --model-path deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  --served-model-name dsv41-flash-1017 \
  --trust-remote-code \
  --tp 8 \
  --ep-size 8 \
  --max-running-requests 4 \
  --mem-fraction-static 0.8 \
  --attention-backend dsv4 \
  --moe-runner-backend flashinfer_mxfp4 \
  --enable-decoder-swa-bounded-replay \
  --cuda-graph-max-bs-decode 64 \
  --speculative-algorithm DSPARK \
  --speculative-dspark-block-size 5 \
  --enable-hierarchical-cache \
  --hicache-io-backend direct \
  --hicache-mem-layout page_first_direct \
  --hicache-ratio 2 \
  --hicache-write-policy write_back \
  --reasoning-parser auto \
  --tool-call-parser auto \
  --watchdog-timeout 3600 \
  --enable-metrics \
  --enable-request-time-stats-logging \
  --enable-cache-report \
  --log-requests \
  --log-requests-level 1 \
  --host 0.0.0.0 \
  --port 8013
```

Environment:

```text
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1
SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=private
```

## Image pin

```text
lmsysorg/sglang@sha256:4a5d132a06a77c8331e15845f2e925adc788b00105097ad55409afa3f4fa4860
```

SGLang reports `0.0.0.dev1+gda64c5cbb`; the image is CUDA 13.0.3 and was built
2026-09-11. The cookbook's H200 Docker image is the mutable tag
`lmsysorg/sglang:dev-dsv41`; this project pinned the digest actually pulled so a
later retag cannot silently change the engine.

## Pinned-image compatibility finding: `per_rank` vs `private`

The cookbook and the newer SGLang source use the env value `per_rank` for the
host engram layout. The pinned image's source (`sglang/srt/layers/engram.py`)
implements the same layout under the name `private`:

```python
assert layout in ("shared", "private"), layout
```

`choose_layout("per_rank")` returns `"per_rank"` and then trips that assertion.
The candidate therefore uses `SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=private`.
This is a naming compatibility fix, not a change of intent: `private` is one
anonymous shard per rank, keeps the all-reduce, and is backed by anonymous
transparent huge pages (`madvise`). The cookbook's `per_rank` description
matches this exactly.

If a later image is pinned, re-check whether the spelling changed; do not
assume `private` is still accepted.

## Why each addition exists

| Setting | Reason |
| --- | --- |
| `--speculative-algorithm DSPARK` | User requested DSPARK. The checkpoint's own draft path; the cookbook states there is no EAGLE/MTP path for V4.1. |
| `--speculative-dspark-block-size 5` | Cookbook value for the DSpark low-latency cells. |
| `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1` | User requested host offload of the two engram tables. Frees ~203 GB of HBM for KV. |
| `SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=private` | Cookbook-recommended per-rank private-shard layout for 2 MiB-page backing; one shard per TP rank, ~25.4 GB/rank. The cookbook calls this `per_rank`; this image's `_HostTable` accepts only `shared`/`private` and asserts otherwise, so `private` is the compatible spelling of the same semantics. |
| HiCache five flags | User requested KV offload. Values are the cookbook H200 donor values, not guessed. |
| `--max-running-requests 4` | DSpark's hook resets an unset value to 48. Four preserves the user's approved active-request target and prevents accidental over-admission. |
| `--watchdog-timeout 3600` | The checkpoint is 510 GB and the engram host table adds ~203 GB of host-side loading. The cookbook agentic cells use 3600 to avoid a premature startup watchdog. |
| metrics/cache-report/logging | Observability only. Needed to prove active requests, KV pool, HiCache hits and resolved backends. Does not change model math. |
| port 8013 / name `dsv41-flash-1017` | Isolation from the live `pratt-high`/8012 service. Not a production cutover. |

## Deliberate omissions

| Omitted | Why |
| --- | --- |
| `--context-length` | Left at the checkpoint default in phase 1. Phase 2a installed **409600** to shrink the H200 indexer's configured scoring width; it remains in the prepared overlap candidate. See `docs/CONTEXT_409600_TEST.md`. |
| `--chunked-prefill-size` | Left at the image default. The detailed H200 agentic cell uses 4096, but the selected low-latency cell does not. |
| `--swa-prefix-tails` | Left at the recipe default. The detailed agentic cell tunes it per concurrency; this is a scoped test. |
| `--max-prefill-tokens`, `--prefill-decode-interval` | Agentic throughput tuning, not part of the selected low-latency cell. |
| EAGLE/MTP flags | Not supported by this checkpoint; DSPARK replaces them. |
| Dynamo, NATS, etcd, Mooncake, PD disaggregation | The selected shape is single-node aggregated with the plain SGLang OpenAI server. |
| Multi-node flags | Single node only. |

## Cookbook warnings that matter here

1. **Do not judge throughput from a backend override.** The cookbook says
   `--attention-backend`, `--moe-runner-backend` and `--fp8-gemm-backend` are
   auto-selected and that forcing the wrong one can fall back to a slow Triton
   path. The selected cell explicitly passes `dsv4` and `flashinfer_mxfp4`; the
   startup log must confirm those are the resolved backends.
2. **DSpark is not proportional to draft tokens.** The step cost is fixed, so
   the gain is accept length divided by that cost. Measure acceptance.
3. **HiCache is host DRAM, not an extension of the active GPU decode pool.**
   A host hit must be demonstrated after GPU eviction; merely enabling the flag
   is not proof.
4. **Engram host offload is bitwise-neutral in the cookbook's claim**, but it
   costs host RAM, a longer load, and huge-page backing. This project verifies
   the tables actually moved to host memory and that the output remains sane.
5. **Output is not bitwise stable across batch composition** on this backend.
   Do not use bitwise equality across different batch sizes as a quality gate.

## Measured runtime allocation (2026-10-18)

The engine is running with the candidate above. Measured values supersede the
estimates in the earlier sections:

| Item | Measured |
| --- | ---: |
| Startup to ready | 585 s first start; **578 s** phase 1 |
| GPU KV pool | **45,420,032 tokens** (4-request); **44,957,952** (mr16/phase 1) |
| HiCache host pool | **90,840,064 tokens** (4-request); **89,915,904** (mr16/phase 1), ratio 2 |
| Engram host table | **386.7 GiB** total; 24,171 MiB per table per rank, 16 mappings |
| Engram huge-page backing | 100% |
| HBM free per GPU after load | ~7.3 GB (4-request); **5.0–5.4 GB** (phase 1) |
| Container RSS after load | 1.34–1.42 TiB |
| Host available after load | ~613–617 GB |

The engram measurement is important: 24,171 MiB per table per rank is a **4-way
shard** of the 101.5 GB table, so with TP8 the runtime keeps two replicated
4-way shards. That is why the host cost is ~386.7 GiB rather than the ~203 GB
of checkpoint table bytes. Use the measured value for capacity planning.

## Checkpoint facts relevant to capacity planning

From `config.json` and the safetensors index/header:

- 40 decoder layers, 384 routed experts, top-6 plus one shared expert.
- 512-wide MLA KV latent; 1,048,576 max positions.
- Two engram layers, ids 1 and 14.
- Each engram embedding tensor: `[384006168, 256]` FP8 plus
  `[384006168, 8]` FP8 scales, ~101.5 GB per layer.
- Two engram tables: ~203.1 GB total.
- Total checkpoint: 510,286,023,000 bytes (~510.3 GB).
- Non-engram weights after host offload: ~307 GB, or ~38.4 GB per GPU at TP8.
