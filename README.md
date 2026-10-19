# DeepSeek-V4.1-Flash-1017 on 8xH200

Isolated SGLang experiment for the DeepSeek-V4.1-Flash-1017 preview checkpoint.
This project is deliberately separate from the live `glm53-inference` project so
the validated GLM-5.3 service and its exact rollback path stay intact.

**Status:** **phase 2b installed and measured on 8012 / pratt-high**
(2026-10-19). It retains the 409,600 context cap and adds
`--prefill-decode-interval 16` / `--chunked-prefill-size 4096`. Mixed
4-warm/4-cold decode rose from **31.3 to 292.4 tok/s** (ratio **0.649** vs
0.072), while the cold-TTFT gate passed (74.7 s single; four-cold max 268.3 s /
wall 268.5 s). Costs: c16 decode 643.4 tok/s (0.974× phase 2a), 16 × 400k cold
wall **1068.9 s**, and the 16-active replay wall **81.6 s** due to one
straggler. This is a scoped result, not production qualification. A scoped
HiCache host hit was observed. See `docs/RESULTS.md`.

**Prepared, not installed:** phase-2c SWA-tails candidate. Phase 2b is
installed on **8012 / pratt-high**; see `docs/SCHEDULER_OVERLAP_TEST.md` for the
separate tails test. A **production boot candidate** is also prepared,
preserving every 2b flag while adding bounded crash restart and a `/raid`
mount dependency. The installed 2b unit is currently disabled at boot and uses
`Restart=no`; see `docs/OPERATIONS.md` for the exact enable/reboot procedure.

## What is being tested

Canonical cookbook selection:

- https://lmsysorg.mintlify.app/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1
- Hardware: **H200**, checkpoint variant **flash-1017**, quant **FP4**,
  strategy **low-latency**, **single node**.
- Selected cell: **Agg TP8 / EP8**.
- Model: `deepseek-ai/DeepSeek-V4.1-Flash-1017` (510.3 GB, 763 B parameters,
  FP8 dense weights with FP4 routed experts and two FP8 engram tables).
- Image: `lmsysorg/sglang:dev-dsv41`, pinned by digest
  `sha256:4a5d132a06a77c8331e15845f2e925adc788b00105097ad55409afa3f4fa4860`
  (SGLang `0.0.0.dev1+gda64c5cbb`, CUDA 13.0.3).

User-requested additions on top of the cookbook cell:

1. **DSPARK** draft: `--speculative-algorithm DSPARK
   --speculative-dspark-block-size 5`. The cookbook explains that DSPARK is this
   checkpoint's own draft path; there is no EAGLE/MTP path for V4.1.
2. **Engram table offload to host RAM**:
   `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1` and
   `SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=private`. This moves the two
   ~101.5 GB FP8 tables out of HBM and gives the freed HBM to the KV pool. The
   cookbook calls this layout `per_rank`; the pinned image's code calls the same
   private-shard layout `private` and rejects the newer `per_rank` spelling.
   **Measured host cost: 386.7 GiB**, not the naive 203 GB checkpoint-byte
   estimate. See `docs/RESULTS.md` for the per-rank measurements.
3. **KV offload with HiCache**, using the cookbook H200 donor settings:
   `--enable-hierarchical-cache --hicache-io-backend direct
   --hicache-mem-layout page_first_direct --hicache-ratio 2
   --hicache-write-policy write_back`.
4. **Explicit `--max-running-requests 4`.** The low-latency cell omits this,
   but the pinned image's DSpark hook resets an unset value to **48**. Four is
   the user's approved active-request target and prevents accidental
   over-admission during this test.

## Test identity and safety boundary

- Test port: **8012** (the live GLM service also uses 8012 and is stopped).
- Test served name: **`pratt-high`**.
- Current installed candidate (phase 2b):
  `configs/systemd/candidates/sglang-dsv41-8012-pratt-high-mr16-nodelay-ctx409600-overlap2b.service`
  (`--prefill-decode-interval 16 --chunked-prefill-size 4096`, context 409600).
- Phase 2a fallback on the same endpoint:
  `configs/systemd/candidates/sglang-dsv41-8012-pratt-high-mr16-nodelay-ctx409600.service`.
- Prepared phase 2c SWA-tails candidate (not installed; separate test):
  `configs/systemd/candidates/sglang-dsv41-8012-pratt-high-mr16-nodelay-ctx409600-tails1024.service`.
- Historical 8013 test candidates:
  `configs/systemd/candidates/sglang-dsv41-8013-mr16-nodelay-ctx409600.service`,
  `...-8013-mr16-nodelay.service`, `...-8013-mr16.service`.
- The test unit is **not enabled** and uses `Restart=no`, so a failed 510 GB
  load does not loop and thrash the GPUs.
- This is not a production qualification. The engine intentionally serves the
  client-facing 8012/pratt-high endpoint, but gateway/client behavior still
  needs separate validation.
- The live GLM unit and its 400k DP4+MTP+HiCache configuration are preserved
  unchanged for rollback.

## Resource budget (measured unless marked estimate)

| Item | Value |
| --- | ---: |
| Checkpoint on disk | 510.3 GB |
| Engram tables on host | **386.7 GiB** measured (16 × 24,171 MiB) |
| Non-engram weights in HBM | ~37.8 GB/GPU measured (plus 1.11 GB DSpark) |
| H200 HBM per GPU | 141 GB |
| Static memory fraction | 0.80 |
| GPU KV pool | **45,420,032 tokens** (4-request start); **44,957,952** (mr16/phase 1); **45,075,200** (phase 2b) |
| HiCache host pool | **90,840,064 tokens** (4-request); **89,915,904** (mr16/phase 2a); **90,150,400** (phase 2b) |
| HBM free per GPU | ~7.3 GB after 4-request load; **~12.9–13.4 GiB after phase 2b load** (cache state differs from 2a) |
| Host RAM | ~2.0 TB total; **~617 GB available after phase 2b load** |
| Container RSS | **1.336 TiB** after phase 2b load |
| Startup | **585 s** first start; **578 s** phase 1; **574 s** phase 2a; **562 s** phase 2b |

The engine is running; the actual GPU KV pool, host HiCache pool, pinned-RAM
footprint, startup time and free memory are recorded in `evidence/` and
`docs/RESULTS.md`.

## Hand-off commands (privileged)

The account running this project has Docker access but **no passwordless sudo**.
The user runs the stop/install/start commands personally:

```bash
# 1. Stop the validated GLM-5.3 service and free the eight GPUs.
sudo systemctl stop sglang-glm53.service

# 2. Install and start the isolated DSV4.1 phase 2b test unit on 8012/pratt-high.
sudo install -o root -g root -m 0644 \
  /home/advtech/dsv41-flash-inference/configs/systemd/candidates/sglang-dsv41-8012-pratt-high-mr16-nodelay-ctx409600-overlap2b.service \
  /etc/systemd/system/sglang-dsv41.service
sudo systemctl daemon-reload
sudo systemctl start sglang-dsv41.service
```

Rollback to the validated GLM-5.3 service:

```bash
sudo systemctl stop sglang-dsv41.service
sudo systemctl start sglang-glm53.service
```

The GLM unit was not edited. Its 400k DP4+MTP+HiCache profile, image digest and
API identity remain exactly as installed before this experiment.

## Documentation

- `HANDOFF.md` — restart-safe current state, phase 2b results, tails runbook,
  rollback and non-claims.
- `docs/DECODE_PROFILE.md` — decode kernel profile, root cause of batch-scaled
  step time and phase 2a/2b measurements.
- `docs/CONTEXT_409600_TEST.md` — phase 2a context-cap reference, apply/verify
  commands, validation sequence, criteria and rollback.
- `docs/SCHEDULER_OVERLAP_TEST.md` — phase 2b overlap results and phase 2c
  tails plan.
- `docs/RESULTS.md` — measured startup, memory, interfaces, decode, long
  context, mixed load and HiCache results.
- `docs/RECIPE.md` — cookbook provenance, exact flags, additions and omissions.
- `docs/VALIDATION.md` — staged checks and evidence requirements.
- `docs/OPERATIONS.md` — startup, monitoring, shutdown and rollback.
- `configs/systemd/candidates/` — pinned candidate units.
- `tests/` — chat, reasoning, tools, long-context, cold-TTFT and decode probes.
- `evidence/` — pre-change snapshot and timestamped test reports.

## Non-claims

A successful startup is not a sustained-load or quality qualification. This
project will distinguish:

- cold prefill vs GPU-cached reuse vs HiCache host restoration vs decode,
- synthetic smoke tests vs real coding/agent tasks,
- a scoped concurrency probe vs production admission capacity.

The public alias, gateway, OpenCode workstations and their metadata are outside
this project and are not changed by it.
