# DeepSeek-V4.1-Flash-1017 on 8xH200

Isolated SGLang experiment for the DeepSeek-V4.1-Flash-1017 preview checkpoint.
This project is deliberately separate from the live `glm53-inference` project so
the validated GLM-5.3 service and its exact rollback path stay intact.

**Status:** preparing (2026-10-18). The preview image and 510.3 GB checkpoint are
being acquired. The engine has **not** been started, and the live GLM-5.3
service is still the authoritative deployment until the hand-off commands below
are run by the user.

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
   ~101.5 GB FP8 tables out of HBM (~203 GB total, ~25.4 GB per TP rank) and
   gives the freed HBM to the KV pool. The cookbook calls this layout
   `per_rank`; the pinned image's code calls the same private-shard layout
   `private` and rejects the newer `per_rank` spelling. This is documented in
   `docs/RECIPE.md`.
3. **KV offload with HiCache**, using the cookbook H200 donor settings:
   `--enable-hierarchical-cache --hicache-io-backend direct
   --hicache-mem-layout page_first_direct --hicache-ratio 2
   --hicache-write-policy write_back`.
4. **Explicit `--max-running-requests 4`.** The low-latency cell omits this,
   but the pinned image's DSpark hook resets an unset value to **48**. Four is
   the user's approved active-request target and prevents accidental
   over-admission during this test.

## Test identity and safety boundary

- Test port: **8013** (the live GLM service remains on 8012).
- Test served name: **`dsv41-flash-1017`**.
- New systemd unit: `configs/systemd/candidates/sglang-dsv41-8013.service`.
- The test unit is **not enabled** and uses `Restart=no`, so a failed 510 GB
  load does not loop and thrash the GPUs.
- This is not a production cutover. Moving the public `pratt-high` alias or
  port 8012 to this model requires a separate change and client validation.
- The live GLM unit and its 400k DP4+MTP+HiCache configuration are preserved
  unchanged for rollback.

## Resource budget

| Item | Estimate |
| --- | ---: |
| Checkpoint on disk | 510.3 GB |
| Engram tables on host | ~203 GB total, ~25.4 GB/rank |
| Non-engram weights in HBM | ~307 GB total, ~38.4 GB/GPU |
| H200 HBM per GPU | 141 GB |
| Static memory fraction | 0.80 (~113 GB/GPU budget) |
| Host RAM | ~2.0 TB total |
| HiCache host tier | ratio 2 of the GPU KV pool, measured at startup |

The actual GPU KV pool, host HiCache pool, pinned-RAM footprint, startup time
and free memory are recorded in `evidence/`; none of the numbers above are
claimed as measured until the engine is running.

## Hand-off commands (privileged)

The account running this project has Docker access but **no passwordless sudo**.
The user runs the stop/install/start commands personally:

```bash
# 1. Stop the validated GLM-5.3 service and free the eight GPUs.
sudo systemctl stop sglang-glm53.service

# 2. Install and start the isolated DSV4.1 test unit.
sudo install -o root -g root -m 0644 \
  /home/advtech/dsv41-flash-inference/configs/systemd/candidates/sglang-dsv41-8013.service \
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

- `docs/RECIPE.md` — cookbook provenance, exact flags, additions and omissions.
- `docs/VALIDATION.md` — staged checks and evidence requirements.
- `docs/OPERATIONS.md` — startup, monitoring, shutdown and rollback.
- `configs/systemd/candidates/` — pinned candidate unit.
- `tests/` — OpenAI chat, reasoning, tool-call and long-context probes.
- `evidence/` — pre-change snapshot and timestamped test reports.

## Non-claims

A successful startup is not a sustained-load or quality qualification. This
project will distinguish:

- cold prefill vs GPU-cached reuse vs HiCache host restoration vs decode,
- synthetic smoke tests vs real coding/agent tasks,
- a scoped concurrency probe vs production admission capacity.

The public alias, gateway, OpenCode workstations and their metadata are outside
this project and are not changed by it.
