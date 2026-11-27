# AngelSpec DFly2 test — DeepSeek-V4.1-Flash-1017

**2026-11-27: DFly2 served the full checkpoint on eight H200s. The operator
selected it for production.** The DSpark control was stopped during startup
at 14:30 UTC to free the GPUs for promotion; it produced no same-image
benchmark. The previous production DSpark unit SHA256 is
`ef89ec375d118ac7aad2d5b8cb9cec2acf152a7b179e05f2e9641c1c9249bf41`
and matches the committed `*overlap2b-production.service` exactly.

## Pins and results

- [Tencent AngelSpec](https://github.com/Tencent/AngelSpec) trains the drafter.
  The serving code is draft [SGLang PR #53538](https://github.com/sgl-project/sglang/pull/53538)
  at `82308b9d9e1685aa9404272cc499c4c195610913`. The PR author tested the
  `-1017` target with TP4/EP4 and reported mixed performance vs DSpark; the
  local test uses TP8/EP8 on H200. The production SGLang commit does not have
  DFLY, and the patch does not apply cleanly. The smaller November nightly
  backport also failed test collection on a missing `VerifyAccept` API.
- Target: existing `deepseek-ai/DeepSeek-V4.1-Flash-1017` revision
  `1bae0e1088993b9b82de08f59de5be5a34e42170`. Drafter:
  [AngelSlim/DS-V4.1-Flash-DFly2-B6](https://huggingface.co/AngelSlim/DS-V4.1-Flash-DFly2-B6)
  revision `d08e55a6897b1974307ea9771038838de5986e66`, four shards
  (8.85 GB) verified against their published SHA256 digests. The real model
  configs agree on hidden and attention dimensions, the token vocabulary,
  and the attention layout for paired layers 28, 32, and 36.
- Local test image: `local/dsv41-angelspec:82308b9d9`, image ID
  `sha256:d8f59019486c911daa7b0e1b6872c91a513ce1cf1a9b096b678a2fb3378c6b2d`.
  Uses PR Python source over the digest-pinned CUDA 13.0.3 November 22
  nightly, with two dependency updates pinned by wheel hash and the Python
  radix tree selected. For an algorithm comparison, use this same image for
  both DFly2 and DSpark. The test keeps the production context cap, 16-request
  limit, 0.80 static memory fraction, HiCache and host Engram placement.
- The experimental image passed **15 CPU tests and 8 subtests**. Both
  launch configurations resolved on the host, with six-token verification
  blocks, FlashMLA, reasoning and tool parsers, and the expected limits.
  A synthetic **dual-cache FlashMLA** call and a CUDA-graph replay with
  changed sparse lengths matched an analytical reference on one H200
  (see `tests/angelspec_flashmla_smoke.py`). These preflight checks loaded no
  target weights.
- Live DSpark, same production container: two rounds each at c1 and c4,
  short deterministic coding prompts, no forced EOS, all ten requests hit
  the 1,024-token cap. Median client-observed aggregate rates:
  **184.0 tok/s at c1**, **393.6 tok/s at c4**. Four active requests observed,
  up to three transiently queued, no retractions or failures. C1 and c4
  share one short prompt per round. See
  `evidence/angelspec-live-dspark-baseline-20261127.json`. This is a current
  cross-build reference. The October phase 2b DSpark results reached
  **181.1 / 414.0 / 643.4 aggregate tok/s** at c1/c4/c16, respectively.
  See `docs/DECODE_PROFILE.md`. Neither baseline isolates algorithm effects.

## Full-checkpoint DFly2 pilot

On the first launch with a cold compilation cache, SGLang logged readiness
at 14:23:29.880 UTC, **19 min 17 s** after the container started. DeepGEMM JIT,
FlashInfer tuning, and HiCache allocation extended startup. The dedicated
compile cache is now retained across test containers. The target and
standalone DFly2 drafter loaded on all eight ranks. Effective settings:
TP8/EP8, 409,600 context, 16 active requests, 0.80 static fraction and six
verification slots. **44,794,624** GPU KV tokens, **89,589,248** HiCache host
tokens; minimum free HBM **22,442 MiB** at readiness and **17,416 MiB** after
the probes.

DFly2 short natural coding (two rounds at each concurrency, greedy sampling,
all 42 requests hit the 1,024-token cap without forced EOS):

| Concurrent | Aggregate tok/s | Median per-request tok/s | Peak active / waiting |
| ---: | ---: | ---: | ---: |
| 1 | 240.7 | 240.7 | 1 / 0 |
| 4 | 621.1 | 160.6 | 4 / 3 |
| 16 | 1,749.3 | 122.3 | 16 / 15 |

No request failures or retractions. The most recent decode batch after the
short suite reported 3.70 accepted tokens per verification, including the
bonus token (53.9% of proposed drafts); that gauge is a snapshot.

- Plain chat, separated reasoning, parsed tool call, and two-turn tool
  completion passed. A sampled two-turn SQL task with schema lookup and an
  executable fixture passed in **14.11 s**, including the first sampled
  request on this process.
- Four independent ~64K cold prompts recalled all codes in **7.53 s** wall.
  Their warmed forced-output replay at c4 yielded **769.6 aggregate tok/s**
  (4,096 tokens in 5.322 s), with **63,744 cached tokens per request** and
  four active. The initial 64K probe completed but could not write its report
  because the temporary evidence directory did not exist; the reported rerun
  used four fresh prefixes.
- A fresh ~400K prompt recalled all three codes with **13.85 s TTFT** and
  **13.98 s** elapsed; scheduler logs summed to **399,973 new tokens** and
  **zero cached tokens** over 98 chunks. This proves scoped retrieval near
  the cap; it does not establish host restoration or broad coding quality.

Raw client evidence: `evidence/angelspec-runtime-20261127/`.

## Production promotion

DFly2 is serving production at `8012/pratt-high`. With the preserved test
compilation cache, the container started at 14:44:57 UTC and logged readiness
at 14:50:58 UTC (**6 min 1 s**). Live settings matched the pilot, including
409,600 context, 16 maximum active requests, HiCache, and six verify slots.
Chat, separated reasoning, tool parsing and a two-turn tool exchange passed
after startup. Zero service restarts and zero retractions were observed;
minimum HBM free after smoke was **24,388 MiB**. The GPU and HiCache pools
matched the pilot. This is a startup and interface check.

To reproduce, run `bash scripts/build_angelspec.sh` from the repository root
and resolve `local/dsv41-angelspec:82308b9d9` to its local image ID with
`docker image inspect --format '{{.Id}}' local/dsv41-angelspec:82308b9d9`.
Place the pinned target in `huggingface/` within your cache root. For example,
with the Hugging Face CLI and that target cache already populated:

```bash
export DSV41_CACHE_ROOT=/raid/user/cache
revision=d08e55a6897b1974307ea9771038838de5986e66
mkdir -p "$DSV41_CACHE_ROOT/sglang/angelspec/runtime-82308b9d9"
hf download AngelSlim/DS-V4.1-Flash-DFly2-B6 --revision "$revision" \
  --local-dir "$DSV41_CACHE_ROOT/sglang/angelspec/DS-V4.1-Flash-DFly2-B6/$revision"
```

Set the cache paths and local image ID in the unit before installing:

```bash
sudo install -o root -g root -m 0644 \
  configs/systemd/candidates/sglang-dsv41-8012-pratt-high-dfly2-production.service \
  /etc/systemd/system/sglang-dsv41.service
sudo systemctl daemon-reload
sudo systemctl start sglang-dsv41.service
```

After the ready log, confirm that the running image matches the pinned ID,
`/v1/models` shows `pratt-high` and 409,600 tokens, and `/server_info`
reports `DFLY`, six verify tokens and HiCache. Run a scoped chat/reasoning
and tool round trip on `8012/pratt-high` after checking the endpoint is idle.

Rollback if needed: stop the service, install the previous
`sglang-dsv41-8012-pratt-high-mr16-nodelay-ctx409600-overlap2b-production.service`
from the same candidates directory, daemon-reload and start. No change to its
pinned image or checkpoint is needed.

## Maintenance test

The existing engine uses all eight GPUs and had about 12 GiB free per GPU
during preflight. A second full engine requires stopping it. The first launch
spent **19 min 17 s** on startup with cold compilation caches. Budget for that
first load, the DSpark control, and the production reload. Wait for the test
container's automatic removal before starting another under the same name.

From the repository root, set `DSV41_CACHE_ROOT` to the local cache root and
verify the two test configs before the window if rebuilt:

```bash
bash scripts/run_angelspec.sh check dfly
bash scripts/run_angelspec.sh check dspark
```

At the start of the agreed window, confirm the live service has zero running
and queued requests:

```bash
curl -fsS http://127.0.0.1:8012/metrics | grep -E '^sglang:num_(running|queue)_reqs'
```

The operator runs `sudo systemctl stop sglang-dsv41.service`. Use the test
listener on `127.0.0.1:8013`:

```bash
bash scripts/run_angelspec.sh run dfly
docker logs -f dsv41-angelspec
```

Wait for readiness and confirm `/server_info` reports DFLY and six verify
tokens. Record startup time, device memory, GPU and host pool sizes, and
warnings. At readiness, run from the project directory:

```bash
base=http://127.0.0.1:8013
model=dsv41-angelspec
algorithm=dfly
python3 tests/openai_probe.py --base-url "$base" --model "$model" --mode smoke
python3 tests/openai_probe.py --base-url "$base" --model "$model" --mode tools
python3 tests/decode_throughput.py --base "$base" --model "$model" \
  --concurrency 1,4,16 --output-tokens 1024 --rounds 2 --label "angelspec-$algorithm"
python3 tests/long_c4_warm.py --base-url "$base" --model "$model" \
  --concurrency 4 --target-tokens 64000 --output-tokens 1024
python3 tests/angelspec_sql_task.py --base-url "$base" --model "$model" \
  --report "evidence/angelspec-$algorithm-sql.json"
python3 tests/cold_ttft.py --base-url "$base" --model "$model" \
  --concurrency 1 --target-tokens 400000 --output-tokens 512
```

Capture speculative accept length, actual running/queued/retracted requests,
client wall time and output tokens, prefix cache hits and correctness. Test
greedy and sampled chat, tool-call round trips, and a repeatable multi-turn
coding task with executable tests before any production decision. The draft
PR currently rejects grammar-constrained decoding; ordinary auto tools need
an independent smoke check. If the pilot passes, stop the test container,
launch `bash scripts/run_angelspec.sh run dspark`, and repeat the same probes
with `algorithm=dspark`. Treat forced-output long decode and natural coding
as separate measurements.

Stop the test container with `docker stop -t 120 dsv41-angelspec`. The
operator runs `sudo systemctl start sglang-dsv41.service` to restore production
on its existing unit and image. Verify `8012/pratt-high`, context 409,600,
DSpark, HiCache, and a brief chat/tool smoke after readiness.
