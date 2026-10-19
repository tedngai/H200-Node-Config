# Handoff — DeepSeek-V4.1-Flash-1017 phase 2b installed and measured

> This is the operational handoff, kept separate from the public recipe
> `README.md`. Paths below are relative to the repository root.

Updated: 2026-10-19. This is the restart-safe handoff for the isolated DSV4.1
experiment. Read `/home/advtech/AGENTS.md`, then this file, then
`docs/RESULTS.md` and `docs/SCHEDULER_OVERLAP_TEST.md`.

## Current state

- Project: `/home/advtech/dsv41-flash-inference`
- Branch: `dsv41-flash-1017`
- Remote: `https://github.com/tedngai/H200-Node-Config.git`
- Live unit: `/etc/systemd/system/sglang-dsv41.service`
- Installed candidate (phase 2b):
  `configs/systemd/candidates/sglang-dsv41-8012-pratt-high-mr16-nodelay-ctx409600-overlap2b.service`
- Phase 2a fallback on the same endpoint:
  `configs/systemd/candidates/sglang-dsv41-8012-pratt-high-mr16-nodelay-ctx409600.service`
- Historical phase 1/2a test endpoint (8013):
  `configs/systemd/candidates/sglang-dsv41-8013-mr16-nodelay-ctx409600.service`
- API: `http://127.0.0.1:8012`, served name **`pratt-high`**.
- Image:
  `lmsysorg/sglang@sha256:4a5d132a06a77c8331e15845f2e925adc788b00105097ad55409afa3f4fa4860`
- Service is **active and idle** at handoff: 0 running, 0 queued, 0 retracted.
  The installed unit is currently **disabled at boot** and uses `Restart=no`.
- Production boot candidate (prepared, **not installed**):
  `configs/systemd/candidates/sglang-dsv41-8012-pratt-high-mr16-nodelay-ctx409600-overlap2b-production.service`.
  It preserves all 2b flags and adds `Restart=on-failure` (30 s),
  `StartLimitIntervalSec=1800`/`StartLimitBurst=3`, and
  `RequiresMountsFor=/raid/advtech/cache/huggingface`. See
  `docs/OPERATIONS.md`.
- Both `sglang-glm53.service` and `sglang-glm53-flash.service` are currently
  **enabled** at boot but inactive. Disable both when installing the DSV4
  production unit; keep them installed for rollback.
- The GLM-5.3 unit is unchanged and stopped. The DSV4 unit intentionally binds
  **8012 / pratt-high**; the gateway and client configs were not changed.

### Effective phase 2b settings

From `/server_info` and startup logs:

| Setting | Value |
| --- | --- |
| `max_running_requests` | **16** |
| `min_free_slots_delay` | **0** |
| `prefill_decode_interval` | **16** |
| `chunked_prefill_size` | **4096** |
| `swa_prefix_tails` | default **64** tails |
| Context length / max model len | **409,600** |
| GPU full-KV pool | **45,075,200 tokens** |
| SWA pool | **51,712 tokens** (`prefix_tails=64`) |
| HiCache host pool | **90,150,400 tokens** |
| Static memory fraction | **0.80** |
| HBM used at idle | **~121.4 GB/GPU** |
| HBM free after load | **12.9–13.4 GiB/GPU** (cache state differs from 2a) |
| Host available after load | **617 GB** |
| Container RSS | **1.336 TiB** |
| Phase 2b restart | 23:31:16 → 23:40:38 UTC, **562 s** |

## Phase 1 result — capacity gate passed

Evidence:

- `evidence/long-400000-c16-warm-20261018T204521Z.json`
- `evidence/replay-c16-warm-20261018T205717Z.json`
- `evidence/mixed-prefill-decode-w4-c4-20261018T210728Z.json`
- `evidence/phase1-nodelay-summary-20261018T210803Z.json`
- `evidence/postload-phase1-20261018T210803Z/`

| Test | Result |
| --- | --- |
| 16 × 400k cold | 991.4 s, all recalled; admission still staggered (queue 0.01–913.1 s) |
| 512-token warm | 35.2 s, 290.367 aggregate / 16.377 median tok/s; peak **15** running, 0 queue, 0 retracted |
| 2048-token warm replay | 96.3 s, **372.979 aggregate / 22.878 median tok/s**; peak **16** running, 0 queue, 0 retracted; 16 active for ~56 s |
| Mixed baseline | no-prefill **268.669 / 67.203**; with four fresh 400k prefills **29.556 / 7.388**; ratio **0.11** |
| Post-load | 0 post-ready runtime errors; smoke and tools passed |

The 512-token pass peaking at 15 is an admission-stagger artifact: all 16
requests were submitted within 0.09 s, but server-side tokenization/ingress
staggered scheduler entry and the first request finished before the last was
admitted. The 2048-token replay is the valid 16-active proof. Do **not** claim
16 simultaneous cold 400k prefills; the cold pass remained staggered.

## Phase 2a result — context cap 409600 (historical)

Only `--context-length 409600` changed. Startup was 574 s;
`max_model_len=409600`, `swa_tokens=59904` and the 44,957,952-token pool were
unchanged. Measured on the same scripted coding prompts:

| Load | Phase 1 | Phase 2a | Ratio |
| --- | ---: | ---: | ---: |
| c1 aggregate | 139.495 | **174.176** | 1.249× |
| c4 aggregate | 256.492 | **394.381** | 1.538× |
| c16 aggregate | 336.409 | **660.484** | **1.963×** |

- 2048-token 16-active replay: **55.576 s / 704.503 tok/s / 41.426 median**,
  peak **16 running**, 0 queue, 0 retracted. Phase 1 was 96.254 s / 372.979 /
  22.878.
- 16 × 400k cold: **978.719 s**, all codes recalled. Prefill logs show one new
  sequence per batch, so it was interleaved, **not** a 16-sequence batch.
- 512-token warm: 474.665 tok/s aggregate but peak 10 running; faster decode
  plus ingress stagger, so the replay is the 16-active proof.
- Mixed 4 warmed + 4 cold 400k: warmed baseline **434.154**, mixed **31.342**,
  ratio **0.072** (phase 1: 0.11). Absolute mixed improved ~6%; the ratio fell
  because the warmed baseline improved more. All four cold prompts recalled.
- 409,025-token total near-limit prompt recalled; over-cap request rejected
  HTTP 400, not truncated.
- Post-load HBM free **7.0–7.5 GiB/GPU (minimum 7.0)**, 0 runtime errors, idle 0/0/0.
  One-time non-fatal `_hc_mix_stats_reduce_kernel` compile warnings remain.

Full evidence: `evidence/phase2a-summary-20261018T225300Z.json` and
`docs/RESULTS.md`.

## Phase 2b result — overlap on 8012/pratt-high (installed)

Candidate:
`configs/systemd/candidates/sglang-dsv41-8012-pratt-high-mr16-nodelay-ctx409600-overlap2b.service`.
Adds `--prefill-decode-interval 16` and `--chunked-prefill-size 4096`; startup
**562 s**. Startup pools: SWA **51,712**, full KV **45,075,200**, HiCache
**90,150,400**.

| Check | Result |
| --- | --- |
| TTFT gate | c1 **74.721 s**; c4 max **268.290 s** / wall **268.498 s** — pass |
| Decode c1/c4 | **181.1 / 414.0 tok/s** (1.04× / 1.05× phase 2a) |
| Decode c16 | **643.4 tok/s** (0.974× phase 2a); transient submission queue |
| 16 × 400k cold | **1068.9 s**, all recalled; completions serialized ~67 s apart |
| 2048 replay | **715.8 aggregate / 42.9 median**, peak 16, 0 queue; wall **81.6 s** due to one straggler |
| Mixed 4-warm/4-cold | **292.4 aggregate / 68.8 median**, ratio **0.649**; phase 2a 31.3 / 0.072; cold wall 284.0 s; peak 4, queue 3, 0 retracted |
| Near-limit | 409,007-token total recalled via pratt-high in 69.4 s |
| Post-load | 0 runtime errors; 12.9–13.4 GiB free HBM/GPU; smoke/tools pass |

This passes the delegated cold-TTFT gate and is a large mixed-load win. Costs:
cold 16-request wall +9.2%, replay wall +46.8% from a straggler, c16 decode
-2.6%. Keep installed for real coding evaluation; not production-qualified.

## Decode profile finding — phase 2a/2b applied

A 2026-10-18 21:29–21:36 UTC diagnostic (`docs/DECODE_PROFILE.md`,
`evidence/decode-profile-20261018T213400Z/`) found that decode step time is set
by batch size, not context: ~33 ms at c1 and ~202 ms at c16 for both short and
400k prompts. 64% of the c16 step is the SM90 Triton DSA indexer scoring the
full **configured** 1,048,576-token width for every verify row.

Phase 2a was **installed and measured**:
`configs/systemd/candidates/sglang-dsv41-8013-mr16-nodelay-ctx409600.service`.
The cap raised c16 aggregate decode from 336.4 to **660.5 tok/s (1.963×)**,
c4 to 394.4 (1.538×), and c1 to 174.2 (1.249×). It also reduced idle HBM from
about 138 GB to **121.4 GB/GPU** and raised post-load free HBM to
**7.0–7.5 GiB/GPU (minimum 7.0)** without changing the KV or SWA pools. Full
results are in `docs/RESULTS.md`; apply/verify/rollback steps remain in
`docs/CONTEXT_409600_TEST.md`.

Phase 2b currently serves that cap on 8012/pratt-high. It leaves c16 decode
within a few percent (643.4 tok/s), raises c1/c4 slightly, and raises mixed
4-warm/4-cold from 31.3 to **292.4 tok/s**. It does not fix cold-call
serialization or the replay straggler. Do not bundle phase 2c tails into 2b;
test them separately.

## Phase 2b installed / phase 2c candidate

Phase 2b is **installed** on 8012/pratt-high. Phase 2c (tails) is prepared and
**not installed**. Both retain `--context-length 409600`.

### Phase 2b — scheduler overlap (interval + chunk)

```text
configs/systemd/candidates/sglang-dsv41-8012-pratt-high-mr16-nodelay-ctx409600-overlap2b.service
```

It adds two flags to phase 2a:

```text
--prefill-decode-interval 16
--chunked-prefill-size 4096
```

- `--prefill-decode-interval 16`: after each prefill batch, the scheduler
  defers new prefill for 16 scheduler/decode rounds. This protects warmed
  decode from continuous prefill competition. `0` disables it.
- `--chunked-prefill-size 4096`: caps each prefill chunk at 4096 new tokens
  instead of 8192. A 400k prompt becomes roughly 98 chunks instead of 49,
  allowing finer prefill/decode interleaving at the cost of more scheduler
  iterations and potentially slower cold prefill.

This candidate keeps `--swa-prefix-tails` at the default 64. The 4096 chunk
changes the measured SWA cap to **51,712** (was 59,904) and the full pool to
**45,075,200** (was 44,957,952). The interval increased cold TTFT modestly:

### Cold TTFT gate

Phase 2a baseline (`tests/cold_ttft.py`, streaming, distinct 400k prompts):
single **61.841 s**; four cold TTFT **62.316 / 123.088 / 183.866 / 243.633 s**,
median **153.477 s**, max **243.633 s**, batch wall **243.983 s**.

Measured phase 2b: single **74.721 s**; four cold TTFT **68.378 / 135.406 /
201.636 / 268.290 s**, median **168.521 s**, max **268.290 s**, batch wall
**268.498 s**. All recalled.

Phase 2b gate: **single ≤ 90 s**, **worst-of-four ≤ 300 s**, **four-cold batch
wall ≤ 360 s**, all codes recalled. Passed. The user delegated this ceiling; it
is a proposed gate, not a production SLA.

### Phase 2c — SWA prefix tails only

```text
configs/systemd/candidates/sglang-dsv41-8012-pratt-high-mr16-nodelay-ctx409600-tails1024.service
```

It adds only:

```text
--swa-prefix-tails 1024
```

Each tail is one 128-token sliding window plus one 256-token page (384 slots).
The default is 64 tails at 16 requests. Expected but **unverified** effect:
`swa_tokens` around 420,000 instead of 59,904, roughly +8 GB/GPU, and a
full-KV pool around 40M instead of 44.96M. Confirm the startup lines. This is a
multi-session prefix-reuse experiment, not a decode-speed change for 16 active
requests; test it separately.

Do not bundle the tails into the interval test. If both help, combine them in a
later candidate.

### Reinstall (already installed; privileged)

Check the engine is idle first, then reinstall phase 2b if needed:

```bash
sudo install -o root -g root -m 0644 \
  /home/advtech/dsv41-flash-inference/configs/systemd/candidates/sglang-dsv41-8012-pratt-high-mr16-nodelay-ctx409600-overlap2b.service \
  /etc/systemd/system/sglang-dsv41.service
sudo systemctl daemon-reload
sudo systemctl restart sglang-dsv41.service
```

The agent must not run privileged install/restart commands. Startup takes about
10 minutes.

### Verify readiness and effective flags

```bash
systemctl status sglang-dsv41.service --no-pager | head -5
curl -s http://127.0.0.1:8012/v1/models | python3 -c \
  'import json,sys; d=json.load(sys.stdin); print([(m["id"],m.get("max_model_len")) for m in d["data"]])'
curl -s http://127.0.0.1:8012/server_info | python3 -c \
  'import json,sys; d=json.load(sys.stdin); print({k:d.get(k) for k in ("max_running_requests","min_free_slots_delay","context_length","prefill_decode_interval","chunked_prefill_size","swa_prefix_tails","enable_hierarchical_cache")})'
```

Expected for phase 2b: model id **pratt-high**, `max_model_len=409600`,
`context_length=409600`, `max_running_requests=16`,
`min_free_slots_delay=0`, `prefill_decode_interval=16`,
`chunked_prefill_size=4096`, `swa_prefix_tails=None` (default 64),
`swa_tokens=51712`, full pool 45,075,200, HiCache enabled. Also check:

```bash
journalctl -u sglang-dsv41.service --since '20 min ago' --no-pager | \
  grep -E 'DSV4 SWA sizing|DSV4 memory calculation|ready to roll|Traceback|RuntimeError|CUDA error|OOM'
```

Check 0 running / 0 queued before accepting the restart.

### Phase 2b tests

Run these sequentially on an idle engine, not concurrently:

```bash
cd /home/advtech/dsv41-flash-inference

python3 tests/cold_ttft.py --base-url http://127.0.0.1:8012 \
  --model pratt-high --concurrency 1 --target-tokens 400000 \
  --output-tokens 512 --label cold-ttft-overlap2b-c1
python3 tests/cold_ttft.py --base-url http://127.0.0.1:8012 \
  --model pratt-high --concurrency 4 --target-tokens 400000 \
  --output-tokens 512 --label cold-ttft-overlap2b-c4

python3 tests/decode_throughput.py --base http://127.0.0.1:8012 \
  --model pratt-high --concurrency 16 --output-tokens 1024 --rounds 1 \
  --label decode-throughput-overlap2b

python3 tests/long_c4_warm.py --base-url http://127.0.0.1:8012 \
  --model pratt-high --concurrency 16 --target-tokens 400000 \
  --output-tokens 512

python3 tests/replay_c16_warm.py --base-url http://127.0.0.1:8012 \
  --model pratt-high --output-tokens 2048

python3 tests/mixed_prefill_decode.py --base-url http://127.0.0.1:8012 \
  --model pratt-high --warm-concurrency 4 --cold-concurrency 4 \
  --target-tokens 400000 --output-tokens 2048

python3 tests/openai_probe.py --base-url http://127.0.0.1:8012 \
  --model pratt-high --mode smoke
python3 tests/openai_probe.py --base-url http://127.0.0.1:8012 \
  --model pratt-high --mode tools
```

### Success criteria

Primary: retain the phase 2b **292.4 aggregate / 68.8 median tok/s** mixed
result (ratio **0.649**) without loss of cold recall, retractions or OOM. The
cold TTFT gate remains **90 s single / 300 s worst-of-four / 360 s four-cold
batch wall**, all codes recalled.

Secondary:

- c16 decode remains near the phase 2b **643.4 tok/s** result;
- 16 × 400k cold still completes and all marker codes are recalled;
- the 2048-token replay still reaches **16 running**, 0 queue, 0 retracted;
- `swa_tokens`, full-KV pool and HBM headroom remain safe;
- no post-ready runtime errors.

Phase 2c is separate: install
`configs/systemd/candidates/sglang-dsv41-8012-pratt-high-mr16-nodelay-ctx409600-tails1024.service`,
verify `swa_prefix_tails=1024` and the reduced pool, then measure multi-session
prefix reuse with long gaps. Do not treat it as a 16-active decode test.

After the runs, capture a post-load evidence directory, update
`docs/RESULTS.md`, `docs/SCHEDULER_OVERLAP_TEST.md`, `PROGRESS.xml`,
`INFERENCE_CONTEXT.md`, commit the project, and push the `dsv41-flash-1017`
branch. Do not push this project's history to `main`.

## Rollback

From phase 2b or 2c back to the 8012 / pratt-high phase 2a unit:

```bash
sudo install -o root -g root -m 0644 \
  /home/advtech/dsv41-flash-inference/configs/systemd/candidates/sglang-dsv41-8012-pratt-high-mr16-nodelay-ctx409600.service \
  /etc/systemd/system/sglang-dsv41.service
sudo systemctl daemon-reload
sudo systemctl restart sglang-dsv41.service
```

Historical phase 1 unit (moves the API to 8013):

```bash
sudo install -o root -g root -m 0644 \
  /home/advtech/dsv41-flash-inference/configs/systemd/candidates/sglang-dsv41-8013-mr16-nodelay.service \
  /etc/systemd/system/sglang-dsv41.service
sudo systemctl daemon-reload
sudo systemctl restart sglang-dsv41.service
```

Deeper mr16 rollback:

```bash
sudo install -o root -g root -m 0644 \
  /home/advtech/dsv41-flash-inference/configs/systemd/candidates/sglang-dsv41-8013-mr16.service \
  /etc/systemd/system/sglang-dsv41.service
sudo systemctl daemon-reload
sudo systemctl restart sglang-dsv41.service
```

Full GLM-5.3 rollback:

```bash
sudo systemctl stop sglang-dsv41.service
sudo systemctl start sglang-glm53.service
```

## Git and documentation state

- Local `dsv41-flash-1017` is the project source of truth and is pushed to
  `origin/dsv41-flash-1017`.
- Phase 1 commit: `88538bb` — "Record phase 1 16-active and mixed
  prefill/decode baseline". Phase 2a preparation commit: `37a683f`; the phase
  2b results update is the current branch head (see `git log`).
- `README.md`, `AGENTS.md`, `docs/RESULTS.md`,
  `docs/DECODE_PROFILE.md`, `docs/CONTEXT_409600_TEST.md`,
  `docs/SCHEDULER_OVERLAP_TEST.md`, `docs/OPERATIONS.md`, and
  `docs/VALIDATION.md` reflect phase 2b/2c.
- The global `PROGRESS.xml`, `INFERENCE_CONTEXT.md`, and
  `NEXT_EXPERIMENT.md` were updated and synced to `glm53-inference/docs`.
  Those sync commits are **local only**; `main` was intentionally not pushed.

## Hard constraints and non-claims

- Do not change `/etc/systemd/system/sglang-glm53.service`. The DSV4 test
  intentionally binds **8012 / pratt-high**; do not reroute the gateway or
  change client configs without explicit approval.
- The service is not enabled and uses `Restart=no`; do not enable it.
- Do not claim production readiness, sustained real coding load, 16
  simultaneous cold prefills, or a 400k HiCache restoration.
- Phase 2b's 16-active replay reaches peak 16 but has an 81.6 s wall and a
  straggler; it is not a real multi-user coding qualification.
- The mixed test used synthetic 400k records; the 9.33× gain is not yet a
  broad coding-quality or task-time evaluation.
- Keep the pinned image digest; do not pull mutable images or replacement
  weights.
- Preserve failed/invalid probes and distinguish cold prefill, GPU-cached
  reuse, host-cache restoration, and decode.

## Quick resume checklist

1. Confirm the live unit and `/server_info`; phase 2b should report
   `context_length=409600`, `prefill_decode_interval=16`,
   `chunked_prefill_size=4096`, model id **pratt-high**, and
   `/v1/models.max_model_len=409600`. If not, check whether a rollback occurred.
2. Confirm 0 running / 0 queued before any restart.
3. Phase 2b is installed on 8012/pratt-high. The next step is real coding /
   agent workload evaluation, not another scheduler flag. Do not add phase 2c
   tails without a separate multi-session reuse test.
4. If reinstalling, verify model id `pratt-high`, `max_model_len=409600`,
   `context_length=409600`, `prefill_decode_interval=16`,
   `chunked_prefill_size=4096`, `swa_prefix_tails=None`,
   `swa_tokens=51712`, full pool 45,075,200 and HBM headroom.
5. Re-run cold TTFT, c16 decode, 16×400k, 2048 replay and mixed if config
   changes. Compare c16 to **643.4 tok/s**, mixed absolute to **292.4 tok/s**,
   ratio to **0.649**, cold wall to **1068.9 s**, and TTFT to
   90/300/360 s.
6. Check recall, queue/retractions, OOM, cold TTFT and post-load HBM.
7. Update docs/evidence, commit and push `dsv41-flash-1017`.
8. Test `--swa-prefix-tails 1024` separately via the
   `sglang-dsv41-8012-pratt-high-mr16-nodelay-ctx409600-tails1024.service`
   candidate.
