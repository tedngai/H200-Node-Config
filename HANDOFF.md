# Handoff — DeepSeek-V4.1-Flash-1017 phase 2a installed and measured

Updated: 2026-10-18. This is the restart-safe handoff for the isolated DSV4.1
experiment. Read `/home/advtech/AGENTS.md`, then this file, then
`docs/RESULTS.md` and `docs/SCHEDULER_OVERLAP_TEST.md`.

## Current state

- Project: `/home/advtech/dsv41-flash-inference`
- Branch: `dsv41-flash-1017`
- Remote: `https://github.com/tedngai/H200-Node-Config.git`
- Live unit: `/etc/systemd/system/sglang-dsv41.service`
- Installed candidate (phase 2a):
  `configs/systemd/candidates/sglang-dsv41-8013-mr16-nodelay-ctx409600.service`
- Previous phase 1 candidate (rollback):
  `configs/systemd/candidates/sglang-dsv41-8013-mr16-nodelay.service`
- API: `http://127.0.0.1:8013`, served name `dsv41-flash-1017`.
- Next 2b install moves the API to **8012 / pratt-high**, per user request.
- Image:
  `lmsysorg/sglang@sha256:4a5d132a06a77c8331e15845f2e925adc788b00105097ad55409afa3f4fa4860`
- Service is **active and idle** at handoff: 0 running, 0 queued, 0 retracted.
- The GLM-5.3 unit is unchanged and stopped. The public `pratt-high` alias and
  port 8012 were not touched.

### Effective phase 2a settings

From `/server_info` and startup logs:

| Setting | Value |
| --- | --- |
| `max_running_requests` | **16** |
| `min_free_slots_delay` | **0** |
| `prefill_decode_interval` | **0** (disabled) |
| `chunked_prefill_size` | **8192** |
| `swa_prefix_tails` | default **64** tails |
| Context length / max model len | **409,600** |
| GPU full-KV pool | **44,957,952 tokens** (unchanged) |
| SWA pool | **59,904 tokens** (`prefix_tails=64`, unchanged) |
| HiCache host pool | **89,915,904 tokens** |
| Static memory fraction | **0.80** |
| HBM used at idle | **~121.4 GB/GPU** |
| HBM free after all load | **7.0–7.5 GiB/GPU (minimum 7.0)** |
| Host available after load | **616 GB** |
| Container RSS | **1.337 TiB** |
| Phase 2a restart | 22:02:18 → 22:11:52 UTC, **574 s** |

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

## Phase 2a result — context cap 409600 (installed)

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

## Decode profile finding — phase 2a applied

A 2026-10-18 21:29–21:36 UTC diagnostic (`docs/DECODE_PROFILE.md`,
`evidence/decode-profile-20261018T213400Z/`) found that decode step time is set
by batch size, not context: ~33 ms at c1 and ~202 ms at c16 for both short and
400k prompts. 64% of the c16 step is the SM90 Triton DSA indexer scoring the
full **configured** 1,048,576-token width for every verify row.

Phase 2a is **installed and measured**:
`configs/systemd/candidates/sglang-dsv41-8013-mr16-nodelay-ctx409600.service`.
The cap raised c16 aggregate decode from 336.4 to **660.5 tok/s (1.963×)**,
c4 to 394.4 (1.538×), and c1 to 174.2 (1.249×). It also reduced idle HBM from
about 138 GB to **121.4 GB/GPU** and raised post-load free HBM to
**7.0–7.5 GiB/GPU (minimum 7.0)** without changing the KV or SWA pools. Full results are in
`docs/RESULTS.md`; apply/verify/rollback steps remain in
`docs/CONTEXT_409600_TEST.md`.

Do not bundle the overlap flags into a phase 2a restart; phase 2a isolated the
cap. The overlap candidate is a separate next step.

## Phase 2b/2c — overlap and tails candidates

Two separate candidates are prepared, **not installed**. Both retain the phase
2a `--context-length 409600`.

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

This candidate keeps `--swa-prefix-tails` at the default 64, so the SWA and
full-KV pools should stay at 59,904 / 44,957,952. The interval may increase
cold time-to-first-token.

### Cold TTFT gate

Phase 2a baseline (`tests/cold_ttft.py`, streaming, distinct 400k prompts):
single **61.841 s**; four cold TTFT **62.316 / 123.088 / 183.866 / 243.633 s**,
median **153.477 s**, max **243.633 s**, batch wall **243.983 s**.

Phase 2b gate: **single ≤ 90 s**, **worst-of-four ≤ 300 s**, **four-cold batch
wall ≤ 360 s**, all codes recalled. The user delegated this ceiling; it is a
proposed gate, not a production SLA.

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

### Install (privileged; the user runs this)

Check the engine is idle first, then install phase 2b:

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
`swa_tokens=59904`, full pool 44,957,952, HiCache enabled. Also check:

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
```

### Success criteria

Primary: absolute mixed warmed decode exceeds the phase 2a **31.342 tok/s**
without loss of cold recall, retractions or OOM. The cold TTFT gate is
**90 s single / 300 s worst-of-four / 360 s four-cold batch wall**, all codes
recalled. Report the ratio against phase 2a's **0.072**; do not use the old
0.11 comparison.

Secondary:

- c16 decode remains near the phase 2a 660 tok/s baseline (measure it, because
  the interval can add scheduling overhead);
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
  2a results/overlap update is the current branch head (see `git log`).
- `README.md`, `AGENTS.md`, `docs/RESULTS.md`,
  `docs/DECODE_PROFILE.md`, `docs/CONTEXT_409600_TEST.md`,
  `docs/SCHEDULER_OVERLAP_TEST.md`, `docs/OPERATIONS.md`, and
  `docs/VALIDATION.md` reflect phase 2a.
- The global `PROGRESS.xml`, `INFERENCE_CONTEXT.md`, and
  `NEXT_EXPERIMENT.md` were updated and synced to `glm53-inference/docs`.
  Those sync commits are **local only**; `main` was intentionally not pushed.

## Hard constraints and non-claims

- Do not change `/etc/systemd/system/sglang-glm53.service`, port 8012, or the
  public `pratt-high` alias as a side effect.
- The service is not enabled and uses `Restart=no`; do not enable it.
- Do not claim production readiness, sustained real coding load, 16
  simultaneous cold prefills, or a 400k HiCache restoration.
- Phase 2a's 16-active result is a synthetic warmed replay with a ~16 s
  plateau, not a real multi-user coding qualification.
- The mixed test used synthetic 400k records and forced output; it is not a
  broad quality evaluation.
- Keep the pinned image digest; do not pull mutable images or replacement
  weights.
- Preserve failed/invalid probes and distinguish cold prefill, GPU-cached
  reuse, host-cache restoration, and decode.

## Quick resume checklist

1. Confirm the live unit and `/server_info`; phase 2a should report
   `context_length=409600` and `/v1/models.max_model_len=409600`. If not,
   check whether a rollback occurred.
2. Confirm 0 running / 0 queued before any restart.
3. The next experiment is phase 2b, the overlap candidate
   (`configs/systemd/candidates/sglang-dsv41-8012-pratt-high-mr16-nodelay-ctx409600-overlap2b.service`),
   prepared but not installed. It adds `--prefill-decode-interval 16` and
   `--chunked-prefill-size 4096`, retains `--context-length 409600`, keeps
   default tails, and serves **8012 / pratt-high**. The cold-TTFT gate is
   90 s single / 300 s worst-of-four / 360 s four-cold batch wall.
4. After phase 2b install, verify model id `pratt-high`, `max_model_len=409600`,
   `context_length=409600`, `prefill_decode_interval=16`,
   `chunked_prefill_size=4096`, `swa_prefix_tails=None` (default 64),
   `swa_tokens=59904` and unchanged pool and HBM headroom.
5. Run cold TTFT, decode c16, 16×400k cold, 2048 replay and mixed test.
   Compare c16 to **660.5 tok/s**, mixed absolute to **31.342 tok/s**, mixed
   ratio to **0.072**, cold wall to **978.7 s**, and TTFT to the gate.
6. Check recall, queue/retractions, OOM, cold TTFT and post-load HBM.
7. Update docs/evidence, commit and push `dsv41-flash-1017`.
8. Test `--swa-prefix-tails 1024` separately via the
   `sglang-dsv41-8012-pratt-high-mr16-nodelay-ctx409600-tails1024.service`
   candidate.
