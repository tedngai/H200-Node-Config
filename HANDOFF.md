# Handoff — DeepSeek-V4.1-Flash-1017 phase 1 complete, phase 2 pending

Updated: 2026-10-18. This is the restart-safe handoff for the isolated DSV4.1
experiment. Read `/home/advtech/AGENTS.md`, then this file, then
`docs/RESULTS.md` and `docs/SCHEDULER_OVERLAP_TEST.md`.

## Current state

- Project: `/home/advtech/dsv41-flash-inference`
- Branch: `dsv41-flash-1017`
- Remote: `https://github.com/tedngai/H200-Node-Config.git`
- Live unit: `/etc/systemd/system/sglang-dsv41.service`
- Installed candidate:
  `configs/systemd/candidates/sglang-dsv41-8013-mr16-nodelay.service`
- Phase 2a candidate (prepared, **not installed**), phase 1 plus
  `--context-length 409600`:
  `configs/systemd/candidates/sglang-dsv41-8013-mr16-nodelay-ctx409600.service`
- API: `http://127.0.0.1:8013`, served name `dsv41-flash-1017`
- Image:
  `lmsysorg/sglang@sha256:4a5d132a06a77c8331e15845f2e925adc788b00105097ad55409afa3f4fa4860`
- Service is **active and idle** at handoff: 0 running, 0 queued, 0 retracted.
- The GLM-5.3 unit is unchanged and stopped. The public `pratt-high` alias and
  port 8012 were not touched.

### Effective phase 1 settings

From `/server_info` and startup logs:

| Setting | Value |
| --- | --- |
| `max_running_requests` | **16** |
| `min_free_slots_delay` | **0** |
| `prefill_decode_interval` | **0** (disabled) |
| `chunked_prefill_size` | **8192** |
| `swa_prefix_tails` | default **64** tails |
| Context length | **1,048,576** |
| GPU full-KV pool | **44,957,952 tokens** |
| SWA pool | **59,904 tokens** (`prefix_tails=64`) |
| HiCache host pool | **89,915,904 tokens** |
| Static memory fraction | **0.80** |
| HBM free after 16-request load | **5.0–5.4 GiB/GPU** |
| Host available after load | **617 GiB** |
| Container RSS | **1.341 TiB** |
| Phase 1 restart | 20:04:52 → 20:14:30 UTC, **578 s** |

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

## Decode profile finding — read before phase 2

A 2026-10-18 21:29–21:36 UTC diagnostic (`docs/DECODE_PROFILE.md`,
`evidence/decode-profile-20261018T213400Z/`) found that decode step time is set
by batch size, not context: ~33 ms at c1 and ~202 ms at c16 for both short and
400k prompts. 64% of the c16 step is the SM90 Triton DSA indexer scoring the
full **configured** 1,048,576-token width for every verify row.

Phase 2a is now **prepared, not installed**:
`configs/systemd/candidates/sglang-dsv41-8013-mr16-nodelay-ctx409600.service`.
It adds only `--context-length 409600` to the installed phase 1 unit. The
scripted phase 1 decode baseline is
`evidence/decode-throughput-baseline-20261018T215614Z.json` (c1 132.9/146.1,
c4 259.0/254.0, c16 335.8/337.0 aggregate tok/s, peak 16 running). The exact
apply, verify, test, acceptance and rollback steps are in
`docs/CONTEXT_409600_TEST.md`. The projected c16 gain is ~1.8×, unmeasured.

Do not add the overlap flags in the same restart: phase 2a isolates the cap.

## Phase 2 plan — prefill/decode overlap candidate

Candidate (prepared, **not installed**):

```text
configs/systemd/candidates/sglang-dsv41-8013-mr16-nodelay-overlap.service
```

It adds three flags to the current phase 1 unit:

```text
--prefill-decode-interval 16
--chunked-prefill-size 4096
--swa-prefix-tails 1024
```

### What each flag does

- `--prefill-decode-interval 16`: after each prefill batch, the scheduler
  defers new prefill for 16 scheduler/decode rounds. This protects warmed
  decode from continuous prefill competition. `0` disables it.
- `--chunked-prefill-size 4096`: caps each prefill chunk at 4096 new tokens
  instead of 8192. A 400k prompt becomes roughly 98 chunks instead of 49,
  allowing finer prefill/decode interleaving at the cost of more scheduler
  iterations and potentially slower cold prefill.
- `--swa-prefix-tails 1024`: reserves SWA-pool room for 1024 cached prefix
  tails. Each tail is one 128-token sliding window plus one 256-token page
  (384 slots). The default is 64 tails at 16 requests. This helps many cached
  long prefixes stay reusable, but it is far more than 16 active requests need.

Expected but **unverified** phase 2 startup effect: `swa_tokens` around
420,000 instead of 59,904, roughly +8 GB/GPU of SWA/state allocation, and a
full-KV pool around 40M tokens instead of 44.96M. Confirm the actual values in
the `DSV4 SWA sizing` and `DSV4 memory calculation` startup lines. If the full
pool or HBM headroom is unacceptable, roll back.

Because phase 2 bundles three changes, a positive result will not attribute
which flag helped. A negative or ambiguous result should be followed by an
isolated test of interval + chunk size first, then the 1024-tail reservation.

### Install (privileged; the user runs this)

Check the engine is idle first, then:

```bash
sudo install -o root -g root -m 0644 \
  /home/advtech/dsv41-flash-inference/configs/systemd/candidates/sglang-dsv41-8013-mr16-nodelay-overlap.service \
  /etc/systemd/system/sglang-dsv41.service
sudo systemctl daemon-reload
sudo systemctl restart sglang-dsv41.service
```

The agent must not run privileged install/restart commands. Startup takes about
10 minutes.

### Verify readiness and effective flags

```bash
systemctl status sglang-dsv41.service --no-pager | head -5
curl -s http://127.0.0.1:8013/server_info | python3 -c \
  'import json,sys; d=json.load(sys.stdin); print({k:d.get(k) for k in ("max_running_requests","min_free_slots_delay","prefill_decode_interval","chunked_prefill_size","swa_prefix_tails","enable_hierarchical_cache")})'
```

Expected: `max_running_requests=16`, `min_free_slots_delay=0`,
`prefill_decode_interval=16`, `chunked_prefill_size=4096`,
`swa_prefix_tails=1024`, HiCache enabled. Also check:

```bash
journalctl -u sglang-dsv41.service --since '20 min ago' --no-pager | \
  grep -E 'DSV4 SWA sizing|DSV4 memory calculation|ready to roll|Traceback|RuntimeError|CUDA error|OOM'
```

Check 0 running / 0 queued before accepting the restart.

### Phase 2 tests

Run these sequentially on an idle engine, not concurrently:

```bash
cd /home/advtech/dsv41-flash-inference

python3 tests/long_c4_warm.py \
  --concurrency 16 --target-tokens 400000 --output-tokens 512

python3 tests/replay_c16_warm.py --output-tokens 2048

python3 tests/mixed_prefill_decode.py \
  --warm-concurrency 4 --cold-concurrency 4 \
  --target-tokens 400000 --output-tokens 2048
```

### Success criteria

Primary: the mixed/baseline warmed-decode ratio improves versus phase 1's
**0.11**, with no loss of cold recall and no increase in retractions or OOM.

Secondary:

- 16 × 400k cold still completes and all marker codes are recalled;
- the 2048-token replay still reaches **16 running**, 0 queue, 0 retracted;
- `swa_tokens`, full-KV pool and HBM headroom remain safe;
- no post-ready runtime errors;
- cold batch wall and cold forward duration are recorded, because the overlap
  settings may trade cold-prefill latency for warmed decode progress.

After the runs, capture a post-load evidence directory, update
`docs/RESULTS.md`, `docs/SCHEDULER_OVERLAP_TEST.md`, `PROGRESS.xml`,
`INFERENCE_CONTEXT.md`, commit the project, and push the `dsv41-flash-1017`
branch. Do not push this project's history to `main`.

## Rollback

Exact phase 1 rollback (after phase 2 is installed):

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
  prefill/decode baseline". This handoff is committed on top of it.
- `README.md`, `AGENTS.md`, `docs/RESULTS.md`,
  `docs/SCHEDULER_OVERLAP_TEST.md`, `docs/OPERATIONS.md`, and
  `docs/VALIDATION.md` reflect phase 1.
- The global `PROGRESS.xml`, `INFERENCE_CONTEXT.md`, and
  `NEXT_EXPERIMENT.md` were updated and synced to `glm53-inference/docs`.
  Those sync commits are **local only**; `main` was intentionally not pushed.

## Hard constraints and non-claims

- Do not change `/etc/systemd/system/sglang-glm53.service`, port 8012, or the
  public `pratt-high` alias as a side effect.
- The service is not enabled and uses `Restart=no`; do not enable it.
- Do not claim production readiness, sustained real coding load, 16
  simultaneous cold prefills, or a 400k HiCache restoration.
- Phase 1's 16-active result is a synthetic warmed replay, not a real
  multi-user coding qualification.
- The mixed test used synthetic 400k records and forced output; it is not a
  broad quality evaluation.
- Keep the pinned image digest; do not pull mutable images or replacement
  weights.
- Preserve failed/invalid probes and distinguish cold prefill, GPU-cached
  reuse, host-cache restoration, and decode.

## Quick resume checklist

1. Confirm the live unit and `/server_info`; the installed unit may still be
   phase 1, so check `/v1/models.max_model_len` before assuming 409600.
2. Confirm 0 running / 0 queued before any restart.
3. If phase 2a is not installed, ask the user to install the ctx409600
   candidate. Do not add the overlap flags in the same restart.
4. Verify `max_model_len=409600`, unchanged phase 1 flags, `swa_tokens=59904`,
   full-pool value and HBM headroom.
5. Run `tests/decode_throughput.py` first and compare with the scripted
   baseline; then 16×400k cold, 2048 replay, mixed test and near-limit/tools.
6. Compare c16 decode to 335.8/337.0 aggregate and the mixed ratio to 0.11;
   check recall, queue/retractions, OOM and headroom.
7. Update docs/evidence, commit and push `dsv41-flash-1017`.
8. Only after a phase 2a decision, isolate the overlap pair; test
   `--swa-prefix-tails 1024` separately afterward.
