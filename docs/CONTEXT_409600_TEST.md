# Phase 2a — context cap 409600 (installed and measured)

Status: **installed 2026-10-18 22:02 UTC; scoped validation passed.** Candidate:

```text
configs/systemd/candidates/sglang-dsv41-8013-mr16-nodelay-ctx409600.service
```

Phase 2b (interval + chunk) was subsequently installed on 8012/pratt-high; see
`docs/SCHEDULER_OVERLAP_TEST.md` and `docs/RESULTS.md`. This runbook remains the
phase 2a reference and rollback.

Preparation summary: `evidence/phase2a-prepared-20261018T215639Z.json`.
Preflight idle snapshot: `evidence/preflight-ctx409600-20261018T215639Z/`.
Measured summary: `evidence/phase2a-summary-20261018T225300Z.json`.
This runbook is retained for reinstall, verification and rollback.

Measured highlights:

- `max_model_len=409600`; `swa_tokens=59904` and the 44,957,952-token full pool
  were unchanged.
- C16 decode **660.5 tok/s** (1.963× phase 1); c4 1.538×; c1 1.249×.
- 16-active 2048-token replay **704.5 tok/s**, peak 16 running, 0 queue,
  0 retracted.
- 16 × 400k cold all recalled in **978.7 s**; prefill remained one sequence
  per batch, not a 16-sequence cold batch.
- Mixed/baseline ratio **0.072** (phase 1: 0.11); absolute mixed decode
  improved about 6%.
- Post-load HBM free **7.0–7.5 GiB/GPU (minimum 7.0)**; 0 runtime errors.

It is byte-identical to the installed phase 1 unit except for one engine flag
and the unit description:

```diff
-  --max-running-requests 16 \
-  --min-free-slots-delay 0 \
+  --max-running-requests 16 \
+  --min-free-slots-delay 0 \
+  --context-length 409600 \
```

Do **not** add the overlap flags in this step. Phase 2a isolates the context
cap so the decode gain is attributable.

## Why

The H200 DSA indexer falls back to a Triton kernel that scores every position
up to the **configured** context width for every DSpark verify row. At c16 that
kernel was 64% of the phase 1 decode step (136 of 212 ms), while the real 400k
prompts use only ~39% of the configured width. Capping the context at
**409,600** shrank the indexer width from 1,048,896 to about 409,920. Measured
c16 throughput rose **1.96×**, close to the projected ~1.8×; c1 rose 1.25×.

The cap **rejects** prompts that exceed it; it does not truncate them.
`/v1/models.max_model_len` reports 409600. The approved 400k target fits with
about 9.6k tokens of headroom including output; a 409,025-token total passed.

## Measured startup

- `/v1/models.max_model_len`: **409600**.
- `DSV4 SWA sizing`: **unchanged** at `swa_tokens=59904, prefix_tails=64`.
- `DSV4 memory calculation`: full pool **unchanged** at 44,957,952 tokens.
- Startup: 22:02:18→22:11:52 UTC, **574 s**.
- Idle HBM used ~121.4 GB/GPU; post-load free HBM **7.0–7.5 GiB/GPU (minimum 7.0)**.

## Client impact

Only clients pointed directly at port 8013 see the 409,600 limit. The public
`pratt-high` alias and port 8012 are unchanged, and no workstation or gateway
configuration is modified by this step. If this engine later becomes the
production endpoint, client context/compaction limits must be at or below
409,600; otherwise large prompts will be rejected.

## Apply or reinstall (privileged; the user runs this)

This candidate is already installed. To reinstall it after a rollback, confirm
the engine is idle (0 running / 0 queued), then:

```bash
sudo install -o root -g root -m 0644 \
  /home/advtech/dsv41-flash-inference/configs/systemd/candidates/sglang-dsv41-8013-mr16-nodelay-ctx409600.service \
  /etc/systemd/system/sglang-dsv41.service
sudo systemctl daemon-reload
sudo systemctl restart sglang-dsv41.service
```

Startup takes about 10 minutes. The agent must not run privileged commands.

## Verify readiness and effective settings

```bash
systemctl status sglang-dsv41.service --no-pager | head -5
curl -s http://127.0.0.1:8013/v1/models | python3 -c \
  'import json,sys; print(json.load(sys.stdin)["data"][0]["max_model_len"])'
curl -s http://127.0.0.1:8013/server_info | python3 -c \
  'import json,sys; d=json.load(sys.stdin); print({k:d.get(k) for k in ("max_running_requests","min_free_slots_delay","chunked_prefill_size","prefill_decode_interval","swa_prefix_tails","hicache_ratio","status")})'
journalctl -u sglang-dsv41.service --since '20 min ago' --no-pager | \
  grep -E 'DSV4 SWA sizing|DSV4 memory calculation|ready to roll|Traceback|RuntimeError|CUDA error|OOM'
```

Accept only if `max_model_len` is **409600**, the three phase 1 flags are
unchanged, `swa_tokens=59904`, no runtime error appears, and the engine reports
0 running / 0 queued.

## Validation sequence

Run sequentially on an idle engine. Do not run them concurrently.

### 1. Decode A/B

```bash
cd /home/advtech/dsv41-flash-inference
python3 tests/decode_throughput.py \
  --concurrency 1,4,16 --output-tokens 1024 --rounds 2 \
  --label decode-throughput-ctx409600
```

Compare against the phase 1 scripted baseline
`evidence/decode-throughput-baseline-20261018T215614Z.json`:

| Load | Baseline aggregate (rounds 1/2) | Baseline median per request |
| --- | ---: | ---: |
| c1 | 132.884 / 146.106 | 132.894 / 146.113 |
| c4 | 259.018 / 253.966 | 65.819 / 64.559 |
| c16 | 335.794 / 337.024 | 21.727 / 21.968 |

Primary decode gate: **c16 aggregate at least 1.5× baseline** (about 505 tok/s)
without queueing or retractions. Report the measured value even if it misses.

### 2. 16 × 400k cold + 512 warm

```bash
python3 tests/long_c4_warm.py \
  --concurrency 16 --target-tokens 400000 --output-tokens 512
```

All 16 marker codes must be recalled. Cold admission is still expected to be
staggered; do not claim 16 simultaneous cold prefills.

### 3. 16-active 2048-token replay

```bash
python3 tests/replay_c16_warm.py --output-tokens 2048
```

Must still reach **16 running, 0 queued, 0 retracted**. Record aggregate and
median rates, and compare to phase 1's 372.979 / 22.878.

### 4. Mixed prefill/decode

```bash
python3 tests/mixed_prefill_decode.py \
  --warm-concurrency 4 --cold-concurrency 4 \
  --target-tokens 400000 --output-tokens 2048
```

All four cold prompts must recall. Compare the mixed/baseline ratio to phase 1's
**0.11**. Phase 2a targets decode speed, not prefill interference, so the ratio
may not improve; a decode gain with an unchanged ratio is still useful.

### 5. Near-limit and interface checks

```bash
python3 tests/openai_probe.py --mode long --prompt-tokens 409000
python3 tests/openai_probe.py --mode smoke
python3 tests/openai_probe.py --mode tools
```

The 409,000-token prompt plus 512 output tokens totals 409,512, just under the
cap. It must recall all three codes. A prompt above 409,600 should be rejected
with an error, not silently truncated; this boundary rejection is not separately
load-tested unless requested.

### 6. Post-load snapshot

Capture metrics, GPU/RAM, recent log and an error scan excluding request logs,
as in `evidence/postload-phase1-20261018T210803Z/`. Confirm HBM headroom and
0 retractions/OOM.

## Result against the criteria

- **Met:** `max_model_len=409600`; `swa_tokens=59904`; full pool unchanged.
- **Met:** c16 decode **1.963×** baseline (c4 1.538×, c1 1.249×).
- **Met:** 16 × 400k cold all recalled in **978.7 s**; 512 warm and 2048
  replay completed.
- **Met:** replay reached **16 running**, 0 queue, 0 retracted, at 704.5 tok/s.
- **Met:** 409,025-token near-limit retrieval; tools and separated reasoning
  passed.
- **Met:** no OOM, runtime error, retraction or HBM regression; post-load free
  HBM 7.0–7.5 GiB/GPU (minimum 7.0).
- **Not met:** the mixed/baseline ratio did **not** improve against 0.11; it
  fell to **0.072** because the warmed baseline improved more. Absolute mixed
  decode improved about 6% (29.556 → 31.342 tok/s). Prefill interference still
  requires the overlap experiment.

These are scoped synthetic tests. They do not qualify sustained coding load,
16 simultaneous cold prefills, production readiness or the public gateway.

## Rollback

Back to installed phase 1:

```bash
sudo install -o root -g root -m 0644 \
  /home/advtech/dsv41-flash-inference/configs/systemd/candidates/sglang-dsv41-8013-mr16-nodelay.service \
  /etc/systemd/system/sglang-dsv41.service
sudo systemctl daemon-reload
sudo systemctl restart sglang-dsv41.service
```

Back to mr16 (before `--min-free-slots-delay 0`):

```bash
sudo install -o root -g root -m 0644 \
  /home/advtech/dsv41-flash-inference/configs/systemd/candidates/sglang-dsv41-8013-mr16.service \
  /etc/systemd/system/sglang-dsv41.service
sudo systemctl daemon-reload
sudo systemctl restart sglang-dsv41.service
```

Full GLM rollback:

```bash
sudo systemctl stop sglang-dsv41.service
sudo systemctl start sglang-glm53.service
```

## After validation

Update `docs/RESULTS.md`, `docs/DECODE_PROFILE.md`, `HANDOFF.md`, `PROGRESS.xml`
and `INFERENCE_CONTEXT.md`; commit and push `dsv41-flash-1017`. If the cap
works, the next isolated step is the overlap pair (`--prefill-decode-interval`
and `--chunked-prefill-size`), with an explicit cold time-to-first-token
ceiling; test `--swa-prefix-tails 1024` separately afterward.
