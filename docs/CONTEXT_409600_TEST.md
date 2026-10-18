# Phase 2a — context cap 409600

Status: **prepared, not installed.** Candidate:

```text
configs/systemd/candidates/sglang-dsv41-8013-mr16-nodelay-ctx409600.service
```

Preparation summary: `evidence/phase2a-prepared-20261018T215639Z.json`.
Preflight idle snapshot: `evidence/preflight-ctx409600-20261018T215639Z/`.

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
kernel is 64% of the decode step (136 of 212 ms), while the real 400k prompts
use only ~39% of the configured width. Capping the context at **409,600**
shrinks the indexer width from 1,048,896 to about 409,920 and should cut the c16
step to about **120 ms** (projected ~1.8× throughput). The single-user
projection is smaller, about 1.2×. These are projections from the measured
profile, not results.

The cap **rejects** prompts that exceed it; it does not truncate them.
`/v1/models.max_model_len` will report 409600. The approved 400k target still
fits with about 9.6k tokens of headroom including output.

## Expected startup changes

- `/v1/models.max_model_len`: **409600** (currently 1048576).
- `DSV4 SWA sizing`: **unchanged** at `swa_tokens=59904, prefix_tails=64`,
  because SWA sizing depends on request count and prefix tails, not context.
- `DSV4 memory calculation`: full pool may move slightly upward because
  `req_to_token` and indexer scratch shrink; record the actual value.
- Startup time remains roughly 10 minutes (weights plus engram tables).
- HBM headroom should be no worse; verify before accepting.

## Client impact

Only clients pointed directly at port 8013 see the 409,600 limit. The public
`pratt-high` alias and port 8012 are unchanged, and no workstation or gateway
configuration is modified by this step. If this engine later becomes the
production endpoint, client context/compaction limits must be at or below
409,600; otherwise large prompts will be rejected.

## Apply (privileged; the user runs this)

The engine is idle at preparation time (0 running / 0 queued). Confirm again,
then:

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

## Success criteria

- `max_model_len=409600` and `swa_tokens=59904`; no unexpected pool loss.
- c16 decode at least 1.5× the scripted baseline; c1/c4 recorded.
- 16 × 400k cold all recalled; 512 warm and 2048 replay complete.
- Replay still reaches 16 running with 0 queue and 0 retracted.
- Mixed test cold prompts all recalled; ratio recorded against 0.11.
- 409k near-limit retrieval passes; tools and separated reasoning still pass.
- No OOM, runtime error, retraction or HBM regression.

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
