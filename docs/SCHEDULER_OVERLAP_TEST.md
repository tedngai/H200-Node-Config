# Scheduler and prefill/decode overlap tests

Status: **phase 2a installed and measured; overlap prepared, not installed.**
The installed unit is
`configs/systemd/candidates/sglang-dsv41-8013-mr16-nodelay-ctx409600.service`
(`--max-running-requests 16 --min-free-slots-delay 0 --context-length 409600`).
Phase 1 proved 16 simultaneous active warmed 400k decodes in a 2048-token
replay and produced the first mixed baseline. Phase 2a then raised c16 decode
1.96× and changed the mixed baseline. The overlap candidate remains a later
step and still needs a privileged restart.

## Phase 1 measured result (2026-10-18)

- 16 × 400k cold pass: 991.414 s, all recalled; admission remained staggered
  (queue 0.01–913.1 s), so this is not 16 simultaneous cold prefill.
- 512-token warm pass: 35.173 s, 290.367 aggregate / 16.377 median tok/s;
  peak **15** running, 0 queue, 0 retracted. Server-side
  tokenization/ingress staggered the 16 arrivals, so this pass does not prove
  16 active.
- 2048-token warm replay: 96.254 s, 372.979 aggregate / 22.878 median tok/s;
  peak **16** running, 0 queue, 0 retracted; 16 active for ~56 s. This is the
  16-active proof.
- Mixed baseline (4 warmed 2048-token decodes, then +4 fresh 400k prefills):
  baseline 268.669 aggregate / 67.203 median tok/s; mixed 29.556 / 7.388;
  ratio **0.11**; four cold prompts recalled, all completing at ~247.4–247.6 s;
  peak running 7, queue 3, retracted 0.
- Post-ready runtime error scan: 0 matches; smoke and tools passed.
- Evidence: `evidence/long-400000-c16-warm-20261018T204521Z.json`,
  `evidence/replay-c16-warm-20261018T205717Z.json`,
  `evidence/mixed-prefill-decode-w4-c4-20261018T210728Z.json`,
  `evidence/phase1-nodelay-summary-20261018T210803Z.json`.

## Phase 2a measured result (2026-10-18)

Phase 2a installed only `--context-length 409600`. It raised c16 aggregate
decode from 336.4 to **660.5 tok/s (1.963×)**, c4 to 394.4 (1.538×), and c1 to
174.2 (1.249×). The 16-active replay reached **704.5 tok/s** (phase 1: 373.0)
with peak 16 running, 0 queue, 0 retracted. Cold 16 × 400k recalled all codes
in 978.7 s; prefill remained one sequence per batch, not a 16-sequence batch.

The mixed comparison changed materially: warmed baseline **434.154**, mixed
**31.342**, ratio **0.072** (phase 1: ratio 0.11, mixed 29.556). Absolute mixed
decode improved ~6%, but the warmed baseline improved more, so the ratio fell.
Post-load free HBM was 7.0–7.5 GiB/GPU (minimum 7.0); runtime errors were 0.

Evidence: `evidence/phase2a-summary-20261018T225300Z.json`,
`evidence/decode-throughput-ctx409600-20261018T221334Z.json`,
`evidence/replay-c16-warm-20261018T224045Z.json`,
`evidence/mixed-prefill-decode-w4-c4-20261018T225014Z.json`.

The overlap test must be compared against the **phase 2a** mixed baseline, not
the old phase 1 numbers.

Overlap acceptance: absolute mixed aggregate should exceed the phase 2a
**31.342 tok/s**, all cold prompts must recall, and cold wall/time-to-first-token
must stay within a ceiling agreed before install. Report the ratio against
**0.072** as well; the ratio alone can fall if the warmed baseline gets faster.

## Why these tests

The 45M-token GPU pool is not the reason only 14 requests were observed in
mr16. The DFlash min-free-slots policy auto-enables at
`max_running_requests >= 8` and for 16 requests resolves to a threshold of 3.
It admits a batch, waits for three slots to free, then admits the remainder.
Phase 1 removed that policy. The 512-token phase 1 pass still peaked at 15
because server-side tokenization/ingress staggered the 16 arrivals; a
2048-token replay then reached 16. Phase 2 tests whether the official H200
agentic prefill/decode overlap settings improve decode progress while long
prefills are active.

## Phase 1 — remove the admission delay

Candidate:

```text
configs/systemd/candidates/sglang-dsv41-8013-mr16-nodelay.service
```

Only change from the current mr16 unit:

```text
--min-free-slots-delay 0
```

`0` is the explicit disable value (values `<= 1` disable the delayer). All
other flags, including `--max-running-requests 16`, HiCache, DSpark and the
pinned image, are unchanged.

Install/restart (privileged; user runs these):

```bash
sudo install -o root -g root -m 0644 \
  /home/advtech/dsv41-flash-inference/configs/systemd/candidates/sglang-dsv41-8013-mr16-nodelay.service \
  /etc/systemd/system/sglang-dsv41.service
sudo systemctl daemon-reload
sudo systemctl restart sglang-dsv41.service
```

Wait for readiness, then verify:

```bash
curl -s http://127.0.0.1:8013/server_info | python3 -c \
  'import json,sys; d=json.load(sys.stdin); print({k:d.get(k) for k in ("max_running_requests","min_free_slots_delay","enable_hierarchical_cache")})'
```

### Phase 1 capacity test

```bash
cd /home/advtech/dsv41-flash-inference
python3 tests/long_c4_warm.py \
  --concurrency 16 --target-tokens 400000 --output-tokens 512

# 16-active proof: replay the same 16 prefixes with longer output
python3 tests/replay_c16_warm.py --output-tokens 2048
```

Success criteria:

- all 16 cold prompts recall their three marker codes;
- the 2048-token warm replay reaches peak `num_running_reqs` **16**; the
  512-token pass alone may peak at 15 because server-side tokenization
  staggers admission (measured phase 1 behavior);
- peak `num_queue_reqs` is 0 or only a transient admission queue;
- peak `num_retracted_reqs` is 0;
- no OOM, CUDA error, watchdog event or runtime traceback;
- record aggregate and median tok/s against the mr16 baseline
  (first warm pass 290.8 aggregate / 17.05 median; replay 317.5 / 16.95).

If peak running remains 14, inspect the resolved server args and scheduler
logs; do not assume the flag took effect.

### Phase 1 mixed prefill/decode baseline

This pass measures warmed decode with and without four fresh 400k prefills.
It uses the same 4 warmed prefixes for both passes and fresh cold prompts for
the mixed pass.

```bash
python3 tests/mixed_prefill_decode.py \
  --warm-concurrency 4 --cold-concurrency 4 \
  --target-tokens 400000 --output-tokens 2048
```

Record:

- baseline and mixed aggregate/median warmed decode tok/s;
- `mixed.aggregate_ratio` and `mixed.median_ratio`;
- `warm_running_seen_before_cold`;
- mixed peak running/queue/retracted/token usage;
- all cold marker codes recalled;
- cold batch wall time.

This is the comparison point for phase 2.

## Phase 2b — scheduler overlap (interval + chunk, no tails)

Candidate:

```text
configs/systemd/candidates/sglang-dsv41-8013-mr16-nodelay-overlap.service
```

Changes on top of the installed phase 2a unit (which retains
`--context-length 409600`):

```text
--prefill-decode-interval 16
--chunked-prefill-size 4096
```

`--swa-prefix-tails` stays at the default 64, so the SWA and full-KV pools
should remain 59,904 / 44,957,952. These are the published scheduler settings,
intended to keep decode progressing while long prefixes are queued. The tail
reservation is deliberately moved to phase 2c so it can be attributed.

Install/restart:

```bash
sudo install -o root -g root -m 0644 \
  /home/advtech/dsv41-flash-inference/configs/systemd/candidates/sglang-dsv41-8013-mr16-nodelay-overlap.service \
  /etc/systemd/system/sglang-dsv41.service
sudo systemctl daemon-reload
sudo systemctl restart sglang-dsv41.service
```

Verify the resolved args:

```bash
curl -s http://127.0.0.1:8013/server_info | python3 -c \
  'import json,sys; d=json.load(sys.stdin); print({k:d.get(k) for k in ("max_running_requests","min_free_slots_delay","context_length","prefill_decode_interval","chunked_prefill_size","swa_prefix_tails")})'
```

### Phase 2b tests

Run c16 decode, then the 16 × 400k capacity test, 2048-token replay and mixed
test:

```bash
python3 tests/decode_throughput.py \
  --concurrency 16 --output-tokens 1024 --rounds 1 \
  --label decode-throughput-overlap

python3 tests/long_c4_warm.py \
  --concurrency 16 --target-tokens 400000 --output-tokens 512

python3 tests/replay_c16_warm.py --output-tokens 2048

python3 tests/mixed_prefill_decode.py \
  --warm-concurrency 4 --cold-concurrency 4 \
  --target-tokens 400000 --output-tokens 2048
```

Primary overlap success criterion: absolute mixed warmed decode exceeds the
phase 2a **31.342 tok/s** without an increase in OOM, retractions or cold-recall
failures, and cold wall/time-to-first-token stays within the pre-agreed
ceiling. Report the mixed/baseline ratio against phase 2a's **0.072**; do not
use the old 0.11 comparison. A small absolute improvement is still useful; do
not expect PD-style isolation from a scheduler interval alone.

Secondary criteria:

- cold 16 × 400k prefill still completes and recalls;
- the 2048-token replay still reaches 16 running;
- token usage and HBM headroom remain safe;
- no watchdog or runtime errors.

## Phase 2c — SWA prefix tails only

Candidate:

```text
configs/systemd/candidates/sglang-dsv41-8013-mr16-nodelay-tails1024.service
```

It adds only `--swa-prefix-tails 1024` to the installed phase 2a unit. Each tail
is one 128-token sliding window plus one 256-token page (384 slots). Expected
but **unverified**: `swa_tokens` around 420,000 instead of 59,904, roughly
+8 GB/GPU, and a full-KV pool around 40M instead of 44.96M. Confirm the actual
startup lines.

This is a multi-session prefix-reuse experiment, not a 16-active decode test.
Install it separately, verify `swa_prefix_tails=1024` and `swa_tokens`, then
measure cache reuse across long gaps. Do not combine it with phase 2b unless
both pass independently and the combination is explicitly tested.

## Rollback

Phase 2a is installed; the overlap unit is not. Rollback from overlap to phase
2a is below.

```bash
sudo install -o root -g root -m 0644 \
  /home/advtech/dsv41-flash-inference/configs/systemd/candidates/sglang-dsv41-8013-mr16-nodelay-ctx409600.service \
  /etc/systemd/system/sglang-dsv41.service
sudo systemctl daemon-reload
sudo systemctl restart sglang-dsv41.service
```

Return to the earlier phase 1 unit:

```bash
sudo install -o root -g root -m 0644 \
  /home/advtech/dsv41-flash-inference/configs/systemd/candidates/sglang-dsv41-8013-mr16-nodelay.service \
  /etc/systemd/system/sglang-dsv41.service
sudo systemctl daemon-reload
sudo systemctl restart sglang-dsv41.service
```

Return to the earlier mr16 unit:

```bash
sudo install -o root -g root -m 0644 \
  /home/advtech/dsv41-flash-inference/configs/systemd/candidates/sglang-dsv41-8013-mr16.service \
  /etc/systemd/system/sglang-dsv41.service
sudo systemctl daemon-reload
sudo systemctl restart sglang-dsv41.service
```

The full GLM-5.3 rollback remains:

```bash
sudo systemctl stop sglang-dsv41.service
sudo systemctl start sglang-glm53.service
```
