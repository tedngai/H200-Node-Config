# Scheduler and prefill/decode overlap tests

Status: **phase 1 installed and measured; phase 2a prepared, not installed.**
The installed unit is
`configs/systemd/candidates/sglang-dsv41-8013-mr16-nodelay.service`
(`--max-running-requests 16 --min-free-slots-delay 0`). Phase 1 proved 16
simultaneous active warmed 400k decodes in a 2048-token replay, and produced the
mixed prefill/decode baseline. Before this overlap test, phase 2a adds only
`--context-length 409600` to address the decode-side indexer cost; see
`docs/CONTEXT_409600_TEST.md`. This overlap candidate remains a later step and
still needs a privileged restart.

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

Phase 2 must improve the **0.11** mixed/baseline ratio without losing cold
recall, 16-active capacity, or memory safety.

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

## Phase 2 — official H200 agentic overlap package

Candidate:

```text
configs/systemd/candidates/sglang-dsv41-8013-mr16-nodelay-overlap.service
```

Changes on top of phase 1:

```text
--prefill-decode-interval 16
--chunked-prefill-size 4096
--swa-prefix-tails 1024
```

These are the published H200 Flash-1017 agentic settings, not arbitrary
values. They are intended to keep decode progressing while long prefixes are
queued and to avoid SWA prefix-tail exhaustion at higher concurrency.

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
  'import json,sys; d=json.load(sys.stdin); print({k:d.get(k) for k in ("max_running_requests","min_free_slots_delay","prefill_decode_interval","chunked_prefill_size","swa_prefix_tails")})'
```

### Phase 2 tests

Run the same 16 × 400k capacity test, 2048-token replay and mixed test:

```bash
python3 tests/long_c4_warm.py \
  --concurrency 16 --target-tokens 400000 --output-tokens 512

python3 tests/replay_c16_warm.py --output-tokens 2048

python3 tests/mixed_prefill_decode.py \
  --warm-concurrency 4 --cold-concurrency 4 \
  --target-tokens 400000 --output-tokens 2048
```

Primary overlap success criterion: the mixed/baseline warmed decode ratio
improves versus phase 1's **0.11**, with no increase in OOM, retractions or
cold-recall failures. A small improvement is still useful; do not expect
PD-style isolation from a scheduler interval alone.

Secondary criteria:

- cold 16 × 400k prefill still completes and recalls;
- the 2048-token replay still reaches 16 running;
- token usage and HBM headroom remain safe;
- no watchdog or runtime errors.

## Rollback

Before phase 2 is installed, the exact phase 1 rollback is already on disk.
After phase 2, return to phase 1:

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
