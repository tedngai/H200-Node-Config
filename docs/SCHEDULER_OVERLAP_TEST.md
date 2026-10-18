# Scheduler and prefill/decode overlap tests

Status: **phase 1 prepared, not installed**. The live service is still the
validated `max-running-requests=16` profile with the default DFlash
min-free-slots policy (peak observed 14 running + 1–2 queued).

## Why these tests

The 45M-token GPU pool is not the reason only 14 requests were observed. The
DFlash min-free-slots policy auto-enables at `max_running_requests >= 8` and
for 16 requests resolves to a threshold of 3. It admits a batch, waits for
three slots to free, then admits the remainder. The first test removes that
policy; the second tests whether the official H200 agentic prefill/decode
overlap settings improve decode progress while long prefills are active.

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
```

Success criteria:

- all 16 cold prompts recall their three marker codes;
- warm pass peak `num_running_reqs` is **16** (not 14);
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

Run the same 16 × 400k capacity test and mixed test:

```bash
python3 tests/long_c4_warm.py \
  --concurrency 16 --target-tokens 400000 --output-tokens 512

python3 tests/mixed_prefill_decode.py \
  --warm-concurrency 4 --cold-concurrency 4 \
  --target-tokens 400000 --output-tokens 2048
```

Primary overlap success criterion: the mixed/baseline warmed decode ratio
improves versus phase 1, with no increase in OOM, retractions or cold-recall
failures. A small improvement is still useful; do not expect PD-style
isolation from a scheduler interval alone.

Secondary criteria:

- cold 16 × 400k prefill still completes and recalls;
- peak running remains 16 in the capacity test;
- token usage and HBM headroom remain safe;
- no watchdog or runtime errors.

## Rollback

Return to the validated mr16 unit:

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
