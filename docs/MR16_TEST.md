# 16-active 400k test

> **Superseded by phase 1 (2026-10-18):** with `--min-free-slots-delay 0`, a
> 2048-token replay reached 16 running, 0 queued, 0 retracted. The mr16 result
> below remains the historical baseline. See `docs/RESULTS.md` and
> `docs/SCHEDULER_OVERLAP_TEST.md`.

## Result (2026-10-18)

**16 requests were accepted and all completed, but 16 simultaneous active
decode was not observed.** Peak was **14 running + 1–2 queued**, with zero
retractions.

- Cold pass: 16 × ~400k prompts, all recalled; batch wall **995.5 s**;
  prefill admission was heavily staggered (queue durations up to 918 s).
- Warm pass: 399,872 device-cached tokens each; batch wall **35.0 s**;
  **290.8 aggregate / 17.05 median per-request tok/s**; peak 14 running,
  2 queued, 0 retracted; `token_usage` 0.18.
- Warm replay: **317.5 aggregate / 16.95 median tok/s**; peak 14 running,
  1 queued, 0 retracted; `token_usage` 0.16.
- Post-load HBM free: **~5.0 GB/GPU**; host available **~613 GB**.
- No OOM or runtime errors.

The likely admission limiter is the DSpark/DFlash min-free-slots policy, not
the `max_running_requests=16` cap. Forcing 16 simultaneous would require
`--min-free-slots-delay 0` and another restart. The current service is running
the 16-request candidate.

## Goal

Test whether the current DeepSeek-V4.1-Flash-1017 profile can admit **16
concurrent 400k-token requests** and sustain a warmed 16-active decode.

## Change

Only one engine flag changes from the validated 4-request profile:

```text
--max-running-requests 4  ->  16
```

Everything else is identical: image digest, TP8/EP8, `dsv4`,
`flashinfer_mxfp4`, DSPARK gamma 5 / 6 draft tokens, SWA bounded replay,
context 1,048,576, static memory 0.80, HiCache ratio 2, engram host tables,
parsers, port 8013 and served name `dsv41-flash-1017`.

Candidate:

```text
configs/systemd/candidates/sglang-dsv41-8013-mr16.service
```

The validated 4-request unit remains
`configs/systemd/candidates/sglang-dsv41-8013.service`.

## Why this is not derived from pool size alone

The GPU pool is 45,420,032 tokens. Sixteen 400k prompts are 6.4M prompt tokens,
but the measured four-request test peaked at `token_usage=0.11` — about 5.0M
pool slots, or ~1.25M slots per 400k request (~3.1× the prompt length) once
DSpark draft KV, SWA and paging overhead are included. At that measured rate,
16 requests would use roughly 20M slots, or ~44% of the pool. That is the
reason 16 is a plausible next step while 90 is not.

The run still needs live gauges: a higher-than-expected `token_usage`,
retractions, free-HBM collapse or decode slowdown is a failure even if the
requests eventually return.

## Install / restart (privileged, user-run)

```bash
sudo install -o root -g root -m 0644 \
  /home/advtech/dsv41-flash-inference/configs/systemd/candidates/sglang-dsv41-8013-mr16.service \
  /etc/systemd/system/sglang-dsv41.service
sudo systemctl daemon-reload
sudo systemctl restart sglang-dsv41.service
```

Startup is expected to take about 10 minutes again. Wait for the ready log and
`/server_info`.

## Test

```bash
cd /home/advtech/dsv41-flash-inference
python3 tests/long_c4_warm.py \
  --concurrency 16 \
  --target-tokens 400000 \
  --output-tokens 512
```

The script:

1. builds 16 distinct ~400k-token prompts and calibrates each with
   `/tokenize`;
2. sends all 16 cold requests concurrently, requiring all marker codes;
3. repeats the same 16 prompts with 512 forced output tokens;
4. samples `/metrics` every 0.5 s during the warm pass.

The cold pass will be **staggered by prefill admission**, not 16 simultaneous
prefills. The warm pass is the actual 16-active decode test.

## Success criteria

- All 16 cold prompts return their three marker codes.
- Warm pass: peak `num_running_reqs = 16`.
- Warm pass: peak `num_queue_reqs = 0` and `num_retracted_reqs = 0`.
- No OOM, CUDA error, traceback or watchdog event.
- `token_usage` remains below 1.0 with headroom.
- HBM free remains positive; record the post-load minimum.
- Report aggregate and median per-request tok/s; do not compare directly to the
  four-request rate without noting batch size.

## Rollback

Restore the validated 4-request unit:

```bash
sudo install -o root -g root -m 0644 \
  /home/advtech/dsv41-flash-inference/configs/systemd/candidates/sglang-dsv41-8013.service \
  /etc/systemd/system/sglang-dsv41.service
sudo systemctl daemon-reload
sudo systemctl restart sglang-dsv41.service
```

Or stop the test and restart GLM:

```bash
sudo systemctl stop sglang-dsv41.service
sudo systemctl start sglang-glm53.service
```
