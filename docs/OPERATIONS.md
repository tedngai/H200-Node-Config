# Operations

**Current state (2026-10-19):** installed and running as
`sglang-dsv41.service` on **8012 / pratt-high** with the phase 2b
`mr16-nodelay-ctx409600-overlap2b` unit. The 8012 phase 2a candidate is the
rollback that preserves the endpoint; phase 2c tails is a separate test.

## Production boot enablement (prepared, not installed)

The tested 2b unit is **not enabled** and has `Restart=no`. The production
candidate is:

```text
configs/systemd/candidates/sglang-dsv41-8012-pratt-high-mr16-nodelay-ctx409600-overlap2b-production.service
```

It preserves every tested 2b flag and adds only operational hardening:

- `Restart=on-failure` with `RestartSec=30`;
- `StartLimitIntervalSec=1800` / `StartLimitBurst=3`, so a bad 510 GB load
  cannot loop forever;
- `RequiresMountsFor=/raid/advtech/cache/huggingface`, so boot fails rather
  than starting with a missing model/cache mount;
- the existing `Requires=docker.service nvidia-fabricmanager.service` and
  `Conflicts=sglang-glm53.service sglang-glm53-flash.service` are retained.

### Install and enable (user-run; needs sudo)

First confirm the engine is idle (`0 running / 0 queued`) and decide the
maintenance window. Startup takes about **9–10 minutes**. Then:

```bash
sudo install -o root -g root -m 0644 \
  /home/advtech/dsv41-flash-inference/configs/systemd/candidates/sglang-dsv41-8012-pratt-high-mr16-nodelay-ctx409600-overlap2b-production.service \
  /etc/systemd/system/sglang-dsv41.service
sudo systemctl daemon-reload

# Keep the GLM units installed for rollback, but stop them and their boot autostart.
sudo systemctl disable --now sglang-glm53.service sglang-glm53-flash.service

# Start DSV4.1 at boot.
sudo systemctl enable sglang-dsv41.service
sudo systemctl restart sglang-dsv41.service
```

Do not use both engines at once; both need all eight GPUs. Disabling the GLM
units does **not** delete or modify them. Rollback to GLM is
`systemctl disable --now sglang-dsv41.service`, then
`systemctl enable --now sglang-glm53.service`.

### Verify before rebooting

```bash
systemctl is-enabled sglang-dsv41.service
systemctl is-enabled sglang-glm53.service sglang-glm53-flash.service
systemctl show sglang-dsv41.service \
  -p Restart -p RestartUSec -p StartLimitBurst -p StartLimitIntervalUSec
curl -s http://127.0.0.1:8012/v1/models | python3 -c \
  'import json,sys; print([(m["id"],m.get("max_model_len")) for m in json.load(sys.stdin)["data"]])'
systemctl show sglang-dsv41.service -p NRestarts -p ActiveState -p SubState
```

Expected: DSV4 `enabled`; both GLM services `disabled`; `Restart=on-failure`;
model `pratt-high`, `max_model_len=409600`; 0 restarts.

### Reboot proof and post-boot check

Only an actual reboot proves boot auto-start. After a scheduled reboot:

```bash
systemctl is-active sglang-dsv41.service
systemctl is-enabled sglang-dsv41.service
docker ps --filter name=sglang-dsv41
curl -s http://127.0.0.1:8012/v1/models
nvidia-smi
systemctl show sglang-dsv41.service -p NRestarts -p ActiveState -p SubState
```

Expect one DSV4 container, eight GPUs, `pratt-high`, and no competing GLM
container. The first request may wait for the ~9–10 minute model load. If the
unit enters `failed`, inspect `journalctl -u sglang-dsv41.service` and
`docker logs sglang-dsv41`; fix the cause before allowing another restart.

This is operational boot/restart hardening, not a claim of sustained
production load or gateway qualification. The single node has no warm standby;
a reboot or crash recovery is a roughly 10-minute outage.

## Manual 2b install / start (not boot-enabled)

Use this only for a temporary 2b test or an endpoint-preserving rollback; for
boot-enabled production use the section above.

The experiment account has Docker access but no passwordless sudo. Run these
personally:

```bash
# Stop the validated GLM-5.3 service first; it owns all eight GPUs.
sudo systemctl stop sglang-glm53.service

# Install the isolated DSV4.1 phase 2b test unit (port 8012, pratt-high).
sudo install -o root -g root -m 0644 \
  /home/advtech/dsv41-flash-inference/configs/systemd/candidates/sglang-dsv41-8012-pratt-high-mr16-nodelay-ctx409600-overlap2b.service \
  /etc/systemd/system/sglang-dsv41.service
sudo systemctl daemon-reload
sudo systemctl start sglang-dsv41.service
```

The unit is not enabled at boot and has `Restart=no`.

## Startup observation

```bash
systemctl status sglang-dsv41.service --no-pager -l
docker logs -f sglang-dsv41
```

Expected startup phases:

1. Container and CUDA/NCCL initialization.
2. Checkpoint load (~510 GB). The engram host-table path may drop the
   checkpoint page cache (`posix_fadvise DONTNEED`) before pre-faulting the
   private shard; this is expected and makes the next restart colder.
3. Engram host table allocation: two tables, ~25.4 GB per rank in `private`
   layout, pinned with `cudaHostRegister`, backed by anonymous THP
   (`madvise`).
4. KV pool and HiCache host pool allocation (`--hicache-ratio 2`).
5. CUDA graph capture (`--cuda-graph-max-bs-decode 64`).
6. Ready log. Look for `The server is fired up and ready to go!` or the
   equivalent readiness line.

Startup is not complete until `/server_info` responds and the ready log appears.
A container that is still loading weights is not a failed start.

### Measured startup (2026-10-18)

- Container start 14:42:34 UTC; ready 14:52:39 UTC; **585 s**.
- Engram host tables: 16 mappings × 24,171 MiB, **386.7 GiB total**, 100% huge
  pages, pinned.
- GPU KV pool: **45,420,032 tokens**; HiCache host pool **90,840,064 tokens**.
- HBM free: ~22.3 GB/GPU at startup, ~7.3 GB/GPU after the first long request.
- Container RSS after load: **1.42 TiB**; host available ~665 GB.
- Non-fatal startup warnings: NUMA affinity not set (no `--cap-add SYS_NICE`)
  and a FABRIC-handle export fallback to POSIX fd transport during CUDA graph
  capture.

## Health and capacity checks

```bash
curl -s http://127.0.0.1:8013/v1/models | python3 -m json.tool
curl -s http://127.0.0.1:8013/server_info | python3 -m json.tool
curl -s http://127.0.0.1:8013/metrics | grep -E \
  'num_requests_running|num_queue_reqs|num_retracted_reqs|token_usage|cache_hit'
```

Record from `/server_info`:

- `context_length`, `tp_size`, `ep_size`, `max_running_requests`;
- `attention_backend`, `moe_runner_backend`, `speculative_algorithm`;
- `enable_hierarchical_cache`, `hicache_ratio`, `hicache_io_backend`,
  `hicache_mem_layout`, `hicache_write_policy`;
- `max_total_tokens`, `max_prefill_tokens`;
- resolved reasoning/tool parsers.

Host and GPU:

```bash
free -b
grep -E 'HugePages|AnonHugePages|Hugetlb' /proc/meminfo
nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free --format=csv
docker stats --no-stream sglang-dsv41
```

Engram host-table log lines to look for:

```text
engram host table layers.1.engram.embed: layout=private, ... resident, ... in huge pages ...
engram host table layers.14.engram.embed: layout=private, ...
```

If the log says `No huge pages`, the table is still functional but lookups are
expected to be much slower. Record it; do not silently accept it as equivalent.

## Running the scoped probes

```bash
cd /home/advtech/dsv41-flash-inference
python3 tests/openai_probe.py --mode smoke
python3 tests/openai_probe.py --mode tools
python3 tests/openai_probe.py --mode throughput --concurrency 1 --output-tokens 256 --rounds 3
python3 tests/openai_probe.py --mode throughput --concurrency 4 --output-tokens 256 --rounds 3
python3 tests/openai_probe.py --mode long --prompt-tokens 100000
```

Run one load test at a time. The script writes a timestamped report under
`evidence/`. Do not run concurrent benchmarks from multiple shells.

## Stop / rollback

Return from phase 2b to the 8012 / pratt-high phase 2a unit:

```bash
sudo install -o root -g root -m 0644 \
  /home/advtech/dsv41-flash-inference/configs/systemd/candidates/sglang-dsv41-8012-pratt-high-mr16-nodelay-ctx409600.service \
  /etc/systemd/system/sglang-dsv41.service
sudo systemctl daemon-reload
sudo systemctl restart sglang-dsv41.service
```

Full GLM rollback:

```bash
sudo systemctl stop sglang-dsv41.service
sudo systemctl start sglang-glm53.service
```

The GLM unit was not modified. Verify the rollback:

```bash
curl -s http://127.0.0.1:8012/server_info | python3 -c \
  'import json,sys; d=json.load(sys.stdin); print(d["model_path"], d["context_length"], d["tp_size"], d["dp_size"], d["max_running_requests"], d["enable_hierarchical_cache"])'
```

Expected: `zai-org/GLM-5.3`, context 400000, TP8, DP4, max running 4, HiCache
enabled.

To remove the test unit entirely after the experiment:

```bash
sudo systemctl disable --now sglang-dsv41.service
sudo rm -f /etc/systemd/system/sglang-dsv41.service
sudo systemctl daemon-reload
```

## Known operating limits

- **Do not run both engines at once.** The 510 GB DSV4.1 checkpoint and the GLM
  checkpoint both need all eight GPUs.
- **Do not infer capacity from `--cuda-graph-max-bs-decode 64`.** The effective
  admission cap is `--max-running-requests 16` in the current phase 2b
  candidate, the context cap is 409,600 tokens, and the measured SWA cap is
  51,712 tokens.
- **HiCache is not RAM decoding.** It retains evicted prefixes; it does not
  extend the active GPU pool or migrate a request between workers.
- **The engram host table is pinned host memory.** It consumes **386.7 GiB**
  measured host RAM and is not part of the GPU KV pool.
- **A cold restart is expensive.** The engram path may drop the checkpoint page
  cache; the next load can be slower than the previous one.
