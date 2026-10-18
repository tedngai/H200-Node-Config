# Operations

## Install / start (privileged, user-run)

The experiment account has Docker access but no passwordless sudo. Run these
personally:

```bash
# Stop the validated GLM-5.3 service first; it owns all eight GPUs.
sudo systemctl stop sglang-glm53.service

# Install the isolated DSV4.1 test unit (port 8013).
sudo install -o root -g root -m 0644 \
  /home/advtech/dsv41-flash-inference/configs/systemd/candidates/sglang-dsv41-8013.service \
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
  admission cap is `--max-running-requests 4` in this candidate.
- **HiCache is not RAM decoding.** It retains evicted prefixes; it does not
  extend the active GPU pool or migrate a request between workers.
- **The engram host table is pinned host memory.** It reduces available host
  RAM by ~203 GB and is not part of the GPU KV pool.
- **A cold restart is expensive.** The engram path may drop the checkpoint page
  cache; the next load can be slower than the previous one.
