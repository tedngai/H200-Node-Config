#!/usr/bin/env bash
# Capture startup state after the DSV4.1 server is ready. Read-only.
set -euo pipefail

ROOT="/home/advtech/dsv41-flash-inference"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${ROOT}/evidence/startup-${TS}"
mkdir -p "${OUT}"

curl -fsS --max-time 10 http://127.0.0.1:8013/server_info \
  -o "${OUT}/server-info.json"
curl -fsS --max-time 10 http://127.0.0.1:8013/metrics \
  > "${OUT}/metrics.txt"
docker logs --tail 10000 sglang-dsv41 > "${OUT}/container.log" 2>&1 || true
docker inspect sglang-dsv41 > "${OUT}/container-inspect.json"
docker stats --no-stream sglang-dsv41 > "${OUT}/docker-stats.txt" 2>&1 || true
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv > "${OUT}/gpu.csv"
free -b > "${OUT}/free.txt"
grep -E 'HugePages|AnonHugePages|Hugetlb' /proc/meminfo > "${OUT}/hugepages.txt"
systemctl status sglang-dsv41 --no-pager -l > "${OUT}/systemctl-status.txt" 2>&1 || true

grep -Ei 'engram host table|engram.*layout|attention backend|moe runner|fp8 gemm|dspark|hierarchical|hicache|max running|context length|memory pool|KV cache|ready|error|traceback|oom|out of memory|watchdog' \
  "${OUT}/container.log" > "${OUT}/startup-highlights.txt" || true

python3 - "${OUT}" <<'PY'
import json, pathlib, re, sys
out = pathlib.Path(sys.argv[1])
info = json.loads((out / "server-info.json").read_text())
keys = [
    "model_path", "served_model_name", "context_length", "tp_size", "ep_size",
    "dp_size", "max_running_requests", "max_queued_requests",
    "mem_fraction_static", "max_total_tokens", "max_prefill_tokens",
    "attention_backend", "moe_runner_backend", "fp8_gemm_backend",
    "speculative_algorithm", "speculative_dspark_block_size",
    "speculative_num_draft_tokens", "enable_hierarchical_cache",
    "hicache_ratio", "hicache_size", "hicache_io_backend",
    "hicache_mem_layout", "hicache_write_policy", "reasoning_parser",
    "tool_call_parser", "enable_decoder_swa_bounded_replay",
    "cuda_graph_max_bs_decode", "watchdog_timeout",
]
summary = {"utc": out.name.removeprefix("startup-")}
for k in keys:
    if k in info:
        summary[k] = info[k]
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

echo "startup evidence: ${OUT}"
