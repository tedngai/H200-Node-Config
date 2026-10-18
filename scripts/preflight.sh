#!/usr/bin/env bash
# Read-only preflight and pre-change snapshot. No sudo, no service changes.
set -euo pipefail

ROOT="/home/advtech/dsv41-flash-inference"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${ROOT}/evidence/preflight-${TS}"
mkdir -p "${OUT}"

echo "preflight ${TS} -> ${OUT}"

# Live GLM state, used for rollback verification.
curl -fsS --max-time 10 http://127.0.0.1:8012/server_info \
  -o "${OUT}/glm-server-info.json"
curl -fsS --max-time 10 http://127.0.0.1:8012/metrics \
  | grep -E 'sglang:(num_requests_running|num_queue_reqs|num_retracted_reqs|token_usage)' \
  > "${OUT}/glm-metrics.txt" || true
systemctl cat sglang-glm53.service > "${OUT}/glm-unit.txt"
docker inspect sglang-glm53 > "${OUT}/glm-container-inspect.json"
docker image inspect \
  lmsysorg/sglang@sha256:9e148f5ac788e856a06166bd6347a831831eb9fcfab4d1770874823a7c29a1a1 \
  > "${OUT}/glm-image-inspect.json" 2>/dev/null || true

# Host capacity.
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv > "${OUT}/gpu.csv"
free -b > "${OUT}/free.txt"
df -B1 /raid / > "${OUT}/df.txt"
grep -E 'HugePages|AnonHugePages|Hugetlb' /proc/meminfo > "${OUT}/hugepages.txt"
ss -ltnp > "${OUT}/listen.txt" 2>/dev/null || true

# Test image identity.
docker image inspect lmsysorg/sglang:dev-dsv41 \
  > "${OUT}/test-image-inspect.json"
docker image inspect lmsysorg/sglang:dev-dsv41 \
  --format '{{.Id}} {{index .RepoDigests 0}} {{.Size}} {{.Created}}' \
  > "${OUT}/test-image-identity.txt"

# Checkpoint cache footprint.
du -sb \
  /raid/advtech/cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017 \
  > "${OUT}/checkpoint-cache-bytes.txt" 2>/dev/null || true

# Candidate unit syntax (does not launch anything).
systemd-analyze verify \
  "${ROOT}/configs/systemd/candidates/sglang-dsv41-8013.service" \
  > "${OUT}/systemd-verify.txt" 2>&1 || true

# Confirm the pinned image accepts every candidate flag we rely on.
HELP="${OUT}/sglang-serve-help.txt"
docker run --rm lmsysorg/sglang:dev-dsv41 sglang serve --help > "${HELP}" 2>&1
for flag in \
  --ep-size \
  --max-running-requests \
  --attention-backend \
  --moe-runner-backend \
  --enable-decoder-swa-bounded-replay \
  --cuda-graph-max-bs-decode \
  --speculative-algorithm \
  --speculative-dspark-block-size \
  --enable-hierarchical-cache \
  --hicache-io-backend \
  --hicache-mem-layout \
  --hicache-ratio \
  --hicache-write-policy \
  --reasoning-parser \
  --tool-call-parser \
  --enable-cache-report \
  --enable-request-time-stats-logging
do
  if grep -q -- "${flag}" "${HELP}"; then
    echo "OK ${flag}"
  else
    echo "MISSING ${flag}"
  fi
done | tee "${OUT}/flag-check.txt"

# Confirm port 8013 is free.
if ss -ltn | awk '{print $4}' | grep -qE '(:|\.)8013$'; then
  echo "PORT 8013 IN USE" | tee "${OUT}/port-8013.txt"
else
  echo "PORT 8013 FREE" | tee "${OUT}/port-8013.txt"
fi

# Summarize the live idle state.
python3 - "${OUT}" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
info = json.loads((out / "glm-server-info.json").read_text())
summary = {
    "utc": out.name.removeprefix("preflight-"),
    "glm": {
        "model_path": info.get("model_path"),
        "served_model_name": info.get("served_model_name"),
        "context_length": info.get("context_length"),
        "tp_size": info.get("tp_size"),
        "dp_size": info.get("dp_size"),
        "max_running_requests": info.get("max_running_requests"),
        "enable_hierarchical_cache": info.get("enable_hierarchical_cache"),
    },
    "test_image": (out / "test-image-identity.txt").read_text().strip(),
    "checkpoint_cache_bytes": (out / "checkpoint-cache-bytes.txt").read_text().strip(),
    "port_8013": (out / "port-8013.txt").read_text().strip(),
}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

echo "done: ${OUT}"
