#!/usr/bin/env bash
# Wait for the DSV4.1 test server to become ready or fail. Read-only.
set -u
LOG="/tmp/opencode/dsv41-startup-monitor.log"
: > "${LOG}"
for i in $(seq 1 240); do
  if curl -fsS --max-time 3 http://127.0.0.1:8013/server_info > /tmp/opencode/dsv41-server-info.json 2>/dev/null; then
    echo "READY after $((i*15))s" | tee -a "${LOG}"
    exit 0
  fi
  if ! docker inspect -f '{{.State.Running}}' sglang-dsv41 2>/dev/null | grep -q true; then
    echo "CONTAINER EXITED after $((i*15))s" | tee -a "${LOG}"
    docker logs --tail 200 sglang-dsv41 >> "${LOG}" 2>&1
    exit 1
  fi
  {
    echo "--- t=$((i*15))s $(date -u +%H:%M:%SZ) ---"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; echo
    free -g | awk '/Mem:/{print "ram used="$3"G avail="$7"G"}'
    docker logs --tail 4 sglang-dsv41 2>&1 | tail -4
  } >> "${LOG}"
  sleep 15
done
echo "TIMEOUT after 3600s" | tee -a "${LOG}"
exit 2
