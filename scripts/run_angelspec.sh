#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 2 || ( $1 != check && $1 != run ) || ( $2 != dfly && $2 != dspark ) ]]; then
  echo "Usage: bash $0 {check|run} {dfly|dspark}" >&2
  exit 2
fi

tag=local/dsv41-angelspec:82308b9d9
image=$(docker image inspect --format '{{.Id}}' "$tag")
test "$(docker image inspect --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$image")" \
  = 82308b9d9e1685aa9404272cc499c4c195610913
cache_root=${DSV41_CACHE_ROOT:-/raid/user/cache}
cache=$cache_root/huggingface
draft=$cache_root/sglang/angelspec/DS-V4.1-Flash-DFly2-B6/d08e55a6897b1974307ea9771038838de5986e66
target_rev=1bae0e1088993b9b82de08f59de5be5a34e42170
test -r "$cache/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/$target_rev/config.json"

mounts=(--volume "$cache:/root/.cache/huggingface:ro")
spec=(--speculative-algorithm DSPARK --speculative-dspark-block-size 5)
if [[ $2 == dfly ]]; then
  test -s "$draft/config.json"
  test -s "$draft/model.safetensors.index.json"
  for shard in "$draft"/model-0000{1,2,3,4}-of-00004.safetensors; do
    test -s "$shard"
  done
  mounts+=(--volume "$draft:/opt/dfly2-draft:ro")
  spec=(--speculative-algorithm DFLY --speculative-draft-model-path /opt/dfly2-draft)
fi

args=(
  --model-path deepseek-ai/DeepSeek-V4.1-Flash-1017 --revision "$target_rev"
  --served-model-name dsv41-angelspec --trust-remote-code
  --tp 8 --ep-size 8 --max-running-requests 16 --min-free-slots-delay 0
  --context-length 409600 --prefill-decode-interval 16 --chunked-prefill-size 4096
  --mem-fraction-static 0.8 --attention-backend dsv4 --dsv4-attn-backend flashmla
  --moe-runner-backend flashinfer_mxfp4 --enable-decoder-swa-bounded-replay
  --cuda-graph-max-bs-decode 64 --cuda-graph-backend-prefill disabled
  --engram-config '{"placement":"cpu","replication":"none"}'
  --enable-hierarchical-cache --hicache-io-backend direct
  --hicache-mem-layout page_first_direct --hicache-ratio 2 --hicache-write-policy write_back
  --reasoning-parser auto --tool-call-parser auto --watchdog-timeout 3600
  --enable-metrics --enable-request-time-stats-logging --enable-cache-report
  --host 127.0.0.1 --port 8013
  "${spec[@]}"
)
common=(
  --rm --pull never --env HF_HUB_OFFLINE=1
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "${mounts[@]}"
)

if [[ $1 == check ]]; then
  # Resolve the same flags using real H200 detection; no weights, ports, or workers.
  exec docker run "${common[@]}" --network none --gpus device=0 --cpus 4 --memory 8g \
    --entrypoint python3 "$image" -c '
import json
import sys
from sglang.srt.arg_groups.model_override_base import resolving_view
from sglang.srt.parser.template_detection import resolve_auto_parsers
from sglang.srt.server_args import prepare_server_args

args = prepare_server_args(sys.argv[1:])
args.resolve_once()
args.check_server_args()
resolve_auto_parsers(args)
cfg = resolving_view(args)
fields = (
    "speculative_algorithm", "speculative_num_draft_tokens", "dsv4_attn_backend",
    "moe_runner_backend", "tp_size", "ep_size", "context_length",
    "max_running_requests", "mem_fraction_static", "enable_hierarchical_cache",
    "reasoning_parser", "tool_call_parser",
)
print(json.dumps({key: getattr(cfg, key) for key in fields}, indent=2))
' "${args[@]}"
fi

if [[ $(systemctl show --property=ActiveState --value sglang-dsv41.service) != inactive ]]; then
  echo "The production service must be stopped during the maintenance window." >&2
  exit 1
fi
gpu_processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
if [[ -n "$gpu_processes" ]]; then
  echo "GPU compute processes are still present; refusing to start a second engine." >&2
  exit 1
fi

compile_cache=$cache_root/sglang/angelspec/runtime-82308b9d9
mkdir -p "$compile_cache"
exec docker run "${common[@]}" --name dsv41-angelspec --detach \
  --mount "type=bind,source=$compile_cache,target=/root/.cache/sglang" \
  --gpus all --shm-size 32g --ulimit memlock=-1:-1 --network host --ipc host \
  "$image" sglang serve "${args[@]}"
