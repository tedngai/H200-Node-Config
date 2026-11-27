#!/usr/bin/env bash
set -euo pipefail

revision=82308b9d9e1685aa9404272cc499c4c195610913
root=$(cd -- "$(dirname -- "$0")/.." && pwd)
work=$(mktemp -d /tmp/opencode/dsv41-dfly2-build.XXXXXX)
trap 'rm -rf -- "$work"' EXIT

git init -q "$work/source"
git -C "$work/source" fetch --depth=1 https://github.com/sgl-project/sglang.git "$revision"
test "$(git -C "$work/source" rev-parse FETCH_HEAD)" = "$revision"
mkdir "$work/context"
git -C "$work/source" archive "$revision" python/sglang \
  test/registered/unit/spec/test_dfly2_dsv41.py \
  test/registered/unit/managers/test_scheduler_sampling_mask_validation.py |
  tar -xf - -C "$work/context"
cp "$root/configs/angelspec/"{Dockerfile,requirements.txt} "$work/context/"
docker build --platform linux/amd64 --tag local/dsv41-angelspec:82308b9d9 "$work/context"
