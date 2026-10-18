#!/usr/bin/env python3
"""Verify the local DeepSeek-V4.1-Flash-1017 safetensors cache without loading weights."""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

REPO = "models--deepseek-ai--DeepSeek-V4.1-Flash-1017"
CACHE = Path("/raid/advtech/cache/huggingface/hub") / REPO


def find_snapshot() -> Path:
    snaps = sorted((CACHE / "snapshots").glob("*"))
    if not snaps:
        raise SystemExit(f"no snapshot under {CACHE}")
    return snaps[-1]


def check_file(path: Path) -> dict:
    with path.open("rb") as f:
        head = f.read(8)
        if len(head) != 8:
            raise ValueError(f"{path.name}: short header length")
        n = struct.unpack("<Q", head)[0]
        if n <= 0 or n > 100 * 1024 * 1024:
            raise ValueError(f"{path.name}: implausible header length {n}")
        header = json.loads(f.read(n))
    end = 8 + n
    tensors = 0
    data_bytes = 0
    for key, value in header.items():
        if key == "__metadata__":
            continue
        tensors += 1
        offsets = value.get("data_offsets")
        if not offsets or len(offsets) != 2:
            raise ValueError(f"{path.name}: bad offsets for {key}")
        end = max(end, 8 + n + int(offsets[1]))
        data_bytes = max(data_bytes, int(offsets[1]))
    size = path.stat().st_size
    if end != size:
        raise ValueError(f"{path.name}: header end {end} != file size {size}")
    return {"file": path.name, "bytes": size, "tensors": tensors, "data_bytes": data_bytes}


def main() -> int:
    snap = find_snapshot()
    index = json.loads((snap / "model.safetensors.index.json").read_text())
    files = sorted(snap.glob("*.safetensors"))
    report = {
        "snapshot": str(snap),
        "file_count": len(files),
        "index_total_size": index.get("metadata", {}).get("total_size"),
        "files": [],
    }
    total = 0
    data_total = 0
    for path in files:
        info = check_file(path)
        total += info["bytes"]
        data_total += info["data_bytes"]
        report["files"].append(info)
    report["sum_file_bytes"] = total
    report["sum_tensor_data_bytes"] = data_total
    report["ok"] = (
        len(files) == 48
        and data_total == report["index_total_size"]
        and (snap / "config.json").is_file()
        and (snap / "tokenizer.json").is_file()
    )
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
