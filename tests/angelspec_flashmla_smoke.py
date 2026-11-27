"""Smoke-test DFly2's two-cache FlashMLA call on one H200 without model weights."""

import torch

from sglang.kernels.ops.attention.flash_mla_interface import (
    flash_mla_with_kvcache,
    get_mla_metadata,
)


def v4_cache(page_count: int, page_size: int, value: float) -> torch.Tensor:
    # V4 stores 448 FP8 values and 64 BF16 values per token, then page-local scales.
    cache = torch.empty((page_count, page_size * 584), device="cuda", dtype=torch.uint8)
    numbers = cache[:, : page_size * 576].view(page_count, page_size, 576)
    numbers[..., :448].view(torch.float8_e4m3fn).copy_(
        torch.full((page_count, page_size, 448), value, device="cuda")
    )
    numbers[..., 448:].view(torch.bfloat16).fill_(value)
    cache[:, page_size * 576 :].fill_(127)  # UE8M0 scale of one
    return cache.view(page_count, page_size, 1, 584)


def main() -> None:
    assert torch.cuda.get_device_capability() == (9, 0)
    rows, heads, dim = 6, 64, 512  # B6, padded heads on SM90, DSV4 latent width
    query = torch.zeros((rows, 1, heads, dim), device="cuda", dtype=torch.bfloat16)
    swa_cache = v4_cache(4, 64, 1.0)
    sparse_cache = v4_cache(2, 256, 2.0)
    swa_indices = torch.arange(192, device="cuda", dtype=torch.int32)
    sparse_indices = torch.arange(512, device="cuda", dtype=torch.int32)
    swa_indices = swa_indices.repeat(rows, 1, 1)
    sparse_indices = sparse_indices.repeat(rows, 1, 1)
    swa_lengths = torch.full((rows,), 134, device="cuda", dtype=torch.int32)
    sparse_lengths = torch.full((rows,), 512, device="cuda", dtype=torch.int32)
    sink = torch.zeros((heads,), device="cuda")

    def decode() -> torch.Tensor:
        return flash_mla_with_kvcache(
            q=query,
            k_cache=swa_cache,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=dim,
            tile_scheduler_metadata=get_mla_metadata()[0],
            is_fp8_kvcache=True,
            indices=swa_indices,
            topk_length=swa_lengths,
            attn_sink=sink,
            extra_k_cache=sparse_cache,
            extra_indices_in_kvcache=sparse_indices,
            extra_topk_length=sparse_lengths,
            use_legacy=True,
        )[0]

    def check(output: torch.Tensor) -> None:
        # Zero queries and zero sink logits give equal weight to valid rows
        # and one extra zero-valued sink.
        expected = (swa_lengths + 2 * sparse_lengths) / (swa_lengths + sparse_lengths + 1)
        torch.testing.assert_close(
            output.float(), expected[:, None, None, None].expand_as(output),
            atol=0.01, rtol=0,
        )

    check(decode())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = decode()
    sparse_lengths.fill_(96)
    graph.replay()
    check(output)
    print(f"{torch.cuda.get_device_name()}: dual-cache FlashMLA and CUDA graph replay passed")


if __name__ == "__main__":
    main()
