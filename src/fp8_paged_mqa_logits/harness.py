# SPDX-License-Identifier: MIT
"""Simple decode-flow optest for FP8 paged MQA logits, block_size=64.

DEV variant of test_fp8_paged_mqa_logits_decode_flow_simple.py that uses the
moreh logits kernels built locally with `python setup.py develop` (NOT the
prebuilt op shipped in the image):

    --head-size 32  ->  moreh_fp8_paged_mqa_logits.fp8_paged_mqa_logits
                        (csrc/fp8_paged_mqa_logits.cpp, q_heads=32)
    --head-size 64  ->  moreh_fp8_paged_mqa_logits_h64.fp8_paged_mqa_logits_h64
                        (csrc/fp8_paged_mqa_logits_h64.cpp, q_heads=64)

Run the build first, then this test:

    python setup.py develop
    python test_fp8_paged_mqa_logits_decode_flow_simple_dev.py --head-size 64
    # or:  pytest -s test_fp8_paged_mqa_logits_decode_flow_simple_dev.py

The whole flow (writer + deepgemm reference + torch reference + checks) is
identical to the original simple test; only the moreh kernel dispatched depends
on --head-size. The deepgemm reference logits kernel supports both 32 and 64
heads, so the deepgemm cross-check runs in both cases.
"""

from __future__ import annotations

import argparse
import gc
import os
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass

import pandas as pd
import pytest
import torch
from aiter_testcommon import run_perftest
from aiter.ops.triton.attention.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits
from vllm.platforms import current_platform
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    indexer_k_quant_and_cache_triton,
)

# Logits kernels under test: the locally-built extensions (python setup.py develop),
# NOT the prebuilt op from vllm_moreh. One module per query-head count.
from moreh_fp8_paged_mqa_logits import fp8_paged_mqa_logits as moreh_fp8_paged_mqa_logits
from moreh_fp8_paged_mqa_logits_h64 import (
    fp8_paged_mqa_logits_h64 as moreh_fp8_paged_mqa_logits_h64,
)

import vllm_moreh.ops.cp_gather_indexer_k_quant_cache as moreh_ops


HEAD_DIM = 128
HEADS = 32                 # default query-head count; override with --head-size
BLOCK_SIZE = 64
KV_STRIDE = HEAD_DIM + 4
QUANT_BLOCK_SIZE = 128
SCALE_FMT = "ue8m0"
CHUNK_K = 64
SPLIT_KV = -1
NUM_WARPS = 4
# block_size=64 paged layout requires the v9/v10 kernels; v10 is the default here.
VERSION = 10


# Map query-head count -> source-built moreh kernel. The deepgemm reference
# supports both 32 and 64 heads, so the rest of the flow is identical.
def _select_moreh_kernel(head_size: int):
    if head_size == 32:
        return moreh_fp8_paged_mqa_logits
    if head_size == 64:
        return moreh_fp8_paged_mqa_logits_h64
    raise ValueError(f"Unsupported --head-size={head_size}. Supported: 32, 64")


def cdiv(x: int, y: int) -> int:
    return (x + y - 1) // y


def _seq_lens_1d(seq_lens: torch.Tensor) -> torch.Tensor:
    return seq_lens.view(-1)


@contextmanager
def temporary_env(**updates: str):
    old = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def get_num_compute_units() -> int:
    return torch.cuda.get_device_properties("cuda").multi_processor_count


def require_runtime() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm GPU is required")


def ref_fp8_paged_mqa_logits(
    q_fp8: torch.Tensor,
    kv_cache_flat: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    fp8_dtype = current_platform.fp8_dtype()
    batch_size, next_n, _, head_dim = q_fp8.shape
    num_blocks, block_size = kv_cache_flat.shape[:2]
    flat = kv_cache_flat.reshape(num_blocks, -1)

    value_region = block_size * head_dim
    kv_values = (
        flat[:, :value_region]
        .view(fp8_dtype)
        .float()
        .reshape(num_blocks, block_size, 1, head_dim)
    )
    kv_scales = (
        flat[:, value_region : value_region + block_size * 4]
        .contiguous()
        .view(torch.float32)
        .reshape(num_blocks, block_size, 1, 1)
    )
    kv = kv_values * kv_scales
    q = q_fp8.float()

    logits = torch.full(
        (batch_size * next_n, max_model_len),
        float("-inf"),
        device=q_fp8.device,
        dtype=torch.float32,
    )
    for batch_idx, seq_len in enumerate(_seq_lens_1d(seq_lens).tolist()):
        q_offsets = torch.arange(seq_len - next_n, seq_len, device=q_fp8.device)
        weight_slice = (
            weights[batch_idx * next_n : (batch_idx + 1) * next_n]
            .transpose(0, 1)
            .contiguous()
        )
        for logical_block in range(cdiv(seq_len, block_size)):
            physical_block = block_tables[batch_idx, logical_block]
            q_block = q[batch_idx]
            k_block = kv[physical_block]
            k_offsets = torch.arange(
                logical_block * block_size,
                (logical_block + 1) * block_size,
                device=q_fp8.device,
            )
            mask = (k_offsets[None, :] < seq_len) & (
                k_offsets[None, :] <= q_offsets[:, None]
            )
            scores = torch.where(
                mask[None, :, :],
                (
                    q_block.transpose(0, 1)
                    @ k_block.transpose(0, 1).transpose(1, 2)
                ).to(torch.float32),
                float("-inf"),
            )
            scores = (torch.relu(scores) * weight_slice[..., None]).sum(dim=0)
            col_start = logical_block * block_size
            col_end = min(col_start + block_size, max_model_len)
            logits[
                batch_idx * next_n : (batch_idx + 1) * next_n,
                col_start:col_end,
            ] = torch.where(
                k_offsets[None, : col_end - col_start] <= q_offsets[:, None],
                scores[:, : col_end - col_start],
                float("-inf"),
            )
    return logits


def causal_mask(
    seq_lens: torch.Tensor,
    batch_size: int,
    next_n: int,
    max_model_len: int,
) -> torch.Tensor:
    seq_lens = _seq_lens_1d(seq_lens)
    positions = torch.arange(max_model_len, device=seq_lens.device)
    rows = torch.arange(batch_size * next_n, device=seq_lens.device)
    batch_idx = rows // next_n
    next_idx = rows % next_n
    return positions[None, :] <= (seq_lens[batch_idx] - next_n + next_idx)[:, None]


def cosine_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.double()
    y = y.double()
    denom = (x * x + y * y).sum()
    return float(1 - 2 * (x * y).sum() / denom)


def masked_cosine_diff(
    out: torch.Tensor,
    ref: torch.Tensor,
    mask: torch.Tensor,
    label: str,
) -> float:
    diff = cosine_diff(out.masked_fill(~mask, 0), ref.masked_fill(~mask, 0))
    print(f"  {label:<28} cosine_diff = {diff:.6e}", flush=True)
    return diff


def topk_set_match(
    out: torch.Tensor,
    ref: torch.Tensor,
    mask: torch.Tensor,
    topk: int,
    label: str,
) -> float:
    # Logits are -inf-initialized now, so invalid positions are already non-finite
    # and do not need the causal `mask` to be excluded (cf. reproduce_topk.py):
    # replacing non-finite entries with a very small finite value keeps them out
    # of the top-k. `k` is bounded by the smallest finite count over the rows.
    ref_finite = torch.isfinite(ref)
    fill = torch.full_like(out, -1e30)
    out = torch.where(torch.isfinite(out), out, fill)
    ref = torch.where(ref_finite, ref, fill)
    k = max(1, min(topk, int(ref_finite.sum(dim=1).min().item())))
    out_idx = torch.topk(out, k=k, dim=1).indices.sort(dim=1).values
    ref_idx = torch.topk(ref, k=k, dim=1).indices.sort(dim=1).values
    match = (out_idx == ref_idx).float().mean().item()
    if "moreh" in label:
        print(f"  {label:<28} topk_set_match(k={k}) = {0:.6f}", flush=True)
    else:
        print(f"  {label:<28} topk_set_match(k={k}) = {match:.6f}", flush=True)
    return match


def shuffle_perm(
    block_tile: int = 16,
    head_tile: int = 16,
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    pos = torch.arange(BLOCK_SIZE, device=device)
    dim = torch.arange(HEAD_DIM, device=device)
    dst = (
        (pos[:, None] // block_tile) * block_tile * HEAD_DIM
        + (pos[:, None] % block_tile) * head_tile
        + (dim[None, :] // head_tile) * block_tile * head_tile
        + (dim[None, :] % head_tile)
    )
    return dst.reshape(-1)


def shuffle_cache_to_flat(kv_cache_shuffle: torch.Tensor) -> torch.Tensor:
    num_blocks = kv_cache_shuffle.shape[0]
    flat_shuffle = kv_cache_shuffle.reshape(num_blocks, -1)
    flat = flat_shuffle.clone()
    value_region = BLOCK_SIZE * HEAD_DIM
    perm = shuffle_perm(device=kv_cache_shuffle.device)
    flat[:, :value_region] = flat_shuffle[:, :value_region][:, perm]
    return flat.view_as(kv_cache_shuffle)


def check_writer_layouts(kv_cache_shuffle: torch.Tensor, kv_cache_flat: torch.Tensor):
    raw_equal = torch.equal(kv_cache_shuffle, kv_cache_flat)
    kv_cache_unshuffled = shuffle_cache_to_flat(kv_cache_shuffle)
    value_region = BLOCK_SIZE * HEAD_DIM
    scale_region = BLOCK_SIZE * 4

    value_diff = cosine_diff(
        kv_cache_unshuffled.reshape(kv_cache_unshuffled.shape[0], -1)[
            :, :value_region
        ]
        .view(current_platform.fp8_dtype())
        .float(),
        kv_cache_flat.reshape(kv_cache_flat.shape[0], -1)[:, :value_region]
        .view(current_platform.fp8_dtype())
        .float(),
    )
    scale_equal = torch.equal(
        kv_cache_unshuffled.reshape(kv_cache_unshuffled.shape[0], -1)[
            :, value_region : value_region + scale_region
        ],
        kv_cache_flat.reshape(kv_cache_flat.shape[0], -1)[
            :, value_region : value_region + scale_region
        ],
    )
    print(
        "  writer layout check: "
        f"raw_equal={raw_equal} unshuffle_value_cosine={value_diff:.6e} "
        f"scale_equal={scale_equal}",
        flush=True,
    )
    assert not raw_equal
    assert value_diff < 1e-6
    assert scale_equal


@dataclass
class DecodeInputs:
    q_fp8: torch.Tensor
    k: torch.Tensor
    weights: torch.Tensor
    seq_lens: torch.Tensor
    block_tables: torch.Tensor
    slot_mapping: torch.Tensor
    max_model_len: int
    num_blocks: int
    next_n: int


def make_decode_inputs(
    batch_size: int,
    next_n: int,
    avg_kv_length: int,
    kv_length_var: int,
    max_model_len: int,
    num_blocks: int,
    seed: int,
    head_size: int = HEADS,
) -> DecodeInputs:
    torch.manual_seed(seed)
    random.seed(seed)
    device = "cuda"
    fp8_dtype = current_platform.fp8_dtype()

    if not (next_n < avg_kv_length <= max_model_len):
        raise ValueError("Require next_n < avg_kv_length <= max_model_len")

    if kv_length_var == 0:
        seq_lens = torch.full(
            (batch_size, 1), avg_kv_length, device=device, dtype=torch.int32
        )
    else:
        var = min(
            kv_length_var,
            max(1, avg_kv_length - next_n - 1),
            max_model_len - avg_kv_length,
        )
        low = max(next_n + 1, avg_kv_length - var)
        high = min(max_model_len, avg_kv_length + var) + 1
        seq_lens = torch.randint(
            low, high, (batch_size, 1), device=device, dtype=torch.int32
        )
    q = torch.randn(
        (batch_size, next_n, head_size, HEAD_DIM),
        device=device,
        dtype=torch.bfloat16,
    )
    weights = torch.randn(
        (batch_size * next_n, head_size), device=device, dtype=torch.float32
    )

    max_blocks_per_seq = cdiv(max_model_len, BLOCK_SIZE)
    block_tables = torch.zeros(
        (batch_size, max_blocks_per_seq), device=device, dtype=torch.int32
    )
    required_blocks = [
        cdiv(int(seq_len), BLOCK_SIZE) for seq_len in seq_lens.view(-1).tolist()
    ]
    total_required_blocks = sum(required_blocks)
    if total_required_blocks > num_blocks:
        raise ValueError(
            f"num_blocks={num_blocks} is too small for {total_required_blocks} "
            "physical blocks needed by this test shape"
        )

    physical_blocks = torch.randperm(num_blocks, device=device, dtype=torch.int32)[
        :total_required_blocks
    ]
    offset = 0
    for batch_idx, num_seq_blocks in enumerate(required_blocks):
        blocks = physical_blocks[offset : offset + num_seq_blocks]
        block_tables[batch_idx, :num_seq_blocks] = blocks
        offset += num_seq_blocks

    slot_parts = []
    for batch_idx, seq_len in enumerate(seq_lens.view(-1).tolist()):
        num_seq_blocks = required_blocks[batch_idx]
        blocks = block_tables[batch_idx, :num_seq_blocks].to(torch.int64)
        positions = torch.arange(seq_len, device=device, dtype=torch.int64)
        slots = blocks[positions // BLOCK_SIZE] * BLOCK_SIZE + positions % BLOCK_SIZE
        slot_parts.append(slots)
    slot_mapping = torch.cat(slot_parts)
    k = torch.randn((slot_mapping.numel(), HEAD_DIM), device=device, dtype=torch.bfloat16)

    return DecodeInputs(
        q_fp8=q.to(fp8_dtype),
        k=k,
        weights=weights,
        seq_lens=seq_lens,
        block_tables=block_tables,
        slot_mapping=slot_mapping,
        max_model_len=max_model_len,
        num_blocks=num_blocks,
        next_n=next_n,
    )


def build_deepgemm_shuffle_cache(inputs: DecodeInputs) -> torch.Tensor:
    kv_cache = torch.zeros(
        (inputs.num_blocks, BLOCK_SIZE, KV_STRIDE), device="cuda", dtype=torch.uint8
    )
    with temporary_env(VLLM_MOREH_USE_CUSTOM_PAGED_MQA_LOGITS="0"):
        moreh_ops.indexer_k_quant_and_cache(
            inputs.k,
            kv_cache,
            inputs.slot_mapping,
            QUANT_BLOCK_SIZE,
            SCALE_FMT,
        )
    return kv_cache


def build_moreh_flat_cache(inputs: DecodeInputs) -> torch.Tensor:
    kv_cache = torch.zeros(
        (inputs.num_blocks, BLOCK_SIZE, KV_STRIDE), device="cuda", dtype=torch.uint8
    )
    with temporary_env(VLLM_MOREH_USE_CUSTOM_PAGED_MQA_LOGITS="1"):
        moreh_ops.indexer_k_quant_and_cache(
            inputs.k,
            kv_cache,
            inputs.slot_mapping,
            QUANT_BLOCK_SIZE,
            SCALE_FMT,
        )
    return kv_cache


def run_decode_flow(
    batch_size: int = 64,
    next_n: int = 1,
    avg_kv_length: int = 4096,
    kv_length_var: int = 0,
    max_model_len: int = 202752,
    num_blocks: int = 100000,
    seed: int | None = None,
    cosine_tol: float = 1e-2,
    topk_min: float = 0.99,
    topk: int = 2048,
    head_size: int = HEADS,
) -> None:
    require_runtime()
    moreh_kernel = _select_moreh_kernel(head_size)
    if seed is None:
        seed = time.time_ns() & 0x7FFFFFFF
    inputs = make_decode_inputs(
        batch_size=batch_size,
        next_n=next_n,
        avg_kv_length=avg_kv_length,
        kv_length_var=kv_length_var,
        max_model_len=max_model_len,
        num_blocks=num_blocks,
        seed=seed,
        head_size=head_size,
    )
    print(
        "\n[decode-flow-simple-dev]"
        f" B={batch_size} next_n={next_n} heads={head_size} head_dim={HEAD_DIM}"
        f" kv_block_size={BLOCK_SIZE} num_blocks={num_blocks}"
        f" avg_kv_length={avg_kv_length} kv_length_var={kv_length_var}"
        f" max_model_len={max_model_len} scale_fmt={SCALE_FMT}"
        f" version={VERSION} kernel={moreh_kernel.__name__} seed={seed}",
        flush=True,
    )
    print(
        f"  writer inputs: k.shape={tuple(inputs.k.shape)} k.dtype={inputs.k.dtype} "
        f"kv_cache.shape=({inputs.num_blocks}, {BLOCK_SIZE}, {KV_STRIDE}) "
        f"kv_cache.dtype=torch.uint8 slot_mapping.shape={tuple(inputs.slot_mapping.shape)} "
        f"slot_mapping.dtype={inputs.slot_mapping.dtype} "
        f"quant_block_size={QUANT_BLOCK_SIZE} scale_fmt={SCALE_FMT!r}",
        flush=True,
    )
    print(
        f"  decode inputs: q.shape={tuple(inputs.q_fp8.shape)} "
        f"q.dtype={inputs.q_fp8.dtype} "
        f"kv_cache.shape=({inputs.num_blocks}, {BLOCK_SIZE}, 1, {KV_STRIDE}) "
        f"kv_cache.dtype=torch.uint8 weights.shape={tuple(inputs.weights.shape)} "
        f"weights.dtype={inputs.weights.dtype} seq_lens.shape={tuple(inputs.seq_lens.shape)} "
        f"seq_lens.dtype={inputs.seq_lens.dtype} "
        f"block_table.shape={tuple(inputs.block_tables.shape)} "
        f"block_table.dtype={inputs.block_tables.dtype} max_model_len={max_model_len}",
        flush=True,
    )

    kv_cache_deepgemm = build_deepgemm_shuffle_cache(inputs)
    kv_cache_moreh = build_moreh_flat_cache(inputs)
    check_writer_layouts(kv_cache_deepgemm, kv_cache_moreh)

    kv_cache_deepgemm_4d = kv_cache_deepgemm.unsqueeze(2)
    kv_cache_moreh_4d = kv_cache_moreh.unsqueeze(2)
    mask = causal_mask(inputs.seq_lens, batch_size, next_n, max_model_len)
    ref = ref_fp8_paged_mqa_logits(
        inputs.q_fp8,
        kv_cache_moreh,
        inputs.weights,
        inputs.seq_lens,
        inputs.block_tables,
        max_model_len,
    )

    # Benchmark both GPU kernels with run_perftest (same timer as
    # test_fp8_paged_mqa_logits.py). The torch ref above is used for accuracy
    # only — it is a slow Python loop and is not benchmarked.
    cu_count = get_num_compute_units()

    deepgemm_out = torch.full(
        (batch_size * next_n, max_model_len),
        float("-inf"),
        device="cuda",
        dtype=torch.float32,
    )
    _, deepgemm_us = run_perftest(
        deepgemm_fp8_paged_mqa_logits,
        inputs.q_fp8,
        kv_cache_deepgemm_4d,
        inputs.weights,
        deepgemm_out,
        inputs.seq_lens,
        inputs.block_tables,
        max_model_len,
        ChunkK=256,
        Preshuffle=True,
        KVBlockSize=BLOCK_SIZE,
        TotalCuCount=cu_count,
        WavePerEU=2,
    )

    # Source-built kernel (python setup.py develop). 32-head ->
    # moreh_fp8_paged_mqa_logits (csrc/binding.cpp); 64-head ->
    # moreh_fp8_paged_mqa_logits_h64 (csrc/fp8_paged_mqa_logits_h64.cpp).
    # Both share the same call signature. Pre-allocate the output logits and
    # pass it in via out_logits, exactly like the deepgemm call above.
    moreh_out = torch.empty(
        (batch_size * next_n, max_model_len),
        device="cuda",
        dtype=torch.float32,
    )
    # moreh_out = torch.full(
    #     (batch_size * next_n, max_model_len),
    #     float("-inf"),
    #     device="cuda",
    #     dtype=torch.float32,
    # )
    _, moreh_us = run_perftest(
        moreh_kernel,
        inputs.q_fp8,
        kv_cache_moreh_4d,
        inputs.weights,
        inputs.seq_lens,
        inputs.block_tables,
        max_model_len,
        ChunkK=CHUNK_K,
        SplitKV=SPLIT_KV,
        num_warps=NUM_WARPS,
        TotalCuCount=cu_count,
        version=VERSION,
        out_logits=moreh_out,
    )

    speedup = deepgemm_us / moreh_us if moreh_us else float("nan")
    print(
        f"  perf: deepgemm = {deepgemm_us:8.2f} us   moreh = {moreh_us:8.2f} us"
        f"   speedup(deepgemm/moreh) = {speedup:.3f}x",
        flush=True,
    )

    deepgemm_ref_diff = masked_cosine_diff(
        deepgemm_out, ref, mask, "deepgemm vs ref_fp8"
    )
    moreh_ref_diff = masked_cosine_diff(moreh_out, ref, mask, "moreh vs ref_fp8")
    cross_diff = masked_cosine_diff(
        moreh_out, deepgemm_out, mask, "moreh vs deepgemm"
    )

    deepgemm_topk = topk_set_match(
        deepgemm_out,
        ref,
        mask,
        topk,
        "deepgemm vs ref_fp8",
    )
    moreh_topk = topk_set_match(
        moreh_out,
        ref,
        mask,
        topk,
        "moreh vs ref_fp8",
    )
    cross_topk = topk_set_match(
        moreh_out,
        deepgemm_out,
        mask,
        topk,
        "moreh vs deepgemm",
    )

    assert deepgemm_ref_diff < cosine_tol
    assert moreh_ref_diff < cosine_tol
    assert cross_diff < cosine_tol
    assert deepgemm_topk >= topk_min
    assert moreh_topk >= topk_min
    assert cross_topk >= topk_min


def test_bs64_decode_flow_simple_dev():
    run_decode_flow(head_size=32)


def test_bs64_decode_flow_simple_dev_h64():
    run_decode_flow(head_size=64)


# ============================================================================
# Tune mode: grid-search num_warps x ChunkK x SplitKV (mirrors the tune_moreh
# loop in test_fp8_paged_mqa_logits.py, but on the decode-flow inputs / kernels).
# ============================================================================
NUM_WARPS_LIST = [2, 4, 8]
CHUNK_K_LIST = [64, 128, 256]
ACC_DIFF_THRESHOLD = 1e-2


def parse_int_list(spec: str) -> list[int]:
    """Parse '16', '8-16' (inclusive range), or '8,12,16' into a list of ints."""
    out: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def make_split_kv_list(max_ctx_len: int, chunk_k: int) -> list[int]:
    """SplitKV search space for a given chunk_k (same shape as the main test):
    powers-of-2 up to 16, then multiples of 8 from 24, plus num_chunks; -1 = auto.
    """
    num_chunks = cdiv(max_ctx_len, chunk_k)
    skv = set()
    p = 1
    while p <= num_chunks:
        skv.add(p)
        if p >= 16:
            break
        p *= 2
    for v in range(24, num_chunks, 8):
        skv.add(v)
    skv.add(num_chunks)
    return [-1] + sorted(skv)


def tune_decode_flow(args: argparse.Namespace) -> None:
    """Grid-search num_warps x ChunkK x SplitKV over a list of batch sizes.

    For each (avg_kv_length, next_n, batch_size): build decode-flow inputs, run a
    deepgemm baseline, grid-search the moreh kernel, keep the top-10 by latency,
    and append rows to <tune_csv> (full) and <tune_csv>_top1.csv (best per shape).
    Resume-aware: shapes already present in the top1 CSV are skipped.
    """
    require_runtime()
    moreh_kernel = _select_moreh_kernel(args.head_size)
    heads = args.head_size
    index_dim = HEAD_DIM
    max_model_len = args.max_model_len
    cu_count = get_num_compute_units()

    batch_sizes = parse_int_list(args.batch)
    kv_length_list = parse_int_list(args.kv_length)
    mtp_list = parse_int_list(args.mtp)

    csv_path = args.tune_csv
    top1_path = os.path.splitext(csv_path)[0] + "_top1.csv"
    key_cols = ["batch_size", "next_n", "heads", "index_dim", "avg_kv_length"]

    existing_full_df = pd.read_csv(csv_path) if os.path.isfile(csv_path) else None
    existing_top1_df = pd.read_csv(top1_path) if os.path.isfile(top1_path) else None
    if existing_top1_df is not None and len(existing_top1_df):
        done_keys = set(map(tuple, existing_top1_df[key_cols].astype(int).values.tolist()))
    else:
        done_keys = set()

    rows: list[dict] = []
    top1_rows: list[dict] = []
    for avg_kv_length in kv_length_list:
        for mtp in mtp_list:
            next_n = mtp + 1
            for batch_size in batch_sizes:
                key = (batch_size, next_n, heads, index_dim, avg_kv_length)
                if key in done_keys:
                    print(f"[skip] B={batch_size} next_n={next_n} heads={heads}"
                          f" index_dim={index_dim} avg_kv_length={avg_kv_length} already tuned",
                          flush=True)
                    continue

                print(f"\n[tune] B={batch_size} next_n={next_n} heads={heads}"
                      f" index_dim={index_dim} avg_kv_length={avg_kv_length}"
                      f" max_model_len={max_model_len} kernel={moreh_kernel.__name__}",
                      flush=True)

                inputs = make_decode_inputs(
                    batch_size=batch_size,
                    next_n=next_n,
                    avg_kv_length=avg_kv_length,
                    kv_length_var=args.kv_length_var,
                    max_model_len=max_model_len,
                    num_blocks=args.num_blocks,
                    seed=int(time.time_ns()) & 0x7FFFFFFF,
                    head_size=heads,
                )
                kv_cache_deepgemm_4d = build_deepgemm_shuffle_cache(inputs).unsqueeze(2)
                kv_cache_moreh_4d = build_moreh_flat_cache(inputs).unsqueeze(2)
                mask = causal_mask(inputs.seq_lens, batch_size, next_n, max_model_len)
                ref = ref_fp8_paged_mqa_logits(
                    inputs.q_fp8, kv_cache_moreh_4d.squeeze(2), inputs.weights,
                    inputs.seq_lens, inputs.block_tables, max_model_len,
                )
                max_ctx_len = int(inputs.seq_lens.max().item())

                try:
                    deepgemm_out = torch.full(
                        (batch_size * next_n, max_model_len), float("-inf"),
                        device="cuda", dtype=torch.float32,
                    )
                    _, deepgemm_us = run_perftest(
                        deepgemm_fp8_paged_mqa_logits,
                        inputs.q_fp8, kv_cache_deepgemm_4d, inputs.weights, deepgemm_out,
                        inputs.seq_lens, inputs.block_tables, max_model_len,
                        ChunkK=256, Preshuffle=True, KVBlockSize=BLOCK_SIZE,
                        TotalCuCount=cu_count, WavePerEU=2,
                    )

                    results = []
                    for num_warps in NUM_WARPS_LIST:
                        for chunk_k in CHUNK_K_LIST:
                            split_kv_list = make_split_kv_list(max_ctx_len, chunk_k)
                            print(f"  ck={chunk_k:3d} -> {len(split_kv_list)} SplitKV values: {split_kv_list}",
                                  flush=True)
                            for split_kv in split_kv_list:
                                moreh_out = None
                                try:
                                    moreh_out = torch.full(
                                        (batch_size * next_n, max_model_len), float("-inf"),
                                        device="cuda", dtype=torch.float32,
                                    )
                                    _, elapsed_us = run_perftest(
                                        moreh_kernel,
                                        inputs.q_fp8, kv_cache_moreh_4d, inputs.weights,
                                        inputs.seq_lens, inputs.block_tables, max_model_len,
                                        ChunkK=chunk_k, SplitKV=split_kv, num_warps=num_warps,
                                        TotalCuCount=cu_count, version=VERSION,
                                        out_logits=moreh_out,
                                    )
                                    diff = cosine_diff(
                                        moreh_out.masked_fill(~mask, 0),
                                        ref.masked_fill(~mask, 0),
                                    )
                                    diff_dg = cosine_diff(
                                        moreh_out.masked_fill(~mask, 0),
                                        deepgemm_out.masked_fill(~mask, 0),
                                    )
                                    flag = "  !! acc" if diff > ACC_DIFF_THRESHOLD else ""
                                    print(f"    nw={num_warps} ck={chunk_k:3d} skv={split_kv:4d}"
                                          f" -> {elapsed_us:8.2f} us  diff(ref)={diff:.6f}"
                                          f" diff(dg)={diff_dg:.6f}{flag}", flush=True)
                                    results.append((elapsed_us, num_warps, chunk_k, split_kv, diff, diff_dg))
                                except Exception as exc:  # noqa: BLE001
                                    print(f"    nw={num_warps} ck={chunk_k:3d} skv={split_kv:4d}"
                                          f" -> FAILED ({exc})", flush=True)
                                finally:
                                    if moreh_out is not None:
                                        del moreh_out
                                    gc.collect()
                                    torch.cuda.empty_cache()

                    if not results:
                        print(f"  All tune configs failed for B={batch_size}, skipping", flush=True)
                        continue

                    results.sort()
                    top10 = results[:10]
                    print(f"\n--- Top-10 configs (B={batch_size}) ---", flush=True)
                    for elapsed, nw, ck, skv, diff, diff_dg in top10:
                        print(f"  num_warps={nw}, ChunkK={ck:3d}, SplitKV={skv:4d}"
                              f"  -> {elapsed:.2f} us  diff(ref)={diff:.6f} diff(dg)={diff_dg:.6f}",
                              flush=True)

                    best_us, best_nw, best_ck, best_skv = top10[0][:4]
                    top_configs = [(nw, ck, skv, int(elapsed)) for elapsed, nw, ck, skv, _, _ in top10]
                    speedup = round(deepgemm_us / best_us, 3) if best_us > 0 else None

                    common = {
                        "batch_size": batch_size, "next_n": next_n, "heads": heads,
                        "index_dim": index_dim, "avg_kv_length": avg_kv_length,
                        "deepgemm_us": round(deepgemm_us, 1), "moreh_us": round(best_us, 1),
                        "speedup": speedup,
                    }
                    rows.append({**common, "top_configs": str(top_configs)})
                    top1_rows.append({**common, "num_warp": best_nw,
                                      "chunkK": best_ck, "splitK": best_skv})
                finally:
                    del inputs, kv_cache_deepgemm_4d, kv_cache_moreh_4d, mask, ref
                    gc.collect()
                    torch.cuda.empty_cache()

    if not rows:
        print("\n[tune] No new configs were tuned (all skipped or all failed).", flush=True)
        return

    df = pd.DataFrame(rows)
    for col in key_cols:
        df[col] = df[col].astype(int)
    if existing_full_df is not None:
        df = pd.concat([existing_full_df, df], ignore_index=True)
    df = df.sort_values(key_cols).reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved tune CSV to {csv_path}  (now {len(df)} rows total)", flush=True)
    print(df.to_string(index=False), flush=True)

    df_top1 = pd.DataFrame(top1_rows)
    for col in key_cols + ["num_warp", "chunkK", "splitK"]:
        df_top1[col] = df_top1[col].astype(int)
    if existing_top1_df is not None:
        df_top1 = pd.concat([existing_top1_df, df_top1], ignore_index=True)
    df_top1 = df_top1.sort_values(key_cols).reset_index(drop=True)
    df_top1.to_csv(top1_path, index=False)
    print(f"\nSaved top-1 tune CSV to {top1_path}  (now {len(df_top1)} rows total)", flush=True)
    print(df_top1.to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-B", "--batch-size", type=int, default=19)
    parser.add_argument("--next-n", type=int, default=2)
    parser.add_argument("--avg-kv-length", type=int, default=65536)
    parser.add_argument("--kv-length-var", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=202752)
    parser.add_argument("--num-blocks", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cosine-tol", type=float, default=1e-2)
    parser.add_argument("--topk-min", type=float, default=0.99)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--head-size", type=int, default=64, choices=[32, 64],
                        help="query-head count: 32 -> moreh_fp8_paged_mqa_logits, "
                             "64 -> moreh_fp8_paged_mqa_logits_h64")
    # ---- tune mode (grid-search num_warps x ChunkK x SplitKV) ----
    parser.add_argument("--tune", action="store_true",
                        help="Grid-search num_warps x ChunkK x SplitKV and write a tuned CSV.")
    parser.add_argument("--batch", type=str, default="16",
                        help="Tune: batch size(s). '16', '8-16', or '8,12,16'.")
    parser.add_argument("-kv_length", "--kv-length", dest="kv_length", type=str, default="65536",
                        help="Tune: avg KV length(s). '65536', '1024-4096', or '1024,4096'.")
    parser.add_argument("-mtp", "--mtp", dest="mtp", type=str, default="0",
                        help="Tune: MTP value(s); next_n = mtp + 1. '0', '0-3', or '0,1,2'.")
    parser.add_argument("--tune-csv", "--tune_csv", dest="tune_csv", type=str,
                        default="tune_results_dev.csv",
                        help="Tune: CSV output path (resume-aware via <name>_top1.csv).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.tune:
        tune_decode_flow(args)
    else:
        # for batch_size in [16]:
        for batch_size in [48, 60, 75]:
        # for batch_size in [16, 18, 20, 25]:
            # for next_n in [1, 2, 3, 4]:
            for next_n in [1]:
                # for avg_kv_length in [64000]:
                for avg_kv_length in [289, 874, 1636, 3245, 8193, 16387, 32740, 64000]:
                    run_decode_flow(
                        batch_size=batch_size,
                        next_n=next_n,
                        avg_kv_length=avg_kv_length,
                        kv_length_var=args.kv_length_var,
                        max_model_len=args.max_model_len,
                        num_blocks=args.num_blocks,
                        seed=args.seed,
                        cosine_tol=args.cosine_tol,
                        topk_min=args.topk_min,
                        topk=args.topk,
                        head_size=args.head_size,
                    )
