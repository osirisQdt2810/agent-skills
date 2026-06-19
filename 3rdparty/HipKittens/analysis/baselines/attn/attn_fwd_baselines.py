import torch
# import tk_kernel
import random
import time
import math
from torch.nn.functional import scaled_dot_product_attention
import aiter

torch.set_printoptions(
    precision=3,        # decimals
    sci_mode=False,     # True → always scientific
    linewidth=220,      # characters per line before folding
    threshold=float("inf")  # print every element, no summarising "..."
)

# Inputs
B = 16
H = 16
H_KV = 16
N = 16384
D = 64
causal = True
dtype = torch.bfloat16

def flops(batch, seqlen, nheads, headdim, causal, mode="fwd"):
    """Calculate FLOPs for attention operation."""
    flop = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return flop

def efficiency(flop, time):
    """Calculate efficiency in TFLOPS."""
    flop = flop / 1e12  # convert to TFLOPS
    time = time / 1e3   # convert to seconds
    return flop / time

def robustness_check(ref, pred):
    ref = ref.float()
    pred = pred.float()
    diff = (ref - pred).abs()
    denom = ref.abs().clamp_min(1e-6)
    mask = (diff > (0.001 + 0.05 * denom))
    error_count = mask.sum().item()
    numel = ref.numel()
    rel_error = error_count / numel
    l2_error = (diff.pow(2).sum().sqrt() / ref.pow(2).sum().sqrt()).item()
    cos = torch.nn.functional.cosine_similarity(ref.flatten(), pred.flatten(), dim=0).item()
    return diff, error_count, numel, rel_error, l2_error, cos, mask  


num_warmup = 500
num_iters = 100

start_event = torch.cuda.Event(enable_timing=True) # in milliseconds
end_event = torch.cuda.Event(enable_timing=True)
flops_ref = flops(B, N, H, D, causal)

# Reference matmul using AITER
for _ in range(num_warmup):
    q = torch.randn(B, N, H, D, dtype=dtype, device='cuda', requires_grad=True)
    k = torch.randn(B, N, H_KV, D, dtype=dtype, device='cuda', requires_grad=True)
    v = torch.randn(B, N, H_KV, D, dtype=dtype, device='cuda', requires_grad=True)
    out_ref, lse_ref = aiter.flash_attn_func(q, k, v, causal=causal, return_lse=True, deterministic=True)
timings_ref = []
torch.manual_seed(0)
random.seed(0)
for _ in range(num_iters):
    q = torch.randn(B, N, H, D, dtype=dtype, device='cuda', requires_grad=True)
    k = torch.randn(B, N, H_KV, D, dtype=dtype, device='cuda', requires_grad=True)
    v = torch.randn(B, N, H_KV, D, dtype=dtype, device='cuda', requires_grad=True)
    torch.cuda.synchronize()
    start_event.record()
    out_ref, lse_ref = aiter.flash_attn_func(q, k, v, causal=causal, return_lse=True, deterministic=True)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    timings_ref.append(elapsed_time)
print(f"{out_ref.dtype=}")
avg_time_ref = sum(timings_ref) / len(timings_ref)
eff_ref = efficiency(flops_ref, avg_time_ref)
print(f"AITER (AMD) reference average execution time: {avg_time_ref:.4f} ms")
print(f"AITER (AMD) reference performance: {eff_ref:.2f} TFLOPS for {B=} {H=} {N=} {D=} {causal=}.\n")


# pytorch 
# Reference matmul using PyTorch
for _ in range(num_warmup):
    q = torch.randn(B, H, N, D, dtype=dtype, device='cuda', requires_grad=True)
    k = torch.randn(B, H_KV, N, D, dtype=dtype, device='cuda', requires_grad=True)
    v = torch.randn(B, H_KV, N, D, dtype=dtype, device='cuda', requires_grad=True)
    out_ref_pytorch = scaled_dot_product_attention(q, k, v, is_causal=causal)
timings_ref = []
torch.manual_seed(0)
random.seed(0)
for _ in range(num_iters):
    q = torch.randn(B, N, H, D, dtype=dtype, device='cuda', requires_grad=True).transpose(1, 2)
    k = torch.randn(B, N, H_KV, D, dtype=dtype, device='cuda', requires_grad=True).transpose(1, 2)
    v = torch.randn(B, N, H_KV, D, dtype=dtype, device='cuda', requires_grad=True).transpose(1, 2)
    torch.cuda.synchronize()
    start_event.record()
    out_ref_pytorch = scaled_dot_product_attention(q, k, v, is_causal=causal)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    timings_ref.append(elapsed_time)
print(f"{out_ref_pytorch.dtype=}")
avg_time_ref = sum(timings_ref) / len(timings_ref)
eff_ref = efficiency(flops_ref, avg_time_ref)
print(f"PyTorch reference average execution time: {avg_time_ref:.4f} ms")
print(f"PyTorch reference performance: {eff_ref:.2f} TFLOPS for {B=} {H=} {N=} {D=} {causal=}.\n")

out_ref_pytorch_transposed = out_ref_pytorch.transpose(1, 2)


# Compare against reference
num_print = 16
print(f"\n PyTorch vs AITER comparison:")
print("\nO outputs:")
print("PyTorch: ", out_ref_pytorch_transposed[0, 0, :num_print, 0], "Max:", out_ref_pytorch_transposed.max().item())
print("AITER: ", out_ref[0, 0, :num_print, 0], "Max:", out_ref.max().item())

print("Robustness check:")
o_diff, o_err_cnt, o_total, o_rel_error, o_l2_error, o_cos, o_mask = robustness_check(out_ref_pytorch_transposed, out_ref)
print(f"O: max_abs={o_diff.max().item():.6f}, max_rel={o_rel_error:.4f}, "
      f"rel_l2={o_l2_error:.4f}, cos={o_cos:.6f}, "
      f"errors={o_err_cnt}/{o_total} ({100*o_err_cnt/o_total:.4f}%)")
