import torch
import random
import math
import time

use_aiter = True
if use_aiter:
    import aiter

torch.cuda.set_device(6)
torch.manual_seed(0)
random.seed(0)

torch.set_printoptions(
    precision=3,        
    sci_mode=False,     
    linewidth=220,      
    threshold=float("inf")  
)

torch.cuda.set_device(5)

# **************************************************
# Benchmarking
# **************************************************

num_warmup = 500
num_iters = 100
start_event = torch.cuda.Event(enable_timing=True) # in milliseconds
end_event = torch.cuda.Event(enable_timing=True)

def flops(batch, seqlen, nheads, headdim, causal, mode="bwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

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


# **************************************************
# Reference
# **************************************************

def expand_kv_for_gqa(K, V, h_q, h_kv):
    """Expand K,V from h_kv heads to h_q heads for GQA by replicating each KV head"""
    group_size = h_q // h_kv
    # Repeat each KV head group_size times: (B, h_kv, N, D) -> (B, h_q, N, D)
    K_expanded = K.repeat_interleave(group_size, dim=1)  
    V_expanded = V.repeat_interleave(group_size, dim=1)
    return K_expanded, V_expanded

def reference_forward(Q, K, V, causal):
    """GQA Reference implementation using BHND layout (batch, heads, seq, dim)"""
    # Convert to float64 and create new leaf tensors with requires_grad
    q_ = Q.detach().to(torch.float64).requires_grad_(True)
    k_ = K.detach().to(torch.float64).requires_grad_(True) 
    v_ = V.detach().to(torch.float64).requires_grad_(True)
    
    # Expand K,V to match Q heads for GQA computation
    k_expanded, v_expanded = expand_kv_for_gqa(k_, v_, h_q, h_kv)
    
    # Manual pytorch implementation of scaled dot product attention
    QK = torch.matmul(q_, k_expanded.transpose(-2, -1))
    QK /= (q_.size(-1) ** 0.5)
    QK = torch.nn.functional.softmax(QK, dim=-1)
    output = torch.matmul(QK, v_expanded)
    
    return output, q_, k_, v_

def simple_flash_backward(Q, K, V, dO, L):
    """GQA version that should match PyTorch exactly"""
    D = Q.shape[-1]
    scale = 1.0 / math.sqrt(D)
    
    # Expand K,V to match Q heads for GQA computation  
    K_expanded, V_expanded = expand_kv_for_gqa(K, V, h_q, h_kv)

    # Recompute scores and probabilities with saved m, l
    S = torch.matmul(Q, K_expanded.transpose(-2, -1)) * scale
    P = torch.exp(S - L.unsqueeze(-1))
    O = torch.matmul(P, V_expanded)

    # dV - need to sum across grouped heads  
    dV_expanded = torch.matmul(P.transpose(-2, -1), dO)  # (B, h_q, N, D)
    dV = torch.zeros_like(V)
    group_size = h_q // h_kv
    for i in range(h_kv):
        start_idx = i * group_size
        end_idx = (i + 1) * group_size
        dV[:, i, :, :] = dV_expanded[:, start_idx:end_idx, :, :].sum(dim=1)

    # softmax backward
    Delta = (dO * O).sum(dim=-1, keepdim=True)                 # (B, h_q, N, 1)
    dS = P * (torch.matmul(dO, V_expanded.transpose(-2, -1)) - Delta)   # (B, h_q, N, N)

    # chain rule through S = (Q K^T) * scale  
    dQ = torch.matmul(dS, K_expanded) * scale  # (B, h_q, N, D)
    
    # dK - need to sum across grouped heads
    dK_expanded = torch.matmul(dS.transpose(-2, -1), Q) * scale  # (B, h_q, N, D)
    dK = torch.zeros_like(K)
    for i in range(h_kv):
        start_idx = i * group_size
        end_idx = (i + 1) * group_size 
        dK[:, i, :, :] = dK_expanded[:, start_idx:end_idx, :, :].sum(dim=1)

    return dQ, dK, dV, Delta

# **************************************************
# Generate inputs
# **************************************************


causal = False
b = 16
h_q = 64  # number of query heads  
h_kv = 8  # number of key/value heads (for GQA)
group_size = h_q // h_kv  # queries per KV head group
n = 16384
d = 128
dtype = torch.bfloat16
mean = 10
std = 0.1  

flops_ref = flops(b, n, h_q, d, causal, mode="bwd")  # Use query heads for FLOP calculation

def generate_tensor(shape, mean, std, dtype, device):
    tensor = torch.randn(shape, dtype=dtype, device=device)
    magnitude = torch.norm(tensor, dim=-1, keepdim=True)
    scaled_tensor = tensor * (torch.randn(magnitude.shape, dtype=dtype, device=device) * std + mean) / magnitude
    return scaled_tensor.contiguous()

def generate_inputs():
    # Generate in BHND format (batch, heads, seq, dim) for GQA
    # Q has h_q heads, but K and V have h_kv heads
    Q = generate_tensor((b, h_q, n, d), mean, std, torch.bfloat16, 'cuda')
    K = generate_tensor((b, h_kv, n, d), mean, std, torch.bfloat16, 'cuda') 
    V = generate_tensor((b, h_kv, n, d), mean, std, torch.bfloat16, 'cuda')
    dO = generate_tensor((b, h_q, n, d), mean, std, torch.bfloat16, 'cuda') 

    Q.requires_grad_(True)
    K.requires_grad_(True)
    V.requires_grad_(True)
    return Q, K, V, dO

# Generate base inputs in BHND format
Q_bhnd, K_bhnd, V_bhnd, dO_bhnd = generate_inputs()

# # **************************************************
# # Pytorch forward and backward
# # **************************************************

# for _ in range(num_warmup):
#     q_pt = Q_bhnd.detach().requires_grad_(True)
#     k_pt = K_bhnd.detach().requires_grad_(True)
#     v_pt = V_bhnd.detach().requires_grad_(True)
#     out_pt = torch.nn.functional.scaled_dot_product_attention(
#                 q_pt, k_pt, v_pt, attn_mask=None, dropout_p=0.0, is_causal=causal)
#     out_pt.backward(dO_bhnd)

# timings_pt = []
# for _ in range(num_iters):
#     q_pt = Q_bhnd.detach().requires_grad_(True)
#     k_pt = K_bhnd.detach().requires_grad_(True)
#     v_pt = V_bhnd.detach().requires_grad_(True)
#     out_pt = torch.nn.functional.scaled_dot_product_attention(
#                 q_pt, k_pt, v_pt, attn_mask=None, dropout_p=0.0, is_causal=causal)
#     torch.cuda.synchronize()
#     start_event.record()
#     out_pt.backward(dO_bhnd)
#     end_event.record()
#     torch.cuda.synchronize()
#     elapsed_time = start_event.elapsed_time(end_event)
#     timings_pt.append(elapsed_time)

# avg_time_pt = sum(timings_pt) / len(timings_pt)
# eff_pt = efficiency(flops_ref, avg_time_pt)
# print(f"PyTorch reference average execution time: {avg_time_pt:.4f} ms")
# print(f"PyTorch reference performance: {eff_pt:.2f} TFLOPS for {b=} h_q={h_q} h_kv={h_kv} {n=} {d=} {causal=}.\n")

# q_grad_pt = q_pt.grad
# k_grad_pt = k_pt.grad
# v_grad_pt = v_pt.grad

# **************************************************
# AITER forward and backward
# **************************************************

if use_aiter:
    timings = []
    print("\nRunning AITER...")

    for _ in range(num_warmup):
        Q_aiter = Q_bhnd.transpose(1, 2).contiguous().detach().requires_grad_(True)  
        K_aiter = K_bhnd.transpose(1, 2).contiguous().detach().requires_grad_(True)  
        V_aiter = V_bhnd.transpose(1, 2).contiguous().detach().requires_grad_(True)  
        dO_aiter = dO_bhnd.transpose(1, 2).contiguous()
        out_aiter, softmax_lse = aiter.flash_attn_func(Q_aiter, K_aiter, V_aiter, causal=causal, return_lse=True, deterministic=False)
        out_aiter.backward(dO_aiter)
    
    for _ in range(num_iters):
        Q_aiter = Q_bhnd.transpose(1, 2).contiguous().detach().requires_grad_(True)  
        K_aiter = K_bhnd.transpose(1, 2).contiguous().detach().requires_grad_(True)  
        V_aiter = V_bhnd.transpose(1, 2).contiguous().detach().requires_grad_(True)  
        dO_aiter = dO_bhnd.transpose(1, 2).contiguous()
        out_aiter, softmax_lse = aiter.flash_attn_func(Q_aiter, K_aiter, V_aiter, causal=causal, return_lse=True, deterministic=False)
        torch.cuda.synchronize()
        start_event.record()
        out_aiter.backward(dO_aiter)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        timings.append(elapsed_time)

    avg_time_aiter = sum(timings) / len(timings)
    eff_aiter = efficiency(flops_ref, avg_time_aiter)
    print(f"AITER (AMD) reference average execution time: {avg_time_aiter:.4f} ms")
    print(f"AITER (AMD) reference performance: {eff_aiter:.2f} TFLOPS for {b=} h_q={h_q} h_kv={h_kv} {n=} {d=} {causal=}.\n")

    q_grad_aiter_bnhd = Q_aiter.grad
    k_grad_aiter_bnhd = K_aiter.grad  
    v_grad_aiter_bnhd = V_aiter.grad
    out_aiter_bnhd = out_aiter
    out_aiter_bhnd = out_aiter.transpose(1, 2)  # BNHD -> BHND
    q_grad_aiter_bhnd = q_grad_aiter_bnhd.transpose(1, 2)  # BNHD -> BHND
    k_grad_aiter_bhnd = k_grad_aiter_bnhd.transpose(1, 2)  # BNHD -> BHND
    v_grad_aiter_bhnd = v_grad_aiter_bnhd.transpose(1, 2)  # BNHD -> BHND


# # # **************************************************
# # Comparisons
# # **************************************************

# num_print = 8

# # TK vs AITER
# print(f"\nTK vs AITER comparison:")
# print("\nO outputs:")
# print("PyTorch: ", out_pt[0, 0, :num_print, 0], "Max:", out_pt.max().item())
# print("AITER: ", out_aiter_bhnd[0, 0, :num_print, 0], "Max:", out_aiter_bhnd.max().item())

# print()
# print("\nGradient K outputs:")
# print("PyTorch: ", k_grad_pt[0, 0, 0, :num_print], "Max:", k_grad_pt.max().item())
# print("AITER: ", k_grad_aiter_bhnd[0, 0, 0, :num_print], "Max:", k_grad_aiter_bhnd.max().item())

# print()
# print("Gradient V outputs:")
# print("PyTorch: ", v_grad_pt[0, 0, 0, :num_print], "Max:", v_grad_pt.max().item())
# print("AITER: ", v_grad_aiter_bhnd[0, 0, 0, :num_print], "Max:", v_grad_aiter_bhnd.max().item())

# print()
# print("Gradient Q outputs:")
# print("PyTorch: ", q_grad_pt[0, 0, 0, :num_print], "Max:", q_grad_pt.max().item())
# print("AITER: ", q_grad_aiter_bhnd[0, 0, 0, :num_print], "Max:", q_grad_aiter_bhnd.max().item())
# # print("Diff: ", (dQ_tk - q_grad_aiter_bnhd)[0, :, 0, 32:48], "Max:", (dQ_tk - q_grad_aiter_bnhd).max().item())


# # **************************************************
# # TK vs AITER (robust tolerances & metrics)
# # **************************************************
# # Compare O and L with AITER
# print(f"\nRobustness checks (TK vs AITER):") 
# o_diff, o_err_cnt, o_total, o_rel_error, o_l2_error, o_cos, o_mask = robustness_check(out_pt, out_aiter_bhnd)
# print(f"O: max_abs={o_diff.max().item():.6f}, max_rel={o_rel_error:.4f}, "
#       f"rel_l2={o_l2_error:.4f}, cos={o_cos:.6f}, "
#       f"errors={o_err_cnt}/{o_total} ({100*o_err_cnt/o_total:.4f}%)")

# # **************************************************
# # TK vs AITER (gradient comparisons)
# # **************************************************
# print(f"\nGradient comparisons (TK vs AITER):") 

# # Compute diffs in float32 to avoid bf16 quantization in the comparison itself
# q_diff, q_err_cnt, q_total, q_rel_error, q_l2_error, q_cos, q_mask = robustness_check(q_grad_pt, q_grad_aiter_bhnd)
# k_diff, k_err_cnt, k_total, k_rel_error, k_l2_error, k_cos, k_mask = robustness_check(k_grad_pt, k_grad_aiter_bhnd)
# v_diff, v_err_cnt, v_total, v_rel_error, v_l2_error, v_cos, v_mask = robustness_check(v_grad_pt, v_grad_aiter_bhnd)

# print(f"Q grad: max_abs={q_diff.max().item():.6f}, max_rel={q_rel_error:.4f}, "
#         f"rel_l2={q_l2_error:.4f}, cos={q_cos:.6f}, "
#       f"errors={q_err_cnt}/{q_total} ({100*q_err_cnt/q_total:.4f}%)")
# print(f"K grad: max_abs={k_diff.max().item():.6f}, max_rel={k_rel_error:.4f}, "
#       f"rel_l2={k_l2_error:.4f}, cos={k_cos:.6f}, "
#       f"errors={k_err_cnt}/{k_total} ({100*k_err_cnt/k_total:.4f}%)")
# print(f"V grad: max_abs={v_diff.max().item():.6f}, max_rel={v_rel_error:.4f}, "
#       f"rel_l2={v_l2_error:.4f}, cos={v_cos:.6f}, "
#       f"errors={v_err_cnt}/{v_total} ({100*v_err_cnt/v_total:.4f}%)")