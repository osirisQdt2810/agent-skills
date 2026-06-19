# Note that this kernel is not optimized
# non-causal

# GQA
# N = 1024
make SRC=attn_bkwd_non_causal_4_warps.cpp TARGET=tk_kernel_bkwd ATTN_B=16 ATTN_H=64 ATTN_H_KV=8 ATTN_N=1024
make SRC=attn_bkwd_prep.cpp TARGET=tk_kernel_bkwd_prep ATTN_B=16 ATTN_H=64 ATTN_H_KV=8 ATTN_N=1024
make SRC=attn_fwd_non_causal.cpp TARGET=tk_kernel_fwd ATTN_B=16 ATTN_H=64 ATTN_H_KV=8 ATTN_N=1024
python test_python.py 16 1024 64 8 0 mi355x_gqa_bkwd_non_causal_4warps.json

# N = 2048
make SRC=attn_bkwd_non_causal_4_warps.cpp TARGET=tk_kernel_bkwd ATTN_B=16 ATTN_H=64 ATTN_H_KV=8 ATTN_N=2048
make SRC=attn_bkwd_prep.cpp TARGET=tk_kernel_bkwd_prep ATTN_B=16 ATTN_H=64 ATTN_H_KV=8 ATTN_N=2048
make SRC=attn_fwd_non_causal.cpp TARGET=tk_kernel_fwd ATTN_B=16 ATTN_H=64 ATTN_H_KV=8 ATTN_N=2048
python test_python.py 16 2048 64 8 0 mi355x_gqa_bkwd_non_causal_4warps.json

# N = 4096
make SRC=attn_bkwd_non_causal_4_warps.cpp TARGET=tk_kernel_bkwd ATTN_B=16 ATTN_H=64 ATTN_H_KV=8 ATTN_N=4096
make SRC=attn_bkwd_prep.cpp TARGET=tk_kernel_bkwd_prep ATTN_B=16 ATTN_H=64 ATTN_H_KV=8 ATTN_N=4096
make SRC=attn_fwd_non_causal.cpp TARGET=tk_kernel_fwd ATTN_B=16 ATTN_H=64 ATTN_H_KV=8 ATTN_N=4096
python test_python.py 16 4096 64 8 0 mi355x_gqa_bkwd_non_causal_4warps.json

# N = 8192
make SRC=attn_bkwd_non_causal_4_warps.cpp TARGET=tk_kernel_bkwd ATTN_B=16 ATTN_H=64 ATTN_H_KV=8 ATTN_N=8192
make SRC=attn_bkwd_prep.cpp TARGET=tk_kernel_bkwd_prep ATTN_B=16 ATTN_H=64 ATTN_H_KV=8 ATTN_N=8192
make SRC=attn_fwd_non_causal.cpp TARGET=tk_kernel_fwd ATTN_B=16 ATTN_H=64 ATTN_H_KV=8 ATTN_N=8192
python test_python.py 16 8192 64 8 0 mi355x_gqa_bkwd_non_causal_4warps.json

# N = 16384
make SRC=attn_bkwd_non_causal_4_warps.cpp TARGET=tk_kernel_bkwd ATTN_B=15 ATTN_H=64 ATTN_H_KV=8 ATTN_N=16384
make SRC=attn_bkwd_prep.cpp TARGET=tk_kernel_bkwd_prep ATTN_B=15 ATTN_H=64 ATTN_H_KV=8 ATTN_N=16384
make SRC=attn_fwd_non_causal.cpp TARGET=tk_kernel_fwd ATTN_B=15 ATTN_H=64 ATTN_H_KV=8 ATTN_N=16384
python test_python.py 15 16384 64 8 0 mi355x_gqa_bkwd_non_causal_4warps.json

# MHA
# N = 1024
make SRC=attn_bkwd_non_causal_4_warps.cpp TARGET=tk_kernel_bkwd ATTN_B=16 ATTN_H=16 ATTN_H_KV=16 ATTN_N=1024
make SRC=attn_bkwd_prep.cpp TARGET=tk_kernel_bkwd_prep ATTN_B=16 ATTN_H=16 ATTN_H_KV=16 ATTN_N=1024
make SRC=attn_fwd_non_causal.cpp TARGET=tk_kernel_fwd ATTN_B=16 ATTN_H=16 ATTN_H_KV=16 ATTN_N=1024
python test_python.py 16 1024 16 16 0 mi355x_mha_bkwd_non_causal_4warps.json

# N = 2048
make SRC=attn_bkwd_non_causal_4_warps.cpp TARGET=tk_kernel_bkwd ATTN_B=16 ATTN_H=16 ATTN_H_KV=16 ATTN_N=2048
make SRC=attn_bkwd_prep.cpp TARGET=tk_kernel_bkwd_prep ATTN_B=16 ATTN_H=16 ATTN_H_KV=16 ATTN_N=2048
make SRC=attn_fwd_non_causal.cpp TARGET=tk_kernel_fwd ATTN_B=16 ATTN_H=16 ATTN_H_KV=16 ATTN_N=2048
python test_python.py 16 2048 16 16 0 mi355x_mha_bkwd_non_causal_4warps.json

# N = 4096
make SRC=attn_bkwd_non_causal_4_warps.cpp TARGET=tk_kernel_bkwd ATTN_B=16 ATTN_H=16 ATTN_H_KV=16 ATTN_N=4096
make SRC=attn_bkwd_prep.cpp TARGET=tk_kernel_bkwd_prep ATTN_B=16 ATTN_H=16 ATTN_H_KV=16 ATTN_N=4096
make SRC=attn_fwd_non_causal.cpp TARGET=tk_kernel_fwd ATTN_B=16 ATTN_H=16 ATTN_H_KV=16 ATTN_N=4096
python test_python.py 16 4096 16 16 0 mi355x_mha_bkwd_non_causal_4warps.json

# N = 8192
make SRC=attn_bkwd_non_causal_4_warps.cpp TARGET=tk_kernel_bkwd ATTN_B=16 ATTN_H=16 ATTN_H_KV=16 ATTN_N=8192
make SRC=attn_bkwd_prep.cpp TARGET=tk_kernel_bkwd_prep ATTN_B=16 ATTN_H=16 ATTN_H_KV=16 ATTN_N=8192
make SRC=attn_fwd_non_causal.cpp TARGET=tk_kernel_fwd ATTN_B=16 ATTN_H=16 ATTN_H_KV=16 ATTN_N=8192
python test_python.py 16 8192 16 16 0 mi355x_mha_bkwd_non_causal_4warps.json

# N = 16384
make SRC=attn_bkwd_non_causal_4_warps.cpp TARGET=tk_kernel_bkwd ATTN_B=16 ATTN_H=16 ATTN_H_KV=16 ATTN_N=16384
make SRC=attn_bkwd_prep.cpp TARGET=tk_kernel_bkwd_prep ATTN_B=16 ATTN_H=16 ATTN_H_KV=16 ATTN_N=16384
make SRC=attn_fwd_non_causal.cpp TARGET=tk_kernel_fwd ATTN_B=16 ATTN_H=16 ATTN_H_KV=16 ATTN_N=16384
python test_python.py 16 16384 16 16 0 mi355x_mha_bkwd_non_causal_4warps.json