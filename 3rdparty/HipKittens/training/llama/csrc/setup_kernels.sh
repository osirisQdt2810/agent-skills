


# Batch size, heads, sequence length, head dimension need to match training configs, otherwise training will crash.
make SRC=attn_bkwd_causal_HBN.cpp TARGET=tk_kernel_bkwd ATTN_B=8 ATTN_H=16 ATTN_H_KV=16 ATTN_N=2048
make SRC=attn_bkwd_prep.cpp TARGET=tk_kernel_bkwd_prep ATTN_B=8 ATTN_H=16 ATTN_H_KV=16 ATTN_N=2048
make SRC=attn_fwd_causal.cpp TARGET=tk_kernel_fwd ATTN_B=8 ATTN_H=16 ATTN_H_KV=16 ATTN_N=2048

# Move to the attention folder. 
cp tk_kernel* ../llama/models/attentions/


