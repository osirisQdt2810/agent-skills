---
name: fp8-mqa-logits-optimization
description: The fp8_paged_mqa_logits (Heads=64) kernel optimization run and its difficulty
metadata:
  type: project
---

`/optimize-kernels` run started 2026-06-19 on `src/fp8_paged_mqa_logits` (GLM-5-FP8 decode
indexer logits kernel, Heads=64, `csrc/fp8_paged_mqa_logits_h64.cpp`).

- **Target:** mean `speedup = deepgemm_us/moreh_us` at avg_kv_length=64000 (batch {48,60,75},
  next_n=1) must reach **1.2x**. Baseline (deepgemm = aiter Triton).
- **Hard finding at calibrate:** baseline kernel is **0.71x** (slower than deepgemm); runtime
  autotune (num_warps×ChunkK×SplitKV) ceiling is only **~0.77x**. Reaching 1.2x needs
  algorithmic rework (~1.6–1.7x faster), not tuning. Profile: ~2.35 TB/s effective HBM BW
  (~44% of MI300X peak), memory-latency/occupancy bound; deepgemm hits ~3.07 TB/s and uses a
  preshuffled KV layout vs moreh's LDS-staged K.
- All calibrate artifacts under `artifacts/fp8_paged_mqa_logits/`; resume via
  `/optimize-kernels --resume`. Details: `artifacts/fp8_paged_mqa_logits/fp8_paged_mqa_logits.context.md`.

Skill bug fixed during this run: see [[skill-bug-handling]] and repo-root `SKILL_ISSUES.md`.
