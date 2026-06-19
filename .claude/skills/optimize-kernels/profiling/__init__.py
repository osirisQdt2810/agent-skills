"""Reusable GPU-profiling helpers for the optimize-kernels goal loop.

These are generic over the kernel under test; the runner's per-problem
profile.py / harness.py (generated into artifacts/<problem>/) drive them.

Modules:
  profile_digest   - canonical digest shape shared by both profiler paths
  merge_v2         - rocprofv2 multi-line counter sets -> merged CSV -> digest
  rocprofv3_digest - rocprofv3 counter CSV (+ ATT pointer) -> digest
  autotune         - generic parameter-sweep helper for the generated harness.py
"""
