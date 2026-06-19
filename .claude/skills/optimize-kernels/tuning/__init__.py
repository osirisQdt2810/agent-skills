"""Parameter-tuning helpers for the optimize-kernels goal loop.

Separate concern from `profiling/` (which collects/digests hardware counters).
The runner's per-problem harness.py (generated into artifacts/<problem>/) drives
these to search a config space, correctness-gated.

Modules:
  autotune - generic config-space sweep (grid/random) for the generated harness.py
"""
