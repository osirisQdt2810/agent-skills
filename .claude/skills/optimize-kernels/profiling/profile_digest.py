#!/usr/bin/env python3
"""
Shared profile digest — the ONE canonical shape that both the rocprofv2 and the
rocprofv3 profiling paths emit, so the goal loop reads the same dict regardless of
which profiler produced it.

A "digest" is built from a list of per-dispatch records (one dict per kernel
dispatch, already filtered to the kernel(s) under test). Counter names are
normalized to a small canonical set; whatever is present is averaged over
dispatches (the first dispatch is dropped as warmup), a few derived metrics are
computed, and a coarse bottleneck hint is attached.

Used by:
  - profiling/merge_v2.py        (rocprofv2 multi-line -> merged CSV -> digest)
  - profiling/rocprofv3_digest.py(rocprofv3 counter CSV -> digest)

The digest dict shape (all keys optional except `n_dispatches`/`kernels`):

  {
    "n_dispatches": int,            # dispatches averaged (after warmup drop)
    "kernels": [str, ...],          # distinct kernel symbol names seen
    "duration_us": float,           # mean per-dispatch time
    "memory_bandwidth_GBps": float, # derived from FetchSize+WriteSize
    "metrics": { <canonical>: float, ... },   # averaged raw counters
    "bottleneck": str,              # coarse heuristic label
    "notes": [str, ...],            # human-readable hints
  }
"""

from __future__ import annotations

# Map many possible source column names (rocprofv2 derived metrics, rocprofv3
# raw HW counters) onto one canonical counter name. Lower-cased lookup.
CANONICAL = {
    # global memory traffic (KB in rocprofv2 FetchSize/WriteSize)
    "fetchsize": "FetchSize_KB",
    "writesize": "WriteSize_KB",
    # busy / utilization (percent 0..100 in rocprofv2 derived metrics)
    "valubusy": "VALUBusy",
    "salubusy": "SALUBusy",
    "valuutilization": "VALUUtilization",
    "memunitbusy": "MemUnitBusy",
    "memunitstalled": "MemUnitStalled",
    "writeunitstalled": "WriteUnitStalled",
    "alustalledbylds": "ALUStalledByLDS",
    # cache
    "l2cachehitrate": "L2CacheHitRate",
    "sl1dcachehitrate": "sL1dCacheHitRate",
    # lds
    "ldsbankconflict": "LDSBankConflict",
    # occupancy-ish raw counters
    "wavefronts": "Wavefronts",
    "sq_level_waves": "SQ_LEVEL_WAVES",
    "grbm_gui_active": "GRBM_GUI_ACTIVE",
    "gpubusy": "GPUBusy",
}

# Timestamp column aliases (ns).
START_KEYS = ("start_timestamp", "start", "begin")
END_KEYS = ("end_timestamp", "end", "stop")
KERNEL_KEYS = ("kernel_name", "kernelname", "name", "kernel")


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def normalize_record(raw: dict) -> dict:
    """Lower-case-key a raw CSV row into {canonical_counter: float, ...} plus
    bookkeeping fields _start_ns / _end_ns / _kernel."""
    low = {str(k).strip().lower(): v for k, v in raw.items()}
    out = {}
    for src, val in low.items():
        canon = CANONICAL.get(src)
        if canon is None:
            continue
        f = _to_float(val)
        if f is not None:
            out[canon] = f
    # timestamps
    for k in START_KEYS:
        if k in low and _to_float(low[k]) is not None:
            out["_start_ns"] = _to_float(low[k])
            break
    for k in END_KEYS:
        if k in low and _to_float(low[k]) is not None:
            out["_end_ns"] = _to_float(low[k])
            break
    # if a duration column already exists (rocprofv2 merge adds Duration(us))
    if "duration(us)" in low and _to_float(low["duration(us)"]) is not None:
        out["_duration_us"] = _to_float(low["duration(us)"])
    for k in KERNEL_KEYS:
        if k in low and low[k] not in (None, ""):
            out["_kernel"] = str(low[k])
            break
    return out


def _dispatch_duration_us(rec: dict):
    if "_duration_us" in rec:
        return rec["_duration_us"]
    if "_start_ns" in rec and "_end_ns" in rec:
        return (rec["_end_ns"] - rec["_start_ns"]) / 1000.0
    return None


def summarize(records, drop_warmup=True) -> dict:
    """Average normalized per-dispatch records into a digest."""
    recs = [normalize_record(r) for r in records]
    recs = [r for r in recs if r]  # keep non-empty
    if drop_warmup and len(recs) > 1:
        recs = recs[1:]
    if not recs:
        return {"n_dispatches": 0, "kernels": [], "metrics": {},
                "bottleneck": "unknown", "notes": ["no dispatches parsed"]}

    kernels = sorted({r["_kernel"] for r in recs if "_kernel" in r})

    # average each canonical counter present
    metrics = {}
    counter_keys = {k for r in recs for k in r if not k.startswith("_")}
    for k in counter_keys:
        vals = [r[k] for r in recs if k in r]
        if vals:
            metrics[k] = sum(vals) / len(vals)

    durations = [d for r in recs if (d := _dispatch_duration_us(r)) is not None]
    duration_us = sum(durations) / len(durations) if durations else None

    digest = {
        "n_dispatches": len(recs),
        "kernels": kernels,
        "metrics": metrics,
    }
    if duration_us is not None:
        digest["duration_us"] = duration_us

    # derived: memory bandwidth (FetchSize/WriteSize are KB; *1.024 / us -> GB/s)
    fetch = metrics.get("FetchSize_KB")
    write = metrics.get("WriteSize_KB")
    if duration_us and (fetch is not None or write is not None):
        total_kb = (fetch or 0.0) + (write or 0.0)
        digest["memory_bandwidth_GBps"] = total_kb * 1.024 / duration_us

    digest["bottleneck"], digest["notes"] = bottleneck_hint(digest)
    return digest


def bottleneck_hint(digest: dict):
    """Coarse heuristic mirroring the runner's signal table. Returns
    (label, notes[])."""
    m = digest.get("metrics", {})
    notes = []
    label = "unknown"

    valu = m.get("VALUBusy")
    memb = m.get("MemUnitBusy")
    mems = m.get("MemUnitStalled")
    lds = m.get("LDSBankConflict")
    l2 = m.get("L2CacheHitRate")

    if lds and lds > 0:
        notes.append(f"LDSBankConflict={lds:.2f} > 0 -> pad shared tiles (+1 column).")
    if l2 is not None and l2 < 50:
        notes.append(f"L2CacheHitRate={l2:.1f}% low -> improve locality / blocking.")

    if memb is not None and mems is not None and memb > 60 and mems > 20:
        label = "bandwidth-bound"
        notes.append("High MemUnitBusy+MemUnitStalled -> cut global traffic, "
                     "coalesce, vectorize (float4).")
    elif valu is not None and valu < 40:
        label = "latency-bound"
        notes.append("Low VALUBusy -> improve ILP / reuse / occupancy.")
    elif valu is not None and valu >= 60:
        label = "compute-bound"
        notes.append("High VALUBusy -> consider MFMA / better math mapping.")

    if "memory_bandwidth_GBps" in digest:
        notes.append(f"Memory bandwidth ~= {digest['memory_bandwidth_GBps']:.1f} GB/s "
                     "(MI300X HBM3 peak ~5.3 TB/s).")
    return label, notes


def print_summary(digest: dict):
    print("=== profile digest ===")
    print(f"kernels      : {', '.join(digest.get('kernels') or ['?'])}")
    print(f"dispatches   : {digest.get('n_dispatches')}")
    if "duration_us" in digest:
        print(f"duration     : {digest['duration_us']:.3f} us/dispatch")
    if "memory_bandwidth_GBps" in digest:
        print(f"bandwidth    : {digest['memory_bandwidth_GBps']:.1f} GB/s")
    print(f"bottleneck   : {digest.get('bottleneck')}")
    m = digest.get("metrics", {})
    for k in sorted(m):
        print(f"  {k:<20} {m[k]:.4f}")
    for n in digest.get("notes", []):
        print(f"  - {n}")
