#!/usr/bin/env python3
"""
rocprofv3 profiling path (Task 1): parse rocprofv3 output into the SAME shared
profile digest the rocprofv2 path emits, so the goal loop reads one shape.

rocprofv3 (1.1.0) advantages over v2 used here:
  * `--kernel-include-regex <re>`  -> filter to the kernel natively at collection
    time (no manual merge+filter step needed).
  * `-i counters.txt` / `--pmc ...` with `-f csv` -> hardware counters. As with
    v2, a single pass can only hold counters that fit one collection group, so a
    multi-line `-i` file (one group per line) gives multi-pass collection.
  * `--att --att-activity <N>` -> Advanced Thread Trace for instruction-level
    stall/latency analysis (decoded into a stats CSV by the ATT plugin).

This module reads:
  * the counter CSV (long format: one row per (dispatch, counter); pivoted to
    wide here), OR a wide CSV if that is what was emitted, and
  * an optional ATT stats CSV, surfacing the top stall reasons as notes.

Generic over any kernel — the kernel regex and kernel-only command come from the
calibrated problem, never assumed.

CLI (invoke by path; works from any cwd — the script bootstraps its own import):
    SKILL="${CLAUDE_SKILL_DIR}"   # the optimize-kernels skill dir; expanded by Claude Code

    # collect counters (multi-pass) for one kernel, then digest
    python3 $SKILL/profiling/rocprofv3_digest.py \\
        --counters $SKILL/profiling/counters/default.txt \\
        --out-dir artifacts/<problem>/profile/v3 --kernel-regex '<kernel-name-regex>' \\
        --run -- <kernel-only harness command...>

    # digest an already-collected run (counters and/or ATT)
    python3 $SKILL/profiling/rocprofv3_digest.py --out-dir artifacts/<problem>/profile/v3
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from profiling import profile_digest
else:
    from . import profile_digest

# long-format column aliases rocprofv3 uses across point releases
DISPATCH_KEYS = ("dispatch_id", "dispatch_index")
CNAME_KEYS = ("counter_name", "name")
CVAL_KEYS = ("counter_value", "value")


def _lower(row):
    return {str(k).strip().lower(): v for k, v in row.items()}


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def find_counter_csv(out_dir):
    """Locate the rocprofv3 counter CSV anywhere under out_dir."""
    cands = glob.glob(os.path.join(out_dir, "**", "*counter_collection*.csv"), recursive=True)
    if not cands:
        # any csv that has a Counter_Name column
        for p in glob.glob(os.path.join(out_dir, "**", "*.csv"), recursive=True):
            try:
                rows = _read_csv(p)
            except OSError:
                continue
            if rows and any(k in _lower(rows[0]) for k in CNAME_KEYS):
                cands.append(p)
    return sorted(cands)


def find_att_stats_csv(out_dir):
    """Locate an ATT stall/stats CSV if the ATT plugin produced one."""
    pats = ["*att*stats*.csv", "*stall*.csv", "*att*.csv"]
    found = []
    for pat in pats:
        found += glob.glob(os.path.join(out_dir, "**", pat), recursive=True)
    return sorted(set(found))


def _is_long_format(rows):
    if not rows:
        return False
    low = _lower(rows[0])
    return any(k in low for k in CNAME_KEYS) and any(k in low for k in CVAL_KEYS)


def pivot_long(rows):
    """Pivot long-format rows (one row per dispatch x counter) into one wide
    record per dispatch: {Kernel_Name, Start_Timestamp, ..., <counter>: val}."""
    by_dispatch = {}
    order = []
    for raw in rows:
        low = _lower(raw)
        did = next((low[k] for k in DISPATCH_KEYS if k in low), None)
        if did is None:
            did = len(order)  # fall back to row order
        if did not in by_dispatch:
            by_dispatch[did] = {}
            order.append(did)
            # carry per-dispatch metadata columns (everything that isn't the
            # counter name/value pair)
            for k, v in raw.items():
                lk = str(k).strip().lower()
                if lk in CNAME_KEYS or lk in CVAL_KEYS:
                    continue
                by_dispatch[did][k] = v
        cname = next((low[k] for k in CNAME_KEYS if k in low), None)
        cval = next((low[k] for k in CVAL_KEYS if k in low), None)
        if cname is not None:
            by_dispatch[did][cname] = cval
    return [by_dispatch[d] for d in order]


def parse_counters(out_dir, kernel_substr=None):
    """Return per-dispatch wide records from the rocprofv3 counter CSV."""
    csvs = find_counter_csv(out_dir)
    if not csvs:
        return []
    rows = []
    for p in csvs:
        rows += _read_csv(p)
    records = pivot_long(rows) if _is_long_format(rows) else rows
    if kernel_substr:
        records = [r for r in records
                   if kernel_substr in str(_lower(r).get("kernel_name", ""))]
    return records


def att_notes(out_dir):
    """Best-effort ATT summary: surface top stall reasons if a stats CSV exists,
    else point at the raw trace for offline inspection."""
    stats = find_att_stats_csv(out_dir)
    if not stats:
        traces = (glob.glob(os.path.join(out_dir, "**", "*.att"), recursive=True)
                  or glob.glob(os.path.join(out_dir, "**", "ui"), recursive=True))
        if traces:
            return [f"ATT trace captured at {os.path.dirname(traces[0])}; "
                    "decode with the rocprof ATT viewer for instruction-level stalls."]
        return []
    notes = []
    for p in stats:
        rows = _read_csv(p)
        if not rows:
            continue
        low0 = _lower(rows[0])
        # find a stall/latency value column and a label column heuristically
        val_col = next((k for k in low0 if any(t in k for t in
                        ("stall", "cycle", "latency", "hit", "count"))), None)
        label_col = next((k for k in low0 if any(t in k for t in
                          ("name", "reason", "type", "instruction", "inst"))), None)
        if not val_col:
            continue
        def fv(r):
            try:
                return float(_lower(r).get(val_col, 0) or 0)
            except ValueError:
                return 0.0
        top = sorted(rows, key=fv, reverse=True)[:5]
        notes.append(f"ATT top by {val_col} (from {os.path.basename(p)}):")
        for r in top:
            lr = _lower(r)
            label = lr.get(label_col, "?") if label_col else "?"
            notes.append(f"    {label}: {lr.get(val_col)}")
    return notes


def run_rocprofv3(out_dir, counters=None, kernel_regex=None, att=False,
                  att_activity=8, cmd=None):
    """Invoke rocprofv3. Counters and ATT generally need separate runs."""
    os.makedirs(out_dir, exist_ok=True)
    full = ["rocprofv3", "-d", out_dir, "-f", "csv"]
    if counters:
        full += ["-i", counters]
    if kernel_regex:
        full += ["--kernel-include-regex", kernel_regex]
    if att:
        full += ["--att", "--att-activity", str(att_activity)]
    full += ["--"] + [str(c) for c in (cmd or [])]
    print("+ " + " ".join(full))
    subprocess.run(full, check=True)


def build_digest(out_dir, kernel_substr=None):
    records = parse_counters(out_dir, kernel_substr)
    digest = profile_digest.summarize(records)
    digest.setdefault("notes", []).extend(att_notes(out_dir))
    return digest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--counters", help="multi-line pmc counter file (metrics/*.txt)")
    ap.add_argument("--kernel-regex", help="--kernel-include-regex at collection time")
    ap.add_argument("--kernel-filter", help="substring to keep when digesting")
    ap.add_argument("--att", action="store_true", help="collect ATT (separate run)")
    ap.add_argument("--att-activity", type=int, default=8)
    ap.add_argument("--run", action="store_true",
                    help="invoke rocprofv3 first (everything after -- is the harness argv)")
    ap.add_argument("--out", help="digest json path (default <out-dir>/digest.json)")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)

    if args.run:
        cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
        if not cmd:
            ap.error("--run requires -- <harness argv>")
        run_rocprofv3(args.out_dir, counters=args.counters,
                      kernel_regex=args.kernel_regex, att=args.att,
                      att_activity=args.att_activity, cmd=cmd)

    digest = build_digest(args.out_dir, args.kernel_filter or args.kernel_regex)
    profile_digest.print_summary(digest)
    out = args.out or os.path.join(args.out_dir, "digest.json")
    with open(out, "w") as f:
        json.dump(digest, f, indent=2)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
