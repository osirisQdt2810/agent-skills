#!/usr/bin/env python3
"""
rocprofv2 deep-metric path (Task 2): multi-line counter sets -> per-pmc CSVs ->
merged-and-filtered table -> shared profile digest.

Why this exists
---------------
rocprofv2 collects at most one *pass* per `pmc:` line, and packing too many
counters onto a single line silently yields null values. So the counter set
(profiling/counters/default.txt) splits counters across MULTIPLE `pmc:` lines;
rocprofv2 then re-runs the app once per line and emits one CSV per line under
`<outdir>/pmc_<i>/`. Those per-pmc CSVs must be merged (matched by Dispatch_ID)
into one wide table and then FILTERED to the exact kernel symbol(s) under test.

The catch: the kernel's mangled symbol name is unknown up front. So the flow is:
    1. dry run  -> discover the kernel symbol name(s)   (discover_symbols)
    2. profile  -> collect the multi-line counter set    (run_rocprofv2)
    3. merge    -> combine per-pmc CSVs + filter to symbol(s)  (merge)
    4. digest   -> average -> canonical digest            (to_digest)

Generic over any kernel — the symbol substring and kernel-only command come from
the calibrated problem, never assumed.

CLI (invoke by path; works from any cwd — the script bootstraps its own import):
    SKILL="${CLAUDE_SKILL_DIR}"   # the optimize-kernels skill dir; expanded by Claude Code

    # discover symbols from any already-collected run (or a 1-line dry run)
    python3 $SKILL/profiling/merge_v2.py --discover --profile-dir <dir>

    # collect + merge + digest in one shot
    python3 $SKILL/profiling/merge_v2.py --counters $SKILL/profiling/counters/default.txt \\
        --profile-dir artifacts/<problem>/profile/v2 --kernel-filter <symbol-substr> \\
        --run -- <kernel-only harness command...>
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from profiling import profile_digest
else:
    from . import profile_digest

COMMON_COLS = [
    "Dispatch_ID", "GPU_ID", "Queue_ID", "PID", "TID",
    "Grid_Size", "Workgroup_Size", "LDS_Per_Workgroup", "Scratch_Per_Workitem",
    "Arch_VGPR", "Accum_VGPR", "SGPR", "Wave_Size", "Kernel_Name",
    "Start_Timestamp", "End_Timestamp", "Correlation_ID",
]


def parse_counters(counters_file):
    """Return list of counter-name lists, one per `pmc:` line. Accepts both
    comma- and space-separated counter lists (metrics/*.txt uses both)."""
    pmcs = []
    with open(counters_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or not line.lower().startswith("pmc:"):
                continue
            body = line[4:].replace(",", " ")
            counters = [c.strip() for c in body.split() if c.strip()]
            if counters:
                pmcs.append(counters)
    return pmcs


def _find_csv(directory):
    files = sorted(glob.glob(os.path.join(directory, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV found in {directory}")
    return files[0]


def _pmc_dirs(profile_dir):
    """Return sorted [(i, dir), ...] for pmc_<i> subdirs, or [(1, profile_dir)]
    if the CSV sits directly in profile_dir (single-pass run)."""
    dirs = []
    for d in sorted(glob.glob(os.path.join(profile_dir, "pmc_*"))):
        m = re.search(r"pmc_(\d+)$", d)
        if m and os.path.isdir(d):
            dirs.append((int(m.group(1)), d))
    if not dirs and glob.glob(os.path.join(profile_dir, "*.csv")):
        dirs = [(1, profile_dir)]
    return sorted(dirs)


def discover_symbols(profile_dir):
    """Read distinct Kernel_Name values from whatever CSV(s) exist under
    profile_dir. Use after a dry run to learn the mangled symbol name(s)."""
    names = set()
    for _, d in _pmc_dirs(profile_dir):
        try:
            df = pd.read_csv(_find_csv(d))
        except (FileNotFoundError, pd.errors.EmptyDataError):
            continue
        if "Kernel_Name" in df.columns:
            names.update(str(n) for n in df["Kernel_Name"].dropna().unique())
    return sorted(names)


def run_rocprofv2(counters_file, outdir, cmd):
    """Invoke rocprofv2 with a multi-line counter file. `cmd` is the harness
    argv list (e.g. [harness_bin, 'profile', M, N, K, ITERS])."""
    os.makedirs(outdir, exist_ok=True)
    full = ["rocprofv2", "-i", counters_file, "-d", outdir] + [str(c) for c in cmd]
    print("+ " + " ".join(full))
    subprocess.run(full, check=True)


def merge(profile_dir, counters_file, kernel_filter=None):
    """Merge all per-pmc CSVs under profile_dir into one DataFrame, optionally
    filtered to kernel names containing any of `kernel_filter` substrings."""
    pmcs = parse_counters(counters_file)
    pmc_dirs = _pmc_dirs(profile_dir)
    if not pmc_dirs:
        raise RuntimeError(f"No pmc_* CSVs found under {profile_dir}")

    frames = []
    for (i, d) in pmc_dirs:
        df = pd.read_csv(_find_csv(d))
        counters = pmcs[i - 1] if i - 1 < len(pmcs) else []
        if i == 1:
            if {"Start_Timestamp", "End_Timestamp"}.issubset(df.columns):
                df["Duration(us)"] = (
                    df["End_Timestamp"].astype("int64")
                    - df["Start_Timestamp"].astype("int64")
                ) / 1000.0
        frames.append((i, counters, df))

    base = frames[0][2]
    keep = [c for c in COMMON_COLS if c in base.columns]
    if "Duration(us)" in base.columns:
        keep.append("Duration(us)")
    keep += [c for c in frames[0][1] if c in base.columns]
    result = base[keep].copy()

    for (i, counters, df) in frames[1:]:
        cols = [c for c in counters if c in df.columns]
        if "Dispatch_ID" in df.columns:
            result = result.merge(
                df[["Dispatch_ID"] + cols], on="Dispatch_ID", how="left",
                suffixes=("", f"_pmc{i}"),
            )

    if kernel_filter:
        pat = "|".join(re.escape(p) for p in kernel_filter)
        result = result[result["Kernel_Name"].astype(str).str.contains(pat, na=False)]
    return result


def to_digest(df):
    """Convert a merged (filtered) DataFrame into the shared profile digest."""
    return profile_digest.summarize(df.to_dict(orient="records"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile-dir", required=True)
    ap.add_argument("--counters", help="multi-line pmc counter file (metrics/*.txt)")
    ap.add_argument("--kernel-filter", nargs="*", default=None,
                    metavar="SUBSTR", help="keep kernels whose name contains any of these")
    ap.add_argument("--discover", action="store_true",
                    help="just print distinct kernel symbol names found, then exit")
    ap.add_argument("--run", action="store_true",
                    help="invoke rocprofv2 first (everything after -- is the harness argv)")
    ap.add_argument("--out", help="write digest json here (default: <profile-dir>/digest.json)")
    ap.add_argument("cmd", nargs=argparse.REMAINDER,
                    help="-- <harness argv> when --run is given")
    args = ap.parse_args(argv)

    if args.discover:
        for n in discover_symbols(args.profile_dir):
            print(n)
        return 0

    if args.run:
        cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
        if not args.counters or not cmd:
            ap.error("--run requires --counters and -- <harness argv>")
        run_rocprofv2(args.counters, args.profile_dir, cmd)

    if not args.counters:
        ap.error("--counters is required (unless --discover)")

    df = merge(args.profile_dir, args.counters, args.kernel_filter)
    print(f"merged {len(df)} rows"
          + (f" (filtered to {args.kernel_filter})" if args.kernel_filter else ""))
    digest = to_digest(df)
    profile_digest.print_summary(digest)

    out = args.out or os.path.join(args.profile_dir, "digest.json")
    with open(out, "w") as f:
        json.dump(digest, f, indent=2)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
