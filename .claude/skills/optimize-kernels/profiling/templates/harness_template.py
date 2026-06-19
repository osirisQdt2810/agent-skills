#!/usr/bin/env python3
"""
TEMPLATE for a per-problem `artifacts/<problem>/harness.py`.

This skill is GENERIC over any GPU kernel. Nothing below is matmul-specific —
every name, shape, build command, and config knob is a placeholder you replace
from the calibrated problem. Do NOT assume GEMM, hipBLAS, M/N/K, or any tiling
scheme; use whatever the actual source and its measurement contract dictate.

WHAT THIS IS FOR
----------------
`verify.py` is the GATE (drives the measurement harness, writes result.json,
never reimplements the metric). `harness.py` is a SEPARATE, agent-owned wrapper
whose jobs are:

  1. Standardize a check / bench / profile interface (a kernel-only profile region
     is what the profiler attaches counters to).
  2. Drive a parameter AUTOTUNE sweep over a config space (whatever knobs this
     kernel exposes: block/tile sizes, unroll factor, launch bounds, vector width,
     ...) — often the algorithm is already fine and only its config is wrong.

SOURCE-CORRECTNESS POLICY (important — read before adapting)
-----------------------------------------------------------
The ORIGINAL source can be wrong — the kernel AND its harness both. The flow is:

  * CALIBRATE (before the loop): if the original is incorrect or its measurement is
    broken, you FIX IT IN PLACE to be correct first — with the user's approval —
    so the baseline measures something real. (Fixing a genuine bug is allowed here;
    silently changing WHAT is measured is not.)
  * LOOP (autonomous): once the original is correct, you must NOT keep editing the
    original harness. `harness.py` is the agent-controlled wrapper you change
    instead, so the corrected original stays frozen and trustworthy. The ONLY
    original file the loop edits is the kernel under test (EDIT_TARGET).

So `HARNESS_REF` below points at the corrected-and-frozen original harness: read
it, compile/import it, drive it — but do not edit it during the loop.

DESIGN RULES (do not violate when adapting this template)
  * Change the original flow as LITTLE as possible. Prefer `import` over copy-paste;
    for C++/HIP, COMPILE the original harness + edit-target rather than pasting their
    bodies here. NEVER touch input-generation / ground-truth code — that keeps the
    metric honest.
  * Correctness stays the gate: a config that fails the correctness check is invalid
    and must score None so autotune never selects it.
  * harness.py may expose extra knobs (e.g. -D defines), but every number it reports
    must come from the original harness / reference, not from new math here.

Fill in every  # TODO  for the specific problem, then delete this banner.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

# Make the skill's bundled profiling lib importable regardless of cwd.
# The GENERATED harness.py lives at artifacts/<problem>/harness.py, so it must point
# at the skill dir explicitly. When generating it, bake in the ABSOLUTE skill path
# you got from Claude Code's ${CLAUDE_SKILL_DIR} substitution, e.g.:
#   SKILL = "/abs/path/to/.claude/skills/optimize-kernels"   # TODO: set for generated file
# (Inside this template file itself, the package root is two dirs up.)
SKILL = os.environ.get(
    "OPTIMIZE_KERNELS_SKILL",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
)
sys.path.insert(0, SKILL)
from profiling import autotune  # noqa: E402

# --- problem-specific paths (TODO) -----------------------------------------
EDIT_TARGET = "optimize-kernels/<problem>/<kernel-source>"   # the ONLY original file the loop edits
HARNESS_REF = "optimize-kernels/<problem>/<harness-source>"  # corrected + frozen during the loop
BUILD_DIR = "artifacts/<problem>/build"
ARCH = "gfx942"

# --- config space to autotune (TODO: pick the knobs THIS kernel exposes) ----
# Generic placeholder knobs mapped to -D compile defines the edit target reads.
# Replace names/values with the real tunables for this kernel; keep it small.
CONFIG_SPACE = {
    "PARAM_A": [64, 128],     # e.g. a block/tile dimension
    "PARAM_B": [8, 16, 32],   # e.g. an unroll / inner-tile factor
}


def build(config=None, label="default"):
    """Compile harness + edit target into a binary, passing config as -D defines.
    Returns the binary path. Raises CalledProcessError on build failure (which the
    autotune sweep treats as an invalid config). TODO: match the real build line
    (sources, libs, flags) for this problem."""
    os.makedirs(BUILD_DIR, exist_ok=True)
    binpath = os.path.join(BUILD_DIR, f"harness_{label}")
    defines = [f"-D{k}={v}" for k, v in (config or {}).items()]
    cmd = ["hipcc", "-O3", f"--offload-arch={ARCH}", "-x", "hip",
           EDIT_TARGET, HARNESS_REF, *defines, "-o", binpath]
    # surface VGPR/SGPR/spill so the loop can watch register pressure
    cmd.insert(1, "-Rpass-analysis=kernel-resource-usage")
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return binpath


def _run_json(argv):
    """Run a harness invocation that prints one JSON object; parse it."""
    out = subprocess.run([str(a) for a in argv], check=True,
                         capture_output=True, text=True).stdout
    return json.loads(out)


def check(binpath, shape):
    """Return True iff the harness reports correctness for `shape`.
    TODO: match the real harness CLI/JSON keys for this problem."""
    res = _run_json([binpath, "check", *shape])
    return bool(res.get("correct", res.get("pass")))


def bench(binpath, shape, iters=100):
    """Return the perf metric the harness reports (higher = better by convention).
    TODO: match the real harness CLI/JSON keys for this problem."""
    res = _run_json([binpath, "bench", *shape, iters])
    return float(res.get("ratio", res.get("metric")))


def profile(binpath, shape, iters=200):
    """Return the kernel-only argv (no baseline/copy-back) for a profiler to wrap.
    TODO: match the real harness' kernel-only mode for this problem."""
    return [binpath, "profile", *shape, iters]


def autotune_cmd(shape, mode="grid", n=None, out="artifacts/<problem>/autotune.json"):
    """Sweep CONFIG_SPACE: build+check+bench each config, keep the best CORRECT one."""
    def evaluate(cfg):
        label = "_".join(f"{k}{v}" for k, v in cfg.items())
        binpath = build(cfg, label=label)        # build failure -> invalid
        if not check(binpath, shape):            # wrong -> invalid (score None)
            return None
        return bench(binpath, shape), {"bin": binpath}

    result = autotune.sweep(CONFIG_SPACE, evaluate, mode=mode, n=n,
                            maximize=True, on_result=autotune.default_logger)
    autotune.write_results(out, result)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "bench", "profile"):
        p = sub.add_parser(name)
        p.add_argument("shape", nargs="+", help="problem dimensions for this kernel")
    at = sub.add_parser("autotune")
    at.add_argument("shape", nargs="+")
    at.add_argument("--mode", choices=["grid", "random"], default="grid")
    at.add_argument("--n", type=int, default=None)
    args = ap.parse_args(argv)

    if args.cmd == "autotune":
        autotune_cmd(args.shape, mode=args.mode, n=args.n)
    elif args.cmd == "check":
        print(check(build(), args.shape))
    elif args.cmd == "bench":
        print(bench(build(), args.shape))
    elif args.cmd == "profile":
        print(" ".join(str(x) for x in profile(build(), args.shape)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
