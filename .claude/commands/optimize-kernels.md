---
description: Autonomously optimize a GPU kernel until it meets a measured target (correctness + performance vs a baseline).
argument-hint: [max-iters]
allowed-tools: Read, Edit, Write, Bash, TodoWrite, AskUserQuestion
---

You are running the **`optimize-kernels`** goal-loop skill. It is generic over any
single GPU kernel; you learn the specific problem from the files the user gives you.
Do NOT assume matmul or any particular kernel.

This skill is **interactive**: you ASK the user for its inputs (like `loop-tasks`) and
wait for answers. The only thing taken from the slash-command argument is `max-iters`.

- `$ARGUMENTS` = **max iterations** (optional, default `15`).

## Phase 1 — Gather inputs (ask ALL of these in ONE batched prompt, not split)

Ask the four inputs together in a single `AskUserQuestion` call (extra context included
there too, as a free-text option — never break it out into a separate follow-up prompt).
Confirm your understanding of:

1. **Reference file(s)** — the kernel source to optimize **and** its measurement
   harness. If several files are given, confirm which single file is the **edit
   target** (the kernel); the rest (harness) are read-only.
2. **Target** — the success threshold for this run. For a perf-vs-baseline gate it's
   the required `kernel / baseline` ratio (e.g. `0.5` = reach 50% of hipBLAS).
3. **Profile** — set up `rocprofv2` profiling for diagnostics? (yes/no; optionally a
   custom counter list). If yes, you wire it up in Phase 2 and the loop uses it when stuck.
4. **Extra context** — any note for THIS run (empty ⇒ skip). Used only this run.

## Conventions (provenance split)

- Original source lives in `optimize-kernels/<problem>/` — derive `<problem>` from the
  directory holding the edit target.
- Everything you generate goes to `artifacts/<problem>/`: `verify.py`, `profile/`,
  `build/`, `runs/`, `result.json`, `<problem>.context.md`, `REPORT.md`.

## Environment (this box)

- AMD **MI300X**, arch **`gfx942`** (CDNA3), ROCm 7.2, `hipcc`, hipBLAS (`-lhipblas`).
- Build HIP from `.cpp`: `hipcc -O3 --offload-arch=gfx942 -x hip <srcs> -o <bin>`.

## Phase 2 — Calibrate (do once, get the user's sign-off before looping)

1. **Read the source** — the edit target and its harness. Identify the entry-point
   contract (signature, layout, dtype), how correctness is checked, the performance
   baseline, which exact file is editable, and **the measurement interface the harness
   actually exposes** (what command/flag/function yields correctness, perf, and a
   kernel-only region for profiling). Do NOT assume the names `check`/`bench`/`profile` —
   discover what this source really offers.
2. **Validate it can be measured.** Confirm it builds, and that the hooks `verify.py`
   (and profiling, if requested) need are present: a correctness ground truth, a
   performance/baseline path, and — for profiling — a way to run only the kernel.
3. **Fix gaps (with approval).** If a hook is **missing or mis-named** (no kernel-only
   region, no correctness reference, wrong entry-point name, won't build), **report it to
   the user and fix it before continuing** — preferably by adapting `verify.py` to the
   interface that exists; only if necessary, add a minimal hook to the harness (e.g. a
   kernel-only profile mode). The harness is *their* source: fix in place with approval,
   and never silently change *what* it measures.
4. **Generate `artifacts/<problem>/verify.py`** — the gate. It builds kernel+harness,
   runs correctness, runs performance vs the baseline, computes one `pass` against
   `--target`, and writes `result.json = {pass, target, build_ok, metrics, summary}`.
   It drives whatever interface the source actually has; it never reimplements the metric.
5. **If profiling was requested, set it up** (see recipe below): create
   `artifacts/<problem>/profile/{counters.txt, profile.py}` and set `PROFILE_CMD`.
6. **Write `artifacts/<problem>/<problem>.context.md`** — a short, stable summary of
   what you learned (entry-point contract, how it's measured, baseline, what's editable).
7. **Baseline + sign-off.** Run `verify.py` once unmodified to capture starting metrics.
   Show the user the `verify.py` summary + baseline `result.json` (and a baseline profile
   if enabled) and confirm before entering the loop.

## Profiling recipe (`rocprofv2`, gfx942 — validated on this box)

- Profiling needs a **kernel-only region** so counters attribute cleanly to the kernel
  under test (no baseline, no copy-back). In this demo the harness exposes a
  `profile M N K ITERS` mode for that; if the source lacks such a region, add one in
  Phase 2 with the user's approval (don't assume it already exists).
- `counters.txt` uses one `pmc:` line per pass (rocprofv2 re-runs the app per pass):
  ```
  pmc: FetchSize WriteSize VALUBusy SALUBusy
  pmc: LDSBankConflict MemUnitBusy MemUnitStalled Wavefronts
  ```
- Run: `rocprofv2 -i counters.txt -d <outdir> <harness_bin> profile M N K ITERS`
- Output: per-dispatch CSVs at `<outdir>/pmc_*/results_*.csv`, columns include
  `FetchSize WriteSize VALUBusy LDSBankConflict MemUnitBusy Wavefronts` plus
  `Start_Timestamp End_Timestamp Kernel_Name`. `FetchSize`/`WriteSize` are in **KB**;
  per-dispatch time = `End-Start` (ns).
- `profile.py` averages over dispatches (drop the first as warmup), derives e.g.
  `memory_bandwidth_GBps = (FetchSize+WriteSize)KB·1024 / mean_dispatch_time_ns`,
  and writes `artifacts/<problem>/profile/profile.json` + a short printed summary
  (bandwidth, VALUBusy, LDSBankConflict, MemUnitBusy/Stalled, occupancy).

## Optimization hints (gfx942 / CDNA3 — an idea menu, NOT a recipe)

These are **not** ordered steps and not always applicable — different kernels have
different bottlenecks. Treat them as a menu of ideas to draw from **after looking at the
profile**; let the measured bottleneck decide which (if any) to try next. Don't assume a
fixed pipeline like "tile → register-block → MFMA".

**Read the profile first.** Rough signals: low `VALUBusy` ⇒ latency-bound (improve ILP /
reuse / occupancy); high `MemUnitBusy`+`MemUnitStalled` ⇒ bandwidth-bound (cut global
traffic, coalesce, vectorize); `LDSBankConflict` > 0 ⇒ pad shared tiles; low occupancy ⇒
fewer registers / `__launch_bounds__`.

**Candidate techniques** (pick by bottleneck, not by order):

- Memory coalescing — adjacent threads read adjacent addresses.
- LDS / shared-memory tiling — stage tiles in `__shared__` for data reuse.
- Register blocking — each thread computes a micro-tile of the output.
- Vectorized 128-bit loads (`float4`).
- Padding shared tiles — avoid LDS bank conflicts (e.g. `+1` column).
- Double-buffering / prefetch — overlap global loads with compute.
- Block-size / occupancy tuning.
- Matrix cores — MFMA (`__builtin_amdgcn_mfma_*`) / rocWMMA where the math maps to them.

**Watch register pressure.** With register blocking / heavy unrolling, compile with
`-Rpass-analysis=kernel-resource-usage` to see per-kernel VGPR/SGPR and spill/scratch.
Spilling (scratch > 0) usually tanks performance — shrink the micro-tile, reduce
unrolling, or use `__launch_bounds__` until it disappears.

## Phase 3 — Run the loop

Once Phase 2 is signed off, bind these and follow the engine:

- `EDIT_TARGET` = the kernel source file the user chose (the ONLY file the loop may edit)
- `VERIFY_CMD` = `python3 artifacts/<problem>/verify.py --target <target>`
- `RESULT_JSON` = `artifacts/<problem>/result.json`
- `TARGET` = the target the user gave
- `MAX_ITERS` = `$ARGUMENTS` (default `15`)
- `PROFILE_CMD` = `python3 artifacts/<problem>/profile/profile.py` if profiling was set up, else empty
- `ARTIFACTS_DIR` = `artifacts/<problem>/`
- `RUN_CONTEXT` = the user's extra-context answer (may be empty)

@.claude/loop-tasks/_BASE.md
