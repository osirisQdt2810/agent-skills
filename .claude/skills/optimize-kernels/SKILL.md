---
name: optimize-kernels
description: Autonomously optimize a GPU kernel until it meets a measured target (correctness + performance vs a baseline).
argument-hint: [max-iters]
allowed-tools: Read, Edit, Write, Bash, TodoWrite, AskUserQuestion
---

You are running the **`optimize-kernels`** goal-loop skill. It is generic over any
single GPU kernel; you learn the specific problem from the files the user gives you.
**Do NOT assume matmul, GEMM, hipBLAS, or any particular kernel** — discover the actual
contract from the source. Examples below are illustrative placeholders, never defaults.

This skill is **interactive**: you ASK the user for its inputs (like `loop-tasks`) and
wait for answers. The only thing taken from the slash-command argument is `max-iters`.

- `$ARGUMENTS`:
  - **new run** (default): an optional number = **max iterations** (default `15`); starts a
    fresh optimization — asks for inputs (Phase 1) and calibrates (Phase 2).
  - **`--resume [problem] [max-iters]`**: continue an INTERRUPTED run from disk instead of
    starting a new one (see "Resume mode"). Use after a session/budget window ends — it works
    in a brand-new session or even a different account, since it reads only `artifacts/`.
    Optional `[problem]` disambiguates; an optional trailing number raises the iteration cap
    (needed if the previous run already hit `MAX_ITERS`).

## Resume mode (`--resume`) — continue an interrupted run, do NOT start a new one

`/optimize-kernels [max-iters]` always means **start a NEW optimization**. To continue the
kernel you were already optimizing after a session/budget window ended (or after switching
accounts), invoke `/optimize-kernels --resume`. In resume mode you **skip Phase 1 and Phase 2
entirely** and pick up from the on-disk checkpoint:

1. **Find the run.** Use the `[problem]` argument if given; otherwise scan
   `artifacts/*/loop_state.json` for one whose `status` is not `done`. None → tell the user
   there's nothing to resume. Several → ask which `<problem>`.
2. **Recover state from disk only** (no conversation memory needed): read
   `artifacts/<problem>/loop_state.json` (target, `max_iters`, `iters_done`, best snapshot)
   and `artifacts/<problem>/<problem>.context.md` (entry-point contract, kernel + harness
   paths, detected `ARCH`, kernel symbol, autotune space, the original `RUN_CONTEXT`). Do
   NOT re-ask the user for inputs.
3. **Re-validate, don't regenerate.** Confirm `verify.py` / `harness.py` and the working
   kernel copy still build (one `verify.py` run). Restore `EDIT_TARGET` from
   `artifacts/<problem>/best/`. Only if a calibrate artifact is genuinely missing/broken do
   you report it and offer to re-calibrate — otherwise keep everything frozen as-is.
4. **Continue the loop.** Bind Phase 3 as usual (`MAX_ITERS` = the recovered cap, or the
   larger one the user passed) and follow the engine — its Resume step (step 0) restores the
   counter and continues from `iters_done`. If the prior run was `stopped` at `MAX_ITERS`
   without passing, a larger `max-iters` is required to make further progress.

> Safety: a **new** run (no `--resume`) that finds an existing non-`done`
> `artifacts/<problem>/loop_state.json` should warn and offer `--resume` before overwriting
> it — so an interrupted optimization is never silently clobbered.

## Phase 1 — Gather inputs (ask ALL of these in ONE batched prompt, not split)

Ask these three inputs together in a single `AskUserQuestion` call (extra context included
there too, as a free-text option — never break it out into a separate follow-up prompt).
**Profiling is NOT an input** — it is always on and you drive it yourself (see Phase 2).
Confirm your understanding of:

1. **Reference file(s)** — the kernel source to optimize **and** its measurement
   harness. If several files are given, confirm which single file is the **edit
   target** (the kernel); the rest (harness) are the original measurement code.
2. **Target** — the success threshold for this run: the required `kernel / baseline` ratio
   (e.g. `0.5` = reach 50% of the baseline). The **baseline is the reference THIS run
   measures against** — a vendor lib / reference impl the harness already times (torch,
   aiter, triton, hipBLAS, …); it essentially always exists. It is NOT some external library
   you go hunt for, and NOT the user's own HIP kernel. The deliverable is always your
   optimized kernel beating this run's baseline by the target. (What MAY be missing is a
   *starting* HIP kernel to improve — then you write an initial correct one; see calibrate.)
3. **Extra context** — any note for THIS run (empty ⇒ skip). Used only this run.

## Conventions (provenance split)

- Original source lives in `optimize-kernels/<problem>/` — derive `<problem>` from the
  directory holding the kernel. **It stays pristine: the loop never edits the user's
  original kernel or harness; it works on copies/wrappers under `artifacts/`** (calibrate
  step 3).
- Everything you generate goes to `artifacts/<problem>/`: the **working kernel copy**
  (`EDIT_TARGET` — the only file the loop edits), `verify.py`, `harness.py`, `profile/`,
  `build/`, `runs/`, `result.json`, `autotune.json`, `<problem>.context.md`, `REPORT.md`.
- This skill's own helper files are **bundled inside the skill**, split by concern:
  `profiling/` (counter collection + digest), `tuning/` (autotune sweep), `templates/`
  (harness starting point), `references/` (distilled idea docs).
  At runtime Claude Code substitutes `${CLAUDE_SKILL_DIR}` with this skill's absolute path,
  so invoke them as `python3 ${CLAUDE_SKILL_DIR}/<dir>/<script>.py` (each script
  bootstraps its own imports, so cwd does not matter). Your generated `profile.py` /
  `harness.py` import/drive these instead of re-implementing CSV parsing, merging,
  digesting, or the sweep loop:
  - `${CLAUDE_SKILL_DIR}/profiling/profile_digest.py` — the ONE canonical digest shape both profilers emit.
  - `${CLAUDE_SKILL_DIR}/profiling/merge_v2.py` — rocprofv2 multi-line counters → merge → kernel-filter → digest.
  - `${CLAUDE_SKILL_DIR}/profiling/rocprofv3_digest.py` — rocprofv3 counter CSV (+ ATT) → digest.
  - `${CLAUDE_SKILL_DIR}/profiling/counters/default.txt` — curated CDNA counter set (gfx942 baseline; validate/extend per detected arch).
  - `${CLAUDE_SKILL_DIR}/tuning/autotune.py` — generic config-space sweep for `harness.py` (tune concern, separate from profiling).
  - `${CLAUDE_SKILL_DIR}/templates/harness_template.py` — annotated starting point for `harness.py`.
  - `${CLAUDE_SKILL_DIR}/references/amd-techniques.md` — distilled AMD/CDNA **idea menu** (arch-tagged, bottleneck-gated); open on-demand in the loop, never the deliverable.
  When you generate `profile.py` / `harness.py`, bake the resolved absolute skill path into
  their `sys.path` (you know it from the expanded `${CLAUDE_SKILL_DIR}`) so they import the
  bundled `profiling` / `tuning` packages.

## Environment (this box)

- AMD ROCm box (ROCm ~7.2): `hipcc`, `rocprofv2`, `rocprofv3`, `rocm_agent_enumerator`,
  `rocminfo`. The **exact GPU arch is NOT fixed** — it may be CDNA2 (`gfx90a`, MI250),
  CDNA3 (`gfx942`, MI300X), or CDNA4 (`gfx950`, MI350/MI355). **Detect it at calibrate**
  (Phase 2) into `ARCH`; do not hardcode. hipBLAS (`-lhipblas`) is available if a
  baseline needs it.
- Build HIP from `.cpp`: `hipcc -O3 --offload-arch=$ARCH -x hip <srcs> -o <bin>`
  (`$ARCH` = the detected arch — feeds the generated `harness.py` `ARCH` too).

## Phase 2 — Calibrate (do once, get the user's sign-off before looping)

1. **Read the source** — the kernel (the optimization target) and its harness. Identify the
   entry-point contract (signature, layout, dtype), how correctness is checked, the
   performance baseline, which file is the kernel, and **the measurement interface the harness
   actually exposes** (what command/flag/function yields correctness, perf, and a
   kernel-only region for profiling). Do NOT assume the names `check`/`bench`/`profile` —
   discover what this source really offers.
2. **Detect the arch, then validate it can be measured.** First detect the GPU arch —
   `rocm_agent_enumerator` (prints e.g. `gfx942`) or `rocminfo | grep -m1 gfx` — and set
   `ARCH`. Build everything with `--offload-arch=$ARCH` and pass `ARCH` to the generated
   `harness.py`. Then confirm it builds and that the hooks `verify.py` + profiling need are
   present: a **correctness ground truth** (always required), a **performance-baseline
   path**, and a way to run **only the kernel** (for counter attribution).
   - **No *starting* HIP kernel?** The **baseline reference** to beat (torch / aiter /
     triton / hipBLAS / …) essentially always exists — that is the gate's denominator. What
     may be absent is a HIP kernel to *improve*. If there's no starting kernel, **write an
     initial correct one from scratch** (the simplest version that passes correctness) as
     the working edit-target, capture its metrics, and optimize from there toward beating
     the baseline. Note this in `<problem>.context.md`.
3. **Correct first, then freeze the originals — the loop edits COPIES, not the user's
   source.** The original can be wrong (the **kernel AND its harness**). With approval, fix
   genuine correctness/measurement bugs first so the baseline measures something real
   (fixing a real bug is allowed; silently changing *what* is measured is not), applying
   each fix in the right place:
   - **Original harness** → corrected, then **frozen**: the loop never edits it again; you
     drive measurement through the generated `harness.py`.
   - **Original kernel** → **copied into `artifacts/<problem>/` as the working `EDIT_TARGET`**
     (e.g. `artifacts/<problem>/<kernel>`). EVERY kernel edit — any correctness fix *and*
     every optimization in the loop — happens on that copy, so the **user's original kernel
     file is never modified** and stays as the pristine record of the starting point. (If
     there is no starting kernel, write the from-scratch initial version directly as this
     working copy.)
4. **Generate `artifacts/<problem>/verify.py`** — the gate. It builds kernel+harness,
   runs correctness, runs performance vs the baseline, computes one `pass` against
   `--target`, and writes `result.json = {pass, target, build_ok, metrics, summary}`.
   It drives whatever interface the source actually has; it never reimplements the metric.
5. **Generate `artifacts/<problem>/harness.py`** — the agent-controlled measurement wrapper
   (distinct from `verify.py`), starting from `$SKILL/templates/harness_template.py`.
   It standardizes a `check`/`bench`/`profile` interface (the kernel-only `profile` mode is
   what the profiler attaches to) and drives the **parameter autotune** sweep. It must change
   the original flow as little as possible (prefer `import` over copy-paste; for C++/HIP,
   COMPILE the frozen original harness + the working edit-target copy with `-D` config defines
   rather than pasting bodies; never touch input-gen / ground-truth). Every number it reports
   comes from the original harness/reference — never new math here.
6. **Set up profiling (always — no user toggle).** Create `artifacts/<problem>/profile/`
   with a thin `profile.py` that collects with `rocprofv2` and/or `rocprofv3` (your choice —
   see "Profiling" below) using the curated counter set, and writes
   `artifacts/<problem>/profile/digest.json` via the shared helpers.
7. **Write `artifacts/<problem>/<problem>.context.md`** — a short, stable summary of what you
   learned (entry-point contract, how it's measured, baseline, what's editable, the kernel
   symbol name(s) you discovered, any autotune config space). Also record everything
   **resume** needs from disk alone: the kernel + harness paths, the detected `ARCH`, the
   target, and this run's `RUN_CONTEXT` — so a new session or even a different account can
   continue with `--resume`.
8. **Baseline + sign-off.** Run `verify.py` once unmodified to capture starting metrics, and
   run a baseline profile. Show the user the `verify.py` summary + baseline `result.json` +
   the baseline profile digest, and confirm before entering the loop.

## Profiling (always on — you choose the tool and the counters)

There is no profile flag: the loop **always** has a profiler wired up and uses it to drive
ideas. You decide *how* to profile:

- **Which tool.** Default to **`rocprofv2`** (robust multi-line counter path). Add
  **`rocprofv3`** when you want its native `--kernel-include-regex` or **`--att`**
  instruction-level stall traces (e.g. once counters say "latency-bound" but not *why*).
  Using both is fine — they emit the same digest shape.
- **Which counters.** Start from the curated `$SKILL/profiling/counters/default.txt` (chosen
  to map onto the digest's bottleneck signals: memory traffic/bandwidth, cache locality,
  compute utilization, LDS/occupancy). Extend or trim it when a specific hypothesis needs a
  specific counter — but keep each `pmc:` line small (overpacking yields nulls).

Common requirement: profiling needs a **kernel-only region** (no baseline, no copy-back) so
counters attribute cleanly. Expose it via the generated `harness.py` `profile` subcommand
(or a kernel-only mode added to the original in Phase 2 with approval). The kernel's
**mangled symbol name is unknown up front**, so always **dry-run first to discover it**, then
filter to it; record the name in `<problem>.context.md`.

Your `profile.py` is a THIN wrapper that runs one of the recipes below for this problem's
kernel-only command and writes `artifacts/<problem>/profile/digest.json` (averages over
dispatches dropping the first as warmup; derives
`memory_bandwidth_GBps = (FetchSize+WriteSize)KB·1.024 / duration_us`; attaches a bottleneck
label).

### Recipe A — `rocprofv2` deep metrics (multi-line counters → merge → filter)

rocprofv2 collects **one pass per `pmc:` line**; packing too many counters onto one line
silently yields **null** values, so the counter file is multi-line. Multiple lines ⇒
multiple per-pmc CSVs under `<outdir>/pmc_<i>/` ⇒ merge by `Dispatch_ID` and filter to the
exact kernel symbol. `merge_v2.py` does all of it:

```bash
SKILL="${CLAUDE_SKILL_DIR}"          # Claude Code expands this to the skill's absolute path
COUNTERS=$SKILL/profiling/counters/default.txt
OUT=artifacts/<problem>/profile/v2

# 1. dry-run to discover the mangled symbol name(s)
rocprofv2 -i "$COUNTERS" -d "$OUT" <kernel-only harness command...>
python3 $SKILL/profiling/merge_v2.py --discover --profile-dir "$OUT"
# 2. collect + merge + filter + digest (one shot; --run does the rocprofv2 call)
python3 $SKILL/profiling/merge_v2.py --counters "$COUNTERS" --profile-dir "$OUT" \
    --kernel-filter '<symbol-substr>' --run -- <kernel-only harness command...>
```

`profile.py` is just the second command wired to this problem's kernel-only invocation.

### Recipe B — `rocprofv3` (native kernel filter + ATT stall traces)

rocprofv3 (1.1.0) adds `--kernel-include-regex` (filter at collection time — no manual
merge), `-i counters.txt -f csv` (same multi-line/multi-pass rule as v2), and
`--att --att-activity <N>` for instruction-level stalls. `rocprofv3_digest.py` pivots the
(long-format) counter CSV and surfaces ATT stall hotspots into the **same digest**:

```bash
SKILL="${CLAUDE_SKILL_DIR}"          # Claude Code expands this to the skill's absolute path
COUNTERS=$SKILL/profiling/counters/default.txt
OUT=artifacts/<problem>/profile/v3

# counters (multi-pass), filtered to the kernel natively, then digest
python3 $SKILL/profiling/rocprofv3_digest.py --counters "$COUNTERS" \
    --out-dir "$OUT" --kernel-regex '<kernel-name-regex>' \
    --run -- <kernel-only harness command...>
# ATT instruction-level stalls (separate run; decode hotspots into notes)
python3 $SKILL/profiling/rocprofv3_digest.py --out-dir "$OUT/att" --att --att-activity 8 \
    --kernel-regex '<kernel-name-regex>' --run -- <kernel-only harness command...>
```

## `harness.py` — the agent-controlled measurement wrapper

`verify.py` is always the gate. `harness.py` is generated alongside it (Phase-2 step 5) and
is how you keep control without touching the frozen original during the loop:

- **Standardized interface.** It exposes `check` / `bench` / `profile` by importing/driving
  the corrected-and-frozen original — never rewriting it. The `profile` subcommand is the
  kernel-only region the profiler attaches to.
- **Parameter autotune.** When the algorithm looks right but its **config params** may be
  wrong (block/tile sizes, unroll factor, launch bounds, vector width — whatever knobs THIS
  kernel exposes), sweep a config space via `$SKILL/tuning/autotune.py`: it builds +
  correctness-checks + benches each config and keeps the best **correct** one. A sweep often
  hits the target with no new algorithm. Run `python3 artifacts/<problem>/harness.py
  autotune <shape>` and record the winning config in `<problem>.context.md`.

Correctness stays the gate inside autotune: a config that fails the build or the correctness
check scores `None` and is never selected.

## Optimization hints (an idea menu, NOT a recipe — gate by arch + bottleneck)

These are **not** ordered steps and not always applicable — different kernels have
different bottlenecks. Treat them as a menu of ideas to draw from **after looking at the
profile**; let the measured bottleneck decide which (if any) to try next. Don't assume a
fixed pipeline.

**Read the profile first.** Rough signals (the digest labels these for you): low `VALUBusy`
⇒ latency-bound (improve ILP / reuse / occupancy); high `MemUnitBusy`+`MemUnitStalled` ⇒
bandwidth-bound (cut global traffic, coalesce, vectorize); `LDSBankConflict` > 0 ⇒ pad
shared tiles; low occupancy ⇒ fewer registers / `__launch_bounds__`.

**Candidate techniques** (pick by bottleneck, not by order):

- Memory coalescing — adjacent threads read adjacent addresses.
- LDS / shared-memory staging — stage data in `__shared__` for reuse.
- Per-thread register blocking — each thread computes a micro-tile of the output.
- Vectorized 128-bit loads (`float4` / `__int128`-width).
- Padding shared tiles — avoid LDS bank conflicts (e.g. `+1` column).
- Double-buffering / prefetch — overlap global loads with compute.
- Block-size / occupancy tuning.
- Matrix cores — MFMA (`__builtin_amdgcn_mfma_*`) / rocWMMA where the math maps to them
  (shapes & availability differ by arch — see the AMD reference below).
- **Parameter autotune** — when the algorithm looks right but the config might be off,
  sweep the config space with `harness.py autotune` instead of hand-guessing one knob at a
  time. Cheap, exhaustive, and correctness-gated.

**AMD-specific techniques (CDNA) — borrow ideas, gate by arch + bottleneck.** For
tile-structured / tensor-core kernels (GEMM, attention, fused-norm, MoE) there is a
deeper, **arch-tagged** idea menu at `${CLAUDE_SKILL_DIR}/references/amd-techniques.md`:
8-wave/4-wave wave scheduling, XCD chiplet swizzle, bank-conflict-free LDS swizzle,
direct global→LDS, MFMA shape selection, register pinning, in-register elementwise math,
wave-vote reductions, … — each tagged CDNA2/3/4 (`gfx90a`/`gfx942`/`gfx950`) and by the
bottleneck it fixes. **Open it on demand** (only when the profile points at a matching
bottleneck *and* the kernel is that kind) and **borrow the idea into `EDIT_TARGET`** — do
not adopt a framework and do not go match an exact shape. It also points at HipKittens /
aiter as **idea sources only** (they may not be checked out locally): if you discover one
is faster than your kernel, **note it in `REPORT.md` and keep optimizing** — a better
existing library is never the deliverable. The deliverable is always your optimized
`EDIT_TARGET` beating this run's baseline by the target.

**Watch register pressure.** With register blocking / heavy unrolling, compile with
`-Rpass-analysis=kernel-resource-usage` to see per-kernel VGPR/SGPR and spill/scratch.
Spilling (scratch > 0) usually tanks performance — shrink the micro-tile, reduce
unrolling, or use `__launch_bounds__` until it disappears.

## Phase 3 — Run the loop

Once Phase 2 is signed off, bind these and follow the engine:

- `EDIT_TARGET` = the **working kernel copy** at `artifacts/<problem>/<kernel>` (copied from
  the frozen original at calibrate step 3; or the from-scratch initial kernel if none
  existed) — the ONLY file the loop may edit; the user's original kernel stays pristine
- `VERIFY_CMD` = `python3 artifacts/<problem>/verify.py --target <target>`
- `RESULT_JSON` = `artifacts/<problem>/result.json`
- `TARGET` = the target the user gave
- `MAX_ITERS` = `$ARGUMENTS` (default `15`)
- `PROFILE_CMD` = `python3 artifacts/<problem>/profile/profile.py` (always set — profiling
  is on by default; the wrapper writes `artifacts/<problem>/profile/digest.json`).
- `ARTIFACTS_DIR` = `artifacts/<problem>/`
- `RUN_CONTEXT` = the user's extra-context answer (may be empty)
- `STATE_JSON` = `artifacts/<problem>/loop_state.json` (the resumable checkpoint; always set
  — this is what `--resume` reads)

Note: `harness.py` is an additional agent-owned tool, not a binding — use it for autotune
sweeps (`python3 artifacts/<problem>/harness.py autotune <shape>`) and as the standardized
check/bench/profile interface. The gate is still `VERIFY_CMD` only.

@../../engine/_BASE.md
