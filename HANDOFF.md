# HANDOFF — optimize-kernels skill work (for transfer to a new account)

> Purpose: capture the full conversation context, decisions, completed work, and the
> **in-progress task** so a new Claude Code session (different account, same repo at
> `/home/phuc-nguyen/workspaces/agents`) can resume with zero loss. Read this top-to-bottom,
> then jump to **§6 RESUME HERE**.

Date of handoff: 2026-06-19. Repo: `/home/phuc-nguyen/workspaces/agents` (git, branch `main`).

---

## 1. What this repo is

A framework for **autonomous "goal loops"** in Claude Code: tasks that keep editing code and
re-measuring until an objective, machine-checked gate passes (or a budget runs out). See
[CLAUDE.md](CLAUDE.md) for the authoritative description. Three layers:

1. **Engine** — `.claude/engine/_BASE.md`: the shared loop, gate contract, guardrails,
   artifacts, escalation. (Plain included markdown, not a skill.)
2. **Generator** — `.claude/skills/loop-tasks/SKILL.md`: interviews the user and emits a new
   runner skill.
3. **Runners** — `.claude/skills/<task>/SKILL.md`: a concrete task type. First one:
   `optimize-kernels` (optimize a GPU kernel until correctness + perf target met).

Environment: runs in a Docker container on an **AMD** box. ROCm 7.2, `hipcc`, `rocprofv2`,
`rocprofv3` (1.1.0), `rocminfo`, `rocm_agent_enumerator`, hipBLAS. **The exact GPU arch is
NOT fixed** — could be gfx90a (MI250, CDNA2), gfx942 (MI300, CDNA3), or gfx950 (MI355, CDNA4).
This matters (see §5).

---

## 2. Chronology of work done in this conversation

### Round A — Built the profiling/tuning library + wired it into the runner (DONE)
Implemented the three "Next/open ideas" tasks from CLAUDE.md:
- **Task 1** — rocprofv3 profiling (ATT + counters, native `--kernel-include-regex`) parsed
  into a shared digest.
- **Task 2** — rocprofv2 deep metrics: multi-line `pmc:` counter sets → per-pmc CSVs → merge
  by `Dispatch_ID` → filter to exact kernel symbol (with dry-run **symbol discovery**).
- **Task 3** — generate a separate `harness.py` (besides `verify.py`) that standardizes
  check/bench/profile by driving the original (import/compile, no copy-paste) and runs a
  **parameter autotune** sweep over a config space (correctness-gated).

Built a Python lib (unit-tested, 36/36 passing, no GPU needed): `profile_digest.py` (shared
digest shape + bottleneck heuristic), `merge_v2.py` (rocprofv2 path), `rocprofv3_digest.py`
(rocprofv3 path), `autotune.py` (sweep), `counters/default.txt` (curated counter set),
`templates/harness_template.py`.

### Round B — Five follow-up requirements from the user (DONE)
1. **Co-locate skill scripts inside the skill** (not repo root).
2. **Fix harness philosophy**: original source (kernel AND harness) can be wrong → the agent
   **corrects it first (with approval), then FREEZES it**; during the autonomous loop it does
   NOT edit the original harness — it controls measurement via the generated `harness.py`.
   (The only original file the loop edits is the kernel = EDIT_TARGET.)
3. **Remove matmul bias** — the skill is generic over any kernel; matmul/GEMM/hipBLAS/M·N·K
   are only illustrative placeholders.
4. **Drop the `profile` input** — profiling is **always-on and self-driven** (the runner
   chooses rocprofv2 and/or rocprofv3 itself).
5. **The agent curates the counters** — `counters/default.txt` chosen to map onto the
   digest's bottleneck signals.

### Round C — Migrate to the Claude Code **skills standard** (DONE)
Verified the official spec (via claude-code-guide) and restructured:
- `.claude/commands/optimize-kernels.md` → `.claude/skills/optimize-kernels/SKILL.md`
- `.claude/commands/loop-tasks.md` → `.claude/skills/loop-tasks/SKILL.md`
- bundled lib → `.claude/skills/optimize-kernels/profiling/`
- engine `.claude/loop-tasks/_BASE.md` → `.claude/engine/_BASE.md` (removes name clash with
  the `loop-tasks` skill)
- Frontmatter: added `name:`; kept `argument-hint`, `allowed-tools`. `$ARGUMENTS` still works.
- Scripts invoked via `${CLAUDE_SKILL_DIR}/profiling/<script>.py` (Claude Code substitutes the
  skill's absolute path at runtime; each script also bootstraps its own `sys.path`).
- Engine include in SKILL.md: `@../../engine/_BASE.md` (relative `@`-include resolves from the
  skill dir).
- Generator updated to emit new runners in the same skills layout.
- CLAUDE.md updated to reflect all of the above. Tests re-run green at the new location.

> Standard facts confirmed: skills live at `.claude/skills/<name>/SKILL.md` (dir name = command
> name); bundle resources in subdirs; `${CLAUDE_SKILL_DIR}` gives the skill's abs path at
> runtime; `@path` includes resolve relative to the skill dir; commands are legacy (both create
> `/name`) so we MOVED, not duplicated. Both `optimize-kernels` and `loop-tasks` invoke as
> `/optimize-kernels [max-iters]` and `/loop-tasks`.

### Round D — IN PROGRESS (this is where to resume) — see §5 and §6.

---

## 3. Current file layout (the skills standard)

```
.claude/
├── engine/_BASE.md                         # shared loop engine (included, not a skill)
├── settings.json
└── skills/
    ├── loop-tasks/SKILL.md                 # generator skill
    └── optimize-kernels/SKILL.md           # runner skill
        └── profiling/                      # bundled lib (invoked via ${CLAUDE_SKILL_DIR})
            ├── __init__.py
            ├── profile_digest.py           # canonical digest shape + bottleneck_hint()
            ├── merge_v2.py                  # rocprofv2 multi-line→merge→filter→digest + discover_symbols
            ├── rocprofv3_digest.py          # rocprofv3 counters(pivot)+ATT→digest
            ├── autotune.py                  # generic config-space sweep (grid/random, correctness-gated)
            ├── counters/default.txt         # curated gfx942 counter set (4 small pmc: lines)
            └── templates/harness_template.py
```

Repo-root reference tooling is the **user's own**, left untouched: `merge.py`, `metrics/`,
`profile_rocprofv2.sh`, `profile_rocprofv3.sh`. Vendored repos (untracked): `3rdparty/HipKittens`,
`3rdparty/aiter`. Paper: `2511.08083v1.pdf` (HipKittens, arXiv 2511.08083). `.gitignore` has
`artifacts`, `__pycache__/`, `*.pyc`.

Worked example problem layout: `optimize-kernels/matmul/` (source) + `artifacts/<problem>/`
(generated; gitignored). The matmul example is just a reference of the layout.

Tests: a scratchpad test exercises the lib (36 assertions). It is NOT in the repo (scratchpad
is session-local). If needed, recreate from the modules' behavior; key checks were: warmup
drop, bandwidth derive, bottleneck labels, merge+filter+discover, v3 long-format pivot, ATT
notes, autotune best/skip-invalid, curated counters parse to 4 small lines.

---

## 4. Conventions / decisions to honor (do not regress)

- **Gate contract**: `result.json = {pass, metrics, summary}`. `pass` is the ONLY success
  signal. `verify.py` drives the original harness; never reimplements the metric.
- **Provenance split**: original source in `optimize-kernels/<problem>/`; everything generated
  in `artifacts/<problem>/`; machinery in `.claude/`.
- **Correct-first, then freeze** (Round B #2): fix genuine bugs in the original during
  calibrate with approval; afterwards the original harness is frozen, loop controls via
  `harness.py`. Only EDIT_TARGET (the kernel) is edited in the loop.
- **Generic over any kernel** (Round B #3): no matmul assumptions.
- **Profiling always-on** (Round B #4): no profile input; runner picks rocprofv2/v3.
- **Skills standard** (Round C): scripts via `${CLAUDE_SKILL_DIR}`, engine via
  `@../../engine/_BASE.md`, keep SKILL.md reasonably short (progressive disclosure → put deep
  material in bundled reference files).
- Commit only when the user asks (nothing has been committed this session).

---

## 5. THE IN-PROGRESS TASK (Round D) — add AMD optimization techniques + generalize arch

User's idea, with three decisions already made via AskUserQuestion:

1. **Approach** = **Distill once + on-demand pointers** (NOT live-read-in-loop, NOT prose-only).
   Read the repos/paper once, write a bundled reference doc, tag techniques by arch + bottleneck,
   include precise file pointers; the loop only opens source on-demand when the kernel shape
   matches and the profile points to the relevant bottleneck.
2. **Depth** = **Paper + source + blog (deep)**. (Already done — see §7; do not re-run.)
3. **Arch** = **Generalize now**: detect arch at calibrate via `rocm_agent_enumerator` /
   `rocminfo`, build with the detected `--offload-arch`, and **tag every hint by CDNA family**
   (CDNA2 gfx90a / CDNA3 gfx942 / CDNA4 gfx950). Counters may differ per arch → validate the
   counter set per arch (note it; keep default.txt as a CDNA baseline).

### Remaining steps (todo)
- [ ] **Write `.claude/skills/optimize-kernels/profiling/references/amd-techniques.md`** — the
  distilled reference, arch-tagged + bottleneck-gated, with the file pointers in §7. Keep it as
  a bundled file (progressive disclosure), not inline in SKILL.md.
- [ ] **Update SKILL.md**:
  - Add **arch detection** to the calibrate phase (rocm_agent_enumerator/rocminfo → set `ARCH`,
    build with `--offload-arch=<ARCH>`); add `ARCH` to the Environment section (replace the
    hardcoded gfx942 framing with "detected at calibrate").
  - In "Optimization hints", **tag existing hints by CDNA family** and add a short pointer:
    "for AMD-specific techniques (wave scheduling, XCD swizzle, LDS swizzle, direct-to-LDS,
    register pinning) see `references/amd-techniques.md`; gate by arch + measured bottleneck."
  - Mention **aiter as an optional stronger baseline** (call `aiter.*` to anchor per-shape
    timing) and as a technique/config source.
- [ ] **Generalize arch in `templates/harness_template.py`**: `ARCH` should come from calibrate
  (env/var), not hardcoded `gfx942`. Add a note that counter availability is arch-dependent.
- [ ] **Update CLAUDE.md**: Environment (arch unknown → detected), and add the references dir to
  the architecture/status notes.
- [ ] Re-run the lib sanity (compile + the digest/merge/autotune checks) after edits.

### Caveats to bake into the reference (be honest)
- HipKittens techniques are **specialized** for tile-structured, tensor-core-heavy kernels
  (GEMM / attention / norm family). They will NOT help a memory-bound elementwise/reduction
  kernel. **Gate by bottleneck and kernel shape.**
- "Apply" has two modes: **borrow the idea** (cheap, usually right) vs **adopt the HK framework**
  (big rewrite, rarely worth it inside a bounded loop). Default to borrowing ideas.
- **License**: both HipKittens and aiter are **MIT**. Borrowing techniques is fine; copying code
  verbatim into the user's kernel needs attribution / license respect.
- Note: `3rdparty/aiter/.claude/skills/` ships its own skills (`opus-kernel-best-practice`,
  `opus-module-build-optimization`) scoped to files under `3rdparty/aiter/` — useful when
  reading aiter's OPUS kernels, but they are aiter's, not ours.

---

## 6. RESUME HERE

1. Re-read `.claude/skills/optimize-kernels/SKILL.md` and `CLAUDE.md` (current state).
2. Do the remaining steps in §5 using the distilled research in §7 (already gathered — do NOT
   re-run the Explore agents or re-read the paper unless you need a specific extra detail).
3. Keep all conventions in §4. Show the user the new reference + SKILL.md diffs for sign-off
   before considering it done. Don't commit unless asked.

---

## 7. DISTILLED RESEARCH (already gathered — embed into the reference doc)

All paths below are under `/home/phuc-nguyen/workspaces/agents/`.

### 7a. AMD CDNA architecture facts (from paper + "AMD GPUs go brrr" blog)
- **CU/SIMD/wave**: each CU has **4 SIMDs**; a **64-thread wave** occupies one SIMD (AMD wave =
  NVIDIA warp, but 64 not 32). Register file is **2× NVIDIA**: 512 regs/SIMD split into **256
  VGPRs + 256 AGPRs** per wave (single-wave-per-SIMD case).
- **Chiplets (XCD)**: MI355X (CDNA4) = **8 XCDs × 32 CUs = 256 CUs**; per-XCD L2, shared LLC
  between L2 and HBM. CDNA3 (MI300X) ≈ 38 CUs/XCD. Miss penalty ~L2 300ns / LLC 500ns; L2 BW
  ~3× LLC BW.
- **MFMA**: heterogeneous matrix-core shapes (16×16×32, 32×32×16, 16×16×8; FP8 16×16×128,
  32×32×64; CDNA4 adds `v_mfma_f32_16x16x128_f8f6f4`). Smaller shapes than NVIDIA → enable deep
  pipelines via finer load/compute stages.
- **Missing vs NVIDIA**: no async matrix instr on shared/tensor mem (no wgmma/tcgen05), **no
  register reallocation** (waves can't share regs), no TMA, no first-class mbarrier. HIPCC
  **cannot use AGPRs as MFMA inputs** → forces register moves unless you pin registers.
- **Advantageous**: 2× register file; small MFMA shapes; **direct global→LDS loads** via
  `buffer_load_dword*` (bypass VGPRs).
- **LDS banks**: 4-byte banks; `ds_read_b128` → 64 banks / 4 phases; `ds_write_b64` → 32 banks
  / 4 phases; `ds_read_b96` → 32 banks / 8 phases. **No universal swizzle** works across all
  shapes/instructions — need per-layout swizzles.
- **Peak (MI355X)**: BF16 ~2.5 PFLOPs; MXFP4/6 ~10.1 PFLOPs; 288 GB HBM; ~8 TB/s.

### 7b. HipKittens technique taxonomy (LICENSE: MIT). Pointers under `3rdparty/HipKittens/`.

| # | Technique | Kernel shape | Bottleneck | Arch | Pointer |
|---|-----------|--------------|-----------|------|---------|
| 1 | **8-wave ping-pong** scheduling (2 wavegroups; one computes MFMA while other loads; `s_barrier` + `s_setprio`) | GEMM, attn fwd | global latency, balanced compute/mem | CDNA3+CDNA4 | `kernels/gemm/fp8fp32/FP8_8wave/8_wave.cu:13-154` |
| 2 | **4-wave interleave** (1 wave/SIMD; fine-grained load/MFMA interleave via `do_interleaved_cluster`; smaller tiles, lower regs/occupancy) | FP8 GEMM (compute-heavy), attn bwd (register-heavy) | occupancy, L2, imbalanced pipelines | CDNA4 (best); CDNA3 limited | `kernels/gemm/fp8fp32/FP8_4wave/4_wave.cu:18-83`, `:186-214` |
| 3 | **XCD chiplet swizzle** (remap blockIdx so consecutive blocks stay on one XCD; + L2 windowed traversal). +18–55% BW on big GEMM | large GEMM | L2/LLC locality | CDNA4 (8 XCD); N/A single-chiplet | `kernels/gemm/fp8fp32/FP8_4wave/4_wave.cu:114-133` |
| 4 | **Bank-conflict-free LDS swizzle** (per-layout masks, e.g. `offset ^= ((offset%512)>>7)<<3` for bf16 16×16) | all LDS users | LDS bank conflicts | CDNA3+CDNA4 (masks differ by gen) | `include/cdna4/types/shared/st_shape.cuh:48-146` |
| 5 | **Direct global→LDS** (`llvm_amdgcn_raw_buffer_load_lds`; bypass VGPRs; light `s_waitcnt vmcnt`) | GEMM, attn | global latency, register pressure | CDNA3+CDNA4 (not CDNA2) | `include/cdna4/ops/warp/memory/tile/global_to_shared.cuh:36-69` |
| 6 | **Precomputed swizzled offsets** (`prefill_swizzled_offsets`; amortize address calc across loads) | multi-load loops | VALU/instr count | all | `.../global_to_shared.cuh:117-149`; use `FP8_8wave/8_wave.cu:75-78` |
| 7 | **MFMA shape selection** wrappers (`mma_ABt`/`mma_AtB` dispatch on dtype/shape) | compute-bound | throughput vs regs | CDNA3+CDNA4 | `include/cdna4/ops/warp/register/tile/assembly/mma.cuh:24-130`; builtins `include/cdna4/common/macros.cuh:774-1107` |
| 8 | **Register-tile (accumulator) layouts** (`rt_16x16/32x32/16x32`, col_l vs row_l to cut transposes) | GEMM/attn | transpose cost, regs | all | `include/cdna4/types/register/rt_shape.cuh:20-47` |
| 9 | **Wave priority + scheduler barriers** (`s_setprio`, `sched_barrier`, `sched_group_barrier`) to force MFMA/VMEM/DS ordering | high-occupancy compute | ILP, load/compute overlap | CDNA4 (rich); CDNA3 limited | `kernels/attn/gqa_backwards/attn_fwd_non_causal.cpp:41-55`; `FP8_8wave/8_wave.cu:116-130` |
| 10 | **Transpose LDS reads** (`ds_read_b64_tr_b16/_b8/_b4`) to match MFMA layout without explicit transpose | shared→register | LDS latency, instr count | CDNA3+CDNA4 (`_trN` differ by gen) | `include/cdna4/common/macros.cuh:238-288`; `.../shared_to_register.cuh:248-267` |
| 11 | **Wave voting reductions** (`__all()` instead of ds_bpermute ladders) | attention softmax/reductions | sync cost | all | `kernels/attn/gqa/kernel.cpp:71-85` |
| 12 | **In-register elementwise math** (RV tiles; avoid LDS round-trips; unroll over packed elems) | memory-bound (rotary/layernorm/softmax) | bandwidth, LDS capacity | all | `kernels/layernorm/kernel.cpp:57-126`; `kernels/rotary/kernel.cpp:44-74` |
| 13 | **Developer-controlled register pinning** (bypass HIPCC's AGPR-as-MFMA-input limitation) | register-heavy (attn bwd) | register file utilization | CDNA3+CDNA4 | paper §3.2.1, App D.3 (interface `register_ranges` + tile struct) |
| — | **Wave specialization (producer-consumer) UNDERPERFORMS on AMD** (no register reallocation → wasted producer regs). Prefer 8-wave/4-wave instead. | GEMM | — | CDNA3+CDNA4 | paper §3.3.1, Table 2 |

**8-wave vs 4-wave rule of thumb** (paper Table 3): 8-wave = fewer LoC, large tiles, great for
balanced/latency-bound (GEMM fwd, attn fwd); 4-wave = more LoC, small tiles + low regs, wins on
compute-heavy FP8 GEMM and register-heavy attn backward (e.g. MHA bwd 4-wave 1091 vs 8-wave 894
TFLOPS; GQA non-causal bwd 4-wave ~1.8–2.3× over AITER). Choose by bottleneck, not by default.

**Headline results** (vs AITER asm / rocBLAS / hipBLASLT / CK / Triton / PyTorch SDPA): GEMM
hotloop <100 LoC competitive with AITER/hipBLASLT; attn fwd 1.0–2.1× AITER, 1.3–4.5× SDPA; attn
bwd 1.8–2.5× all baselines; fused dropout-residual-layernorm & RoPE 1.1–2.2× AITER. XCD swizzle
+18.7% TFLOPS / +55% BW on big GEMM. Tested on MI325X (CDNA3), MI350X/MI355X (CDNA4), ROCm 7.0.

### 7c. aiter (ROCm/aiter) — LICENSE: MIT. Pointers under `3rdparty/aiter/`.
- **What it is**: AMD's production op library (CK-generated kernels + FlyDSL JIT + Triton +
  hand asm + OPUS C++ templates). Consumed from **Python**.
- **Entry points** (use as **stronger baseline** + technique source):
  - BF16 GEMM: `aiter.gemm_a16w16(A, B, bias=None, otype=torch.bfloat16)` — `aiter/tuned_gemm.py:253-317`
  - A8W8 GEMM: `aiter.gemm_a8w8(...)`
  - Flash attention: `aiter.flash_attn_func(q,k,v,...)` — `aiter/ops/mha.py:2298-2380`
  - MLA decode: `aiter/mla.py:19-116`; RMSNorm: `aiter/ops/rmsnorm.py:50-73`; fused MoE:
    `aiter/fused_moe.py:75-150`; quant (fp8/mx): `aiter/ops/quant.py:42-94`
  - Perf helper: `aiter.test_common.run_perftest(...)` (warmup + CUDA events → µs/iter)
- **Config-driven selection (relevant to our autotune)**: tuned CSV tables keyed by
  `(gfx, cu_num, M, N, K, dtype, ...)` → `(kernelId, splitK, us, kernelName, tflops, bw)`.
  Example: `aiter/configs/a8w8_tuned_gemm.csv`; loader+fallback `aiter/tuned_gemm.py:59-79,117-162`;
  overridable via env `AITER_CONFIG_GEMM_*`; capture shapes with `AITER_TUNE_GEMM=1`.
  Arch-aware codegen example: `aiter/ops/flydsl/gemm_kernels.py:44` (`KERNEL_ASYNC_COPY =
  get_rocm_arch() != "gfx942"`).
- **Techniques to mine**: online-softmax fused MHA pipeline (`csrc/kernels/mha_native/fused/
  pipeline.hpp:9-60`), split-K persistent GEMM, MXFP4/FP8 quant+MoE fusion, OPUS MFMA templates
  (`csrc/include/opus/opus.hpp`).
- **Arch coverage**: gfx942 + gfx950 (OPUS kernels gfx950-only); gfx90a not first-class.
- **How to use as baseline**: call the matching `aiter.*` op on the same shape, time with
  `run_perftest`, set the loop's target relative to it ("beat X% of aiter").

### 7d. Blog narrative (for framing the reference's intro)
- "AMD GPUs go brrr" + "HipKittens": AMD has competitive peak FLOPs but a software gap; the fix
  is **tile-based abstractions** (interface generalizes; backends are arch-specific). The big
  AMD-specific lessons: wave specialization fails (no reg realloc) → use 8-wave/4-wave; bank
  conflicts need per-shape swizzles; chiplet (XCD) scheduling drives cache reuse; small MFMA
  shapes + 2× register file are AMD's path to deep pipelines. Competing AMD DSLs (Mojo,
  TileLang, Triton) leave large performance on the table (e.g. Mojo MHA ~50% of peak).

---

## 8. Quick command reference (current, skills-standard)

```bash
SKILL=.claude/skills/optimize-kernels        # ${CLAUDE_SKILL_DIR} at runtime
# rocprofv2 deep metrics:
rocprofv2 -i $SKILL/profiling/counters/default.txt -d <OUT> <kernel-only-cmd>   # dry run
python3 $SKILL/profiling/merge_v2.py --discover --profile-dir <OUT>             # symbol(s)
python3 $SKILL/profiling/merge_v2.py --counters $SKILL/profiling/counters/default.txt \
    --profile-dir <OUT> --kernel-filter '<symbol-substr>' --run -- <kernel-only-cmd>
# rocprofv3:
python3 $SKILL/profiling/rocprofv3_digest.py --counters $SKILL/profiling/counters/default.txt \
    --out-dir <OUT> --kernel-regex '<name>' --run -- <kernel-only-cmd>
python3 $SKILL/profiling/rocprofv3_digest.py --out-dir <OUT>/att --att --att-activity 8 \
    --kernel-regex '<name>' --run -- <kernel-only-cmd>
# autotune (generated per problem):
python3 artifacts/<problem>/harness.py autotune <shape>
# arch detect (for Round D):
rocm_agent_enumerator        # e.g. prints gfx942 ; or: rocminfo | grep -m1 gfx
```
