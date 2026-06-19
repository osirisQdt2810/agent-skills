# AMD CDNA optimization techniques — distilled idea menu

A bottleneck-gated, arch-tagged menu of AMD-specific kernel **ideas to borrow**,
distilled once from the HipKittens paper (arXiv 2511.08083), the HipKittens and aiter
source, and AMD kernel-optimization blogs. The `optimize-kernels` loop reads this
**on demand** — only after a profile says which bottleneck to attack.

> **This is an idea index, not code to paste and not a list of kernels to adopt.** The
> distilled *concepts* below stand on their own — you do **not** need the source repos
> checked out to use them. The `file:line` pointers are a convenience *if* HipKittens /
> aiter happen to be available locally (or you choose to clone them to dig for more
> ideas); they are not required, and nothing in the loop should block on their presence.

---

## 0. The prime directive (READ FIRST)

The deliverable of this loop is **always one thing: a kernel (`EDIT_TARGET`) that you
write/optimize until it beats the run's reference by the target X%.** Everything in
this file exists only to give you *ideas* toward that. Concretely:

- **Borrow ideas, don't adopt kernels.** Take a *concept* — "remap blockIdx so
  consecutive blocks share an XCD", "XOR-swizzle the LDS offset to kill bank
  conflicts", "interleave loads with MFMA" — and implement it **in `EDIT_TARGET`** in
  plain HIP, in the kernel's own style. You do **not** need to match the shape, dtype,
  or tiling of the kernel you got the idea from.
- **A better existing implementation is NOT a substitute for the work.** If, while
  mining ideas, you find that aiter / HipKittens / a vendor lib is itself faster than
  the user's kernel, that is **not** the answer to hand back. **Note it in `REPORT.md`**
  so the user knows it exists, **borrow its ideas**, and **keep looping** on
  `EDIT_TARGET`. Never end the loop early by presenting someone else's kernel as the
  result.
- **The reference/target is defined by the run, not by this file.** The gate's baseline
  is whatever the harness measures against (see the SKILL's calibrate phase, including
  the *no-performance-baseline → bootstrap one from scratch* case). aiter is **not** the
  target; it is an idea source (§7).
- **"Adopt the HK framework"** (pull in HipKittens tile types/ops and rewrite on top of
  them) is a different, heavyweight move — big rewrite, new build surface, high
  regression risk. Inside a bounded loop this is **rarely** worth it; default to
  borrowing the idea.
- **License.** HipKittens and aiter are **MIT**. Borrowing a technique is free. If you
  copy non-trivial code verbatim, preserve attribution + the MIT notice and tell the
  user — but prefer re-implementing the idea.

**Pointers drift / may be absent.** Line numbers were captured at distill time; the
repos can change or not be present. Treat `file:line` as "start looking here *if you
have it*". If it doesn't match (or you cloned a newer copy), `grep` for the named
symbol — `do_interleaved_cluster`, `ds_read_b64_tr_b16`, `prefill_swizzled_offsets`,
`gemm_a16w16`, etc.

---

## 1. When a technique applies (the two-part gate)

Both must hold before a technique here is worth an iteration. If either fails, stay
with the generic menu in `SKILL.md` + the profile; don't reach here.

1. **The bottleneck matches.** Read `profile/digest.json` (`bottleneck` + `metrics` +
   `notes`) and use the lookup in §6. Applying a compute-pipeline trick to a
   bandwidth-bound kernel wastes iterations.
2. **The kernel is the right *kind*.** Most of these come from tile-structured,
   tensor-core-heavy kernels — **GEMM / attention / fused-norm (layernorm, RMSNorm,
   rotary) / MoE**. They will **not** help a plain memory-bound elementwise or
   reduction kernel; for those the generic ideas (coalesce, vectorize, raise
   occupancy) are the whole story. A few entries here are genuinely general
   (in-register elementwise, wave-vote reductions, precomputed offsets) — tagged
   "all shapes".

---

## 2. CDNA architecture facts (orientation)

Arch is **detected at calibrate** (`rocm_agent_enumerator` / `rocminfo`) into `ARCH` —
never assume it. Three CDNA generations may appear; techniques are tagged by family:

| Family | gfx | Example part | Notes |
|--------|-----|--------------|-------|
| CDNA2 | `gfx90a` | MI250 / MI250X | older; **no** direct-global→LDS, limited scheduler intrinsics |
| CDNA3 | `gfx942` | MI300X / MI325X | direct-to-LDS, MFMA-rich; XCD present, fewer CUs/XCD |
| CDNA4 | `gfx950` | MI350X / MI355X | richest: 8 XCDs, more scheduler control, new FP8/MX MFMA shapes |

Core facts (broadly CDNA unless noted):

- **CU / SIMD / wave.** Each CU has **4 SIMDs**; a **64-lane wave** runs on one SIMD
  (AMD wave = NVIDIA warp but **64**, not 32). Register file ~**2× NVIDIA**: ~512
  regs/SIMD = **256 VGPR + 256 AGPR** per wave (single-wave-per-SIMD case).
- **Chiplets (XCD).** CDNA4 MI355X = **8 XCDs × 32 CUs = 256 CUs**; per-XCD L2, a
  shared LLC between L2 and HBM. CDNA3 MI300X ≈ 38 CUs/XCD. Rough miss penalty L2
  ~300 ns / LLC ~500 ns; L2 BW ≈ 3× LLC. **Single-chiplet parts ⇒ XCD swizzle is N/A.**
- **MFMA matrix cores.** Heterogeneous shapes — 16×16×32, 32×32×16, 16×16×8; FP8
  16×16×128, 32×32×64; **CDNA4 adds** `v_mfma_f32_16x16x128_f8f6f4`. Shapes are
  *smaller* than NVIDIA's — that is what enables deep load/compute pipelines.
- **What AMD lacks vs NVIDIA.** No async matrix instr on shared/tensor memory (no
  `wgmma`/`tcgen05`), **no register reallocation** (waves can't hand registers to each
  other), no TMA, no first-class `mbarrier`. HIPCC **cannot use AGPRs as MFMA inputs**
  → inserts register moves unless you pin registers yourself.
- **What AMD has going for it.** 2× register file; small MFMA shapes; **direct
  global→LDS** loads (`buffer_load_dword*`) bypassing VGPRs (CDNA3+; **not** CDNA2).
- **LDS banks.** 4-byte banks. `ds_read_b128` → 64 banks / 4 phases; `ds_write_b64` →
  32 banks / 4 phases; `ds_read_b96` → 32 banks / 8 phases. **No single swizzle works
  for all shapes/instructions** — masks are per-layout and differ by gen.
- **Peak (MI355X, ballpark).** BF16 ~2.5 PFLOP/s; MXFP4/6 ~10.1 PFLOP/s; 288 GB HBM;
  ~8 TB/s. (MI300X HBM3 peak ~5.3 TB/s — the figure the digest cites.)

---

## 3. The big lesson (why these techniques, in one paragraph)

AMD has competitive peak FLOPs but a software gap; a **tile-based abstraction** closes
most of it. The AMD-specific lessons worth internalizing: **(a)** classic NVIDIA-style
wave specialization (producer/consumer) *underperforms* on AMD because there's no
register reallocation — prefer **8-wave / 4-wave** ping-pong/interleave; **(b)** LDS
bank conflicts need **per-shape swizzles**, there is no universal one; **(c)** **chiplet
(XCD) scheduling** drives cache reuse on multi-XCD parts; **(d)** small MFMA shapes + the
2× register file are AMD's path to **deep software pipelines**. Pick by measured
bottleneck, never by default.

---

## 4. Technique taxonomy (HipKittens — MIT; pointers under `3rdparty/HipKittens/` *if present*)

"Borrow the idea" unless noted. If the repo isn't checked out, the description is
enough to implement the idea; clone it only to dig deeper.

| # | Technique | Kernel kind | Bottleneck it helps | Arch | Pointer (if repo present) |
|---|-----------|-------------|---------------------|------|---------------------------|
| 1 | **8-wave ping-pong** — 2 wavegroups, one does MFMA while the other loads; `s_barrier` + `s_setprio` alternate them | GEMM, attn fwd | global latency; balanced compute/mem | CDNA3+4 | `kernels/gemm/fp8fp32/FP8_8wave/8_wave.cu:13-154` |
| 2 | **4-wave interleave** — 1 wave/SIMD, fine-grained load/MFMA interleave (`do_interleaved_cluster`); smaller tiles, lower regs | FP8 GEMM (compute-heavy), attn bwd (register-heavy) | occupancy; L2; imbalanced pipelines | CDNA4 best; CDNA3 limited | `kernels/gemm/fp8fp32/FP8_4wave/4_wave.cu:18-83`, `:186-214` |
| 3 | **XCD chiplet swizzle** — remap `blockIdx` so consecutive blocks stay on one XCD; + L2 windowed traversal. +18–55% on big GEMM | large GEMM | L2/LLC locality | CDNA4 (8 XCD); **N/A single-chiplet** | `kernels/gemm/fp8fp32/FP8_4wave/4_wave.cu:114-133` |
| 4 | **Bank-conflict-free LDS swizzle** — per-layout XOR masks, e.g. `offset ^= ((offset%512)>>7)<<3` for bf16 16×16 | any LDS user | LDS bank conflicts | CDNA3+4 (masks differ by gen) | `include/cdna4/types/shared/st_shape.cuh:48-146` |
| 5 | **Direct global→LDS** — `llvm_amdgcn_raw_buffer_load_lds`; bypass VGPRs; light `s_waitcnt vmcnt` | GEMM, attn | global latency; register pressure | CDNA3+4 (**not CDNA2**) | `include/cdna4/ops/warp/memory/tile/global_to_shared.cuh:36-69` |
| 6 | **Precomputed swizzled offsets** — `prefill_swizzled_offsets`; amortize address math across many loads | multi-load loops | VALU / instruction count | all | `.../global_to_shared.cuh:117-149`; used at `FP8_8wave/8_wave.cu:75-78` |
| 7 | **MFMA shape selection** — `mma_ABt`/`mma_AtB` wrappers dispatch on dtype/shape | compute-bound | throughput vs regs | CDNA3+4 | `include/cdna4/ops/warp/register/tile/assembly/mma.cuh:24-130`; builtins `include/cdna4/common/macros.cuh:774-1107` |
| 8 | **Register-tile (accumulator) layouts** — `rt_16x16/32x32/16x32`, col_l vs row_l to cut transposes | GEMM / attn | transpose cost; regs | all | `include/cdna4/types/register/rt_shape.cuh:20-47` |
| 9 | **Wave priority + scheduler barriers** — `s_setprio`, `sched_barrier`, `sched_group_barrier` to force MFMA/VMEM/DS ordering | high-occupancy compute | ILP; load/compute overlap | CDNA4 rich; CDNA3 limited | `kernels/attn/gqa_backwards/attn_fwd_non_causal.cpp:41-55`; `FP8_8wave/8_wave.cu:116-130` |
| 10 | **Transpose LDS reads** — `ds_read_b64_tr_b16/_b8/_b4` to match MFMA layout with no explicit transpose | shared→register | LDS latency; instr count | CDNA3+4 (`_trN` differ by gen) | `include/cdna4/common/macros.cuh:238-288`; `.../shared_to_register.cuh:248-267` |
| 11 | **Wave-vote reductions** — `__all()` instead of `ds_bpermute` ladders | attention softmax / reductions | sync cost | all | `kernels/attn/gqa/kernel.cpp:71-85` |
| 12 | **In-register elementwise math** — keep tiles in registers, avoid LDS round-trips, unroll over packed elems | memory-bound (rotary/layernorm/softmax) | bandwidth; LDS capacity | all | `kernels/layernorm/kernel.cpp:57-126`; `kernels/rotary/kernel.cpp:44-74` |
| 13 | **Developer-controlled register pinning** — work around HIPCC's "AGPR not an MFMA input" limitation | register-heavy (attn bwd) | register-file utilization | CDNA3+4 | paper §3.2.1, App. D.3 (`register_ranges` + tile struct) |
| — | **ANTI-PATTERN: wave specialization (producer/consumer)** — underperforms on AMD (no reg realloc → wasted producer regs). Use 8-wave/4-wave instead. | GEMM | — | CDNA3+4 | paper §3.3.1, Table 2 |

### 8-wave vs 4-wave — the choice (paper Table 3)

- **8-wave**: fewer lines of code, **large tiles**; great for **balanced or
  latency-bound** kernels (GEMM forward, attention forward).
- **4-wave**: more code, **small tiles + low registers**; wins on **compute-heavy FP8
  GEMM** and **register-heavy attention backward** (e.g. MHA bwd: 4-wave 1091 vs
  8-wave 894 TFLOP/s; GQA non-causal bwd ~1.8–2.3× over AITER).
- Decide by **bottleneck**: register-heavy / occupancy-starved → lean 4-wave;
  latency-bound with headroom → 8-wave.

### Headline results — context for how ambitious to be, NOT a swap target

vs AITER asm / rocBLAS / hipBLASLT / CK / Triton / PyTorch SDPA: GEMM hotloop <100 LoC
competitive with AITER/hipBLASLT; attn fwd 1.0–2.1× AITER and 1.3–4.5× SDPA; attn bwd
1.8–2.5× all baselines; fused dropout-residual-layernorm & RoPE 1.1–2.2× AITER; XCD
swizzle +18.7% TFLOP/s, +55% BW on big GEMM. Measured on MI325X (CDNA3),
MI350X/MI355X (CDNA4), ROCm 7.0. Use these to judge whether your kernel still has
headroom — then go get it in `EDIT_TARGET`.

---

## 5. Bottleneck → idea lookup (drive off the digest's label)

Read `profile/digest.json`: take `bottleneck` + the `metrics`/`notes`, then:

| digest signal | meaning | candidate ideas (also gate on kernel kind) |
|---------------|---------|---------------------------------------------|
| `bottleneck = "bandwidth-bound"` (high `MemUnitBusy`+`MemUnitStalled`) | memory pipe saturated | generic first: coalesce, `float4`, cut traffic. Tile/TC kernels: **#5** direct→LDS, **#12** in-register math |
| `bottleneck = "latency-bound"` (low `VALUBusy`) | not enough work in flight | **#1** 8-wave ping-pong, **#9** scheduler barriers, **#2** 4-wave (raise occupancy), **#6** precompute offsets |
| `bottleneck = "compute-bound"` (high `VALUBusy`) | math-limited | **#7** MFMA shape selection, **#8** register-tile layouts (cut transposes), **#2** 4-wave for FP8 |
| `LDSBankConflict > 0` | shared-mem bank conflicts | **#4** LDS XOR swizzle, **#10** transpose LDS reads; generic `+1` pad as fallback |
| `L2CacheHitRate` low | poor locality / blocking | **#3** XCD swizzle (multi-XCD only), better tiling, **autotune** the tile dims |
| low occupancy / spills (`-Rpass-analysis`) | too many regs/wave | **#2** 4-wave (low regs), **#13** register pinning, `__launch_bounds__`, shrink micro-tile |
| reduction/softmax sync cost | ladder reductions dominate | **#11** wave-vote reductions |

Rule: try the **cheapest matching idea** first (often a generic one or an autotune
sweep), re-profile, escalate only if the bottleneck persists.

---

## 6. aiter (ROCm/aiter — MIT) — an IDEA SOURCE (not a baseline, not a candidate)

aiter is AMD's production op library (CK-generated kernels + FlyDSL JIT + Triton + hand
asm + OPUS C++ templates), consumed from Python. Use it **only** to mine ideas and
config seeds. It is **not** the run's reference and **not** something to hand back —
re-read §0.

### 6a. Mine techniques from its source (pointers under `3rdparty/aiter/` *if present*)

- Online-softmax fused MHA pipeline — `csrc/kernels/mha_native/fused/pipeline.hpp:9-60`
- Split-K / persistent GEMM scheduling (CK-generated + `tuned_gemm` paths)
- MXFP4/FP8 quant + MoE fusion — `aiter/fused_moe.py`, `aiter/ops/quant.py`
- OPUS MFMA C++ templates — `csrc/include/opus/opus.hpp` (OPUS kernels are gfx950-only)
- BF16/A8W8 GEMM, flash-attn, MLA, RMSNorm entry points — `aiter/tuned_gemm.py`,
  `aiter/ops/mha.py`, `aiter/mla.py`, `aiter/ops/rmsnorm.py`

### 6b. Seed our autotune from its tuned config tables

aiter ships **tuned CSV tables** keyed by `(gfx, cu_num, M, N, K, dtype, ...)` →
`(kernelId, splitK, us, kernelName, tflops, bw)`. When your kernel exposes the same
knobs, these are a strong **seed** for `harness.py autotune`'s config space — don't
guess tile/split-K from scratch when aiter already tuned that arch+shape.

- Example table: `aiter/configs/a8w8_tuned_gemm.csv`
- Loader + fallback: `aiter/tuned_gemm.py:59-79, 117-162` (env-overridable
  `AITER_CONFIG_GEMM_*`; capture shapes with `AITER_TUNE_GEMM=1`)
- **Arch-aware codegen** reminder: `aiter/ops/flydsl/gemm_kernels.py:44`
  (`KERNEL_ASYNC_COPY = get_rocm_arch() != "gfx942"`) — the right knob flips by arch;
  match the calibrated `ARCH`.

### 6c. Optional headroom check (report-only)

If aiter is importable, you *may* time the matching op on the same shape purely to see
**how much headroom remains** and to record it in `REPORT.md`. This does **not** change
the gate or the deliverable: keep optimizing `EDIT_TARGET` toward beating the run's own
reference by X%. If aiter isn't importable, skip it — never block the loop on it.

```python
# report-only; NOT the gate:
from aiter.test_common import run_perftest
import aiter
out, us = run_perftest(aiter.gemm_a16w16, A, B, bias=None, otype=torch.bfloat16)
```

Arch coverage: aiter targets **gfx942 + gfx950** first-class (OPUS = gfx950 only);
**gfx90a is not first-class** — on CDNA2 treat aiter paths as may-not-apply.

---

## 7. Counter-set caveat across arch

`profiling/counters/default.txt` is curated for **gfx942**. Counter names/availability
**differ by arch** — some derived metrics on gfx942 may be absent or renamed on gfx90a
/ gfx950. At calibrate, after detecting `ARCH`, dry-run the counter file and check for
all-null columns; drop/rename the unavailable counters for that arch and note it in
`<problem>.context.md`. Keep `default.txt` as the CDNA3 baseline.

---

## 8. Sources (distilled once; not shipped/fed — clone yourself to dig further)

- HipKittens paper — `2511.08083v1.pdf` (arXiv 2511.08083), §3.2–3.3, Tables 2–3, App. D.
- HipKittens source — `github.com/HazyResearch/HipKittens` (MIT). *(May not be local.)*
- aiter source — `github.com/ROCm/aiter` (MIT). *(May not be local.)*
- AMD kernel-optimization blogs (e.g. "AMD GPUs go brrr" / the HipKittens release post).
