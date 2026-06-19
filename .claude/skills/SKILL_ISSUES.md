# Skill issues found while running `/optimize-kernels` (do NOT edit `.claude/` during a run)

This file collects bugs / rough edges in the **original skills under `.claude/`** that I hit
while running a goal loop. Per the user's instruction, I do **not** edit `.claude/` during a
run — I work around the issue inside `artifacts/<problem>/` and log it here so the skills can
be updated later in a dedicated session.

Format: one entry per issue. Each says **where**, **what's wrong**, **how it shows up**, the
**workaround** I used in `artifacts/`, and a **suggested fix** for the skill.

---

## 1. `profiling/merge_v2.py` — off-by-one mapping of pmc dirs → counter lines (rocprofv2 7.2)  ✅ FIXED in `.claude/`

> Status: this is a **generic** (kernel-independent) bug, so per the user's instruction it was
> fixed directly in `.claude/skills/optimize-kernels/profiling/merge_v2.py` (and recorded here).
> `merge()` now selects counter columns by their actual presence in each CSV (columns minus
> `COMMON_COLS`) instead of by the `pmc_<i>` directory index, so the extra info pass is handled.
>
> Root cause of the "4 pmc lines → 5 folders" puzzle: this box's **rocprofv2 (HIP 7.2)** runs an
> extra leading **info/discovery pass** (`pmc_1`, no counter columns) before the N real counter
> passes *when the counter file has more than one `pmc:` line*. With a single `pmc:` line it does
> NOT add the extra pass (1 line → 1 folder). Verified empirically.

- **Where:** `.claude/skills/optimize-kernels/profiling/merge_v2.py`, `merge()` (the line
  `counters = pmcs[i - 1] if i - 1 < len(pmcs) else []`).
- **What's wrong:** `merge()` assumes the per-pass CSV directory `pmc_<i>` holds the counters
  of the *i-th* `pmc:` line (`pmcs[i-1]`). On this box's **rocprofv2 (HIP 7.2.x)**, rocprofv2
  emits a **spurious metadata-only `pmc_1`** (columns: only `Dispatch_ID … Correlation_ID`,
  no counters) and shifts the real counter passes to `pmc_2 … pmc_{N+1}`. So the actual
  contents are:
  - `pmc_1` → (no counters, metadata only)
  - `pmc_2` → line 1 counters (`FetchSize, WriteSize, MemUnitBusy, MemUnitStalled`)
  - `pmc_3` → line 2 counters (`L2CacheHitRate, sL1dCacheHitRate, WriteUnitStalled`)
  - `pmc_4` → line 3 counters (`VALUBusy, SALUBusy, VALUUtilization`)
  - `pmc_5` → line 4 counters (`LDSBankConflict, ALUStalledByLDS, Wavefronts`)

  But `merge()` looks for line-2 counters in `pmc_2` (which actually has line-1 counters),
  line-3 in `pmc_3`, etc. Since `cols = [c for c in counters if c in df.columns]` then finds
  **no matching columns**, every counter is dropped.
- **Symptom:** `merge_v2.py` prints `merged N rows` but the digest has `"metrics": {}` and
  `"bottleneck": "unknown"` — duration is correct, all counters are missing. (The raw
  `pmc_*/results_*.csv` files DO contain valid counter values.)
- **Workaround (in `artifacts/fp8_paged_mqa_logits/profile/profile.py`):** do the merge
  locally without trusting the `pmc_<i> → pmcs[i-1]` index. For each `pmc_*` CSV, pick up
  **whatever counter columns are actually present** (any column beyond the known
  `COMMON_COLS`), merge them all by `Dispatch_ID`, filter to the kernel symbol, then feed the
  records to `profile_digest.summarize()` (which is fine and reused as-is).
- **Suggested skill fix:** in `merge()`, don't map by directory index. Instead, for each
  `pmc_*` CSV take the set-difference of its columns against `COMMON_COLS` (+ derived) as the
  counter columns to merge. That is robust to (a) the spurious metadata pass and (b) any
  reordering/renumbering of passes across rocprof versions. Optionally also skip a pass whose
  only columns are `COMMON_COLS`.

---
<!-- add further issues below -->
