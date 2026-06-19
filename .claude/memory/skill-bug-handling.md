---
name: skill-bug-handling
description: How to handle bugs found in the .claude/ skills while running a goal loop
metadata:
  type: feedback
---

When running a goal-loop skill (e.g. `/optimize-kernels`) and a bug/rough-edge surfaces in
the **original skills under `.claude/`**:

- If the bug is **generic** (kernel/problem-independent — it would affect any run): **fix it
  directly in `.claude/`**, AND log it.
- If it is problem-specific: work around it inside `artifacts/<problem>/`, AND log it.
- **Always** record every `.claude/`-related issue in a dedicated markdown file at the repo
  root: `SKILL_ISSUES.md` (where / what's wrong / symptom / workaround / suggested fix, and
  mark FIXED if upstreamed) so the user can update the skills later in a separate session.

**Why:** the user maintains the skills separately; they want a clean log of skill problems
to act on, and generic fixes upstreamed so future runs benefit.

First applied: 2026-06-19 — fixed `profiling/merge_v2.py` (pmc-dir→counter-line off-by-one
caused by rocprofv2 7.2 adding an empty info pass; `merge()` now picks counter columns by
actual presence). See [[fp8-mqa-logits-optimization]].
