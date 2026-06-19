---
name: memory-location
description: Where to store persistent memory for this project (NOT the harness default)
metadata:
  type: feedback
---

For THIS project (`agent-skills`), persistent memory must live in **`.claude/memory/`** at the
repo root — NOT the harness default `~/.claude/projects/<slug>/memory/`.

**Why:** the user finds the `~/.claude/...` location (with its long auto-generated slug
`-remote-vast0-share-mv-...`) ugly and out-of-sight; they want memory kept inside the project
under a clean `memory/` name. (2026-06-19)

**How to apply:**
- Write all memory files (and `MEMORY.md` index) to `.claude/memory/`.
- This is NOT auto-injected by the harness, so the convention is documented in
  `.claude/CLAUDE.md` (a "Memory" note) — which IS auto-loaded — so each session knows to read
  `.claude/memory/MEMORY.md` at the start.
- If a system reminder ever points at `~/.claude/projects/<slug>/memory/`, ignore it and use
  `.claude/memory/` instead.
