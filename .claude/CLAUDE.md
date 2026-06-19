# CLAUDE.md

## What this repo is

A small framework for running **autonomous "goal loops"** in Claude Code: tasks that keep
editing code and re-measuring until an **objective, machine-checked gate** passes (or a
budget runs out). The aim is to let the user kick off a convergent task with a **short
command + a few inputs** instead of hand-writing a long prompt every time.

## Why it exists (the goal)

The user repeatedly hits tasks of the shape *"iterate on code until a measurable target is
met"* — e.g. optimize a kernel until it reaches some score, fix/extend code until it serves
and passes an eval, investigate a behavior and fix it. Writing a fresh long prompt for each
is error-prone. This repo turns that pattern into **reusable skills**: define the task type
once, then invoke it with just the variable inputs.

Key ideas this design rests on:
- **goal = agent loop + a stopping condition.** The repetition is already in the agent's
  inner reason→act→observe loop once it drives tools itself; the "goal" only adds an
  objective gate that decides when to stop. (It is *not* `loop + condition`.)
- **workflow vs agent.** Deterministic parts (build, measure, parse) are frozen into
  scripts; the creative part (generating ideas) is left to the agent loop. Mix both.
- **one source of truth for success.** A single `result.json {pass, metrics, summary}`
  written by a verify command. Trust `pass`; never recompute or edit the metric.

## Architecture (3 layers)

Everything follows the **Claude Code skills standard**: each invokable piece is a skill
directory `.claude/skills/<name>/SKILL.md`; the shared engine is a plain included file.

1. **Engine** — [.claude/engine/_BASE.md](.claude/engine/_BASE.md): the shared loop,
   the gate contract, guardrails, artifacts, and escalation rules. Runners supply
   task-specific **bindings** then `@`-include this file (relative to the skill dir). The
   loop logic lives here once. (Not a skill — just shared markdown.)
2. **Generator** — [.claude/skills/loop-tasks/SKILL.md](.claude/skills/loop-tasks/SKILL.md):
   interviews the user about a new task type and **emits a new runner skill** bound to the
   engine.
3. **Runners** — `.claude/skills/<task>/SKILL.md` (first one:
   [optimize-kernels](.claude/skills/optimize-kernels/SKILL.md)): a concrete task type.
   Interactive — it **asks** for its inputs (reference files, target, extra context) in one
   batched prompt; only `max-iters` is the `$ARGUMENTS` value. Profiling is **not** an input —
   the runner drives it automatically. It runs a one-time **calibrate phase** (read +
   correct + validate the source, generate the verify adapter + the agent-controlled
   `harness.py` + profiling + per-problem context, capture a baseline, get sign-off) and
   then runs the loop. A runner **bundles its own helper files inside the skill dir**, split
   by concern — `profiling/` (counters + digest), `tuning/` (autotune), `templates/`, and
   `references/` (distilled idea docs, e.g.
   [optimize-kernels/references/](.claude/skills/optimize-kernels/references/)) — invoked at
   runtime via `${CLAUDE_SKILL_DIR}/...`.

## Conventions

**Provenance split** — every file has a clear owner:
- `<source-dir>/<problem>/` (e.g. `optimize-kernels/<problem>/`) = **original source**
  (the user's): the file(s) under test + their measurement harness. How the loop treats them
  — edit in place vs keep pristine and work on a copy — is the **runner's** choice.
- `artifacts/<problem>/` = **everything Claude generates** (owner: Claude), **always at the
  project root** (never nested in the source tree). For optimize-kernels that's the working
  kernel copy the loop edits, `verify.py` (gate adapter), `harness.py` (measurement wrapper),
  `profile/`, `build/`, `runs/`, `result.json`, `autotune.json`, `<problem>.context.md`,
  `REPORT.md`.
- `.claude/` = the goal-loop machinery: the engine (`.claude/engine/`), the runner skills
  (`.claude/skills/<name>/SKILL.md`) **with their own bundled helper files** split by concern
  (e.g. optimize-kernels: `profiling/`, `tuning/`, `templates/`, `references/`), and settings.
  No per-problem source, no artifacts.

**Correct-first, then freeze the gate** — before looping, make the thing being measured
correct (fix genuine bugs in the gate/measurement at calibrate, with approval) and then
**freeze it**, so the loop can't change what "success" means. *How each runner protects the
user's source is runner-specific* — e.g. optimize-kernels keeps the original kernel + harness
pristine and has the loop edit a copy in `artifacts/` (details in its
[SKILL.md](.claude/skills/optimize-kernels/SKILL.md)).

**Gate contract** — `result.json = {pass, metrics, summary}`. `pass` is the only success
signal. The verify adapter *drives* the original harness; it never reimplements the metric.

**Guardrails (in the engine, always on)** — edit only the target file; snapshot and keep a
best-so-far; roll back on regression; correctness before speed; respect `max-iters`; never
touch the measurement/harness/baseline; never delete or move files outside `artifacts/`
(`CLAUDE.md`, `.claude/`, the source tree are off-limits); measure only on an idle GPU (busy
→ poll 30s up to 30 min, else checkpoint + stop); report honestly with numbers.

**Resumable across sessions** — the engine checkpoints to disk every iteration
(`loop_state.json` + `best/`), and only "commits" an iteration once the gate result is
written, so an interrupted one is retried, not lost. A run therefore survives a
session/budget window (even an account switch); continue it from disk with the runner's
resume mode (e.g. `/optimize-kernels --resume`), which skips calibrate and picks up from
the checkpoint.

## What the user provides

- **To create a runner** (`/loop-tasks`): name + verb, stable context, run-input contract,
  benchmark/gate command + baseline source (computed/supplied/none), profile (optional),
  strategy hints, target shape, guardrails, max-iters default.
- **To run `optimize-kernels`**: reference files, target, extra context — asked together in
  one prompt; `max-iters` is the slash-command argument. Profiling is automatic (the runner
  drives rocprofv2/rocprofv3 itself and picks the counters), not a user input.

## Usage

- `/loop-tasks` — interview to generate a new runner skill for a task type.
- `/optimize-kernels [max-iters]` — run the kernel-optimization goal loop (it asks for the
  rest of its inputs).
- `/optimize-kernels --resume [problem] [max-iters]` — continue an interrupted run from its
  on-disk checkpoint instead of starting a new one (survives session/budget windows and even
  account switches; reads only `artifacts/`).

## Permissions

- A runner's frontmatter `allowed-tools` pre-authorizes its tools. **It is read at
  invocation time**, so it applies to the *next* invocation, not a loop already running.
- `Bash` is intentionally left OUT of `optimize-kernels`'s `allowed-tools` (the exact
  commands are problem-specific, not generic). Approve the verify command once with "don't
  ask again" and the loop stops prompting for it.
- Project allow-rules can also live in `.claude/settings.json`.

## Environment

- Runs **inside a Docker container** on an AMD **ROCm** box (ROCm ~7.2) with `hipcc`,
  hipBLAS, `rocprofv2`, `rocprofv3` (1.1.0), `rocm_agent_enumerator`, and `rocminfo`. The
  **GPU arch is not fixed** — it may be CDNA2 (`gfx90a`), CDNA3 (`gfx942`, MI300X), or CDNA4
  (`gfx950`, MI350/MI355). The runner **detects it at calibrate** and builds with
  `--offload-arch=<detected>`; counter sets are validated per arch. (Earlier end-to-end
  validation was on `gfx942`.)

## Status — how far we've got

- **Engine, generator, and the `optimize-kernels` runner are built and working** (skills
  standard).
- Validated end-to-end on a sample problem: interview → calibrate (auto-generate `verify.py`
  + `harness.py` + profiling + per-problem context) → baseline + sign-off → autonomous loop
  with best-so-far / rollback / per-iteration logging → `REPORT.md`. (Earlier profiling +
  `allowed-tools` runs were on `gfx942`.)
- The runner carries its own helper library, **split by concern** under
  [.claude/skills/optimize-kernels/](.claude/skills/optimize-kernels/): `profiling/`
  (`profile_digest.py`, `merge_v2.py`, `rocprofv3_digest.py`, curated `counters/default.txt`),
  `tuning/` (`autotune.py`), `templates/` (`harness_template.py`), `references/`
  (`amd-techniques.md`). Unit-tested without a GPU.
- Profiling is **always-on and self-driven**; arch is **detected at calibrate** (CDNA2/3/4)
  with AMD hints arch-tagged + bottleneck-gated. Engine-level: the loop is **resumable**
  across sessions/accounts (`--resume` + disk checkpoint) and **pauses on a busy GPU**
  (poll 30s, 30-min timeout → checkpoint + stop).

## Next / open ideas (to do in a NEW session)

System-level directions:
- Use `/loop-tasks` to author the other task types we scoped (fix-until-serving+eval-passes;
  investigate-then-fix), reusing the same engine.
- Tighten the calibrate phase (source validation + auto-fix of missing/mis-named hooks).
- Smoother permissions so a fresh runner invocation needs zero prompts.
