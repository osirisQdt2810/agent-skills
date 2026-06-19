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

1. **Engine** — [.claude/loop-tasks/_BASE.md](.claude/loop-tasks/_BASE.md): the shared loop,
   the gate contract, guardrails, artifacts, and escalation rules. Runners supply
   task-specific **bindings** then `@`-include this file. The loop logic lives here once.
2. **Generator** — [.claude/commands/loop-tasks.md](.claude/commands/loop-tasks.md):
   interviews the user about a new task type and **emits a new runner skill** bound to the
   engine.
3. **Runners** — `.claude/commands/<task>.md` (first one:
   [optimize-kernels.md](.claude/commands/optimize-kernels.md)): a concrete task type.
   Interactive — it **asks** for its inputs (reference files, target, profile?, extra
   context) in one batched prompt; only `max-iters` is a command argument. It runs a
   one-time **calibrate phase** (read + validate the source, generate the verify adapter +
   optional profiling + per-problem context, capture a baseline, get sign-off) and then
   runs the loop.

## Conventions

**Provenance split** — every file has a clear owner:
- `<source-dir>/<problem>/` (e.g. `optimize-kernels/<problem>/`) = **original source**
  (the user's): the file under optimization + its measurement harness. The loop may edit
  **only the designated target file**; the harness is read-only.
- `artifacts/<problem>/` = **everything Claude generates**: `verify.py` (the gate adapter),
  `profile/`, `build/`, `runs/`, `result.json`, `<problem>.context.md`, `REPORT.md`.
- `.claude/` = the goal-loop machinery only (engine + commands + settings). No source code,
  no artifacts.

**Gate contract** — `result.json = {pass, metrics, summary}`. `pass` is the only success
signal. The verify adapter *drives* the original harness; it never reimplements the metric.

**Guardrails (in the engine, always on)** — edit only the target file; snapshot and keep a
best-so-far; roll back on regression; correctness before speed; respect `max-iters`; never
touch the measurement/harness/baseline; report honestly with numbers.

## What the user provides

- **To create a runner** (`/loop-tasks`): name + verb, stable context, run-input contract,
  benchmark/gate command + baseline source (computed/supplied/none), profile (optional),
  strategy hints, target shape, guardrails, max-iters default.
- **To run a runner**: reference files, target, profile?, extra context — asked together in
  one prompt; `max-iters` is the slash-command argument.

## Usage

- `/loop-tasks` — interview to generate a new runner skill for a task type.
- `/optimize-kernels [max-iters]` — run the kernel-optimization goal loop (it asks for the
  rest of its inputs).

## Permissions

- A runner's frontmatter `allowed-tools` pre-authorizes its tools. **It is read at
  invocation time**, so it applies to the *next* invocation, not a loop already running.
- `Bash` is intentionally left OUT of `optimize-kernels`'s `allowed-tools` (the exact
  commands are problem-specific, not generic). Approve the verify command once with "don't
  ask again" and the loop stops prompting for it.
- Project allow-rules can also live in `.claude/settings.json`.

## Environment

- Runs **inside a Docker container** (see [docker.sh](docker.sh)) on an AMD **MI300X** box,
  arch **gfx942** (CDNA3), ROCm 7.2, with `hipcc`, hipBLAS, and `rocprofv2`.
- `~/.claude` is bind-mounted host→container, so Claude Code sessions/history/config are
  shared between host and container (sessions are local files keyed by `$HOME` + cwd, not by
  account).

## Status — how far we've got

- **Engine, generator, and the `optimize-kernels` runner are built and working.**
- The full flow has been **validated end-to-end** on a sample problem: interview →
  calibrate (auto-generate `verify.py` + rocprofv2 profiling + per-problem context) →
  baseline + sign-off → autonomous loop with best-so-far / rollback / per-iteration logging
  → `REPORT.md`. rocprofv2 profiling on gfx942 and the `allowed-tools` permission flow both
  work.
- A worked example problem lives under `optimize-kernels/` + `artifacts/` (kept as a
  reference of the layout and outputs).

## Next / open ideas

> The user has further improvement ideas to pursue in a **new session**. Record them here
> as they're decided.

Candidate system-level directions (not problem-specific):
- Use `/loop-tasks` to author the other task types we scoped (fix-until-serving+eval-passes;
  investigate-then-fix), reusing the same engine.
- Tighten the calibrate phase (source validation + auto-fix of missing/mis-named hooks).
- Smoother permissions so a fresh runner invocation needs zero prompts.
