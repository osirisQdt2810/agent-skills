---
name: loop-tasks
description: Interview-driven generator — turn a task description into a new goal-loop skill (a runner bound to the shared engine).
argument-hint: [free-form task description, optional]
---

You are running **`loop-tasks`**, the generator. Your job is NOT to optimize anything
now — it is to **interview the user and emit a new runner skill** at
`.claude/skills/<skill>/SKILL.md` that binds the shared engine `.claude/engine/_BASE.md`.
Following the Claude Code skills standard, each runner is its own skill directory; bundle
any task-specific helper scripts inside it (e.g. `.claude/skills/<skill>/profiling/`) and
invoke them via `${CLAUDE_SKILL_DIR}/...` at runtime.

If `$ARGUMENTS` is non-empty, treat it as an initial task description: parse out
whatever fields you can, then ask ONLY for what's missing. Otherwise start from scratch.

## Interview (ask ONE field at a time, wait for the answer, stay adaptive)

Ask in this order. Skip anything already answered. Echo back your understanding before
moving on.

1. **Skill name + verb.** What should the new skill be called, and which verb is it?
   (optimize-to-a-number / fix-until-pass / investigate-then-fix). The verb sets the
   default loop and gate semantics. *(Use AskUserQuestion for the verb.)*
2. **Context (stable).** Domain knowledge that never changes between runs — environment,
   conventions, anything the agent must always know. Goes into a context file, written once.
3. **Run-input contract.** Which file(s) must the user pass on each run (e.g. the file to
   edit + the harness)? This is what the runner validates at start.
4. **Benchmark / gate.** The command(s) that build + measure and the **baseline source**
   (`computed` each run / `supplied` value / `none` = pass-fail only).
   *(Use AskUserQuestion for the baseline source.)* This drives the **calibrate phase**: you
   will draft a `verify.py` adapter that runs these and writes `result.json = {pass, metrics,
   summary}`, and show the user a sample `result.json` for sign-off.
5. **Profile (optional).** A diagnostic command for when the loop gets stuck. Empty ⇒ none.
6. **Custom strategy hints (optional).** "Try X", "avoid Y", "debug like Z".
7. **Target shape.** Which metric(s) define success and their default thresholds. (The
   concrete value is passed at runtime, not baked.)
8. **Guardrails.** Which file(s) are the ONLY ones the loop may edit; what is off-limits
   (the harness, the verify adapter, result.json, baselines/tests are always off-limits).
9. **MAX_ITERS default** (default 15).

## Emit the skill

When the fields are gathered:

1. Write `.claude/skills/<skill>/SKILL.md` modeled on the existing
   `.claude/skills/optimize-kernels/SKILL.md`: frontmatter (`name`, `description`,
   `argument-hint`, `allowed-tools`), then an **interactive input-gathering step** — the
   runner ASKS the user for its run inputs (reference file(s), target, extra context) and
   takes only **`max-iters`** from `$ARGUMENTS`; then conventions, a numbered **calibrate
   phase** (read inputs → correct + validate the source can be measured, fixing genuine bugs
   with approval then freezing the original → generate the `verify.py` adapter + any
   diagnostics + per-problem context → baseline + sign-off), the captured strategy hints,
   then a **bindings block** (`EDIT_TARGET`, `VERIFY_CMD`, `RESULT_JSON`, `TARGET`,
   `MAX_ITERS`, `PROFILE_CMD`, `ARTIFACTS_DIR`, `RUN_CONTEXT`) followed by the engine include
   `@../../engine/_BASE.md` (relative paths in an `@`-include resolve from the skill dir).
2. If the task type needs task-specific helper scripts, bundle them under
   `.claude/skills/<skill>/` (e.g. a `profiling/` package) and have the SKILL.md reference
   them via `${CLAUDE_SKILL_DIR}/...` — keep them generic over problem instances.
3. Keep the new skill **generic** over its problem instances — bake only stable task-type
   knowledge, never one specific problem's details.
4. Show the user the generated file for approval. Do not run the loop here — once approved,
   the user invokes `/<skill>` themselves with their inputs.

Reuse the shared engine; never duplicate the loop/guardrails/artifacts logic into the new
skill — that all lives in `.claude/engine/_BASE.md`.
