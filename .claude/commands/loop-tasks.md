---
description: Interview-driven generator — turn a task description into a new goal-loop skill (a runner command bound to the shared engine).
argument-hint: [free-form task description, optional]
---

You are running **`loop-tasks`**, the generator. Your job is NOT to optimize anything
now — it is to **interview the user and emit a new runner command** under
`.claude/commands/<skill>.md` that binds the shared engine `.claude/loop-tasks/_BASE.md`.

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

1. Write `.claude/commands/<skill>.md` modeled on the existing `optimize-kernels.md`:
   frontmatter (`description`, `argument-hint`), then an **interactive input-gathering
   step** — the runner ASKS the user for its run inputs (reference file(s), target,
   profile?, extra context) and takes only **`max-iters`** from `$ARGUMENTS`; then
   conventions, a numbered **calibrate phase** (read inputs → validate the source can be
   measured & fix gaps with approval → generate the `verify.py` adapter + optional
   profiling + per-problem context → baseline + sign-off), the captured strategy hints,
   then a **bindings block** (`EDIT_TARGET`, `VERIFY_CMD`,
   `RESULT_JSON`, `TARGET`, `MAX_ITERS`, `PROFILE_CMD`, `ARTIFACTS_DIR`, `RUN_CONTEXT`)
   followed by `@.claude/loop-tasks/_BASE.md`.
2. Keep the new skill **generic** over its problem instances — bake only stable task-type
   knowledge, never one specific problem's details.
3. Show the user the generated file for approval. Do not run the loop here — once approved,
   the user invokes `/<skill>` themselves with their inputs.

Reuse the shared engine; never duplicate the loop/guardrails/artifacts logic into the new
skill — that all lives in `_BASE.md`.
