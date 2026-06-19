# Goal-Loop Engine (`_BASE`)

You are running an **autonomous goal loop**: keep changing code and re-measuring
until an objective, machine-checked gate passes — or until you run out of budget,
at which point you stop and report honestly.

This file is the shared engine. A *runner* command supplies the task-specific
**bindings** below before including this file. Never invent bindings that the
runner did not provide; if a required one is missing, stop and ask.

## Bindings (provided by the runner)

| Binding | Meaning |
|---|---|
| `EDIT_TARGET` | The file(s) you are allowed to modify to improve the metric. |
| `VERIFY_CMD` | Command that builds + measures and writes `result.json`. The objective gate. |
| `RESULT_JSON` | Path to the canonical result the verify command writes. |
| `TARGET` | Success threshold for this run (passed to `VERIFY_CMD`). |
| `MAX_ITERS` | Hard cap on optimize→verify iterations (default 15). |
| `PROFILE_CMD` | *(optional)* Command for deeper diagnostics. Empty ⇒ skip. |
| `ARTIFACTS_DIR` | Where per-iteration logs, the best-so-far snapshot, and `REPORT.md` go. |
| `RUN_CONTEXT` | *(optional)* Free-text note the user gave for THIS run. May be empty. |

## The gate contract (non-negotiable)

The only source of truth for "did it work" is `RESULT_JSON`, written by `VERIFY_CMD`:

```json
{ "pass": <bool>, "metrics": { ... }, "summary": "<string>" }
```

- `pass` is the **hard gate**. You stop successfully only when `pass == true`.
- `metrics` is raw data you read to decide the next idea. Never recompute the
  verdict yourself or eyeball numbers — trust `pass`.
- You did **not** succeed unless the verify command ran and `pass == true` on its
  own output. Editing the metric, the harness, or `RESULT_JSON` is cheating and
  is forbidden (see Guardrails).

## Default loop

Run this until `pass == true` or you hit `MAX_ITERS`:

1. **Understand.** On iteration 1, read `EDIT_TARGET`, the verify/harness code
   (read-only), `RUN_CONTEXT`, and the runner's context file. State a baseline:
   run `VERIFY_CMD` once unchanged to capture the starting `metrics`.
2. **Hypothesize.** Pick ONE concrete, motivated change likely to move the metric.
   Write down the rationale (one or two lines) before editing.
3. **Act.** Apply the change to `EDIT_TARGET` only.
4. **Measure.** Run `VERIFY_CMD` with `TARGET`. Read `RESULT_JSON`.
5. **Decide.**
   - `pass == true` → stop, write `REPORT.md`, report success.
   - regressed vs best-so-far (or build/correctness broke) → **roll back** to the
     best-so-far snapshot, log why, try a different idea.
   - improved but not passing → keep it as the new best-so-far.
6. **Diagnose (if stuck).** If two iterations bring no improvement and `PROFILE_CMD`
   is set, run it and let the profile drive the next hypothesis.
7. Loop.

## Guardrails (always on)

- **Only edit `EDIT_TARGET`.** Never modify the harness, the verify adapter,
  `RESULT_JSON`, baselines, or test cases. If a fix seems to require touching the
  measurement, STOP and ask the user — do not do it silently.
- **Correctness first.** A faster-but-wrong result is a failure. If correctness
  breaks, the iteration failed regardless of speed.
- **Always keep a best-so-far.** Before each edit, snapshot `EDIT_TARGET`. Never
  let the working file end up worse than the best verified state.
- **Respect `MAX_ITERS`.** Do not exceed it. Count build failures as iterations.
- **No destructive / out-of-scope actions** (deleting unrelated files, touching
  files outside `EDIT_TARGET`, network installs) without asking.
- **Be honest.** If you did not reach the target, say so plainly with the numbers.

## Artifacts (write under `ARTIFACTS_DIR`)

- Per iteration: append a line to `ARTIFACTS_DIR/log.md` —
  `iter N | hypothesis | key metrics | kept/rolled-back`.
- `ARTIFACTS_DIR/best/` — copy of the best-so-far `EDIT_TARGET` + its `result.json`.
- On exit (pass or give-up): write `ARTIFACTS_DIR/REPORT.md` with: starting vs
  final metrics, what worked / what didn't, the diff that won, and (if not passed)
  the best result reached + concrete next ideas.

## Escalation — stop and ask the user when

- `MAX_ITERS` reached without `pass`. Report best-so-far + next ideas; don't loop on.
- The verify command itself is broken (build error in the harness, not the kernel;
  malformed `result.json`). This is a harness bug — surface it, don't work around it.
- You have **literally no idea left to try**, or a needed change would cross a guardrail.

**Do NOT escalate early just because the remaining ideas feel hard or low-confidence.**
A lower-confidence idea still counts as an idea: keep iterating — apply it, measure it,
roll back if it regresses — until `pass == true` or you actually hit `MAX_ITERS`. Running
out of *high-confidence* ideas is not the same as being stuck. The hints in the runner are
only a starting menu; exhausting them is not a stopping condition while iterations remain.
"This is a steep target and my next idea might not work" is the normal state of the loop,
not a reason to stop and ask. The only budget that ends the loop is `MAX_ITERS`.
