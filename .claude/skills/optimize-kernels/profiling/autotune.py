#!/usr/bin/env python3
"""
Generic parameter-autotuning sweep: used by a per-problem harness.py to search a
configuration space (whatever knobs the kernel exposes — block/tile sizes, unroll
factor, launch bounds, vector width, ...) for the config that maximizes (or
minimizes) a measured metric — often enough to hit the target without inventing a
new algorithm.

It is intentionally tiny and dependency-free. The caller supplies:
  * a `space`: {param_name: [candidate values, ...]}
  * an `evaluate(config) -> (score, info)` callback that BUILDS + MEASURES the
    kernel for that config and returns a float score (or None / raises to mark
    the config invalid — e.g. it failed to build, spilled, or was incorrect).

Correctness stays the gate: `evaluate` must return None/raise for any config
that fails the correctness check, so the sweep never selects a wrong-but-fast one.
"""

from __future__ import annotations

import itertools
import json
import random


def grid(space):
    """Yield every config in the full cartesian product of `space`."""
    keys = list(space)
    for combo in itertools.product(*(space[k] for k in keys)):
        yield dict(zip(keys, combo))


def sample(space, n, seed=0):
    """Yield up to `n` distinct random configs from the space."""
    all_cfgs = list(grid(space))
    rng = random.Random(seed)
    rng.shuffle(all_cfgs)
    yield from all_cfgs[:n]


def sweep(space, evaluate, *, mode="grid", n=None, seed=0, maximize=True,
          on_result=None):
    """Run `evaluate` over the config space and return the best.

    Returns {"best_config", "best_score", "best_info", "results": [...]}.
    Each result is {"config", "score", "ok", "info", "error"}.
    """
    if mode == "random":
        cfgs = list(sample(space, n or 16, seed))
    else:
        cfgs = list(grid(space))
        if n:
            cfgs = cfgs[:n]

    results = []
    best = None
    for i, cfg in enumerate(cfgs):
        rec = {"config": cfg, "score": None, "ok": False, "info": {}, "error": None}
        try:
            out = evaluate(cfg)
            score, info = out if isinstance(out, tuple) else (out, {})
            if score is None:
                rec["error"] = "invalid (evaluate returned None)"
            else:
                rec.update(score=float(score), ok=True, info=info or {})
        except Exception as e:  # build/correctness failure -> skip this config
            rec["error"] = f"{type(e).__name__}: {e}"
        results.append(rec)
        if on_result:
            on_result(i, len(cfgs), rec)
        if rec["ok"]:
            better = (best is None
                      or (rec["score"] > best["score"] if maximize
                          else rec["score"] < best["score"]))
            if better:
                best = rec

    return {
        "best_config": best["config"] if best else None,
        "best_score": best["score"] if best else None,
        "best_info": best["info"] if best else {},
        "results": results,
    }


def default_logger(i, total, rec):
    cfg = ",".join(f"{k}={v}" for k, v in rec["config"].items())
    if rec["ok"]:
        print(f"[{i + 1}/{total}] {cfg} -> {rec['score']:.4f}")
    else:
        print(f"[{i + 1}/{total}] {cfg} -> SKIP ({rec['error']})")


def write_results(path, result):
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"-> {path}")
    if result["best_config"]:
        print(f"BEST {result['best_config']} score={result['best_score']:.4f}")
    else:
        print("BEST none (all configs failed)")
