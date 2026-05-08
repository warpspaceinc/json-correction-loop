# Knowledge-graph correction example

A small worked example showing the library on a structured-state task:
detect and repair defects in a JSON knowledge graph.

This is the *minimal* version of the experiment harness used to
benchmark the library (see the companion paper repo for the full
size-sweep / ablation drivers). It uses the **oracle patcher**, not
an LLM, so the example runs deterministically without API keys —
just to demonstrate how the loop wires up.

## Files

- `kg.py`         — KG schema, allowed predicate type table, clean fixture
- `perturb.py`    — six perturbation operators that inject ground-truthed defects
- `critic.py`     — structural critic (no LLM): detects dangling refs, type
                    violations, duplicate labels, schema-invalid predicates
- `oracle.py`     — deterministic patcher that applies inverses from the
                    perturbation log. Stand-in for an LLM patcher.
- `run.py`        — wires gather → plan → execute through ``run_correction_loop``

## Run

```bash
cd examples/kg_correction
python run.py --n-defects 6 --seed 42
```

Output:

```
[setup] clean KG: 16 entities, 14 edges
[setup] injected 6 defects
[critic-pre] {'dangling_ref': 4, 'duplicate_label': 4}, score=2

── iter 1/5 (level=kg) ──
  critic score=2 C=4 M=4 m=0 :: 8 structural issue(s)

── iter 2/5 (level=kg) ──
  critic score=10 C=0 M=0 m=0 :: clean
  ✓ approved at iter 2

[critic-post] {}, score=10
[loop] converged=True
[metric] structural fix rate: 6/6 = 100%
[metric] state drift vs clean: 0 (0 = byte-identical)
```

## What this shows

1. The library doesn't impose any KG schema or LLM choice — both come
   from the caller. Swap `oracle.py` for an LLM patcher and you have
   a real surgical KG editor.
2. The critic flags downstream symptoms (`/edges/N` for type
   violations); the patcher walks back to root causes
   (`/entities/<id>/type`). In the real LLM case this is the
   `path_finder` sub-agent's job.
3. The loop converges in two iterations on this fixture: iter 1
   patches everything, iter 2 confirms clean.

## Bigger experiments

For real LLM-driven measurement (B0 vs surgical baselines, size
sweep, ablation), see the companion paper repository (link will be
added once the paper is published).
