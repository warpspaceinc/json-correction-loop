# Experiments

Empirical evaluation of the `gather → plan → execute` loop on a
synthetic **knowledge-graph (KG) correction** task. We perturb a
clean KG with a known defect log, run the loop until it converges
or hits a hardcap, and measure (1) whether the critic-flagged
defects are repaired, (2) cost in tokens / wall clock, and (3)
whether unflagged content drifts.

> All numbers below were collected with `gpt-4o-mini` via OpenRouter
> in May 2026. Reproducing requires the perturbation harness (see
> [Reproducing](#reproducing) at the end).

## TL;DR

| Condition | Fix% | Tokens | Wall | Iters |
|-----------|-----:|-------:|-----:|------:|
| **size=100** — full-regen baseline | **0%** | 73,740 | 435s | 5 (hardcap) |
| **size=100** — full sub-agent stack | **100%** | 17,117 | 26s | 4 |

At 100 entities the prevailing pattern (re-emit the entire JSON on
every critic pass) **repairs zero defects** because the model
saturates `max_tokens` mid-array and returns truncated output.
This loop, with `path_finder` + context narrowing, achieves
**100% fix rate at ~5× fewer tokens and ~17× faster**.

## Setup

### Fixture

A synthetic KG with controllable size:

- Entities: typed nodes with `id`, `type`, `label`, attributes.
  Labels are intentionally fictional (`Director 0001`,
  `Film 0042`) so the LLM has no training prior that "auto-corrects"
  paraphrase defects.
- Edges: typed predicates between entities. ~1.8 edges per entity.
- Sizes evaluated: 16/14, 50/90, 100/183 (entities/edges).

### Perturbation operators

Six operators inject ground-truthed defects:

1. `relation_swap` — swap predicate type on an edge.
2. `entity_drop` — delete an entity, leave edges dangling.
3. `entity_paraphrase` — rename an entity's label.
4. `type_violation` — flip an entity's type, breaking edge type
   constraints.
5. `dangling_ref` — change an edge's source/target to a non-existent
   id.
6. `duplicate` — duplicate an entity under a fresh id.

Each run perturbs the clean KG with a mix of operators (default 8
defects, mixed seeds), records the ground-truth defect log, then
runs the correction loop on the perturbed KG.

### Critic

A pure-Python structural critic (no LLM). Detects:

- `dangling_ref` — edge endpoint is missing.
- `type_violation` — edge predicate types don't match endpoint types.
- `duplicate_label` — two entities share a label.
- `schema_invalid` — required field missing / wrong shape.

Returns a `CriticReport` with one `CriticIssue` per defect, each
keyed by JSON pointer (`/entities/{id}` or `/edges/{idx}`) so the
planner has stable item IDs to operate on.

### Conditions

| Code | Condition | Components |
|------|-----------|------------|
| `oracle` | Upper bound | Deterministic inverse using the ground-truth defect log + a heal-fallback for cascade symptoms. No LLM. |
| `B0` | Full-regen baseline | LLM sees full perturbed KG + critic issue list, returns full corrected KG. |
| `B1` | Single-shot patch | One LLM call emits RFC 6902 ops; ops applied via `jsonpatch`. No loop. |
| `O1` | Loop + patcher | `B1` wrapped in the `gather → plan → execute` loop. No sub-agents. |
| `O2` | + `path_finder` | `O1` + path_finder maps symptom pointers to root-cause pointers. Full KG context to all calls. |
| `O2N` | + context narrowing | `O2` + scope sub-agent and patcher to the entities/edges implicated by flagged paths. |

The library's full sub-agent stack is `O2N`. `B0`, `B1`, and `O1`
are documented baselines, kept available so users can quantify the
contribution of each component.

## Phase A — pipeline validation (no LLM)

Before plugging in any LLM, we exercise the loop with the
deterministic oracle patcher to confirm `gather → plan → execute`
wires correctly across realistic defect counts.

| Defects per run | Fix% | Drift | Loop |
|-----------------|-----:|------:|------|
| 8 (×6 seeds)    | **100%** every run | 0 | converges |
| 4               | 100% | 0 | converges |
| 16              | 100% | 0 | converges |
| 24              | 90% | 0 | converges (over-saturated KG) |

Validates the loop machinery itself — convergence detection, manifest
attribution, hardcap behavior — independent of any LLM.

## Phase B — full-regen vs surgical (size=16)

Same fixture, six seeds × six defects each, gpt-4o-mini.

| Condition | Fix% | Drift | Tokens | Wall | Iters |
|-----------|-----:|------:|-------:|-----:|------:|
| oracle (UB) | 100 | 0.2 | 0 | 0s | 2.0 |
| B0 full-regen | 100 | 4.5 | 2,506 | 14.5s | 2.2 |
| B1 single-shot patch | 55 | 5.5 | 1,774 | 3.1s | 2.0 |
| O1 loop + patch | 64 | 4.8 | 6,218 | 9.6s | 4.7 |
| O2 (+ path_finder, full ctx) | 84 | 5.5 | 13,244 | 19.7s | 4.5 |
| **O2N (+ narrowing)** | **97** | **4.5** | **8,482** | 17.2s | 4.3 |

Key observations at this scale:

- `B0` and `O2N` reach the same fix rate.
- `B0` is *cheaper* in tokens (2.5K vs 8.5K). Surgical looks expensive.
- `path_finder` alone (`O1` → `O2`) closes a +20pp gap by redirecting
  symptom-targeted patches to root causes.
- Context narrowing (`O2` → `O2N`) **improves both cost (-36%) and
  fix rate (+13pp)** — it's a correctness component, not just an
  optimization.

## Phase B — size sweep

The story flips at scale. Same fixture, scaled up.

| Size (E/X) | Cond | Fix% | Tokens | Wall | Iters |
|------------|------|-----:|-------:|-----:|------:|
| 50 (50/90) | B0 | 100 | 10,095 | 60.7s | 2 |
| 50 (50/90) | O2N | 100 | 18,995 | 24.6s | 4 |
| **100 (100/183)** | **B0** | **0** | **73,740** | **513s** | **5 (hardcap)** |
| **100 (100/183)** | **O2N** | **100** | **17,114** | 26s | 4 |

At size=100 the perturbed KG is ~6KB JSON, well within
`gpt-4o-mini`'s context. **`B0` still fixed 0 of 8 defects.** The
proximate cause is completion-token saturation: regenerating 100
entities + 183 edges produces JSON that consistently hits the 8K
`max_tokens` ceiling, the model truncates mid-array, the structural
critic flags more defects (truncation introduces dangling refs),
the next iteration retries. Five times, until hardcap.

The crossover is sharp:

- At 50 entities, `O2N` costs 2× more tokens than `B0` for the same
  100% fix rate.
- At 100 entities, `O2N` costs ~5× **fewer** tokens than `B0` and is
  **the only one that works**.
- A real-world KG-editing application sits well beyond the
  crossover.

## Where the wins come from — ablation

Holding size=16 fixed (where every condition can complete), we can
attribute each fix-rate point to a component:

| Step | Δ Fix% | Why |
|------|-------:|-----|
| `B1` → `O1` | +9pp | Loop wrapping lets a second pass clean up cascade symptoms. |
| `O1` → `O2` | +20pp | `path_finder` redirects to root-cause pointers (e.g. flips an entity's type back instead of patching every edge that fails type-check). |
| `O2` → `O2N` | +13pp | Narrowed context removes distractor entities so the patcher's tool-calling stays focused on the implicated slice. |

Single-shot RFC 6902 patching (`B1`) saturates around 55%. Closing
the gap to 97% requires the loop **and** the sub-agent stack, not
either alone.

## Drift

Drift = number of (entity_id ∪ edge_id) keys whose post-state JSON
differs from the clean baseline, **excluding** keys whose change is
the legitimate fix.

At size=16, drift is roughly comparable across conditions (~4-5
keys/run). The drift mostly comes from label hallucination on
restored entities — the patcher invents a plausible-sounding label
when the ground-truth label was perturbed by `entity_paraphrase`.
A stronger semantic critic that flags label drift would reduce this.

At size=100 a fair drift comparison isn't meaningful — `B0` doesn't
finish, so its post-state is the broken truncation.

## Reproducing

The perturbation harness, raw run data, and analysis scripts live in
a separate repository alongside the experimental design notes. The
scripts wire through this library's public API (`run_correction_loop`,
`make_identity_planner`, `make_callback_executor`); the only
non-library code is the synthetic KG fixture, the perturbation
operators, the structural critic, and the patcher prompts.

The minimum reproducible smoke test ships with the library:

```bash
pip install -e ".[dev]"
python examples/kg_correction/run.py
```

This runs the deterministic oracle patcher (no LLM, ~1 second) and
prints the gather/plan/execute trace so you can see the loop
mechanics. To run an LLM condition (`B0` / `O2N` / etc.) point the
run script at the LLM-driven critic + patcher modules in your fork
of the harness repo.
