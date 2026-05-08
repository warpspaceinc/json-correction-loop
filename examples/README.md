# Examples

Runnable examples in increasing order of complexity. Each is
self-contained — copy any file out and adapt it to your domain.

| File / dir | LLM? | What it shows |
|---|:---:|---|
| [`01_quickstart.py`](01_quickstart.py)             | no  | The smallest end-to-end loop. Faked critic + patcher. |
| [`02_observability.py`](02_observability.py)       | no  | Custom `StorageBackend` + `EventSink` Protocols — JSONL + ANSI console. |
| [`03_oscillation_aware.py`](03_oscillation_aware.py) | no | Identity vs. oscillation-aware planner on a deliberately stuck loop. |
| [`04_with_llm_patcher.py`](04_with_llm_patcher.py) | **yes** | Real OpenAI / OpenRouter LLM emitting RFC 6902 ops on a tiny domain. |
| [`kg_correction/`](kg_correction/)                 | no  | Knowledge-graph correction with a perturbation harness + structural critic + oracle patcher. The setup behind the paper's main benchmark. |
| [`config_refactor/`](config_refactor/)             | no  | Non-KG domain: JSON Schema validator as the critic, deterministic patcher. Drop-in template for config-file refactoring. |

## Running

The no-LLM examples need only the library installed:

```bash
pip install json-correction-loop
python examples/01_quickstart.py
```

The `kg_correction/` and `config_refactor/` examples have local
imports — run from inside their directory:

```bash
cd examples/config_refactor
pip install jsonschema  # critic dep
python run.py
```

`04_with_llm_patcher.py` requires:

```bash
pip install openai jsonpatch
export OPENROUTER_API_KEY=sk-or-...   # or OPENAI_API_KEY
python examples/04_with_llm_patcher.py
```

## What each example demonstrates

### 01 — quickstart

**The shape of the loop.** `gather_fn → plan_fn → execute_fn`, plus
the `kg_target_parser`-style hook that converts critic issues into
addressable target paths. No persistence, no events, no LLM. About
40 LOC of meaningful code.

### 02 — observability

**The Protocol pattern in action.** Shows that `StorageBackend` and
`EventSink` are duck-typed Protocols — anything with the right method
signatures plugs in. The example writes one JSONL line per iteration
and prints colored events to stderr. Useful as a starting template for
production observability stacks (Rich, structlog, OTEL, MongoDB,
file-tree commits, etc.).

### 03 — oscillation-aware planner

**When critics and patchers fundamentally disagree.** Demonstrates
the `make_oscillation_aware_planner` policy: any target re-flagged
for `threshold` consecutive iterations is dropped from subsequent
plans. Without this, philosophically-stuck loops burn tokens until
hardcap.

### 04 — with LLM patcher

**The smallest example that calls a real model.** A tiny "team"
config with three members; two have schema violations. The LLM
proposes RFC 6902 patch ops; we apply them with `jsonpatch`. No
sub-agents, no narrowing — just the bare loop with a real backend so
you can verify your API setup before trying the full stack.

### kg_correction/

**The benchmark.** Synthetic 16-entity / 14-edge knowledge graph
fixture, six perturbation operators with ground-truthed defect log,
structural critic (no LLM), deterministic oracle patcher. The
upper-bound condition behind the numbers in
[`EXPERIMENTS.md`](../EXPERIMENTS.md).

### config_refactor/

**A non-KG domain.** The same library applied to JSON Schema-valid
config trees. The critic is a `jsonschema` validator; the patcher
applies typed coercions for the violation classes the schema
generates. A clean drop-in starting point if your domain has a
schema you can validate against.

## Suggesting an example

If you've adapted the library to a domain we don't cover here and the
adapter is small, please open a PR with a new example directory.
Useful examples we'd particularly like to receive:

- LLM agent memory / scratchpad scrubbing
- Generated structured documents (resumes, multi-section reports)
- Multi-critic composition (style + structural + semantic)
- A `patch_evaluator` sub-agent ablation against the bare patcher
