# Config-tree refactor — JSON Schema as the critic

A non-KG domain showing the same library applied to **configuration
file refactoring**. The state is a config tree; the critic is a JSON
Schema validator (no LLM); the patcher is the deterministic example
from `01_quickstart` adapted to the schema. Useful as a starting
template if your domain has a schema you can validate against.

## Files

- `config.py`     — clean config + a deliberately broken version
- `schema.py`     — JSON Schema definition + critic that runs it
- `patcher.py`    — minimal deterministic fixes (typed defaults)
- `run.py`        — wires it all through `run_correction_loop`

## Run

```bash
cd examples/config_refactor
python run.py
```

You'll see the critic flag schema-violation defects (missing required
fields, type mismatches, unknown enum values) and the patcher restore
the config to schema validity in one or two iterations.

## What this shows

1. The library doesn't care that the domain isn't a KG — same `gather
   → plan → execute` shape works.
2. **Schema validators are great critics.** They report defects with
   stable JSON pointers (`/services/web/port`) — exactly the contract
   the planner / executor want.
3. The patcher in this example is deterministic to keep the demo
   reproducible. In production, swap it for an LLM patcher (see
   `examples/04_with_llm_patcher.py`) and the loop converges the same
   way — fewer iterations because LLM patches are smarter, but the
   contract is unchanged.

## Why this is interesting

The headline result on the KG benchmark — **full-regen breaks at
modest scale, surgical patching stays bounded** — applies to any
domain where:

- the document is large enough that re-emitting it stresses the
  model's `max_tokens` budget, and
- only a small fraction of fields actually need to change per fix.

Config files in production are routinely both. A 50KB Helm values
file with a single typo is the canonical case.
