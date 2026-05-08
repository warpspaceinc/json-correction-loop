# Changelog

All notable changes to `json-correction-loop` will be documented in
this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-05-09

Initial public release. Domain-neutral `gather → plan → execute`
correction loop with surgical RFC 6902 patcher and four sub-agent
slots:

- `path_finder` — symptom → root-cause pointer redirect
- `template_filler` — empty-container enumeration
- `request_validator` — upstream rejection of malformed requests
- `patch_evaluator` — score patches against intent before commit

Composable convergence policies (`QualityStablePolicy`,
`HardcapPolicy`) and `StorageBackend` / `EventSink` Protocols. The
library imports no specific LLM client, persistence layer, or event
sink — adapters are Protocols you plug in.

Bundled examples:

- `examples/01_quickstart.py` — fakes-only end-to-end loop, no LLM.
- `examples/kg_correction/` — synthetic knowledge-graph fixture, six
  perturbation operators with ground-truthed defect log, structural
  critic, deterministic oracle patcher, end-to-end driver.

Empirical evaluation in [EXPERIMENTS.md](EXPERIMENTS.md).
