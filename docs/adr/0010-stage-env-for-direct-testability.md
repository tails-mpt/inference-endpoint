# ADR 0010: `[stage.env]` in shark_pipeline.toml for direct testability

**Status:** Accepted

## Context

Slipstream sets schema `[[input]]` values via baby-shark's `global_env`
parameter when it invokes `pipeline.run_pipeline()`. That mechanism is
internal to slipstream — the standalone `shark pipeline run` CLI doesn't
expose it.

Without a fix, the only way to test this tool would be to deploy through
slipstream — which has its own infra requirements. That makes iteration
on `run.sh` and `parallelism.py` painful.

Approaches considered:

| | Approach | Why rejected |
|---|---|---|
| Add a CLI flag to baby-shark to forward `os.environ` to stages | Cross-cutting change in baby-shark; broader than this tool's needs |
| Document that direct testing isn't supported | Punishes contributors who don't have slipstream infra |
| Use `[stage.env]` block with `${VAR}` interpolation | Already supported by baby-shark; minimal change ↓ |

## Decision

Add a `[stage.env]` block to `shark_pipeline.toml` that uses `${VAR}`
interpolation (depends on ADR 0007) to pull every schema input from
the developer's shell:

```toml
[stage.env]
TARGET_MODEL      = "${TARGET_MODEL}"
ACCELERATORS      = "${ACCELERATORS}"
ENGINE            = "${ENGINE:-sglang}"
PORT              = "${PORT:-8000}"
HF_TOKEN          = "${HF_TOKEN:-}"
# ... and the rest of schema.toml
```

Required inputs (`TARGET_MODEL`, `ACCELERATORS`) intentionally have no
default — a missing export fails fast at TOML load. Optional inputs use
`${VAR:-default}` matching the schema's defaults.

## Consequences

- **Tool runs via `shark pipeline run` directly.** `export VAR=...` then
  invoke; same VM ends up running the same `run.sh`.
- **Slipstream still works unchanged.** Per the priority chain in
  baby-shark's `pipeline.py:574`,
  `ARTIFACT_* > global_env > [stage.env]`. So slipstream's `global_env`
  wins over our defaults; this block is purely additive.
- **Schema duplication.** `schema.toml` lists each input once; this
  block lists them again. They must stay in sync. Documented in the
  comment block; mitigated by the small surface area.
- **Hard dependency on ADR 0007.** Without `${VAR}` interpolation in
  `shark_pipeline.toml`, this whole block is just literal strings. The
  two ADRs ship together.
