# ADR 0002: Single job, runtime engine dispatch (sglang vs vllm)

**Status:** Accepted

## Context

The tool must serve via either **sglang** or **vllm**, picked per launch.
Three architectures considered:

| | Approach | Tradeoff |
|---|---|---|
| **a** | Two parallel stage dirs (`serve_sglang/`, `serve_vllm/`); slipstream picks one | Cleanest envs, but slipstream has no pipeline-branching primitive — `[[input]]` only sets env vars |
| **b** | Two job dirs in two separate slipstream tools | Duplicates almost everything; users must remember which tool to launch |
| **c** | One job dir, branch on `$ENGINE` inside `run.sh` | Single repo, single schema; engine-divergence localized to ~20 lines |

Slipstream's input mechanism doesn't support stage selection (verified by
inspecting `tails-mpt/number-generator`'s schema.toml — `[[input]]` blocks
just become env vars). So (a) is structurally not available.

## Decision

Single job dir at `.slipstream/pipeline/jobs/inference-endpoint/`. `run.sh`
branches on the `ENGINE` env var to install the correct engine and call
`launch.py`, which builds engine-specific argv.

```bash
case "$ENGINE" in
  sglang) pip install --quiet "sglang==0.5.6" ;;
  vllm)   pip install --quiet vllm ;;
esac
exec python /repo/src/launch.py
```

## Consequences

- **All engine-divergence is in one Python file.** `launch.py` has two
  `_build_*_argv` functions; everything else (parallelism, health-poll,
  HF auth) is shared.
- **Adding a third engine is a case-arm + a builder function.** No
  schema changes, no pipeline changes.
- **Dependency conflicts dodged.** sglang and vllm pin overlapping
  torch/transformers/flashinfer; one shared env wouldn't resolve.
- **Engine choice is a runtime decision, not a deploy-time one.** The
  same launch can switch engines by changing one input.
