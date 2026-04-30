# Architecture Decision Records

Each file captures a single decision made during this tool's design. The
intent is that someone joining the project six months from now can re-litigate
any one of them without having to rebuild the full context.

Format: lightweight [MADR](https://adr.github.io/madr/) — Context, Decision,
Consequences. Keep them short. Supersede an old one with a new ADR rather
than editing the original.

## Index

| # | Title | Status |
|---|---|---|
| [0001](0001-slipstream-tool-baby-shark-pipeline.md) | Build as a slipstream tool wrapping a baby-shark pipeline | Accepted |
| [0002](0002-single-job-runtime-engine-dispatch.md) | Single job, runtime engine dispatch (sglang vs vllm) | Accepted |
| [0003](0003-pip-install-engine-at-runtime.md) | Pip-install the engine at run.sh time | Accepted |
| [0004](0004-user-specified-gpu-spec.md) | User specifies GPU spec; tool validates fit, doesn't auto-pick | Accepted |
| [0005](0005-auto-compute-tp-dp-ep.md) | Auto-compute TP/DP/EP from model + GPU | Accepted |
| [0006](0006-vendor-parallelism-utils-from-ai-factory.md) | Vendor parallelism utilities from ai-factory | Accepted (with tech debt) |
| [0007](0007-baby-shark-var-interpolation.md) | Add `${VAR}` interpolation to baby-shark TOML | Accepted |
| [0008](0008-hf-only-speculative-drafts.md) | Speculative decoding from HF only — no GCS drafts | Accepted |
| [0009](0009-fp8-gated-to-hopper.md) | Reject fp8 quantization on pre-Hopper GPUs | Accepted |
| [0010](0010-stage-env-for-direct-testability.md) | `[stage.env]` in shark_pipeline.toml for direct testability | Accepted |
