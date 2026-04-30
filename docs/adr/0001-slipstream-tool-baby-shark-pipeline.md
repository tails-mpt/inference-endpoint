# ADR 0001: Build as a slipstream tool wrapping a baby-shark pipeline

**Status:** Accepted

## Context

Goal: stand up an OpenAI-compatible inference endpoint for any HuggingFace
model on a user-chosen GPU shape, on demand, in a way teammates can invoke
without local SkyPilot setup.

The team already has two relevant pieces of infrastructure:

- **baby-shark** — a SkyPilot wrapper that runs jobs and pipelines from a
  `shark.toml` / `shark_pipeline.toml` declaration.
- **slipstream** — pulls in repos with a `.slipstream/` dir and runs them as
  user-facing "tools," surfacing the underlying baby-shark pipeline behind a
  simple `[[input]]` schema. Reference example: `tails-mpt/number-generator`.

We could also have written a standalone Python CLI, a Cloud Run service, or
a custom Terraform module.

## Decision

Build as a slipstream tool whose `.slipstream/pipeline/` is a baby-shark
pipeline with a single (n=1) stage that launches the inference server.

Layout mirrors `tails-mpt/number-generator`:
```
inference-endpoint/
├── src/                            <- mounted at /repo on the VM
└── .slipstream/
    ├── schema.toml                 <- user-facing inputs
    └── pipeline/
        ├── shark_pipeline.toml
        └── jobs/inference-endpoint/
            ├── shark.toml
            ├── environment.yml
            └── run.sh
```

## Consequences

- **Inherits a familiar input contract.** Schema fields become env vars on
  the VM with no extra plumbing.
- **No platform invention.** SkyPilot does provisioning; baby-shark does
  pipeline state; slipstream does UX. We just write the job.
- **Testable two ways.** Through slipstream (production path) or directly
  via `shark pipeline run` (dev path — see ADR 0010).
- **Locked in to baby-shark + slipstream.** Migrating off this stack means
  rewriting the orchestration layer. Acceptable: both are owned by the
  team and nothing else needs the same primitives.
