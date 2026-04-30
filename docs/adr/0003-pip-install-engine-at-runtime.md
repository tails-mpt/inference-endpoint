# ADR 0003: Pip-install the engine at run.sh time

**Status:** Accepted

## Context

Following ADR 0002, the job has one conda env but needs to support two
mutually-incompatible engines (sglang, vllm). Three ways to handle the
env split:

| | Approach | Cost |
|---|---|---|
| **a** | One `environment.yml` with both engines | Won't resolve — torch/transformers/flashinfer pin clashes |
| **b** | `setup_commands` in `shark.toml` runs `micromamba env create` per engine before the entrypoint | Adds env-create overhead to every launch and depends on `$ENGINE` being available in `setup_commands` scope |
| **c** | Minimal base env in `environment.yml`; `run.sh` pip-installs the engine on top | Simple, deterministic, no shark plumbing changes |

## Decision

Option (c). `environment.yml` declares only python + huggingface_hub.
`run.sh` reads `$ENGINE` and runs `pip install --quiet "sglang==0.5.6"` or
`pip install --quiet vllm` into the same env before launching.

## Consequences

- **~2-3 min cold-start cost per launch.** Negligible vs. the model
  download (~10 min for 70B) and shard load (~5 min). Not on the hot path
  anyone cares about.
- **No conda/mamba surgery required.** baby-shark sees one env, one entry
  point. `setup_commands` stays free for things like docker setup.
- **Engine version pinning lives in `run.sh`.** Bumping sglang from 0.5.6
  to 0.6 is a one-line change. No schema impact.
- **First-class debugging.** If a launch fails inside engine install, the
  error lands in the same job log as everything else; no two-tier
  conda-env-vs-pip-failure to disentangle.
- **No shared engine cache between launches.** Each fresh VM re-pip-installs.
  Acceptable for now; persistent disk could fix this (see TODO).
