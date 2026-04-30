# ADR 0006: Vendor parallelism utilities from ai-factory

**Status:** Accepted (with explicit tech debt)

## Context

The parallelism math (memory estimation from HF safetensors, TP/DP/EP
derivation, quantization-aware sizing) already exists in `ai-factory` at
`src/utils/model_utils.py`. About 250 lines.

Four ways to reuse it:

| | Approach | Pros | Cons |
|---|---|---|---|
| **a** | Re-implement from scratch | Independent code | Wasted effort; will diverge from ai-factory |
| **b** | Vendor (copy) into this repo | Zero dependencies; ships as one repo | Duplicates code; bug fixes need to be made twice |
| **c** | `pip install` ai-factory from git | Single source of truth | ai-factory isn't installable today (no `pyproject.toml` package config); pulls heavy deps (subprocess to nvidia-smi, SLA types, deploy scripts) |
| **d** | Extract a shared library `tails-mpt/ai-inference-utils` | Cleanest long-term; both repos depend on it | New repo to maintain; one-shot refactor of ai-factory's imports |

Option (d) is the right answer. Option (c) is blocked on ai-factory not
shipping as a package. Option (a) is wasteful.

## Decision

For v0.1, vendor (option **b**). Ship the working tool now. Track the
extraction (option **d**) as tech debt in the README's TODO section and
revisit when the cost of duplication starts to bite.

The vendored module lives at `src/parallelism.py` with a header comment
identifying the source. Three changes from the original:

1. Drop SLA derivation (out of scope for serving).
2. Take `GPUInventory` from `src/gpu_specs.py` (env-driven from
   `ACCELERATORS`), not nvidia-smi — VM contract is set before the VM
   exists.
3. Add a `compute_or_die()` CLI entry point that writes
   `/tmp/parallelism.json` for `launch.py`.

## Consequences

- **Bug fixes in ai-factory don't auto-propagate here.** Either re-vendor
  manually when ai-factory's util changes, or accelerate the extraction
  in option (d).
- **Repo is self-contained.** No cross-repo install dance to test or run
  the tool — important for a v0.1 with one user.
- **Tech debt is visible.** TODO section in the README calls out
  de-vendoring as priority work, so this ADR isn't a "decided and
  forgotten" choice.
- **The right next move is auditable.** The proposed extracted
  library would contain: `ParallelismConfig`, `calculate_parallelism`,
  `estimate_model_memory_gb`, `validate_model_accessible`,
  `GPUInventory` constructor variants. ~300 LOC, ~3 deps.
