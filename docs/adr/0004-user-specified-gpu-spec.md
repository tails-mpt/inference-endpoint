# ADR 0004: User specifies GPU spec; tool validates fit, doesn't auto-pick

**Status:** Accepted

## Context

Given a model, the tool could either:

- **Auto-pick the GPU shape** from model size (e.g. 70B → "you need H100:2").
- **Require the user** to specify `ACCELERATORS` and just validate the fit.

Auto-pick is friendlier on paper, but in practice the choice is dominated
by two factors that have nothing to do with the model:

1. **Cost.** L4 spot is $0.20/hr; H100:8 spot is $20/hr. The user knows
   their budget; we don't.
2. **Quota and capacity.** GCP A100/H100 spot routinely stocks out per
   region. The user knows what their account has access to; we don't.

Auto-picking would mean either burning capacity to find out or maintaining
a real-time view of cloud availability, which is not this tool's job.

## Decision

`ACCELERATORS` is a required `[[input]]` (e.g. `A100:8`, `H100:4`, `L4:1`).
The tool's responsibility is to **fail fast if the model can't fit** on the
spec the user provided, with a clear message including the suggested next
step (bigger GPU, fewer GPUs, add quantization).

Implementation: `parallelism.py` parses the spec via the static VRAM table
in `gpu_specs.py`, runs the fit math (see ADR 0005), and `sys.exit` with
an actionable error if TP > GPU count.

## Consequences

- **Maintain a static GPU → VRAM table.** Adding a new GPU model in the
  SkyPilot catalog is a one-line edit. Not magic, but fine.
- **Errors are actionable, not abstract.** "needs TP=8 but only 4 GPUs
  available" beats "out of memory" mid-launch.
- **The README must teach a sizing intuition.** We added a GPU spec
  table and example configurations to compensate for not auto-picking.
- **A future "auto" mode is additive, not breaking.** If we ever want
  smart defaults, `ACCELERATORS=auto` could pick from a budget input —
  doesn't change the existing contract.
