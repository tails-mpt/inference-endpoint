# ADR 0005: Auto-compute TP/DP/EP from model + GPU

**Status:** Accepted

## Context

Both engines need TP, DP, and (for MoE) EP set explicitly at launch. Asking
the user to compute these by hand is a bad UX — it requires knowing the
model's parameter count, attention head count, expert count, and the
target GPU's VRAM. Easy to get wrong, and silent failures are common
(wrong TP → OOM at load; TP doesn't divide heads → assertion crash).

ADR 0004 fixed `ACCELERATORS` as user input. So we have:

- The model ID → introspect via HuggingFace metadata.
- The GPU spec → look up VRAM from the static table.

The math to derive parallelism from these two pieces is mechanical.

## Decision

Vendor ai-factory's parallelism algorithm (see ADR 0006) and run it before
the engine install in `run.sh`.

Strategy (from `src/utils/model_utils.py:calculate_parallelism`):
1. Estimate weight memory from HF safetensors metadata, scaled by
   quantization (fp8 → 0.5×, int4 → 0.25×). Add 20% overhead for CUDA
   context, activations, and KV-cache bootstrap.
2. **TP** = smallest power of 2 such that `weight_mem / TP ≤ vram_per_gpu`.
3. Validate `num_attention_heads % TP == 0`; if not, search for next valid
   divisor.
4. **DP** = `gpu_count // TP`.
5. **EP** = `TP` if model has `num_local_experts` (MoE), else 1.

Output goes to `/tmp/parallelism.json`, which `launch.py` reads to build
engine-specific argv. The user never specifies TP/DP/EP.

Strategy emphasizes **maximizing DP** (more replicas → better concurrency
under tight latency SLAs) and **using powers of 2** (always divides
real-world attention head counts: 32, 64, 128).

## Consequences

- **One concrete output, two consumers.** `parallelism.py` produces
  `/tmp/parallelism.json`; `launch.py` reads it. Easy to inspect and test.
- **Powers of 2 is a known limitation.** Models with min_tp=5 jump to
  TP=8 even if TP=5 would technically fit. In practice powers of 2 cover
  ~all real workloads.
- **PP not yet computed.** Beyond a single node, the math gets more
  involved. Pending work, see TODO.
- **Fail-fast over fail-late.** Bad fit detected before pip-install ($0
  cost), not after weight load (~$1 in wasted spot time).
