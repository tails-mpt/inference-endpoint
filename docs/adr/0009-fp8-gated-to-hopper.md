# ADR 0009: Reject fp8 quantization on pre-Hopper GPUs

**Status:** Accepted

## Context

Discovered live during testing. Launching Llama-3.1-70B with
`QUANTIZATION=fp8` on `A100:4` passed our fit check, ran for ~5 minutes
through engine install and weight download, then crashed inside sglang's
weight-load step:

```
ValueError("type fp8e4nv not supported in this architecture.
           The supported fp8 dtypes are ('fp8e4b15', 'fp8e5')")
```

`fp8e4nv` is the NVIDIA-native E4M3 FP8 format. Hardware tensor-core
support for it was introduced in **Hopper** (H100, H200) and continues
through Blackwell (B200, B300). Pre-Hopper architectures — Ampere
(A100, A10), Ada (L4), Turing (T4) — lack the tensor-core path that
sglang's and vllm's Triton FP8 kernels compile to.

So `--quantization fp8` is silently a Hopper+ feature. The fit-math
correctly halved the weight memory (141 → 70 GB), but the math was
moot: the kernel won't compile on Ampere regardless of whether weights
fit.

## Decision

Maintain a `FP8_SUPPORTED` set in `gpu_specs.py` listing the
architectures that have hardware fp8e4nv:

```python
FP8_SUPPORTED = {"H100", "H100-MEGA-80GB", "H200", "B200", "B300"}
```

Validate in `compute_or_die()` immediately after parsing accelerators,
before any expensive work. Fail with an actionable suggestion (try
`int8wo`, or use a Hopper+ GPU). Do **not** auto-substitute `int8wo`
for `fp8` — they have different quality/throughput tradeoffs and the
choice should be deliberate.

## Consequences

- **Saves ~5 minutes and ~$0.50 per misconfigured launch.** Validation
  fires before the engine pip-install, not after a 30 GB model download.
- **Error message guides the user.** "fp8 requires Hopper+; got A100;
  try int8wo or use H100" is actionable; "Triton kernel compile failed"
  is not.
- **`FP8_SUPPORTED` is a maintenance burden.** New GPU architectures
  need to be added explicitly. Cost: one line per new GPU. Cheap.
- **No similar guard yet for int4 / int8 paths.** Those route through
  torchao Triton kernels which generally work on Ampere; if we
  discover a similar architecture-specific failure, add a guard with
  the same shape.
