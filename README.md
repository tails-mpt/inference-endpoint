# inference-endpoint

Slipstream tool that stands up an OpenAI-compatible inference endpoint for any
HuggingFace model on **sglang** or **vllm**, on a user-specified GPU shape.
TP/DP/EP are computed automatically from model size + GPU spec.

## Inputs

### Required

| Name | Description |
|------|-------------|
| `TARGET_MODEL` | HuggingFace model ID, e.g. `Qwen/Qwen2.5-1.5B-Instruct`, `meta-llama/Llama-3.1-70B-Instruct`. |
| `ACCELERATORS` | SkyPilot GPU spec, e.g. `L4:1`, `A100:8`, `H100:4`. Cost and availability are user-controlled — we don't auto-pick. |

### Optional

| Name | Default | Description |
|------|---------|-------------|
| `ENGINE` | `sglang` | `sglang` or `vllm`. |
| `SERVED_MODEL_NAME` | basename of `TARGET_MODEL` | Alias clients use. |
| `PORT` | `8000` | Same default for both engines. |
| `MAX_MODEL_LEN` | (engine default) | sglang `--context-length`; vllm `--max-model-len` (`auto` to fit KV cache). |
| `DTYPE` | (engine default) | `auto`, `bfloat16`, `float16`, `float32`. |
| `QUANTIZATION` | none | sglang: `fp8` or torchao methods (`int4wo-128`, `int8wo`, `int8dq`). vllm: `fp8`, `bitsandbytes`, `torchao`. |
| `MEM_FRACTION` | `0.85` | Fraction of GPU memory the server may use. |
| `HF_TOKEN` | empty | Required for gated models. |
| `SPEC_ENABLED` | `false` | Speculative decoding on/off. |
| `SPEC_DRAFT_MODEL` | empty | HuggingFace ID of the speculator, e.g. `RedHatAI/Qwen3-14B-speculator.eagle3`. |
| `SPEC_METHOD` | `eagle3` | sglang: `EAGLE`/`EAGLE3`/`STANDALONE`. vllm: `eagle`/`eagle3`. |
| `SPEC_NUM_TOKENS` | `3` | Tokens drafted per step. |

## How parallelism is chosen

`src/parallelism.py` follows ai-factory's heuristic:

1. Pull weight count from HF (safetensors metadata → file sizes → param count).
2. Add 20% overhead for CUDA context, activations, KV cache bootstrap.
3. Apply quantization reduction if `QUANTIZATION` is set (fp8 → 0.5×, int4wo → 0.25×).
4. **TP** = smallest power of 2 such that `weight_mem / TP ≤ vram_per_gpu`, then validated to divide `num_attention_heads`.
5. **DP** = `gpu_count // TP`.
6. **EP** = `TP` if the model is MoE (config has `num_local_experts`), else 1.

If TP exceeds the requested GPU count, the job fails fast with an actionable message.

## GPU spec table

`src/gpu_specs.py` maps SkyPilot accelerator names to per-GPU VRAM:

| GPU | VRAM | | GPU | VRAM |
|---|---|---|---|---|
| `T4` | 16 GB | | `A100` | 40 GB |
| `L4` | 24 GB | | `A100-80GB` | 80 GB |
| `A10` / `A10G` | 24 GB | | `H100` | 80 GB |
| `V100` | 16 GB | | `H200` | 141 GB |
| `V100-32GB` | 32 GB | | `B200` | 192 GB |

Add to `VRAM_GB` in `gpu_specs.py` if SkyPilot supports a GPU not listed here.

## Layout

```
inference-endpoint/
├── README.md
├── pyproject.toml
├── src/
│   ├── gpu_specs.py        # ACCELERATORS string → GPUInventory
│   ├── parallelism.py      # vendored from ai-factory; HF intro + TP/DP/EP math
│   └── launch.py           # engine-aware argv builder, execvp's the server
└── .slipstream/
    ├── schema.toml         # user-facing inputs
    └── pipeline/
        ├── shark_pipeline.toml
        └── jobs/inference-endpoint/
            ├── shark.toml         # accelerators = "${ACCELERATORS}"
            ├── environment.yml    # base env (engine installed at runtime)
            └── run.sh             # parallelism check → engine install → launch + health
```

## Dependency on baby-shark

`shark.toml` uses `accelerators = "${ACCELERATORS}"` — driven by the slipstream
`[[input]]`. This requires baby-shark with `${VAR}` interpolation in TOML
loading (PR [tails-mpt/baby-shark#110]). Without it, the launch fails with a
literal-string accelerator value.

## Limitations of the fit check

- TP/PP weight sharding is approximated — actual layouts vary between engines.
- Activation memory is folded into a flat 20% overhead.
- Quantized checkpoints (`gguf`, `awq`) are detected by file naming only; explicit `QUANTIZATION` is more reliable.
- KV cache pre-allocation differs between sglang (`mem-fraction-static`) and vllm (`gpu-memory-utilization`); we pass `MEM_FRACTION` to both.

[tails-mpt/baby-shark#110]: https://github.com/tails-mpt/baby-shark/pull/110
