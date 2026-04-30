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

## Running

This tool runs through slipstream in production. To test it directly via
baby-shark on GCP, follow these steps.

### Prerequisites

```bash
# GCS bucket (existing or new) — used by baby-shark for pipeline state
gcloud storage buckets create gs://YOUR-BUCKET --location=us-west1

# ADC auth for SkyPilot
gcloud auth application-default login

# baby-shark with ${VAR} interpolation (PR tails-mpt/baby-shark#110)
git clone https://github.com/tails-mpt/baby-shark.git
cd baby-shark
git checkout feat/shark-toml-env-interpolation   # remove once PR lands
uv sync

# Sanity check
uv run shark sky check
```

### Launch

```bash
cd inference-endpoint

# Required
export BUCKET="gs://YOUR-BUCKET"
export TARGET_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
export ACCELERATORS="L4:1"

# Optional — defaults match schema.toml
export ENGINE="sglang"
export HF_TOKEN="hf_..."   # only for gated models

uv run --project /path/to/baby-shark \
  shark pipeline run --pipeline-file .slipstream/pipeline/shark_pipeline.toml
```

The pipeline's `[stage.env]` block uses `${VAR}` interpolation to pull these
exports into the VM's environment, so the same shark.toml works under
slipstream (which uses `global_env`) and via direct `shark pipeline run`.

### What success looks like

1. SkyPilot provisions an L4 spot VM in GCP (~3 min).
2. `run.sh` on the VM logs:
   - `=== Computing parallelism ===` → `TP=1, DP=1, EP=1`
   - `=== Installing engine: sglang ===` (~2 min)
   - `=== Launching server ===`
   - `Endpoint live at http://<external-ip>:8000`
3. From your laptop:
   ```bash
   curl http://<external-ip>:8000/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{"model":"Qwen2.5-1.5B-Instruct",
          "messages":[{"role":"user","content":"Hi"}],
          "max_tokens":20}'
   ```

### Stopping

The server runs forever by design (`teardown_cluster = false`). Tear it down
explicitly when you're done:

```bash
uv run --project /path/to/baby-shark shark instance list
uv run --project /path/to/baby-shark shark instance destroy --name <cluster-name>
```

### Cost notes

| GPU spec | Approx. spot $/hr | Cold start to live endpoint |
|---|---|---|
| `L4:1` | $0.20 | ~6 min |
| `A100:1` | $0.50 | ~7 min |
| `H100:8` | $20 | ~10 min |

Spot can be preempted. For demos, set `use_spot = false` in `shark.toml`.

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
