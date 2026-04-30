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

### Example configurations

Each entry exercises a different code path. Swap the exports below into the
launch block above.

#### Single-GPU, public model

Smallest "real" test. No HF token needed.

```bash
export TARGET_MODEL="Qwen/Qwen2.5-7B-Instruct"
export ACCELERATORS="A100:1"
```

Expected: `TP=1, DP=1, EP=1`. ~7 min cold start, ~$0.50/hr.

#### Multi-GPU tensor parallelism (gated model)

Forces `TP > 1`. Validates the parallelism math and HF auth.

```bash
# https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct — accept the license first
export HF_TOKEN="hf_..."
export TARGET_MODEL="meta-llama/Llama-3.1-70B-Instruct"
export ACCELERATORS="A100:4"
```

Expected: `TP=4, DP=1, EP=1`. ~12-15 min cold start, ~$5/hr.

#### Mixture-of-experts

Forces `EP > 1`. Validates the MoE branch.

```bash
export TARGET_MODEL="mistralai/Mixtral-8x7B-Instruct-v0.1"
export ACCELERATORS="A100-80GB:2"
```

Expected: `TP=2, DP=1, EP=2`. ~15 min cold start.

#### Speculative decoding (HF draft)

Validates the speculative path on a public draft.

```bash
export TARGET_MODEL="Qwen/Qwen3-14B-FP8"
export ACCELERATORS="H100:1"
export SPEC_ENABLED="true"
export SPEC_DRAFT_MODEL="RedHatAI/Qwen3-14B-speculator.eagle3"
export SPEC_METHOD="eagle3"   # or 'EAGLE3' for sglang
```

### Quota notes

A100/H100 spot quota is per-region in GCP and often zero by default. If
`sky launch` retries forever:

```bash
gcloud compute regions describe us-west1 --format="value(quotas)" \
  | tr ',' '\n' | grep -i nvidia
```

If the quota you want is zero, request it in the GCP console (Compute Engine
→ Quotas) or fall back to a smaller GPU spec.

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

## TODO

Things this v0.1 doesn't do yet. Listed roughly by priority — the security &
cost items are the riskiest gaps if anyone uses this beyond personal testing.

### Security & access control

- [ ] **API-key auth.** Currently anyone with the external IP can hit the
      endpoint. sglang accepts `--api-key`; vllm accepts `--api-key`. Add
      `API_KEY` schema input and thread it into both launchers.
- [ ] **TLS / HTTPS.** Endpoint is raw HTTP — credentials and prompts go
      cleartext. Either terminate TLS at a reverse proxy (caddy / nginx) on
      the same VM, or front with GCP HTTPS Load Balancer.
- [ ] **Firewall scoping.** SkyPilot's `ports` opens 8000 to `0.0.0.0/0`.
      Restrict to known CIDRs via `ports` + custom GCP firewall rules.
- [ ] **HF token leakage.** `HF_TOKEN` is currently passed via `[stage.env]`
      in plaintext into the SkyPilot YAML. Switch to a secret-manager pull
      (GCP Secret Manager / GitHub Actions secret) at run.sh time.

### Cost & lifecycle

- [ ] **Auto-shutdown on idle.** Server runs forever unless explicitly
      destroyed. Add idle-timeout: poll request rate, shut down after N
      minutes of zero traffic. Optional schema input `IDLE_TIMEOUT_MIN`.
- [ ] **Budget cap.** No hard ceiling. Add `MAX_RUN_HOURS` that schedules a
      `gcloud compute instances delete --max-run-duration=Nh` on launch
      (already used by ai-factory's find_gpu.sh).
- [ ] **Spot preemption recovery.** Today, preemption = endpoint dies. Move
      to SkyPilot managed jobs (`sky.jobs.launch()`) for auto-recovery.
- [ ] **Persistent HF cache.** Every fresh cluster re-downloads weights
      (~10 min for 70B). Mount a persistent disk at `~/.cache/huggingface`
      across launches in the same project.

### Token accounting & observability

- [ ] **Per-request token counting.** Both engines emit Prometheus metrics
      at `/metrics` (sglang: `--enable-metrics` already on; vllm: built-in).
      Scrape into a dashboard, attribute by API key once auth lands.
- [ ] **Request log to GCS.** No durable record of who asked what. Stream
      access logs from sglang/vllm to a GCS bucket via `--log-requests` +
      a fluentbit sidecar.
- [ ] **Cost-per-1k-tokens reporting.** Compute from GPU $/hr × utilization
      × throughput. Useful as an output artifact.
- [ ] **Latency percentiles in MLflow.** Mirror number-generator's MLflow
      pattern for benchmark runs.

### Untested code paths

- [ ] **vllm engine.** Argv builder exists but never run end-to-end. Smoke
      test on `Qwen/Qwen2.5-1.5B-Instruct` / `L4:1`.
- [ ] **Speculative decoding.** Both engines wired up; not validated.
      Smoke test: sglang EAGLE3 with `RedHatAI/Qwen3-14B-speculator.eagle3`.
- [ ] **MoE / EP path.** Math validated locally (Mixtral on H100:4 → EP=2);
      live launch never attempted.
- [ ] **Multi-cloud.** Only GCP tested. SkyPilot supports AWS, Azure,
      Lambda, RunPod — should work via just changing `cloud` in shark.toml,
      not verified.

### Fit-check & parallelism

- [ ] **Pipeline parallelism (PP).** We compute TP and DP, never PP.
      Models too large for TP-on-one-node need PP across nodes.
- [ ] **Auto-pick fp8 vs int8wo by architecture.** Today the user has to
      know fp8 is Hopper+. Could auto-substitute int8wo on Ampere when fp8
      is requested, with a warning.
- [ ] **Activation memory modeling at long context.** Flat 20% overhead is
      wrong for `MAX_MODEL_LEN=128k`. Account for sequence length × hidden
      dim × layers in the estimate.
- [ ] **Honor `num_key_value_heads` (GQA) in TP validation.** Currently we
      validate against `num_attention_heads`; for GQA models, `num_kv_heads`
      is the tighter constraint.

### Infrastructure & DX

- [ ] **Tests.** Zero tests in this repo. At minimum: unit tests for
      `parse_accelerators`, `_next_power_of_two`, `calculate_parallelism`
      (we have local smoke tests but not committed).
- [ ] **CI.** No GitHub Actions. Add lint + unit-test workflow.
- [ ] **De-vendor parallelism math.** `src/parallelism.py` is copied from
      ai-factory. Either depend on ai-factory as a package (after it adds
      `pyproject.toml`) or extract a shared `tails-mpt/ai-inference-utils`
      library. See discussion in commit history.
- [ ] **More GPUs in `gpu_specs.py`.** Add MI300X, RTX 6000 Ada, etc. as
      SkyPilot adds them.
- [ ] **GGUF / AWQ / GPTQ paths.** Currently only fp8 + torchao. Llama.cpp
      / AWQ could run much smaller models on cheaper GPUs.
- [ ] **Local-only mode.** No way to run sglang/vllm against a local model
      directory. Useful for testing draft-model pairs developed in-house.

### Known caveats already documented

- Baby-shark `${VAR}` interpolation: still on PR
  [tails-mpt/baby-shark#110]; without merge, this tool depends on the
  feature branch.
- fp8 quantization requires Hopper+ (guarded in `parallelism.py`).

[tails-mpt/baby-shark#110]: https://github.com/tails-mpt/baby-shark/pull/110
