# Architecture overview

High-level view of how a request to spin up an inference endpoint flows
from user input to a live OpenAI-compatible server.

For why each piece is the way it is, see [docs/adr/](adr/README.md).

## Layering

```
┌────────────────────────────────────────────────────────────────────┐
│  user                                                              │
│    └─ provides TARGET_MODEL, ACCELERATORS, ENGINE, ...             │
└─────────────────────────────────┬──────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────┐
│  slipstream  (production path)                                     │
│    reads .slipstream/schema.toml                                   │
│    sets schema vars in os.environ + global_env                     │
│    invokes baby-shark pipeline                                     │
└─────────────────────────────────┬──────────────────────────────────┘
                                  │  ── OR ──
┌─────────────────────────────────▼──────────────────────────────────┐
│  shark pipeline run  (dev path, ADR 0010)                          │
│    user exports schema vars; [stage.env] interpolates them         │
└─────────────────────────────────┬──────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────┐
│  baby-shark                                                        │
│    reads shark_pipeline.toml + shark.toml with ${VAR} interp.      │
│      (ADR 0007 — PR tails-mpt/baby-shark#110)                      │
│    builds SkyPilot YAML; provisions VM via SkyPilot                │
│    streams stage logs back to user                                 │
└─────────────────────────────────┬──────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────┐
│  GCP VM (run.sh runs here)                                         │
│                                                                    │
│   1.  parallelism.py                                               │
│         parse ACCELERATORS → GPUInventory (gpu_specs.py)           │
│         introspect HF model (config.json + safetensors metadata)   │
│         estimate weight memory + 20% overhead                      │
│         compute TP/DP/EP (ADR 0005)                                │
│         validate fp8/Hopper compatibility (ADR 0009)               │
│         write /tmp/parallelism.json                                │
│         (fail fast on misfit before any download)                  │
│                                                                    │
│   2.  pip install sglang OR vllm  (ADR 0003)                       │
│         engine selected by $ENGINE (ADR 0002)                      │
│                                                                    │
│   3.  launch.py                                                    │
│         read /tmp/parallelism.json                                 │
│         build engine-specific argv                                 │
│           sglang → --tp-size, --dp-size, --ep-size, etc.           │
│           vllm   → --tensor-parallel-size, --enable-expert-...     │
│         os.execvp the server                                       │
│                                                                    │
│   4.  health-poll loop                                             │
│         curl /health every 5s for up to HEALTH_TIMEOUT (2400s)     │
│         print public IP + curl examples on success                 │
│         tail server.log + exit 1 on timeout/crash                  │
│                                                                    │
│   5.  wait $SERVER_PID  (job stays alive while server runs)        │
└────────────────────────────────────────────────────────────────────┘
```

## File map

| File | Role | Notes |
|---|---|---|
| `.slipstream/schema.toml` | User-facing input contract | Slipstream UI reads this |
| `.slipstream/pipeline/shark_pipeline.toml` | baby-shark pipeline declaration | One n=1 stage; `[stage.env]` mirrors schema for direct testing (ADR 0010) |
| `.slipstream/pipeline/jobs/inference-endpoint/shark.toml` | Stage manifest | `accelerators = "${ACCELERATORS}"` (ADR 0007); `repo_dir` mounts `src/` at `/repo` |
| `.slipstream/pipeline/jobs/inference-endpoint/environment.yml` | Base conda env | Minimal: python + huggingface_hub. Engine pip-installed at runtime (ADR 0003) |
| `.slipstream/pipeline/jobs/inference-endpoint/run.sh` | Stage entrypoint | Orchestrates the 5 steps in the diagram above |
| `src/gpu_specs.py` | Static GPU → VRAM table; FP8 capability set | Adding a GPU = one-line edit |
| `src/parallelism.py` | HF introspection + TP/DP/EP math | Vendored from ai-factory (ADR 0006) |
| `src/launch.py` | Engine-aware argv builder + `execvp` | All engine-divergence localized here |

## Cross-cutting properties

- **Fail-fast cost ordering.** Validation (cheap) runs before pip install
  (medium), which runs before model download (expensive). A misconfig is
  caught at the cheapest step possible.
- **One log path.** All stage output goes through SkyPilot's job log;
  the engine itself logs to `~/server.log` on the VM, accessible via
  `shark sky ssh` (ADR 0002 keeps both engines using the same path).
- **No Python on the user's machine for production launches.** The
  whole tool runs on the VM; the user just provides inputs through
  slipstream's UI.
- **Server runs forever by default.** `teardown_cluster = false`. Users
  destroy clusters explicitly. See README TODO for auto-shutdown work.
