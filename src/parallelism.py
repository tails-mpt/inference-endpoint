"""Compute optimal TP/DP/EP for a target HF model on a given GPU spec.

Vendored from ai-factory/src/utils/model_utils.py with three changes:
  1. Drop the SLA derivation — out of scope for serving.
  2. GPUInventory comes from gpu_specs.parse_accelerators (env-driven), not
     nvidia-smi — the slipstream contract is that ACCELERATORS is the source
     of truth, decided before the VM exists.
  3. CLI entry point writes /tmp/parallelism.json for launch.py to consume.

Usage as a module:
    from parallelism import compute_or_die
    config = compute_or_die()  # reads env vars, writes /tmp/parallelism.json

Usage as a script (called from run.sh):
    python -m parallelism
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from gpu_specs import GPUInventory, parse_accelerators


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass
class ParallelismConfig:
    tp_size: int
    dp_size: int
    ep_size: int = 1


# ---------------------------------------------------------------------------
# HF introspection
# ---------------------------------------------------------------------------


def validate_hf_credentials() -> str | None:
    """Return HF_TOKEN if set. Warn (don't fail) if missing — public models work."""
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or ""
    )
    if not token:
        print("Warning: HF_TOKEN not set. Public models will work; gated models will fail.")
        return None
    return token


def validate_model_accessible(model: str) -> dict[str, Any]:
    """Confirm the model exists on HF and pull metadata used for memory + TP math."""
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import (
        GatedRepoError,
        HfHubHTTPError,
        RepositoryNotFoundError,
    )

    api = HfApi()
    try:
        info = api.model_info(model, files_metadata=True)
    except RepositoryNotFoundError:
        sys.exit(f"Error: Model {model!r} not found on HuggingFace.")
    except GatedRepoError:
        sys.exit(
            f"Error: Model {model!r} is gated. "
            f"Accept the license at https://huggingface.co/{model} and set HF_TOKEN."
        )
    except HfHubHTTPError as e:
        sys.exit(f"Error: Cannot access model {model!r}: {e}")

    result: dict[str, Any] = {"model_id": info.id}

    try:
        config_path = Path(hf_hub_download(model, "config.json"))
        config = json.loads(config_path.read_text())
        if (heads := config.get("num_attention_heads")):
            result["num_attention_heads"] = heads
        if (experts := config.get("num_local_experts") or config.get("num_experts")):
            result["num_local_experts"] = experts
    except Exception:
        pass  # Non-critical — TP math will fall back to power-of-2 only.

    if info.safetensors:
        result["safetensors"] = {
            "parameters": info.safetensors.parameters,
            "total": info.safetensors.total,
        }

    weight_bytes = sum(
        (sib.size or 0)
        for sib in (info.siblings or [])
        if sib.rfilename and sib.rfilename.endswith((".safetensors", ".bin"))
    )
    if weight_bytes > 0:
        result["weight_files_gb"] = weight_bytes / 1e9

    return result


# ---------------------------------------------------------------------------
# Memory estimation
# ---------------------------------------------------------------------------


BYTES_PER_PARAM: dict[str, float] = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "float16": 2.0,
    "bfloat16": 2.0,
    "fp8": 1.0,
    "f8_e4m3": 1.0,
    "int8": 1.0,
    "int8wo": 1.0,
    "int8dq": 1.0,
    "int4": 0.5,
}


def estimate_model_memory_gb(
    model_meta: dict[str, Any],
    quantization: str | None = None,
) -> float:
    """Estimate weight memory in GB, with 20% overhead for CUDA + KV bootstrap.

    Sources in priority order:
      1. safetensors metadata (param counts per dtype) — most accurate
      2. sum of safetensors/bin file sizes — reliable fallback
      3. total param count assuming bf16 — last resort
    """
    safetensors = model_meta.get("safetensors", {})
    parameters = safetensors.get("parameters")
    total_params = safetensors.get("total")
    weight_files_gb = model_meta.get("weight_files_gb")

    weight_gb: float | None = None
    source = ""

    if parameters and isinstance(parameters, dict):
        total_bytes = 0.0
        for dtype, count in parameters.items():
            bpp = BYTES_PER_PARAM.get(dtype.lower().replace("float", "fp"), 2.0)
            total_bytes += count * bpp
        weight_gb = total_bytes / 1e9
        source = "safetensors metadata"
    elif weight_files_gb:
        weight_gb = weight_files_gb
        source = "weight file sizes"
    elif total_params:
        weight_gb = total_params * 2.0 / 1e9
        source = "total param count (assumed bf16)"

    if weight_gb is None:
        print("Warning: could not determine model size from HF metadata; assuming 20 GB.")
        return 20.0

    print(f"  Model weight estimate: {weight_gb:.1f} GB (from {source})")

    if quantization:
        q = quantization.lower()
        if q == "fp8":
            weight_gb *= 0.5
        elif q.startswith("int4wo") or q == "int4":
            weight_gb *= 0.25
        elif q.startswith("int8"):
            weight_gb *= 0.5

    return weight_gb * 1.2  # CUDA context, activations, KV bootstrap


# ---------------------------------------------------------------------------
# Parallelism math
# ---------------------------------------------------------------------------


def _next_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def calculate_parallelism(
    model_memory_gb: float,
    gpu: GPUInventory,
    quantization: str | None = None,
    num_attention_heads: int | None = None,
    num_local_experts: int | None = None,
) -> ParallelismConfig:
    """Pick TP/DP/EP for a model on a GPU spec.

    Strategy: minimize TP (to maximize DP) since higher DP wins on concurrency
    under tight latency SLAs. For MoE models, EP = TP to distribute experts
    across GPUs without sharding individual expert weights.
    """
    usable_vram = gpu.vram_gb
    if usable_vram <= 0:
        sys.exit(f"Error: GPU VRAM too small ({gpu.vram_gb:.1f} GB).")

    min_tp = math.ceil(model_memory_gb / usable_vram)
    tp = _next_power_of_two(min_tp)

    if num_attention_heads and tp > 1 and num_attention_heads % tp != 0:
        original_tp = tp
        candidate = tp * 2
        found = False
        while candidate <= gpu.count:
            if num_attention_heads % candidate == 0:
                tp = candidate
                found = True
                break
            candidate *= 2
        if not found:
            for d in range(min_tp, gpu.count + 1):
                if num_attention_heads % d == 0:
                    tp = d
                    found = True
                    break
        if not found:
            tp = original_tp
        else:
            print(
                f"  Adjusted TP from {original_tp} to {tp}"
                f" (num_attention_heads={num_attention_heads} must be divisible by TP)"
            )

    if tp > gpu.count:
        print(f"Error: model needs TP={tp} but only {gpu.count} GPUs available.")
        print(f"  Model memory estimate: {model_memory_gb:.1f} GB")
        print(f"  Usable VRAM per GPU: {usable_vram:.1f} GB")
        if not quantization:
            print("  Try a larger accelerator, or set QUANTIZATION=fp8 / int4wo-128.")
        sys.exit(1)

    dp = gpu.count // tp
    ep = tp if num_local_experts else 1

    return ParallelismConfig(tp_size=tp, dp_size=dp, ep_size=ep)


# ---------------------------------------------------------------------------
# CLI entry point — called from run.sh
# ---------------------------------------------------------------------------


PARALLELISM_OUT = "/tmp/parallelism.json"


def compute_or_die() -> dict[str, Any]:
    """End-to-end: env vars → model meta → parallelism config → JSON on disk."""
    target_model = os.environ.get("TARGET_MODEL", "").strip()
    accelerators = os.environ.get("ACCELERATORS", "").strip()
    quantization = os.environ.get("QUANTIZATION", "").strip() or None

    if not target_model:
        sys.exit("Error: TARGET_MODEL is required.")
    if not accelerators:
        sys.exit("Error: ACCELERATORS is required (e.g. 'A100:8').")

    validate_hf_credentials()

    print(f"Resolving GPU spec: {accelerators}")
    gpu = parse_accelerators(accelerators)
    print(f"  → {gpu.count}x {gpu.name} ({gpu.vram_gb:.0f} GB each, {gpu.vram_gb * gpu.count:.0f} GB total)")

    print(f"Inspecting model: {target_model}")
    meta = validate_model_accessible(target_model)
    if "num_local_experts" in meta:
        print(f"  MoE detected (num_local_experts={meta['num_local_experts']}) — EP will be enabled.")

    mem_gb = estimate_model_memory_gb(meta, quantization=quantization)
    print(f"  Estimated GPU memory needed: {mem_gb:.1f} GB (incl. 20% overhead)")

    config = calculate_parallelism(
        model_memory_gb=mem_gb,
        gpu=gpu,
        quantization=quantization,
        num_attention_heads=meta.get("num_attention_heads"),
        num_local_experts=meta.get("num_local_experts"),
    )
    print(f"Parallelism: TP={config.tp_size}, DP={config.dp_size}, EP={config.ep_size}")

    out = {**asdict(config), "gpu_count": gpu.count, "gpu_vram_gb": gpu.vram_gb}
    Path(PARALLELISM_OUT).write_text(json.dumps(out, indent=2))
    print(f"Wrote {PARALLELISM_OUT}")
    return out


if __name__ == "__main__":
    compute_or_die()
