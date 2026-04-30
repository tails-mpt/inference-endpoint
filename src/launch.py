"""Launch sglang or vllm with computed parallelism + user-supplied options.

Reads /tmp/parallelism.json (written by parallelism.py) and translates the
shared schema env vars into engine-specific argv. Then execvp's the server.

Engine flag conventions (cross-checked against ai-factory's deploy scripts):
                         sglang                          vllm
  TP                     --tp-size                       --tensor-parallel-size
  DP                     --dp-size                       --data-parallel-size
  EP (MoE)               --ep-size                       --enable-expert-parallel (bool)
  Mem fraction           --mem-fraction-static           --gpu-memory-utilization
  Quantization (online)  --quantization fp8 |            --quantization fp8|bitsandbytes|torchao
                         --torchao-config int4wo-128
  Speculative            --speculative-algorithm + flags --speculative-config '<json>'
  Served name            --served-model-name             --served-model-name
  Max model len          --context-length (when >0)      --max-model-len (or 'auto')
  Dtype                  --dtype                         --dtype
"""

from __future__ import annotations

import json
import os
import shlex
import sys
from pathlib import Path


PARALLELISM_OUT = "/tmp/parallelism.json"
DEFAULT_PORT = 8000
DEFAULT_HOST = "0.0.0.0"
DEFAULT_MEM_FRACTION = "0.85"


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _load_parallelism() -> dict[str, int]:
    p = Path(PARALLELISM_OUT)
    if not p.exists():
        sys.exit(f"Error: {PARALLELISM_OUT} not found. Did parallelism.py run first?")
    return json.loads(p.read_text())


def _build_sglang_argv(par: dict[str, int]) -> list[str]:
    target_model = _env("TARGET_MODEL")
    served_name = _env("SERVED_MODEL_NAME") or Path(target_model).name
    port = _env("PORT") or str(DEFAULT_PORT)
    host = _env("HOST") or DEFAULT_HOST
    mem_fraction = _env("MEM_FRACTION") or DEFAULT_MEM_FRACTION
    dtype = _env("DTYPE")
    quantization = _env("QUANTIZATION")
    max_model_len = _env("MAX_MODEL_LEN")

    argv = [
        "python", "-m", "sglang.launch_server",
        "--model-path", target_model,
        "--served-model-name", served_name,
        "--host", host,
        "--port", port,
        "--tp-size", str(par["tp_size"]),
        "--mem-fraction-static", mem_fraction,
        "--enable-metrics",
    ]
    if par["dp_size"] > 1:
        argv += ["--dp-size", str(par["dp_size"])]
    if par["ep_size"] > 1:
        argv += ["--ep-size", str(par["ep_size"])]
    if dtype:
        argv += ["--dtype", dtype]
    if max_model_len:
        argv += ["--context-length", max_model_len]
    if quantization:
        # fp8 uses --quantization; torchao methods (int4wo-*, int8wo, ...) use --torchao-config.
        if quantization.lower() == "fp8":
            argv += ["--quantization", "fp8"]
        else:
            argv += ["--torchao-config", quantization]

    if _env("SPEC_ENABLED").lower() == "true":
        method = _env("SPEC_METHOD") or "EAGLE3"
        draft = _env("SPEC_DRAFT_MODEL")
        n = _env("SPEC_NUM_TOKENS") or "3"
        if not draft:
            sys.exit("Error: SPEC_ENABLED=true but SPEC_DRAFT_MODEL is empty.")
        argv += [
            "--speculative-algorithm", method.upper(),
            "--speculative-draft-model-path", draft,
            "--speculative-num-draft-tokens", n,
        ]

    return argv


def _build_vllm_argv(par: dict[str, int]) -> list[str]:
    target_model = _env("TARGET_MODEL")
    served_name = _env("SERVED_MODEL_NAME") or Path(target_model).name
    port = _env("PORT") or str(DEFAULT_PORT)
    host = _env("HOST") or DEFAULT_HOST
    mem_fraction = _env("MEM_FRACTION") or DEFAULT_MEM_FRACTION
    dtype = _env("DTYPE")
    quantization = _env("QUANTIZATION")
    max_model_len = _env("MAX_MODEL_LEN") or "auto"

    argv = [
        "vllm", "serve", target_model,
        "--served-model-name", served_name,
        "--host", host,
        "--port", port,
        "--tensor-parallel-size", str(par["tp_size"]),
        "--gpu-memory-utilization", mem_fraction,
        "--max-model-len", max_model_len,
    ]
    if par["dp_size"] > 1:
        argv += ["--data-parallel-size", str(par["dp_size"])]
    if par["ep_size"] > 1:
        argv += ["--enable-expert-parallel"]
    if dtype:
        argv += ["--dtype", dtype]
    if quantization:
        argv += ["--quantization", quantization]

    if _env("SPEC_ENABLED").lower() == "true":
        method = _env("SPEC_METHOD") or "eagle3"
        draft = _env("SPEC_DRAFT_MODEL")
        n = int(_env("SPEC_NUM_TOKENS") or "3")
        if not draft:
            sys.exit("Error: SPEC_ENABLED=true but SPEC_DRAFT_MODEL is empty.")
        spec_config = json.dumps(
            {"model": draft, "method": method.lower(), "num_speculative_tokens": n}
        )
        argv += ["--speculative-config", spec_config]

    return argv


def main() -> None:
    engine = _env("ENGINE", "sglang").lower()
    par = _load_parallelism()

    builders = {"sglang": _build_sglang_argv, "vllm": _build_vllm_argv}
    if engine not in builders:
        sys.exit(f"Error: ENGINE must be one of {list(builders)}, got: {engine!r}")

    argv = builders[engine](par)
    print(f"[launch] {engine} argv: {' '.join(shlex.quote(a) for a in argv)}", flush=True)
    os.execvp(argv[0], argv)


if __name__ == "__main__":
    main()
