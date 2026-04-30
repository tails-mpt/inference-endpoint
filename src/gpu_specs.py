"""Parse SkyPilot ACCELERATORS strings into a GPU inventory.

The slipstream user provides ACCELERATORS as a SkyPilot accelerator spec
(e.g. "A100:8", "H100:4", "L4:1"). We map that to per-GPU VRAM via a
static table so parallelism math can run without nvidia-smi — useful both
on the VM and during local validation.
"""

from dataclasses import dataclass


# Per-GPU VRAM in GB, keyed by SkyPilot accelerator name.
# Cross-checked against SkyPilot catalog and ai-factory/config/gpu_pricing.json.
VRAM_GB: dict[str, float] = {
    "T4": 16.0,
    "L4": 24.0,
    "A10": 24.0,
    "A10G": 24.0,
    "V100": 16.0,
    "V100-32GB": 32.0,
    "A100": 40.0,
    "A100-80GB": 80.0,
    "H100": 80.0,
    "H100-MEGA-80GB": 80.0,
    "H200": 141.0,
    "B200": 192.0,
    "B300": 288.0,
}


@dataclass
class GPUInventory:
    """GPU spec for parallelism math. Mirrors ai-factory's shape."""

    name: str
    count: int
    vram_gb: float


def parse_accelerators(spec: str) -> GPUInventory:
    """Parse a SkyPilot accelerator spec like 'A100:8' into a GPUInventory.

    Raises ValueError on unknown GPU names or malformed input so we fail
    fast with an actionable message instead of mis-allocating downstream.
    """
    if not spec or ":" not in spec:
        raise ValueError(
            f"ACCELERATORS must be 'NAME:COUNT' (e.g. 'A100:8'), got: {spec!r}"
        )
    name, _, count_str = spec.partition(":")
    name = name.strip()
    try:
        count = int(count_str.strip())
    except ValueError:
        raise ValueError(f"GPU count must be an integer, got: {count_str!r}")
    if count <= 0:
        raise ValueError(f"GPU count must be positive, got: {count}")
    if name not in VRAM_GB:
        known = ", ".join(sorted(VRAM_GB))
        raise ValueError(
            f"Unknown GPU type {name!r}. Known: {known}. "
            f"Add it to gpu_specs.VRAM_GB if SkyPilot supports it."
        )
    return GPUInventory(name=name, count=count, vram_gb=VRAM_GB[name])
