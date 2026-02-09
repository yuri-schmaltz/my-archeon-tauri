from __future__ import annotations

import os
from typing import Any, Dict


def _detect_system_ram_gb() -> float:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        return float(page_size * pages) / (1024 ** 3)
    except Exception:
        return 0.0


def detect_runtime() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "device": "cpu",
        "cuda_available": False,
        "gpu_name": None,
        "vram_gb": 0.0,
        "ram_gb": round(_detect_system_ram_gb(), 2),
        "mode": "cpu",
        "quantization": "none",
        "recommendation": "CPU fallback",
    }

    try:
        import torch

        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            vram_gb = float(props.total_memory) / (1024 ** 3)

            info.update(
                {
                    "device": f"cuda:{idx}",
                    "cuda_available": True,
                    "gpu_name": props.name,
                    "vram_gb": round(vram_gb, 2),
                }
            )

            if vram_gb < 6:
                info["mode"] = "cpu"
                info["quantization"] = "none"
                info["recommendation"] = "Use CPU or very small model"
            elif vram_gb < 12:
                info["mode"] = "low_vram"
                info["quantization"] = "int8"
                info["recommendation"] = "Enable low VRAM mode"
            else:
                info["mode"] = "full"
                info["quantization"] = "none"
                info["recommendation"] = "Use full model path"
    except Exception:
        # Torch is optional for sidecar orchestration.
        pass

    return info
