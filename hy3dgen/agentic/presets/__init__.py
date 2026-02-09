from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

PRESET_GENERIC = "generic"
PRESET_VEHICLE_AUTOMOTIVE = "vehicle_automotive"


def preset_dir(preset: str) -> Path:
    return Path(__file__).parent / preset


def load_preset_schema(preset: str) -> Dict[str, Any]:
    if preset == PRESET_GENERIC:
        return {
            "id": PRESET_GENERIC,
            "description": "Generic mesh workflow",
            "required_parts": ["body"],
            "optional_parts": [],
            "rig": {"type": "generic", "bones": ["root"]},
        }

    schema_path = preset_dir(preset) / "schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"Preset schema not found: {schema_path}")
    return json.loads(schema_path.read_text(encoding="utf-8"))
