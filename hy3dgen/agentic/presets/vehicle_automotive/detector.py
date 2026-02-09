from __future__ import annotations

from typing import Dict, List


def detect_prompt_parts(prompt: str) -> Dict[str, bool]:
    p = (prompt or "").lower()

    return {
        "wheels": any(k in p for k in ["wheel", "tire", "rim", "car", "vehicle"]),
        "doors": any(k in p for k in ["door", "coupe", "sedan", "truck"]),
        "windows": any(k in p for k in ["window", "glass", "windshield"]),
        "spoiler": any(k in p for k in ["spoiler", "wing"]),
        "headlights": any(k in p for k in ["headlight", "lamp", "lights"]),
    }


def default_vehicle_parts(prompt: str) -> List[str]:
    detected = detect_prompt_parts(prompt)
    parts = ["body", "wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"]

    if detected["doors"]:
        parts.extend(["door_left", "door_right"])
    if detected["windows"]:
        parts.extend(["window_front", "window_rear"])
    if detected["spoiler"]:
        parts.append("spoiler")
    if detected["headlights"]:
        parts.extend(["headlight_left", "headlight_right"])

    return parts
