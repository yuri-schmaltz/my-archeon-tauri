from __future__ import annotations

from typing import Dict, List

import numpy as np
import trimesh


def _face_groups_by_x(mesh: trimesh.Trimesh) -> Dict[str, List[int]]:
    if len(mesh.faces) == 0:
        return {"left": [], "right": [], "center": []}

    centers = mesh.triangles_center
    left = np.where(centers[:, 0] < -0.05)[0].tolist()
    right = np.where(centers[:, 0] > 0.05)[0].tolist()
    center = np.where(np.abs(centers[:, 0]) <= 0.05)[0].tolist()
    return {"left": left, "right": right, "center": center}


def segment_vehicle_mesh(mesh: trimesh.Trimesh, parts: List[str]) -> Dict[str, Dict[str, List[int]]]:
    x_groups = _face_groups_by_x(mesh)

    semantic: Dict[str, Dict[str, List[int]]] = {
        "body": {"faces": x_groups["center"]}
    }

    for part in parts:
        if part in semantic:
            continue
        if part.endswith("_left") or part.endswith("_fl") or part.endswith("_rl"):
            semantic[part] = {"faces": x_groups["left"]}
        elif part.endswith("_right") or part.endswith("_fr") or part.endswith("_rr"):
            semantic[part] = {"faces": x_groups["right"]}
        else:
            semantic[part] = {"faces": x_groups["center"]}

    return semantic
