from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import trimesh

from .types import PlanIR


STEP_ORDER = {
    "import": 0,
    "normalize": 1,
    "partgraph": 2,
    "preset": 3,
    "clean": 4,
    "op": 5,
    "validate": 6,
    "export": 7,
}


class Validator:
    def validate_plan(self, plan: PlanIR) -> Dict[str, Any]:
        errors: List[str] = []
        warnings: List[str] = []

        if not plan.steps:
            errors.append("Plan has no steps")
            return {"valid": False, "errors": errors, "warnings": warnings}

        if plan.steps[0].type != "import":
            errors.append("First step must be 'import'")
        if plan.steps[-1].type != "export":
            errors.append("Last step must be 'export'")

        last_rank = -1
        for step in plan.steps:
            if step.type not in STEP_ORDER:
                errors.append(f"Unknown step type: {step.type}")
                continue

            rank = STEP_ORDER[step.type]
            if rank < last_rank:
                errors.append(
                    f"Step order regression at {step.id} ({step.type})"
                )
            last_rank = rank

            if step.type == "op" and step.op_class not in {"A", "B", "C", "D", "E", "F"}:
                errors.append(f"Invalid op_class in {step.id}: {step.op_class}")

        return {
            "valid": not errors,
            "errors": errors,
            "warnings": warnings,
        }

    def validate_mesh(self, mesh: trimesh.Trimesh) -> Dict[str, Any]:
        errors: List[str] = []
        warnings: List[str] = []

        if mesh is None:
            return {"valid": False, "errors": ["Mesh is None"], "warnings": []}

        verts = len(mesh.vertices)
        faces = len(mesh.faces)
        if verts == 0:
            errors.append("Mesh has 0 vertices")
        if faces == 0:
            errors.append("Mesh has 0 faces")

        try:
            bounds = np.asarray(mesh.bounds)
            if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
                errors.append("Mesh bounds are invalid")
        except Exception:
            errors.append("Failed to compute mesh bounds")

        if not mesh.is_watertight:
            warnings.append("Mesh is not watertight")

        return {
            "valid": not errors,
            "errors": errors,
            "warnings": warnings,
            "metrics": {
                "vertices": verts,
                "faces": faces,
                "watertight": bool(mesh.is_watertight),
                "winding_consistent": bool(mesh.is_winding_consistent),
            },
        }

    def validate_export(self, path: str) -> Dict[str, Any]:
        p = Path(path)
        errors: List[str] = []
        if not p.exists():
            errors.append(f"Export does not exist: {p}")
        elif p.stat().st_size == 0:
            errors.append(f"Export is empty: {p}")

        return {
            "valid": not errors,
            "errors": errors,
            "warnings": [],
        }
