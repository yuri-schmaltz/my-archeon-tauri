from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .presets import PRESET_GENERIC, PRESET_VEHICLE_AUTOMOTIVE
from .types import PlanIR, PlanStep


OP_CLASS_ORDER = ["A", "B", "C", "D", "E", "F"]


OP_CLASS_KEYWORDS = {
    "A": ["generate", "create", "new mesh", "from prompt", "synthesize"],
    "B": ["texture", "material", "pbr", "albedo", "paint", "roughness", "metallic"],
    "C": ["uv", "unwrap", "seam", "island"],
    "D": ["symmetry", "proportion", "scale", "dimension", "align"],
    "E": ["clean", "repair", "weld", "fill", "normals", "manifold"],
    "F": ["rig", "skeleton", "bones", "animate", "humanoid", "vehicle rig"],
}


def _default_step_params(op_class: str) -> Dict[str, object]:
    if op_class == "A":
        return {"engine": "meshtron_stub", "mode": "prompt_to_mesh"}
    if op_class == "B":
        return {"engine": "get3d_stub", "resolution": 2048}
    if op_class == "C":
        return {"method": "xatlas_or_planar"}
    if op_class == "D":
        return {"symmetry_axis": "x", "preserve_scale": True}
    if op_class == "E":
        return {"remove_degenerate_faces": True, "fix_normals": True}
    if op_class == "F":
        return {"rig_type": "auto"}
    return {}


class Planner:
    version = "1.0"

    @staticmethod
    def _seed_from(prompt: str, preset: str) -> int:
        digest = hashlib.sha256(f"{preset}:{prompt}".encode("utf-8")).hexdigest()
        return int(digest[:8], 16)

    @staticmethod
    def _has_keyword(prompt: str, keywords: Iterable[str]) -> bool:
        p = (prompt or "").lower()
        return any(k in p for k in keywords)

    def _infer_op_classes(
        self,
        prompt: str,
        preset: str,
        input_mesh: Optional[str],
    ) -> List[str]:
        classes: List[str] = []
        for op_class in OP_CLASS_ORDER:
            if self._has_keyword(prompt, OP_CLASS_KEYWORDS[op_class]):
                classes.append(op_class)

        # Default generation when no input mesh was supplied.
        if not input_mesh and "A" not in classes:
            classes.insert(0, "A")

        # Automotive preset should enforce proportion/symmetry and rigging defaults.
        if preset == PRESET_VEHICLE_AUTOMOTIVE:
            for required in ("D", "F"):
                if required not in classes:
                    classes.append(required)

        # Include cleanup op by default as post-op guard.
        if "E" not in classes:
            classes.append("E")

        # Keep deterministic order.
        classes = [c for c in OP_CLASS_ORDER if c in classes]
        return classes

    def create_plan(
        self,
        prompt: str,
        preset: str = PRESET_GENERIC,
        input_mesh: Optional[str] = None,
        output_path: str = "output.glb",
        export_format: str = "glb",
        seed: Optional[int] = None,
    ) -> PlanIR:
        preset = preset or PRESET_GENERIC
        seed = self._seed_from(prompt, preset) if seed is None else int(seed)
        op_classes = self._infer_op_classes(prompt, preset, input_mesh)

        steps: List[PlanStep] = []

        def add_step(step_type: str, name: str, **kwargs: object) -> None:
            idx = len(steps) + 1
            steps.append(
                PlanStep(
                    id=f"s{idx:02d}_{step_type}",
                    type=step_type,  # type: ignore[arg-type]
                    name=name,
                    params=dict(kwargs.get("params", {})),
                    op_class=kwargs.get("op_class"),  # type: ignore[arg-type]
                    depends_on=list(kwargs.get("depends_on", [])),  # type: ignore[arg-type]
                )
            )

        add_step("import", "import_mesh_or_prompt", params={"input_mesh": input_mesh})
        add_step("normalize", "normalize_mesh", depends_on=[steps[-1].id])
        add_step("partgraph", "build_partgraph", depends_on=[steps[-1].id])
        add_step("preset", "apply_preset", params={"preset": preset}, depends_on=[steps[-1].id])
        add_step("clean", "baseline_cleanup", depends_on=[steps[-1].id])

        for op_class in op_classes:
            add_step(
                "op",
                f"op_{op_class}",
                op_class=op_class,
                params=_default_step_params(op_class),
                depends_on=[steps[-1].id],
            )

        add_step("validate", "validate_outputs", depends_on=[steps[-1].id])
        add_step(
            "export",
            "export_mesh",
            params={"path": output_path, "format": export_format},
            depends_on=[steps[-1].id],
        )

        signature = hashlib.sha256(
            json.dumps(
                {
                    "prompt": prompt,
                    "preset": preset,
                    "seed": seed,
                    "steps": [s.to_dict() for s in steps],
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

        return PlanIR(
            version=self.version,
            prompt=prompt,
            preset=preset,
            seed=seed,
            steps=steps,
            metadata={
                "signature": signature,
                "op_classes": op_classes,
                "deterministic": True,
            },
        )

    @staticmethod
    def save(plan: PlanIR, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
        return path
