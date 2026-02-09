from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import trimesh

from .planner import Planner
from .presets import PRESET_VEHICLE_AUTOMOTIVE, load_preset_schema
from .presets.vehicle_automotive.detector import default_vehicle_parts
from .presets.vehicle_automotive.segmenter import segment_vehicle_mesh
from .runtime import detect_runtime
from .types import ExecutionResult, PlanIR
from .validator import Validator


class PlanExecutor:
    def __init__(self) -> None:
        self.validator = Validator()

    def execute(
        self,
        plan: PlanIR,
        workspace: str | Path,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        workspace_path = Path(workspace)
        workspace_path.mkdir(parents=True, exist_ok=True)

        runtime = detect_runtime()
        trace = []
        artifacts: Dict[str, str] = {}
        validation: Dict[str, Any] = {}

        plan_check = self.validator.validate_plan(plan)
        if not plan_check["valid"]:
            return ExecutionResult(
                success=False,
                plan=plan.to_dict(),
                artifacts=artifacts,
                validation={"plan": plan_check},
                runtime=runtime,
                trace=[{"step": "validate_plan", "ok": False, "errors": plan_check["errors"]}],
            )

        mesh: Optional[trimesh.Trimesh] = None
        partgraph: Dict[str, Any] = {}
        preset_schema = load_preset_schema(plan.preset)

        Planner.save(plan, workspace_path / "plan.json")
        artifacts["plan"] = str(workspace_path / "plan.json")

        for step in plan.steps:
            if not step.enabled:
                trace.append({"step": step.id, "ok": True, "skipped": True})
                continue

            try:
                if step.type == "import":
                    mesh = self._handle_import(step.params, plan)

                elif step.type == "normalize":
                    if mesh is None:
                        raise ValueError("normalize requires an imported/generated mesh")
                    mesh = self._normalize(mesh)

                elif step.type == "partgraph":
                    if mesh is None:
                        raise ValueError("partgraph requires a mesh")
                    partgraph = self._build_partgraph(mesh, plan)
                    partgraph_path = workspace_path / "partgraph.json"
                    partgraph_path.write_text(json.dumps(partgraph, indent=2), encoding="utf-8")
                    artifacts["partgraph"] = str(partgraph_path)

                elif step.type == "preset":
                    preset_payload = {
                        "preset": plan.preset,
                        "schema": preset_schema,
                    }
                    preset_path = workspace_path / "preset_applied.json"
                    preset_path.write_text(json.dumps(preset_payload, indent=2), encoding="utf-8")
                    artifacts["preset"] = str(preset_path)

                elif step.type == "clean":
                    if mesh is None:
                        raise ValueError("clean requires a mesh")
                    mesh = self._cleanup(mesh)

                elif step.type == "op":
                    if mesh is None and step.op_class != "A":
                        raise ValueError(f"op {step.op_class} requires a mesh")
                    mesh, new_artifacts = self._run_op(
                        mesh=mesh,
                        op_class=step.op_class,
                        step_params=step.params,
                        plan=plan,
                        partgraph=partgraph,
                        schema=preset_schema,
                        workspace=workspace_path,
                    )
                    artifacts.update(new_artifacts)

                elif step.type == "validate":
                    if mesh is None:
                        raise ValueError("validate requires a mesh")
                    validation["mesh"] = self.validator.validate_mesh(mesh)

                elif step.type == "export":
                    if mesh is None:
                        raise ValueError("export requires a mesh")
                    export_path = self._export(mesh, workspace_path, step.params)
                    artifacts["mesh"] = str(export_path)
                    validation["export"] = self.validator.validate_export(str(export_path))

                else:
                    raise ValueError(f"Unsupported step type: {step.type}")

                trace.append({"step": step.id, "ok": True, "type": step.type})
            except Exception as exc:
                trace.append(
                    {
                        "step": step.id,
                        "ok": False,
                        "type": step.type,
                        "error": str(exc),
                    }
                )
                return ExecutionResult(
                    success=False,
                    plan=plan.to_dict(),
                    artifacts=artifacts,
                    validation=validation,
                    runtime=runtime,
                    trace=trace,
                )

        success = bool(validation.get("mesh", {}).get("valid", True)) and bool(
            validation.get("export", {}).get("valid", True)
        )

        return ExecutionResult(
            success=success,
            plan=plan.to_dict(),
            artifacts=artifacts,
            validation=validation,
            runtime=runtime,
            trace=trace,
        )

    def _handle_import(self, params: Dict[str, Any], plan: PlanIR) -> trimesh.Trimesh:
        input_mesh = params.get("input_mesh")
        if input_mesh:
            loaded = trimesh.load(input_mesh, force="mesh", process=False)
            if isinstance(loaded, trimesh.Scene):
                if not loaded.geometry:
                    raise ValueError("Imported scene has no geometry")
                loaded = trimesh.util.concatenate([g.copy() for g in loaded.geometry.values()])
            if not isinstance(loaded, trimesh.Trimesh):
                raise TypeError(f"Unsupported imported mesh type: {type(loaded)}")
            return loaded

        return self._generate_prompt_mesh(plan.prompt, plan.preset)

    def _generate_prompt_mesh(self, prompt: str, preset: str) -> trimesh.Trimesh:
        p = (prompt or "").lower()
        if preset == PRESET_VEHICLE_AUTOMOTIVE or any(k in p for k in ["car", "vehicle", "truck"]):
            body = trimesh.creation.box(extents=(2.2, 0.95, 0.65))
            roof = trimesh.creation.box(extents=(1.2, 0.8, 0.35))
            roof.apply_translation((0.1, 0.0, 0.45))

            wheels = []
            for x in (-0.75, 0.75):
                for y in (-0.52, 0.52):
                    wheel = trimesh.creation.cylinder(radius=0.26, height=0.2, sections=28)
                    wheel.apply_transform(
                        trimesh.transformations.rotation_matrix(np.pi / 2.0, [0.0, 1.0, 0.0])
                    )
                    wheel.apply_translation((x, y, -0.25))
                    wheels.append(wheel)

            return trimesh.util.concatenate([body, roof, *wheels])

        if any(k in p for k in ["character", "humanoid", "person"]):
            body = trimesh.creation.capsule(radius=0.24, height=1.0, count=[24, 24])
            body.apply_translation((0.0, 0.0, 0.2))
            return body

        return trimesh.creation.icosphere(subdivisions=3, radius=0.6)

    def _normalize(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        mesh = mesh.copy()
        bounds = np.asarray(mesh.bounds)
        center = (bounds[0] + bounds[1]) * 0.5
        extents = bounds[1] - bounds[0]
        max_extent = float(np.max(extents)) if extents.size else 0.0
        if max_extent <= 1e-8:
            return mesh
        mesh.apply_translation(-center)
        mesh.apply_scale(1.6 / max_extent)
        return mesh

    def _build_partgraph(self, mesh: trimesh.Trimesh, plan: PlanIR) -> Dict[str, Any]:
        if plan.preset == PRESET_VEHICLE_AUTOMOTIVE:
            parts = default_vehicle_parts(plan.prompt)
            semantic = segment_vehicle_mesh(mesh, parts)
            return {
                "preset": plan.preset,
                "parts": parts,
                "semantic": semantic,
            }

        return {
            "preset": plan.preset,
            "parts": ["body"],
            "semantic": {"body": {"faces": list(range(len(mesh.faces)))}}
        }

    def _cleanup(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        mesh = mesh.copy()
        mesh.remove_unreferenced_vertices()
        mesh.remove_infinite_values()
        mesh.update_faces(mesh.nondegenerate_faces())

        try:
            from trimesh import repair

            repair.fix_normals(mesh)
            repair.fix_inversion(mesh)
        except Exception:
            pass

        return mesh

    def _run_op(
        self,
        mesh: Optional[trimesh.Trimesh],
        op_class: Optional[str],
        step_params: Dict[str, Any],
        plan: PlanIR,
        partgraph: Dict[str, Any],
        schema: Dict[str, Any],
        workspace: Path,
    ) -> tuple[trimesh.Trimesh, Dict[str, str]]:
        if op_class is None:
            raise ValueError("op step missing op_class")

        artifacts: Dict[str, str] = {}

        if op_class == "A":
            if mesh is None:
                mesh = self._generate_prompt_mesh(plan.prompt, plan.preset)

        elif op_class == "B":
            if mesh is None:
                raise ValueError("Texture op requires mesh")
            mesh = mesh.copy()
            self._paint_mesh(mesh, plan.prompt)

        elif op_class == "C":
            if mesh is None:
                raise ValueError("UV op requires mesh")
            mesh = mesh.copy()
            self._ensure_uv(mesh)

        elif op_class == "D":
            if mesh is None:
                raise ValueError("Symmetry op requires mesh")
            mesh = mesh.copy()
            verts = mesh.vertices.copy()
            # Enforce centered proportional model in X axis without destructive remeshing.
            verts[:, 0] = verts[:, 0] - np.mean(verts[:, 0])
            x_max = float(np.max(np.abs(verts[:, 0]))) if len(verts) else 0.0
            if x_max > 1e-8:
                verts[:, 0] /= x_max
            mesh.vertices = verts

        elif op_class == "E":
            if mesh is None:
                raise ValueError("Cleanup op requires mesh")
            mesh = self._cleanup(mesh)

        elif op_class == "F":
            if mesh is None:
                raise ValueError("Rigging op requires mesh")
            rig_type = step_params.get("rig_type") or schema.get("rig", {}).get("type", "generic")
            rig = {
                "rig_type": rig_type,
                "bones": schema.get("rig", {}).get("bones", ["root"]),
                "parts": partgraph.get("parts", ["body"]),
            }
            rig_path = workspace / "rig.json"
            rig_path.write_text(json.dumps(rig, indent=2), encoding="utf-8")
            artifacts["rig"] = str(rig_path)

        else:
            raise ValueError(f"Unsupported op_class: {op_class}")

        if mesh is None:
            raise ValueError("Op pipeline did not produce a mesh")

        return mesh, artifacts

    def _paint_mesh(self, mesh: trimesh.Trimesh, prompt: str) -> None:
        color_map = {
            "red": [220, 40, 40, 255],
            "blue": [40, 90, 220, 255],
            "green": [45, 170, 80, 255],
            "white": [220, 220, 220, 255],
            "black": [35, 35, 35, 255],
            "yellow": [230, 180, 20, 255],
        }
        base = [200, 200, 200, 255]
        text = (prompt or "").lower()

        for key, value in color_map.items():
            if key in text:
                base = value
                break

        colors = np.tile(np.asarray(base, dtype=np.uint8), (len(mesh.faces), 1))
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh)
        mesh.visual.face_colors = colors

    def _ensure_uv(self, mesh: trimesh.Trimesh) -> None:
        if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
            return

        vertices = np.asarray(mesh.vertices)
        min_coords = vertices.min(axis=0)
        max_coords = vertices.max(axis=0)
        ranges = max_coords - min_coords
        ranges[ranges == 0] = 1.0

        uv = (vertices[:, :2] - min_coords[:2]) / ranges[:2]
        uv = np.clip(uv, 0.0, 1.0)

        material = trimesh.visual.material.SimpleMaterial(diffuse=[255, 255, 255, 255])
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)

    def _export(self, mesh: trimesh.Trimesh, workspace: Path, params: Dict[str, Any]) -> Path:
        out = params.get("path") or "output.glb"
        fmt = (params.get("format") or "glb").lower()

        out_path = Path(out)
        if not out_path.is_absolute():
            out_path = workspace / out_path

        out_path.parent.mkdir(parents=True, exist_ok=True)
        if fmt in {"glb", "gltf"}:
            mesh.export(out_path, include_normals=True)
        else:
            mesh.export(out_path, file_type=fmt)
        return out_path
