from pathlib import Path

import trimesh

from hy3dgen.agentic.executor import PlanExecutor
from hy3dgen.agentic.planner import Planner


def test_executor_writes_plan_and_mesh(tmp_path: Path):
    planner = Planner()
    executor = PlanExecutor()

    plan = planner.create_plan(
        prompt="create a blue vehicle with rigged wheels",
        preset="vehicle_automotive",
        output_path="artifacts/output.glb",
    )

    result = executor.execute(plan, workspace=tmp_path)

    assert result.success
    assert "plan" in result.artifacts
    assert "mesh" in result.artifacts
    assert Path(result.artifacts["plan"]).exists()
    assert Path(result.artifacts["mesh"]).exists()

    loaded = trimesh.load(result.artifacts["mesh"], force="mesh", process=False)
    assert isinstance(loaded, trimesh.Trimesh)
    assert len(loaded.faces) > 0
