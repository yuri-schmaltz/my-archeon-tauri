import trimesh

from hy3dgen.agentic.types import PlanIR
from hy3dgen.agentic.validator import Validator


def test_validate_plan_rejects_wrong_last_step():
    validator = Validator()
    plan = PlanIR.from_dict(
        {
            "version": "1.0",
            "prompt": "x",
            "preset": "generic",
            "seed": 1,
            "steps": [
                {"id": "s01", "type": "import", "name": "import"},
                {"id": "s02", "type": "validate", "name": "validate"},
            ],
        }
    )

    result = validator.validate_plan(plan)
    assert not result["valid"]
    assert any("Last step must be 'export'" in err for err in result["errors"])


def test_validate_mesh_accepts_basic_mesh():
    validator = Validator()
    mesh = trimesh.creation.box(extents=(1, 1, 1))
    result = validator.validate_mesh(mesh)
    assert result["valid"]
    assert result["metrics"]["vertices"] > 0
    assert result["metrics"]["faces"] > 0
