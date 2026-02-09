from hy3dgen.agentic.planner import Planner


def test_planner_is_deterministic_for_same_prompt_and_preset():
    planner = Planner()
    plan_a = planner.create_plan(
        prompt="create a red sports car with rigged wheels",
        preset="vehicle_automotive",
    )
    plan_b = planner.create_plan(
        prompt="create a red sports car with rigged wheels",
        preset="vehicle_automotive",
    )

    assert plan_a.seed == plan_b.seed
    assert plan_a.metadata["signature"] == plan_b.metadata["signature"]
    assert plan_a.to_dict() == plan_b.to_dict()


def test_planner_contains_export_and_vehicle_ops():
    planner = Planner()
    plan = planner.create_plan(
        prompt="vehicle with rig and clean topology",
        preset="vehicle_automotive",
    )

    step_types = [s.type for s in plan.steps]
    op_classes = [s.op_class for s in plan.steps if s.type == "op"]

    assert step_types[0] == "import"
    assert step_types[-1] == "export"
    assert "D" in op_classes
    assert "F" in op_classes
