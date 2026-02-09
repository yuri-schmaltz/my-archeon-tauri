from hy3dgen.agentic.executor import PlanExecutor
from hy3dgen.agentic.planner import Planner
from hy3dgen.sidecar.main import handle_request


def test_sidecar_health_request():
    planner = Planner()
    executor = PlanExecutor()

    response = handle_request(
        {"id": "1", "method": "health", "params": {}},
        planner,
        executor,
    )

    assert response["ok"] is True
    assert response["result"]["status"] == "ok"


def test_sidecar_generate_plan_request():
    planner = Planner()
    executor = PlanExecutor()

    response = handle_request(
        {
            "id": "2",
            "method": "generate_plan",
            "params": {
                "prompt": "create a rigged car",
                "preset": "vehicle_automotive",
            },
        },
        planner,
        executor,
    )

    assert response["ok"] is True
    assert response["result"]["preset"] == "vehicle_automotive"
    assert len(response["result"]["steps"]) > 0
