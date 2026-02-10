from __future__ import annotations

import asyncio

from hy3dgen.agentic import PlanExecutor, PlanIR, Planner, detect_runtime


logger = logging.getLogger("hy3dgen.sidecar")


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        stream=sys.stderr,
    )


def _json_error(req_id: Any, message: str) -> Dict[str, Any]:
    return {"id": req_id, "ok": False, "error": message}


async def handle_request(request: Dict[str, Any], planner: Planner, executor: PlanExecutor) -> Dict[str, Any]:
    req_id = request.get("id")
    method = request.get("method")
    params = request.get("params", {})

    if not isinstance(params, dict):
        return _json_error(req_id, "params must be an object")

    try:
        if method == "health":
            result = {
                "name": "my-archeon-tauri-sidecar",
                "version": "0.1.0",
                "status": "ok",
            }

        elif method == "detect_runtime":
            result = detect_runtime()

        elif method == "generate_plan":
            plan = planner.create_plan(
                prompt=str(params.get("prompt", "")),
                preset=str(params.get("preset", "generic")),
                input_mesh=params.get("input_mesh"),
                output_path=str(params.get("output_path", "output.glb")),
                export_format=str(params.get("export_format", "glb")),
                seed=params.get("seed"),
            )
            result = plan.to_dict()

        elif method == "execute_plan":
            workspace = params.get("workspace") or tempfile.mkdtemp(prefix="archeon-sidecar-")
            plan_payload = params.get("plan")
            if plan_payload:
                plan = PlanIR.from_dict(plan_payload)
            else:
                plan = planner.create_plan(
                    prompt=str(params.get("prompt", "")),
                    preset=str(params.get("preset", "generic")),
                    input_mesh=params.get("input_mesh"),
                    output_path=str(params.get("output_path", "output.glb")),
                    export_format=str(params.get("export_format", "glb")),
                    seed=params.get("seed"),
                )

            exec_result = executor.execute(
                plan=plan,
                workspace=Path(workspace),
                context=params.get("context"),
            )
            result = exec_result.to_dict()

        elif method == "mesh_ops":
            from hy3dgen.meshops.engine import MeshOpsEngine
            from hy3dgen.api.schemas import MeshOpsRequest
            engine = MeshOpsEngine()
            # Sidecar receives plain dict, we validate to MeshOpsRequest
            ops_req = MeshOpsRequest.model_validate(params)
            artifacts = await engine.process_async(ops_req)
            result = {"artifacts": [a.model_dump() for a in artifacts]}

        else:
            return _json_error(req_id, f"Unknown method: {method}")

        return {"id": req_id, "ok": True, "result": result}
    except Exception as exc:
        logger.exception("Request handling failed")
        return _json_error(req_id, str(exc))


async def run_stdio(planner: Planner, executor: PlanExecutor) -> int:
    # Use a thread for blocking stdin read if necessary, but for JSONL it's usually fine
    # with a simple loop if the processing itself is async.
    # For Tauri, we want to be responsive.
    
    loop = asyncio.get_event_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    while True:
        raw = await reader.readline()
        if not raw:
            break
            
        line = raw.decode().strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                response = _json_error(None, "request must be a JSON object")
            else:
                response = await handle_request(request, planner, executor)
        except json.JSONDecodeError as exc:
            response = _json_error(None, f"invalid json: {exc}")

        sys.stdout.write(json.dumps(response, ensure_ascii=True) + "\n")
        sys.stdout.flush()

    return 0


async def main_async() -> int:
    _configure_logging()

    parser = argparse.ArgumentParser(description="Archeon sidecar bridge")
    parser.add_argument("--method", type=str, default=None, help="Single-shot method name")
    parser.add_argument(
        "--params",
        type=str,
        default="{}",
        help="JSON object with method params for single-shot mode",
    )
    parser.add_argument(
        "--stdio",
        action="store_true",
        help="JSONL mode (read requests from stdin and write responses to stdout)",
    )
    args = parser.parse_args()

    planner = Planner()
    executor = PlanExecutor()

    if args.method:
        try:
            params = json.loads(args.params)
            if not isinstance(params, dict):
                raise ValueError("--params must be a JSON object")
        except Exception as exc:
            sys.stderr.write(f"Invalid --params: {exc}\n")
            return 2

        response = await handle_request(
            {"id": "cli", "method": args.method, "params": params},
            planner,
            executor,
        )
        sys.stdout.write(json.dumps(response, ensure_ascii=True) + "\n")
        return 0 if response.get("ok") else 1

    # Default to stdio mode so it can run directly as Tauri sidecar.
    return await run_stdio(planner, executor)


def main() -> int:
    try:
        return asyncio.run(main_async())
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
