import argparse
import uvicorn
import sys
import logging
from fastapi import FastAPI
from contextlib import asynccontextmanager
import asyncio

from hy3dgen.manager import ModelManager, PriorityRequestManager

from hy3dgen.api.routes import router
import hy3dgen.api.routes as routes_module

# Logging
import logging
from fastapi import FastAPI
from contextlib import asynccontextmanager
import asyncio

from hy3dgen.manager import ModelManager, PriorityRequestManager

from hy3dgen.api.routes import router
import hy3dgen.api.routes as routes_module
from hy3dgen.utils.system import setup_logging

# Logging
logger = setup_logging("api_server")

def create_app(args=None):
    if args is None:
         parser = argparse.ArgumentParser()
         # Defaults matching main()
         parser.add_argument("--model_path", type=str, default='tencent/Hunyuan3D-2')
         parser.add_argument("--subfolder", type=str, default='hunyuan3d-dit-v2-0-turbo')
         parser.add_argument("--texgen_model_path", type=str, default='tencent/Hunyuan3D-2')
         parser.add_argument('--port', type=int, default=8081)
         parser.add_argument('--host', type=str, default='127.0.0.1')
         parser.add_argument('--device', type=str, default='cuda')
         parser.add_argument('--low_vram_mode', action='store_true', default=True)
         args, _ = parser.parse_known_args()

    logger.info(f"Initializing Archeon 3D API Server on {args.host}:{args.port}")
    
    # 1. Setup Backend Managers
    model_mgr = ModelManager(capacity=1 if args.low_vram_mode else 3, device=args.device)
    
    def get_loader(model_path, subfolder):
        from hy3dgen.inference import InferencePipeline
        return lambda: InferencePipeline(
            model_path=model_path, 
            tex_model_path=args.texgen_model_path, 
            subfolder=subfolder,
            device=args.device, 
            enable_t2i=True, 
            enable_tex=True, 
            low_vram_mode=args.low_vram_mode
        )
    
    model_mgr.register_model("Normal", get_loader(args.model_path, args.subfolder))
    
    request_manager = PriorityRequestManager(model_mgr, max_concurrency=1)
    
    # 2. Inject Manager into Routes
    routes_module.request_manager = request_manager
    
    # 3. Define App Lifespan
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Starting Worker Manager...")
        asyncio.create_task(request_manager.start())
        
        # Auto-open browser in a separate thread to avoid blocking or signals
        def open_browser(url):
            import time
            import webbrowser
            time.sleep(1.5) # Wait for server to bind
            try:
                logger.info(f"Opening browser at {url}")
                webbrowser.open(url)
            except Exception as e:
                logger.warning(f"Failed to open browser: {e}")

        import threading
        url = f"http://{args.host}:{args.port}"
        browser_thread = threading.Thread(target=open_browser, args=(url,))
        browser_thread.daemon = True
        browser_thread.start()
        
        yield
        logger.info("Stopping Worker Manager...")
        await request_manager.stop()
    
    from fastapi.middleware.cors import CORSMiddleware
    
    app = FastAPI(title="Archeon 3D API", version="1.0.0", lifespan=lifespan)
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.include_router(router)

    @app.get("/v1/system/monitor")
    async def get_system_status():
        import torch
        status = {"gpu": {"available": torch.cuda.is_available()}}
        if status["gpu"]["available"]:
            status["gpu"]["vram"] = {
                "total": torch.cuda.get_device_properties(0).total_memory,
                "allocated": torch.cuda.memory_allocated(0),
                "reserved": torch.cuda.memory_reserved(0),
                "free": torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)
            }
        return status

    @app.get("/v1/history")
    async def get_history():
        import json
        import os
        history_path = "/home/yurix/Documentos/my-archeon-tauri/data/history.json"
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                return json.load(f)
        return {"entries": []}

    @app.post("/v1/history/add")
    async def add_history(entry: dict):
        import json
        import os
        history_path = "/home/yurix/Documentos/my-archeon-tauri/data/history.json"
        data = {"entries": []}
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r') as f:
                    data = json.load(f)
            except: pass
        
        entry["timestamp"] = entry.get("timestamp", __import__("datetime").datetime.now().isoformat())
        data["entries"].unshift(entry) if hasattr(list, "unshift") else data["entries"].insert(0, entry)
        
        with open(history_path, 'w') as f:
            json.dump(data, f, indent=2)
        return {"status": "ok"}

    @app.get("/v1/i18n")
    async def get_i18n():
        from hy3dgen.i18n import TRANSLATIONS
        return TRANSLATIONS

    @app.get("/v1/system/downloads")
    async def get_downloads():
        # In a real scenario, we'd hook into huggingface_hub callbacks.
        # For now, we'll check if models exist and return a status.
        import os
        from huggingface_hub import scan_cache_dir
        
        status = {"active": False, "progress": 100, "models": []}
        try:
            cache = scan_cache_dir()
            for repo in cache.repos:
                if "Hunyuan3D" in repo.repo_id:
                    status["models"].append({"id": repo.repo_id, "size": repo.size_on_disk})
        except: pass
        return status

    
    # 4. Mount Frontend (Static Files) & Auto-Open
    from fastapi.staticfiles import StaticFiles
    import os
    
    # Resolve path to src (root/src)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../.."))
    frontend_dir = os.path.join(project_root, "src")
    
    if os.path.exists(frontend_dir):
        app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
        logger.info(f"Frontend served from {frontend_dir}")
    else:
        logger.warning(f"Frontend directory not found at {frontend_dir}")

    return app, args

def main():
    app, args = create_app()
    # 5. Run
    uvicorn.run(app, host=args.host, port=args.port) 

if __name__ == "__main__":
    main()
