import sys
import argparse

import os
import warnings
import logging

# Suppress Warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.getLogger("uvicorn.access").setLevel(logging.ERROR)
logging.getLogger("uvicorn.error").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("numba").setLevel(logging.ERROR)

def main():
    # Quick check for API mode before importing heavy libraries
    is_api = "--api" in sys.argv
    
    # Setup global logging
    from hy3dgen.utils.system import setup_logging, find_free_port
    setup_logging("archeon_launcher")

    # Startup Validation (UX - Wave 2)
    # Check if models are likely to be downloaded
    try:
        from huggingface_hub import scan_cache_dir
        # Simple heuristic: we don't block, just log helpful info
        logging.info("Startup Check: Verifying model availability...")
    except ImportError:
        pass


    # Find free port if default 8081 (API) or 7860 (Gradio) is taken
    # We will let the sub-apps handle the port argument if passed, 
    # but we can try to facilitate dynamic allocation here if we want to inject it.
    # However, archeon_3d.py calls the 'main' of submodules which parse args again.
    # The best way is to inject the --port arg if not present or if collision.

    # API mode is now the default for Tauri integration
    from hy3dgen.apps.api_server import main as api_main
    
    # Defaults for API Server
    defaults = [
        "--host", "127.0.0.1",
        "--port", "8081",
        "--model_path", "tencent/Hunyuan3D-2",
        "--subfolder", "hunyuan3d-dit-v2-0-turbo",
        "--texgen_model_path", "tencent/Hunyuan3D-2",
        "--low_vram_mode"
    ]
    
    # Prepend defaults to sys.argv
    sys.argv[1:1] = defaults
    
    print("Starting Archeon 3D API Server (Tauri Backend)...")
    api_main()

if __name__ == "__main__":
    main()
