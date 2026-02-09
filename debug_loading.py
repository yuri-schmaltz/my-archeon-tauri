
import sys
import os
import torch
from hy3dgen.texgen.pipelines import Hunyuan3DPaintPipeline

# Ensure we can import hy3dgen
sys.path.append(os.getcwd())

print("Starting debug loading...")
try:
    # Use the default path used in gradio_app.py
    pipeline = Hunyuan3DPaintPipeline.from_pretrained(
        "tencent/Hunyuan3D-2",
        subfolder="hunyuan3d-paint-v2-0-turbo",
        low_vram_mode=True
    )
    print("Loading successful!")
except Exception as e:
    print(f"Loading failed: {e}")
    import traceback
    traceback.print_exc()
