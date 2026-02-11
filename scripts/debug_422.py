import httpx
import json
import sys

def debug_422():
    # Force use of 127.0.0.1 to avoid localhost resolution issues
    url = "http://127.0.0.1:8081/v1/jobs"
    payload = {
      "mode": "image_to_3d",
      "request_id": "job_debug_manual_999",
      "schema_version": "1.0",
      "input": {
        "text_prompt": "Debug Tank",
        "images": [
          {
            "image_id": "test_img",
            "uri": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
            "role": "reference"
          }
        ]
      },
      "quality": {
        "preset": "standard",
        "steps": 30,
        "seed": 42,
        "determinism": "best_effort"
      },
      "constraints": {
        "target_formats": ["glb"],
        "background": "remove"
      },
      "postprocess": {
        "cleanup": True,
        "retopo": False,
        "decimate": False,
        "bake_textures": True,
        "mesh_simplify_target_tris": 0,
        "remove_hidden": False,
        "fix_normals": True,
        "generate_collision": False
      },
      "batch": {
        "enabled": False,
        "items": [],
        "concurrency_hint": 1
      },
      "output": {
        "artifact_prefix": "debug",
        "return_preview_renders": True
      }
    }
    
    print(f"--- Probing {url} ---")
    try:
        response = httpx.post(url, json=payload, timeout=20.0)
        print(f"HTTP Status: {response.status_code}")
        if response.status_code == 422:
            print("\n[CRITICAL] VALIDATION ERROR DETAILS:")
            print(json.dumps(response.json(), indent=2))
        else:
            print("\nResponse Body:")
            print(response.text)
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")

if __name__ == "__main__":
    debug_422()
