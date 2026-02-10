
print("Script started...", flush=True)
import os
import sys
import argparse
import logging
import time
import glob
import base64
import httpx
from pathlib import Path

def setup_batch_logging():
    # Simplistic logger to avoid file issues
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger("batch_tester")

def find_images(root_dir):
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp']
    images = []
    root = Path(root_dir)
    for ext in extensions:
        images.extend(root.rglob(ext))
        images.extend(root.rglob(ext.upper()))
    return sorted(list(set(images)))

def main():
    parser = argparse.ArgumentParser(description="Batch process images via Archeon API")
    parser.add_argument("--input_dir", type=str, default="/home/yurix/Downloads/pins", help="Directory containing images")
    parser.add_argument("--output_dir", type=str, default="batch_outputs", help="Directory to save results")
    parser.add_argument("--steps", type=int, default=30, help="Inference steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images")
    parser.add_argument("--api_url", type=str, default="http://127.0.0.1:8081", help="API Server URL")
    parser.add_argument("--skip_texture", action="store_true", help="Skip texturing")
    
    print("Parsing args...", flush=True)
    args = parser.parse_args()
    print("Args parsed.", flush=True)
    logger = setup_batch_logging()
    
    logger.info(f"Starting batch processing via API: {args.api_url}")
    print(f"Searching for images in {args.input_dir}...", flush=True)
    
    # 1. Find Images
    image_files = find_images(args.input_dir)
    print(f"Found {len(image_files)} images.", flush=True)
    logger.info(f"Found {len(image_files)} images.")
    
    if args.limit > 0:
        image_files = image_files[:args.limit]
        
    if not image_files:
        logger.warning("No images found! Exiting.")
        return

    # 2. Process Loop
    output_root = Path(args.output_dir)
    output_root.mkdir(exist_ok=True, parents=True)
    
    success_count = 0
    fail_count = 0
    
    for i, img_path in enumerate(image_files):
        # Prepare output directory
        base_name = img_path.stem
        safe_name = "".join([c if c.isalnum() else "_" for c in base_name])
        img_output_dir = output_root / safe_name
        img_output_dir.mkdir(exist_ok=True)

        if any(img_output_dir.iterdir()):
            print(f"[{i+1}/{len(image_files)}] Skipping {img_path.name}: Output exists.", flush=True)
            continue

        logger.info(f"[{i+1}/{len(image_files)}] Processing {img_path.name}...")
        try:
            # 3a. Read Image as Base64 (Data URI)
            with open(img_path, "rb") as f:
                img_bytes = f.read()
                b64_str = base64.b64encode(img_bytes).decode('utf-8')
                mime_type = "image/png" if img_path.suffix.lower() == ".png" else "image/jpeg"
                data_uri = f"data:{mime_type};base64,{b64_str}"

            # 3b. Submit Job
            t0 = time.time()
            # 3b. Submit Job
            t0 = time.time()
            job_req_id = f"batch_{safe_name}_{int(time.time())}"
            
            payload = {
                "request_id": job_req_id,
                "schema_version": "1.0",
                "mode": "image_to_3d",
                "input": {
                    "text_prompt": "Image to 3D",
                    "images": [
                        {
                            "image_id": "source_img",
                            "uri": data_uri,
                            "role": "reference"
                        }
                    ]
                },
                "constraints": {
                    "target_formats": ["glb", "obj", "stl"],
                    "materials": {
                        "pbr": not args.skip_texture,
                        "texture_resolution": 1024,
                        "maps": ["basecolor", "normal", "roughness", "metallic"],
                        "single_material": True
                    } if not args.skip_texture else None,
                    "topology": {
                        "watertight": True, 
                        "manifold": True, 
                        "no_self_intersections": False
                    }
                },
                "quality": {
                    "preset": "standard",
                    "steps": args.steps,
                    "seed": args.seed,
                    "determinism": "best_effort"
                },
                "postprocess": {
                    "cleanup": True,
                    "retopo": False,
                    "decimate": False,
                    "bake_textures": False,
                    "remove_hidden": True,
                    "fix_normals": True,
                    "generate_collision": False
                },
                "batch": {
                    "enabled": False
                },
                "output": {
                    "artifact_prefix": job_req_id,
                    "return_preview_renders": False
                }
            }
            
            # Use httpx synchronously for simplicity or just run async?
            # Creating a sync wrapper or just use httpx.post
            try:
                resp = httpx.post(f"{args.api_url}/v1/jobs", json=payload, timeout=300.0)
            except Exception as conn_err:
                 logger.error(f"Connection failed: {conn_err}")
                 break

            if resp.status_code != 200:
                logger.error(f"API Error submitting {img_path.name}: {resp.text}")
                fail_count += 1
                continue
                
            job_data = resp.json()
            job_id = job_data["request_id"]
            
            # 3c. Poll for Completion
            while True:
                try:
                    poll_resp = httpx.get(f"{args.api_url}/v1/jobs/{job_id}", timeout=60.0)
                    status_data = poll_resp.json()
                except Exception as poll_err:
                    logger.warning(f"Poll failed: {poll_err}")
                    time.sleep(1)
                    continue

                status = status_data["status"].upper()
                
                if status == "COMPLETED":
                    duration = time.time() - t0
                    
                    # 3d. Download Artifacts
                    artifacts = status_data.get("artifacts", [])
                    downloaded = 0
                    for art in artifacts:
                        # Art uri might be local file path from server perspective
                        # But since we are likely on localhost, we can try to cp or download via HTTP if served.
                        # The server serves /outputs.
                        # We need to map file path to URL if not provided.
                        # BUT, api_server returns 'uri' as absolute path usually.
                        # We can use the /outputs mount if we know the relative path.
                        
                        art_path = art["uri"] # Absolute path on server
                        # Assuming server and client on same machine (Tauri usage)
                        # We can copy the file if accessible, or download via URL if mapped
                        
                        # Try direct copy first (same FS)
                        try:
                            import shutil
                            art_filename = Path(art_path).name
                            dest = img_output_dir / art_filename
                            shutil.copy2(art_path, dest)
                            downloaded += 1
                            
                            # Convert to STL/OBJ if it's a GLB/OBJ
                            if dest.suffix.lower() in ['.glb', '.gltf', '.obj']:
                                try:
                                    import trimesh
                                    mesh = trimesh.load(dest, force='mesh')
                                    
                                    # Export STL
                                    stl_path = img_output_dir / f"{safe_name}.stl"
                                    mesh.export(stl_path)
                                    
                                    # Export OBJ (if original was not OBJ)
                                    if dest.suffix.lower() != '.obj':
                                        obj_path = img_output_dir / f"{safe_name}.obj"
                                        mesh.export(obj_path)
                                        
                                    logger.info(f"Converted {safe_name} to STL/OBJ")
                                except Exception as conv_err:
                                    logger.error(f"Conversion failed for {safe_name}: {conv_err}")

                        except Exception as cp_err:
                            logger.warning(f"Could not copy local file: {cp_err}. Trying download?")
                            # TODO: Logic to download if remote
                    
                    if downloaded > 0:
                        logger.info(f"Success: {img_path.name} -> {downloaded} files in {duration:.2f}s")
                        success_count += 1
                    else:
                        logger.warning(f"Completed but no artifacts found/copied for {img_path.name}")
                        fail_count += 1
                    break
                    
                elif status == "FAILED":
                    error_msg = status_data.get("error", {}).get("message", "Unknown error")
                    logger.error(f"Job Failed for {img_path.name}: {error_msg}")
                    fail_count += 1
                    break
                
                time.sleep(1) # Poll interval
                
        except Exception as e:
            logger.error(f"Script Error processing {img_path.name}: {e}")
            fail_count += 1

    logger.info("="*30)
    logger.info(f"Batch Processing Complete.")
    logger.info(f"Total: {len(image_files)}")
    logger.info(f"Success: {success_count}")
    logger.info(f"Failed: {fail_count}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBatch processing interrupted.")
        sys.exit(0)
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
