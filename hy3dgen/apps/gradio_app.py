# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import sys
import os
import random
import asyncio
import uuid
import webbrowser
import argparse
from contextlib import asynccontextmanager
from pathlib import Path

import gradio as gr
import numpy as np
import trimesh
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from hy3dgen.utils.system import get_user_cache_dir, setup_logging
from hy3dgen.manager import PriorityRequestManager, ModelManager
from hy3dgen.inference import InferencePipeline
from hy3dgen.apps.ui_templates import HTML_TEMPLATE_MODEL_VIEWER, HTML_PLACEHOLDER, CSS_STYLES
import hy3dgen.i18n as i18n

# Global Log Setup
logger = setup_logging("gradio_app")

# Global Manager
request_manager = None

MAX_SEED = int(1e7)
# Use robust cross-platform cache dir
SAVE_DIR = str(get_user_cache_dir() / "gradio_cache")
HAS_T2I = False
TURBO_MODE = True
HAS_TEXTUREGEN = True

def gen_save_folder():
    os.makedirs(SAVE_DIR, exist_ok=True)
    new_folder = os.path.join(SAVE_DIR, str(uuid.uuid4()))
    os.makedirs(new_folder, exist_ok=True)
    return new_folder

def _coerce_to_trimesh(mesh):
    if mesh is None:
        raise ValueError("Cannot export None mesh")

    if hasattr(mesh, 'mesh_v') and hasattr(mesh, 'mesh_f'):
        mesh = trimesh.Trimesh(vertices=mesh.mesh_v, faces=mesh.mesh_f)

    if isinstance(mesh, trimesh.Scene):
        if not mesh.geometry:
            raise ValueError("Cannot export empty scene")
        mesh = trimesh.util.concatenate([g.copy() for g in mesh.geometry.values()])

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type for export: {type(mesh)}")

    return mesh


def _apply_white_visual(mesh):
    white_color = np.array([235, 235, 235, 255], dtype=np.uint8)
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh)
    mesh.visual.face_colors = np.tile(white_color, (len(mesh.faces), 1))


def _extract_texture_image(mesh):
    if not hasattr(mesh, "visual") or mesh.visual is None:
        return None
    material = getattr(mesh.visual, "material", None)
    if material is None:
        return None

    if hasattr(material, "baseColorTexture") and material.baseColorTexture is not None:
        return material.baseColorTexture
    if hasattr(material, "image") and material.image is not None:
        return material.image
    return None


def _texture_is_mostly_black(mesh, threshold=1.0):
    tex_img = _extract_texture_image(mesh)
    if tex_img is None:
        return False

    tex = np.asarray(tex_img)
    if tex.size == 0:
        return True
    if tex.ndim == 2:
        mean_value = float(tex.mean())
    else:
        mean_value = float(tex[..., :3].mean())
    return mean_value <= threshold


def _normalize_mesh_for_preview(mesh):
    preview = mesh.copy()
    try:
        preview.remove_infinite_values()
        preview.remove_unreferenced_vertices()
    except Exception:
        pass

    bounds = getattr(preview, "bounds", None)
    if bounds is None:
        return preview

    bounds = np.asarray(bounds, dtype=np.float64)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        return preview

    extents = bounds[1] - bounds[0]
    max_extent = float(np.max(extents))
    if max_extent <= 1e-8:
        return preview

    center = (bounds[0] + bounds[1]) * 0.5
    preview.apply_translation(-center)
    # Keep the model large enough for immediate framing in web viewers.
    preview.apply_scale(1.6 / max_extent)
    return preview


def export_mesh(mesh, save_folder, textured=False, file_type='glb', for_preview=False):
    mesh = _coerce_to_trimesh(mesh).copy()

    base_name = 'textured_mesh' if textured else 'white_mesh'
    if for_preview:
        base_name = f"{base_name}_preview"
    path = os.path.join(save_folder, f'{base_name}.{file_type}')

    if for_preview:
        mesh = _normalize_mesh_for_preview(mesh)

    try:
        from trimesh import repair
        repair.fix_normals(mesh)
        repair.fix_inversion(mesh)
    except Exception as e:
        logger.warning(f"Failed to repair mesh normals: {e}")

    if not textured:
        _apply_white_visual(mesh)
    else:
        visual_type = mesh.visual.__class__.__name__ if hasattr(mesh, 'visual') else 'None'
        logger.info(f"Exporting textured mesh with visual type: {visual_type}")

        if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'material'):
            material = mesh.visual.material
            if isinstance(material, trimesh.visual.material.PBRMaterial):
                material.baseColorFactor = [255, 255, 255, 255]
                material.metallicFactor = 0.0
                material.roughnessFactor = 0.9
            elif isinstance(material, trimesh.visual.material.SimpleMaterial):
                if hasattr(material, 'diffuse'):
                    material.diffuse = [255, 255, 255, 255]
                if hasattr(material, 'ambient'):
                    material.ambient = [255, 255, 255, 255]

        if _texture_is_mostly_black(mesh):
            logger.warning("Detected near-black texture map. Falling back to neutral material for export.")
            _apply_white_visual(mesh)

    if file_type not in ['glb', 'obj']:
        mesh.export(path)
    else:
        mesh.export(path, include_normals=True)

    logger.info(
        f"Exported {'textured' if textured else 'white'} "
        f"{'preview ' if for_preview else ''}mesh to {path}"
    )
    return path

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

def build_model_viewer_html(save_folder, height=660, width=790, textured=False, preview=False):
    if textured:
        related_path = "./textured_mesh_preview.glb" if preview else "./textured_mesh.glb"
        output_html_path = os.path.join(save_folder, 'textured_mesh.html')
    else:
        related_path = "./white_mesh_preview.glb" if preview else "./white_mesh.glb"
        output_html_path = os.path.join(save_folder, 'white_mesh.html')
    
    template_html = HTML_TEMPLATE_MODEL_VIEWER
    with open(output_html_path, 'w', encoding='utf-8') as f:
        template_html = template_html.replace('#src#', f'{related_path}')
        f.write(template_html)
    rel_path = os.path.relpath(output_html_path, SAVE_DIR)
    iframe_tag = f'<iframe src="/static/{rel_path}" style="width: 100%; height: 100%; border: none; border-radius: 8px; box-shadow: 0 4px 12px rgb(0 0 0 / 0.28);"></iframe>'
    return iframe_tag

# Helper for HTML Progress
def render_progress_bar(percent, message):
     p = max(0, min(100, int(percent)))
     # Format: Message - 50%
     return f"""
     <div class="archeon-progress-container">
         <div class="archeon-progress-fill" style="width: {p}%;"></div>
         <div class="archeon-progress-text">{message} &nbsp; <b>{p}%</b></div>
     </div>
     """

async def unified_generation(model_key, caption, negative_prompt, image, mv_image_front, mv_image_back, mv_image_left, mv_image_right, steps, guidance_scale, seed, octree_resolution, check_box_rembg, num_chunks, tex_steps, tex_guidance_scale, tex_seed, randomize_seed, do_texture=True, progress=gr.Progress()):
    import time
    mv_mode = model_key == "Multiview"
    mv_images = {}
    if mv_mode:
        if mv_image_front:
            mv_images['front'] = mv_image_front
        if mv_image_back:
            mv_images['back'] = mv_image_back
        if mv_image_left:
            mv_images['left'] = mv_image_left
        if mv_image_right:
            mv_images['right'] = mv_image_right
    seed = int(randomize_seed_fn(seed, randomize_seed))
    
    # Thread-safe progress tracking
    import queue
    progress_queue = queue.Queue()
    
    def gradio_progress_callback(percent, message):
        progress_queue.put((percent, message))

    params = {
        'model_key': model_key,
        'text': caption,
        'negative_prompt': negative_prompt,
        'image': image,
        'mv_images': mv_images if mv_mode else None,
        'num_inference_steps': int(steps),
        'guidance_scale': guidance_scale,
        'seed': seed,
        'octree_resolution': int(octree_resolution),
        'do_rembg': check_box_rembg,
        'num_chunks': int(num_chunks),
        'do_texture': do_texture,
        'tex_steps': int(tex_steps),
        'tex_guidance_scale': float(tex_guidance_scale),
        'tex_seed': int(tex_seed),
        'progress_callback': gradio_progress_callback
    }
    
    logger.info("ACTION: Generation Request Submitted")
    
    # Save Config (Persistence Step)
    try:
        from hy3dgen.utils.config import save_config
        save_config({
            'num_inference_steps': int(steps),
            'guidance_scale': float(guidance_scale),
            'seed': int(seed),
            'octree_resolution': int(octree_resolution),
            'do_rembg': check_box_rembg,
            'num_chunks': int(num_chunks),
            'tex_steps': int(tex_steps),
            'tex_guidance_scale': float(tex_guidance_scale)
        })
    except Exception as e:
        logger.warning(f"Failed to auto-save config: {e}")
    
    # Run generation in background task
    task = asyncio.create_task(request_manager.submit(params))
    start_time = time.time()
    
    # Loop continuously while task runs to update UI
    last_msg = "Initializing..."
    last_pct = 0
    
    while not task.done():
        # Process all pending progress messages
        while not progress_queue.empty():
             last_pct, last_msg = progress_queue.get()
        
        elapsed = time.time() - start_time
        timer_text = f"Stop ({elapsed:.1f}s)"
        progress_html = render_progress_bar(last_pct, last_msg)
        
        # Yield updates: [file_out, html_gen_mesh, seed, progress_html, btn_stop]
        # Use gr.skip() for outputs we don't want to change yet
        yield (
            gr.skip(), 
            gr.skip(), 
            gr.skip(), 
            gr.update(visible=True, value=progress_html), 
            gr.update(value=timer_text)
        )
        
        await asyncio.sleep(0.1)
        
    # Task Finished
    try:
        result = await task
        mesh = result["mesh"]
    except asyncio.CancelledError:
        logger.info("Generation Calcelled by User")
        yield (gr.skip(), gr.skip(), gr.skip(), gr.update(visible=False), gr.update(value="Stop Generation"))
        return
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise gr.Error(f"Generation Failed: {str(e)}")
    
    save_folder = gen_save_folder()

    # Always provide a clean white preview first.
    path_white = export_mesh(mesh, save_folder, textured=False, for_preview=False)
    export_mesh(mesh, save_folder, textured=False, for_preview=True)
    html_white = build_model_viewer_html(save_folder, textured=False, preview=True)
    yield (
        gr.DownloadButton(value=path_white, visible=True),
        html_white,
        seed,
        gr.update(visible=False),
        gr.update(value="Stop Generation")
    )

    if do_texture:
        textured_mesh = result.get("textured_mesh")
        if textured_mesh is None:
            logger.warning("Texture generation failed, using untextured mesh.")
            textured_mesh = mesh
        elif _texture_is_mostly_black(textured_mesh):
            logger.warning("Generated texture is near-black; using untextured mesh fallback.")
            textured_mesh = mesh

        path_textured = export_mesh(textured_mesh, save_folder, textured=True, for_preview=False)
        export_mesh(textured_mesh, save_folder, textured=True, for_preview=True)
        html_textured = build_model_viewer_html(save_folder, textured=True, preview=True)
        yield (
            gr.DownloadButton(value=path_textured, visible=True),
            html_textured,
            seed,
            gr.update(visible=False),
            gr.update(value="Stop Generation")
        )

def build_app(example_is=None, example_ts=None, example_mvs=None):
    # Gradio 6.3+: theme and css are handled in mount_gradio_app
    with gr.Blocks(
        title=i18n.get('app_title'),
        analytics_enabled=False,
        fill_height=True
    ) as demo:
        # State to track current model mode based on tab
        model_key_state = gr.State("Normal")

        # Load Persistent Config
        from hy3dgen.utils.config import get_setting
        
        # Defaults
        def_res = get_setting('octree_resolution', 256)
        def_chunks = get_setting('num_chunks', 8000)
        def_rembg = get_setting('do_rembg', True)
        def_steps = get_setting('num_inference_steps', 50)
        def_cfg = get_setting('guidance_scale', 5.0)
        def_tex_steps = get_setting('tex_steps', 30)
        def_tex_cfg = get_setting('tex_guidance_scale', 5.0)
        def_seed = get_setting('seed', 1234)

        with gr.Row(elem_classes="main-row"):
            with gr.Column(scale=1, elem_classes="left-col"):
                
                with gr.Group(elem_classes="scroll-area"):

                    with gr.Tabs(selected='tab_img_prompt'):
                        with gr.Tab(i18n.get('tab_img_prompt'), id='tab_img_prompt') as tab_ip:
                            with gr.Column(elem_classes="prompt-container"):
                                 image = gr.Image(label=i18n.get('lbl_image'), type='pil', image_mode='RGBA', sources=['upload', 'clipboard'], height=400)
                        
                        with gr.Tab(i18n.get('tab_mv_prompt'), id='tab_mv_prompt') as tab_mv_p:
                            with gr.Column(elem_classes="prompt-container"):
                                with gr.Row(variant='compact'):
                                    mv_image_front = gr.Image(label=i18n.get('lbl_front'), type='pil', image_mode='RGBA', sources=['upload', 'clipboard'])
                                    mv_image_left = gr.Image(label=i18n.get('lbl_left'), type='pil', image_mode='RGBA', sources=['upload', 'clipboard'])
                                    mv_image_back = gr.Image(label=i18n.get('lbl_back'), type='pil', image_mode='RGBA', sources=['upload', 'clipboard'])
                                    mv_image_right = gr.Image(label=i18n.get('lbl_right'), type='pil', image_mode='RGBA', sources=['upload', 'clipboard'])

                        with gr.Tab(i18n.get('tab_text_prompt'), id='tab_txt_prompt', visible=HAS_T2I) as tab_tp:
                            with gr.Column(elem_classes="prompt-container"):
                                caption = gr.Textbox(label=i18n.get('tab_text_prompt'), placeholder=i18n.get('ph_text_prompt'), lines=5, max_lines=5)
                                negative_prompt = gr.Textbox(label='Negative Prompt', placeholder=i18n.get('ph_negative_prompt'), lines=4, max_lines=4)

                        
                        # Settings now inside scroll area group
                        with gr.Column(visible=True, elem_classes="panel-container") as gen_settings_container:
                            with gr.Tabs(selected='tab_options' if TURBO_MODE else 'tab_export'):
                                with gr.Tab("Quality", id='tab_options', visible=TURBO_MODE):
                                    gr.Radio(label=i18n.get('lbl_gen_mode'), choices=['Turbo', 'Fast', 'Standard'], value='Turbo')
                                    gr.Radio(label='Decoding Mode', choices=['Low', 'Standard', 'High'], value='Standard')
                                
                                with gr.Tab("Advanced"):
                                    with gr.Group():
                                        with gr.Row():
                                            check_box_rembg = gr.Checkbox(value=def_rembg, label=i18n.get('lbl_rembg'))
                                            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                                        seed = gr.Slider(label=i18n.get('lbl_seed'), minimum=0, maximum=MAX_SEED, step=1, value=def_seed)
                                        with gr.Row():
                                            octree_resolution = gr.Slider(maximum=512, minimum=16, value=def_res, label=i18n.get('lbl_octree'), info="Higher = sharper geometry, more VRAM.")
                                            num_chunks = gr.Slider(maximum=5000000, minimum=1000, value=def_chunks, label=i18n.get('lbl_chunks'), info="Memory management for large meshes.")
                                            
                                with gr.Tab("Texture"):
                                     with gr.Group():
                                        with gr.Row():
                                            tex_steps = gr.Slider(maximum=100, minimum=1, value=def_tex_steps, step=1, label='Steps', info="Texture refinement steps.")
                                            tex_guidance_scale = gr.Number(value=def_tex_cfg, label='Guidance', info="Texture prompt adherence.")
                                        tex_seed = gr.Slider(label="Texture Seed", minimum=0, maximum=MAX_SEED, step=1, value=def_seed)

                # Exposed Critical Parameters (Moved to Footer)
                with gr.Group(elem_classes="footer-area"):
                   with gr.Row():
                       num_steps = gr.Slider(maximum=100, minimum=1, value=def_steps, step=1, label=i18n.get('lbl_steps'), info=i18n.get('info_steps'))
                       cfg_scale = gr.Number(value=def_cfg, label=i18n.get('lbl_guidance'), info="Prompt strictness")
            
            # Left Column Ends Here

            with gr.Column(scale=1, elem_classes="right-col"):
                with gr.Tabs(selected='gen_mesh_panel', elem_classes="scroll-area"):
                    with gr.Tab(i18n.get('lbl_output'), id='gen_mesh_panel'):
                        with gr.Column(elem_id="gen_output_container"):
                            html_gen_mesh = gr.HTML(HTML_PLACEHOLDER, label='Output', elem_id="model_3d_viewer")
                
                # Custom Progress Bar (Between Viewer and Footer)
                progress_html = gr.HTML(visible=False, elem_id="progress_bar_container")

                # Footer Action Area (Moved from Left)
                with gr.Row(elem_classes="footer-area"):
                    btn = gr.Button(value=i18n.get('btn_generate'), variant='primary', scale=2)
                    btn_stop = gr.Button(value="Stop Generation", variant='stop', visible=False, scale=2)
                    file_out = gr.DownloadButton(label="Download .glb", variant='primary', visible=True, scale=1)
                
        # Helper to toggle buttons
        def on_gen_start():
            logger.info("UI EVENT: Generation started.")
            # Hide Generate, Show Stop with Init Timer
            return gr.update(visible=False), gr.update(visible=True, value="Stop (Preparing...)")
        
        def on_gen_finish():
            logger.info("UI EVENT: Generation finished (or stopped). Restoring UI.")
            # Show Generate, Hide Stop, Hide Progress
            return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False, value="")

        # Explicit Tab Selection Handlers
        def on_image_tab_select():
            return "Normal", gr.update(visible=True)
            
        def on_mv_tab_select():
            return "Multiview", gr.update(visible=True)
            
        def on_text_tab_select():
            return "Normal", gr.update(visible=True)

        # Bind events strictly to specific tabs
        tab_ip.select(fn=on_image_tab_select, outputs=[model_key_state, gen_settings_container])
        tab_mv_p.select(fn=on_mv_tab_select, outputs=[model_key_state, gen_settings_container])
        if HAS_T2I:
            tab_tp.select(fn=on_text_tab_select, outputs=[model_key_state, gen_settings_container])

        # Wire events
        # Event Chain: Textured Generation (Single Flow)
        # 1. Start: Swap Buttons
        succ1_1 = btn.click(on_gen_start, outputs=[btn, btn_stop])
        
        # 2. Generate
        succ1_2 = succ1_1.then(
            unified_generation, 
            inputs=[model_key_state, caption, negative_prompt, image, mv_image_front, mv_image_back, mv_image_left, mv_image_right, num_steps, cfg_scale, seed, octree_resolution, check_box_rembg, num_chunks, tex_steps, tex_guidance_scale, tex_seed, randomize_seed], 
            outputs=[file_out, html_gen_mesh, seed, progress_html, btn_stop]
        )
        
        # 3. Finish: Swap Back
        succ1_2.then(on_gen_finish, outputs=[btn, btn_stop, progress_html])
        
        # Stop Action: Cancel Generation and Swap Back
        btn_stop.click(
            fn=on_gen_finish,
            outputs=[btn, btn_stop, progress_html],
            cancels=[succ1_2]
        )

    return demo

def main():
    global request_manager, SAVE_DIR, HAS_T2I, HAS_TEXTUREGEN
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='tencent/Hunyuan3D-2')
    parser.add_argument("--subfolder", type=str, default='hunyuan3d-dit-v2-0-turbo')
    parser.add_argument("--texgen_model_path", type=str, default='tencent/Hunyuan3D-2')
    # Default port will be handled dynamically if not set, but argparse defaults to 7860 here if we don't change logic. 
    # Better to allow 7860 as default preference but find another if taken.
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--cache-path', type=str, default=None)
    parser.add_argument('--enable_t23d', action='store_true')
    parser.add_argument('--disable_tex', action='store_true', help='Disable texture generation (avoids custom_rasterizer requirement)')
    parser.add_argument('--low_vram_mode', action='store_true', default=True)
    parser.add_argument('--no-browser', action='store_true', help='Do not open the browser automatically')
    args = parser.parse_args()
    
    if args.cache_path:
        SAVE_DIR = args.cache_path
    # Else SAVE_DIR is already set to XDG location by global init
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    HAS_T2I = args.enable_t23d
    HAS_TEXTUREGEN = not args.disable_tex
    
    if args.disable_tex:
        logger.info("Texturization disabled (custom_rasterizer not required)")

    model_mgr = ModelManager(capacity=1 if args.low_vram_mode else 3, device=args.device)
    def get_loader(model_path, subfolder):
        return lambda: InferencePipeline(
            model_path=model_path, tex_model_path=args.texgen_model_path, subfolder=subfolder,
            device=args.device, enable_t2i=args.enable_t23d, enable_tex=not args.disable_tex,
            low_vram_mode=args.low_vram_mode
        )
    model_mgr.register_model("Normal", get_loader("tencent/Hunyuan3D-2", "hunyuan3d-dit-v2-0-turbo"))

    model_mgr.register_model("Multiview", get_loader("tencent/Hunyuan3D-2mv", "hunyuan3d-dit-v2-mv-turbo"))
    request_manager = PriorityRequestManager(model_mgr, max_concurrency=1)
    
    # Define lifespan for FastAPI app (Gradio 6.3+ / FastAPI 0.93+)
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        logger.info("Starting PriorityRequestManager...")
        asyncio.create_task(request_manager.start())
        logger.info("PriorityRequestManager started successfully")
        yield
        # Shutdown (if needed)
    
    # Cria a aplicação FastAPI com lifespan
    app = FastAPI(lifespan=lifespan)
    
    static_dir = Path(SAVE_DIR).absolute()
    app.mount("/static", StaticFiles(directory=static_dir, html=True), name="static")
    demo = build_app()
    
    # Injeta CSS via head customizada (Gradio 6.3+)
    custom_head = f"<style>{CSS_STYLES}</style>"
    app = gr.mount_gradio_app(
        app, 
        demo, 
        path="/",
        head=custom_head,
        theme=gr.themes.Base(
            font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
            font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "Consolas", "monospace"],
        )
    )
    url = f"http://{args.host}:{args.port}"
    print(f"\nHunyuan3D-2 Pro Unified is running at: {url}\n")
    if not args.no_browser:
        # Redirect stderr to devnull to suppress browser logs (e.g. DEPRECATED_ENDPOINT)
        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            old_stderr = os.dup(2)
            sys.stderr.flush()
            os.dup2(devnull, 2)
            os.close(devnull)
            
            webbrowser.open(url)
        except Exception:
            # Fallback if redirection fails
            webbrowser.open(url)
        finally:
            # Restore stderr
            if 'old_stderr' in locals():
                os.dup2(old_stderr, 2)
                os.close(old_stderr)
    uvicorn.run(app, host=args.host, port=args.port, workers=1)

if __name__ == '__main__':
    main()
