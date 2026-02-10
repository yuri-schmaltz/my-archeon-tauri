# Archeon 3D Internationalization

_current_lang = "en"

TRANSLATIONS = {
    "en": {
        "app_title": "Archeon 3D",
        "tab_image": "Image",
        "tab_multiview": "MultiView",
        "tab_text": "Text",
        "tab_mesh_prompt": "Mesh",
        "input_image": "Input Image",
        "front": "Front",
        "back": "Back",
        "left": "Left",
        "right": "Right",
        "prompt": "Prompt",
        "negative_prompt": "Negative Prompt",
        "steps": "Steps",
        "seed": "Seed",
        "do_texture": "Generate Texture",
        "btn_generate": "Generate 3D Model",
        "model_downloading": "Downloading Models...",
        "export_model": "Export Model",
        "mesh_source": "Source Mesh (OBJ/GLB/BLEND)",
        "mesh_decimate": "Decimate (Simplify)",
        "mesh_repair": "Repair (Watertight)",
        "mesh_autotex": "Auto-Texture",
        "mesh_target_tris": "Target Tris",
        "btn_process_mesh": "Process Mesh",
        "gallery_title": "Asset Gallery",
        "vram_usage": "VRAM Usage",
        "api_status": "API Status"
    },
    "pt": {
        "app_title": "Archeon 3D",
        "tab_image": "Imagem",
        "tab_multiview": "Multivista",
        "tab_text": "Texto",
        "tab_mesh_prompt": "Malha",
        "input_image": "Imagem de Entrada",
        "front": "Frente",
        "back": "Trás",
        "left": "Esquerda",
        "right": "Direita",
        "prompt": "Prompt",
        "negative_prompt": "Prompt Negativo",
        "steps": "Passos",
        "seed": "Semente",
        "do_texture": "Gerar Textura",
        "btn_generate": "Gerar Modelo 3D",
        "model_downloading": "Descarregando Modelos...",
        "export_model": "Exportar Modelo",
        "mesh_source": "Malha de Origem (OBJ/GLB/BLEND)",
        "mesh_decimate": "Decimar (Simplificar)",
        "mesh_repair": "Reparar (Watertight)",
        "mesh_autotex": "Auto-Textura",
        "mesh_target_tris": "Triângulos Alvo",
        "btn_process_mesh": "Processar Malha",
        "gallery_title": "Galeria de Assets",
        "vram_usage": "Uso de VRAM",
        "api_status": "Status da API"
    },
    "zh": {
        "app_title": "Archeon 3D",
        "tab_image": "图片",
        "tab_multiview": "多视角",
        "tab_text": "文本",
        "tab_mesh_prompt": "网格",
        "input_image": "输入图片",
        "front": "前",
        "back": "后",
        "left": "左",
        "right": "右",
        "prompt": "提示词",
        "negative_prompt": "反向提示词",
        "steps": "步数",
        "seed": "种子",
        "do_texture": "生成纹理",
        "btn_generate": "生成 3D 模型",
        "model_downloading": "下载模型中...",
        "export_model": "导出模型",
        "mesh_source": "源网格 (OBJ/GLB/BLEND)",
        "mesh_decimate": "抽取 (减面)",
        "mesh_repair": "修复 (水密)",
        "mesh_autotex": "自动纹理",
        "mesh_target_tris": "目标面数",
        "btn_process_mesh": "处理网格",
        "gallery_title": "模型仓库",
        "vram_usage": "显存使用",
        "api_status": "API 状态"
    }
}

def set_language(lang_code):
    global _current_lang
    if lang_code in TRANSLATIONS:
        _current_lang = lang_code

def get(key, default=None):
    return TRANSLATIONS.get(_current_lang, {}).get(key, default or key)
