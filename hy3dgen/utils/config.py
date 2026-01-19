
import json
import os
from pathlib import Path
from typing import Dict, Any
import logging

from hy3dgen.utils.system import get_user_data_dir

logger = logging.getLogger("config_manager")

CONFIG_FILE_NAME = "archeon_settings.json"

def get_config_path() -> Path:
    """Returns the path to the persistent configuration file."""
    data_dir = get_user_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / CONFIG_FILE_NAME

def load_config() -> Dict[str, Any]:
    """Loads configuration from disk. Returns empty dict if file doesn't exist or is invalid."""
    path = get_config_path()
    if not path.exists():
        return {}
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load user config from {path}: {e}")
        return {}

def save_config(config: Dict[str, Any]) -> None:
    """Saves the provided configuration dictionary to disk."""
    path = get_config_path()
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        logger.info(f"User config saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save user config to {path}: {e}")

def get_setting(key: str, default: Any = None) -> Any:
    """Helper to get a single setting value."""
    cfg = load_config()
    return cfg.get(key, default)

def update_setting(key: str, value: Any) -> None:
    """Helper to update a single setting value."""
    cfg = load_config()
    cfg[key] = value
    save_config(cfg)
