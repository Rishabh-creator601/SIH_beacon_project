"""Configuration loading.

The default configuration lives in ``config/default.yaml``. A user
scenario file only needs to contain the keys it wants to change; it is
deep-merged over the defaults.
"""

import copy
import sys
from pathlib import Path

import yaml

if getattr(sys, "frozen", False):
    # Packaged .exe (PyInstaller): bundled read-only files are unpacked to a
    # temporary folder, while outputs must go next to the .exe so they survive.
    PROJECT_ROOT = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    OUTPUT_ROOT = Path(sys.executable).resolve().parent
else:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    OUTPUT_ROOT = PROJECT_ROOT
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"


def deep_merge(base, override):
    """Return a copy of ``base`` with ``override`` merged in recursively.

    Dicts are merged key by key; any other value (including lists such as
    the target list) replaces the base value entirely.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path=None):
    """Load the default config, optionally merged with a user config file."""
    with open(DEFAULT_CONFIG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if path:
        with open(path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        cfg = deep_merge(cfg, user)
    return cfg
