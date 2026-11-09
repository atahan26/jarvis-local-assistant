"""Configuration loading.

config.yaml is authoritative for every tunable (spec §12). config.local.yaml, if
present, is deep-merged on top of it so local tuning never has to touch the
version-controlled file.

Path values under ``paths:`` are resolved as follows:
  * ``data_dir`` and ``log_dir`` are resolved against the project root if relative
  * every other path key is resolved against ``data_dir`` if relative
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
LOCAL_CONFIG_PATH = PROJECT_ROOT / "config.local.yaml"

_MISSING = object()


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge; scalars and lists in *override* win outright."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


class Config:
    """Dotted-path access over the merged YAML tree, plus resolved paths."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    # -- generic access ----------------------------------------------------

    def get(self, dotted: str, default: Any = _MISSING) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is _MISSING:
                    raise KeyError(f"config key not found: {dotted}")
                return default
            node = node[part]
        return node

    def section(self, name: str) -> dict[str, Any]:
        return self.get(name)

    @property
    def data(self) -> dict[str, Any]:
        return self._data

    # -- paths -------------------------------------------------------------

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def data_dir(self) -> Path:
        return self._resolve_root(self.get("paths.data_dir"))

    @property
    def bin_dir(self) -> Path:
        return self._resolve_data(self.get("paths.bin_dir"))

    @property
    def models_dir(self) -> Path:
        return self._resolve_data(self.get("paths.models_dir"))

    @property
    def downloads_dir(self) -> Path:
        return self._resolve_data(self.get("paths.downloads_dir"))

    @property
    def log_dir(self) -> Path:
        return self._resolve_root(self.get("paths.log_dir"))

    def _resolve_root(self, value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else (PROJECT_ROOT / p)

    def _resolve_data(self, value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else (self.data_dir / p)

    # -- component paths ---------------------------------------------------

    @property
    def whisper_exe(self) -> Path:
        return self.bin_dir / self.get("whisper.exe")

    @property
    def whisper_model(self) -> Path:
        return self.models_dir / self.get("whisper.model")

    @property
    def llama_exe(self) -> Path:
        return self.bin_dir / self.get("llm.exe")

    @property
    def llm_model(self) -> Path:
        return self.models_dir / self.get("llm.model")

    @property
    def tts_model(self) -> Path:
        return self.models_dir / self.get("tts.model")

    @property
    def tts_voices(self) -> Path:
        return self.models_dir / self.get("tts.voices")

    # -- endpoints ---------------------------------------------------------

    @property
    def whisper_url(self) -> str:
        return f"http://{self.get('whisper.host')}:{self.get('whisper.port')}"

    @property
    def llm_url(self) -> str:
        return f"http://{self.get('llm.host')}:{self.get('llm.port')}"


def load(path: Path | None = None) -> Config:
    path = path or CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"config.yaml not found at {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if LOCAL_CONFIG_PATH.exists():
        with LOCAL_CONFIG_PATH.open("r", encoding="utf-8") as fh:
            data = _deep_merge(data, yaml.safe_load(fh) or {})
    return Config(data)
