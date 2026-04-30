"""
Configuration loader for the Face Access Control Platform.
"""

import copy
import os
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG = {
    "recognition": {
        "backend": "deepface",
        "deepface_model": "ArcFace",
        "deepface_detector": "opencv",
        "confidence_threshold": 0.6,
        "model_path": "data/models/",
    },
    "scenario": {
        "type": "access_control",
    },
    "camera": {
        "device_index": 0,
        "detection_window": 5,
        "frame_interval": 100,
    },
    "face_collection": {
        "capture_count": 100,
        "save_count": 10,
        "face_data_dir": "data/face_data/",
    },
    "security": {
        "bcrypt_rounds": 12,
        "max_login_attempts": 5,
        "lockout_minutes": 30,
        "min_password_length": 8,
    },
    "logging": {
        "log_file": "data/logs.json",
        "max_entries": 10000,
    },
    "data": {
        "users_file": "data/users.json",
        "admin_config_file": "data/admin_config.json",
        "features_file": "data/models/face_features.npy",
    },
}


class Config:
    """Application configuration with YAML file support and environment variable overrides."""

    def __init__(self, config_path: str | None = None):
        self._data: dict[str, Any] = {}
        # Order: defaults -> file overrides -> env overrides
        self._apply_defaults()
        if config_path and os.path.exists(config_path):
            self._load_from_file(config_path)
        self._apply_env_overrides()

    def _load_from_file(self, path: str) -> None:
        with open(path, encoding="utf-8") as f:
            file_data = yaml.safe_load(f) or {}
        self._deep_merge(self._data, file_data)

    def _apply_defaults(self) -> None:
        self._deep_merge(self._data, copy.deepcopy(_DEFAULT_CONFIG))

    def _apply_env_overrides(self) -> None:
        prefix = "FACE_CTL_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                # Support nested keys like FACE_CTL_RECOGNITION_BACKEND
                parts = config_key.split("__")
                target = self._data
                for part in parts[:-1]:
                    target = target.setdefault(part, {})
                target[parts[-1]] = value

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                Config._deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    def get(self, *keys: str, default: Any = None) -> Any:
        """Get a nested config value. Usage: config.get('recognition', 'backend')"""
        result = self._data
        for key in keys:
            if isinstance(result, dict) and key in result:
                result = result[key]
            else:
                return default
        return result

    @property
    def data_dir(self) -> Path:
        return Path(self.get("data", "users_file", default="data/users.json")).parent

    def __repr__(self) -> str:
        b = self.get("recognition", "backend")
        s = self.get("scenario", "type")
        return f"Config(backend={b}, scenario={s})"
