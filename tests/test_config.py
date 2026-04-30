"""Tests for the Config module."""
import os
import tempfile

from src.core.config import Config


def _clean_env():
    """Remove all FACE_CTL env vars."""
    for k in list(os.environ.keys()):
        if k.startswith("FACE_CTL_"):
            del os.environ[k]


class TestConfig:
    def setup_method(self):
        _clean_env()

    def teardown_method(self):
        _clean_env()

    def test_default_values(self):
        config = Config()
        assert config.get("recognition", "backend") == "deepface"
        assert config.get("scenario", "type") == "access_control"
        assert config.get("security", "max_login_attempts") == 5

    def test_yaml_loading(self):
        yaml_content = """
recognition:
  backend: "lbph"
  confidence_threshold: 0.8
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = Config(f.name)
            assert config.get("recognition", "backend") == "lbph"
            assert config.get("recognition", "confidence_threshold") == 0.8
            os.unlink(f.name)

    def test_env_override(self):
        os.environ["FACE_CTL_RECOGNITION__BACKEND"] = "insightface"
        config = Config()
        assert config.get("recognition", "backend") == "insightface"

    def test_get_with_default(self):
        config = Config()
        assert config.get("nonexistent", "key", default="fallback") == "fallback"

    def test_missing_config_file(self):
        config = Config("/nonexistent/path.yaml")
        assert config.get("recognition", "backend") == "deepface"
