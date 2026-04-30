"""Tests for the Auth module."""
import tempfile

import pytest

from src.core.auth import AuthManager


class TestAuthManager:
    def setup_method(self):
        self.tmpfile = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        self.tmpfile.close()
        self.auth = AuthManager(
            config_file=self.tmpfile.name,
            bcrypt_rounds=4,
            max_attempts=3,
            lockout_minutes=1,
        )

    def teardown_method(self):
        import os
        os.unlink(self.tmpfile.name)

    def test_default_password(self):
        ok, _ = self.auth.authenticate("admin", "admin123")
        assert ok is True

    def test_wrong_password(self):
        ok, _ = self.auth.authenticate("admin", "wrong")
        assert ok is False

    def test_change_password(self):
        ok, _ = self.auth.change_password("admin123", "NewPass123")
        assert ok is True
        ok, _ = self.auth.authenticate("admin", "NewPass123")
        assert ok is True
        ok, _ = self.auth.authenticate("admin", "admin123")
        assert ok is False

    def test_change_password_wrong_old(self):
        ok, msg = self.auth.change_password("wrong", "NewPass123")
        assert ok is False

    def test_change_password_too_short(self):
        ok, msg = self.auth.change_password("admin123", "short")
        assert ok is False
        assert "长度" in msg

    def test_password_strength(self):
        ok, _ = AuthManager.validate_password_strength("Abc12345")
        assert ok is True
        ok, _ = AuthManager.validate_password_strength("short")
        assert ok is False
        ok, _ = AuthManager.validate_password_strength("alllowercase1")
        assert ok is False
        ok, _ = AuthManager.validate_password_strength("ALLUPPERCASE1")
        assert ok is False
        ok, _ = AuthManager.validate_password_strength("NoDigitsHere")
        assert ok is False

    def test_account_lockout(self):
        for _ in range(3):
            self.auth.authenticate("admin", "wrong")
        ok, msg = self.auth.authenticate("admin", "admin123")
        assert ok is False
        assert "锁定" in msg

    def test_legacy_hash_migration(self):
        ok, _ = self.auth.authenticate("admin", "admin123")
        assert ok is True
        import json
        with open(self.tmpfile.name, "r") as f:
            data = json.load(f)
        assert data["admin_password_hash"].startswith("$2b$")
