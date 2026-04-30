"""
Authentication and password security module.
Uses bcrypt for secure password hashing with automatic salting.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import bcrypt


class AuthManager:
    """Manages administrator authentication with bcrypt hashing and account lockout."""

    def __init__(
        self,
        config_file: str = "data/admin_config.json",
        bcrypt_rounds: int = 12,
        max_attempts: int = 5,
        lockout_minutes: int = 30,
    ):
        self._config_file = Path(config_file)
        self._bcrypt_rounds = bcrypt_rounds
        self._max_attempts = max_attempts
        self._lockout_minutes = lockout_minutes
        self._failed_attempts: dict[str, int] = {}
        self._lockout_until: dict[str, float] = {}
        self._config: dict[str, Any] = {}
        self._ensure_config()

    def _ensure_config(self) -> None:
        self._config_file.parent.mkdir(parents=True, exist_ok=True)
        if self._config_file.exists() and self._config_file.stat().st_size > 0:
            with open(self._config_file, "r", encoding="utf-8") as f:
                self._config = json.load(f)
        else:
            self._config = {
                "admin_password_hash": self._hash_legacy("admin123"),
                "password_changed": False,
            }
            self._save()

    def _save(self) -> None:
        with open(self._config_file, "w", encoding="utf-8") as f:
            json.dump(self._config, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _hash_legacy(password: str) -> str:
        return hashlib.sha256(password.encode("utf-8")).hexdigest()

    def hash_password(self, password: str) -> str:
        salt = bcrypt.gensalt(rounds=self._bcrypt_rounds)
        return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

    def verify_password(self, password: str, stored_hash: str) -> bool:
        try:
            if stored_hash.startswith("$2b$") or stored_hash.startswith("$2a$"):
                return bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8"))
        except (ValueError, TypeError):
            pass
        if len(stored_hash) == 64:
            return self._hash_legacy(password) == stored_hash
        return False

    def _is_locked(self, username: str) -> bool:
        import time
        until = self._lockout_until.get(username, 0)
        if until and time.time() < until:
            return True
        if until and time.time() >= until:
            self._lockout_until.pop(username, None)
            self._failed_attempts.pop(username, None)
        return False

    def _record_failure(self, username: str) -> None:
        import time
        self._failed_attempts[username] = self._failed_attempts.get(username, 0) + 1
        if self._failed_attempts[username] >= self._max_attempts:
            self._lockout_until[username] = time.time() + self._lockout_minutes * 60

    def authenticate(self, username: str, password: str) -> tuple[bool, str]:
        if self._is_locked(username):
            import time
            remaining = int(self._lockout_until[username] - time.time())
            return False, f"账户已锁定，请 {remaining} 秒后再试"
        stored_hash = self._config.get("admin_password_hash", "")
        if not stored_hash:
            return False, "系统配置错误：未找到管理员密码"
        if self.verify_password(password, stored_hash):
            self._failed_attempts.pop(username, None)
            if not stored_hash.startswith("$2b$") and not stored_hash.startswith("$2a$"):
                self._config["admin_password_hash"] = self.hash_password(password)
                self._save()
            return True, "登录成功"
        self._record_failure(username)
        remaining = self._max_attempts - self._failed_attempts.get(username, 0)
        return False, f"密码错误，还剩 {remaining} 次尝试机会"

    def change_password(self, old_password: str, new_password: str) -> tuple[bool, str]:
        stored_hash = self._config.get("admin_password_hash", "")
        if not self.verify_password(old_password, stored_hash):
            return False, "原密码错误"
        if len(new_password) < 8:
            return False, "新密码长度不能少于 8 个字符"
        if new_password == old_password:
            return False, "新密码不能与原密码相同"
        self._config["admin_password_hash"] = self.hash_password(new_password)
        self._config["password_changed"] = True
        self._save()
        return True, "密码修改成功"

    @property
    def password_changed(self) -> bool:
        return self._config.get("password_changed", False)

    @staticmethod
    def validate_password_strength(password: str) -> tuple[bool, str]:
        if len(password) < 8:
            return False, "密码长度不能少于 8 个字符"
        if not any(c.isupper() for c in password):
            return False, "密码必须包含至少一个大写字母"
        if not any(c.islower() for c in password):
            return False, "密码必须包含至少一个小写字母"
        if not any(c.isdigit() for c in password):
            return False, "密码必须包含至少一个数字"
        return True, "密码强度合格"
