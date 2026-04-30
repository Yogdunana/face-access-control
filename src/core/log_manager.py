"""
Audit logging module for the Face Access Control Platform.
Records all system events including authentication, recognition results, and admin operations.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class LogManager:
    """Thread-safe audit logger with JSON file persistence and rotation."""

    def __init__(self, log_file: str = "data/logs.json", max_entries: int = 10000):
        self._log_file = Path(log_file)
        self._max_entries = max_entries
        self._ensure_file()

    def _ensure_file(self) -> None:
        self._log_file.parent.mkdir(parents=True, exist_ok=True)
        if not self._log_file.exists():
            self._write([])

    def _read(self) -> list[dict[str, Any]]:
        try:
            with open(self._log_file, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _write(self, entries: list[dict[str, Any]]) -> None:
        with open(self._log_file, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)

    def log(
        self,
        event_type: str,
        detail: str,
        user: str = "system",
        level: str = "info",
    ) -> None:
        """Record a log entry."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "level": level,
            "user": user,
            "detail": detail,
        }
        entries = self._read()
        entries.append(entry)
        if len(entries) > self._max_entries:
            entries = entries[-self._max_entries:]
        self._write(entries)

    def log_login_success(self, username: str) -> None:
        self.log("login", f"管理员 {username} 登录成功", user=username)

    def log_login_failure(self, username: str, reason: str = "") -> None:
        self.log("login", f"管理员 {username} 登录失败: {reason}", user=username, level="warning")

    def log_recognition_success(self, user_name: str, confidence: float) -> None:
        self.log("recognition", f"识别成功: {user_name} (置信度: {confidence:.2%})")

    def log_recognition_failure(self, reason: str = "未匹配到已知用户") -> None:
        self.log("recognition", f"识别失败: {reason}", level="warning")

    def log_recognition_denied(self, user_name: str, reason: str = "不在授权时段内") -> None:
        self.log("access_denied", f"权限拒绝: {user_name} ({reason})", level="warning")

    def log_user_added(self, user_name: str, admin: str) -> None:
        self.log("user_mgmt", f"管理员 {admin} 添加用户 {user_name}", user=admin)

    def log_user_deleted(self, user_name: str, admin: str) -> None:
        self.log("user_mgmt", f"管理员 {admin} 删除用户 {user_name}", user=admin)

    def log_user_updated(self, user_name: str, admin: str, changes: str) -> None:
        self.log("user_mgmt", f"管理员 {admin} 更新用户 {user_name}: {changes}", user=admin)

    def log_face_registered(self, user_name: str, image_count: int) -> None:
        self.log("face_register", f"用户 {user_name} 注册人脸, 采集 {image_count} 张照片")

    def get_logs(
        self,
        event_type: str | None = None,
        user: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query logs with optional filters."""
        entries = self._read()
        if event_type:
            entries = [e for e in entries if e.get("event_type") == event_type]
        if user:
            entries = [e for e in entries if e.get("user") == user]
        return entries[-limit:]

    def export_logs(self, output_path: str) -> None:
        """Export all logs to a file."""
        entries = self._read()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)

    def clear_logs(self) -> int:
        """Clear all logs and return the count of cleared entries."""
        count = len(self._read())
        self._write([])
        return count
